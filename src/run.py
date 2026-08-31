#!/usr/bin/env python
"""
Experiment entrypoint (README.md): config in -> CV score + OOF + experiments.csv row.

    python src/run.py --config configs/e0001_lgbm_base.yaml

Protocol enforced here, not left to the caller:
  * folds come from data/folds.parquet and are never recomputed (rule 3)
  * the metric comes from src/metrics.py (rule 4)
  * a row is appended to experiments.csv before anything is reported (rule 5)
  * OOF predictions land in oof/<exp_id>.parquet (§3.2)
  * the naive `geo3` reference is scored on the identical folds every run, so `delta`
    is always an exact paired comparison rather than a number copied from a log
  * early stopping uses the most recent TRAINING anchor, never the validation fold
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import Panel                    # noqa: E402
from features import build                # noqa: E402
from metrics import rmsle, score_all      # noqa: E402


def log(m: str) -> None:
    print(m, flush=True)


def assert_no_lookahead(p: Panel, anchor: int, X: np.ndarray, keep: np.ndarray,
                        blocks: list[str], mw=None) -> None:
    """
    Rebuild the features on a panel truncated at the anchor. If any block reached past it,
    the two matrices differ. Cheap insurance against the failure mode that has already
    cost us most of a day (DATA.md §10).
    """
    # The cache MUST be disabled here. This function rebuilds features on a panel with the
    # future erased; a cache hit would return the original, un-erased values, the comparison
    # would pass trivially, and the guard would stop guarding without any visible symptom.
    import features as _f
    _was = _f.CACHE_ENABLED
    _f.enable_cache(False)
    arrays = p.prefix_arrays() + p.ewm_arrays() + p.raw_arrays()
    saved = [a.copy() for a in arrays]
    try:
        for a in arrays:
            # prefix arrays have n_days+1 columns and are frozen at their value AT the anchor;
            # raw/EWM arrays have n_days columns and everything after the anchor is zeroed,
            # which is what "the future does not exist" means for a daily series.
            if a.shape[1] > p.n_days:
                a[:, anchor + 2:] = a[:, anchor + 1][:, None]
            elif a.dtype.kind == "f" and a is not None:
                a[:, anchor + 1:] = 0.0
        X2, _ = build(p, anchor, keep, blocks, mw)
    finally:
        for a, s_ in zip(arrays, saved):
            a[...] = s_
        _f.enable_cache(_was)
    bad = ~np.isclose(np.nan_to_num(X), np.nan_to_num(X2), rtol=1e-9, atol=1e-9)
    if bad.any():
        cols = np.where(bad.any(axis=0))[0]
        raise AssertionError(f"LOOK-AHEAD at anchor {anchor}: feature columns {cols.tolist()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--screen", action="store_true", help="2 folds only (tier=screen)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    exp_id = cfg["exp_id"]
    t0 = time.time()
    log(f"\n=== {exp_id} : {cfg['change']} ===")
    log(f"    parent={cfg['parent_id']}  blocks={cfg['feature_blocks']}  seed={cfg['seed']}")

    spec = json.loads((ROOT / "data" / "fold_spec.json").read_text())
    folds = pl.read_parquet(ROOT / "data" / "folds.parquet")
    fold_ids = sorted(folds["fold_id"].unique().to_list())
    if args.screen:
        fold_ids = fold_ids[-2:]
    tier = "screen" if args.screen else cfg.get("tier", "confirm")

    p = Panel()
    if cfg.get("feature_cache"):
        import features as _feat
        _feat.enable_cache(True, force=bool(cfg.get("feature_cache_force")))
        if _feat.CACHE_ENABLED:                  # enable_cache refuses if there is no room
            log(f"    feature cache ON  gen={_feat._code_hash()}  "
                f"budget={_feat.CACHE_BUDGET_GB:.0f}G  dir={_feat.CACHE_DIR}")
        else:
            log("    feature cache requested but REFUSED -- building features directly")
    import lightgbm as lgb

    oof, per_fold, per_fold_naive, best_iters = [], [], [], []
    for k in fold_ids:
        fs = spec["folds"][k]
        va = date.fromisoformat(fs["valid_anchor"])
        vai = p.idx(va)
        tr_anchors = [date.fromisoformat(x) for x in fs["train_anchors"]]
        MH = int(cfg.get("min_history_days", 0))
        if MH:
            # Extend the training anchors EARLIER than the frozen 90-day-history default.
            # Feature windows longer than the available history are simply truncated, so the
            # early anchors carry shorter effective lookbacks -- the "variable-length windows"
            # idea. This lets anchor 2025-02-13 in, whose target (14 Feb - 15 Mar 2025) is the
            # only observation of the gift season in TARGET position that the data contains.
            # Validation populations and targets are untouched, so folds stay frozen (rule 3).
            latest = va - timedelta(days=30)
            earliest = p.dmin + timedelta(days=MH - 1)
            tr_anchors, a_ = [], latest
            while a_ >= earliest:
                tr_anchors.append(a_); a_ -= timedelta(days=7)
            tr_anchors = sorted(tr_anchors)
        if cfg.get("max_train_anchors"):
            tr_anchors = tr_anchors[-int(cfg["max_train_anchors"]):]
        # The early-stopping anchor must be separated from the fit anchors by the full
        # horizon. With a 7-day anchor grid and 365-day feature lookbacks, an ES anchor only
        # 7 days after the last fit anchor is a NEAR-DUPLICATE of training rows (features
        # differ by 7/365 days, targets overlap 23 of 30 days), so its loss falls
        # monotonically and early stopping never fires -- exactly what e0016 showed
        # (best_iteration 30000/30000 while validation RMSLE degraded by +0.044).
        # README.md: duplicate/near-duplicate rows must not straddle train and valid.
        es_anchor = tr_anchors[-1]                      # = A_valid - 30, never the fold itself
        gap = int(cfg.get("es_gap_days", 0))
        FIXED = int(cfg.get("fixed_rounds", 0))
        if FIXED > 0:
            # The ES diagnostic (reports/eda/diag_es.json) showed the validation curve is
            # FLAT -- iterations 80..740 sit within 0.001 of the optimum on every fold -- and
            # total regret from ES noise is only +0.00011. So the holdout buys almost nothing
            # in tree count, while costing ~4 anchors of training data; on fold 4 that cost
            # (+0.004) exceeded the benefit. With a fixed round count we can train on
            # everything. Inseparable change (README.md): dropping the holdout and fixing
            # the rounds cannot be tested apart.
            fit_anchors = tr_anchors
        else:
            fit_anchors = [a for a in tr_anchors[:-1] if (es_anchor - a).days >= gap]
        if not fit_anchors:
            raise ValueError(f"fold {k}: es_gap_days={gap} leaves no fit anchors")

        Xtr, ytr = [], []
        for a in fit_anchors:
            ai = p.idx(a)
            keep = p.active_in(ai - 29, ai)
            p.set_floor(ai, cfg.get("feature_truncate_days"))
            X, names = build(p, ai, keep, cfg["feature_blocks"], cfg.get("feature_max_window"))
            Xtr.append(X); ytr.append(p.target(ai)[keep])
        AUX = list(cfg.get("aux_horizons", []))
        if AUX:
            # Multi-task alternative to stacking: instead of feeding a short-horizon MODEL's
            # prediction in, add the short-horizon TARGETS as extra training rows with the
            # horizon as a feature. Leak-free by construction (each row's label is its own
            # horizon, all in the past), and it costs no training data -- unlike the nested
            # stacking scheme, which must hold anchors out to stay clean.
            Xa, ya = [], []
            for hh in AUX + [30]:
                for a in fit_anchors:
                    ai2 = p.idx(a); k2 = p.active_in(ai2 - 29, ai2)
                    p.set_floor(ai2, cfg.get("feature_truncate_days"))
                    Xb2, _ = build(p, ai2, k2, cfg["feature_blocks"], cfg.get("feature_max_window"))
                    Xa.append(np.column_stack([Xb2, np.full(Xb2.shape[0], hh, np.float32)]))
                    ya.append(p.wsum("gmv", ai2 + 1, ai2 + hh)[k2])
            Xtr = np.concatenate(Xa); ytr_raw = np.concatenate(ya)
            names = names + ["horizon"]
        else:
            Xtr = np.concatenate(Xtr); ytr_raw = np.concatenate(ytr)
        log(f"    fold {k}: {len(fit_anchors)} fit anchors ({fit_anchors[0]}..{fit_anchors[-1]}), "
            f"ES anchor {es_anchor} (gap {(es_anchor - fit_anchors[-1]).days}d), "
            f"{Xtr.shape[0]:,} train rows")
        ytr = np.log1p(ytr_raw)

        if FIXED > 0:
            Xes, yes, yes_raw = None, None, None
        else:
            ei = p.idx(es_anchor)
            ekeep = p.active_in(ei - 29, ei)
            p.set_floor(ei, cfg.get("feature_truncate_days"))
            Xes, _ = build(p, ei, ekeep, cfg["feature_blocks"], cfg.get("feature_max_window"))
            yes_raw = p.target(ei)[ekeep]; yes = np.log1p(yes_raw)

        vkeep = p.active_in(vai - 29, vai)
        p.set_floor(vai, cfg.get("feature_truncate_days"))
        Xva, names = build(p, vai, vkeep, cfg["feature_blocks"], cfg.get("feature_max_window"))
        if k == fold_ids[0]:
            # guard must run on the raw block output; the horizon column is appended after,
            # otherwise the rebuilt matrix has one column fewer and isclose() cannot broadcast
            assert_no_lookahead(p, vai, Xva, vkeep, cfg["feature_blocks"], cfg.get("feature_max_window"))
        if cfg.get("feature_exclude_patterns"):
            # anchor_drift.py found the test cut-off is 3.9x further from the training
            # cut-offs than they are from each other, and attributed it almost entirely to
            # LIFETIME and 365-day features: they grow mechanically with calendar time, so a
            # user at the test anchor has 409 days of history against ~290 at the last
            # training anchor. Thresholds learned on them land in a different part of the
            # distribution at test. Dropping them trades CV for stationarity.
            import re as _re
            pats = [_re.compile(x) for x in cfg["feature_exclude_patterns"]]
            sel = [i for i, n in enumerate(names) if not any(q.search(n) for q in pats)]
            log(f"    exclude patterns: dropping {len(names) - len(sel)} of {len(names)} features")
            Xva = Xva[:, sel]; Xtr = Xtr[:, sel]
            if not (FIXED > 0):
                Xes = Xes[:, sel]
            names = [names[i] for i in sel]
        if cfg.get("feature_whitelist"):
            WL = json.loads(Path(cfg["feature_whitelist"]).read_text())
            sel = [i for i, n in enumerate(names) if n in set(WL)]
            log(f"    whitelist: keeping {len(sel)} of {len(names)} features")
            Xva = Xva[:, sel]; Xtr = Xtr[:, sel]
            if not (FIXED > 0):
                Xes = Xes[:, sel]
            names = [names[i] for i in sel]
        if cfg.get("aux_horizons"):
            Xva = np.column_stack([Xva, np.full(Xva.shape[0], 30, np.float32)])
            names = names + ["horizon"]
            log(f"    look-ahead check passed; {Xva.shape[1]} features")
        fv = folds.filter(pl.col("fold_id") == k).sort("user_id")
        yva = fv["target"].to_numpy()
        assert np.array_equal(fv["user_id"].to_numpy(), p.users[vkeep]), "fold population drift"

        params = dict(cfg["lgb_params"]); params["seed"] = cfg["seed"]
        cb = [lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False),
              lgb.log_evaluation(0)]

        if cfg.get("model") == "magnitude_only":
            # IDEAS.md §I13.  The magnitude term captures only 80.2% of its measured ceiling
            # (0.4814 achieved vs 0.6001) against 89.6% for the buy flag -- the one term the
            # project never scored against a bound, because everything was scored END-TO-END.
            # This trains the magnitude half ALONE, on buyers only, and emits it raw so it can
            # be scored as corr(L, .|Z=1) on the buyer subset.  It is deliberately NOT a
            # submittable model: on zero-target users its output is meaningless.  That is the
            # point -- e0010 multiplied the two halves and scored the product, which is why the
            # term stayed invisible.  `magnitude_all` is the same code path training on ALL
            # users, i.e. the same-harness control the comparison actually needs.
            pos = ytr_raw > 0
            if cfg.get("magnitude_train_on") == "all":
                trX, trY = Xtr, ytr                       # control arm
            else:
                trX, trY = Xtr[pos], ytr[pos]             # treatment arm
            log(f"    magnitude_only: training on {trX.shape[0]:,} rows "
                f"({'ALL users [CONTROL]' if cfg.get('magnitude_train_on') == 'all' else 'BUYERS only'})")
            if FIXED > 0:
                model = lgb.train(params, lgb.Dataset(trX, trY, feature_name=names),
                                  num_boost_round=FIXED, callbacks=[lgb.log_evaluation(0)])
                best_iters.append(FIXED)
            else:
                pose = yes_raw > 0
                esX, esY = (Xes, yes) if cfg.get("magnitude_train_on") == "all" else (Xes[pose], yes[pose])
                model = lgb.train(params, lgb.Dataset(trX, trY, feature_name=names),
                                  num_boost_round=cfg["num_boost_round"],
                                  valid_sets=[lgb.Dataset(esX, esY, feature_name=names)],
                                  callbacks=cb)
                best_iters.append(model.best_iteration)
            mv = model.predict(Xva, num_iteration=getattr(model, "best_iteration", None) or FIXED)
            pred = np.maximum(np.expm1(mv), 0.0)
        elif cfg.get("model", "single") == "two_part":
            # RMSLE is minimised by predicting E[log1p(y)|x], and with a hurdle split
            #     E[L|x] = P(y>0|x) * E[L|x, y>0]
            # so the two parts multiply in LOG space -- not p * E[y|y>0] in linear space.
            cp = {**params, "objective": "binary", "metric": "binary_logloss"}
            clf = lgb.train(cp, lgb.Dataset(Xtr, (ytr_raw > 0).astype(np.int8), feature_name=names),
                            num_boost_round=cfg["num_boost_round"],
                            valid_sets=[lgb.Dataset(Xes, (yes_raw > 0).astype(np.int8),
                                                    feature_name=names)], callbacks=cb)
            pos, pose = ytr_raw > 0, yes_raw > 0
            reg = lgb.train(params, lgb.Dataset(Xtr[pos], ytr[pos], feature_name=names),
                            num_boost_round=cfg["num_boost_round"],
                            valid_sets=[lgb.Dataset(Xes[pose], yes[pose], feature_name=names)],
                            callbacks=cb)
            model = reg
            best_iters.append((clf.best_iteration, reg.best_iteration))
            ph = clf.predict(Xva, num_iteration=clf.best_iteration)
            mh = reg.predict(Xva, num_iteration=reg.best_iteration)
            pred = np.maximum(np.expm1(ph * mh), 0.0)
        elif cfg.get("model") == "hlgauss":
            # IDEAS.md §I1.  Same estimand, different loss geometry: discretise L = log1p(y),
            # train softmax cross-entropy against Gaussian-smoothed bin targets, read out
            #     M = sum_k p_k * c_k   ~ E[L|x]
            # which is the functional RMSLE elicits.  This is NOT ZILN/OptDist/the hurdle: those
            # estimate E[y|x] and were killed here for that reason (PAPERS_FEATURES_AND_IDEAS
            # §0.1).  The ONLY thing that differs from the branch below is the objective.
            import hlgauss as HL
            Kq = int(cfg.get("hl_bins", 12))
            edges, centres, vmax = HL.make_bins(ytr, Kq)
            K = len(centres)
            sigma = float(cfg.get("hl_sigma_ratio", 0.75)) * (vmax / Kq)
            qtr = HL.soft_targets(ytr, edges, sigma).astype(np.float32)
            init = HL.prior_init(qtr)
            hard = np.clip(np.searchsorted(edges, np.clip(ytr, 0, vmax), "right") - 1, 0, K - 1)
            # gate the run: a mis-specified hand-port trains, scores and returns a plausible
            # null (BACKLOG.md 2026-08-20).  Cheap -- 3 rounds on a 20k-row slice.
            md = HL.port_exact_check(Xtr[:20000], hard[:20000], K, params)
            log(f"    hlgauss: K={K} (requested {Kq}) on [0,{vmax:.3f}] sigma={sigma:.4f}"
                f"  PORT GATE max|d raw|={md:.2e} {'PASS' if md < 1e-6 else 'FAIL'}")
            if md >= 1e-6:
                raise RuntimeError(f"hlgauss objective mis-specified (port gate {md:.3e})")
            hp = {**params, "objective": HL.hl_objective(qtr, K), "num_class": K,
                  "metric": "None"}
            hp.pop("early_stopping_round", None)
            dtr = lgb.Dataset(Xtr, np.zeros(Xtr.shape[0]), feature_name=names,
                              init_score=np.tile(init, (Xtr.shape[0], 1)).ravel(order="F"))

            def _readout(raw):
                r = raw.reshape(-1, K) if raw.ndim == 1 else raw
                return HL.softmax_rows(r + init) @ centres

            if FIXED > 0:
                model = lgb.train(hp, dtr, num_boost_round=FIXED,
                                  callbacks=[lgb.log_evaluation(0)])
                best_iters.append(FIXED)
                bi = FIXED
            else:
                # early-stop on RMSLE of the readout, i.e. the same quantity the L2 arm stops
                # on -- so the arms differ by their objective and by nothing else.
                dva = lgb.Dataset(Xes, yes, feature_name=names,
                                  init_score=np.tile(init, (Xes.shape[0], 1)).ravel(order="F"))

                def _feval(preds, _d):
                    return "rmsle_readout", float(np.sqrt(np.mean(
                        (_readout(np.asarray(preds)) - yes) ** 2))), False

                model = lgb.train(hp, dtr, num_boost_round=cfg["num_boost_round"],
                                  valid_sets=[dva], feval=_feval, callbacks=cb)
                best_iters.append(model.best_iteration)
                bi = model.best_iteration
            pred = np.maximum(np.expm1(_readout(
                model.predict(Xva, raw_score=True, num_iteration=bi))), 0.0)
        else:
            if FIXED > 0:
                model = lgb.train(params, lgb.Dataset(Xtr, ytr, feature_name=names),
                                  num_boost_round=FIXED, callbacks=[lgb.log_evaluation(0)])
                best_iters.append(FIXED)
            else:
                model = lgb.train(params, lgb.Dataset(Xtr, ytr, feature_name=names),
                                  num_boost_round=cfg["num_boost_round"],
                                  valid_sets=[lgb.Dataset(Xes, yes, feature_name=names)],
                                  callbacks=cb)
                best_iters.append(model.best_iteration)
            pred = np.maximum(np.expm1(model.predict(Xva, num_iteration=model.best_iteration)), 0.0)

        # The naive reference must not depend on the feature set -- `geo3` is filtered out of
        # short-lookback members (it needs 90 days), which used to crash the run. Compute it
        # from the panel instead, so every member is scored against the same baseline.
        if "geo3" in names:
            naive = np.maximum(Xva[:, names.index("geo3")].astype(np.float64), 0.0)
        else:
            _fl = p.floor; p.floor = 0
            blks = [p.wsum("gmv", vai - 29 - 30 * j, vai - 30 * j) for j in range(3)]
            p.floor = _fl
            naive = np.maximum(np.expm1(np.mean([np.log1p(b) for b in blks], axis=0))[vkeep], 0.0)

        s, sn = rmsle(yva, pred), rmsle(yva, naive)
        per_fold.append(s); per_fold_naive.append(sn)
        oof.append(pl.DataFrame({"fold_id": np.full(yva.size, k, np.int8),
                                 "anchor_date": pl.Series("anchor_date", [va] * yva.size, dtype=pl.Date),
                                 "user_id": fv["user_id"].to_numpy(),
                                 "y_true": yva, "y_pred": pred, "y_naive": naive}))
        log(f"    fold {k} {va}  n={yva.size:>7,} trees={model.best_iteration:>4d}  "
            f"rmsle={s:.5f}  naive={sn:.5f}  delta={s - sn:+.5f}")

    oof = pl.concat(oof)
    (ROOT / "oof").mkdir(exist_ok=True)
    oof.write_parquet(ROOT / "oof" / f"{exp_id}.parquet")

    pf = np.array(per_fold); pfn = np.array(per_fold_naive)
    agg = score_all(oof["y_true"].to_numpy(), oof["y_pred"].to_numpy())
    agg_n = score_all(oof["y_true"].to_numpy(), oof["y_naive"].to_numpy())
    wins = int((pf < pfn).sum())
    runtime = (time.time() - t0) / 60

    imp = sorted(zip(names, model.feature_importance("gain")), key=lambda x: -x[1])[:15]

    log(f"\n  cv_mean = {pf.mean():.5f} +/- {pf.std():.5f}   folds {np.round(pf, 5).tolist()}")
    log(f"  naive   = {pfn.mean():.5f} +/- {pfn.std():.5f}   folds {np.round(pfn, 5).tolist()}")
    log(f"  delta vs naive = {pf.mean() - pfn.mean():+.5f}   wins {wins}/{len(pf)} folds")
    log(f"  last fold (most test-like) = {pf[-1]:.5f}  (naive {pfn[-1]:.5f}, "
        f"delta {pf[-1] - pfn[-1]:+.5f})")
    log(f"  tie-breakers: gini_pred={agg['gini_pred']:.4f} (true {agg['gini_true']:.4f})  "
        f"total_rel_err={agg['total_rel_err']:+.4f}  [naive: {agg_n['gini_pred']:.4f}, "
        f"{agg_n['total_rel_err']:+.4f}]")
    log(f"  best_iteration per fold: {best_iters}   runtime {runtime:.1f} min")
    if cfg.get("feature_cache"):
        import features as _feat
        cs = _feat.cache_stats()
        log(f"  cache: {cs['hit']} hits / {cs['miss']} misses ({cs['hit_rate']:.0%}), "
            f"{cs['written']} written ({cs['bytes'] / 1e9:.1f}G), "
            f"generation now {cs['gen_gb']:.1f}G"
            + (f", {cs['skipped_budget']} writes skipped on budget"
               if cs["skipped_budget"] else ""))
    log(f"\n  top-15 features by gain:")
    for n, g in imp:
        log(f"    {n:26s} {g:>14,.0f}")

    row = {
        "exp_id": exp_id, "parent_id": cfg["parent_id"],
        "date": datetime.now().isoformat(timespec="seconds"),
        "approach": cfg["approach"], "change": cfg["change"], "tier": tier,
        "n_features": len(names), "cv_mean": round(float(pf.mean()), 5),
        "cv_std": round(float(pf.std()), 5),
        "folds": json.dumps([round(float(x), 5) for x in pf]),
        "delta": round(float(pf.mean() - pfn.mean()), 5),
        "significant": "yes" if wins >= 4 or abs(pf.mean() - pfn.mean()) > 2 * pf.std() else "no",
        "lb": "", "runtime_min": round(runtime, 1), "seed": cfg["seed"],
        "config": args.config, "verdict": cfg.get("verdict", ""),
        "gini_pred": round(agg["gini_pred"], 4),
        "total_rel_err": round(agg["total_rel_err"], 4),
        "notes": cfg.get("notes", ""),
        "best_iters": json.dumps(best_iters),
    }
    rd = ROOT / "runs"; rd.mkdir(exist_ok=True)
    (rd / f"{exp_id}.json").write_text(json.dumps(row, indent=2))
    try:
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "src" / "collect.py")], check=False)
    except Exception:
        pass
    log(f"\n  wrote runs/{exp_id}.json -> experiments.csv")


if __name__ == "__main__":
    main()
