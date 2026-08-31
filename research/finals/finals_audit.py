#!/usr/bin/env python
"""
FINALS AUDIT — inspect every submission file that has a measured public-LB score,
check it is a valid submission, and characterise it in the space the metric lives in.

After affine calibration the score is pure rho:  RMSLE = sd_L * sqrt(1 - rho^2).
So for each scored file we can invert its LB score to a rho on a single scale, and we
can measure directly (no LB needed) how far its log-prediction moments sit from the
probe-solved test truth moments -- the only two free knobs post-processing has.

Outputs:
  1. validity of every scored file (rows, id set, NaN, negatives)
  2. log-space mu / sd vs the probe-solved targets, and the RMSLE cost of the miss
  3. pairwise correlation of the top cluster (how much hedging power a 2nd pick has)
  4. LB-inverted rho and gaps in units of the private-200k noise band
"""
import numpy as np, pandas as pd, hashlib, os

SD_L = 2.3178          # refined probe-solved sd of log1p(y_test)   EXPERIMENTS.md 1i
MU_L = 2.3303          # refined probe-solved mean

# every file that has a measured public LB, newest lineage first
SCORED = [
    ("e0301_usercv48_cal", 1.646456),
    ("e0303_arch4_cal",    1.646483),
    ("e0300_cal",          1.646589),
    ("e0162",              1.646602),
    ("e0161",              1.646670),
    ("e0152",              1.646697),
    ("e0150",              1.646700),
    ("e0361_seqext_cal",   1.646806),
    ("e0201_cal",          1.646831),
    ("e0270_cal",          1.646836),
    ("e0200_cal",          1.646868),
    ("e0302_add_d48_cal",  1.647049),
    ("e0151",              1.647480),
    ("e0201_blend",        1.647923),
    ("e0270_blend",        1.647898),
    ("e0200_blend",        1.648859),
    ("e0141",              1.648800),
    ("e0266_cal",          1.650086),
    ("e0146g",             1.651202),
    ("e0145",              1.653230),
    ("e0140",              1.655200),
    ("e0120",              1.655300),
    ("e0090",              1.655247),
    ("e0064",              1.655900),
    ("e0049",              1.656200),
    ("e0060",              1.656700),
    ("e0020",              1.657800),
    ("e0142",              1.678500),
    ("e0001",              1.676600),
]

def load(name):
    for cand in (f"subs/{name}.csv",):
        if os.path.exists(cand):
            df = pd.read_csv(cand)
            df.columns = [c.strip() for c in df.columns]
            return df
    return None

def main():
    sample = pd.read_csv("sample_submit.csv")
    sample.columns = [c.strip() for c in sample.columns]
    ids_ref = set(sample.iloc[:, 0].astype(np.int64))
    print(f"sample_submit: {len(sample):,} rows, {len(ids_ref):,} unique ids, cols={list(sample.columns)}\n")

    rows, logs = [], {}
    for name, lb in SCORED:
        df = load(name)
        if df is None:
            rows.append(dict(name=name, lb=lb, status="FILE MISSING"))
            continue
        ids = df.iloc[:, 0].astype(np.int64).to_numpy()
        p = df.iloc[:, 1].astype(np.float64).to_numpy()
        ok_ids = (len(ids) == len(ids_ref)) and (set(ids) == ids_ref)
        nan = int(np.isnan(p).sum()); neg = int((p < 0).sum())
        order = np.argsort(ids)
        lp = np.log1p(np.maximum(p[order], 0.0))
        logs[name] = lp
        rho = float(np.sqrt(max(0.0, 1.0 - (lb / SD_L) ** 2)))
        rows.append(dict(name=name, lb=lb, n=len(ids), ids_ok=ok_ids, nan=nan, neg=neg,
                         mu=lp.mean(), sd=lp.std(), rho=rho,
                         md5=hashlib.md5(open(f"subs/{name}.csv", "rb").read()).hexdigest()[:8]))

    R = pd.DataFrame(rows)
    print("== 1. validity + moments of every LB-scored file ==")
    print("   (mu target = %.4f   sd target = optimal shrink, see col 'sd')\n" % MU_L)
    show = R.dropna(subset=["n"]).copy()
    show["mu_err"] = show["mu"] - MU_L
    # RMSLE cost of a pure level miss, second order: b^2 / (2*RMSLE)
    show["cost_mu"] = show["mu_err"] ** 2 / (2 * show["lb"])
    print(show[["name", "lb", "rho", "n", "ids_ok", "nan", "neg", "mu", "sd", "mu_err",
                "cost_mu", "md5"]].to_string(index=False,
          formatters={"lb": "{:.6f}".format, "rho": "{:.6f}".format, "mu": "{:.4f}".format,
                      "sd": "{:.4f}".format, "mu_err": "{:+.4f}".format,
                      "cost_mu": "{:.6f}".format}))
    miss = R[R["status"].notna()] if "status" in R else R.iloc[0:0]
    if len(miss):
        print("\n  MISSING FILES:", list(miss["name"]))

    print("\n== 2. pairwise correlation of the top cluster (log space, all 250k users) ==")
    top = [n for n, _ in SCORED[:12] if n in logs]
    M = np.vstack([logs[n] for n in top])
    C = np.corrcoef(M)
    print(pd.DataFrame(C, index=top, columns=[t[:9] for t in top]).to_string(
        float_format=lambda v: f"{v:.5f}"))

    print("\n== 3. distance of each top file from the top-cluster consensus ==")
    cons = M.mean(axis=0)
    for i, n in enumerate(top):
        d = M[i] - cons
        print(f"  {n:20s}  corr vs consensus {np.corrcoef(M[i],cons)[0,1]:.6f}   "
              f"||dev||/sd {d.std()/M[i].std():.5f}   mean dev {d.mean():+.5f}")

    print("\n== 4. public-LB gaps in units of the private-200k paired noise band ==")
    SD200 = 0.00013     # scripts/robustness.py bootstrap, fold 4, 200k paired RMSLE sd
    SD50  = 0.00025     # same, 50k -- the noise ON the public numbers themselves
    best = R["lb"].min()
    for _, r in R.dropna(subset=["n"]).sort_values("lb").iterrows():
        g = r["lb"] - best
        print(f"  {r['name']:20s} LB {r['lb']:.6f}  gap {g:+.6f}  "
              f"= {g/SD200:5.2f} sd(private-200k)  {g/SD50:5.2f} sd(public-50k)")

if __name__ == "__main__":
    main()
