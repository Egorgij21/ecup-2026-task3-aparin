#!/usr/bin/env python
"""
Causal-forces damping test on OOF (Armstrong & Collopy 1993, adapted).

The mean-reverting geo3 baseline B = log1p(y_naive) is the "no-trend / regressing-force"
prediction. The model's deviation d = M - B (M = log1p(y_pred)) is the trend it extrapolates
beyond mean-reversion. Armstrong's rule: DAMP that deviation where the force is contrary /
regressing -- and for a regressing force, up-spikes (d>0) should be damped MORE than dips.

We fit each damping rule LEAVE-ONE-FOLD-OUT (params on the other 4 folds, applied to the held
fold) and score rho(L, M') per fold. Score is rho after affine calibration (§1b), so rho is the
right target. Controls: g=1 no-op (== M), and d shuffled across users (random damping).

Honest prior: §1q measured E[L|M] ~ M to 0.0010 (conditionally calibrated) and §1t found
per-segment slope b_g ~ 1, so the model is not obviously over-extrapolating on the folds -> we
expect g* ~ 1, null. The asymmetric and tail variants are the genuinely-untested part.
"""
import numpy as np, polars as pl

def rho(a, b): return float(np.corrcoef(a, b)[0, 1])

def main():
    d = pl.read_parquet("oof/e0049.parquet").sort(["fold_id", "user_id"])
    L = np.log1p(d["y_true"].to_numpy())
    M = np.log1p(np.maximum(d["y_pred"].to_numpy(), 0.0))
    B = np.log1p(np.maximum(d["y_naive"].to_numpy(), 0.0))
    fold = d["fold_id"].to_numpy()
    dev = M - B
    folds = sorted(set(fold))
    print(f"  n={L.size:,}  mean dev(M-B)={dev.mean():+.4f}  frac up-spike(d>0)={np.mean(dev>0):.3f}")
    print(f"  baseline rho(L,M)={rho(L,M):.5f}   rho(L,B geo3)={rho(L,B):.5f}\n")

    def lofo(apply_rule, fit_rule):
        """fit_rule(Lo,Mo,Bo,devo)->params on 4 folds; apply_rule(params,M,B,dev)->M' on held fold."""
        rr = []
        for k in folds:
            tr = fold != k; te = fold == k
            params = fit_rule(L[tr], M[tr], B[tr], dev[tr])
            Mp = apply_rule(params, M[te], B[te], dev[te])
            rr.append(rho(L[te], Mp))
        return np.array(rr)

    # --- 1. global linear damp: M' = B + g*dev ; g* = argmax rho on train (grid)
    def fit_global(Lo, Mo, Bo, do):
        gs = np.linspace(0.5, 1.3, 81)
        return gs[np.argmax([rho(Lo, Bo + g*do) for g in gs])]
    r_glob = lofo(lambda g, M, B, dv: B + g*dv, fit_global)
    g_star = fit_global(L, M, B, dev)

    # --- 2. asymmetric (causal-forces): damp up-spikes and dips separately
    def fit_asym(Lo, Mo, Bo, do):
        up, dn = do > 0, do < 0
        best, bp = -9, (1.0, 1.0)
        for gp in np.linspace(0.5, 1.2, 29):
            for gn in np.linspace(0.5, 1.2, 29):
                Mp = Bo + np.where(up, gp*do, np.where(dn, gn*do, 0.0))
                r = rho(Lo, Mp)
                if r > best: best, bp = r, (gp, gn)
        return bp
    def app_asym(p, M, B, dv):
        gp, gn = p
        return B + np.where(dv > 0, gp*dv, np.where(dv < 0, gn*dv, 0.0))
    r_asym = lofo(app_asym, fit_asym)
    gp_s, gn_s = fit_asym(L, M, B, dev)

    # --- 3. tail damp: shrink only large extrapolations |dev|>q by factor lambda
    def fit_tail(Lo, Mo, Bo, do):
        q = np.quantile(np.abs(do), 0.9)
        best, bl = -9, 0.0
        for lam in np.linspace(0, 1, 41):
            excess = np.sign(do)*np.maximum(np.abs(do)-q, 0.0)
            Mp = Mo - lam*excess
            r = rho(Lo, Mp)
            if r > best: best, bl = r, lam
        return (bl, q)
    def app_tail(p, M, B, dv):
        lam, q = p
        return M - lam*np.sign(dv)*np.maximum(np.abs(dv)-q, 0.0)
    r_tail = lofo(app_tail, fit_tail)

    # --- controls
    # no-op: g=1 -> M' = M
    r_noop = np.array([rho(L[fold==k], M[fold==k]) for k in folds])
    # shuffled dev (random damping at the fitted global g)
    rng = np.random.default_rng(0)
    def fit_shuf(Lo, Mo, Bo, do): return g_star
    def app_shuf(g, M, B, dv):
        return B + g*dv[rng.permutation(len(dv))]
    r_shuf = lofo(app_shuf, fit_shuf)

    base = r_noop.mean()
    print(f"  {'rule':28s} {'mean rho':>9s} {'d vs M':>10s}  per-fold d")
    for name, r in [("no-op (M, control)", r_noop),
                    (f"global damp (g*={g_star:.3f})", r_glob),
                    (f"asymmetric (g+={gp_s:.2f} g-={gn_s:.2f})", r_asym),
                    ("tail damp (|d|>p90)", r_tail),
                    ("shuffled-dev (control)", r_shuf)]:
        dd = r - r_noop
        print(f"  {name:28s} {r.mean():>9.5f} {r.mean()-base:>+10.5f}  {np.round(dd,5).tolist()}")
    print(f"\n  sd_L={L.std():.4f}; a rho gain g maps to ~{-L.std()*(-0.6/np.sqrt(1-0.66**2)):.2f}*g RMSLE")
    print("  causal-forces works iff a damp rule beats no-op AND the shuffled control, LOFO.")

if __name__ == "__main__":
    main()
