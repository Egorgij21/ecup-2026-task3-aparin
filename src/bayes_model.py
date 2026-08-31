#!/usr/bin/env python
"""
BAYES_EXP.md B1: covariate-dependent BG/NBD + Gamma-Gamma, fitted by ML-II with JAX gradients.

See `src/bayes_cov.py` for why the per-user latents are marginalised exactly instead of being
approximated by §4.2's mean-field guide.

Parameters, all fitted jointly by L-BFGS-B on the exact marginal log-likelihood:

    log r                                    shared shape of the rate prior
    alpha_0, beta_alpha  (1 + d)             log alpha_u  -- rate scale
    a_0,     beta_a      (1 + d)             log a_u      -- dropout Beta
    b_0,     beta_b      (1 + d)             log b_u
    log p_gg, log q_gg                       shared Gamma-Gamma shapes
    nu_0,    beta_nu     (1 + d)             log nu_u     -- spend scale

`beta_* = 0` recovers e0170 exactly, which is the null this is tested against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                            # noqa: E402
from jax.scipy.special import gammaln as jgammaln                  # noqa: E402

__all__ = ["BayesFit", "fit_bayes", "unpack"]

CLIP = 12.0          # bound on every linear predictor, in log space


def _lin(intercept, beta, X):
    return jnp.clip(intercept + X @ beta, -CLIP, CLIP)


def unpack(p, d: int):
    i = 0
    def take(n):
        nonlocal i
        v = p[i:i + n]; i += n
        return v
    log_r = take(1)[0]
    alpha_0 = take(1)[0]; beta_alpha = take(d)
    a_0 = take(1)[0]; beta_a = take(d)
    b_0 = take(1)[0]; beta_b = take(d)
    log_p = take(1)[0]; log_q = take(1)[0]
    nu_0 = take(1)[0]; beta_nu = take(d)
    assert i == p.shape[0]
    return log_r, alpha_0, beta_alpha, a_0, beta_a, b_0, beta_b, log_p, log_q, nu_0, beta_nu


def _bgnbd_ll(log_r, alpha_0, beta_alpha, a_0, beta_a, b_0, beta_b, X, x, t_x, T):
    """Per-user BG/NBD marginal log-likelihood with covariate-dependent priors."""
    r = jnp.exp(log_r)
    log_alpha = _lin(alpha_0, beta_alpha, X)
    log_a = _lin(a_0, beta_a, X)
    log_b = _lin(b_0, beta_b, X)
    alpha = jnp.exp(log_alpha); a = jnp.exp(log_a); b = jnp.exp(log_b)

    rx = r + x
    ln_A1 = jgammaln(rx) - jgammaln(r) + r * log_alpha
    ln_A2 = jgammaln(a + b) + jgammaln(b + x) - jgammaln(b) - jgammaln(a + b + x)
    ln_A3 = -rx * jnp.log(alpha + T)
    d = x > 0
    # b + x - 1 > 0 whenever x >= 1 and b > 0; the `where` keeps the dead branch finite so
    # reverse-mode AD does not propagate a NaN through it
    bx1 = jnp.where(d, b + x - 1.0, 1.0)
    ln_A4 = jnp.where(d, log_a - jnp.log(bx1) - rx * jnp.log(alpha + t_x), -jnp.inf)
    return ln_A1 + ln_A2 + jnp.logaddexp(ln_A3, ln_A4)


def _gg_ll(log_p, log_q, nu_0, beta_nu, X, n, m, mask):
    """Gamma-Gamma marginal log-likelihood, covariate-dependent scale, on `mask` users."""
    p = jnp.exp(log_p); q = jnp.exp(log_q)
    nu = jnp.exp(_lin(nu_0, beta_nu, X))
    # `safe_n` must reach EVERY use of n, `pn` included.  Masked-out users have n = 0, and
    # gammaln(0) = inf with digamma(0) = NaN; reverse-mode AD then multiplies that NaN by the
    # mask's zero cotangent and gets NaN, so the forward value looks fine and the gradient is
    # silently destroyed.  This is the standard JAX where-trick failure and it cost a debug
    # cycle here: L-BFGS-B stopped at iteration 1 with a finite objective.
    safe_m = jnp.where(mask, m, 1.0)
    safe_n = jnp.where(mask, n, 1.0)
    pn = p * safe_n
    v = (jgammaln(pn + q) - jgammaln(pn) - jgammaln(q)
         + q * jnp.log(nu) + (pn - 1.0) * jnp.log(safe_m) + pn * jnp.log(safe_n)
         - (pn + q) * jnp.log(nu + safe_m * safe_n))
    return jnp.where(mask, v, 0.0)


@dataclass
class BayesFit:
    params: np.ndarray
    d: int
    names: list[str]
    nll: float
    n: int
    converged: bool
    n_iter: int
    which: str = "full"
    extra: dict = field(default_factory=dict)

    # -- per-user parameter arrays -----------------------------------------
    def arrays(self, X: np.ndarray) -> dict:
        (log_r, alpha_0, beta_alpha, a_0, beta_a, b_0, beta_b,
         log_p, log_q, nu_0, beta_nu) = unpack(jnp.asarray(self.params), self.d)
        cl = lambda i, bt: np.clip(np.asarray(i) + X @ np.asarray(bt), -CLIP, CLIP)
        return {"r": float(np.exp(log_r)),
                "alpha": np.exp(cl(alpha_0, beta_alpha)),
                "a": np.exp(cl(a_0, beta_a)),
                "b": np.exp(cl(b_0, beta_b)),
                "p_gg": float(np.exp(log_p)), "q_gg": float(np.exp(log_q)),
                "nu": np.exp(cl(nu_0, beta_nu))}

    def summary(self, X: np.ndarray) -> dict:
        A = self.arrays(X)
        return {"r": A["r"], "mean_alpha": float(A["alpha"].mean()),
                "mean_a": float(A["a"].mean()), "mean_b": float(A["b"].mean()),
                "mean_a_plus_b": float((A["a"] + A["b"]).mean()),
                "p_gg": A["p_gg"], "q_gg": A["q_gg"], "mean_nu": float(A["nu"].mean()),
                "nll": self.nll, "n_iter": self.n_iter, "converged": self.converged}


def fit_bayes(X, x, t_x, T, n_gg, m_x, which: str = "full",
              init: np.ndarray | None = None, maxiter: int = 600,
              verbose: bool = False) -> BayesFit:
    """ML-II fit.  `which` selects which latents get covariates:

        "none"   -- beta_* all zero (reproduces e0170, the null)
        "lam"    -- covariates on the rate prior only
        "full"   -- covariates on rate, dropout and spend  (BAYES_EXP B1)
    """
    X = np.asarray(X, np.float64)
    n_users, d = X.shape
    Xj = jnp.asarray(X)
    xj, txj, Tj = jnp.asarray(x, jnp.float64), jnp.asarray(t_x, jnp.float64), jnp.asarray(T, jnp.float64)
    nj, mj = jnp.asarray(n_gg, jnp.float64), jnp.asarray(m_x, jnp.float64)
    maskj = jnp.asarray((np.asarray(n_gg) >= 2) & (np.asarray(m_x) > 0))

    # a free-beta mask so one code path serves all three variants
    use = {"none": (0, 0, 0), "lam": (1, 0, 0), "full": (1, 1, 1)}[which]
    gate = jnp.concatenate([
        jnp.array([1.0]),                                   # log r
        jnp.array([1.0]), jnp.full(d, float(use[0])),       # alpha
        jnp.array([1.0]), jnp.full(d, float(use[1])),       # a
        jnp.array([1.0]), jnp.full(d, float(use[1])),       # b
        jnp.array([1.0, 1.0]),                              # p, q
        jnp.array([1.0]), jnp.full(d, float(use[2])),       # nu
    ])

    def nll(p):
        p = p * gate
        (log_r, alpha_0, beta_alpha, a_0, beta_a, b_0, beta_b,
         log_p, log_q, nu_0, beta_nu) = unpack(p, d)
        ll = _bgnbd_ll(log_r, alpha_0, beta_alpha, a_0, beta_a, b_0, beta_b, Xj, xj, txj, Tj)
        lg = _gg_ll(log_p, log_q, nu_0, beta_nu, Xj, nj, mj, maskj)
        return -(jnp.sum(ll) + jnp.sum(lg))

    val_grad = jax.jit(jax.value_and_grad(nll))

    def f(p):
        v, g = val_grad(jnp.asarray(p))
        v = float(v)
        g = np.asarray(g, np.float64)
        # check BOTH -- a finite objective with a NaN gradient stops L-BFGS-B at iteration 1
        # and reports success, which is exactly how the safe_n bug above hid itself
        if not np.isfinite(v) or not np.all(np.isfinite(g)):
            return 1e15, np.zeros_like(p)
        return v, g

    n_par = 1 + 2 * (1 + d) + 1 + (1 + d) + 2 + (1 + d) - (1 + d)  # = 1 + 3*(1+d) + 2 + (1+d)
    n_par = 1 + (1 + d) * 4 + 2
    if init is None:
        p0 = np.zeros(n_par)
        p0[0] = np.log(1.1)                     # r
        p0[1] = np.log(18.0)                    # alpha_0
        p0[1 + (1 + d)] = np.log(0.02)          # a_0
        p0[1 + 2 * (1 + d)] = np.log(2.4)       # b_0
        p0[1 + 3 * (1 + d)] = np.log(1.0)       # log p_gg
        p0[1 + 3 * (1 + d) + 1] = np.log(3.0)   # log q_gg
        p0[1 + 3 * (1 + d) + 2] = np.log(110.0)  # nu_0
    else:
        p0 = np.asarray(init, np.float64).copy()
    assert p0.size == n_par, (p0.size, n_par)

    res = minimize(f, p0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxfun": maxiter * 2, "ftol": 1e-13,
                            "gtol": 1e-8})
    params = np.asarray(res.x) * np.asarray(gate)
    if verbose:
        print(f"    {which}: nll {res.fun:,.1f} in {res.nit} iters ({res.message})")
    return BayesFit(params, d, [], float(res.fun), n_users, bool(res.success), int(res.nit),
                    which)


def fit_bayes_loc(X, x, t_x, T, n_gg, m_x, b0: BayesFit, maxiter: int = 2000
                  ) -> BayesFit:
    """B1 with §4.2's variance floor, in the form this parameterisation actually needs.

    WHY THIS EXISTS.  Fitting all of (r, a, b, p_gg, q_gg) freely alongside covariates does
    not converge -- `r` climbs past 140 and is still rising at 2000 iterations, i.e. the
    Gamma prior on the rate is collapsing to a point mass and the per-user random effect is
    being optimised out of existence.  The optimum is on the boundary (r -> inf) and no
    iteration count reaches it.

    I previously argued that marginalising the latents analytically removed §4.2's need for
    a variance floor because "there is no free sigma to collapse".  **That was wrong.**  The
    role of sigma is played by `1/r` and `1/(a+b)`, and they collapse exactly the same way --
    the hazard was renamed, not removed.

    The fix is §4.2's floor, stated in this parameterisation: freeze every DISPERSION at its
    B0 value and let covariates move only the LOCATIONS.

        r, p_gg, q_gg, conc = a + b     frozen at B0
        log alpha_u   = alpha_0 + X . beta_alpha        (rate location)
        logit theta_u = th_0    + X . beta_th           (dropout mean, concentration fixed)
        log nu_u      = nu_0    + X . beta_nu           (spend location)

    3(1+d) free parameters instead of 4(1+d)+3, a bounded objective, and the shrinkage that
    made B0 work is preserved by construction.
    """
    X = np.asarray(X, np.float64)
    n_users, d = X.shape
    A0 = b0.arrays(X)
    r_f = float(A0["r"])
    a_f = float(np.mean(A0["a"])); b_f = float(np.mean(A0["b"]))
    conc_f = a_f + b_f
    th_f = a_f / conc_f
    p_f, q_f = float(A0["p_gg"]), float(np.mean(A0["q_gg"]))
    nu_f = float(np.mean(A0["nu"]))

    Xj = jnp.asarray(X)
    xj, txj, Tj = (jnp.asarray(v, jnp.float64) for v in (x, t_x, T))
    nj, mj = jnp.asarray(n_gg, jnp.float64), jnp.asarray(m_x, jnp.float64)
    maskj = jnp.asarray((np.asarray(n_gg) >= 2) & (np.asarray(m_x) > 0))
    log_r_f, log_p_f, log_q_f = np.log(r_f), np.log(p_f), np.log(q_f)

    def split(p):
        alpha_0 = p[0]; beta_alpha = p[1:1 + d]
        th_0 = p[1 + d]; beta_th = p[2 + d:2 + 2 * d]
        nu_0 = p[2 + 2 * d]; beta_nu = p[3 + 2 * d:3 + 3 * d]
        return alpha_0, beta_alpha, th_0, beta_th, nu_0, beta_nu

    def nll(p):
        alpha_0, beta_alpha, th_0, beta_th, nu_0, beta_nu = split(p)
        theta = jax.nn.sigmoid(jnp.clip(th_0 + Xj @ beta_th, -CLIP, CLIP))
        a_u = theta * conc_f
        b_u = (1.0 - theta) * conc_f
        ll = _bgnbd_ll_ab(log_r_f, _lin(alpha_0, beta_alpha, Xj), a_u, b_u, xj, txj, Tj)
        lg = _gg_ll(log_p_f, log_q_f, nu_0, beta_nu, Xj, nj, mj, maskj)
        return -(jnp.sum(ll) + jnp.sum(lg))

    val_grad = jax.jit(jax.value_and_grad(nll))

    def f(p):
        v, g = val_grad(jnp.asarray(p))
        v = float(v); g = np.asarray(g, np.float64)
        if not np.isfinite(v) or not np.all(np.isfinite(g)):
            return 1e15, np.zeros_like(p)
        return v, g

    p0 = np.zeros(3 + 3 * d)
    p0[0] = np.log(float(np.mean(A0["alpha"])))
    p0[1 + d] = float(np.log(th_f / (1 - th_f)))
    p0[2 + 2 * d] = np.log(nu_f)
    res = minimize(f, p0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxfun": maxiter * 2, "ftol": 1e-13,
                            "gtol": 1e-8})

    # repack into the standard layout so BayesFit.arrays() works unchanged
    alpha_0, beta_alpha, th_0, beta_th, nu_0, beta_nu = split(np.asarray(res.x))
    theta = 1.0 / (1.0 + np.exp(-np.clip(th_0 + X @ beta_th, -CLIP, CLIP)))
    full = np.zeros(1 + (1 + d) * 4 + 2)
    full[0] = log_r_f
    full[1] = alpha_0; full[2:2 + d] = beta_alpha
    full[1 + 3 * d + 3] = log_p_f; full[1 + 3 * d + 4] = log_q_f
    full[1 + 3 * d + 5] = nu_0; full[2 + 3 * d + 5:2 + 4 * d + 5] = beta_nu
    fit = BayesFit(full, d, [], float(res.fun), n_users, bool(res.success), int(res.nit), "loc")
    fit.extra = {"a_u": theta * conc_f, "b_u": (1.0 - theta) * conc_f, "conc": conc_f}
    return fit


def _bgnbd_ll_ab(log_r, log_alpha, a, b, x, t_x, T):
    """BG/NBD log-likelihood taking a, b directly (not via log-linear predictors)."""
    r = jnp.exp(log_r)
    alpha = jnp.exp(log_alpha)
    rx = r + x
    ln_A1 = jgammaln(rx) - jgammaln(r) + r * log_alpha
    ln_A2 = jgammaln(a + b) + jgammaln(b + x) - jgammaln(b) - jgammaln(a + b + x)
    ln_A3 = -rx * jnp.log(alpha + T)
    d = x > 0
    bx1 = jnp.where(d, b + x - 1.0, 1.0)
    ln_A4 = jnp.where(d, jnp.log(a) - jnp.log(bx1) - rx * jnp.log(alpha + t_x), -jnp.inf)
    return ln_A1 + ln_A2 + jnp.logaddexp(ln_A3, ln_A4)


def _self_test() -> None:
    """`which='none'` must reproduce src/btyd.py's independent scipy fit."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from btyd import fit_bgnbd, fit_gg

    rng = np.random.default_rng(0)
    n = 40000
    x = rng.poisson(2.0, n).astype(np.float64)
    T = rng.uniform(60, 400, n)
    t_x = np.where(x > 0, rng.uniform(0, 1, n) * T, 0.0)
    n_buy = x + 1
    m_x = rng.gamma(2.0, 30.0, n)
    X = rng.normal(size=(n, 5))

    ref_bg = fit_bgnbd(x, t_x, T)
    ref_gg = fit_gg(n_buy, m_x)
    fit = fit_bayes(X, x, t_x, T, n_buy, m_x, which="none")
    A = fit.arrays(X)
    print(f"  scipy  BG/NBD r={ref_bg.r:.4f} alpha={ref_bg.alpha:.4f} a={ref_bg.a:.4f} b={ref_bg.b:.4f}")
    print(f"  jax    BG/NBD r={A['r']:.4f} alpha={A['alpha'][0]:.4f} a={A['a'][0]:.4f} b={A['b'][0]:.4f}")
    print(f"  scipy  GG p={ref_gg.p:.4f} q={ref_gg.q:.4f} nu={ref_gg.nu:.3f}")
    print(f"  jax    GG p={A['p_gg']:.4f} q={A['q_gg']:.4f} nu={A['nu'][0]:.3f}")
    tot_ref = ref_bg.nll + ref_gg.nll
    print(f"  nll: scipy {tot_ref:,.2f}  jax {fit.nll:,.2f}  diff {fit.nll - tot_ref:+.3f}")
    assert abs(fit.nll - tot_ref) < 0.05 * abs(tot_ref) ** 0.0 + 5.0, "the two fits disagree"

    # covariates must strictly improve the in-sample likelihood (nested models)
    f_lam = fit_bayes(X, x, t_x, T, n_buy, m_x, which="lam")
    f_full = fit_bayes(X, x, t_x, T, n_buy, m_x, which="full")
    print(f"  nll none {fit.nll:,.1f} -> lam {f_lam.nll:,.1f} -> full {f_full.nll:,.1f}")
    assert f_lam.nll <= fit.nll + 1e-3 and f_full.nll <= f_lam.nll + 1e-3, "nesting violated"
    print("src/bayes_model.py: all self-tests passed")


if __name__ == "__main__":
    _self_test()
