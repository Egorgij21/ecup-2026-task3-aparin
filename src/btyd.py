#!/usr/bin/env python
"""
BTYD: BG/NBD (timing) + Gamma-Gamma (value), fitted by maximum likelihood.

Spec: BTYD.md.  Strategy: CLAUDE.md.  Measured data facts: DATA.md.

Everything here is per-anchor and causal by construction: the RFM summary is built by
filtering the event table to `day_index <= anchor` before any aggregation, so a look-ahead
is structurally impossible rather than merely unlikely -- `assert_no_lookahead` below
re-derives the summary from a truncated table and requires bit-identical output anyway.

Two prediction functionals are produced, and the difference between them is the only thing
a generative model brings to an RMSLE task (BTYD.md §3.3):

    log1p(E[y])     the naive BTYD output, E[X(30)] * E[M]
    E[log1p(y)]     the Bayes-optimal prediction under RMSLE, by simulating the fitted
                    process forward and averaging in LOG space

Deviations from BTYD.md, both deliberate and both reported in the results:

  1. §3.2 writes the Gamma-Gamma credibility weight as `(q-1)/(p*x + q-1)` with `x` the
     REPEAT count, while §2 defines `m_x` as the mean over ALL buy-days (x+1 of them).
     Those two are inconsistent: the weight must use the number of observations that m_x
     actually averages.  We use `n = x + 1` and expose the literal-spec variant as
     `gg_count="repeat"` so the choice is measurable rather than assumed.
  2. §3.3's simulation sketch draws `n ~ Poisson(lambda*30) * alive`, which lets a live
     customer buy for the whole horizon and so overstates E[X(30)] -- the customer can also
     die DURING the 30 days.  We simulate the actual BG/NBD forward process (Exp(lambda)
     interarrivals, dropout with probability p after each transaction).  This is what makes
     §8's "MC must reproduce the closed-form E[X(t)]" check meaningful; the sketch would
     fail it by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp2f1, logsumexp

__all__ = ["RFM", "build_rfm", "BGNBD", "fit_bgnbd", "GammaGamma", "fit_gg",
           "simulate_elog1p", "assert_no_lookahead"]


# ---------------------------------------------------------------------------- RFM


@dataclass
class RFM:
    """The four BTYD sufficient statistics, plus what is needed to handle non-buyers.

    x     repeat purchase occasions = buy_days - 1, floored at 0
    t_x   days from first buy-day to last buy-day
    T     days from first buy-day to the anchor
    m_x   mean GMV per buy-day over the observed buy-days (0 where there are none)
    """

    user_id: np.ndarray
    x: np.ndarray            # float64
    t_x: np.ndarray
    T: np.ndarray
    m_x: np.ndarray
    n_buy: np.ndarray        # int64, = x + 1 for buyers, 0 for never-buyers
    tenure: np.ndarray       # anchor - first ACTIVE day; the fallback T for never-buyers

    @property
    def buyer(self) -> np.ndarray:
        return self.n_buy > 0

    def check(self) -> None:
        """BTYD.md §8, the structural asserts."""
        b = self.buyer
        assert np.all(np.isfinite(self.x)) and np.all(self.x >= 0)
        assert np.all(self.t_x[b] >= 0) and np.all(self.t_x[b] <= self.T[b]), "t_x > T"
        assert np.all(self.T[b] >= 0)
        assert np.all(self.m_x[self.n_buy >= 1] > 0), "m_x must be > 0 wherever a buy-day exists"
        assert np.all(self.m_x[~b] == 0)
        assert np.array_equal(self.n_buy[b], (self.x[b] + 1).astype(np.int64))
        assert np.all(self.T <= self.tenure + 1e-9), "T measured from first buy cannot exceed tenure"


def build_rfm(events: pl.DataFrame, users: np.ndarray, anchor: int) -> RFM:
    """RFM at `anchor` (a day index) for exactly `users`, from days <= anchor only.

    `events` must carry columns (user_id, di, gmv) where `di` is the day index.
    """
    past = events.filter(pl.col("di") <= anchor)

    act = (past.group_by("user_id")
               .agg(pl.col("di").min().alias("first_act")))
    buy = (past.filter(pl.col("gmv") > 0)
               .group_by("user_id")
               .agg(pl.len().alias("n_buy"),
                    pl.col("di").min().alias("first_buy"),
                    pl.col("di").max().alias("last_buy"),
                    pl.col("gmv").sum().alias("gmv_sum")))

    base = pl.DataFrame({"user_id": users}).join(act, on="user_id", how="left") \
                                           .join(buy, on="user_id", how="left")
    assert base.height == users.size
    assert base["first_act"].null_count() == 0, "population contains users with no history"

    n_buy = base["n_buy"].fill_null(0).to_numpy().astype(np.int64)
    first_buy = base["first_buy"].to_numpy()
    last_buy = base["last_buy"].to_numpy()
    gmv_sum = base["gmv_sum"].fill_null(0.0).to_numpy().astype(np.float64)
    first_act = base["first_act"].to_numpy().astype(np.float64)

    has = n_buy > 0
    x = np.maximum(n_buy.astype(np.float64) - 1.0, 0.0)
    t_x = np.zeros(users.size)
    T = np.zeros(users.size)
    t_x[has] = (last_buy[has] - first_buy[has]).astype(np.float64)
    T[has] = float(anchor) - first_buy[has].astype(np.float64)
    m_x = np.zeros(users.size)
    m_x[has] = gmv_sum[has] / n_buy[has]
    tenure = float(anchor) - first_act

    # Never-buyers have no first transaction, so BTYD's clock has no origin for them
    # (BTYD.md §2/§5.3).  We give them the activity clock: observed for `tenure` days with
    # zero repeat purchases.  Their prediction is then a function of tenure alone -- not one
    # single constant, but close to it.  Reported separately in the results.
    T[~has] = tenure[~has]

    r = RFM(users, x, t_x, T, m_x, n_buy, tenure)
    r.check()
    return r


def assert_no_lookahead(events: pl.DataFrame, users: np.ndarray, anchor: int) -> None:
    """Zero the panel after `anchor`, rebuild the summary, require it unchanged.

    Same guard as `assert_causal_features` elsewhere in the repo, and it is not vacuous:
    a version of `build_rfm` that aggregated before filtering passes nothing here.
    """
    full = build_rfm(events, users, anchor)
    truncated = build_rfm(events.filter(pl.col("di") <= anchor), users, anchor)
    for nm in ("x", "t_x", "T", "m_x", "n_buy", "tenure"):
        a, b = getattr(full, nm), getattr(truncated, nm)
        assert np.array_equal(a, b), f"look-ahead in RFM field {nm}"
    # vacuity check: the truncation must actually have removed something
    assert events.filter(pl.col("di") > anchor).height > 0, "nothing was truncated; guard is vacuous"


# ---------------------------------------------------------------------------- BG/NBD


@dataclass
class BGNBD:
    r: float
    alpha: float
    a: float
    b: float
    nll: float
    n: int
    converged: bool

    def as_dict(self) -> dict:
        return {"r": self.r, "alpha": self.alpha, "a": self.a, "b": self.b,
                "nll": self.nll, "n": self.n, "converged": self.converged}

    # -- likelihood ---------------------------------------------------------
    @staticmethod
    def _loglik(params: np.ndarray, x, t_x, T) -> np.ndarray:
        r, alpha, a, b = params
        rx = r + x
        ln_A1 = gammaln(rx) - gammaln(r) + r * np.log(alpha)
        ln_A2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
        ln_A3 = -rx * np.log(alpha + T)
        d = x > 0
        # A4 exists only for x > 0; b + x - 1 > 0 is then guaranteed for b > 0
        ln_A4 = np.where(d,
                         np.log(a) - np.log(np.where(d, b + x - 1.0, 1.0)) - rx * np.log(alpha + t_x),
                         -np.inf)
        stacked = np.stack([ln_A3, ln_A4])
        return ln_A1 + ln_A2 + logsumexp(stacked, axis=0)

    # -- prediction ---------------------------------------------------------
    def p_alive(self, x, t_x, T) -> np.ndarray:
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        d = x > 0
        with np.errstate(over="ignore"):
            ratio = np.where(d,
                             (a / np.where(d, b + x - 1.0, 1.0))
                             * np.exp((r + x) * (np.log(alpha + T) - np.log(alpha + t_x))),
                             0.0)
        return 1.0 / (1.0 + ratio)

    def expected_x(self, x, t_x, T, t: float = 30.0) -> np.ndarray:
        """E[X(t) | x, t_x, T] -- expected REPEAT transactions in the next `t` days.

        Diverges for a <= 1; the caller must check `self.a > 1` (BTYD.md §3.1).
        """
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        assert np.all(np.asarray(a) > 1.0), "E[X(t)] closed form requires a > 1"
        rx = r + x
        z = t / (alpha + T + t)
        hyp = hyp2f1(rx, b + x, a + b + x - 1.0, z)
        num = ((a + b + x - 1.0) / (a - 1.0)) * (1.0 - ((alpha + T) / (alpha + T + t)) ** rx * hyp)
        return num * self.p_alive(x, t_x, T)


def fit_bgnbd(x, t_x, T, x0=(1.0, 1.0, 1.0, 1.0), verbose: bool = False) -> BGNBD:
    """MLE over log-parametrised (r, alpha, a, b), all > 0.  L-BFGS-B, several restarts."""
    x = np.asarray(x, np.float64); t_x = np.asarray(t_x, np.float64); T = np.asarray(T, np.float64)

    def nll(lp):
        p = np.exp(np.clip(lp, -20, 20))
        v = BGNBD._loglik(p, x, t_x, T)
        if not np.all(np.isfinite(v)):
            return 1e12
        return -float(v.sum())

    best = None
    starts = [np.log(np.asarray(x0, np.float64)),
              np.log(np.array([0.5, 5.0, 0.5, 2.0])),
              np.log(np.array([2.0, 20.0, 2.0, 5.0])),
              np.log(np.array([0.2, 1.0, 1.5, 10.0]))]
    for s in starts:
        res = minimize(nll, s, method="L-BFGS-B",
                       options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
        if best is None or res.fun < best.fun:
            best = res
        if verbose:
            print(f"    start {np.exp(s).round(3)} -> nll {res.fun:,.2f} {res.message}")
    r, alpha, a, b = np.exp(best.x)
    return BGNBD(float(r), float(alpha), float(a), float(b), float(best.fun), int(x.size),
                 bool(best.success))


# ---------------------------------------------------------------------------- Gamma-Gamma


@dataclass
class GammaGamma:
    p: float
    q: float
    nu: float
    nll: float
    n: int
    converged: bool

    def as_dict(self) -> dict:
        return {"p": self.p, "q": self.q, "nu": self.nu, "nll": self.nll,
                "n": self.n, "converged": self.converged}

    def population_mean(self) -> float:
        assert self.q > 1.0, f"population mean requires q > 1, got {self.q:.4f}"
        return self.p * self.nu / (self.q - 1.0)

    def expected_m(self, n, m_x) -> np.ndarray:
        """Credibility-weighted E[M | n, m_x]; `n` is the count that m_x averages."""
        n = np.asarray(n, np.float64); m_x = np.asarray(m_x, np.float64)
        w_ind = self.p * n / (self.p * n + self.q - 1.0)
        return (1.0 - w_ind) * self.population_mean() + w_ind * m_x


def fit_gg(n, m, verbose: bool = False) -> GammaGamma:
    """MLE over log-parametrised (p, q, nu) on users with n >= 2 and m > 0."""
    n = np.asarray(n, np.float64); m = np.asarray(m, np.float64)
    keep = (n >= 2) & (m > 0)
    n, m = n[keep], m[keep]
    log_m, log_n = np.log(m), np.log(n)

    def nll(lp):
        p, q, nu = np.exp(np.clip(lp, -20, 20))
        pn = p * n
        v = (gammaln(pn + q) - gammaln(pn) - gammaln(q)
             + q * np.log(nu) + (pn - 1.0) * log_m + pn * log_n
             - (pn + q) * np.log(nu + m * n))
        if not np.all(np.isfinite(v)):
            return 1e12
        return -float(v.sum())

    best = None
    for s in (np.log(np.array([1.0, 1.0, 1.0])),
              np.log(np.array([2.0, 4.0, 50.0])),
              np.log(np.array([0.5, 10.0, 200.0])),
              np.log(np.array([6.0, 4.0, 15.0]))):
        res = minimize(nll, s, method="L-BFGS-B",
                       options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
        if best is None or res.fun < best.fun:
            best = res
        if verbose:
            print(f"    start {np.exp(s).round(3)} -> nll {res.fun:,.2f} {res.message}")
    p, q, nu = np.exp(best.x)
    return GammaGamma(float(p), float(q), float(nu), float(best.fun), int(n.size),
                      bool(best.success))


# ---------------------------------------------------------------------------- simulation


def simulate_elog1p(bg: BGNBD, gg: GammaGamma, rfm: RFM, n_gg: np.ndarray,
                    horizon: float = 30.0, n_draws: int = 200, seed: int = 0,
                    max_tx: int = 200) -> dict:
    """Forward-simulate the fitted generative process; return E[log1p(y)] and E[y].

    Exact BG/NBD posterior, conditioned on the branch that matters for the future:

        alive ~ Bernoulli(P(alive | x, t_x, T))
        alive branch:  lambda ~ Gamma(r + x, rate = alpha + T)
                       p_drop ~ Beta(a, b + x)
        then Exp(lambda) interarrivals, dropout w.p. p_drop after each transaction.

    Gamma-Gamma posterior for the individual spend scale:

        nu_i ~ Gamma(q + p*n, rate = nu + n*m_x)      (prior Gamma(q, nu) when n = 0)
        sum of k future occasions ~ Gamma(k*p, rate = nu_i)

    E[p/nu_i] reproduces `GammaGamma.expected_m` exactly, so the two routes agree in
    expectation by construction and any difference in the OUTPUT is the log/linear gap.
    """
    rng = np.random.default_rng(seed)
    r, alpha, a, b = bg.r, bg.alpha, bg.a, bg.b
    x, t_x, T, m_x = rfm.x, rfm.t_x, rfm.T, rfm.m_x
    n_gg = np.asarray(n_gg, np.float64)
    N = x.size

    pa = bg.p_alive(x, t_x, T)
    shape_lam = r + x
    rate_lam = alpha + T
    # every parameter may be a scalar (e0170) or a per-user array (BAYES_EXP B1)
    beta_a = np.broadcast_to(np.asarray(a, np.float64), (N,))
    beta_b = np.broadcast_to(np.asarray(b, np.float64) + x, (N,))
    shape_lam = np.broadcast_to(np.asarray(shape_lam, np.float64), (N,))
    rate_lam = np.broadcast_to(np.asarray(rate_lam, np.float64), (N,))
    # GG posterior; users with no buy-day fall back to the prior (n = 0, m_x = 0)
    gg_shape = gg.q + gg.p * n_gg
    gg_rate = gg.nu + n_gg * m_x

    sum_log1p = np.zeros(N)
    sum_log1p_sq = np.zeros(N)
    sum_y = np.zeros(N)
    sum_n = np.zeros(N)
    n_zero = np.zeros(N)

    for s in range(n_draws):
        alive = rng.random(N) < pa
        lam = rng.gamma(shape_lam, 1.0 / rate_lam)
        p_drop = rng.beta(beta_a, beta_b)

        # forward BG/NBD process, vectorised over users, iterated over transaction index
        n_tx = np.zeros(N, np.int64)
        clock = np.zeros(N)
        active = alive.copy()
        for _ in range(max_tx):
            if not active.any():
                break
            idx = np.flatnonzero(active)
            clock[idx] += rng.exponential(1.0 / lam[idx])
            hit = clock[idx] <= horizon
            n_tx[idx[hit]] += 1
            survive = rng.random(hit.sum()) >= p_drop[idx[hit]]
            still = np.zeros(idx.size, bool)
            still[hit] = survive
            active[idx] = still
        else:                                              # pragma: no cover
            raise RuntimeError("max_tx reached; the fitted rate is implausibly high")

        nu_i = rng.gamma(gg_shape, 1.0 / gg_rate)
        y = np.zeros(N)
        buy = n_tx > 0
        y[buy] = rng.gamma(n_tx[buy] * gg.p, 1.0 / nu_i[buy])

        l1 = np.log1p(y)
        sum_log1p += l1
        sum_log1p_sq += l1 * l1
        sum_y += y
        sum_n += n_tx
        n_zero += ~buy

    mean_l = sum_log1p / n_draws
    # posterior predictive sd of log1p(Y): BAYES_EXP.md's `s_u`, the one column §10 claims a
    # GBDT cannot manufacture ("how much does the history actually pin this user down").
    var_l = np.maximum(sum_log1p_sq / n_draws - mean_l ** 2, 0.0)
    return {"e_log1p": mean_l,
            "sd_log1p": np.sqrt(var_l),
            "p_zero": n_zero / n_draws,
            "e_y": sum_y / n_draws,
            "e_n": sum_n / n_draws,
            "p_alive": pa}


# ---------------------------------------------------------------------------- self-test


def _self_test() -> None:
    """BTYD.md §8: the closed forms must agree with a simulation of the same process."""
    rng = np.random.default_rng(0)
    print("  MC vs closed-form E[X(30)] (BTYD.md §8):")
    for (r, alpha, a, b) in [(0.24, 4.4, 0.79, 2.43), (1.0, 10.0, 1.5, 3.0), (0.5, 2.0, 2.0, 8.0)]:
        if a <= 1.0:
            print(f"    r={r} alpha={alpha} a={a} b={b}: skipped, closed form needs a > 1")
            continue
        bg = BGNBD(r, alpha, a, b, 0.0, 0, True)
        n = 20000
        x = rng.integers(0, 12, n).astype(np.float64)
        T = rng.uniform(30, 400, n)
        t_x = np.where(x > 0, rng.uniform(0, 1, n) * T, 0.0)
        gg = GammaGamma(1.0, 3.0, 2.0, 0.0, 0, True)          # spend is irrelevant to E[X]
        out = simulate_elog1p(bg, gg, RFM(np.arange(n), x, t_x, T, np.ones(n), (x + 1).astype(np.int64),
                                          T + 5),
                              n_gg=x + 1, n_draws=300, seed=1)
        closed = bg.expected_x(x, t_x, T, 30.0)
        rel = abs(out["e_n"].mean() - closed.mean()) / closed.mean()
        print(f"    r={r} alpha={alpha} a={a} b={b}: MC {out['e_n'].mean():.5f} "
              f"vs closed {closed.mean():.5f}  rel err {rel:.4%}")
        assert rel < 0.02, "simulated process does not reproduce the closed-form E[X(t)]"

    # Gamma-Gamma: the simulated spend must reproduce the credibility formula
    gg = GammaGamma(2.5, 3.5, 40.0, 0.0, 0, True)
    n_obs = np.array([1.0, 3.0, 10.0])
    m_obs = np.array([20.0, 60.0, 150.0])
    draws = rng.gamma(gg.q + gg.p * n_obs[:, None], 1.0 / (gg.nu + n_obs[:, None] * m_obs[:, None]),
                      size=(3, 400000))
    mc = (gg.p / draws).mean(1)
    cf = gg.expected_m(n_obs, m_obs)
    print(f"  MC vs closed-form E[M]: {np.round(mc, 3)} vs {np.round(cf, 3)}")
    assert np.allclose(mc, cf, rtol=0.02), "GG posterior draw does not match expected_m"

    # parameter recovery: simulate a BG/NBD panel, refit, require the fit to find it back
    true = (0.6, 8.0, 1.4, 3.0)
    n = 60000
    lam = rng.gamma(true[0], 1.0 / true[1], n)
    p = rng.beta(true[2], true[3], n)
    T = rng.uniform(60, 400, n)
    xs, txs = np.zeros(n), np.zeros(n)
    clock = np.zeros(n); alive = np.ones(n, bool)
    for _ in range(400):
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        clock[idx] += rng.exponential(1.0 / lam[idx])
        hit = clock[idx] <= T[idx]
        xs[idx[hit]] += 1
        txs[idx[hit]] = clock[idx[hit]]
        still = np.zeros(idx.size, bool)
        still[hit] = rng.random(hit.sum()) >= p[idx[hit]]
        alive[idx] = still
    fit = fit_bgnbd(xs, txs, T)
    print(f"  BG/NBD recovery: true {true} -> fitted "
          f"({fit.r:.3f}, {fit.alpha:.3f}, {fit.a:.3f}, {fit.b:.3f})")
    err = abs(fit.expected_x(xs, txs, T, 30).mean()
              - BGNBD(*true, 0.0, 0, True).expected_x(xs, txs, T, 30).mean())
    print(f"  mean E[X(30)] error vs truth: {err:.5f}")
    assert err < 0.02, "BG/NBD parameter recovery failed"

    print("src/btyd.py: all self-tests passed")


if __name__ == "__main__":
    _self_test()
