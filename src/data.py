#!/usr/bin/env python
"""
Loading and the dense (user x day) panel used by every downstream module.

The raw file is sparse -- 30.6 M of a possible 102.25 M user-days (DATA.md §3). We never
materialise a padded dataframe; instead we build one dense matrix per value column
(250 000 x 409 float32, 409 MB each) and immediately reduce it to a prefix-sum along the
time axis, which makes any window aggregate an O(1) subtraction.

Peak RSS ~7 GB. Runtime ~25 s on `compute`.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train.parquet"

# value columns we carry into the dense panel (source name -> short name)
VALUE_COLS = {
    "gmv": "gmv",
    "to_ord": "ord",
    "to_cart": "cart",
    "searches": "srch",
    "gmv_search": "gmvs",
    "gmv_cat": "gmvc",
    # --- funnel columns, added 2026-08-13 -------------------------------------------------
    # Ten of the sixteen raw numeric columns had never entered the pipeline. Measured against
    # log1p(target) at anchor 2025-10-16, `search_to_ord` scores Spearman 0.5668 -- level with
    # `to_ord` (0.5731), our single strongest raw signal -- and the ten together add +0.00489
    # incremental R^2 over the four we were using.
    #
    # They are not a recombination of what we had. `gmv_search`/`gmv_cat` give GMV BY CHANNEL;
    # these give ORDER COUNTS by channel plus per-day conversion flags, and order counts beat
    # GMV as a predictor here. The search-vs-catalogue attribution mix is a genuinely separate
    # measurement of how a user shops.
    "search": "sev",             # search events
    "cat": "cev",                # catalogue events
    "search_to_cart": "s2c",
    "search_to_ord": "s2o",
    "cat_to_cart": "c2c",
    "cat_to_ord": "c2o",
    # the has_* columns are per-day binaries, so a window sum is "days on which it happened" --
    # a different quantity from the event count above, not a duplicate of it
    "has_search_to_cart": "hs2c",
    "has_search_to_ord": "hs2o",
    "has_cat_to_cart": "hc2c",
    "has_cat_to_ord": "hc2o",
}


class Panel:
    """Dense prefix-summed panel. All window queries go through `wsum` / `wdays`."""

    def __init__(self, path: Path = TRAIN, verbose: bool = True):
        df = pl.read_parquet(path)
        self.dmin: date = df["event_date"].min()
        self.dmax: date = df["event_date"].max()
        self.n_days = (self.dmax - self.dmin).days + 1
        self.users = np.sort(df["user_id"].unique().to_numpy())
        self.n_users = self.users.size

        ui = np.searchsorted(self.users, df["user_id"].to_numpy())
        di = (df["event_date"].to_numpy().astype("datetime64[D]")
              - np.datetime64(self.dmin)).astype(np.int32)

        # presence + its prefix sum
        P = np.zeros((self.n_users, self.n_days), dtype=bool)
        P[ui, di] = True
        self.cs_days = np.concatenate(
            [np.zeros((self.n_users, 1), np.int32), np.cumsum(P, axis=1, dtype=np.int32)], axis=1)

        # last-activity / last-order day index, as of every day (running max)
        self.last_act = np.empty((self.n_users, self.n_days), np.int16)
        cur = np.full(self.n_users, -1, np.int32)
        for t in range(self.n_days):
            cur = np.where(P[:, t], t, cur)
            self.last_act[:, t] = cur
        self.first_act = np.argmax(P, axis=1).astype(np.int16)   # every user has >=3 rows

        # value columns -> prefix sums (float64: gmv prefix reaches ~1e5, fp32 would round)
        # buy-days and "empty visit" days (DATA.md §3: 14.85% of rows are zero-activity visits)
        gmv_raw = np.zeros((self.n_users, self.n_days), np.float32)
        np.add.at(gmv_raw, (ui, di), df["gmv"].to_numpy().astype(np.float32))
        B = gmv_raw > 0
        self.cs_buy = np.concatenate(
            [np.zeros((self.n_users, 1), np.int32), np.cumsum(B, axis=1, dtype=np.int32)], axis=1)
        empty = ((df["searches"] == 0) & (df["to_cart"] == 0) & (df["to_ord"] == 0)
                 & (df["search"] == 0) & (df["cat"] == 0)).to_numpy()
        E = np.zeros((self.n_users, self.n_days), dtype=bool)
        E[ui[empty], di[empty]] = True
        self.cs_empty = np.concatenate(
            [np.zeros((self.n_users, 1), np.int32), np.cumsum(E, axis=1, dtype=np.int32)], axis=1)
        del E

        # time-weighted prefix sums -> centre of mass of activity/GMV inside any window
        tvec = np.arange(self.n_days, dtype=np.float32)
        self.cs_tw_days = np.concatenate(
            [np.zeros((self.n_users, 1)), np.cumsum(P * tvec, axis=1, dtype=np.float64)], axis=1)
        self.cs_tw_gmv = np.concatenate(
            [np.zeros((self.n_users, 1)), np.cumsum(gmv_raw * tvec, axis=1, dtype=np.float64)], axis=1)

        # squared-value and time-weighted prefix sums -> exact windowed VARIANCE statistics.
        # Var(x) = E[x^2] - E[x]^2, so a prefix sum of x^2 makes dispersion O(1) per window,
        # the same trick the level features use for the mean.
        ord_tmp = np.zeros((self.n_users, self.n_days), np.float32)
        np.add.at(ord_tmp, (ui, di), df["to_ord"].to_numpy().astype(np.float32))
        t2vec = (np.arange(self.n_days, dtype=np.float64) ** 2)
        self.cs_sq = {}
        for nm, raw in (("gmv", gmv_raw), ("ord", ord_tmp)):
            self.cs_sq[nm] = np.concatenate(
                [np.zeros((self.n_users, 1)),
                 np.cumsum(raw.astype(np.float64) ** 2, axis=1, dtype=np.float64)], axis=1)
        self.cs_tw2_days = np.concatenate(
            [np.zeros((self.n_users, 1)),
             np.cumsum(P * t2vec, axis=1, dtype=np.float64)], axis=1)
        Bf = B.astype(np.float64)
        self.cs_tw_buy = np.concatenate(
            [np.zeros((self.n_users, 1)),
             np.cumsum(Bf * np.arange(self.n_days, dtype=np.float64), axis=1, dtype=np.float64)], axis=1)
        self.cs_tw2_buy = np.concatenate(
            [np.zeros((self.n_users, 1)),
             np.cumsum(Bf * t2vec, axis=1, dtype=np.float64)], axis=1)
        del ord_tmp, Bf

        # causal EWM: E[t] = x[t] + decay * E[t-1]; column A uses only days <= A by construction
        ord_raw = np.zeros((self.n_users, self.n_days), np.float32)
        np.add.at(ord_raw, (ui, di), df["to_ord"].to_numpy().astype(np.float32))
        self.ewm: dict[str, np.ndarray] = {}
        for nm, raw in (("gmv", gmv_raw), ("ord", ord_raw)):
            for hl in (7, 30, 90):
                dec = float(0.5 ** (1.0 / hl))
                acc = np.zeros((self.n_users, self.n_days), np.float32)
                run = np.zeros(self.n_users, np.float32)
                for t in range(self.n_days):
                    run = raw[:, t] + dec * run
                    acc[:, t] = run
                self.ewm[f"{nm}_hl{hl}"] = acc
        # keep the raw daily matrices: the time-series-shape features (skew, autocorr,
        # spectral, entropy, run-lengths) cannot be expressed as prefix sums.
        self.raw = {"gmv": gmv_raw.copy(), "ord": ord_raw.copy()}
        del gmv_raw, ord_raw, B

        self.cs: dict[str, np.ndarray] = {}
        for src, short in VALUE_COLS.items():
            A = np.zeros((self.n_users, self.n_days), np.float32)
            np.add.at(A, (ui, di), df[src].to_numpy().astype(np.float32))
            self.cs[short] = np.concatenate(
                [np.zeros((self.n_users, 1)), np.cumsum(A, axis=1, dtype=np.float64)], axis=1)
            if short == "ord":
                O = A > 0
                self.last_ord = np.empty((self.n_users, self.n_days), np.int16)
                cur = np.full(self.n_users, -1, np.int32)
                for t in range(self.n_days):
                    cur = np.where(O[:, t], t, cur)
                    self.last_ord[:, t] = cur
                del O
            del A
        del df, P, ui, di

        if verbose:
            print(f"  Panel: {self.n_users:,} users x {self.n_days} days "
                  f"({self.dmin} .. {self.dmax}), {len(self.cs)} value columns", flush=True)

    # ---------------------------------------------------------------- truncation
    # `floor` implements "features gathered starting from N days ago" EXACTLY: every window
    # is clamped to begin at `floor`, recency/tenure are capped at the visible span, and the
    # EWMs are re-based in closed form. This is the honest version of the name-based
    # `feature_max_window` filter, which only guessed each feature's reach from its name.
    floor: int = 0

    def _a(self, a: int) -> int:
        return max(a, self.floor, 0)

    def set_floor(self, anchor: int, days: int | None) -> None:
        self.floor = 0 if not days else max(anchor - int(days) + 1, 0)

    def span(self, anchor: int) -> float:
        return float(anchor - self.floor + 1)

    def ewm_at(self, key: str, anchor: int) -> np.ndarray:
        """EWM at `anchor` counting only days >= floor.

        EWM(t) = sum_{s<=t} x[s] * d^(t-s), so the contribution of everything before `floor`
        is exactly d^(t-floor+1) * EWM(floor-1) and can be subtracted off."""
        v = self.ewm[key][:, anchor]
        j = self.floor - 1
        if j >= 0:
            dec = 0.5 ** (1.0 / int(key.split("hl")[1]))
            v = v - (dec ** (anchor - j)) * self.ewm[key][:, j]
        return np.maximum(v, 0.0)

    # ---------------------------------------------------------------- helpers
    def prefix_arrays(self) -> list[np.ndarray]:
        """Every (n_users, n_days+1) prefix-sum array -- used by the look-ahead guard."""
        return list(self.cs.values()) + list(self.cs_sq.values()) + [
            self.cs_days, self.cs_buy, self.cs_empty, self.cs_tw_days, self.cs_tw_gmv,
            self.cs_tw2_days, self.cs_tw_buy, self.cs_tw2_buy]

    def ewm_arrays(self) -> list[np.ndarray]:
        return list(self.ewm.values())

    def raw_arrays(self) -> list[np.ndarray]:
        """Raw daily matrices. block_sbc / block_tsfeat read these DIRECTLY rather than via
        prefix sums, so the look-ahead guard must freeze them too or it proves nothing about
        those blocks."""
        return list(self.raw.values())

    def wbuy(self, a: int, b: int) -> np.ndarray:
        a = self._a(a); b = min(b, self.n_days - 1)
        if b < a:
            return np.zeros(self.n_users)
        return (self.cs_buy[:, b + 1] - self.cs_buy[:, a]).astype(np.float64)

    def wempty(self, a: int, b: int) -> np.ndarray:
        a = self._a(a); b = min(b, self.n_days - 1)
        if b < a:
            return np.zeros(self.n_users)
        return (self.cs_empty[:, b + 1] - self.cs_empty[:, a]).astype(np.float64)

    def wsumsq(self, col: str, a: int, b: int) -> np.ndarray:
        a = self._a(a); b = min(b, self.n_days - 1)
        if b < a:
            return np.zeros(self.n_users)
        return self.cs_sq[col][:, b + 1] - self.cs_sq[col][:, a]

    def wdate_std(self, which: str, a: int, b: int) -> np.ndarray:
        """sd of the day indices on which activity ('days') or purchases ('buy') fell."""
        a = self._a(a); b = min(b, self.n_days - 1)
        if which == "days":
            n = self.wdays(a, b); s1 = self.cs_tw_days[:, b + 1] - self.cs_tw_days[:, a]
            s2 = self.cs_tw2_days[:, b + 1] - self.cs_tw2_days[:, a]
        else:
            n = self.wbuy(a, b); s1 = self.cs_tw_buy[:, b + 1] - self.cs_tw_buy[:, a]
            s2 = self.cs_tw2_buy[:, b + 1] - self.cs_tw2_buy[:, a]
        nn = np.maximum(n, 1.0)
        var = np.maximum(s2 / nn - (s1 / nn) ** 2, 0.0)
        return np.where(n >= 2, np.sqrt(var), -1.0)

    def wcom(self, which: str, a: int, b: int) -> np.ndarray:
        """Centre of mass (mean day index) of activity/GMV in [a,b], relative to b.
        0 = all mass on the last day, larger = older. Returns -1 when the window is empty."""
        a = self._a(a); b = min(b, self.n_days - 1)
        cs_tw = self.cs_tw_days if which == "days" else self.cs_tw_gmv
        num = cs_tw[:, b + 1] - cs_tw[:, a]
        den = self.wdays(a, b) if which == "days" else self.wsum("gmv", a, b)
        return np.where(den > 0, b - num / np.maximum(den, 1e-9), -1.0)

    def idx(self, d: date) -> int:
        return (d - self.dmin).days

    def day(self, i: int) -> date:
        return self.dmin + timedelta(days=int(i))

    def wsum(self, col: str, a: int, b: int) -> np.ndarray:
        """sum of `col` over day-index window [a, b] inclusive, clipped to the history"""
        a = self._a(a); b = min(b, self.n_days - 1)
        if b < a:
            return np.zeros(self.n_users)
        return self.cs[col][:, b + 1] - self.cs[col][:, a]

    def wdays(self, a: int, b: int) -> np.ndarray:
        """number of present (active) days in [a, b]"""
        a = self._a(a); b = min(b, self.n_days - 1)
        if b < a:
            return np.zeros(self.n_users)
        return (self.cs_days[:, b + 1] - self.cs_days[:, a]).astype(np.float64)

    def active_in(self, a: int, b: int) -> np.ndarray:
        """boolean mask: >=1 active day in [a, b]"""
        return self.wdays(a, b) > 0

    def target(self, anchor: int, horizon: int = 30) -> np.ndarray:
        """sum of gmv over [anchor+1, anchor+horizon] -- the competition target"""
        return self.wsum("gmv", anchor + 1, anchor + horizon)

    def recency(self, anchor: int) -> np.ndarray:
        """days since last activity as of `anchor` (n_days if never active)"""
        la = self.last_act[:, anchor].astype(np.float64)
        r = np.where(la < 0, self.n_days, anchor - la)
        return np.minimum(r, self.span(anchor)) if self.floor else r

    def recency_order(self, anchor: int) -> np.ndarray:
        lo = self.last_ord[:, anchor].astype(np.float64)
        r = np.where(lo < 0, self.n_days, anchor - lo)
        return np.minimum(r, self.span(anchor)) if self.floor else r

    def tenure(self, anchor: int) -> np.ndarray:
        t = anchor - self.first_act.astype(np.float64)
        return np.minimum(t, self.span(anchor)) if self.floor else t
