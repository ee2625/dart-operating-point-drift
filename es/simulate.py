"""Parametric null and power simulation (spec v3 §6.2(b); review round 2, N1 and N2).
Each node's hourly score and P&L = seasonal mean (local hour x weekday class x month) + a daily AR(1)
component + hourly noise. rho couples the daily components (rho = 0: no timing skill;
rho > 0: high-score days are loss days). Quarters one year apart are independent draws.

Conventions, for the methodology review:
- ``NodeParams.sd_s`` and ``sd_p`` are the SDs of the AR(1) *innovations*; the stationary SD of a daily component is
  sd / sqrt(1 - phi^2). ``sd_eps`` and ``sd_eta`` are the hourly noise SDs.
- Seasonal means (Ruling 64): one per (hour, weekday class, month) cell, the class being 1 on a local Saturday or
  Sunday as in ``es.data`` (``NodeParams.mu_w`` and ``nu_w``). A cell seen on fewer than 5 days of the fit takes the
  (hour, month) mean instead (``NodeParams.mu`` and ``nu``), and ``NodeParams.fallback_cells`` records which cells did.
  The daily component and the noise are fitted to the residuals from these means.
- Daily AR(1) (Ruling 84a): those residuals are not the daily component. The cell means take out each (weekday class,
  month) group's mean of it and the hourly noise adds sd^2 / hours to the daily means, so the plain AR(1) fit reads a
  true 0.8 as 0.68 and a P&L without any daily component as one of SD 0.4. ``fit_node`` corrects phi and the SDs by
  indirect inference (``_PersistenceMap``): it simulates the daily AR(1) on the fit's own calendar, de-means and adds
  the noise exactly as the fit does, and finds by bisection the phi whose plain statistic is the observed one; the
  stationary SD then reproduces the observed variance. The plain estimates stay in ``NodeParams.*_raw``, and
  ``phi_status`` records a clipped, absent or skipped correction.
- rho: the P&L's daily component is ``-rho * z_score + sqrt(1 - rho^2) * z_own``, where z_score is the score's daily
  component standardised to the sample SD of the P&L's own component z_own. At rho = 0 the P&L does not depend on the
  score model at all, only on its own seasonal mean, daily component and noise.
- Strata of the H-E cube (Ruling 68): ``strata="primary"`` (the default) is the registered stratification, node x local
  hour x weekday class x month, so every stratum is one (hour, weekday class, month) cell. ``strata="decile"`` is the
  sensitivity: node x calendar-propensity deciles of the score's seasonal mean over the window's rows, taken by value
  like ``es.measures.add_calendar_deciles`` (the calendar-only model's score is the seasonal mean here); a decile can
  mix weekdays and weekends, hours and months, which primary strata never do.
- The H-E reference set is ``rotation_test``'s own default (the full cyclic group, Ruling 69); no shift floor is set
  here.
- Persistence (Ruling 63): ``persistence="fit"`` simulates the fitted phi; ``persistence="upper"`` simulates the upper
  end of its interval, ``phi_u = min(0.99, phi + 1.96 * sqrt((1 - phi^2) / n_fit_days))``, for the score and the P&L
  alike, with the innovation SD rescaled so that the stationary SD is unchanged. ``n_fit_days`` is the number of day
  pairs the AR(1) fit used. The multiplicative ``phi_scale`` of the brief is deprecated (it left the innovation SD fixed
  and so inflated the stationary SD as well).
- H-B: the alternative and base blocks are drawn independently from the same parameters (delta = 0 is equal
  portability; delta > 0 shifts the alternative block's deployment scores up by delta score-SDs). ``common_shift``
  shifts both blocks' deployment scores by that many score-SDs, equal portability under drift. A score-SD is one number
  per node and window, the fitted stationary SD of the score (``_score_sd``), shared by both blocks. ``calibrate_delta``
  finds the delta whose simulated gap, the median over nodes of the paired |log10 R| difference (the H-B statistic), is
  a target.
- H-B bootstrap (Ruling 92a): ``hb_size`` runs the registered ``block_contrast_bootstrap`` on each simulated pair of
  blocks, or by default (``fast=True``) ``_hb_bootstrap_fast``, which computes the same numbers, to the last bit, from the
  drawn arrays without building frames or a resampled sample row by row; ``test_simulate`` holds the equality to 32 random
  grids and the fast path falls back to the registered function for any grid it is not built for.
- P&L noise: by default one SD per node (``sd_eta``). With ``hetero=True`` each (local hour, weekday class) cell draws
  its own fitted SD (``NodeParams.sd_eta_cell``), so that the calendar strata differ in loss volatility. The cells
  average to the same noise variance as the pooled SD; only the P&L noise is heteroskedastic, not the score's.
- ``he_size_power`` reports ``mean_selc``, the pooled SELC of the simulated alerts, computed from the cubes. It does
  not read it off ``rotation_test``, whose statistic is the studentized excess (Ruling 55g), not a SELC.
- Simplifications of the spec's model, all of them also in the brief: Gaussian shocks, white hourly noise, no
  season-specific noise variance, no DST structure."""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from .bootstrap import (RowCube, basic_ci, basic_p_one_sided, block_contrast_bootstrap, percentile_ci, rotation_test,
                        stationary_counts)
from .measures import _deciles_by_value

#: A (hour, weekday class, month) cell needs this many days of the fit for its own seasonal mean (Ruling 64).
_MIN_MEAN_DAYS = 5

#: The upper end of the persistence interval is phi + Z * sqrt((1 - phi^2) / n), capped (Ruling 63).
_PERSISTENCE_Z = 1.96
_PHI_CAP = 0.99

_PERSISTENCE_MODES = ("fit", "upper")
_STRATA_MODES = ("primary", "decile")

#: The correction of the fitted daily persistence (Ruling 84a, I-1): the persistence is searched for in this range,
#: with about ``_CORRECTION_PATHS`` simulated days per evaluation of the map, ``_CORRECTION_STEPS`` bisection steps, a
#: fixed seed, and no correction below ``_CORRECTION_MIN_PAIRS`` day pairs. The top of the range is where a year of data
#: stops being informative: after the cell means have taken out each group's mean, the lag-one statistic of a true 0.9,
#: 0.95 and 0.98 is 0.77, 0.80 and 0.81 while the share of the variance that survives falls from 0.55 to 0.17, so a
#: higher bound would only turn an observed statistic above the map into a huge stationary SD.
_PHI_SEARCH = (-0.5, 0.95)
_CORRECTION_PATHS = 40_000
_CORRECTION_STEPS = 12
_CORRECTION_SEED = 20261022
_CORRECTION_MIN_PAIRS = 30

#: The fast path of the H-B bootstrap (Ruling 92a) keeps a table of (24 days + 1) x days counts per node and block, so it
#: is built for windows of up to ``_FAST_MAX_DAYS`` days (the registered ones are 92); longer windows use the registered
#: function. A resampled calibration window's upper-tail quantile is read from its largest rows alone: the
#: ``_FAST_TOP_FACTOR`` times as many rows as there are elements above the quantile, and at least ``_FAST_TOP_MIN``.
#: A replicate whose draw holds too few elements of these rows is read from the whole window instead, so the answer does
#: not depend on these two numbers, only the run time does.
_FAST_MAX_DAYS = 400
_FAST_TOP_FACTOR = 4
_FAST_TOP_MIN = 96


@dataclass
class NodeParams:
    """One node's fitted model.

    ``mu`` and ``nu`` map (hour, month) to the seasonal mean of the score and of the P&L; they are the level of a
    weekday-class cell that has too few days. ``mu_w`` and ``nu_w`` map (hour, weekday class, month) to the mean of that
    cell, for the cells with at least 5 days, and ``fallback_cells`` lists the (hour, weekday class, month) cells of the
    fit that had fewer and so use ``mu`` / ``nu``. ``phi_*`` and ``sd_*`` are the AR(1) persistence and innovation SD of
    the daily score (s) and P&L (p) component, ``n_fit_days`` the number of calendar-adjacent day pairs they were
    fitted on (0 when unknown, which ``persistence="upper"`` refuses). ``sd_eps`` and ``sd_eta`` are the hourly noise
    SDs of the score and the P&L. ``sd_eta_cell`` maps (hour, weekday class) to the P&L noise SD of that cell; a cell it
    lacks (fewer than 30 residuals) takes ``sd_eta``.

    ``fit_node`` corrects ``phi_*`` and ``sd_*`` for what the fit itself does to a daily component (the cell means take
    out each (weekday class, month) group's mean of it, and the hourly noise leaks into the daily means); ``*_raw``
    keep the plain AR(1) estimates for reporting and ``phi_status`` says, per component ("score", "pnl"), whether the
    correction ran: "corrected", "clipped_low" or "clipped_high" (the plain estimate lay outside what the map can
    reach, so the end of the search range is used), "no_signal" (the daily means hold no variance beyond the hourly
    noise) or "uncorrected" (too few day pairs)."""
    mu: dict
    nu: dict
    phi_s: float
    sd_s: float
    sd_eps: float
    phi_p: float
    sd_p: float
    sd_eta: float
    sd_eta_cell: dict = field(default_factory=dict)
    n_fit_days: int = 0
    mu_w: dict = field(default_factory=dict)
    nu_w: dict = field(default_factory=dict)
    fallback_cells: tuple = ()
    phi_s_raw: float | None = None
    sd_s_raw: float | None = None
    phi_p_raw: float | None = None
    sd_p_raw: float | None = None
    phi_status: dict = field(default_factory=dict)


def _adjacent(index):
    """Which consecutive entries of a daily series' date index are calendar-adjacent days."""
    day = pd.DatetimeIndex(pd.to_datetime(index))
    return np.asarray((day[1:] - day[:-1]) == pd.Timedelta(days=1))


def _lag1(X, adjacent):
    """Lag-one correlation over calendar-adjacent pairs, for every row of ``X`` (one daily series per row): the
    statistic ``_ar1`` reports before it clips it."""
    prev, nxt = X[:, :-1][:, adjacent], X[:, 1:][:, adjacent]
    pc, nc = prev - prev.mean(axis=1, keepdims=True), nxt - nxt.mean(axis=1, keepdims=True)
    den = np.sqrt((pc ** 2).sum(axis=1) * (nc ** 2).sum(axis=1))
    return np.where(den > 0, (pc * nc).sum(axis=1) / np.where(den > 0, den, 1.0), 0.0)


def _ar1(series):
    """(phi, innovation SD, number of pairs) of a daily series indexed by calendar date. phi is the lag-one correlation
    over pairs of calendar-adjacent days only, so a missing day breaks the chain instead of being read as a one-day lag;
    the SD is that of x_t - phi * x_{t-1} over the same pairs. With no gap this is the usual estimator. This is the
    plain estimate: ``fit_node`` corrects it (``_PersistenceMap``)."""
    adjacent = _adjacent(series.index)
    x = series.to_numpy(dtype=float)
    x = x - x.mean()
    prev, nxt = x[:-1][adjacent], x[1:][adjacent]
    if len(prev) < 2 or np.var(x) == 0 or np.var(prev) == 0 or np.var(nxt) == 0:
        return 0.0, float(np.std(x)), int(len(prev))
    phi = float(np.clip(np.corrcoef(prev, nxt)[0, 1], -_PHI_CAP, _PHI_CAP))
    return phi, float(np.std(nxt - phi * prev)), int(len(prev))


_FIT_COLS = ("settle_date", "settle_hour", "score", "pnl_open")


def _check_fit_rows(rows):
    """Like ``es.bootstrap`` and ``es.measures``: an empty frame, a missing column, a NaN or a second node is an error,
    not a fit of garbage (a NaN would otherwise run on into every simulated score)."""
    if len(rows) == 0:
        raise ValueError("fit_node: the frame is empty, so there is nothing to fit")
    missing = [c for c in _FIT_COLS if c not in rows.columns]
    if missing:
        raise ValueError(f"fit_node: missing column(s) {missing}")
    nan = [c for c in _FIT_COLS if rows[c].isna().any()]
    if nan:
        raise ValueError(f"fit_node: NaN in {nan}; nothing is dropped silently")
    if {"market", "node"} <= set(rows.columns) and len(rows[["market", "node"]].drop_duplicates()) > 1:
        raise ValueError("fit_node: the frame holds more than one node; fit one node at a time")


def _noise_sd(dev, df):
    """SD of the hourly deviations from their day's mean ``dev``, over the residual degrees of freedom ``df``."""
    return float(np.sqrt(np.sum(dev ** 2) / df)) if df > 0 else 0.0


#: A (hour, weekday class) cell needs this many residuals to get its own noise SD; with fewer the pooled SD stands in.
_MIN_CELL_OBS = 30


def _wclass(dates):
    """The pipeline's weekday class (``es.data.build_panel``): 1 on a local Saturday or Sunday, else 0."""
    return (pd.to_datetime(pd.Series(dates)).dt.dayofweek.to_numpy() >= 5).astype(int)


def _components(left, right, n_nodes):
    """Number of connected components of the bipartite graph whose edges are the (left, right) index pairs."""
    parent = list(range(n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in set(zip(left.tolist(), right.tolist())):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in range(n_nodes)})


def _residual_df(day_code, cell_code):
    """Residual degrees of freedom of "cell mean + day mean" on rows that sit in a day and a mean cell: the rows less
    the rank of the day and cell indicators, which is days + cells - the connected components of the graph whose edges
    are the rows. (A two-way table of 12 months has 12 components; weekday cells make it 24.)"""
    n_days, n_cells = int(day_code.max()) + 1, int(cell_code.max()) + 1
    return len(day_code) - n_days - n_cells + _components(day_code, cell_code + n_days, n_days + n_cells)


def _cell_noise_sd(frame, dev, kappa, df):
    """P&L noise SD by (hour, weekday class), from ``dev``, each residual less its day's mean. Cells with fewer than
    ``_MIN_CELL_OBS`` residuals are left out.

    Taking out the day's mean removes the daily component but mixes into every hour a 1/H share of the day's noise
    (H hours a day), and a seasonal mean estimated from n days absorbs a 1/n share of what it was taken from. A row
    whose mean came from n days has therefore a mean square of kappa * [(1 - 2/H) var_h + mean_h(var) / H], with
    kappa = 1 - 1/n (``kappa`` holds it per row, a cell averaging its rows'). Inverted here, per weekday class, that
    gives the cell variances (exact for complete days). Without it a quiet hour reads loud: SD 0.5 amid hours of SD 3
    would read 0.66."""
    n_hours = frame["settle_hour"].nunique()
    if df <= 0 or n_hours < 3:
        return {}
    cells = (pd.DataFrame({"h": frame["settle_hour"].to_numpy(), "w": frame["wclass"].to_numpy(),
                           "e2": dev.to_numpy() ** 2, "k": kappa}).groupby(["h", "w"])
             .agg(ms=("e2", "mean"), k=("k", "mean"), n=("e2", "size")))
    cells = cells[(cells["n"] >= _MIN_CELL_OBS) & (cells["k"] > 0.5)]
    out = {}
    for _, g in cells.groupby(level="w"):
        shape = g["ms"] / g["k"]                                     # (1 - 2/H) var_h + mean_var / H
        mean_var = shape.mean() / (1 - 1 / n_hours)                  # the class's mean noise variance
        var = (shape - mean_var / n_hours) / (1 - 2 / n_hours)
        out.update({(int(h), int(w)): float(np.sqrt(max(v, 0.0))) for (h, w), v in var.items()})
    return out


class _PersistenceMap:
    """What the fit does to a daily AR(1) component, and the inverse of it (Ruling 84a, I-1).

    The cell means a fit subtracts take out, from the daily component, the mean of every (weekday class, month) group of
    days, and the hourly noise adds sd^2 / hours to the daily means. The plain AR(1) fit of the resulting daily series
    therefore reads a persistent component as less persistent and less variable than it is (a true 0.8 as 0.68), and
    reads iid hourly noise as a daily component. This class knows the calendar of one fit: the day of every row, and the
    cell mean each row was de-meaned by. For a candidate phi it simulates a daily AR(1) on that calendar with the
    stationary variance that reproduces the variance of the observed daily series, applies the de-meaning and the noise
    share exactly as the fit does, and returns the lag-one correlation the plain fit would report; ``solve`` finds by
    bisection the phi whose simulated statistic is the observed one, and derives the stationary and innovation SD
    from the corrected variance decomposition. The random numbers are fixed, so the map is a deterministic and smooth
    function of phi, and rows are put in a canonical order first, so the result does not depend on the order of the
    input rows."""

    def __init__(self, day_code, hour, own, c3, cp, sigma, adjacent, seed=_CORRECTION_SEED):
        order = np.lexsort((hour, day_code))
        day_code, own, c3, cp, sigma = (np.asarray(a)[order] for a in (day_code, own, c3, cp, sigma))
        n_rows, self.n_days = len(day_code), int(day_code.max()) + 1
        n3, n_pool = int(c3.max()) + 1, int(cp.max()) + 1
        self.adjacent = adjacent

        def counts(rows, cell, width):
            """Day x cell counts of the given rows, sparse: about 4% of the entries are not zero."""
            return sparse.csr_matrix((np.ones(len(rows)), (day_code[rows], cell[rows])), shape=(self.n_days, width))

        every = np.arange(n_rows)
        k3, kp = counts(every, c3, n3), counts(every, cp, n_pool)
        self.k3_t, self.kp_t = k3.T.tocsr(), kp.T.tocsr()
        self.k3_own, self.kp_fall = counts(every[own], c3, n3), counts(every[~own], cp, n_pool)
        self.n3, self.n_pool = np.asarray(k3.sum(axis=0)).ravel(), np.asarray(kp.sum(axis=0)).ravel()
        self.n_day = np.asarray(k3.sum(axis=1)).ravel()
        n_paths = max(12, min(200, _CORRECTION_PATHS // self.n_days))
        rng = np.random.default_rng(seed)
        self.innovations = rng.standard_normal((self.n_days, n_paths))                  # days x paths
        # the hourly noise, at row level, taken through the same de-meaning to the daily series (independent of phi)
        eps = rng.standard_normal((n_paths, n_rows)) * sigma
        o3, o_pool = np.argsort(c3, kind="stable"), np.argsort(cp, kind="stable")
        starts3 = np.r_[0, np.flatnonzero(np.diff(c3[o3])) + 1]
        starts_pool = np.r_[0, np.flatnonzero(np.diff(cp[o_pool])) + 1]
        mean3 = np.add.reduceat(eps[:, o3], starts3, axis=1) / self.n3
        mean_pool = np.add.reduceat(eps[:, o_pool], starts_pool, axis=1) / self.n_pool
        used = np.where(own, mean3[:, c3], mean_pool[:, cp])
        day_starts = np.r_[0, np.flatnonzero(np.diff(day_code)) + 1]
        self.noise = np.ascontiguousarray((np.add.reduceat(eps - used, day_starts, axis=1) / self.n_day).T)
        self.noise_variance = float(np.mean(np.var(self.noise, axis=0)))

    def _kept(self, phi):
        """Unit-stationary-variance AR(1) paths on the fit's days (days x paths), less what the cell means take out."""
        e = self.innovations
        x = np.empty_like(e)
        x[0] = e[0]
        c = math.sqrt(1.0 - phi ** 2)
        for t in range(1, self.n_days):
            x[t] = phi * x[t - 1] + c * e[t]
        mean3 = (self.k3_t @ x) / self.n3[:, None]
        mean_pool = (self.kp_t @ x) / self.n_pool[:, None]
        return x - (self.k3_own @ mean3 + self.kp_fall @ mean_pool) / self.n_day[:, None]

    def statistic(self, phi, signal_variance):
        """(mean lag-one correlation of the simulated daily series, share of a unit stationary variance that survives
        the de-meaning), for a daily component of persistence phi whose stationary variance makes the series' variance
        equal the observed one (``signal_variance`` plus the noise's)."""
        kept = self._kept(phi)
        share = float(np.mean(np.var(kept, axis=0)))
        series = math.sqrt(signal_variance / share) * kept + self.noise
        return float(np.mean(_lag1(series.T, self.adjacent))), share

    def solve(self, observed):
        """(phi, innovation SD, status) of the daily component behind the observed daily series (a pandas Series)."""
        x = observed.to_numpy(dtype=float)
        signal_variance = float(np.var(x)) - self.noise_variance
        if not signal_variance > 0:
            return 0.0, 0.0, "no_signal"
        target = float(_lag1(x[None, :], self.adjacent)[0])
        lo, hi = _PHI_SEARCH
        if target <= self.statistic(lo, signal_variance)[0]:
            phi, status = lo, "clipped_low"
        elif target >= self.statistic(hi, signal_variance)[0]:
            phi, status = hi, "clipped_high"
        else:
            for _ in range(_CORRECTION_STEPS):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if self.statistic(mid, signal_variance)[0] < target else (lo, mid)
            phi, status = 0.5 * (lo + hi), "corrected"
        stationary = math.sqrt(signal_variance / self.statistic(phi, signal_variance)[1])
        return phi, stationary * math.sqrt(1.0 - phi ** 2), status


def fit_node(rows):
    """Fit one node: ``rows`` has settle_date, settle_hour, score and pnl_open (market and node are ignored for a
    single node). The seasonal means are the (hour, weekday class, month) cell means, a cell seen on fewer than 5 days
    taking the (hour, month) mean and being recorded in ``fallback_cells``; the daily component is an AR(1) fitted to
    the daily means of the residuals from these means; the hourly noise SD is the within-day SD of the residuals.

    The hourly noise SD divides by the degrees of freedom left after the cell means and the day means (N - days -
    cells + components of the day-cell graph, 12 months' worth for a two-way table), not by N - 1, which reads about 4%
    low in a year of data. The P&L noise SD is also estimated by (hour, weekday class), ``sd_eta_cell``, for cells with
    at least 30 residuals. Raises ValueError for an empty frame, a missing column, a NaN or more than one node.

    The daily AR(1) of the score and of the P&L is corrected for what the fit itself does to it (``_PersistenceMap``,
    Ruling 84a, I-1): ``phi_*`` and ``sd_*`` are the corrected values (the stationary SD reproduces the variance of the
    daily means beyond the hourly noise they carry), ``*_raw`` the plain AR(1) estimates, ``phi_status`` how each
    correction ended. It is skipped, and recorded as such, for fewer than 30 day pairs. Deterministic."""
    _check_fit_rows(rows)
    r = rows.assign(month=pd.to_datetime(rows["settle_date"]).dt.month, wclass=_wclass(rows["settle_date"]))
    by_hm = r.groupby(["settle_hour", "month"])
    by_hwm = r.groupby(["settle_hour", "wclass", "month"])
    mu = {(int(h), int(m)): float(v) for (h, m), v in by_hm["score"].mean().items()}
    nu = {(int(h), int(m)): float(v) for (h, m), v in by_hm["pnl_open"].mean().items()}
    days_cell = by_hwm["settle_date"].transform("nunique").to_numpy()
    own = days_cell >= _MIN_MEAN_DAYS            # the rows whose (hour, weekday class, month) cell has days enough
    estimated = (r[own].groupby(["settle_hour", "wclass", "month"])
                 .agg(score=("score", "mean"), pnl=("pnl_open", "mean")))
    mu_w = {(int(h), int(w), int(m)): float(v) for (h, w, m), v in estimated["score"].items()}
    nu_w = {(int(h), int(w), int(m)): float(v) for (h, w, m), v in estimated["pnl"].items()}
    fallback = tuple(sorted((int(h), int(w), int(m))
                            for h, w, m in r[~own].groupby(["settle_hour", "wclass", "month"]).size().index))
    n_used = np.where(own, days_cell, by_hm["settle_date"].transform("nunique").to_numpy())
    kappa = 1.0 - 1.0 / n_used.astype(float)                         # the share of its variance a row's mean leaves
    s_mean = np.where(own, by_hwm["score"].transform("mean"), by_hm["score"].transform("mean"))
    p_mean = np.where(own, by_hwm["pnl_open"].transform("mean"), by_hm["pnl_open"].transform("mean"))
    r = r.assign(s_res=r["score"].to_numpy() - s_mean, p_res=r["pnl_open"].to_numpy() - p_mean)
    by_day = r.groupby("settle_date")
    daily = by_day[["s_res", "p_res"]].mean().sort_index()
    phi_s_raw, sd_s_raw, n_pairs = _ar1(daily["s_res"])
    phi_p_raw, sd_p_raw, _ = _ar1(daily["p_res"])
    key = pd.DataFrame({"h": r["settle_hour"].to_numpy(), "w": np.where(own, r["wclass"].to_numpy(), -1),
                        "m": r["month"].to_numpy()})
    df = _residual_df(by_day.ngroup().to_numpy(), key.groupby(["h", "w", "m"]).ngroup().to_numpy())
    dev_s, dev_p = r["s_res"] - by_day["s_res"].transform("mean"), r["p_res"] - by_day["p_res"].transform("mean")
    sd_eps, sd_eta, cell_sd = _noise_sd(dev_s, df), _noise_sd(dev_p, df), _cell_noise_sd(r, dev_p, kappa, df)
    phi_s, sd_s, phi_p, sd_p = phi_s_raw, sd_s_raw, phi_p_raw, sd_p_raw
    status = {"score": "uncorrected", "pnl": "uncorrected"}
    if n_pairs >= _CORRECTION_MIN_PAIRS:
        hour, adjacent = r["settle_hour"].to_numpy(), _adjacent(daily.index)
        day_code, c3, cp = by_day.ngroup().to_numpy(), by_hwm.ngroup().to_numpy(), by_hm.ngroup().to_numpy()
        pnl_sigma = np.array([cell_sd.get(c, sd_eta) for c in zip(hour.tolist(), r["wclass"].tolist())])
        for name, sigma, series in (("score", np.full(len(r), sd_eps), daily["s_res"]),
                                    ("pnl", pnl_sigma, daily["p_res"])):
            phi, sd, status[name] = _PersistenceMap(day_code, hour, own, c3, cp, sigma, adjacent).solve(series)
            if name == "score":
                phi_s, sd_s = phi, sd
            else:
                phi_p, sd_p = phi, sd
    return NodeParams(mu, nu, phi_s, sd_s, sd_eps, phi_p, sd_p, sd_eta, cell_sd, n_pairs, mu_w, nu_w, fallback,
                      phi_s_raw=phi_s_raw, sd_s_raw=sd_s_raw, phi_p_raw=phi_p_raw, sd_p_raw=sd_p_raw, phi_status=status)


def persistence_upper(phi, sd, n_fit_days):
    """(phi_u, sd_u): the upper end of the persistence interval, ``min(0.99, phi + 1.96 * sqrt((1 - phi^2) /
    n_fit_days))``, and the innovation SD that keeps the stationary SD unchanged, ``sd * sqrt((1 - phi_u^2) /
    (1 - phi^2))``. Raises ValueError for fewer than 1 day pair and for |phi| above 0.99."""
    if n_fit_days < 1:
        raise ValueError(f"persistence_upper: n_fit_days must be at least 1 (a NodeParams built by hand has 0), "
                         f"got {n_fit_days}")
    if not abs(phi) <= _PHI_CAP:
        raise ValueError(f"persistence_upper: phi must lie in [-{_PHI_CAP}, {_PHI_CAP}], got {phi}")
    phi_u = min(_PHI_CAP, phi + _PERSISTENCE_Z * math.sqrt((1.0 - phi ** 2) / n_fit_days))
    return phi_u, sd * math.sqrt((1.0 - phi_u ** 2) / (1.0 - phi ** 2))


def _calendar_deciles(mu_arr):
    """Calendar-propensity deciles of one node's window, by value (the rule of ``measures.add_calendar_deciles``)."""
    return _deciles_by_value(pd.Series(mu_arr)).to_numpy().astype(int)


@dataclass
class WindowTemplate:
    """One node's seasonal means over a window of days (24 hours each), its two stratifications, and the P&L noise SD
    of each row (``eta_arr``: that of its (hour, weekday class) cell, the pooled SD where the fit has none).

    ``strata_primary`` numbers the window's (hour, weekday class, month) cells, the registered strata;
    ``strata_decile`` the calendar-propensity deciles of ``mu_arr`` (0..9), the sensitivity. A template built by hand
    without the calendar has no primary strata (``WindowTemplate.build`` always sets them)."""
    params: NodeParams
    n_days: int
    mu_arr: np.ndarray
    nu_arr: np.ndarray
    strata_decile: np.ndarray | None = None
    eta_arr: np.ndarray | None = None
    strata_primary: np.ndarray | None = None

    def __post_init__(self):
        if self.strata_decile is None:
            self.strata_decile = _calendar_deciles(self.mu_arr)
        if self.eta_arr is None:
            self.eta_arr = np.full(self.n_days * 24, float(self.params.sd_eta))

    def strata_codes(self, strata):
        """The stratum number of every row, 0..n-1 within this node: ``"primary"`` or ``"decile"``."""
        codes = self.strata_primary if strata == "primary" else self.strata_decile
        if codes is None:
            raise ValueError(f"WindowTemplate: no {strata} strata; build the template with WindowTemplate.build")
        return codes

    @classmethod
    def build(cls, params, days):
        """The template of ``days``. A row's seasonal mean is that of its (hour, weekday class, month) cell, else the
        (hour, month) mean. A cell with neither takes the node's mean level, with a UserWarning (a fit of the whole
        year, as the spec's, has every cell)."""
        months = np.repeat([pd.Timestamp(d).month for d in days], 24)
        hours = np.tile(np.arange(24), len(days))
        wclass = np.repeat([int(pd.Timestamp(d).dayofweek >= 5) for d in days], 24)
        mu_default = float(np.mean(list(params.mu.values())))
        nu_default = float(np.mean(list(params.nu.values())))
        cells = list(zip(hours.tolist(), wclass.tolist(), months.tolist()))
        mu = np.array([params.mu_w[c] if c in params.mu_w else params.mu.get((c[0], c[2]), mu_default) for c in cells])
        nu = np.array([params.nu_w[c] if c in params.nu_w else params.nu.get((c[0], c[2]), nu_default) for c in cells])
        distinct = set(cells)
        absent = sorted(c for c in distinct if (c not in params.mu_w and (c[0], c[2]) not in params.mu)
                        or (c not in params.nu_w and (c[0], c[2]) not in params.nu))
        if absent:
            warnings.warn(f"WindowTemplate.build: the fit has no seasonal mean for {len(absent)} of {len(distinct)} "
                          f"(hour, weekday class, month) cells of the window (months "
                          f"{sorted({m for _, _, m in absent})}); they take the node's mean level",
                          UserWarning, stacklevel=2)
        eta = np.array([params.sd_eta_cell.get((h, w), params.sd_eta) for h, w, _ in cells])
        primary = np.unique((months * 2 + wclass) * 24 + hours, return_inverse=True)[1]
        return cls(params, len(days), mu, nu, eta_arr=eta, strata_primary=primary)


def _sim_ar1(n, phi, sd, rng):
    x = np.empty(n)
    x[0] = rng.normal(0.0, sd / np.sqrt(max(1e-9, 1.0 - phi ** 2)))
    shocks = rng.normal(0.0, sd, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + shocks[t]
    return x


def _check_persistence(persistence, phi_scale, who):
    """Validate the persistence switch; warn once per public call about the deprecated ``phi_scale``."""
    if persistence not in _PERSISTENCE_MODES:
        raise ValueError(f"{who}: persistence must be 'fit' or 'upper', got {persistence!r}")
    if phi_scale != 1.0:
        if persistence != "fit":
            raise ValueError(f"{who}: phi_scale cannot be combined with persistence={persistence!r}; phi_scale is "
                             "deprecated, use persistence='upper'")
        warnings.warn(f"{who}: phi_scale is deprecated (it multiplied phi and left the innovation SD fixed, so it "
                      "inflated the stationary SD too); use persistence='upper'", DeprecationWarning, stacklevel=3)


def _check_strata(strata, who):
    if strata not in _STRATA_MODES:
        raise ValueError(f"{who}: strata must be 'primary' or 'decile', got {strata!r}")


def _daily_params(p, phi_scale, persistence):
    """(phi_s, sd_s, phi_p, sd_p) as simulated: the fitted ones, or the upper end of the persistence interval with the
    stationary SDs unchanged, or (deprecated) phi times phi_scale with the innovation SDs unchanged."""
    if persistence == "upper":
        phi_s, sd_s = persistence_upper(p.phi_s, p.sd_s, p.n_fit_days)
        phi_p, sd_p = persistence_upper(p.phi_p, p.sd_p, p.n_fit_days)
        return phi_s, sd_s, phi_p, sd_p
    return min(_PHI_CAP, p.phi_s * phi_scale), p.sd_s, min(_PHI_CAP, p.phi_p * phi_scale), p.sd_p


def _simulate(template, rho, rng, phi_scale=1.0, hetero=False, persistence="fit"):
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"simulate: rho must lie in [-1, 1], got {rho}")
    p, n = template.params, template.n_days
    phi_s, sd_s, phi_p, sd_p = _daily_params(p, phi_scale, persistence)
    xs = _sim_ar1(n, phi_s, sd_s, rng)
    z = _sim_ar1(n, phi_p, sd_p, rng)
    zs = (xs - xs.mean()) / (xs.std() + 1e-12) * (z.std() + 1e-12)
    xp = -rho * zs + np.sqrt(max(0.0, 1.0 - rho ** 2)) * z
    score = template.mu_arr + np.repeat(xs, 24) + rng.normal(0.0, p.sd_eps, n * 24)
    pnl = template.nu_arr + np.repeat(xp, 24) + rng.normal(0.0, template.eta_arr if hetero else p.sd_eta, n * 24)
    return score, pnl


def simulate(template, rho, rng, phi_scale=1.0, hetero=False, persistence="fit"):
    """One window of hourly (score, P&L) for a node.

    ``persistence="fit"`` uses the fitted phi of the daily score and P&L components; ``"upper"`` uses the upper end of
    their intervals (``persistence_upper``: needs ``NodeParams.n_fit_days``), with the stationary SDs unchanged.
    ``phi_scale`` is deprecated: it multiplies both fitted persistences (capped at 0.99) with the innovation SDs
    unchanged, so the stationary SD of a daily component grows by sqrt((1 - phi^2) / (1 - (phi_scale * phi)^2)) as well;
    it cannot be combined with ``persistence="upper"``. ``hetero`` draws each hour's P&L noise with the SD of its
    (hour, weekday class) cell instead of the pooled one; the generator is consumed identically in every case, so the
    draws are common to the scenarios. Raises ValueError unless -1 <= rho <= 1 and persistence is 'fit' or 'upper'."""
    _check_persistence(persistence, phi_scale, "simulate")
    return _simulate(template, rho, rng, phi_scale, hetero, persistence)


def _cube(templates_dep, templates_cal, rho, rng, alpha, phi_scale, hetero=False, persistence="fit",
          strata="primary"):
    """The H-E cube of one window: per node, a threshold tau (the 1 - alpha quantile of an independent calibration
    quarter, P1) applied to a deployment quarter drawn at coupling rho. Rows are complete (24 hours x days x nodes)
    and (day, cell) pairs are unique. Strata are numbered node after node, so no two nodes share one."""
    n = templates_dep[0].n_days
    day_idx, cells, sid, node_idx, pnl_all, alert_all = [], [], [], [], [], []
    base = 0
    for k, (tpl_d, tpl_c) in enumerate(zip(templates_dep, templates_cal)):
        cal_score, _ = _simulate(tpl_c, 0.0, rng, phi_scale, False, persistence)
        tau = np.quantile(cal_score, 1 - alpha)
        score, pnl = _simulate(tpl_d, rho, rng, phi_scale, hetero, persistence)
        day_idx.append(np.repeat(np.arange(n), 24))
        cells.append(k * 24 + np.tile(np.arange(24), n))
        codes = tpl_d.strata_codes(strata)
        sid.append(base + codes)
        base += int(codes.max()) + 1
        node_idx.append(np.full(n * 24, k))
        pnl_all.append(pnl)
        alert_all.append(score >= tau)
    return RowCube.from_arrays(np.concatenate(day_idx), n, np.concatenate(cells),
                               np.concatenate(sid), np.concatenate(pnl_all), np.concatenate(alert_all),
                               node_idx=np.concatenate(node_idx))


def _nodes(params_by_node, who):
    if not params_by_node:
        raise ValueError(f"{who}: needs at least one node")
    return sorted(params_by_node)


def _pooled_selc(cubes):
    """Pooled loss-weighted SELC of the cubes' alerts: total excess over total loss. Taken from the cubes and not from
    ``rotation_test``, whose statistic is the studentized excess (Ruling 55g), not a SELC."""
    loss = [float(c.loss.sum()) for c in cubes]
    total = sum(loss)
    return sum(c.selc() * L for c, L in zip(cubes, loss)) / total if total > 0 else 0.0


def he_size_power(params_by_node, cal_days_list, dep_days_list, rho, n_sim, alpha_tests=(0.05 / 3, 0.05),
                  seed=20261022, alpha=0.05, phi_scale=1.0, hetero=False, persistence="fit", strata="primary"):
    """Rejection rate of the (pooled) H-E rotation test. Window i deploys on dep_days_list[i]
    and calibrates on cal_days_list[i] (the same quarter one year earlier).

    ``hetero=True`` is the scenario with the fitted heteroskedastic P&L noise (SD by hour and weekday class); the
    default is the pooled SD. ``persistence="upper"`` is the scenario with the upper end of the persistence interval
(see ``simulate``); ``phi_scale`` is deprecated. ``strata="primary"`` stratifies the SELC by node x hour x weekday
    class x month (the registered strata), ``"decile"`` by the calendar-propensity deciles (the sensitivity). The
    rotation reference set is ``rotation_test``'s default. ``mean_selc`` is the mean pooled SELC of the simulated
    alerts (the scale of Task 11's effect target), ``mean_t_obs`` the mean of the statistic ``rotation_test`` reports,
    whatever it is.

    Raises ValueError for no node, no window, calibration and deployment lists of different lengths, an empty
    calibration window, n_sim < 1, an unknown persistence or strata, and (from ``rotation_test``, when the first
    simulation reaches it) a deployment window too short for its reference set."""
    nodes = _nodes(params_by_node, "he_size_power")
    _check_persistence(persistence, phi_scale, "he_size_power")
    _check_strata(strata, "he_size_power")
    if len(cal_days_list) != len(dep_days_list):
        raise ValueError("he_size_power: needs one calibration window per deployment window, got "
                         f"{len(cal_days_list)} and {len(dep_days_list)}")
    if not dep_days_list:
        raise ValueError("he_size_power: needs at least one window")
    if any(len(c) == 0 for c in cal_days_list):
        raise ValueError("he_size_power: a calibration window is empty")
    if n_sim < 1:
        raise ValueError(f"he_size_power: n_sim must be at least 1, got {n_sim}")
    rng = np.random.default_rng(seed)
    tpl_cal = [[WindowTemplate.build(params_by_node[k], days) for k in nodes] for days in cal_days_list]
    tpl_dep = [[WindowTemplate.build(params_by_node[k], days) for k in nodes] for days in dep_days_list]
    pvals, t_obs, selcs = [], [], []
    for _ in range(n_sim):
        cubes = [_cube(td, tc, rho, rng, alpha, phi_scale, hetero, persistence, strata)
                 for td, tc in zip(tpl_dep, tpl_cal)]
        res = rotation_test(cubes)
        pvals.append(res["p"])
        t_obs.append(res["t_obs"])
        selcs.append(_pooled_selc(cubes))
    pvals = np.asarray(pvals)
    out = {f"reject_{a:g}": float(np.mean(pvals <= a)) for a in alpha_tests}
    out.update({"mean_selc": float(np.mean(selcs)), "mean_t_obs": float(np.mean(t_obs)), "n_sim": n_sim, "rho": rho,
                "phi_scale": phi_scale, "hetero": hetero, "persistence": persistence, "strata": strata})
    return out


def _hb_templates(params_by_node, cal_days, dep_days):
    """(calibration template, deployment template) of every node, in sorted node order."""
    return [(WindowTemplate.build(params_by_node[k], cal_days), WindowTemplate.build(params_by_node[k], dep_days))
            for k in sorted(params_by_node)]


def _score_sd(template):
    """The unit of ``delta`` and ``common_shift`` for one node and window: the fitted stationary SD of the score over
    the window, sqrt(variance of the seasonal pattern + daily variance + hourly noise variance). One number, not the
    realised SD of each simulated block: that made the shift random, and different in the two blocks, which adds
    variance the bootstrap cannot see (Ruling 84a, M-1). It does not change with ``persistence``, which holds the
    stationary SD."""
    p = template.params
    return math.sqrt(float(np.var(template.mu_arr)) + p.sd_s ** 2 / (1.0 - p.phi_s ** 2) + p.sd_eps ** 2)


def _hb_draw(templates, rng, delta, phi_scale, persistence="fit", common_shift=0.0):
    """One draw of the alternative and the base block, as arrays: per node (in the order of ``templates``)
    ``((calibration, deployment) of the alternative block, (calibration, deployment) of the base block)``, each the hourly
    scores of its window, day after day (24 rows a day). Both blocks come from the same parameters, independently. Their
    deployment scores are shifted up by ``common_shift`` of the node's score SD (``_score_sd``; both blocks: equal
    portability under drift) and, for the alternative block only, by a further ``delta``."""
    blocks = []
    for tc, td in templates:
        sd = _score_sd(td)
        pair = []
        for block in ("alt", "base"):
            cs, _ = _simulate(tc, 0.0, rng, phi_scale, False, persistence)
            ds, _ = _simulate(td, 0.0, rng, phi_scale, False, persistence)
            shift = common_shift + (delta if block == "alt" else 0.0)
            if shift:
                ds = ds + shift * sd
            pair.append((cs, ds))
        blocks.append(tuple(pair))
    return blocks


def _frames_of(nodes, blocks, cal_days, dep_days):
    """The two blocks of ``_hb_draw`` as the score frames ``block_contrast_bootstrap`` reads (market, node, split,
    settle_date, score)."""
    frames = {"alt": [], "base": []}
    cal_dates, dep_dates = np.repeat(cal_days, 24), np.repeat(dep_days, 24)
    for k, pair in zip(nodes, blocks):
        for block, (cs, ds) in zip(("alt", "base"), pair):
            for split, dates, sc in (("calib", cal_dates, cs), ("deploy", dep_dates, ds)):
                frames[block].append(pd.DataFrame({"market": k[0], "node": k[1], "split": split,
                                                   "settle_date": dates, "score": sc}))
    return pd.concat(frames["alt"]), pd.concat(frames["base"])


def _hb_frames(nodes, templates, cal_days, dep_days, rng, delta, phi_scale, persistence="fit", common_shift=0.0):
    """One draw of the alternative and the base block as the score frames ``block_contrast_bootstrap`` reads (market,
    node, split, settle_date, score): ``_hb_draw`` in frames."""
    return _frames_of(nodes, _hb_draw(templates, rng, delta, phi_scale, persistence, common_shift), cal_days, dep_days)


def _hb_sims(templates, n_sim, seed, delta, phi_scale, persistence, common_shift):
    """The simulations ``hb_size`` and ``calibrate_delta`` share: per draw the blocks (``_hb_draw``) and the seed of the
    bootstrap, taken from the same generator so that both see the same stream."""
    rng = np.random.default_rng(seed)
    for _ in range(n_sim):
        blocks = _hb_draw(templates, rng, delta, phi_scale, persistence, common_shift)
        yield blocks, int(rng.integers(10 ** 9))


def _rate(cal, dep, alpha):
    """R of P1 for one node and block: the share of the deployment rows at or above tau, the 1 - alpha quantile of the
    calibration scores, over alpha. The arithmetic of ``es.bootstrap._log_error`` on the real days, so that the two agree
    to the last bit."""
    tau = np.quantile(cal, 1 - alpha)
    return float(np.count_nonzero(dep >= tau) / float(dep.size)) / alpha


def _block_rates(blocks, alpha):
    """(R of every node in the alternative block, R of every node in the base block), in the order of ``blocks``."""
    return (np.array([_rate(cal, dep, alpha) for (cal, dep), _ in blocks]),
            np.array([_rate(cal, dep, alpha) for _, (cal, dep) in blocks]))


def _abs_log_r(rate, n_dep_rows, alpha):
    """|log10 R| per node, floored at half an alert like ``es.bootstrap._log_error`` (the zero-activity floor). It takes
    ``math.log10``, the function the registered code calls; numpy's ``log10`` is not guaranteed to agree with it to the
    last bit on every platform (it does on this one)."""
    floor = 1.0 / (2.0 * n_dep_rows * alpha)
    return np.array([abs(math.log10(max(float(r), floor))) for r in np.asarray(rate, dtype=float)])


def _paired_gaps(rate_alt, rate_base, n_dep_rows, alpha):
    """|log10 R|(alt) - |log10 R|(base), per node, for one simulation."""
    return _abs_log_r(rate_alt, n_dep_rows, alpha) - _abs_log_r(rate_base, n_dep_rows, alpha)


def _node_gap(rate_alt, rate_base, n_dep_rows, alpha):
    """Mean over nodes of the paired gap for one simulation."""
    return float(np.mean(_paired_gaps(rate_alt, rate_base, n_dep_rows, alpha)))


def _median_gap(rate_alt, rate_base, n_dep_rows, alpha):
    """Median over nodes of the paired gap for one simulation: the H-B statistic (the bootstrap's ``t_hat``)."""
    return float(np.median(_paired_gaps(rate_alt, rate_base, n_dep_rows, alpha)))


# --- the fast path of the H-B bootstrap (Ruling 92a) -----------------------------------------------------------------------
# ``es.bootstrap.block_contrast_bootstrap`` re-places tau in every replicate with ``np.quantile`` of the resampled
# calibration window, built row by row with ``np.repeat``: once per node, block and replicate, nearly all of its run time.
# On a complete grid (24 hours of every day, in every window) the resample is a count per day, so the quantile is one
# element or two of the sorted window found from cumulative counts, and the alert share a table lookup. Everything is
# computed with the registered function's own arithmetic (numpy's linear quantile, ``math.log10``, the same order of
# operations), so the answer is the same bit for bit, not close: ``test_simulate`` checks it against the registered
# function on random grids.

def _order_pair_full(v, day, counts_t, lo):
    """Elements ``lo`` and ``lo + 1`` (ascending, 0-based) of each replicate's resampled calibration sample, the rows
    repeated by the count of their day, from all the rows. ``v`` are the rows' scores in ascending order, ``day`` their
    days, ``counts_t`` the (days, replicates) counts."""
    reached = np.cumsum(counts_t[day], axis=0, dtype=np.int32)             # elements of the sample up to each row
    return v[(reached <= lo).sum(0)], v[(reached <= lo + 1).sum(0)]


def _order_pair_top(v, day, counts_t, lo, k):
    """The same two elements from the ``k`` largest rows alone, and which replicates those rows do not suffice for
    (their resampled sample holds too few elements of these rows; the elements returned for them are void)."""
    above = v.size - 1 - lo                                  # elements of the sample above the one at index lo
    top, top_day = v[::-1][:k], day[::-1][:k]
    reached = np.cumsum(counts_t[top_day], axis=0, dtype=np.int32)         # elements of the sample down from the largest
    last = k - 1
    return (top[np.minimum((reached <= above).sum(0), last)], top[np.minimum((reached <= above - 1).sum(0), last)],
            reached[-1] <= above)


def _replicate_thresholds(cal, counts_t, q):
    """``np.quantile(np.repeat(cal, counts[b][day of each row]), q)`` for every replicate b, to the last bit, from the
    scores ``cal`` of one calibration window (24 rows a day, days in order) and ``counts_t``, the (days, replicates) day
    counts of the resamples. Each resample must hold as many days as the window, as the stationary bootstrap's do: the
    sample then has ``cal.size`` elements, and numpy's linear rule puts the quantile at ``h = (n - 1) q`` between the
    elements floor(h) and floor(h) + 1, weighting them by h - floor(h)."""
    n = cal.size
    h = (n - 1) * q
    lo = int(math.floor(h))
    gamma = h - lo
    order = np.argsort(cal, kind="stable")
    v, day = cal[order], order // 24
    k = min(n, max(_FAST_TOP_FACTOR * (n - lo), _FAST_TOP_MIN))
    if k < n:
        a, b, short = _order_pair_top(v, day, counts_t, lo, k)
        if short.any():
            rest = np.flatnonzero(short)
            a[rest], b[rest] = _order_pair_full(v, day, counts_t[:, rest], lo)
    else:
        a, b = _order_pair_full(v, day, counts_t, lo)
    diff = b - a
    return b - diff * (1 - gamma) if gamma >= 0.5 else a + diff * gamma          # numpy's lerp, both of its branches


def _replicate_alert_counts(dep, tau, counts):
    """For every replicate b, the number of deployment rows at or above ``tau[b]``, each counted as often as its day is
    drawn (``counts[b]``): ``(np.repeat(counts[b], 24) * (dep >= tau[b])).sum()``, exactly. ``dep`` are the scores of
    one deployment window (24 rows a day, days in order)."""
    n_days = dep.size // 24
    order = np.argsort(dep, kind="stable")
    below = np.zeros((dep.size + 1, n_days), dtype=np.int32)
    below[np.arange(1, dep.size + 1), order // 24] = 1
    np.cumsum(below, axis=0, out=below)                      # below[r, d]: rows of day d among the r smallest scores
    rows_below = np.searchsorted(dep[order], tau, side="left")                 # the rows with a score under tau
    return np.einsum("bd,bd->b", 24 - below[rows_below], counts)


def _check_fast_blocks(blocks, alpha, n_rep):
    """The grids and arguments ``_hb_bootstrap_fast`` reproduces; ValueError for anything else. Returns the number of
    calibration and of deployment days."""
    if not blocks:
        raise ValueError("calibration-and-deployment bootstrap: no node, so there is nothing to resample")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"calibration-and-deployment bootstrap (fast path): alpha must lie in (0, 1), got {alpha}")
    if n_rep < 1:
        raise ValueError(f"calibration-and-deployment bootstrap (fast path): n_rep must be at least 1, got {n_rep}")
    n_cal, n_dep = blocks[0][0][0].size, blocks[0][0][1].size
    for pair in blocks:
        for cal, dep in pair:
            if cal.ndim != 1 or dep.ndim != 1 or cal.size != n_cal or dep.size != n_dep:
                raise ValueError("calibration-and-deployment bootstrap (fast path): every node and block needs the same "
                                 "number of days in its calibration window and in its deployment window")
    if n_cal % 24 or n_dep % 24 or n_cal == 0 or n_dep == 0:
        raise ValueError(f"calibration-and-deployment bootstrap (fast path): a window is a whole number of days of 24 "
                         f"hours, a multiple of 24 rows; got {n_cal} and {n_dep}")
    if max(n_cal, n_dep) > 24 * _FAST_MAX_DAYS:
        raise ValueError(f"calibration-and-deployment bootstrap (fast path): windows of up to {_FAST_MAX_DAYS} days, got "
                         f"{n_cal // 24} and {n_dep // 24} days")
    if math.floor((n_cal - 1) * (1.0 - alpha)) > n_cal - 2:
        raise ValueError(f"calibration-and-deployment bootstrap (fast path): alpha = {alpha} leaves no element above the "
                         f"quantile of a window of {n_cal} rows")
    if any(np.isnan(x).any() for pair in blocks for cd in pair for x in cd):
        raise ValueError("calibration-and-deployment bootstrap: NaN in ['score']; nothing is dropped silently")
    return n_cal // 24, n_dep // 24


def _hb_bootstrap_fast(blocks, alpha=0.05, n_rep=2000, mean_block=7, seed=20261022):
    """What ``es.bootstrap.block_contrast_bootstrap`` returns for the complete grids ``hb_size`` draws, to the last bit
    (``t_hat``, every element of ``t_star``, ``p``, ``ci``, ``ci_basic``, ``n_nodes``), far faster.

    ``blocks`` is ``_hb_draw``'s output: per node ``((calibration, deployment) of the alternative block, (calibration,
    deployment) of the base block)``, the hourly scores of each window, day after day (24 rows a day), the days in
    increasing order and the same in every window of a split. The days are resampled as the registered function does
    (``stationary_counts`` with ``seed`` for the calibration days and ``seed + 1`` for the deployment days). Raises
    ValueError for a grid or an ``alpha`` it cannot reproduce (windows over ``_FAST_MAX_DAYS`` days, a quantile with no
    element above it) and for a NaN score; ``hb_size`` sends those to the registered function."""
    n_cal, n_dep = _check_fast_blocks(blocks, alpha, n_rep)
    counts_cal = np.ascontiguousarray(stationary_counts(n_cal, mean_block, n_rep, seed).T, dtype=np.int32)
    counts_dep = stationary_counts(n_dep, mean_block, n_rep, seed + 1)
    n_star = counts_dep.sum(1) * 24.0                                       # deployment rows of each resample
    floor = 1.0 / (2.0 * n_star * alpha)
    e_hat, e_star = [], []
    for pair in blocks:
        hat, star = [], []
        for cal, dep in pair:
            hat.append(_abs_log_r([_rate(cal, dep, alpha)], float(dep.size), alpha)[0])
            tau = _replicate_thresholds(cal, counts_cal, 1 - alpha)
            share = _replicate_alert_counts(dep, tau, counts_dep) / n_star / alpha
            star.append(np.array([abs(math.log10(x)) for x in np.maximum(share, floor)]))
        e_hat.append(hat[0] - hat[1])
        e_star.append(star[0] - star[1])
    t_hat = float(np.median(e_hat))
    t_star = np.median(e_star, axis=0)
    return {"t_hat": t_hat, "t_star": t_star, "p": basic_p_one_sided(t_hat, t_star), "ci": percentile_ci(t_star),
            "ci_basic": basic_ci(t_hat, t_star), "n_nodes": len(blocks)}


def _fast_applies(cal_days, dep_days, alpha):
    """Whether the grids of ``hb_size`` are ones ``_hb_bootstrap_fast`` reproduces: windows of increasing, distinct days
    (the position of a day in the sorted days is its position in the window), at most ``_FAST_MAX_DAYS`` of them, and an
    ``alpha`` with an element above its quantile."""
    for days in (cal_days, dep_days):
        if not 1 <= len(days) <= _FAST_MAX_DAYS:
            return False
        index = pd.Index(days)
        if not (index.is_unique and index.is_monotonic_increasing):
            return False
    n = 24 * len(cal_days)
    return 0.0 < alpha < 1.0 and math.floor((n - 1) * (1.0 - alpha)) <= n - 2


def _check_shift(common_shift, who):
    if not math.isfinite(common_shift):
        raise ValueError(f"{who}: common_shift must be finite, got {common_shift}")


def hb_size(params_by_node, cal_days, dep_days, n_sim, n_rep=299, alpha_tests=(0.05 / 3, 0.05),
            seed=20261022, alpha=0.05, delta=0.0, phi_scale=1.0, persistence="fit", common_shift=0.0, fast=True):
    """Rejection rate of the H-B test when both blocks share parameters (delta = 0, the
    equal-portability null) or when the alternative block's deployment scores are shifted
    up by delta score-SDs (power).

    ``common_shift`` shifts BOTH blocks' deployment scores by that many score-SDs, on top of ``delta`` for the
    alternative block: with delta = 0 the blocks drift alike, equal portability under drift. ``persistence="upper"``
    simulates the upper end of the score's persistence interval (see ``simulate``); ``phi_scale`` is deprecated.

    ``n_rep`` is the number of bootstrap replicates B. The p-value is (1 + k) / (B + 1), so a valid test rejects at
    level a with probability floor(a (B + 1)) / (B + 1): with the default 299, (B + 1) a is whole for a = 0.05 / 3 (5)
    and for 0.05 (15), so both registered levels are exactly attainable. With 200 the level at 0.05 / 3 would be
    3 / 201 = 0.0149, a size read 10% too low (the registered B = 2000 gives 33 / 2001 = 0.0165).

    Besides the rejection rates, the dict gives the simulated effect on the scale of the registered effect size:
    ``mean_node_gap``, the mean over simulations and nodes of |log10 R|(alt) - |log10 R|(base), and ``mean_t_hat``, the
    mean of the test's own statistic, the median over nodes of that difference. The 95th percentile of the number of
    nodes outside each band, ``nodrift_fail_q95_*`` (base block), is the no-drift context and is NaN in a row with a
    delta or a common_shift.

    ``fast`` (Ruling 92a): the bootstrap is ``_hb_bootstrap_fast``, which returns what
    ``es.bootstrap.block_contrast_bootstrap`` returns, bit for bit, in about a tenth of the time. ``fast=False`` calls the
    registered function itself, as does ``fast=True`` for grids the fast path is not built for (days not increasing or
    repeated, windows over 400 days, an ``alpha`` with no element above its quantile), so the result is the same either
    way.

    Raises ValueError for no node, a node key that is not (market, node), an empty window, n_sim or n_rep < 1, a
    non-finite common_shift and an unknown persistence."""
    nodes = _nodes(params_by_node, "hb_size")
    _check_persistence(persistence, phi_scale, "hb_size")
    _check_shift(common_shift, "hb_size")
    if not all(isinstance(k, tuple) and len(k) == 2 for k in nodes):
        raise ValueError("hb_size: params_by_node must be keyed by (market, node)")
    if len(cal_days) == 0 or len(dep_days) == 0:
        raise ValueError("hb_size: the calibration and the deployment window must each hold at least one day")
    if n_sim < 1 or n_rep < 1:
        raise ValueError(f"hb_size: n_sim and n_rep must be at least 1, got {n_sim} and {n_rep}")
    templates = _hb_templates(params_by_node, cal_days, dep_days)
    n_dep_rows = len(dep_days) * 24
    use_fast = bool(fast) and _fast_applies(cal_days, dep_days, alpha)
    pvals, t_hats, gaps, fails = [], [], [], []
    for blocks, boot_seed in _hb_sims(templates, n_sim, seed, delta, phi_scale, persistence, common_shift):
        if use_fast:
            res = _hb_bootstrap_fast(blocks, alpha=alpha, n_rep=n_rep, seed=boot_seed)
        else:
            alt, base = _frames_of(nodes, blocks, cal_days, dep_days)
            res = block_contrast_bootstrap(alt, base, alpha=alpha, n_rep=n_rep, seed=boot_seed)
        pvals.append(res["p"])
        t_hats.append(res["t_hat"])
        rate_alt, rate = _block_rates(blocks, alpha)
        gaps.append(_node_gap(rate_alt, rate, n_dep_rows, alpha))
        fails.append({"0.5_2": int(((rate < 0.5) | (rate > 2)).sum()),
                      "0.67_1.5": int(((rate < 2 / 3) | (rate > 1.5)).sum()),
                      "0.8_1.25": int(((rate < 0.8) | (rate > 1.25)).sum())})
    pvals = np.asarray(pvals)
    out = {f"reject_{a:g}": float(np.mean(pvals <= a)) for a in alpha_tests}
    f = pd.DataFrame(fails)
    no_drift = delta == 0 and common_shift == 0
    out.update({f"nodrift_fail_q95_{c}": float(f[c].quantile(0.95)) if no_drift else float("nan") for c in f.columns})
    out.update({"mean_t_hat": float(np.mean(t_hats)), "mean_node_gap": float(np.mean(gaps)),
                "n_sim": n_sim, "delta": delta, "phi_scale": phi_scale, "persistence": persistence,
                "common_shift": common_shift})
    return out


def _interp_delta(grid, gaps, target):
    """The delta at which the gap curve reaches ``target``: linear interpolation on the running maximum of ``gaps``
    (so a dip from Monte Carlo noise cannot make the curve non-monotone), at the first crossing. Raises ValueError for a
    target outside [gaps[0], max(gaps)]."""
    mono = np.maximum.accumulate(np.asarray(gaps, dtype=float))
    if not mono[0] <= target <= mono[-1]:
        raise ValueError(f"calibrate_delta: target_gap {target:.4g} is outside the gap range of the grid "
                         f"[{mono[0]:.4g}, {mono[-1]:.4g}] (gaps {[round(float(g), 4) for g in gaps]} at deltas "
                         f"{list(grid)}); widen the grid or change the target")
    i = int(np.argmax(mono >= target))
    if i == 0 or mono[i] == target:
        return float(grid[i])
    return float(grid[i - 1] + (target - mono[i - 1]) / (mono[i] - mono[i - 1]) * (grid[i] - grid[i - 1]))


def calibrate_delta(params_by_node, cal_days, dep_days, target_gap, grid=(0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8),
                    n_sim=30, n_rep=100, seed=20261022, alpha=0.05, common_shift=0.0, persistence="fit"):
    """The ``delta`` (alternative block's shift, in score SDs) whose simulated gap equals ``target_gap``, the gap being
    the MEDIAN over nodes of the paired |log10 R|(alt) - |log10 R|(base) (Ruling 84a, M-4): that is the H-B statistic,
    and the effect size to match is half of its observed value (the median paired AQ3 difference), not of the mean over
    nodes, which a few nodes can dominate.

    The gap of every grid point is the mean over simulations of that median, what ``hb_size`` reports as ``mean_t_hat``
    for the same arguments and seed (the same simulations); the grid shares one seed, so the curve is drawn on common
    random numbers. ``delta`` is interpolated linearly, monotonically (``_interp_delta``). Raises ValueError if
    ``target_gap`` lies outside the gap range of the grid, for a grid that is not an increasing sequence of at least two
    positive deltas, a non-finite target and n_sim < 1. ``n_rep`` is accepted so that a call can mirror ``hb_size``'s
    arguments; the gap does not depend on the bootstrap, which is therefore not run.

    Returns {"delta", "grid", "grid_gaps", "target_gap", "statistic"}: ``grid_gaps`` is the measured gap at each grid
    delta and ``statistic`` says what it is."""
    nodes = _nodes(params_by_node, "calibrate_delta")
    _check_persistence(persistence, 1.0, "calibrate_delta")
    _check_shift(common_shift, "calibrate_delta")
    grid = [float(g) for g in grid]
    if len(grid) < 2 or grid[0] <= 0 or any(b <= a for a, b in zip(grid, grid[1:])):
        raise ValueError(f"calibrate_delta: grid must be an increasing sequence of at least two positive deltas, "
                         f"got {grid}")
    if not math.isfinite(target_gap):
        raise ValueError(f"calibrate_delta: target_gap must be finite, got {target_gap}")
    if n_sim < 1:
        raise ValueError(f"calibrate_delta: n_sim must be at least 1, got {n_sim}")
    if not all(isinstance(k, tuple) and len(k) == 2 for k in nodes):
        raise ValueError("calibrate_delta: params_by_node must be keyed by (market, node)")
    templates = _hb_templates(params_by_node, cal_days, dep_days)
    n_dep_rows = len(dep_days) * 24
    gaps = []
    for delta in grid:
        per_sim = [_median_gap(*_block_rates(blocks, alpha), n_dep_rows, alpha)
                   for blocks, _ in _hb_sims(templates, n_sim, seed, delta, 1.0, persistence, common_shift)]
        gaps.append(float(np.mean(per_sim)))
    return {"delta": _interp_delta(grid, gaps, float(target_gap)), "grid": grid, "grid_gaps": gaps,
            "target_gap": float(target_gap),
            "statistic": "median over nodes of the paired |log10 R| difference, mean over simulations"}
