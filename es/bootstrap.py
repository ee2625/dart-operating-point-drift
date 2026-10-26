"""Resampling and rotation inference (spec v3 §5.1, §6.3; review round 2 N1; controller Rulings 45, 55, 68, 69 and 118).
- Estimand intervals: circular stationary bootstrap over settlement days (Politis and
  Romano 1994), alerts fixed on the real path.
- H-E test (Rulings 55 and 69): circular day-rotation of alerts against fixed P&L within each window; windows are
  rotated independently. A window's shift set is {0} u {min_shift, ..., D - min_shift}, 0 being its observed state;
  the default min_shift = 1 makes it the full cyclic group {0, 1, ..., D - 1}. The reference set is the grid of the
  windows' shift sets, and p = (1 + #{non-identity cells with T >= T_obs}) / (number of cells). T is the pooled excess
  studentized by its within-stratum hypergeometric SD, each alert path at its own counts. Pooled SELC remains the
  reported estimand (``selc_obs``). The strata are those of the cube; ``RowCube.from_rows`` defaults to the primary
  strata of Ruling 68 (node x hour x weekday class x month).
- H-B test: calibration days and deployment days resampled jointly, tau re-placed.
- Calendar-aligned option of the level bootstrap (E5, H-S; Ruling 118): ``block_level_bootstrap(aligned=True)`` draws once
  over the (month, day) positions of the calibration and deployment windows and gives each day the count of its position,
  so a replicate keeps the matching of the two windows (the same calendar span in different years) that independent draws
  break. It is off by default, and ``block_contrast_bootstrap`` (H-B) has no such option: its scheme stays as registered.

Validity of the rotation p (Ruling 69). p is a randomization p-value, P(p <= a) <= a, under exchangeability of each
window's observed alert path with its rotations, and only when the shift set is the full cyclic group. The group is what
makes every cell of the grid the observed state of an equally likely world: whichever rotation the observed path is, it
meets the same set of paths. The set {0} u {14, ..., D - 14} that preceded Ruling 69 is not a group. Under day-to-day
persistence the rotations nearest the identity are the ones most like the observed path, leaving them out lets the
observed state rank too high, and the test over-rejected (alerts and losses following independent daily AR(1)
processes, a true null). A larger ``min_shift`` stays available as a parameter and carries no validity claim. The
exchangeability itself is an assumption, not something the test checks.

Design properties of the rotation test, for the methodology review:
- Why T and not the excess. Rotating a window moves its alerts into other calendar strata, where the dispersion of
  the loss can differ, so an unstudentized excess has another spread on a rotated path than on the observed one and the
  null is mis-scaled: where one weekday carried the alerts and ten times the loss dispersion it rejected 17% of true
  nulls at 0.05, against 3.2% for T (tests/es/test_bootstrap.py runs that case). T standardizes each path by the SD it
  would have if its alerts were chosen at random within the strata at its own counts. It cannot repair strata that
  are too coarse: where a stratum mixes rows with different alert rates and loss levels (propensity deciles that
  straddle calendar groups), SELC itself is positive in expectation under a calendar-only null and T inherits the
  shift; that is why Ruling 68 made hour x weekday class x month the primary strata and the deciles a sensitivity.
- Rotation drops an alert whose target (day, cell) has no row. On a complete day x cell grid the alert count is kept
  exactly; with gaps a rotated path holds fewer alerts than the observed one. T is standardized at each path's own
  counts, so the null's spread does not depend on that loss (the unstudentized excess's did). ``retained_share``
  reports, per window, the min and mean share of the observed alerts that survive: on 12 nodes x 92 days with hours
  missing at random it is 99.6%, 98.0% and 95.3% at 0.5%, 2% and 5% missing hours (tests/es/test_bootstrap.py pins
  these figures, over the 91 non-zero shifts of the default reference set). With gaps the rotations do not form a group
  acting on the alert paths, so the exchangeability above is then approximate.
- A window of D days has n_w = D - 2*min_shift + 1 rotations, so n_null = prod(n_w + 1) - 1 and the smallest p is
  1 / prod(n_w + 1). By default (min_shift 1) n_w + 1 = D: a window has D cells, n_null = D - 1, p is a multiple of
  1 / D and at least 1 / D, which is 0.0109 at D = 92, below alpha/3 = 0.0167; 60 days is the shortest window that can
  reject at alpha/3, and 59 cannot however well aligned. Two windows of 92 days give 1 / 8464. For min_shift m > 1 the
  smallest single-window p is 1 / (D - 2m + 2), i.e. 1 / (D - 26) for m = 14."""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .measures import STRATA_PRIMARY


def stationary_counts(n_days, mean_block=7, n_rep=2000, seed=20261022):
    """Day counts of ``n_rep`` circular stationary-bootstrap resamples (Politis and Romano 1994): row b holds how
    often each of the ``n_days`` days is drawn, and sums to ``n_days``. The first index is uniform; every next index
    starts a new block (uniform) with probability 1 / mean_block, else continues at the next day, wrapping from the
    last day to the first."""
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    counts = np.zeros((n_rep, n_days), dtype=np.int64)
    for b in range(n_rep):
        idx = np.empty(n_days, dtype=np.int64)
        idx[0] = rng.integers(n_days)
        new_block = rng.random(n_days) < p
        starts = rng.integers(n_days, size=n_days)
        for t in range(1, n_days):
            idx[t] = starts[t] if new_block[t] else (idx[t - 1] + 1) % n_days
        counts[b] = np.bincount(idx, minlength=n_days)
    return counts


def _codes(frame, cols):
    return frame.groupby(list(cols), sort=True, dropna=False, observed=True).ngroup().to_numpy()


_ROW_COLS = ("market", "node", "settle_date", "settle_hour", "pnl_open", "alert")


def _check_rows(rows, strata):
    cols = list(dict.fromkeys(_ROW_COLS + tuple(strata)))
    if len(rows) == 0:
        raise ValueError("RowCube.from_rows: the frame is empty")
    missing = [c for c in cols if c not in rows.columns]
    if missing:
        raise ValueError(f"RowCube.from_rows: missing column(s) {missing}")
    nan = [c for c in cols if rows[c].isna().any()]
    if nan:
        raise ValueError(f"RowCube.from_rows: NaN in {nan}; nothing is dropped silently")


def _require_binary(alert, where):
    """ValueError unless every value of ``alert`` is 0 or 1 (NaN and inf are neither). An alert is a yes/no flag: the
    variance of the rotation statistic, A (N - A) / (N (N - 1)) Q, is that of choosing A rows only if A counts rows."""
    other = (alert != 0) & (alert != 1)
    if other.any():
        raise ValueError(f"{where}: alert must be 0 or 1 (a yes/no flag), got {int(other.sum())} other value(s), e.g. "
                         f"{alert[other][0]!r}")


def _check_arrays(n_days, day_idx, cell_idx, sid, pnl, alert, node_idx):
    arrays = {"day_idx": day_idx, "cell_idx": cell_idx, "sid": sid, "pnl": pnl, "alert": alert, "node_idx": node_idx}
    if any(a.ndim != 1 for a in arrays.values()) or len({len(a) for a in arrays.values() if a.ndim == 1}) != 1:
        shapes = ", ".join(f"{k} {np.shape(v)}" for k, v in arrays.items())
        raise ValueError("RowCube.from_arrays: day_idx, cell_idx, sid, pnl, alert and node_idx need one entry per "
                         f"row, got {shapes}")
    if len(day_idx) == 0:
        raise ValueError("RowCube.from_arrays: no rows")
    for name in ("pnl", "alert"):
        bad = int((~np.isfinite(arrays[name])).sum())
        if bad:
            raise ValueError(f"RowCube.from_arrays: {name} holds {bad} non-finite value(s); NaN and inf are not "
                             "allowed")
    _require_binary(alert, "RowCube.from_arrays")
    for name in ("day_idx", "cell_idx", "sid", "node_idx"):
        values = arrays[name]
        if values.dtype.kind not in "iu":
            raise ValueError(f"RowCube.from_arrays: {name} must hold integers, got dtype {values.dtype}")
        if name != "day_idx" and values.min() < 0:
            raise ValueError(f"RowCube.from_arrays: {name} must not be negative, got a minimum of {values.min()}")
    if day_idx.min() < 0 or day_idx.max() >= n_days:
        raise ValueError(f"RowCube.from_arrays: day_idx must lie in [0, {n_days}), got values from {day_idx.min()} "
                         f"to {day_idx.max()}")


@dataclass
class RowCube:
    """Hourly rows as flat arrays, so that days can be reweighted and alerts rotated without pandas.

    Per row: ``day_idx`` (0..n_days-1), ``cell_idx`` (its market, node, hour and occurrence), ``sid`` (stratum),
    ``node_idx`` (market, node); ``nd_day`` holds the day of each distinct (node, day) pair. Per stratum: ``strat_n``
    (its rows N), ``strat_loss`` (their loss L) and ``strat_ss`` (the sum of squared deviations of the loss from the
    stratum mean, Q - L^2 / N). None of the three depends on the alert path, so a rotation need not recompute them.

    Every statistic takes day weights ``w`` (bootstrap counts; None means 1 for each day) and equals the statistic
    of the frame in which day d appears w_d times, each copy as a distinct day. ``alert`` replaces the real alert
    path (rotation). SELC is 0.0 when the total loss is 0, as in ``es.measures``; SNET is 0.0 when the weights leave
    no node-days (``es.measures.snet`` cannot meet that case). ``selc`` and ``snet`` raise ValueError when their
    excess or total is not finite, instead of returning 0.0.
    """
    n_days: int
    day_idx: np.ndarray
    cell_idx: np.ndarray
    n_cells: int
    sid: np.ndarray
    n_strata: int
    loss: np.ndarray
    profit: np.ndarray
    pnl: np.ndarray
    alert: np.ndarray
    days: np.ndarray | None = None
    node_idx: np.ndarray | None = None
    nd_day: np.ndarray | None = None
    strat_n: np.ndarray | None = None
    strat_loss: np.ndarray | None = None
    strat_ss: np.ndarray | None = None

    @classmethod
    def from_arrays(cls, day_idx, n_days, cell_idx, sid, pnl, alert, days=None, node_idx=None):
        """Raises ValueError for: no rows or arrays that differ in length; a non-finite ``pnl`` or ``alert``; an
        ``alert`` with a value other than 0 and 1 (a bool or integer array is fine); a ``day_idx``, ``cell_idx``,
        ``sid`` or ``node_idx`` that is not integer, or negative, or (``day_idx``) outside [0, n_days); and two rows
        sharing a (day_idx, cell_idx) pair, which rotating the alerts would overwrite."""
        pnl, alert = np.asarray(pnl, dtype=float), np.asarray(alert, dtype=float)
        day_idx, cell_idx, sid = np.asarray(day_idx), np.asarray(cell_idx), np.asarray(sid)
        node_idx = np.zeros(len(day_idx), dtype=np.int64) if node_idx is None else np.asarray(node_idx)
        _check_arrays(int(n_days), day_idx, cell_idx, sid, pnl, alert, node_idx)
        day_idx, cell_idx, sid, node_idx = (a.astype(np.int64, copy=False) for a in (day_idx, cell_idx, sid, node_idx))
        n_cells, n_strata = int(cell_idx.max()) + 1, int(sid.max()) + 1
        pair, times = np.unique(day_idx * n_cells + cell_idx, return_counts=True)
        if (times > 1).any():
            first = int(pair[times > 1][0])
            raise ValueError(f"RowCube.from_arrays: {int((times > 1).sum())} (day_idx, cell_idx) pair(s) occur more "
                             f"than once, e.g. day {first // n_cells}, cell {first % n_cells}; rotating the alerts "
                             "would overwrite one row with another")
        loss = np.clip(-pnl, 0, None)
        strat_n = np.bincount(sid, minlength=n_strata).astype(float)
        strat_loss = np.bincount(sid, loss, n_strata)
        squares = np.bincount(sid, loss * loss, n_strata)
        with np.errstate(invalid="ignore", divide="ignore"):
            strat_ss = np.where(strat_n > 0, np.maximum(squares - strat_loss ** 2 / strat_n, 0.0), 0.0)
        pairs = np.unique(node_idx * int(n_days) + day_idx)
        return cls(n_days=int(n_days), day_idx=day_idx, cell_idx=cell_idx, n_cells=n_cells, sid=sid,
                   n_strata=n_strata, loss=loss, profit=np.clip(pnl, 0, None), pnl=pnl, alert=alert, days=days,
                   node_idx=node_idx, nd_day=pairs % int(n_days),
                   strat_n=strat_n, strat_loss=strat_loss, strat_ss=strat_ss)

    @classmethod
    def from_rows(cls, rows, strata=STRATA_PRIMARY):
        """The cube of a frame of hourly rows (market, node, settle_date, settle_hour, pnl_open, alert and the
        ``strata`` columns). The strata default to ``STRATA_PRIMARY`` (Ruling 68: market, node, settle_hour, wclass,
        month), so the frame needs ``wclass`` and ``month``; ``STRATA_DECILE`` (market, node, cal_decile) is the
        sensitivity. Raises ValueError for an empty frame, a missing column, or a NaN in any column it reads, and for an
        ``alert`` with a value other than 0 and 1."""
        _check_rows(rows, strata)
        r = rows.reset_index(drop=True)
        days = np.array(sorted(r["settle_date"].unique()))
        pos = {d: i for i, d in enumerate(days)}
        occ = r.groupby(["market", "node", "settle_date", "settle_hour"], observed=True).cumcount()
        cell = _codes(r.assign(_occ=occ), ["market", "node", "settle_hour", "_occ"])
        return cls.from_arrays(r["settle_date"].map(pos).to_numpy(), len(days), cell,
                               _codes(r, strata), r["pnl_open"].to_numpy(float),
                               r["alert"].to_numpy(float), days, _codes(r, ["market", "node"]))

    def _day_weights(self, w):
        """The day weights as floats, one per day (1 each for None). Any other length is an error: a longer vector used
        to pass silently, with only its first n_days entries read."""
        if w is None:
            return np.ones(self.n_days)
        w = np.asarray(w, dtype=float)
        if w.shape != (self.n_days,):
            raise ValueError(f"RowCube: day weights have shape {w.shape}, expected ({self.n_days},), one per day")
        return w

    def _row_weights(self, w):
        return np.ones(len(self.day_idx)) if w is None else self._day_weights(w)[self.day_idx]

    def _excess(self, values, w=None, alert=None):
        """(sum_s [C_s - A_s / N_s * V_s], sum_s V_s): the alerted ``values`` beyond within-stratum random selection
        at the same counts, and the total of ``values``."""
        a = self.alert if alert is None else alert
        wr = self._row_weights(w)
        V = np.bincount(self.sid, wr * values, self.n_strata)
        A = np.bincount(self.sid, wr * a, self.n_strata)
        N = np.bincount(self.sid, wr, self.n_strata)
        C = np.bincount(self.sid, wr * a * values, self.n_strata)
        with np.errstate(invalid="ignore", divide="ignore"):
            expected = np.where(N > 0, A / N * V, 0.0)
        return float((C - expected).sum()), float(V.sum())

    def selc_parts(self, w=None, alert=None):
        return self._excess(self.loss, w, alert)

    def excess_var(self, alert=None):
        """(excess, variance) of an alert path, the observed one by default; unweighted, as the rotation test is.

        excess = sum_s [C_s - A_s L_s / N_s], the same number as ``selc_parts(alert=alert)[0]``. variance is that of
        the captured loss C_s when A_s of the N_s rows of each stratum are chosen at random (the hypergeometric
        reference SELC is built on), summed over the independent strata: sum_s A_s (N_s - A_s) / (N_s (N_s - 1)) *
        (Q_s - L_s^2 / N_s) over strata with N_s >= 2. Both use the path's own counts A_s. Raises ValueError if the
        per-stratum statistics are missing or not finite (a NaN count would otherwise read as an empty stratum and
        leave a finite, wrong statistic), and if the alert path holds a value other than 0 and 1 (NaN and inf
        included), whether it is passed in or is the cube's own."""
        if self.strat_n is None or self.strat_loss is None or self.strat_ss is None:
            raise ValueError("RowCube.excess_var: the per-stratum statistics are missing; build the cube with "
                             "from_arrays or from_rows")
        if not (np.isfinite(self.strat_n).all() and np.isfinite(self.strat_loss).all()
                and np.isfinite(self.strat_ss).all()):
            raise ValueError("RowCube.excess_var: the per-stratum statistics are not finite")
        a = self.alert if alert is None else np.asarray(alert, dtype=float)
        _require_binary(a, "RowCube.excess_var")
        return self._excess_var(a)

    def _excess_var(self, a):
        """``excess_var`` of an alert path already known to hold only 0 and 1, on a cube whose statistics were checked:
        what the rotation loop calls for each shift, the rotation of a checked path being one too."""
        A = np.bincount(self.sid, a, self.n_strata)
        C = np.bincount(self.sid, a * self.loss, self.n_strata)
        N, L = self.strat_n, self.strat_loss
        with np.errstate(invalid="ignore", divide="ignore"):
            expected = np.where(N > 0, A / N * L, 0.0)
        pairs = N >= 2
        variance = np.where(pairs, A * (N - A) / np.where(pairs, N * (N - 1.0), 1.0) * self.strat_ss, 0.0)
        return float((C - expected).sum()), float(variance.sum())

    def snet_parts(self, w=None, alert=None):
        """(excess NET, node-days): node-days, SNET's divisor, sum over nodes the (weighted) days each node has."""
        excess, _ = self._excess(self.loss - self.profit, w, alert)
        return excess, float(self._day_weights(w)[self.nd_day].sum())

    def snet(self, w=None, alert=None):
        excess, node_days = self.snet_parts(w, alert)
        if not (math.isfinite(excess) and math.isfinite(node_days)):
            raise ValueError(f"RowCube.snet: the excess ({excess}) or the node-days ({node_days}) is not finite")
        return excess / node_days if node_days else 0.0

    def day_weight(self, w=None):
        """D in J_lambda = NET + lambda * D * ESred: the number of days, or the sum of the day weights."""
        return float(self.n_days if w is None else self._day_weights(w).sum())

    def j_lambda(self, lam, w=None, alert=None):
        return self.net(w, alert) + lam * self.day_weight(w) * self.esred(w, alert=alert)

    def selc(self, w=None, alert=None):
        excess, total = self.selc_parts(w, alert)
        if not (math.isfinite(excess) and math.isfinite(total)):
            raise ValueError(f"RowCube.selc: the excess ({excess}) or the total loss ({total}) is not finite")
        return excess / total if total > 0 else 0.0

    def net(self, w=None, alert=None):
        a = self.alert if alert is None else alert
        return float((self._row_weights(w) * a * (self.loss - self.profit)).sum())

    @staticmethod
    def _es(daily, day_w, frac):
        """Expected shortfall of the daily P&L: the mean of the k worst days, as a positive loss, in the frame where
        day d is repeated day_w[d] times. k = max(1, ceil(frac * total weight)), as in ``expected_shortfall``."""
        order = np.argsort(daily, kind="stable")
        vals, wts = daily[order], day_w[order].astype(float)
        k = max(1, math.ceil(frac * wts.sum()))
        before = np.concatenate([[0.0], np.cumsum(wts)[:-1]])
        take = np.minimum(wts, np.maximum(0.0, k - before))
        return float(-(vals * take).sum() / k)

    def esred(self, w=None, frac=0.05, alert=None):
        a = self.alert if alert is None else alert
        day_w = self._day_weights(w)
        without = np.bincount(self.day_idx, self.pnl, self.n_days)
        with_ = np.bincount(self.day_idx, self.pnl * (1 - a), self.n_days)
        return self._es(without, day_w, frac) - self._es(with_, day_w, frac)

    def rotated_alerts(self, shift):
        """The alerts moved ``shift`` days later within each cell, circularly. An alert whose target (day, cell) has no
        row is lost, so the alert count is kept exactly only on a complete day x cell grid."""
        dense = np.zeros((self.n_days, self.n_cells))
        dense[self.day_idx, self.cell_idx] = self.alert
        return np.roll(dense, int(shift), axis=0)[self.day_idx, self.cell_idx]


def _rotation_paths(cube, min_shift):
    """One window under rotation. Returns the shifts min_shift..n_days-min_shift, then over the window's shift set
    {0} u those shifts (index 0 is the observed path) the excess and the variance of the alert path, and over the
    non-zero shifts the share of the observed alerts that survive the rotation (NaN if there are none). The observed
    path and the cube's statistics are checked once, by ``excess_var``; a rotation of a path of 0s and 1s holds only
    0s and 1s, so the shifts go through ``_excess_var``."""
    shifts = np.arange(min_shift, cube.n_days - min_shift + 1)
    excess, variance, kept = np.empty(len(shifts) + 1), np.empty(len(shifts) + 1), np.empty(len(shifts))
    excess[0], variance[0] = cube.excess_var()
    for i, shift in enumerate(shifts):
        rotated = cube.rotated_alerts(shift)
        excess[i + 1], variance[i + 1] = cube._excess_var(rotated)
        kept[i] = rotated.sum()
    observed = float(cube.alert.sum())
    return shifts, excess, variance, (kept / observed if observed > 0 else np.full(len(shifts), np.nan))


def _check_min_shift(min_shift, where):
    """``min_shift`` as a Python int; ValueError unless it is an integer of at least 1. A bool is refused (True is an
    int that nobody means as one day), and so is a float, 2.0 included; numpy integers are fine. A min_shift of 0
    would count the observed path again as the shifts 0 and n_days."""
    if isinstance(min_shift, bool) or not isinstance(min_shift, numbers.Integral) or min_shift < 1:
        raise ValueError(f"{where}: min_shift must be an integer of at least 1, got {min_shift!r}")
    return int(min_shift)


def rotation_excess(cube, min_shift=1):
    """(shifts, SELC excess of the window with its alerts rotated by each shift), for the non-zero shifts
    min_shift..n_days-min_shift; the unstudentized excess, in loss units. The default min_shift of 1 gives every
    non-zero shift of the full cyclic group (Ruling 69). Raises ValueError for a min_shift that is not an integer of
    at least 1 (a bool included)."""
    min_shift = _check_min_shift(min_shift, "rotation_excess")
    shifts, excess, _, _ = _rotation_paths(cube, min_shift)
    return shifts, excess[1:]


def rotation_test(cubes, min_shift=1):
    """H-E, as amended by Rulings 55 and 69: a randomization test on the full grid, with a studentized statistic.

    Each window (cube) is rotated independently. Its shift set is {0} u {min_shift, ..., n_days - min_shift}, where 0
    is the observed state: by default (min_shift 1) the full cyclic group {0, 1, ..., n_days - 1}. The reference set is
    the grid of the windows' shift sets; its cell (0, ..., 0) is the observed one. A cell's statistic is
    T = sum_w excess_w / sqrt(sum_w var_w), each window's alert path (observed or rotated) with its own excess and
    variance (``RowCube.excess_var``); T = 0 where the variance is 0 (the excess is then 0 too: no stratum has a choice
    to make). Then p = (1 + #{other cells with T >= T_obs}) / (number of cells) and n_null = prod(n_w + 1) - 1,
    n_w = n_days - 2 * min_shift + 1; by default n_w + 1 = n_days, so n_null = prod(n_days) - 1 and a single window's p
    is k / n_days, k = 1, ..., n_days. A window without alerts adds nothing to T, so pooling with one reproduces the
    single-window p and T.

    Validity. p is a randomization p-value under exchangeability of each window's observed alert path with its
    rotations, when the shift set is the full cyclic group (the default). A restricted set (min_shift > 1) is not a
    group and carries no such claim: it over-rejects under day-to-day persistence (Ruling 69).

    Returns t_obs (T of the observed state), p, n_null, selc_obs (pooled SELC, sum of excess / sum of loss: the
    reported estimand, not the test statistic) and retained_share, one {"min", "mean"} per window: the share of its
    observed alerts that survive a rotation, over its non-zero shifts (NaN for a window without alerts).

    Raises ValueError for a min_shift that is not an integer of at least 1 (a bool included), for no window, for a
    window of fewer than 2 * min_shift days (its null would be empty), and unless every T is finite. With no loss at
    all SELC is 0.0 by convention: T is 0.0 and p is 1.0."""
    min_shift = _check_min_shift(min_shift, "rotation_test")
    cubes = list(cubes)
    if not cubes:
        raise ValueError("rotation_test: needs at least one window")
    short = [c.n_days for c in cubes if c.n_days < 2 * min_shift]
    if short:
        raise ValueError(f"rotation_test: window(s) of {', '.join(map(str, short))} days leave no admissible rotation: "
                         f"with min_shift {min_shift} a window needs at least {2 * min_shift} days, "
                         "or its null is empty")
    excess, variance, shares = np.zeros(1), np.zeros(1), []
    for cube in cubes:
        _, e, v, share = _rotation_paths(cube, min_shift)
        excess = (excess[:, None] + e[None, :]).ravel()          # the first cell stays the observed one
        variance = (variance[:, None] + v[None, :]).ravel()
        shares.append({"min": float(np.min(share)), "mean": float(np.mean(share))})
    unusable = ~(np.isfinite(excess) & np.isfinite(variance))     # first: below, NaN > 0 is False and T would be 0
    if unusable.any():
        raise ValueError(f"rotation_test: the excess or its variance is not finite in {int(unusable.sum())} of "
                         f"{excess.size} grid cells (NaN or inf in the alerts or the losses)")
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(variance > 0, excess / np.sqrt(variance), 0.0)
    total = sum(c.selc_parts()[1] for c in cubes)
    selc_obs = sum(c.selc_parts()[0] for c in cubes) / total if total > 0 else 0.0
    return {"t_obs": float(t[0]), "p": float((1 + np.sum(t[1:] >= t[0])) / t.size), "n_null": int(t.size - 1),
            "selc_obs": float(selc_obs), "retained_share": shares}


def _per_node(scores, split, days):
    pos = {d: i for i, d in enumerate(days)}
    sub = scores[scores["split"] == split]
    return {k: (g["settle_date"].map(pos).to_numpy(), g["score"].to_numpy(float))
            for k, g in sub.groupby(["market", "node"], observed=True)}


def _log_error(cal, dep, wc, wd, alpha):
    ci, cs = cal
    di, ds = dep
    sample = np.repeat(cs, wc[ci].astype(int))
    wdep = wd[di].astype(float)
    n = wdep.sum()
    if sample.size == 0 or n == 0:
        return np.nan
    tau = np.quantile(sample, 1 - alpha)
    r = float((wdep * (ds >= tau)).sum() / n) / alpha
    return abs(math.log10(max(r, 1.0 / (2.0 * n * alpha))))


_SCORE_COLS = ("market", "node", "split", "settle_date", "score")


def _check_scores(frames):
    """Like ``_check_rows``: an empty frame, a missing column or a NaN would otherwise be skipped silently (groupby
    drops NaN keys, a NaN score makes tau NaN so that its node never alerts) or end in an unrelated error."""
    for f in frames:
        if len(f) == 0:
            raise ValueError("calibration-and-deployment bootstrap: a score frame is empty")
        missing = [c for c in _SCORE_COLS if c not in f.columns]
        if missing:
            raise ValueError(f"calibration-and-deployment bootstrap: missing column(s) {missing}")
        nan = [c for c in _SCORE_COLS if f[c].isna().any()]
        if nan:
            raise ValueError(f"calibration-and-deployment bootstrap: NaN in {nan}; nothing is dropped silently")


def _month_day(day):
    """(month, day of month) of a settlement date: a ``datetime.date``, a Timestamp or a numpy datetime64."""
    stamp = pd.Timestamp(day)
    return stamp.month, stamp.day


def _aligned_counts(cal_days, dep_days, mean_block, n_rep, seed):
    """(calibration counts, deployment counts) of ``n_rep`` calendar-aligned resamples (Ruling 118), of shapes
    (n_rep, len(cal_days)) and (n_rep, len(dep_days)).

    The calendar index is the sorted union of the (month, day) of the two windows' days. ONE
    ``stationary_counts(len(index), mean_block, n_rep, seed)`` is drawn over it and each day takes the count of its
    (month, day) position, so a position that both windows have carries the same count in both. A leap day that only one
    window has, and a day that a window lacks, need no special case: the position is in the index, and the window without
    such a day just does not read its count. The window totals therefore vary by replicate (they stay at the window's length
    only when both windows hold the same positions). A window that holds one (month, day) twice, as a span of more than a
    year does, gives both days the same count. The index runs from 1 January, so a window that straddles New Year is cut
    there; the quarters of a calendar year do not straddle it."""
    cal_md, dep_md = [_month_day(d) for d in cal_days], [_month_day(d) for d in dep_days]
    at = {md: i for i, md in enumerate(sorted(set(cal_md) | set(dep_md)))}
    counts = stationary_counts(len(at), mean_block, n_rep, seed)
    return counts[:, [at[md] for md in cal_md]], counts[:, [at[md] for md in dep_md]]


def _cal_dep_bootstrap(frames, combine, alpha, n_rep, mean_block, seed, aggregate=np.nanmedian, aligned=False):
    """Resample the calibration days (seed) and the deployment days (seed + 1) by independent circular stationary
    draws, the same days for every node, and re-place tau per node (P1: the 1 - alpha quantile of its resampled
    calibration scores). Per node, ``combine`` reduces the frames' |log10 R| (floored like ``policies.activity``);
    ``aggregate`` reduces the nodes. Returns (value on the real days, value per replicate, number of nodes).
    Raises ValueError for a frame that fails ``_check_scores`` and when no node has calibration and deployment rows in
    every frame: the statistic would be NaN, and a NaN gives the smallest possible p-value, not no p-value.

    ``aligned=False`` is the registered scheme above, the one H-B uses; its output is what it was before Ruling 118.
    ``aligned=True`` (Ruling 118, pre-registration review M2) draws once over the calendar instead (``_aligned_counts``): the
    calibration and deployment windows are the same calendar span in different years, so a (month, day) has a day in both
    (bar a leap day and the days a window lacks), and two independent draws break that matching in every replicate. For a
    score with a calendar component the mismatch alone moves tau against the deployment rate and inflates |log10 R|, so the
    interval measures how far re-randomizing the calendar moves R, not the sampling error of the realized quarter. The
    aligned draw gives a calibration day and the deployment day on the same (month, day) the same count. ``t_hat`` is on the
    real days, all weights 1, in both."""
    _check_scores(frames)

    def days_of(split):
        return np.array(sorted(set().union(*(set(f.loc[f.split == split, "settle_date"]) for f in frames))))
    cal_days, dep_days = days_of("calib"), days_of("deploy")
    cal = [_per_node(f, "calib", cal_days) for f in frames]
    dep = [_per_node(f, "deploy", dep_days) for f in frames]
    nodes = sorted(set.intersection(*[set(x) for x in cal + dep]))
    if not nodes:
        raise ValueError("calibration-and-deployment bootstrap: no node has calibration and deployment rows in every "
                         "frame, so there is nothing to resample")

    def stat(wc, wd):
        return float(aggregate([combine([_log_error(c[k], d[k], wc, wd, alpha) for c, d in zip(cal, dep)])
                                for k in nodes]))

    t_hat = stat(np.ones(len(cal_days)), np.ones(len(dep_days)))
    if aligned:
        cc, cd = _aligned_counts(cal_days, dep_days, mean_block, n_rep, seed)
    else:
        cc = stationary_counts(len(cal_days), mean_block, n_rep, seed)
        cd = stationary_counts(len(dep_days), mean_block, n_rep, seed + 1)
    t_star = np.array([stat(cc[b], cd[b]) for b in range(n_rep)])
    return t_hat, t_star, len(nodes)


def block_contrast_bootstrap(scores_alt, scores_base, alpha=0.05, n_rep=2000, mean_block=7, seed=20261022):
    """H-B1 and H-B2: median over nodes of |log10 R|(alt) - |log10 R|(base) under P1, with its one-sided basic-bootstrap
    p-value ``p``, the percentile interval ``ci`` and the basic interval ``ci_basic`` that goes with that p-value
    (report it next to p). Only the nodes present in both frames, in both splits, are used (``n_nodes``)."""
    t_hat, t_star, n_nodes = _cal_dep_bootstrap([scores_alt, scores_base], lambda e: e[0] - e[1],
                                                alpha, n_rep, mean_block, seed)
    return {"t_hat": t_hat, "t_star": t_star, "p": basic_p_one_sided(t_hat, t_star),
            "ci": percentile_ci(t_star), "ci_basic": basic_ci(t_hat, t_star), "n_nodes": n_nodes}


LOG2 = math.log10(2.0)


def _failure_count(values):
    """Nodes outside the band, summed over the nodes that could be evaluated; NaN if none could (np.nansum says 0.0)."""
    values = np.asarray(values, dtype=float)
    return np.nan if np.isnan(values).all() else float(np.nansum(values))


def block_level_bootstrap(scores, alpha=0.05, n_rep=2000, mean_block=7, seed=20261022, measure="median", aligned=False):
    """E5 and H-S: one block under P1, resampled like block_contrast_bootstrap. measure="median" gives the
    median over nodes of |log10 R|; measure="failures" counts nodes outside [0.5, 2] (|log10 R| > log10 2,
    which matches activity()'s fail_0.5_2 whenever a node has more than 20 deployment hours at alpha 5%). A replicate
    in which no node can be evaluated (every node lost all its calibration or all its deployment days) is NaN for both
    measures, never 0.0; ``ci`` and ``ci_basic`` are then NaN too.

    ``aligned`` (Ruling 118) chooses the draw. False, the default, is the registered scheme, independent draws of the
    calibration days (seed) and the deployment days (seed + 1), and the result is exactly what it was before the option
    existed. True draws once over the (month, day) positions of the two windows, so that a replicate keeps their matching
    (see ``_cal_dep_bootstrap``), and adds ``"scheme": "aligned"`` to the result. The default result carries no such key, so
    that nothing about it changes; a caller that stores both intervals labels the independent one itself.
    ``block_contrast_bootstrap`` (H-B) has no such option and stays on the registered scheme."""
    if measure == "median":
        combine, aggregate = (lambda e: e[0]), np.nanmedian
    elif measure == "failures":
        combine, aggregate = (lambda e: np.nan if np.isnan(e[0]) else float(e[0] > LOG2)), _failure_count
    else:
        raise ValueError(f"unknown measure {measure!r}")
    t_hat, t_star, n_nodes = _cal_dep_bootstrap([scores], combine, alpha, n_rep, mean_block, seed, aggregate,
                                                aligned=aligned)
    out = {"t_hat": t_hat, "t_star": t_star, "ci": percentile_ci(t_star), "ci_basic": basic_ci(t_hat, t_star),
           "n_nodes": n_nodes}
    if aligned:
        out["scheme"] = "aligned"
    return out


def basic_p_one_sided(t_hat, t_star):
    t_star = np.asarray(t_star, dtype=float)
    if np.isnan(t_hat) or np.isnan(t_star).any():
        raise ValueError("basic_p_one_sided: t_hat or a resample is NaN, so there is no p-value (NaN fails every "
                         "comparison, so the count would be 0 and p the smallest possible, 1 / (B + 1))")
    return float((1 + np.sum(t_star - t_hat >= t_hat)) / (len(t_star) + 1))


def percentile_ci(t_star, level=0.95):
    lo, hi = np.quantile(np.asarray(t_star, dtype=float), [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def basic_ci(t_hat, t_star, level=0.95):
    """The basic (pivotal) bootstrap interval (2 t_hat - q_hi, 2 t_hat - q_lo), q the percentile bounds of t_star: the
    interval that goes with ``basic_p_one_sided``, which reflects the percentile interval about the estimate."""
    lo, hi = percentile_ci(t_star, level)
    return float(2 * t_hat - hi), float(2 * t_hat - lo)
