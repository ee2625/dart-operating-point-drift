"""Descriptive mixed model (spec v3 §5.1; review M6), leave-one-operator-out, and
multiplicity helpers (Holm for the primary family, BH within exploratory families).
``main`` also writes ``bh.csv``, a development rehearsal on AQ3 of the three BH families of ``es.secondary``.
Run: DART_PROJECT_ROOT=... /opt/anaconda3/bin/python3 -m es.run_inference"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .analyses import ANALOGS
from .config import OUT_DIR, ensure_dirs
from .data import build_panel, load_dev_raw
from .scores import cached_scores

#: Columns the BH rehearsal reads from the other development outputs (the AQ3 filter, then es.secondary's families).
_ECON_COLUMNS = ("analysis", "policy", "comparator", "position", "fees", "SELC", "rotation_p")
_MONITOR_COLUMNS = ("analysis", "block", "monitor", "spearman", "spearman_p_one_sided")
#: The label columns of economics/summary.csv. They are read verbatim: "None", "NA" or "null" in one of them is a label,
#: never a missing value (pandas would turn it into NaN), and a number is missing only where its cell is empty.
_ECON_TEXT = ("policy", "comparator", "position", "fees")
#: What the audit's bootstrap reads of a score frame (es.bootstrap checks for these five columns).
_SCORE_COLUMNS = ["market", "node", "split", "settle_date", "score"]
#: Relative slack of the threshold comparisons in ``bh`` and ``holm``. ``p * m / i`` and ``alpha / (m - rank)`` are
#: floating-point quotients, and a decimal p-value that sits exactly on its threshold (0.05 at rank 3 of 3 for q = 0.05)
#: can land one ulp on the wrong side of it. 1e-12 is far below every gap between p-values and thresholds in the
#: registered grids, so it changes the decision only where exact arithmetic says "equal".
_RTOL = 1e-12


def _pvalues(pvals, who):
    """``pvals`` as a 1-D float array. A NaN, a value outside [0, 1] and anything that is not 1-D raise ValueError:
    a NaN sorts last and would shift every other test's threshold without being counted as a test."""
    p = np.asarray(pvals, dtype=float)
    if p.ndim != 1:
        raise ValueError(f"{who}: the p-values must be a 1-D sequence, got an array of shape {p.shape}")
    nan = np.flatnonzero(np.isnan(p))
    if nan.size:
        raise ValueError(f"{who}: NaN p-value at position(s) {nan.tolist()}; drop it before adjusting, "
                         "it is not a test that was run")
    bad = np.flatnonzero((p < 0) | (p > 1))
    if bad.size:
        raise ValueError(f"{who}: p-values must lie in [0, 1], got {p[bad].tolist()} at position(s) {bad.tolist()}")
    return p


def _level(value, name, who):
    level = float(value)
    if not 0.0 < level <= 1.0:                       # a NaN fails the comparison too
        raise ValueError(f"{who}: the level {name} must lie in (0, 1], got {value!r}")
    return level


def bh(pvals, q=0.05):
    """Benjamini-Hochberg step-up: reject the k smallest p-values, k the largest rank i with p_(i) <= i q / m.

    Returns a boolean array in the order of ``pvals`` (empty for an empty input). Tied p-values are always decided
    together. Raises ValueError for a NaN, a value outside [0, 1], input that is not 1-D, or a level outside (0, 1].
    The comparison has a relative slack of ``_RTOL`` (1e-12): a p-value exactly on its threshold is rejected."""
    p = _pvalues(pvals, "bh")
    q = _level(q, "q", "bh")
    order = np.argsort(p, kind="stable")
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    passed = np.zeros(len(p), dtype=bool)
    below = np.nonzero(ranked <= q * (1 + _RTOL))[0]
    if below.size:
        passed[order[: below.max() + 1]] = True
    return passed


def holm(pvals, alpha=0.05):
    """Holm step-down: reject p_(1), p_(2), ... in order while p_(i) <= alpha / (m - i + 1), and stop at the first miss.

    Returns a boolean array **in the order of ``pvals``** (the sealed entry point zips it with its hypothesis names;
    empty for an empty input). Tied p-values are always decided together. Raises ValueError for a NaN, a value
    outside [0, 1], input that is not 1-D, or a level outside (0, 1]. The comparison has a relative slack of
    ``_RTOL`` (1e-12): a p-value exactly on its threshold is rejected."""
    p = _pvalues(pvals, "holm")
    alpha = _level(alpha, "alpha", "holm")
    order = np.argsort(p, kind="stable")
    passed = np.zeros(len(p), dtype=bool)
    for rank, idx in enumerate(order):
        if p[idx] <= alpha / (len(p) - rank) * (1 + _RTOL):
            passed[idx] = True
        else:
            break
    return passed


def fit_mixed(cells):
    """REML fit of the descriptive mixed model of spec v3 §5.1,
    ``|log10 R| ~ family x analysis + (1|operator) + (1|operator:node) + (1|operator:analysis)``, on the P1 / xgb cells:
    ``log_error ~ C(block) * C(analysis)`` grouped by operator (market), with

    - ``(1|operator)``: a random intercept per operator, ``re_formula="~1"`` with ``groups="market"``, the column name,
      so that the summary prints it as "market Var" ("Group Var" if ``groups`` is an array);
    - ``(1|operator:node)``: the variance component ``node``, a random intercept per node within an operator;
    - ``(1|operator:analysis)``: the variance component ``operator_analysis``, a random effect per analysis within an
      operator.

    A ``re_formula`` is needed: with a ``vc_formula`` and none statsmodels adds no random intercept for the group
    (``model.k_re == 0``), which left ``(1|operator)`` out of the brief's call (controller Ruling 80). The column name
    as ``groups`` only relabels it: ``groups=cells["market"]`` with ``re_formula="1"`` is the same fit. With four
    operators the variance is weakly identified, so statsmodels may warn that the MLE is on the boundary; the summary
    prints ``Converged``. The standard errors are descriptive (review M6)."""
    model = smf.mixedlm("log_error ~ C(block) * C(analysis)", cells, groups="market", re_formula="~1",
                        vc_formula={"node": "0 + C(node)", "operator_analysis": "0 + C(analysis)"})
    return model.fit(reml=True)


def _aq3_records(path, required, text_columns=()):
    """The AQ3 rows of the development CSV at ``path`` as dicts. The caller checks that the file exists (a missing file
    means that the task that writes it has not run; here it raises FileNotFoundError). A file that lacks a column in
    ``required`` or has no AQ3 row raises ValueError: that is a wrong or stale file, not a missing one. Floats are
    parsed exactly (``float_precision="round_trip"``), so a p-value that was written is the p-value that is read.

    ``text_columns`` are read as text and only an empty cell is missing (``keep_default_na=False``); without them
    pandas' defaults apply."""
    options = {"float_precision": "round_trip"}
    if text_columns:
        options.update(dtype={c: str for c in text_columns}, keep_default_na=False, na_values=[""])
    frame = pd.read_csv(path, **options)
    absent = [c for c in required if c not in frame.columns]
    if absent:
        raise ValueError(f"{path}: no column(s) {absent} (it has {list(frame.columns)})")
    frame = frame[frame["analysis"] == "AQ3"]
    if frame.empty:
        raise ValueError(f"{path}: no AQ3 rows")
    return frame.to_dict(orient="records")


def _write_csv_atomically(frame, path):
    """``frame`` to ``path`` through a temporary file beside it and ``os.replace`` (as run_economics does), so that
    neither a reader nor a crash ever finds a half-written file; the temporary file is removed if the write fails."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_bh(inf):
    """``bh.csv``: the three BH families of ``es.secondary`` on AQ3, a development rehearsal of the sealed run's.

    The audit family is computed from the score cache (xgb seed 42 and the variants of ``AUDIT_VARIANTS``). The
    economic and monitor families are read from the outputs of Tasks 12 and 14; if one of those files does not exist,
    bh.csv holds the families that do and a line says which is missing. Both files are read, and checked, before the
    scores are asked for: a file that exists but gives no row for its family raises ValueError (for the economic family:
    no AQ3 row with policy "P1" and comparator "none").

    A ``bh.csv`` of an earlier run is removed first and the new one is written atomically, so a failed run never
    leaves a stale file beside the fresh ``mixed_model.txt`` and ``looo.csv``. The audit asks for 36 frames but 21
    distinct ones (DART3's 7 come up 18 times): each is requested once and kept in a slim copy with the five columns
    the bootstrap reads, since 21 full frames would not fit in memory (plan 2A's probe: 338 MB for 990,597 rows)."""
    (inf / "bh.csv").unlink(missing_ok=True)
    # imported here, not at module level: es.secondary imports bh from this module
    from .secondary import audit_family, bh_table, economic_family, monitor_family

    econ_path, monitor_path = OUT_DIR / "economics" / "summary.csv", OUT_DIR / "monitors" / "monitor_eval.csv"
    econ = _aq3_records(econ_path, _ECON_COLUMNS, _ECON_TEXT) if econ_path.exists() else None
    monitors = _aq3_records(monitor_path, _MONITOR_COLUMNS) if monitor_path.exists() else None
    for family, records, path, task in (("economic robustness", econ, econ_path, "Task 12"),
                                        ("drift monitors", monitors, monitor_path, "Task 14")):
        if records is None:
            print(f"missing: {family} ({path} not found; {task} has not run), so bh.csv does not hold it")
    econ_rows = economic_family(econ) if econ is not None else []
    if econ is not None and not econ_rows:
        seen = sorted({str(r["comparator"]) for r in econ if r["policy"] == "P1"})
        raise ValueError(f"{econ_path}: no AQ3 row has policy 'P1' and comparator 'none' "
                         f"(its P1 rows carry the comparators {seen})")
    monitor_rows = monitor_family(monitors) if monitors is not None else []
    panel = build_panel(load_dev_raw())
    frames = {}

    def score(kind, block, estimator, seed):
        key = (kind, block, estimator, seed)
        if key not in frames:
            full = cached_scores(panel, ANALOGS["AQ3F" if kind == "frozen" else "AQ3"], block, estimator, seed)
            frames[key] = full[_SCORE_COLUMNS].copy()
        return frames[key]

    rows = audit_family(score)
    table = bh_table(rows + econ_rows + monitor_rows)
    _write_csv_atomically(table, inf / "bh.csv")
    counts = [f"{family} {int(g['included'].sum())} tests, {int(g['bh_reject'].sum())} rejected"
              for family, g in table.groupby("family", sort=False)]
    print(f"wrote {inf / 'bh.csv'}: " + "; ".join(counts))


def main():
    ensure_dirs()
    acts = pd.read_csv(OUT_DIR / "analogs" / "activity_all.csv", float_precision="round_trip")
    cells = acts[(acts.policy == "P1") & (acts.estimator == "xgb")
                 & acts.block.isin(["G6", "LEV6", "DART3", "ROLL_DISP9", "DART18", "G1_24"])]
    res = fit_mixed(cells)
    inf = OUT_DIR / "inference"          # from OUT_DIR at call time, so that a test can point everything elsewhere
    inf.mkdir(parents=True, exist_ok=True)       # only now: a missing input or a failed fit leaves no empty folder
    (inf / "mixed_model.txt").write_text(str(res.summary()))
    looo = []
    for op in sorted(cells["market"].unique()):
        sub = cells[cells["market"] != op]
        g = sub.groupby(["analysis", "block"])
        looo.append(pd.DataFrame({"dropped": op, "failures": g["fail_0.5_2"].sum(),
                                  "median_log_error": g["log_error"].median()}).reset_index())
    pd.concat(looo).to_csv(inf / "looo.csv", index=False)
    _write_bh(inf)


if __name__ == "__main__":
    main()
