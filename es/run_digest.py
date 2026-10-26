"""Collect the development outputs into one Markdown digest, ``outputs/es/DIGEST.md``, for the author and for the
pre-registration methodology gate (G1).

A header block states the data cut-off, the git short SHA of dart_dev HEAD (if git can say), the run time of every
output (its file mtime, UTC) and that no sealed data has been read. Then, one section per group of outputs:

- (a) the published numbers that the regenerated pipeline does not reproduce (``legacy/comparison.csv``) and the list of
  corrections in ``legacy/CHANGES.md``;
- (b) the analogues: eta, the registered family, score persistence, the rule constants and seeds, the H-E placebo
  pairings and the activity of every block and policy;
- (c) the development economics of the primary pipeline: SELC, NET, ES reduction and their intervals, the decomposition,
  the comparators, the spike-exposure split, the rotation test;
- (d) size and power of H-E and H-B, each power checked against the floor, and the family decision;
- (e) the label-free monitors; (f) the 2x2 label control; (g) the CAISO settlement note and the calibrated fees;
- (h) the inference outputs: BH families, leave-one-operator-out, the first lines of the mixed model.

The digest reads output files and two code constants (``es.config.SEAL_LAST_DEV_DATE`` and
``es.positions.CALIBRATED_FEES``). It reads no price, no score cache, and imports no module that could (a test checks
the imports). A file that does not exist prints ``_missing: <name>_`` in its place and one that cannot be parsed prints
``_unreadable: <name>: <reason>_``. A table that cannot be built from a file prints an ``**ERROR: ...**`` line and the
rest of the digest is still written; ``main`` then says so on standard output. A run that has not finished (the economics
``run_meta.json`` status, ``protocol_inputs.json`` before stage ``sim``) is flagged at the top of its part. The text depends
only on the files read (and on git HEAD), never on the time it is made, so the digest of unchanged outputs is
byte-identical.

Run: DART_PROJECT_ROOT=... /opt/anaconda3/bin/python3 -m es.run_digest"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEV_ROOT, OUT_DIR, SEAL_LAST_DEV_DATE
from .positions import CALIBRATED_FEES

#: A table longer than this is cut and says so (the longest expected one, the monitor evaluation, has about 250 rows).
MAX_ROWS = 600
#: A monitor row whose calibration window is longer than this many days gets the note (Ruling 83: only ranks across
#: nodes are interpreted, not the level of the domain AUC).
LONG_CALIBRATION_DAYS = 100
LONG_CALIBRATION_NOTE = "longer calibration window: only ranks interpreted"
#: Lines of ``inference/mixed_model.txt`` that are shown.
MIXED_MODEL_LINES = 40
#: The power floor, unless ``protocol_inputs.json`` records its own (``rule.min_power``).
POWER_FLOOR = 0.5
FEE_DOC = "docs/fee_rederivation_2026-10-23.md"
SEALED_SENTENCE = "No sealed data has been read."
ERROR_MARK = "**ERROR:"

#: Every output the digest reads, relative to ``OUT_DIR``; the header lists the mtime of each.
SOURCES = (
    "legacy/comparison.csv", "legacy/CHANGES.md",
    "analogs/protocol_inputs.json", "analogs/eta_selection.csv", "analogs/placebo_he.csv", "analogs/activity_all.csv",
    "analogs/simulation.csv",
    "economics/summary.csv", "economics/bootstrap_ci.csv", "economics/decomposition.csv", "economics/rotation.csv",
    "economics/run_meta.json", "economics/volmanagement_check.csv",
    "monitors/monitor_eval.csv",
    "labels_pr/label_control_2x2.csv", "labels_pr/label_control_pr.csv",
    "settlement_status.md", "settlement_status_addendum.md",
    "inference/bh.csv", "inference/looo.csv", "inference/mixed_model.txt",
)

#: Columns of the tables, in order; a column a file lacks is left out.
COMPARISON_COLS = ["item", "analysis", "block", "estimator", "metric", "value", "regenerated", "as_stated_value",
                   "stated_analysis", "abs_diff", "status", "source_line", "note"]
_ECON_NUMBERS = ["alert_rate", "SELC", "SELC_decile", "SNET", "NET", "ESred", "CAP", "excess",
                 "J_0.0", "J_0.25", "J_0.5", "J_1.0"]
POLICY_COLS = ["analysis", "policy", *_ECON_NUMBERS, "rotation_p"]
COMPARATOR_COLS = ["analysis", "comparator", *_ECON_NUMBERS]
SPLIT_COLS = ["analysis", "policy", "position", *_ECON_NUMBERS, "rotation_p"]
CI_COLS = ["analysis", "stat", "estimate", "lo", "hi", "approximate", "n_windows"]
PR_METRICS = ["precision_q", "recall_q", "precision_eval", "recall_eval", "prevalence_eval", "precision_lift"]

#: The two rejection-rate columns of ``simulation.csv`` (named as ``es.simulate`` names them: ``reject_{alpha:g}``) and
#: how the digest labels them.
A3_COL, P05_COL = f"reject_{0.05 / 3:g}", f"reject_{0.05:g}"
RATE_LABELS = {A3_COL: "rate at alpha/3 (0.0167)", P05_COL: "rate at 0.05"}
E_SIZE = ["H-E single A, size", "H-E pooled A+B, size"]
E_POWER = ["H-E pooled A+B, power curve", "H-E pooled A+B, power"]
B_POWER = ["H-B1 power", "H-B2 power"]
#: (hypothesis, the simulation test whose power it is).
POWER_TESTS = (("H-E", "H-E pooled A+B, power"), ("H-B1", "H-B1 power"), ("H-B2", "H-B2 power"))
E_SIZE_COLS = ["test", "scenario", "persistence", "strata", "n_sim", A3_COL, P05_COL]
NODRIFT_COLS = ["nodrift_fail_q95_0.5_2", "nodrift_fail_q95_0.67_1.5", "nodrift_fail_q95_0.8_1.25"]
B_SIZE_COLS = ["law", "persistence", "scenario", "n_sim", A3_COL, P05_COL, *NODRIFT_COLS]
CURVE_COLS = ["test", "rho", "mean_selc", "target_selc", "n_sim", A3_COL, P05_COL, "note"]
B_POWER_COLS = ["test", "law", "target_log_error_gap", "calibrated_delta", "mean_t_hat", "n_sim", A3_COL, P05_COL,
                "note", "error"]


# ---------------------------------------------------------------------------------------------------------------
# Cells and tables
# ---------------------------------------------------------------------------------------------------------------

def _esc(text) -> str:
    """``text`` safe inside a Markdown table cell: a pipe is escaped and a line break becomes a space."""
    return str(text).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _fmt(value) -> str:
    """One cell: 4 significant digits, no decimals from 1000 up (so no exponent for a large number), nothing for a
    missing value, integers and booleans as they are, text escaped. The CSVs hold the exact numbers."""
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number != number:
            return ""
        if number in (float("inf"), float("-inf")):
            return "inf" if number > 0 else "-inf"
        return f"{number:.0f}" if abs(number) >= 1000 else f"{number:.4g}"
    return _esc(value)


def _md(frame: pd.DataFrame) -> str:
    """``frame`` as a Markdown pipe table (``DataFrame.to_markdown``): every cell formatted by ``_fmt``, numeric columns
    right-aligned. At most ``MAX_ROWS`` rows, with a line saying so when it cuts. A frame without rows is a note."""
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        return "_(no rows)_"
    shown = frame.head(MAX_ROWS)
    cells = shown.astype(object).apply(lambda column: column.map(_fmt))
    cells.columns = [_esc(c) for c in shown.columns]
    align = ["right" if pd.api.types.is_numeric_dtype(shown.iloc[:, i]) and not pd.api.types.is_bool_dtype(shown.iloc[:, i])
             else "left" for i in range(shown.shape[1])]
    text = cells.to_markdown(index=False, disable_numparse=True, colalign=align)
    if len(frame) > MAX_ROWS:
        text += f"\n\n_(first {MAX_ROWS} of {len(frame)} rows; the rest are in the CSV)_"
    return text


def _cols(frame: pd.DataFrame, names) -> pd.DataFrame:
    return frame[[c for c in names if c in frame.columns]]


def _join(*blocks: str) -> str:
    """Markdown blocks separated by one blank line (a block's own leading and trailing newlines do not add to it)."""
    return "\n\n".join(block.strip("\n") for block in blocks if block)


def _kv(mapping: dict) -> str:
    """A bullet per key: strings as they are, everything else as compact JSON, in backticks."""
    def show(value):
        return value if isinstance(value, str) else json.dumps(value, default=str)
    return "\n".join(f"- {key}: `{show(value)}`" for key, value in mapping.items())


def _demote(text: str, levels: int = 2) -> str:
    """Markdown ``text`` with every heading ``levels`` deeper (at most level 6), so that an embedded note nests under
    the section that holds it. Lines inside a code fence are left alone."""
    out, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced:
            m = re.match(r"(#{1,6})(\s.*)$", line)
            if m:
                line = "#" * min(len(m.group(1)) + levels, 6) + m.group(2)
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------------
# Files that may be missing or unreadable
# ---------------------------------------------------------------------------------------------------------------

def _load(path: Path, reader):
    """(content, None), or (None, the line that stands in for the content): ``_missing: <name>_`` for a file that does
    not exist, ``_unreadable: <name>: <reason>_`` for one that ``reader`` cannot parse."""
    try:
        if not path.exists():
            return None, f"_missing: {path.name}_"
        return reader(path), None
    except (OSError, ValueError, KeyError) as exc:             # ValueError: pandas' parse errors and JSONDecodeError
        reason = (str(exc).splitlines() or [""])[0]
        return None, f"_unreadable: {path.name}: {type(exc).__name__}: {reason}_"


def _read_json_object(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
    return obj


def _csv(path: Path):
    return _load(path, pd.read_csv)


def _text(path: Path):
    return _load(path, lambda p: p.read_text(encoding="utf-8"))


def _json(path: Path):
    return _load(path, _read_json_object)


def _from_csv(path: Path, build) -> str:
    frame, note = _csv(path)
    return note if frame is None else build(frame)


def _from_text(path: Path, build) -> str:
    text, note = _text(path)
    return note if text is None else build(text)


def _from_json_object(path: Path, build) -> str:
    obj, note = _json(path)
    return note if obj is None else build(obj)


def _error(label: str, exc: Exception) -> str:
    reason = (str(exc).splitlines() or [""])[0]
    return (f"{ERROR_MARK} {label} could not be built ({type(exc).__name__}: {reason}). "
            "The input does not look like what es/run_digest.py expects.**")


def _guarded(label: str, thunk) -> str:
    """``thunk()``, or an ERROR line when it raises: one unexpected file must not stop the whole digest."""
    try:
        return thunk()
    except Exception as exc:                                    # noqa: BLE001 - see the module docstring
        return _error(label, exc)


def _part(label: str, heading: str, thunk) -> str:
    """``### heading`` and what ``thunk`` returns (an ERROR line, naming the section and the heading, if it raises)."""
    return _join(f"### {heading}", _guarded(f"{label} {heading}", thunk))


# ---------------------------------------------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------------------------------------------

def _git_short_sha(repo) -> str | None:
    """The short SHA of ``repo``'s HEAD, or None when git cannot say (no git, no repository, no commit)."""
    try:
        res = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             timeout=20, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout.strip() or None


def _mtime(path: Path) -> str:
    """The file's modification time in UTC, ``missing`` if it does not exist, ``unreadable`` if the file system errs."""
    try:
        if not path.exists():
            return "missing"
        return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    except OSError:
        return "unreadable"


def _header(out: Path, dev_root: Path) -> list[str]:
    sha = _git_short_sha(dev_root)
    stamps = pd.DataFrame({"output": list(SOURCES),
                           "modified (UTC)": [_mtime(out / s) for s in SOURCES]})
    facts = [f"- Data cut-off: {SEAL_LAST_DEV_DATE} (the last settlement date development code may read; "
             "`es.config.SEAL_LAST_DEV_DATE`).",
             f"- {SEALED_SENTENCE}",
             f"- dart_dev HEAD: `{sha}`" if sha else "- dart_dev HEAD: not available (no git, no repository or no commit).",
             "- Run timestamps: the modification time of each output file the digest reads, in UTC (`missing`: not "
             "produced yet)."]
    return [f"# Development digest (data up to {SEAL_LAST_DEV_DATE} only)", "\n".join(facts), _md(stamps)]


# ---------------------------------------------------------------------------------------------------------------
# (a) regenerated vs published
# ---------------------------------------------------------------------------------------------------------------

def _comparison_block(cmp: pd.DataFrame) -> str:
    matched = cmp["match"].astype(str).str.strip().str.lower().eq("true")
    bad = cmp[~matched]
    counts = cmp.groupby("status", sort=False).size().rename("rows").reset_index()
    if bad.empty:
        return _join(_md(counts), f"All {len(cmp)} published numbers reproduce as stated.")
    return _join(_md(counts), f"{len(bad)} of {len(cmp)} published numbers do not reproduce as stated (`match == False`):",
                 _md(_cols(bad, COMPARISON_COLS)))


def _changes_block(text: str) -> str:
    items = re.findall(r"^### (D\d+\..*\S)[ \t]*$", text, flags=re.M)
    lines = "\n".join(f"- {item}" for item in items) if items else "_(no `### D<n>.` heading found in the file)_"
    return _join("The cause and the corrected value of every such row are in `legacy/CHANGES.md`:", lines)


def _section_a(out: Path) -> list[str]:
    return [_part("(a)", "Published numbers: regenerated vs published",
                  lambda: _from_csv(out / "legacy" / "comparison.csv", _comparison_block)),
            _part("(a)", "Corrections listed in legacy/CHANGES.md",
                  lambda: _from_text(out / "legacy" / "CHANGES.md", _changes_block))]


# ---------------------------------------------------------------------------------------------------------------
# (b) analogues
# ---------------------------------------------------------------------------------------------------------------

def _protocol_facts(meta: dict) -> str:
    facts = {"stage": meta.get("stage"), "eta": meta.get("eta")}
    facts.update({k: meta[k] for k in ("rho_star", "analog_selc", "analog_selc_pooled") if k in meta})
    warning = ""
    if meta.get("stage") != "complete":
        warning = (f"**WARNING: protocol_inputs.json is at stage `{meta.get('stage')}`: the registered family has not "
                   "been computed. Run `python -m es.run_analogs --stage sim`.**")
    return _join(_kv(facts), warning)


def _reasons(entry: dict) -> list[str]:
    reasons = entry.get("reasons") or []
    return [str(reasons)] if isinstance(reasons, str) else [str(r) for r in reasons]


def _verdict(entry: dict) -> str:
    """The registered decision of one hypothesis: only a recorded True or False is a decision."""
    confirmatory = entry.get("confirmatory")
    if confirmatory is True:
        return "confirmatory"
    if confirmatory is False:
        return "replaced by its estimand"
    return "unknown (no `confirmatory` entry)"


def _family_block(fam: dict) -> str:
    rows = []
    for hid, f in fam.items():
        if hid == "notes" or not isinstance(f, dict):
            continue
        rows.append({"hypothesis": hid, "decision": _verdict(f),
                     "approximate": f.get("approximate"), "interim_approximate": f.get("interim_approximate"),
                     "power": f.get("power"), "reasons": "; ".join(_reasons(f)) or "none"})
    notes = fam.get("notes")
    return _join(_md(pd.DataFrame(rows)), "\n".join(f"- {note}" for note in notes) if isinstance(notes, list) else "")


def _phi_block(phi: dict) -> str:
    rows = [{"block": block, **values} for block, values in phi.items() if isinstance(values, dict)]
    return _join("`phi_score_median`: the persistence of the block's score (AR(1)), median over nodes, as the simulation "
                 "draws it; `phi_score_raw_median`: before the correction for a daily component.",
                 _md(pd.DataFrame(rows)))


def _rule_block(rule) -> str:
    if isinstance(rule, dict):
        return _kv(rule)
    return "_(no `rule` block in protocol_inputs.json: stage `sim` writes it.)_"


def _eta_block(tab: pd.DataFrame) -> str:
    wide = tab.pivot(index="analysis", columns="eta", values="median_log_error")
    wide = wide.reindex(tab["analysis"].drop_duplicates())
    wide.loc["mean of AQ1-AQ4 (AQ3F excluded)"] = wide.drop(index="AQ3F", errors="ignore").mean()
    return _join("Median |log10 R| of the self-correcting policy P5 on each analogue, by eta. eta is the one with the "
                 "least mean over the four quarters (AQ3F is not pooled; a tie goes to the smaller eta).",
                 _md(wide.reset_index().rename_axis(columns=None)))


def _placebo_block(plc: pd.DataFrame) -> str:
    caption = ""
    if "counted" in plc.columns:
        caption = ("`counted` marks the pairings the registration rule reads (Ruling 94: only the 1-year pairings "
                   "count); the others are reported only.")
    return _join(caption, _md(plc))


def _activity_block(act: pd.DataFrame) -> str:
    xgb = act[act["estimator"] == "xgb"]
    rows = []
    for (analysis, block), g in xgb.groupby(["analysis", "block"], sort=False):
        row = {"analysis": analysis, "block": block}
        for policy, gp in g.groupby("policy", sort=False):
            row[policy] = f"{int(gp['fail_0.5_2'].sum())}/{len(gp)} ({gp['log_error'].median():.2f})"
        rows.append(row)
    return _join("Each cell: nodes with R outside the band [0.5, 2] / nodes (median |log10 R| over nodes), by analogue, "
                 "block and policy; the logit estimator's rows are in `analogs/activity_all.csv`.",
                 _md(pd.DataFrame(rows)))


def _section_b(out: Path) -> list[str]:
    label = "(b)"
    meta, note = _json(out / "analogs" / "protocol_inputs.json")
    parts = []
    if meta is None:
        parts.append(note)
    else:
        parts.append(_part(label, "Protocol inputs", lambda: _protocol_facts(meta)))
        if isinstance(meta.get("family"), dict):
            parts.append(_part(label, "Registered family (computed by `registration_family`, never decided by hand)",
                               lambda: _family_block(meta["family"])))
        if isinstance(meta.get("phi"), dict) and meta["phi"]:
            parts.append(_part(label, "Score persistence (phi) by block", lambda: _phi_block(meta["phi"])))
        parts.append(_part(label, "Rule constants and seeds", lambda: _rule_block(meta.get("rule"))))
    parts += [
        _part(label, "Eta selection", lambda: _from_csv(out / "analogs" / "eta_selection.csv", _eta_block)),
        _part(label, "H-E placebo pairings", lambda: _from_csv(out / "analogs" / "placebo_he.csv", _placebo_block)),
        _part(label, "Activity of the analogues (estimator xgb)",
              lambda: _from_csv(out / "analogs" / "activity_all.csv", _activity_block))]
    return parts


# ---------------------------------------------------------------------------------------------------------------
# (c) economics
# ---------------------------------------------------------------------------------------------------------------

def _run_block(meta: dict) -> str:
    warning = ""
    status = meta.get("status")
    if status != "complete":
        done, total = len(meta.get("analyses_done") or []), len(meta.get("analyses") or [])
        warning = (f"**WARNING: the economics run is `{status}`: {done} of {total} analyses are done, so the tables "
                   "below can be partial.**")
        if status == "failed":
            warning += f" **It failed on `{meta.get('failed_on')}`: {meta.get('error')}**"
    return _join(_kv(meta), warning)


def _at_premium_calibrated(summ: pd.DataFrame) -> pd.DataFrame:
    return summ[(summ["position"] == "premium") & (summ["fees"] == "calibrated")]


def _policies_block(summ: pd.DataFrame) -> str:
    rows = _at_premium_calibrated(summ)
    rows = rows[rows["policy"] != "comparator"]
    return _join("P1: fixed calibration-quantile cutoff. P4: rolling cutoff. `P4@r*`: P4 with its rate matched to P1's "
                 "alert counts. P5: self-correcting cutoff. P6: exact-rate cutoff (the deployment scores' own quantile). "
                 "`rotation_p` is the H-E rotation-test p-value of the P1 row.", _md(_cols(rows, POLICY_COLS)))


def _comparators_block(summ: pd.DataFrame) -> str:
    rows = _at_premium_calibrated(summ)
    return _join("`calendar@count`: after-the-fact top-k of the calendar-only (G6) score, k = P1's alert count per node. "
                 "`VOL_P1`, `VOL_P4`, `VOL_P6`: the `abs_mean_168h` score under the P1, P4 and P6 cutoffs; `VOL@count`: "
                 "its top-k with P1's counts. `curve_P1_a<alpha>`, `curve_P6_a<alpha>`: P1 and P6 on the model score at "
                 "cutoff alpha 0.01, 0.025, 0.05 and 0.10.",
                 _md(_cols(rows[rows["policy"] == "comparator"], COMPARATOR_COLS)))


def _split_block(summ: pd.DataFrame) -> str:
    rows = summ[summ["position"].astype(str).str.startswith("premium[") & (summ["fees"] == "calibrated")]
    return _join("The premium position split by its spike exposure (`es.positions.spike_exposure`, learnt on the training "
                 "rows: short_spikes are steady gains with rare large losses, long_spikes the rest), calibrated fees.",
                 _md(_cols(rows, SPLIT_COLS)))


def _decomposition_block(dec: pd.DataFrame) -> str:
    wide = dec.set_index("analysis").T
    wide.index.name = "measure_part"
    return _join("E1: `timing = P1 - P4@r*`, `level = P4@r* - P4`, `total = P1 - P4` (`es.measures.decomposition`).",
                 _md(wide.reset_index().rename_axis(columns=None)))


def _ci_block(ci: pd.DataFrame, e1: bool) -> str:
    is_e1 = ci["stat"].astype(str).str.startswith("E1:")
    rows = ci[is_e1] if e1 else ci[~is_e1]
    return _join("Day-bootstrap percentile intervals. `|ex-CAISO` repeats an analysis without CAISO; "
                 "`E4:dSELC|<comparator>` is SELC(P1) - SELC(comparator), paired over the same resampled days "
                 "(Ruling 81); `E4:ESred|<comparator>` is the comparator's own ESred and `E4:dESred|<comparator>` is "
                 "ESred(P1) - ESred(comparator), paired in the same way (Ruling 118); `approximate` flags the ES-based "
                 "statistics (ESred, and J_lambda for lambda > 0), whose interval the producer calls approximate.",
                 _md(_cols(rows, CI_COLS)))


def _rotation_block(rot: pd.DataFrame) -> str:
    return _join("`t_obs` is the studentized T of the observed state (unitless: it is not SELC), `selc_obs` the pooled "
                 "SELC, `p` the randomization p-value over the full cyclic group and `n_null` the size of the reference "
                 "set; `retained_*` is the share of alerts that survive a rotation.", _md(rot))


def _volmanagement_block(vm: pd.DataFrame) -> str:
    return _join("Mean / SD of the position's P&L over the P1-alerted rows and over the others (not annualised).", _md(vm))


def _section_c(out: Path) -> list[str]:
    label, econ = "(c)", out / "economics"
    return [
        _part(label, "Run", lambda: _from_json_object(econ / "run_meta.json", _run_block)),
        _part(label, "Premium position, calibrated fees: the policies",
              lambda: _from_csv(econ / "summary.csv", _policies_block)),
        _part(label, "Comparators (premium position, calibrated fees)",
              lambda: _from_csv(econ / "summary.csv", _comparators_block)),
        _part(label, "Spike-exposure split", lambda: _from_csv(econ / "summary.csv", _split_block)),
        _part(label, "Decomposition (E1)", lambda: _from_csv(econ / "decomposition.csv", _decomposition_block)),
        _part(label, "Bootstrap intervals: SELC, NET, E2-E4",
              lambda: _from_csv(econ / "bootstrap_ci.csv", lambda ci: _ci_block(ci, e1=False))),
        _part(label, "Bootstrap intervals: E1 decomposition",
              lambda: _from_csv(econ / "bootstrap_ci.csv", lambda ci: _ci_block(ci, e1=True))),
        _part(label, "Rotation test (H-E)", lambda: _from_csv(econ / "rotation.csv", _rotation_block)),
        _part(label, "Volatility-management check",
              lambda: _from_csv(econ / "volmanagement_check.csv", _volmanagement_block))]


# ---------------------------------------------------------------------------------------------------------------
# (d) size and power
# ---------------------------------------------------------------------------------------------------------------

def _sim_rows(sim: pd.DataFrame, tests, columns, strata: str | None = None) -> pd.DataFrame:
    """The rows of ``simulation.csv`` for ``tests`` (under ``strata``, if given), with the columns they have, the rates
    labelled and the columns that are empty in these rows left out."""
    rows = sim[sim["test"].isin(tests)]
    if strata is not None:
        rows = rows[rows["strata"] == strata]
    return _cols(rows, columns).dropna(axis=1, how="all").rename(columns=RATE_LABELS)


def _e_size_block(sim: pd.DataFrame, strata: str) -> str:
    if strata == "primary":
        caption = ("No timing skill (rho = 0), so the rate is the size, to be read against 0.0167 and 0.05. `single A` is "
                   "one quarter, `pooled A+B` two quarters pooled; persistence `fit` or `upper` end; scenario "
                   "homoskedastic (pooled P&L SD) or heteroskedastic (SD fitted by hour and weekday class).")
    else:
        caption = ("The same rows under the calendar-propensity decile strata (Ruling 68): reported, never governing "
                   "the rule.")
    return _join(caption, _md(_sim_rows(sim, E_SIZE, E_SIZE_COLS, strata)))


def _hb_size_block(sim: pd.DataFrame) -> str:
    table = _sim_rows(sim, ["H-B size"], B_SIZE_COLS)
    caption = ""
    if any(c in table.columns for c in NODRIFT_COLS):
        caption = ("`nodrift_fail_q95_<band>`: the 95th percentile over simulations of the number of nodes outside the "
                   "band (base block, no drift): the simulated no-drift context for the failure counts in (b). It is "
                   "defined for the no-drift rows only.")
    return _join(caption, _md(table))


def _power_block(sim: pd.DataFrame, floor: float) -> str:
    """The registered power of each hypothesis (its rate at alpha/3) against ``floor``. A power that is not a number is
    counted as below the floor, as ``registration_family`` counts it."""
    rows, below = [], []
    for hid, test in POWER_TESTS:
        found = sim[sim["test"] == test]
        if found.empty:
            below.append(f"{hid} (no row)")
            rows.append({"hypothesis": hid, "test": test, "floor check": "no row in simulation.csv"})
        for _, r in found.iterrows():
            a3 = r.get(A3_COL)
            unavailable = a3 is None or pd.isna(a3)
            is_below = unavailable or float(a3) < floor
            if is_below:
                below.append(f"{hid} (unavailable)" if unavailable else f"{hid} ({float(a3):.2f})")
            flag = "ok" if not is_below else f"below {floor:g}" + (" (unavailable)" if unavailable else "")
            rows.append({"hypothesis": hid, "test": test, "n_sim": r.get("n_sim"), RATE_LABELS[A3_COL]: a3,
                         RATE_LABELS[P05_COL]: r.get(P05_COL), "floor check": flag})
    verdict = (f"**Below the floor of {floor:g}: {', '.join(below)}.**" if below
               else f"Every power is at or above the floor of {floor:g}.")
    return _join(f"Power is the rejection rate at alpha/3 of the simulated alternative; the floor is {floor:g}. The rule "
                 "(`es.run_analogs.registration_family`) replaces an H-B test below the floor by its estimand; H-E's "
                 "power is reported and not read by the rule.", verdict, _md(pd.DataFrame(rows)))


def _decision_block(meta, note) -> str:
    if meta is None:
        return note
    fam = meta.get("family")
    if not isinstance(fam, dict):
        return (f"_Not computed: protocol_inputs.json is at stage `{meta.get('stage')}` "
                "(run `python -m es.run_analogs --stage sim`)._")
    lines = []
    for hid, f in fam.items():
        if hid == "notes" or not isinstance(f, dict):
            continue
        reasons = "; ".join(_reasons(f))
        lines.append(f"- **{hid}**: {_verdict(f)}" + (f" ({reasons})" if reasons else ""))
    return "\n".join(lines)


def _section_d(out: Path) -> list[str]:
    label = "(d)"
    sim, sim_note = _csv(out / "analogs" / "simulation.csv")
    meta, meta_note = _json(out / "analogs" / "protocol_inputs.json")
    rule = meta.get("rule") if meta else None
    floor = rule.get("min_power", POWER_FLOOR) if isinstance(rule, dict) else POWER_FLOOR
    parts = []
    if sim is None:
        parts.append(sim_note)
    else:
        parts += [
            _part(label, "Power per hypothesis", lambda: _power_block(sim, floor)),
            _part(label, "H-E size, primary strata", lambda: _e_size_block(sim, "primary")),
            _part(label, "H-E size, decile strata (sensitivity, not used by the rule)",
                  lambda: _e_size_block(sim, "decile")),
            _part(label, "H-B size by law, persistence and scenario", lambda: _hb_size_block(sim)),
            _part(label, "H-E power curve", lambda: _md(_sim_rows(sim, E_POWER, CURVE_COLS))),
            _part(label, "H-B power", lambda: _md(_sim_rows(sim, B_POWER, B_POWER_COLS)))]
    parts.append(_part(label, "Family decision (from protocol_inputs.json)", lambda: _decision_block(meta, meta_note)))
    return parts


# ---------------------------------------------------------------------------------------------------------------
# (e) monitors, (f) label control
# ---------------------------------------------------------------------------------------------------------------

def _monitor_block(mon: pd.DataFrame) -> str:
    frame = mon.copy()
    if "calib_days" in frame.columns:
        days = pd.to_numeric(frame["calib_days"], errors="coerce")
        frame["note"] = np.where(days > LONG_CALIBRATION_DAYS, LONG_CALIBRATION_NOTE, "")
    return _join("Spearman of each monitor with |log10 R| across nodes (one-sided p) and AUROC for node failure. "
                 "`calib_days` is the calibration window in days (descriptive). `placebo_median` and `placebo_n` are the "
                 "descriptive no-drift reference (Ruling 83): the median over nodes of the same statistic on a split "
                 f"inside the calibration window. Rows with calib_days > {LONG_CALIBRATION_DAYS} are noted.", _md(frame))


def _by_block(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows grouped by block (blocks in the order of the file), analyses sorted within a block, so that the four cells
    of the 2x2 control sit next to each other."""
    order = {b: i for i, b in enumerate(frame["block"].drop_duplicates())}
    return frame.assign(_b=frame["block"].map(order)).sort_values(["_b", "analysis"], kind="stable").drop(columns="_b")


def _control_block(ctl: pd.DataFrame) -> str:
    return _join("Nodes outside the band [0.5, 2] of n_nodes (P1, alpha 0.05, XGBoost seed 42). Analyses are "
                 "`LC-{full|h1}-{q2023|q2324}`: calibration on the full year or on H1, label cutoff q from 2023 or from "
                 "2023-24; the four cells of a block are adjacent.", _md(_by_block(ctl)))


def _pr_block(pr: pd.DataFrame) -> str:
    g = pr.groupby(["analysis", "block"], sort=False)
    out = g[[c for c in PR_METRICS if c in pr.columns]].median()
    out.insert(0, "nodes", g.size())
    if "n_nan_outcome" in pr.columns:
        out["n_nan_outcome"] = g["n_nan_outcome"].sum()
    return _join("Per analysis and block: the median over nodes of each metric (`n_nan_outcome` is summed).",
                 _md(_by_block(out.reset_index())))


def _section_e(out: Path) -> list[str]:
    return [_from_csv(out / "monitors" / "monitor_eval.csv", _monitor_block)]


def _section_f(out: Path) -> list[str]:
    label, folder = "(f)", out / "labels_pr"
    return [_part(label, "Failures and median R", lambda: _from_csv(folder / "label_control_2x2.csv", _control_block)),
            _part(label, "Precision and recall (median over nodes)",
                  lambda: _from_csv(folder / "label_control_pr.csv", _pr_block))]


# ---------------------------------------------------------------------------------------------------------------
# (g) settlement and fees, (h) inference
# ---------------------------------------------------------------------------------------------------------------

def _fees_block() -> str:
    rows = [{"market": market, "INC (d=+1)": f"{sides[1]:.2f}", "DEC (d=-1)": f"{sides[-1]:.2f}"}
            for market, sides in CALIBRATED_FEES.items()]
    return _join("Side-specific fees in $/MWh at the calibrated setting (`es.positions.CALIBRATED_FEES`; derivation and "
                 f"sources in `{FEE_DOC}`).", _md(pd.DataFrame(rows)))


def _section_g(out: Path) -> list[str]:
    parts = [_guarded("(g) settlement_status.md", lambda: _from_text(out / "settlement_status.md", _demote))]
    addendum, note = _text(out / "settlement_status_addendum.md")
    if addendum is not None:
        parts.append(_guarded("(g) settlement_status_addendum.md", lambda: _demote(addendum)))
    elif not note.startswith("_missing"):    # optional: no marker when it is absent, one when it cannot be read
        parts.append(note)
    parts.append(_part("(g)", "Calibrated fees", _fees_block))
    return parts


#: Ruling 118 (G1, M5): the audit-robustness BH family is descriptive. Its p-values come from the calibration-and-deployment
#: bootstrap that the size study found oversized, so its flags are diagnostics and, as in the paper, no count of its rejections is
#: printed. es.paper_registry.DESCRIPTIVE_BH_FAMILIES holds the same tuple (a test compares them; this module imports nothing heavy).
DESCRIPTIVE_BH_FAMILIES = ("audit robustness",)
DESCRIPTIVE_TEXT = "n/a (descriptive)"
DESCRIPTIVE_NOTE = ("The audit-robustness family is descriptive: its p-values come from the calibration-and-deployment bootstrap whose "
                    "simulated size exceeds nominal, so its flags are diagnostics and no count of its rejections is reported.")


def _bh_summary(bh: pd.DataFrame) -> str:
    g = bh.groupby("family", sort=False)
    summary = pd.DataFrame({"tests": g.size(), "included": g["included"].sum(),
                            "BH rejections": g["bh_reject"].sum()}).reset_index()
    summary["BH rejections"] = [DESCRIPTIVE_TEXT if fam in DESCRIPTIVE_BH_FAMILIES else n
                                for fam, n in zip(summary["family"], summary["BH rejections"])]
    return _join("BH within each family (q = 0.05, the `es.secondary.bh_table` default). `included` counts the tests with "
                 "a p-value: a NaN is no test that ran, so it is left out of the adjustment and never rejected.",
                 DESCRIPTIVE_NOTE, _md(summary))


def _bh_tests(bh: pd.DataFrame) -> str:
    return _join("`bh_reject` is the Benjamini-Hochberg decision of each test. For the audit-robustness family it is a diagnostic: "
                 "that family is descriptive.", _md(bh))


def _mixed_block(text: str) -> str:
    lines = text.splitlines()
    body = "```\n" + "\n".join(lines[:MIXED_MODEL_LINES]) + "\n```"
    note = ""
    if len(lines) > MIXED_MODEL_LINES:
        note = f"_(first {MIXED_MODEL_LINES} of {len(lines)} lines; the full file is inference/mixed_model.txt)_"
    return _join(body, note)


def _section_h(out: Path) -> list[str]:
    label, folder = "(h)", out / "inference"
    return [_part(label, "Benjamini-Hochberg families", lambda: _from_csv(folder / "bh.csv", _bh_summary)),
            _part(label, "Tests", lambda: _from_csv(folder / "bh.csv", _bh_tests)),
            _part(label, "Leave-one-operator-out", lambda: _from_csv(folder / "looo.csv", _md)),
            _part(label, f"Mixed model (first {MIXED_MODEL_LINES} lines)",
                  lambda: _from_text(folder / "mixed_model.txt", _mixed_block))]


# ---------------------------------------------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------------------------------------------

SECTIONS = (
    ("(a) Regenerated vs published", _section_a),
    ("(b) Analogues: eta, registered family, placebo pairings, activity", _section_b),
    ("(c) Economics, primary pipeline", _section_c),
    ("(d) Size and power", _section_d),
    ("(e) Monitors", _section_e),
    ("(f) Label 2x2 control", _section_f),
    ("(g) CAISO settlement and calibrated fees", _section_g),
    ("(h) Inference: BH families, leave-one-operator-out, mixed model", _section_h),
)


def build_digest(out_dir=None, dev_root=None) -> str:
    """The digest of the outputs under ``out_dir`` (default ``OUT_DIR``) as Markdown text. Reads files only; writes
    nothing. ``dev_root`` is where git is asked for HEAD (default ``DEV_ROOT``)."""
    out = Path(OUT_DIR if out_dir is None else out_dir)
    dev = Path(DEV_ROOT if dev_root is None else dev_root)
    blocks = _header(out, dev)
    for title, build in SECTIONS:
        blocks.append(f"## {title}")
        blocks.append(_guarded(title.split()[0], lambda: _join(*build(out))))
    return _join(*blocks) + "\n"


def main():
    text = build_digest()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "DIGEST.md"
    tmp = path.with_name(path.name + ".tmp")
    try:                                         # a reader or a crash never finds half a digest
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    for label in re.findall(re.escape(ERROR_MARK) + r" (.+?) could not be built", text):
        print(f"WARNING: {label} could not be built; see its ERROR line in the digest")
    print(path)
    return path


if __name__ == "__main__":
    main()
