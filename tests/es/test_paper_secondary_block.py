"""The macros of the exploratory secondary economic analysis (controller Ruling 148): the development economic analysis with the plain-DART-lag
block as the alert model, written by es.run_economics_block into outputs/es/economics_block/DART3/ and read by es.paper_registry."""
import re

import pandas as pd
import pytest

from es import paper_registry as reg
from es.config import OUT_DIR, REVISION_ROOT

MACROS = REVISION_ROOT / "paper" / "generated" / "macros.tex"
FOLDER = OUT_DIR / "economics_block" / "DART3"
needs_outputs = pytest.mark.skipif(not (MACROS.is_file() and (FOLDER / "summary.csv").is_file()),
                                   reason="the built paper (generated/macros.tex) and the outputs of the secondary analysis are not in this tree")


def _entries():
    return reg._econ_block_entries()


def test_the_secondary_entries_are_named_apart_from_every_primary_entry():
    names = [e.name for e in _entries()]
    assert len(names) == len(set(names))
    assert all("SecDart" in n for n in names)
    primary = {e.name for e in reg._econ_entries()}
    assert not set(names) & primary                                                   # no name of the primary analysis is touched
    assert not [n for n in primary if "SecDart" in n]                                 # and the primary analysis does not use the infix


def test_the_secondary_entries_read_the_folder_of_the_block_and_nothing_else():
    entries = _entries()
    assert {e.source.split("/")[0] + "/" + e.source.split("/")[1] for e in entries} == {"economics_block/DART3"}
    assert {e.source.split("/")[-1] for e in entries} == {"bootstrap_ci.csv", "summary.csv", "rotation.csv"}
    assert {e.stage for e in entries} == {"dev"}
    stats = set(re.findall(r'stat == "([^"]+)"', " ".join(e.query for e in entries)))
    assert stats == set(reg.SEC_DART_STATS)


def test_the_secondary_entries_cover_the_four_quarters_and_the_pooled_rehearsal_with_an_interval_each():
    entries = _entries()
    per_stat = len(reg.SEC_DART_STATS)
    # four quarters: the stats with their intervals, the alert rate and the rotation p; the pooled rehearsal: the stats with their intervals and the rotation p
    assert len(entries) == len(reg.QUARTER_ANALOGUES) * (2 * per_stat + 2) + 2 * per_stat + 1
    names = {e.name for e in entries}
    for stat in reg.SEC_DART_STATS:
        for aw in [reg.ANALYSIS_WORD[a] for a in reg.QUARTER_ANALOGUES] + ["PooledRehearsal"]:
            assert f"{reg.STAT_WORD[stat]}SecDart{aw}" in names and f"{reg.STAT_WORD[stat]}SecDart{aw}CI" in names
    assert {"alertRatePOneSecDartAqThree", "alertRatePOneSecDartAqFour", "alertRatePOneSecDartAqOne", "alertRatePOneSecDartAqTwo",
            "rotPSecDartPooledRehearsal"} <= names


def test_every_secondary_stat_has_a_word_and_a_format():
    for stat in reg.SEC_DART_STATS:
        assert stat in reg.STAT_WORD and stat in reg.STAT_FMT


def test_the_entries_are_part_of_the_registry_once():
    all_names = [e.name for e in reg.ENTRIES]
    assert len(all_names) == len(set(all_names))
    assert {e.name for e in _entries()} <= set(all_names)


@needs_outputs
def test_the_built_macros_hold_every_secondary_name_with_a_value():
    text = MACROS.read_text(encoding="utf-8")
    defined = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", text, flags=re.M))
    for e in _entries():
        assert e.name in defined, e.name
        assert "PENDING" not in defined[e.name] and "pending" not in defined[e.name], (e.name, defined[e.name])


@needs_outputs
def test_the_alert_rate_macros_are_the_rates_of_the_summary_file():
    text = MACROS.read_text(encoding="utf-8")
    defined = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", text, flags=re.M))
    s = pd.read_csv(FOLDER / "summary.csv")
    for aid in reg.QUARTER_ANALOGUES:
        row = s[(s.analysis == aid) & (s.policy == "P1") & (s.position == "premium") & (s.fees == "calibrated") & (s.comparator == "none")]
        assert len(row) == 1, aid
        assert defined[f"alertRatePOneSecDart{reg.ANALYSIS_WORD[aid]}"] == f"{100 * float(row['alert_rate'].iloc[0]):.1f}\\%", aid


@needs_outputs
def test_the_paired_contrast_macros_are_the_estimates_of_the_bootstrap_file():
    text = MACROS.read_text(encoding="utf-8")
    defined = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", text, flags=re.M))
    b = pd.read_csv(FOLDER / "bootstrap_ci.csv")
    row = b[(b.analysis == "AQ4") & (b.stat == "E4:dESred|calendar@count")]
    assert len(row) == 1
    est = float(row["estimate"].iloc[0])
    printed = defined["desredCalCountSecDartAqFour"].replace("\\textminus{}", "-").replace("\\$", "").replace("{,}", "")
    assert abs(float(printed) - round(est)) < 1e-9, (printed, est)
