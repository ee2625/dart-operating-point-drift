"""Tests of es/run_economics_block.py (Task S1): the development economic analysis with another feature block as the alert
model, an exploratory secondary analysis (Ruling 148). Fast, synthetic, no real data and no network.

What is pinned:
- the score function: the combined block ("G1_24") becomes the chosen block, the calendar block ("G6") and every other
  request pass through untouched, and the arguments reach the underlying function as they came;
- the refusals of the command line, all before any data is loaded: an unknown block, an unknown or repeated analysis, a
  non-positive resample count, the primary output directory (also through an alias, a sub-folder or ``--overwrite``) and a
  directory that already holds a summary.csv;
- the hook on ``es.run_economics.economics_for``: installed for the run, restored after a normal run and after an exception;
- ``run_all`` end to end on synthetic score frames through the hook, against ``economics_for`` called directly with the
  block's frame in place of the combined one (the mechanism, not just its parts);
- the start-up message, the development-only loaders, and the provenance file the run leaves beside its CSVs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _module():
    import es.run_economics_block as module
    return module


def _economics_tests():
    """The helpers of test_run_economics (synthetic score frames, a stand-in for economics_for), imported as
    test_paper_g1 imports test_paper_tables."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_run_economics
    return test_run_economics


def _recorder():
    """A stand-in for cached_scores: records every call in full and returns a string naming the block it was asked for."""
    calls = []

    def base(panel, analysis, block, estimator="xgb", seed=42, *args, **kwargs):
        calls.append((panel, analysis, block, estimator, seed, args, kwargs))
        return f"frame:{block}"
    base.calls = calls
    return base


def _must_not_be_called(*args, **kwargs):
    raise AssertionError("the data loaders were called: the command line should have stopped before them")


# ---------------------------------------------------------------------------------------------------------------
# The score function
# ---------------------------------------------------------------------------------------------------------------

def test_the_score_function_maps_the_combined_block_to_the_chosen_block_and_passes_the_calendar_block_through():
    module = _module()
    base, panel, analysis = _recorder(), object(), object()
    score_fn = module.score_fn_for("DART3", base)
    assert score_fn(panel, analysis, "G1_24", "xgb", 42) == "frame:DART3"
    assert score_fn(panel, analysis, "G6", "xgb", 42) == "frame:G6"
    assert [c[2] for c in base.calls] == ["DART3", "G6"]
    for p, a, _, estimator, seed, args, kwargs in base.calls:          # nothing but the block is touched
        assert p is panel and a is analysis and (estimator, seed, args, kwargs) == ("xgb", 42, (), {})


def test_the_score_function_hands_the_estimator_and_seed_and_extra_arguments_on_unchanged():
    module = _module()
    base = _recorder()
    module.score_fn_for("MID3", base)("panel", "analysis", "G1_24", "logit", 7, 0.05, version="v9")
    assert base.calls == [("panel", "analysis", "MID3", "logit", 7, (0.05,), {"version": "v9"})]


def test_the_score_function_asks_for_the_combined_block_itself_when_that_is_the_chosen_block():
    """The regression run of the brief (block G1_24) must reproduce the primary analysis, so it maps G1_24 to G1_24."""
    module = _module()
    base = _recorder()
    score_fn = module.score_fn_for("G1_24", base)
    assert score_fn("p", "a", "G1_24", "xgb", 42) == "frame:G1_24" and score_fn("p", "a", "G6", "xgb", 42) == "frame:G6"


@pytest.mark.parametrize("requested", ["LEV6", "DART18", "DART3", "MID3", "g1_24", "G1_24 "])
def test_the_score_function_leaves_every_other_request_alone(requested):
    module = _module()
    base = _recorder()
    assert module.score_fn_for("ROLL15", base)("p", "a", requested, "xgb", 42) == f"frame:{requested}"


def test_the_score_function_reads_the_cached_scores_of_this_module_at_call_time_when_no_base_is_given(monkeypatch):
    import es.scores
    module = _module()
    assert module.cached_scores is es.scores.cached_scores        # the default of economics_for, unpatched
    base = _recorder()
    monkeypatch.setattr(module, "cached_scores", base)
    assert module.score_fn_for("DART3")("p", "a", "G1_24", "xgb", 42) == "frame:DART3"
    assert [c[2] for c in base.calls] == ["DART3"]


def test_the_combined_and_the_calendar_names_are_those_economics_for_asks_for():
    """economics_for asks its score_fn for these two literal names (es/run_economics.py: _scores_for)."""
    import inspect

    import es.run_economics as econ
    module = _module()
    source = inspect.getsource(econ._scores_for)
    assert f'score_fn(panel, analysis, "{module.PRIMARY_BLOCK}", "xgb", 42)' in source
    assert f'score_fn(panel, analysis, "{module.CALENDAR_BLOCK}", "xgb", 42)' in source
    assert (module.PRIMARY_BLOCK, module.CALENDAR_BLOCK) == ("G1_24", "G6")


# ---------------------------------------------------------------------------------------------------------------
# The block name, the analyses, the resample count
# ---------------------------------------------------------------------------------------------------------------

def test_every_key_of_blocks_is_accepted_as_the_alert_block():
    from es.blocks import BLOCKS
    module = _module()
    assert all(module.check_block(name) == name for name in BLOCKS)


@pytest.mark.parametrize("name", ["NOPE", "none", "None", "", "g1_24", "DART 3", None, 3])
def test_an_unknown_block_is_rejected_and_the_message_names_it_and_lists_the_valid_blocks(name):
    from es.blocks import BLOCKS
    module = _module()
    with pytest.raises(ValueError) as err:
        module.check_block(name)
    message = str(err.value)
    assert f"unknown block {name!r}" in message
    assert all(valid in message for valid in BLOCKS)


def test_main_rejects_an_unknown_block_before_anything_is_loaded(monkeypatch, capsys, tmp_path):
    module = _module()
    monkeypatch.setattr(module, "_load_inputs", _must_not_be_called)
    with pytest.raises(SystemExit) as exc:
        module.main(["--block", "NOPE", "--out-dir", str(tmp_path / "out")])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "unknown block 'NOPE'" in err and "DART3" in err
    assert not (tmp_path / "out").exists()


def test_main_needs_a_block(monkeypatch, capsys):
    module = _module()
    monkeypatch.setattr(module, "_load_inputs", _must_not_be_called)
    with pytest.raises(SystemExit) as exc:
        module.main([])
    assert exc.value.code == 2 and "--block" in capsys.readouterr().err


def test_the_default_analyses_are_the_four_development_quarters_in_the_order_of_the_brief():
    from es.analyses import ANALOGS
    module = _module()
    assert module.parse_analyses(None) == ["AQ3", "AQ4", "AQ1", "AQ2"] == list(module.DEFAULT_ANALYSES)
    assert "AQ3F" not in module.DEFAULT_ANALYSES and "AQ3F" in ANALOGS       # AQ3F is not a quarter of its own


def test_a_comma_list_of_analyses_keeps_its_order_and_tolerates_spaces():
    from es.analyses import ANALOGS
    module = _module()
    assert module.parse_analyses("AQ2, AQ1 ,AQ4") == ["AQ2", "AQ1", "AQ4"]
    assert all(module.parse_analyses(key) == [key] for key in ANALOGS)


@pytest.mark.parametrize("text", ["AQ9", "AQ3,AQ9", "aq3", "SA", "25H1-26H1", "AQ3,AQ3", "", "AQ3,,AQ4", " ", ","])
def test_unknown_empty_and_repeated_analyses_are_rejected(text):
    module = _module()
    with pytest.raises(ValueError, match="analys"):
        module.parse_analyses(text)


def test_the_message_for_an_unknown_analysis_names_it_and_lists_the_valid_keys():
    from es.analyses import ANALOGS
    module = _module()
    with pytest.raises(ValueError) as err:
        module.parse_analyses("AQ3,SA")
    assert "'SA'" in str(err.value) and all(key in str(err.value) for key in ANALOGS)


@pytest.mark.parametrize("argv", [["--analyses", "AQ9"], ["--analyses", "AQ3,AQ3"], ["--n-rep", "0"], ["--n-rep", "-5"],
                                  ["--n-rep", "many"]])
def test_main_rejects_bad_analyses_and_resample_counts_before_anything_is_loaded(argv, monkeypatch, capsys, tmp_path):
    module = _module()
    monkeypatch.setattr(module, "_load_inputs", _must_not_be_called)
    with pytest.raises(SystemExit) as exc:
        module.main(["--block", "DART3", "--out-dir", str(tmp_path / "out")] + argv)
    assert exc.value.code == 2 and capsys.readouterr().err.strip()
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------------------------------------------
# The output directory
# ---------------------------------------------------------------------------------------------------------------

@pytest.fixture
def layout(tmp_path, monkeypatch):
    """A temporary stand-in for outputs/es holding a primary economics folder with a summary.csv in it."""
    module = _module()
    out = tmp_path / "outputs" / "es"
    primary = out / "economics"
    primary.mkdir(parents=True)
    (primary / "summary.csv").write_text("analysis\nAQ4\n")
    monkeypatch.setattr(module, "OUT_DIR", out)
    return module, out, primary


def test_the_default_output_directory_is_a_new_folder_per_block_beside_the_primary_one(layout):
    module, out, primary = layout
    default = module.default_out_dir("DART3")
    assert default == out / "economics_block" / "DART3" and default != primary
    assert module.check_out_dir(default, overwrite=False) == default.resolve()
    assert not default.exists()                                   # checking creates nothing


def test_the_primary_output_directory_is_refused(layout):
    module, out, primary = layout
    with pytest.raises(ValueError, match="primary"):
        module.check_out_dir(primary, overwrite=False)


def test_the_real_primary_output_directory_is_refused():
    """Not the temporary stand-in: the folder the primary analysis wrote (nothing is created or touched by the check)."""
    from es.config import OUT_DIR
    module = _module()
    with pytest.raises(ValueError, match="primary"):
        module.check_out_dir(OUT_DIR / "economics", overwrite=False)
    with pytest.raises(ValueError, match="primary"):
        module.check_out_dir(OUT_DIR / "economics", overwrite=True)


def test_overwrite_never_opens_the_primary_output_directory(layout):
    module, out, primary = layout
    with pytest.raises(ValueError, match="primary"):
        module.check_out_dir(primary, overwrite=True)
    assert (primary / "summary.csv").read_text() == "analysis\nAQ4\n"


def test_the_primary_directory_is_refused_through_an_alias_or_from_inside(layout):
    module, out, primary = layout
    (primary / "inner").mkdir()
    (out / "alias").symlink_to(primary, target_is_directory=True)
    aliases = [out / "economics" / ".." / "economics",             # a path that goes round
               Path(str(primary) + "/"),                             # a trailing slash
               out / "alias",                                        # a symlink to it
               out / "alias" / "inner",                              # a folder inside it, through the symlink
               primary / "inner",                                    # a folder inside it
               primary / "not_yet_made" / "deeper"]                  # a folder that does not exist yet, inside it
    for alias in aliases:
        for overwrite in (False, True):
            with pytest.raises(ValueError, match="primary"):
                module.check_out_dir(alias, overwrite=overwrite)


def test_the_primary_directory_is_refused_by_its_path_even_when_it_does_not_exist_yet(tmp_path, monkeypatch):
    """No identity to compare then (a fresh checkout): the path alone has to do."""
    module = _module()
    out = tmp_path / "outputs" / "es"
    out.mkdir(parents=True)
    monkeypatch.setattr(module, "OUT_DIR", out)
    assert not (out / "economics").exists()
    for target in (out / "economics", out / "economics" / "block" / "DART3", out / "x" / ".." / "economics"):
        for overwrite in (False, True):
            with pytest.raises(ValueError, match="primary"):
                module.check_out_dir(target, overwrite=overwrite)
    assert module.check_out_dir(out / "economics_block" / "DART3") == (out / "economics_block" / "DART3").resolve()
    assert not (out / "economics").exists()                       # and the check creates nothing


def test_the_primary_directory_is_refused_under_another_spelling_of_its_name(layout):
    """macOS volumes are case-insensitive: the same folder under another case is the same folder."""
    module, out, primary = layout
    other_case = out / "ECONOMICS"
    if not other_case.exists():
        pytest.skip("this file system tells the two spellings apart")
    with pytest.raises(ValueError, match="primary"):
        module.check_out_dir(other_case, overwrite=True)


def test_a_directory_that_holds_a_summary_csv_is_refused_unless_overwrite_is_given(layout, tmp_path):
    module, out, primary = layout
    earlier = out / "economics_block" / "DART3"
    earlier.mkdir(parents=True)
    (earlier / "summary.csv").write_text("analysis\n")
    with pytest.raises(ValueError, match="summary.csv"):
        module.check_out_dir(earlier, overwrite=False)
    assert module.check_out_dir(earlier, overwrite=True) == earlier.resolve()


def test_a_folder_without_a_summary_csv_is_fine_even_if_it_holds_other_files(layout):
    module, out, primary = layout
    folder = out / "economics_block" / "DART3"
    folder.mkdir(parents=True)
    (folder / "run.log").write_text("log\n")
    (folder / "run_meta.json").write_text("{}\n")
    assert module.check_out_dir(folder, overwrite=False) == folder.resolve()


def test_a_regular_file_is_not_an_output_directory(layout, tmp_path):
    module, out, primary = layout
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("x")
    with pytest.raises(ValueError, match="not a directory"):
        module.check_out_dir(not_a_dir, overwrite=True)


def test_main_refuses_the_primary_directory_with_or_without_overwrite_and_touches_nothing(layout, monkeypatch, capsys):
    module, out, primary = layout
    monkeypatch.setattr(module, "_load_inputs", _must_not_be_called)
    before = sorted((p.name, p.read_bytes()) for p in primary.iterdir())
    for extra in ([], ["--overwrite"]):
        with pytest.raises(SystemExit) as exc:
            module.main(["--block", "DART3", "--out-dir", str(primary)] + extra)
        assert exc.value.code == 2 and "primary" in capsys.readouterr().err
    assert sorted((p.name, p.read_bytes()) for p in primary.iterdir()) == before


def test_main_refuses_an_existing_summary_csv_unless_overwrite_is_given(layout, monkeypatch, capsys):
    module, out, primary = layout
    monkeypatch.setattr(module, "_load_inputs", _must_not_be_called)
    earlier = out / "economics_block" / "DART3"
    earlier.mkdir(parents=True)
    (earlier / "summary.csv").write_text("keep me\n")
    with pytest.raises(SystemExit) as exc:
        module.main(["--block", "DART3", "--out-dir", str(earlier)])
    assert exc.value.code == 2 and "summary.csv" in capsys.readouterr().err
    assert (earlier / "summary.csv").read_text() == "keep me\n"


# ---------------------------------------------------------------------------------------------------------------
# The hook on es.run_economics.economics_for
# ---------------------------------------------------------------------------------------------------------------

def test_alert_block_installs_a_wrapper_for_the_run_and_restores_the_original_afterwards():
    import es.run_economics as econ
    module = _module()
    original = econ.economics_for
    with module.alert_block("DART3", _recorder()):
        assert econ.economics_for is not original
    assert econ.economics_for is original


def test_alert_block_restores_the_original_when_the_body_raises():
    import es.run_economics as econ
    module = _module()
    original = econ.economics_for
    with pytest.raises(RuntimeError, match="boom"):
        with module.alert_block("DART3", _recorder()):
            assert econ.economics_for is not original
            raise RuntimeError("boom")
    assert econ.economics_for is original


def test_the_wrapper_hands_the_mapped_score_function_to_economics_for_and_everything_else_on_unchanged(monkeypatch):
    import es.run_economics as econ
    module = _module()
    seen = []

    def economics_for(panel, analysis, fmm, nyi, eta, score_fn=None, keep_rows=False):
        seen.append((panel, analysis, fmm, nyi, eta, keep_rows, score_fn(panel, analysis, "G1_24", "xgb", 42),
                     score_fn(panel, analysis, "G6", "xgb", 42)))
        return "rows"
    monkeypatch.setattr(econ, "economics_for", economics_for)
    with module.alert_block("DART3", _recorder()):
        assert econ.economics_for("p", "a", "f", "n", 0.05, keep_rows=True) == "rows"      # as run_all calls it
        assert econ.economics_for("p", "a", "f", "n", 0.05) == "rows"
    assert seen == [("p", "a", "f", "n", 0.05, True, "frame:DART3", "frame:G6"),
                    ("p", "a", "f", "n", 0.05, False, "frame:DART3", "frame:G6")]


def test_a_score_function_given_by_the_caller_is_wrapped_not_replaced(monkeypatch):
    import es.run_economics as econ
    module = _module()
    monkeypatch.setattr(econ, "economics_for", lambda panel, analysis, fmm, nyi, eta, score_fn=None, keep_rows=False:
                        (score_fn("p", "a", "G1_24", "xgb", 42), score_fn("p", "a", "G6", "xgb", 42)))
    callers, never = _recorder(), _recorder()
    with module.alert_block("DART3", never):
        assert econ.economics_for("p", "a", "f", "n", 0.05, score_fn=callers) == ("frame:DART3", "frame:G6")
    assert [c[2] for c in callers.calls] == ["DART3", "G6"] and never.calls == []


# ---------------------------------------------------------------------------------------------------------------
# main(): the run, with the loaders and economics_for stubbed
# ---------------------------------------------------------------------------------------------------------------

@pytest.fixture
def stubbed(layout, monkeypatch):
    """main() with the development loaders stubbed (no real data), economics_for replaced by the fake of
    test_run_economics, and cached_scores replaced by a recorder. Returns (module, out, record, base)."""
    module, out, primary = layout
    import es.run_economics as econ
    tre = _economics_tests()
    record, base = [], _recorder()
    fake = tre._fake_economics_for(record)

    def economics_for(panel, analysis, fmm, nyi, eta, score_fn=None, keep_rows=False):
        # what the wrapper hands down: the block it was asked for, in place of the combined one
        record.append(("scores", analysis.id, score_fn("panel", analysis, "G1_24", "xgb", 42),
                       score_fn("panel", analysis, "G6", "xgb", 42)))
        return fake(panel, analysis, fmm, nyi, eta, score_fn, keep_rows)
    monkeypatch.setattr(econ, "economics_for", economics_for)
    monkeypatch.setattr(module, "cached_scores", base)
    monkeypatch.setattr(module, "_load_inputs", lambda: ("panel", "fmm", "nyi", 0.05, "protocol_inputs.json (test)"))
    return module, out, record, base, economics_for


def test_main_runs_the_analyses_into_the_default_folder_and_the_primary_one_is_not_touched(stubbed, capsys):
    module, out, record, base, hook_target = stubbed
    primary = out / "economics"
    before = sorted((p.name, p.read_bytes()) for p in primary.iterdir())
    assert module.main(["--block", "DART3", "--analyses", "AQ3,AQ4", "--n-rep", "6"]) == 0
    folder = out / "economics_block" / "DART3"
    assert sorted(p.name for p in folder.iterdir()) == [
        "block_meta.json", "bootstrap_ci.csv", "decomposition.csv", "rotation.csv", "run_meta.json", "summary.csv",
        "volmanagement_check.csv"]
    assert sorted((p.name, p.read_bytes()) for p in primary.iterdir()) == before
    meta = json.loads((folder / "run_meta.json").read_text())
    assert meta["status"] == "complete" and meta["analyses_done"] == ["AQ3", "AQ4"] and meta["n_rep"] == 6
    assert (meta["eta"], meta["eta_source"]) == (0.05, "protocol_inputs.json (test)")
    rot = pd.read_csv(folder / "rotation.csv")
    assert list(rot["window"]) == ["AQ3", "AQ4", "AQ3+AQ4 (rehearsal of the pooled H-E test)"]     # the pooled rehearsal runs
    assert [r[:2] for r in record if r[0] == "scores"] == [("scores", "AQ3"), ("scores", "AQ4")]
    assert all(r[2:] == ("frame:DART3", "frame:G6") for r in record if r[0] == "scores")


def test_main_restores_economics_for_after_a_normal_run(stubbed):
    import es.run_economics as econ
    module, out, record, base, hook_target = stubbed
    assert econ.economics_for is hook_target
    module.main(["--block", "DART3", "--analyses", "AQ3", "--n-rep", "6"])
    assert econ.economics_for is hook_target


def test_main_restores_economics_for_after_an_exception_and_lets_the_exception_through(stubbed, monkeypatch):
    import es.run_economics as econ
    module, out, record, base, hook_target = stubbed

    def failing(panel, analysis, fmm, nyi, eta, score_fn=None, keep_rows=False):
        raise RuntimeError("boom in the middle of the run")
    monkeypatch.setattr(econ, "economics_for", failing)
    with pytest.raises(RuntimeError, match="boom in the middle"):
        module.main(["--block", "DART3", "--analyses", "AQ3", "--n-rep", "6"])
    assert econ.economics_for is failing
    meta = json.loads((out / "economics_block" / "DART3" / "run_meta.json").read_text())
    assert meta["status"] == "failed" and meta["failed_on"] == "AQ3"


def test_the_hook_is_not_in_place_while_the_data_are_loaded(stubbed, monkeypatch):
    """Only the run itself is under the hook: the loaders (and the checks before them) are not."""
    import es.run_economics as econ
    module, out, record, base, hook_target = stubbed
    seen = []
    monkeypatch.setattr(module, "_load_inputs", lambda: (seen.append(econ.economics_for is hook_target),
                                                         ("panel", "fmm", "nyi", 0.05, "x.json"))[1])
    module.main(["--block", "DART3", "--analyses", "AQ3", "--n-rep", "6"])
    assert seen == [True]


def test_the_start_up_message_names_the_block_its_features_the_analyses_and_the_destination_before_any_data_is_read(
        stubbed, monkeypatch, capsys):
    from es.blocks import BLOCKS, features
    module, out, record, base, hook_target = stubbed
    printed_before_loading = []

    def load():
        printed_before_loading.append(capsys.readouterr().out)
        return "panel", "fmm", "nyi", 0.05, "x.json"
    monkeypatch.setattr(module, "_load_inputs", load)
    module.main(["--block", "DART3", "--analyses", "AQ3,AQ4", "--n-rep", "6"])
    (header,) = printed_before_loading
    assert "DART3" in header and "plain DART lags" in header
    assert f"{len(features('DART3'))} features" in header and f"{len(features('G1_24'))} features" in header
    assert all(name in header for name in BLOCKS["DART3"])               # the three lags are named
    assert "AQ3, AQ4" in header
    assert str(out / "economics_block" / "DART3") in header
    assert "6 resamples" in header
    assert "2026-06-30" in header                                        # the development-only promise


def test_describe_run_for_a_block_with_many_features_counts_them_and_does_not_list_them_all():
    from es.blocks import BLOCKS, features
    module = _module()
    text = module.describe_run("G1_24", ["AQ4"], Path("/somewhere"), 2000)
    assert f"{len(features('G1_24'))} features" in text and f"{len(BLOCKS['G1_24'])} price-history" in text
    assert "/somewhere" in text and "AQ4" in text and "2000 resamples" in text
    assert BLOCKS["G1_24"][0] not in text


def test_the_provenance_file_records_the_block_its_features_and_the_run_settings(stubbed):
    from es.blocks import BLOCKS, features
    module, out, record, base, hook_target = stubbed
    module.main(["--block", "DART3", "--analyses", "AQ3,AQ4", "--n-rep", "6"])
    folder = out / "economics_block" / "DART3"
    info = json.loads((folder / "block_meta.json").read_text())
    assert info["alert_block"] == "DART3" and info["replaces"] == "G1_24" and info["calendar_comparator_block"] == "G6"
    assert info["features"] == features("DART3") and info["n_features"] == len(features("DART3"))
    assert info["primary_n_features"] == len(features("G1_24"))
    assert info["analyses"] == ["AQ3", "AQ4"] and info["n_rep"] == 6
    assert info["out_dir"] == str(folder.resolve()) and "exploratory" in info["label"]
    assert info["argv"] == ["--block", "DART3", "--analyses", "AQ3,AQ4", "--n-rep", "6"]


def test_the_provenance_file_is_written_before_the_run_so_a_failed_run_is_labelled_too(stubbed, monkeypatch):
    import es.run_economics as econ
    module, out, record, base, hook_target = stubbed

    def failing(panel, analysis, fmm, nyi, eta, score_fn=None, keep_rows=False):
        raise RuntimeError("boom")
    monkeypatch.setattr(econ, "economics_for", failing)
    with pytest.raises(RuntimeError):
        module.main(["--block", "LEV6", "--analyses", "AQ4", "--n-rep", "6"])
    assert json.loads((out / "economics_block" / "LEV6" / "block_meta.json").read_text())["alert_block"] == "LEV6"


def test_main_with_an_explicit_out_dir_writes_there_and_overwrite_replaces_an_earlier_run(stubbed, tmp_path):
    module, out, record, base, hook_target = stubbed
    target = tmp_path / "scratch" / "regression"
    argv = ["--block", "G1_24", "--analyses", "AQ4", "--n-rep", "6", "--out-dir", str(target)]
    assert module.main(argv) == 0 and (target / "summary.csv").exists()
    with pytest.raises(SystemExit):                                  # the second time it is an earlier run's folder
        module.main(argv)
    first = (target / "summary.csv").read_bytes()
    assert module.main(argv + ["--overwrite"]) == 0
    assert (target / "summary.csv").read_bytes() == first             # same inputs, same numbers
    assert not list(target.glob("*.tmp"))


def test_the_default_resample_count_is_that_of_the_primary_analysis(stubbed):
    from es.run_economics import N_REP
    module, out, record, base, hook_target = stubbed
    args = module._parser().parse_args(["--block", "DART3"])
    assert args.n_rep == N_REP == 2000 and args.analyses is None and args.out_dir is None and args.overwrite is False


# ---------------------------------------------------------------------------------------------------------------
# The loaders: development data only, the same three as es.run_economics.main
# ---------------------------------------------------------------------------------------------------------------

def test_the_loaders_are_the_development_loaders_of_the_primary_analysis_called_the_same_way(tmp_path, monkeypatch):
    from es.config import DATA_ROOT
    module = _module()
    out = tmp_path / "outputs" / "es"
    (out / "analogs").mkdir(parents=True)
    (out / "analogs" / "protocol_inputs.json").write_text(json.dumps({"eta": 0.05, "other": 1}))
    monkeypatch.setattr(module, "OUT_DIR", out)
    calls = []
    monkeypatch.setattr(module, "load_caiso_fmm", lambda *a, **k: calls.append(("fmm", a, k)) or "fmm")
    monkeypatch.setattr(module, "load_nyiso_integrated", lambda *a, **k: calls.append(("nyi", a, k)) or "nyi")
    monkeypatch.setattr(module, "load_dev_raw", lambda *a, **k: calls.append(("raw", a, k)) or "raw")
    monkeypatch.setattr(module, "build_panel", lambda raw: calls.append(("panel", (raw,), {})) or "panel")
    panel, fmm, nyi, eta, eta_source = module._load_inputs()
    assert (panel, fmm, nyi, eta) == ("panel", "fmm", "nyi", 0.05)
    assert eta_source == str(out / "analogs" / "protocol_inputs.json")
    assert ("fmm", (DATA_ROOT / "caiso_fmm_dev",), {}) in calls          # no dev_only override: the guard stays on
    assert ("nyi", (DATA_ROOT / "nyiso_rtint_dev",), {}) in calls
    assert ("raw", (), {}) in calls and ("panel", ("raw",), {}) in calls


def test_the_wrapper_never_asks_for_a_sealed_window_or_the_sealed_run_machinery():
    import inspect
    source = inspect.getsource(_module())
    for forbidden in ("dev_only=False", "sealed_run", "score_analysis", "SEALED", "sealed_pull"):
        assert forbidden not in source, forbidden


# ---------------------------------------------------------------------------------------------------------------
# run_all through the hook on synthetic score frames: the mechanism end to end
# ---------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mapped_run(tmp_path_factory):
    """run_all on one synthetic analysis under alert_block("DART3"), where the 'DART3' frame is the combined frame with a
    different score, and economics_for called directly with that frame in place of the combined one."""
    import es.run_economics as econ
    module = _module()
    tre = _economics_tests()
    combined, g6 = tre._syn_scores()
    block_frame = combined.assign(score=np.random.default_rng(11).normal(size=len(combined)))
    calls = []

    def base(panel, analysis, block, estimator, seed):
        calls.append((analysis.id, block, estimator, seed))
        return {"G1_24": combined, "DART3": block_frame, "G6": g6}[block].copy()
    out_dir = tmp_path_factory.mktemp("mapped_run")
    original = econ.economics_for
    with module.alert_block("DART3", base):
        econ.run_all(None, None, None, 0.01, [tre._syn_analysis()], out_dir, n_rep=8, eta_source="synthetic")
    rows, kept = econ.economics_for(None, tre._syn_analysis(), None, None, 0.01, score_fn=tre._stub(block_frame, g6),
                                    keep_rows=True)
    return {"calls": calls, "out_dir": out_dir, "rows": rows, "kept": kept, "original": original,
            "combined": combined, "block_frame": block_frame, "g6": g6}


def test_run_all_under_the_hook_asks_for_the_block_and_the_calendar_block_and_never_for_the_combined_one(mapped_run):
    assert mapped_run["calls"] == [("SYN", "DART3", "xgb", 42), ("SYN", "G6", "xgb", 42)]


def test_run_all_under_the_hook_leaves_the_module_as_it_found_it(mapped_run):
    import es.run_economics as econ
    assert econ.economics_for is mapped_run["original"]


def test_the_summary_under_the_hook_is_that_of_economics_for_with_the_blocks_frame_in_place_of_the_combined_one(
        mapped_run, tmp_path):
    direct = tmp_path / "direct.csv"
    pd.DataFrame(mapped_run["rows"]).to_csv(direct, index=False)
    got, want = pd.read_csv(mapped_run["out_dir"] / "summary.csv"), pd.read_csv(direct)
    pd.testing.assert_frame_equal(got, want, check_exact=False, rtol=1e-12, atol=1e-12)
    assert len(got) == 388 and set(got["analysis"]) == {"SYN"}


def test_that_comparison_is_not_vacuous_the_blocks_alerts_are_not_the_combined_blocks_alerts(mapped_run):
    from es.config import ALPHA
    from es.policies import p1_fixed
    key = ["market", "node", "timestamp_utc"]
    alerts = {name: set(map(tuple, p1[p1["alert"].astype(bool)][key].to_numpy()))
              for name, p1 in (("combined", p1_fixed(mapped_run["combined"], ALPHA)),
                               ("block", p1_fixed(mapped_run["block_frame"], ALPHA)))}
    assert alerts["combined"] and alerts["block"] and alerts["combined"] != alerts["block"]
    kept_p1 = mapped_run["kept"]["P1"]
    assert set(map(tuple, kept_p1[kept_p1["alert"].astype(bool)][key].to_numpy())) == alerts["block"]


def test_the_other_four_files_under_the_hook_hold_the_analysis_and_its_ex_caiso_run(mapped_run):
    folder = mapped_run["out_dir"]
    cis = pd.read_csv(folder / "bootstrap_ci.csv")
    assert list(dict.fromkeys(cis["analysis"])) == ["SYN", "SYN|ex-CAISO"]
    assert {"E4:dSELC|calendar@count", "E4:dESred|calendar@count", "E4:dSELC|VOL@count", "E4:dESred|VOL@count"} <= set(cis["stat"])
    assert list(pd.read_csv(folder / "rotation.csv")["window"]) == ["SYN"]
    meta = json.loads((folder / "run_meta.json").read_text())
    assert meta["status"] == "complete" and meta["analyses_done"] == ["SYN"] and meta["n_rep"] == 8
