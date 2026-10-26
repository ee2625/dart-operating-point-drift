import datetime as dt

import pandas as pd
import pytest


def test_static_parts_match_the_july_pipeline():
    from es.impl_table import blocks, collect_static, logit_params, scaler_spec, windows, xgb_params
    x = xgb_params()
    assert (x["n_estimators"], x["max_depth"], x["learning_rate"]) == (300, 4, 0.05)
    assert (x["subsample"], x["colsample_bytree"], x["reg_lambda"], x["eval_metric"], x["random_state"]) == (0.8, 0.8, 2.0, "aucpr", 42)
    lg = logit_params()
    assert (lg["C"], lg["class_weight"], lg["max_iter"]) == (1.0, "balanced", 2000)
    assert "1.4826" in scaler_spec()["spread"]                       # raises if src.transfer disagrees
    b = blocks()
    assert [b[k]["n_price_features"] for k in ("G6", "LEV6", "MID3", "DART3", "MIDDART6", "ROLL15",
                                                "ROLL_DISP9", "ROLL_LOC6", "DART18", "G1_24")] == [0, 6, 3, 3, 6, 15, 9, 6, 18, 24]
    assert len(b["calendar_G6"]) == 10
    w = {r["id"]: r for r in windows()}
    assert w["24H2-25H2"]["train"] == {"start": "2023-01-01", "end": "2024-07-01", "convention": "UTC half-open [start, end)"}
    assert w["SA"]["deploy"]["convention"].startswith("local settlement dates")
    assert set(collect_static()) == {"xgboost", "logistic", "scaler", "windows", "blocks", "rules", "versions"}
    from es.paper_registry import PINS
    from es.impl_table import rules
    r = rules()
    assert r["bootstrap"]["replicates"] == PINS["bootB"] == 2000 and r["P4_windows_days"][0] == PINS["pFourWindow"] == 56


def test_coverage_refuses_a_missing_duplicate_count():
    from es.impl_table import collect_coverage
    panel = pd.DataFrame({"market": ["MISO"], "node": ["A"], "timestamp_utc": pd.to_datetime(["2023-01-01 05:00"], utc=True)})
    with pytest.raises(KeyError, match="duplicates_dropped"):
        collect_coverage(panel.copy(), panel, None, None)        # raw.attrs lacks the count: never print a sentinel


def test_coverage_from_synthetic_frames():
    from es.impl_table import collect_coverage
    ts = pd.date_range("2023-04-13 08:00", periods=48, freq="h", tz="UTC")
    panel = pd.DataFrame({"market": ["CAISO"] * 48, "node": ["A"] * 24 + ["B"] * 24, "timestamp_utc": list(ts[:24]) * 2,
                          "settle_date": [t.tz_convert("America/Los_Angeles").date() for t in ts[:24]] * 2})
    raw = panel.copy(); raw.attrs["duplicates_dropped"] = 7
    fmm = pd.DataFrame({"node": ["A"] * 3, "timestamp_utc": ts[:3] + pd.Timedelta(days=91), "fmm_lmp": [1.0, 2.0, 3.0]})
    cov = collect_coverage(raw, panel, fmm, None).set_index("series")
    assert cov.loc["CAISO", "nodes"] == 2 and cov.loc["CAISO", "hourly_rows"] == 48
    assert cov.loc["CAISO", "first_local_date"] == "2023-04-13"
    assert cov.loc["CAISO FMM (settlement leg)", "first_local_date"] == "2023-07-13"
    assert int(cov["duplicates_dropped_total"].iloc[0]) == 7
    assert "NYISO" in cov.index and "ISONE" in cov.index                 # every market has a row, even when empty




# ---------------------------------------------------------------------------------------------
# Controller Rulings 59 and 65 (pre-review and fix round 1): scaler fallback, missing-data rule, coverage notes,
# duplicates, numbers fixed by the spec, entries read from the code, and the implementation.json contract.
# ---------------------------------------------------------------------------------------------
import re


def _cited(text):
    """(path, line number, source line) for every 'dir/file.py:123' citation in a generated sentence."""
    from es.impl_table import DEV_ROOT
    return [(path, int(line), (DEV_ROOT / path).read_text().splitlines()[int(line) - 1])
            for path, line in re.findall(r"((?:src|es|v2)/[\w/]+\.py):(\d+)", text)]


@pytest.fixture(scope="module")
def static_rules():
    from es.impl_table import rules
    return rules()


def test_scaler_fallback_is_read_from_the_pipeline_and_pinned():
    import inspect
    from src import transfer
    from es.impl_table import scaler_spec
    # src/transfer.py:452 declares the threshold as the default of fit_scaler's `minimum_scale` (no module constant)
    assert inspect.signature(transfer.fit_scaler).parameters["minimum_scale"].default == 1e-6
    fb = scaler_spec()["fallback"]
    assert "training standard deviation" in fb and "1.4826 x MAD <= 1e-6" in fb and "then 1.0" in fb
    assert "calendar_is_peak_hour" in fb and "calendar_is_weekend" in fb
    # the behaviour the sentence describes, on the real fit_scaler (src/transfer.py:464-466)
    x = pd.DataFrame({"flag": [0.0] * 8 + [1.0] * 4, "const": [3.0] * 12,
                      "wide": [1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 31.0, 40.0, 41.0, 50.0]})
    _, sp = transfer.fit_scaler(x, ["flag", "const", "wide"])
    assert sp["flag"] == pytest.approx(x["flag"].std())              # 1.4826 x MAD = 0: the sample standard deviation
    assert sp["const"] == 1.0                                        # that standard deviation is 0: 1.0
    assert sp["wide"] == pytest.approx(1.4826 * (x["wide"] - x["wide"].median()).abs().median())


def test_scaler_spec_raises_when_the_pipeline_has_no_fallback(monkeypatch):
    from src import transfer
    from es.impl_table import scaler_spec

    def mad_only(train, feature_names, minimum_scale=1e-6):          # fit_scaler without its fallback
        names = list(feature_names)
        centre = train[names].median()
        return centre, (train[names] - centre).abs().median() * 1.4826

    monkeypatch.setattr(transfer, "fit_scaler", mad_only)
    with pytest.raises(RuntimeError, match="fallback"):
        scaler_spec()


def test_scaler_spec_raises_when_the_pipeline_is_not_median_over_mad(monkeypatch):
    from src import transfer
    from es.impl_table import scaler_spec

    def raw_mad(train, feature_names, minimum_scale=1e-6):           # the MAD without the 1.4826 factor
        names = list(feature_names)
        centre = train[names].median()
        return centre, (train[names] - centre).abs().median()

    monkeypatch.setattr(transfer, "fit_scaler", raw_mad)
    with pytest.raises(RuntimeError, match="not median/1.4826"):
        scaler_spec()


def test_the_fallback_applies_to_the_zero_one_calendar_flags():
    from es.blocks import G6
    from es.impl_table import calendar_fallback_features
    from v2.panel_prep import add_features
    ts = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h", tz="UTC")
    cal = add_features(pd.DataFrame({"market": "NYISO", "node": "n", "timestamp_utc": ts, "da_lmp": 1.0, "rt_lmp": 1.0}))
    flags = [c for c in G6 if set(cal[c].unique()) <= {0, 1}]
    assert flags == ["calendar_is_peak_hour", "calendar_is_weekend"]
    assert calendar_fallback_features() == flags


def test_calendar_fallback_features_raises_when_fit_scaler_disagrees(monkeypatch):
    from src import transfer
    from es.impl_table import calendar_fallback_features

    def mad_only(train, feature_names, minimum_scale=1e-6):          # a flag gets spread 0, not its standard deviation
        names = list(feature_names)
        centre = train[names].median()
        return centre, (train[names] - centre).abs().median() * 1.4826

    def std_only(train, feature_names, minimum_scale=1e-6):          # every feature gets the standard deviation
        names = list(feature_names)
        return train[names].median(), train[names].std()

    monkeypatch.setattr(transfer, "fit_scaler", mad_only)
    with pytest.raises(RuntimeError, match=r"calendar_is_peak_hour the spread .*standard deviation fallback"):
        calendar_fallback_features()
    monkeypatch.setattr(transfer, "fit_scaler", std_only)
    with pytest.raises(RuntimeError, match=r"calendar_day_of_week_cos the spread .*1\.4826 x MAD"):
        calendar_fallback_features()


def _g1_panel(n=420, drop_rows=(), nan_rt_rows=()):
    import numpy as np
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"market": "NYISO", "node": "A", "timestamp_utc": ts,
                       "da_lmp": 50.0 + rng.normal(size=n), "rt_lmp": 50.0 + rng.normal(size=n)})
    df.loc[list(nan_rt_rows), "rt_lmp"] = float("nan")
    return df.drop(index=list(drop_rows)).reset_index(drop=True), ts


def _rolling_min_observed(w, T=400):
    """The smallest number of observed hours in the w-hour window ending 48 h before row T for which
    v2.panel_prep.add_g1_features still computes the rolling mean (found by bisection on the real function)."""
    from v2.panel_prep import add_g1_features
    window = list(range(T - 48 - w + 1, T - 48 + 1))

    def computed(k):                                                  # keep a price for the first k hours of the window only
        df, ts = _g1_panel(nan_rt_rows=window[k:])
        out = add_g1_features(df)
        return bool(out.loc[out["timestamp_utc"] == ts[T], f"dart_spread_roll_mean_{w}h_shift_48h"].notna().iloc[0])

    assert computed(w) and not computed(0)
    lo, hi = 1, w
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if computed(mid) else (mid + 1, hi)
    return lo


def test_missing_data_rule_matches_add_g1_features(static_rules):
    import inspect
    from v2 import panel_prep
    rule = static_rules["missing_data"]
    assert rule.startswith("nothing is imputed")
    # rolling tolerance: v2/panel_prep.py:92 `mp = max(2, w // 2)` is an inline expression, so it is cited and pinned here
    found = {w: _rolling_min_observed(w) for w in panel_prep.G1_ROLL_WINDOWS}
    assert found == {24: 12, 72: 36, 168: 84}
    assert "max(2, w // 2)" in rule
    assert "(" + ", ".join(f"{w} h: {k}" for w, k in found.items()) + ")" in rule
    # a lag needs the exact hour: with the row 48 h before T absent, the 48 h lag is missing and the 72 h lag is not
    T = 300
    df, ts = _g1_panel(drop_rows=[T - 48])
    row = panel_prep.add_g1_features(df).query("timestamp_utc == @ts[@T]").iloc[0]
    assert pd.isna(row["dart_spread_lag_48h"]) and pd.notna(row["dart_spread_lag_72h"])
    assert "exactly that hour" in rule and "absent" in rule
    # each node drops duplicate timestamps, keeping the first (v2/panel_prep.py:82); the keep-first rule is pinned by source
    base, _ = _g1_panel()
    dup = base.iloc[[100]].copy()
    dup["rt_lmp"] = 999.0
    twin = pd.concat([base.iloc[:101], dup, base.iloc[101:]], ignore_index=True)
    out = panel_prep.add_g1_features(twin)
    assert out["timestamp_utc"].is_unique and len(out) == len(base)
    assert 'index.duplicated(keep="first")' in inspect.getsource(panel_prep.add_g1_features)
    assert "each node drops duplicate timestamps, keeping the first" in rule


def test_missing_data_rule_cites_the_inner_merge_of_all_four_ingesters(static_rules):
    rule = static_rules["missing_data"]
    assert ("inner-merges the day-ahead and real-time series on timestamp and node (and node type for ISO-NE and MISO), so an "
            "hour with a price on one side only is absent from the panel") in rule
    ingesters = [c for c in _cited(rule) if c[0].startswith("v2/ingest_")]
    assert sorted(path for path, _, _ in ingesters) == [
        "v2/ingest_caiso.py", "v2/ingest_isone.py", "v2/ingest_miso.py", "v2/ingest_nyiso.py"]
    assert all("da.merge(rt" in source and 'how="inner"' in source for _, _, source in ingesters)


def test_information_policy_numbers_come_from_the_code_or_are_marked_as_spec_fixed(static_rules):
    from es.paper_registry import CODE_CONSTANTS
    from v2.panel_prep import G1_LAGS, G1_ROLL_WINDOWS
    r = static_rules
    assert "48, 72 and 168 h" in r["information_policy"] and "24, 72 and 168 h" in r["information_policy"]
    assert G1_LAGS == (48, 72, 168) and G1_ROLL_WINDOWS == (24, 72, 168)
    assert "216 h" in r["information_policy"] and "the longest rolling window plus the shift, §3.3" in r["information_policy"]
    fixed = r["spec_fixed"]
    assert {"P4_windows_days", "P5_eta_grid", "feature_lookback_h", "information_shift_h", "coverage_rule", "position_mw"} <= set(fixed)
    # G3 item 14: the reader has no design specification, so each value says in plain words what fixes it and where the paper describes it
    assert all(re.search(r"§\d+(\.\d+)*", v) and "spec" not in v and "I3" not in v for v in fixed.values()), fixed
    assert "§3.7.3" in fixed["P4_windows_days"] and "§3.7.3" in fixed["P5_eta_grid"] and "§3.3" in fixed["feature_lookback_h"]
    assert "§3.3" in fixed["information_shift_h"] and "§6.1.1" in fixed["coverage_rule"] and "§3.7.1" in fixed["position_mw"]
    # the typed copies agree with the ones behind the macros (es.paper_registry.CODE_CONSTANTS)
    assert r["P4_windows_days"] == [56, CODE_CONSTANTS["pFourWindowAlt"]] and r["P5_eta_grid"] == [0.005, 0.01, 0.02, 0.05]
    assert CODE_CONSTANTS["lookbackH"] == 216 and CODE_CONSTANTS["lagMinH"] == 48 and CODE_CONSTANTS["lagMaxH"] == 168
    assert r["position_mw"] == CODE_CONSTANTS["posMW"] == 1
    cov = r["coverage_rule"]
    assert f"{CODE_CONSTANTS['covMinNodes']} usable nodes" in cov and f"{CODE_CONSTANTS['covMinShare']}%" in cov
    assert "set before the sealed windows were opened, §6.1.1" in cov
    assert r["seeds"]["primary"] == 42 == CODE_CONSTANTS["seedPrimary"]


def test_rules_raise_when_the_spec_numbers_disagree_with_the_code(monkeypatch):
    import es.impl_table as it
    monkeypatch.setattr(it, "SPEC_LOOKBACK_H", 217)
    with pytest.raises(RuntimeError, match="spec v3 fixes a 48 h shift and a 217 h lookback"):
        it.rules()
    monkeypatch.setattr(it, "SPEC_LOOKBACK_H", 216)
    monkeypatch.setattr(it, "SPEC_SHIFT_H", 24)
    with pytest.raises(RuntimeError, match="spec v3 fixes a 24 h shift"):
        it.rules()


def test_rules_agree_with_the_constants_behind_the_macros(static_rules):
    """A golden test: the table and the macros read the same code through two routes and must agree."""
    from es.paper_pins import band_text, pct_of
    from es.paper_registry import CODE_CONSTANTS as C
    from es.paper_registry import MARKET_WORD
    from es.paper_tex import fmt_dec, fmt_g, tex_escape
    r = static_rules
    for market, word in MARKET_WORD.items():                                    # calibrated fees, INC and DEC per market
        assert r["fees_calibrated"][market] == {"INC": C[f"fee{word}Inc"], "DEC": C[f"fee{word}Dec"]}
    assert ", ".join(fmt_dec(c, 2) for c in r["fees_uniform_grid"]) == C["feeGrid"]
    assert r["seeds"]["all"] == [int(s) for s in C["seedList"].split(", ")] and len(r["seeds"]["all"]) == C["nSeeds"]
    assert r["seeds"]["primary"] == C["seedPrimary"]
    assert r["es_fraction_pct"] == C["esFrac"]
    assert ", ".join(fmt_g(lam, 6) for lam in r["lambda_grid"]) == C["lamGrid"]
    assert ", ".join(fmt_g(pct_of(a), 6) for a in r["alpha_sweep"]) == C["alphaGrid"] and pct_of(r["alpha"]) == C["alphaPct"]
    assert f"at least {C['minPositives']} positive training labels" in r["feasibility"]
    b = r["bootstrap"]
    assert (b["replicates"], b["mean_block_days"], b["seed"], b["rotation_min_shift_days"]) == (
        C["bootB"], C["bootBlockDays"], C["bootSeed"], C["rotMinShiftDays"])
    assert r["P4_windows_days"] == [C["pFourWindow"], C["pFourWindowAlt"]]
    assert r["time_zones"]["MISO"] == C["tzMiso"]
    bands = r["failure_bands"]
    assert band_text(bands["0.5_2"]) == C["bandWide"] and band_text(bands["0.8_1.25"]) == C["bandTight"]
    assert band_text(bands["0.67_1.5"]) == C["bandMid"] == "[0.67, 1.5]"
    assert " $\\times$ ".join(tex_escape(s) for s in r["strata_primary"]) == C["strataPrimary"]
    assert " $\\times$ ".join(tex_escape(s) for s in r["strata_sensitivity"]) == C["strataDecile"]


def test_importing_impl_table_does_not_import_run_legacy():
    """The constants come from es.paper_pins (mirrors, tested equal to run_legacy's in test_paper_pins.py); the one use of
    es.run_legacy, the seed scope, is a call-time import."""
    import subprocess
    import sys
    from es import impl_table, run_legacy
    assert (impl_table.SEEDS, impl_table.ALPHAS, impl_table.BANDS) == (run_legacy.SEEDS, run_legacy.ALPHAS, run_legacy.BANDS)
    code = "import sys, es.impl_table; print('es.run_legacy' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=impl_table.DEV_ROOT, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
    assert r.stdout.strip() == "False"


def test_docstring_and_help_text_do_not_overclaim(capsys):
    from es import impl_table
    doc = " ".join(impl_table.__doc__.split())
    assert "Every number comes from" not in doc and "about two minutes" not in doc
    assert "set by the design of the study" in doc and "rules()" in doc
    with pytest.raises(SystemExit):
        impl_table.main(["--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "2 minutes" not in help_text and "--with-data" in help_text


# --- the XGBoost class weight -------------------------------------------------------------------

def test_xgboost_entry_describes_the_class_weight_and_fit_model_passes_it():
    import importlib.metadata as md
    from xgboost import XGBClassifier
    from src import transfer
    from es.impl_table import XGB_KEYS, xgb_params
    from es.scores import xgb_factory
    x = xgb_params()
    text = x["scale_pos_weight"]
    assert isinstance(text, str)                                      # never the factory's placeholder 1.0
    assert text.startswith("n_negative / n_positive of the node's feature-complete training rows (src/transfer.py:")
    assert f"every other parameter at the xgboost {md.version('xgboost')} default except n_jobs = 4 (the thread count)" in text
    (path, line, source), = _cited(text)
    assert "model_factory(scale_pos_weight=negatives / max(positives, 1))" in source
    got, default = xgb_factory(42)(1.0).get_params(), XGBClassifier().get_params()
    differs = {k for k in got if not (got[k] == default[k] or (got[k] != got[k] and default[k] != default[k]))}   # nan == nan
    assert differs == set(XGB_KEYS) | {"scale_pos_weight", "n_jobs"}
    # what fit_model really passes: only the feature-complete rows count (the 10 incomplete positives would make it 1.5)
    seen = {}

    class Recorder:
        def fit(self, X, y):
            seen["n_fit"] = len(y)

    def factory(**kwargs):
        seen.update(kwargs)
        return Recorder()

    toy = pd.DataFrame({"f": [0.0] * 60 + [1.0] * 30 + [float("nan")] * 10, "label": [0] * 60 + [1] * 30 + [1] * 10})
    assert transfer.fit_model(factory, toy, ["f"], "label") is not None
    assert seen == {"scale_pos_weight": 2.0, "n_fit": 90}


def test_xgb_checks_raise_on_a_changed_factory_or_class_weight(monkeypatch):
    import es.impl_table as it
    from xgboost import XGBClassifier
    from src import transfer

    def exact_trees(seed):
        return lambda w: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                       reg_lambda=2.0, scale_pos_weight=w, eval_metric="aucpr", random_state=seed, n_jobs=4,
                                       tree_method="exact")

    def wrong_weight(model_factory, train, feature_names, label_column):
        model_factory(scale_pos_weight=0.5)
        return object()

    with monkeypatch.context() as m:
        m.setattr(it, "xgb_factory", exact_trees)
        with pytest.raises(RuntimeError, match="tree_method"):
            it.xgb_params()
    with monkeypatch.context() as m:
        m.setattr(transfer, "fit_model", wrong_weight)
        with pytest.raises(RuntimeError, match=r"scale_pos_weight=0\.5"):
            it.xgb_params()


def test_the_primary_seed_is_the_default_of_xgb_params():
    import inspect
    from es.impl_table import PRIMARY_SEED, logit_params, xgb_params
    assert inspect.signature(xgb_params).parameters["seed"].default == PRIMARY_SEED == 42
    assert xgb_params()["random_state"] == PRIMARY_SEED and xgb_params(7)["random_state"] == 7
    assert logit_params()["class_weight"] == "balanced"


# --- entries read from the code ------------------------------------------------------------------

def test_policy_entries_are_read_from_es_policies(static_rules):
    import inspect
    from es import policies
    p5 = inspect.signature(policies.p5_self_correcting).parameters
    assert (p5["lo"].default, p5["hi"].default) == (0.001, 0.5)
    text = static_rules["P5_clip"]
    assert "[0.001, 0.5]" in text and "alpha_(d+1) = min(0.5, max(0.001, alpha_d + eta (alpha - r_d)))" in text
    assert [source.strip().startswith("def p5_self_correcting(") for _, _, source in _cited(text)] == [True]
    iters = inspect.signature(policies.p4_matched).parameters["iters"].default
    assert (policies._RATE_LO, policies._RATE_HI, iters) == (1e-4, 0.5, 40)
    text = static_rules["P4_matched"]
    assert "[0.0001, 0.5]" in text and "40 iterations" in text and "P1" in text
    sources = [source.strip() for _, _, source in _cited(text)]
    assert len(sources) == 2 and any(s.startswith("def p4_matched(") for s in sources)
    assert any(s.startswith("_RATE_LO, _RATE_HI =") for s in sources)


def _stress_rows():
    day1, day2 = pd.Timestamp("2024-01-10").date(), pd.Timestamp("2024-01-11").date()
    return pd.DataFrame({
        "market": "NYISO", "node": "N.Y.C.", "settle_date": [day1] * 3 + [day2] * 3,
        "timestamp_utc": pd.to_datetime(["2024-01-10 05:00", "2024-01-10 06:00", "2024-01-10 07:00",
                                         "2024-01-11 05:00", "2024-01-11 06:00", "2024-01-11 07:00"], utc=True),
        "abs_mean_168h": [95.0, 0.0, 0.0, 10.0, 99.0, 99.0], "dart_settle": 0.0}), day1, day2


def test_stress_fee_entry_matches_positions(static_rules):
    import numpy as np
    from es import positions
    train = pd.DataFrame({"market": "NYISO", "node": "N.Y.C.", "abs_mean_168h": np.arange(101.0)})
    thr = positions.stress_threshold(train)
    assert float(thr.iloc[0]) == pytest.approx(90.0)                  # the 90th percentile of the training rows
    rows, day1, day2 = _stress_rows()
    fee = positions.attach_pnl(rows, "inc", fees=1.0, stress=thr).groupby("settle_date")["fee"].agg(["min", "max"])
    assert fee.loc[day1].tolist() == [2.0, 2.0]                       # first hour 95 >= 90: every hour of the day doubles
    assert fee.loc[day2].tolist() == [1.0, 1.0]                       # first hour 10 < 90: later hours at 99 do not count
    text = static_rules["stress_fee"]
    assert "90th percentile" in text and "multiplied by 2" in text and "first hour" in text and "training rows" in text
    assert "the trailing 168-hour mean of |DART|, lagged 48 h, at its first hour" in text and "abs_mean_168h" not in text
    sources = [source.strip() for _, _, source in _cited(text)]
    assert len(sources) == 3 and all(any(s.startswith(head) for s in sources) for head in (
        "def stress_threshold(", "def _stressed_days(", "def attach_pnl("))


def test_monitor_entry_matches_run_monitors(static_rules):
    import inspect
    from unittest import mock
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import BaseCrossValidator, StratifiedGroupKFold
    from es import run_monitors
    # the splitter: the one class of its kind that the module names
    assert [n for n, o in vars(run_monitors).items() if isinstance(o, type) and issubclass(o, BaseCrossValidator)] == [
        "StratifiedGroupKFold"]
    rng = np.random.default_rng(0)
    cal, dep = rng.normal(size=(120, 2)), rng.normal(size=(56, 2))
    ids = np.r_[np.repeat(np.arange(19800, 19830), 4), np.repeat(np.arange(19800, 19814), 4)]          # 30 + 14 settlement days
    with mock.patch.object(run_monitors, "LogisticRegression", wraps=LogisticRegression) as model, \
            mock.patch.object(run_monitors, "StratifiedGroupKFold", wraps=StratifiedGroupKFold) as splitter:
        run_monitors.domain_auc(cal, dep, ids)
    assert model.call_args.kwargs == {"max_iter": 2000, "class_weight": "balanced"}
    assert model.call_count == run_monitors.N_FOLDS == 5
    assert splitter.call_args.kwargs == {"n_splits": 5, "shuffle": False}
    with pytest.raises(ValueError, match="needs at least 5"):                      # fewer than 5 day groups
        run_monitors.domain_auc(cal[:8], dep[:8], np.r_[np.repeat([1, 2], 4), np.repeat([1, 2], 4)])
    assert inspect.signature(run_monitors.psi).parameters["bins"].default == 10
    days = run_monitors.MONITOR_DAYS                  # Ruling 66b; this fails loudly if es/run_monitors.py does not define it
    text = static_rules["monitors"]
    assert "max_iter = 2000, class_weight = balanced" in text and "10 reference-quantile bins" in text
    # Ruling 73: class-stratified 5-fold CV blocked by local settlement day, not by week
    assert ("scored by out-of-fold AUROC over 5-fold StratifiedGroupKFold CV, which is class-stratified and blocked by local "
            "settlement day (every day stays whole in one fold, no shuffling; at least 5 distinct day groups are required, "
            "and so are both classes in the training and test part of every fold)") in text
    assert "ISO week" not in text and "GroupKFold blocked" not in text
    # Ruling 93: implementation facts read from the code only; no simulated null level, and no decimal number at all
    assert "no-shift" not in text and "optimistic" not in text and "calib_days" not in text
    assert not re.search(r"\d\.\d", text)
    # Ruling 66b: the window
    assert f"every monitor sees only deployment days 1 to {days}" in text
    assert "W1_norm, PSI and domain_auc compare the calibration rows with the deployment rows of those days" in text
    assert "early_rate is the alert rate over those days" in text
    assert f"the outcome they are scored against, |log10 R| and the failure flag, comes from deployment days {days + 1} to the end" in text
    # Ruling 83: the placebo columns, named from the code
    assert run_monitors.PLACEBO_COLUMNS == {"W1_norm": "W1_norm_placebo", "PSI": "psi_placebo", "domain_auc": "domain_auc_placebo"}
    assert "W1_norm_placebo, psi_placebo and domain_auc_placebo repeat the three input statistics" in text
    assert f"its last {days} days against the earlier ones; they are not monitors" in text and "placebo_median" in text
    assert "Ruling" not in text                                                    # internal rulings stay out of the paper
    cites = _cited(text)
    assert len(cites) == 4 and {path for path, _, _ in cites} == {"es/run_monitors.py"}
    sources = [source.strip() for _, _, source in cites]
    assert any(s.startswith("MONITOR_DAYS") for s in sources) and any(s.startswith("N_FOLDS") for s in sources)
    assert any("class_weight" in s for s in sources) and any(s.startswith("def placebo_split(") for s in sources)


def test_run_monitors_defines_monitor_days():
    """Controller Ruling 66b: every monitor sees deployment days 1 to MONITOR_DAYS. Fails until Task 14 defines the constant."""
    from es import run_monitors
    assert isinstance(run_monitors.MONITOR_DAYS, int) and not isinstance(run_monitors.MONITOR_DAYS, bool)
    assert run_monitors.MONITOR_DAYS >= 1


def _stub_the_monitor_days_citation(monkeypatch):
    """es/run_monitors.py may not define MONITOR_DAYS yet (Task 14). The tests that inject a value skip only the citation of its
    definition; test_monitor_entry_matches_run_monitors checks that citation against the real module."""
    import es.impl_table as it
    real = it._cite
    monkeypatch.setattr(it, "_cite", lambda where, needle: "es/run_monitors.py:1" if needle is it._MONITOR_DAYS_LINE else real(where, needle))


def test_monitor_days_is_read_from_run_monitors_and_a_missing_one_is_loud(monkeypatch):
    from es import run_monitors
    from es.impl_table import monitor_rule
    _stub_the_monitor_days_citation(monkeypatch)
    monkeypatch.setattr(run_monitors, "MONITOR_DAYS", 21, raising=False)           # not Ruling 66b's 14: it is read, never typed
    text = monitor_rule()
    assert "every monitor sees only deployment days 1 to 21" in text and "deployment days 22 to the end" in text
    for bad in (None, 0, -3, 14.0, "14", True):                                     # anything but a positive integer
        monkeypatch.setattr(run_monitors, "MONITOR_DAYS", bad, raising=False)
        with pytest.raises(RuntimeError, match="MONITOR_DAYS is"):
            monitor_rule()
    monkeypatch.delattr(run_monitors, "MONITOR_DAYS")
    with pytest.raises(RuntimeError, match="MONITOR_DAYS is missing"):
        monitor_rule()


def test_the_early_rate_probe_checks_the_window_it_really_uses(monkeypatch):
    """early_rate covers deployment days 1 to MONITOR_DAYS by default: probed on a toy node, not read off its signature (it is
    days=None now and reads MONITOR_DAYS when it is called)."""
    from es import run_monitors
    from es.impl_table import _check_early_rate, monitor_rule
    _check_early_rate()                                                           # the real one, days=None
    _stub_the_monitor_days_citation(monkeypatch)

    def counting(window):
        def early_rate(alerts, days=window):                                      # a function with a window of its own
            early = alerts[run_monitors.deployment_day(alerts) < days]
            return early.groupby(["market", "node"])["alert"].mean()
        return early_rate

    for wrong in (7, 15):                                                          # a window too short, and one day too long
        with monkeypatch.context() as m:
            m.setattr(run_monitors, "early_rate", counting(wrong))
            with pytest.raises(RuntimeError, match="early_rate does not cover exactly the first 14 deployment days"):
                _check_early_rate()
            with pytest.raises(RuntimeError, match="early_rate does not cover exactly"):
                monitor_rule()
    with monkeypatch.context() as m:
        m.setattr(run_monitors, "early_rate", counting(14))                       # the same window, fixed in the signature
        assert "deployment days 1 to 14" in monitor_rule()
    with monkeypatch.context() as m:                                               # no `days` parameter at all
        m.setattr(run_monitors, "early_rate", lambda alerts: counting(run_monitors.MONITOR_DAYS)(alerts))
        assert "deployment days 1 to 14" in monitor_rule()


def test_calendar_entry_matches_add_features(static_rules):
    import numpy as np
    from v2.panel_prep import add_features
    ts = pd.DatetimeIndex(["2023-01-06 06:00", "2023-01-06 07:00", "2023-01-06 22:00", "2023-01-06 23:00",
                           "2023-01-07 12:00", "2023-01-08 12:00", "2023-01-09 12:00"], tz="America/New_York").tz_convert("UTC")
    cal = add_features(pd.DataFrame({"market": "NYISO", "node": "n", "timestamp_utc": ts, "da_lmp": 1.0, "rt_lmp": 1.0}))
    assert cal["calendar_is_peak_hour"].tolist() == [0, 1, 1, 0, 1, 1, 1]     # local hours 7 to 22
    assert cal["calendar_is_weekend"].tolist() == [0, 0, 0, 0, 1, 1, 0]       # Saturday and Sunday
    local = ts.tz_convert("America/New_York")
    for name, value, period in (("hour", local.hour, 24), ("day_of_week", local.dayofweek, 7),
                                ("day_of_year", local.dayofyear, 365.25), ("month", local.month, 12)):
        assert np.allclose(cal[f"calendar_{name}_sin"], np.sin(2 * np.pi * value / period))
        assert np.allclose(cal[f"calendar_{name}_cos"], np.cos(2 * np.pi * value / period))
    text = static_rules["calendar"]
    assert "peak hour = local hours 7 to 22" in text and "weekend = Saturday and Sunday" in text
    assert "periods hour 24, day of week 7, day of year 365.25, month 12" in text
    assert "settlement clock" in text and [c[0] for c in _cited(text)] == ["v2/panel_prep.py"]


def test_seed_scope_is_read_from_the_run_plan(static_rules):
    from es.run_legacy import SEED_BLOCKS, SEEDS, run_plan
    plan = list(run_plan())
    extra = [(a.id, block, est, seed, variants) for a, block, est, seed, variants in plan if seed != 42]
    assert sorted({e[0] for e in extra}) == ["2024-2025", "25H1-26H1"] and {e[3] for e in extra} == set(SEEDS) - {42}
    assert {e[2] for e in extra} == {"xgb"} and {str(e[4]) for e in extra} == {"[('P1', 0.05)]"}
    assert len(extra) == 2 * len(SEED_BLOCKS) * 4
    text = static_rules["seeds"]["extra_seeds_scope"]
    assert "the 4 extra seeds (0, 1, 7, 13) run only for the seed-sensitivity blocks" in text
    assert all(block in text for block in SEED_BLOCKS)
    assert "of 2 analyses (2024-2025, 25H1-26H1), XGBoost, P1 at alpha 0.05" in text and "every other run uses the primary seed 42" in text
    ((path, line, source),) = _cited(text)
    assert path == "es/run_legacy.py" and "# D8: seeds" in source


def test_window_conventions_are_read_from_the_windows(static_rules):
    from es.impl_table import windows
    seen = {}
    for r in windows():
        seen.setdefault(r["group"], set()).update(r[w]["convention"] for w in ("train", "calib", "deploy", "label_train"))
    utc, local = "UTC half-open [start, end)", "local settlement dates [start, end]"
    assert seen == {"legacy": {utc}, "label_control": {utc}, "analog": {local}, "sealed": {local}}
    text = static_rules["window_conventions"]
    head, tail = text.split(local)
    assert "legacy (7 analyses)" in head and "label_control (4 analyses)" in head       # the label-control windows are UTC
    assert "analog (5 analyses)" in tail and "sealed (4 analyses)" in tail and "never pooled" in tail
    assert "new analyses" not in text


def test_cite_points_at_the_line_that_holds_the_needle_and_refuses_anything_else():
    from src import transfer
    from es.impl_table import DEV_ROOT, _cite
    cite = _cite(transfer.fit_model, "model_factory(scale_pos_weight=")
    path, line = cite.rsplit(":", 1)
    assert path == "src/transfer.py" and "model_factory(scale_pos_weight=" in (DEV_ROOT / path).read_text().splitlines()[int(line) - 1]
    assert _cite(DEV_ROOT / "es" / "policies.py", "_RATE_LO, _RATE_HI =").startswith("es/policies.py:")
    with pytest.raises(RuntimeError, match="no line"):
        _cite(transfer.fit_model, "this text is not in fit_model")
    with pytest.raises(RuntimeError, match="expected exactly one"):
        _cite(transfer.fit_scaler, "spread")


# --- coverage notes -----------------------------------------------------------------------------

FALL_BACK_NY = ["2023-11-05 05:00", "2023-11-05 06:00"]                # New York: the repeated 01:00 local hour on 2023-11-05


def _hourly(market, nodes, start, end, drop=None):
    """Hourly UTC rows per node from start to end inclusive, minus drop[node] and drop["*"] (timestamps)."""
    drop = drop or {}
    frames = []
    for node in nodes:
        gone = set(pd.to_datetime(list(drop.get("*", [])) + list(drop.get(node, [])), utc=True))
        ts = [t for t in pd.date_range(start, end, freq="h", tz="UTC") if t not in gone]
        frames.append(pd.DataFrame({"market": market, "node": node, "timestamp_utc": ts}))
    return pd.concat(frames, ignore_index=True)


def _synthetic_world():
    start, end = "2023-11-04 00:00", "2023-11-07 23:00"
    nyiso = _hourly("NYISO", ["A", "B"], start, end, drop={"*": FALL_BACK_NY, "A": ["2023-11-06 10:00"]})
    panel = pd.concat([nyiso, _hourly("CAISO", ["X"], start, end), _hourly("MISO", ["M"], start, end)], ignore_index=True)
    nyi = _hourly("NYISO", ["A", "B", "C"], start, end, drop={"*": FALL_BACK_NY}).drop(columns="market")
    fmm = _hourly("CAISO", ["X"], start, end).drop(columns="market")
    raw = panel.copy()
    raw.attrs["duplicates_dropped"] = 0
    return raw, panel, fmm, nyi


def _one(notes, prefix):
    hit = [n for n in notes if n.startswith(prefix)]
    assert len(hit) == 1, (prefix, notes)
    return hit[0]


def test_coverage_notes_derive_the_gaps_from_the_series():
    from es.impl_table import collect_coverage
    raw, panel, fmm, nyi = _synthetic_world()
    notes = collect_coverage(raw, panel, fmm, nyi).attrs["notes"]
    fb = _one(notes, "NYISO panel: both rows")
    assert "01:00 local hour are absent on the fall-back day (2023-11-05), 2 hours per node" in fb
    assert "The July downloaders drop them (§6.1.1)" in fb
    gap = _one(notes, "NYISO panel: 1 hour absent outside the fall-back rule")
    assert "2023-11-06 10:00Z (1 of 2 nodes; local settlement date 2023-11-06: 23 of 24 hours present)" in gap
    assert "The NYISO integrated series has every one of these node-hours" in gap
    assert "are present on the fall-back day (2023-11-05)" in _one(notes, "CAISO panel: both rows")
    assert "are present" in _one(notes, "CAISO FMM (settlement leg): both rows")
    sett = _one(notes, "NYISO integrated RT LBMP (settlement leg): both rows")
    assert "are absent" in sett and "July downloaders" not in sett
    assert not any(n.startswith(("MISO", "ISONE")) for n in notes)         # fixed EST: nothing to say; no ISONE rows given


def test_coverage_notes_report_what_the_settlement_series_lacks_and_a_late_start():
    from es.impl_table import collect_coverage
    raw, panel, fmm, nyi = _synthetic_world()
    nyi = nyi[nyi["timestamp_utc"] != pd.Timestamp("2023-11-06 10:00", tz="UTC")]
    late = panel[~((panel["node"] == "B") & (panel["market"] == "NYISO") & (panel["timestamp_utc"] < pd.Timestamp("2023-11-04 10:00", tz="UTC")))]
    notes = collect_coverage(raw, late, fmm, nyi).attrs["notes"]
    gap = _one(notes, "NYISO panel: 11 hours absent outside the fall-back rule")
    assert ("2023-11-04 00:00Z to 09:00Z (1 of 2 nodes; local settlement dates 2023-11-03: 0 of 24 hours present, "
            "2023-11-04: 18 of 24 hours present)") in gap
    assert "2023-11-06 10:00Z (1 of 2 nodes; local settlement date 2023-11-06: 23 of 24 hours present)" in gap
    assert "The NYISO integrated series lacks 1 of these 11 node-hours" in gap


def test_coverage_notes_cover_a_midnight_run_the_run_cap_and_a_partial_fall_back():
    from es.impl_table import _local_day_hours, collect_coverage
    start, end = "2023-11-04 00:00", "2023-11-07 23:00"

    def notes_for(drop):
        ny = _hourly("NYISO", ["A", "B"], start, end, drop=drop)
        raw = ny.copy()
        raw.attrs["duplicates_dropped"] = 0
        return collect_coverage(raw, ny, None, None).attrs["notes"]

    # a run that crosses UTC midnight: five hours of one node, all on one local settlement date
    crossing = list(pd.date_range("2023-11-06 22:00", "2023-11-07 02:00", freq="h", tz="UTC"))
    gap = _one(notes_for({"*": FALL_BACK_NY, "A": crossing}), "NYISO panel: 5 hours absent")
    assert "2023-11-06 22:00Z to 2023-11-07 02:00Z (1 of 2 nodes; local settlement date 2023-11-06: 19 of 24 hours present)" in gap
    # more than six runs: the first six are listed and the rest are counted
    lone = [pd.Timestamp("2023-11-04 03:00", tz="UTC") + pd.Timedelta(hours=2 * k) for k in range(8)]
    gap = _one(notes_for({"*": FALL_BACK_NY, "A": lone}), "NYISO panel: 8 hours absent")
    assert gap.count("(1 of 2 nodes;") == 6 and "2 more runs" in gap
    # a partial fall-back: node A lacks both repeated hours, node B only one of them; then one hour in every node
    fb = _one(notes_for({"A": FALL_BACK_NY, "B": FALL_BACK_NY[1:]}), "NYISO panel: of the 2 rows")
    assert "on the fall-back day (2023-11-05), 1 to 2 hours per node are absent" in fb
    fb = _one(notes_for({"*": FALL_BACK_NY[1:]}), "NYISO panel: of the 2 rows")
    assert "on the fall-back day (2023-11-05), 1 hour per node is absent" in fb
    # the length of a local settlement day: 24 hours, 23 in spring and 25 in autumn
    from datetime import date
    assert [_local_day_hours(date(2023, m, d), "America/New_York") for m, d in ((6, 1), (3, 12), (11, 5))] == [24, 23, 25]


def test_coverage_notes_name_the_settlement_nodes_that_are_not_panel_nodes():
    from es.impl_table import collect_coverage
    raw, panel, fmm, nyi = _synthetic_world()
    notes = collect_coverage(raw, panel, fmm, nyi).attrs["notes"]
    assert _one(notes, "NYISO integrated RT LBMP (settlement leg): 3 names") == (
        "NYISO integrated RT LBMP (settlement leg): 3 names, 1 of them not panel nodes (C).")
    assert not any(n.startswith("CAISO FMM (settlement leg): ") and " names" in n for n in notes)       # all its nodes are panel nodes
    other = fmm.assign(node="Z")                                            # a series that lacks the panel's node X
    note = _one(collect_coverage(raw, panel, other, nyi).attrs["notes"], "CAISO FMM (settlement leg): 1 name")
    assert note == "CAISO FMM (settlement leg): 1 name, 1 of them not panel nodes (Z); 1 panel node not in it (X)."


# --- duplicates ---------------------------------------------------------------------------------

def _dups(**rows):
    """raw.attrs["duplicates_dropped_where"]: market -> (rows, distinct hours, nodes, rows at 00:00 local on the 1st)."""
    out = {m: {"rows": 0, "hours": 0, "nodes": 0, "month_start_local": 0} for m in ("NYISO", "ISONE", "MISO", "CAISO")}
    for market, (n, hours, nodes, starts) in rows.items():
        out[market] = {"rows": n, "hours": hours, "nodes": nodes, "month_start_local": starts}
    return out


def test_duplicates_note_is_derived_from_the_dropped_rows_and_refuses_a_split_that_does_not_add_up():
    from es.impl_table import collect_coverage
    raw, panel, fmm, nyi = _synthetic_world()

    def note(total, where=None):
        raw.attrs["duplicates_dropped"] = total
        raw.attrs.pop("duplicates_dropped_where", None)
        if where is not None:
            raw.attrs["duplicates_dropped_where"] = where
        return _one(collect_coverage(raw, panel, fmm, nyi).attrs["notes"], "Duplicates")

    # all NYISO and every row at 00:00 local on the 1st: the month-boundary claim is derived
    n = note(390, _dups(NYISO=(390, 39, 10, 390)))
    assert ("390 duplicate rows, all at NYISO month boundaries (each at 00:00 local on the 1st of a month: "
            "39 month starts, 10 nodes)") in n
    assert "NYISO 390, ISONE 0, MISO 0, CAISO 0" in n and "first copy is kept" in n
    # NYISO only, but not every row at a month start: no boundary claim
    n = note(390, _dups(NYISO=(390, 40, 10, 380)))
    assert "all at NYISO month boundaries" not in n and "380 of them at 00:00 local on the 1st" in n
    # a split over two markets: no NYISO-only claim
    n = note(390, _dups(NYISO=(300, 30, 10, 300), ISONE=(90, 10, 9, 90)))
    assert "all at NYISO" not in n and "NYISO 300, ISONE 90" in n and "all at 00:00 local on the 1st" in n
    # nothing examined: the total, and the earlier finding is cited as a finding, not claimed for this run
    n = note(390)
    assert "390 duplicate rows" in n and "not examined" in n and "a review of the earlier version of this work found them at NYISO month boundaries" in n
    assert "each at 00:00" not in n
    # a split that does not add up to the loader's count is refused
    with pytest.raises(ValueError, match="add up"):
        note(391, _dups(NYISO=(390, 39, 10, 390)))
    assert "no duplicate rows" in note(0, _dups())


def _month_start_frame():
    """The frame a loader would dedup: NYISO node A and B duplicated at 00:00 EST on 1 Dec, one CAISO duplicate at 01:00 PST."""
    rows = [("NYISO", "A", "2023-12-01 05:00"), ("NYISO", "A", "2023-12-01 05:00"), ("NYISO", "A", "2023-12-01 06:00"),
            ("NYISO", "B", "2023-12-01 05:00"), ("NYISO", "B", "2023-12-01 05:00"), ("NYISO", "B", "2023-12-01 06:00"),
            ("CAISO", "X", "2023-12-01 09:00"), ("CAISO", "X", "2023-12-01 09:00")]
    return pd.DataFrame(rows, columns=["market", "node", "timestamp_utc"]).assign(
        timestamp_utc=lambda f: pd.to_datetime(f["timestamp_utc"], utc=True), da_lmp=1.0, rt_lmp=[1.0, 2.0] * 4)


def _fake_loader(pre, data):
    """A stand-in for es.data.load_dev_raw with its structure: node-map restriction, then drop_duplicates, then the count."""
    def load(dirs=None, markets=None):
        frame = data.restrict_to_node_map(pre.copy())
        out = frame.drop_duplicates(["market", "node", "timestamp_utc"], keep="first").reset_index(drop=True)
        out.attrs["duplicates_dropped"] = len(frame) - len(out)
        return out
    return load


def test_load_with_dropped_returns_the_rows_the_loader_dropped(monkeypatch):
    import es.data as data
    from es.impl_table import load_dev_raw_with_dropped, summarize_dropped
    identity = lambda frame: frame                                       # noqa: E731
    monkeypatch.setattr(data, "restrict_to_node_map", identity)
    monkeypatch.setattr(data, "load_dev_raw", _fake_loader(_month_start_frame(), data))
    raw, dropped = load_dev_raw_with_dropped()
    assert len(raw) == 5 and raw.attrs["duplicates_dropped"] == 2 + 1 == len(dropped)
    assert data.restrict_to_node_map is identity                         # the tap is undone
    where = summarize_dropped(dropped)
    assert where["NYISO"] == {"rows": 2, "hours": 1, "nodes": 2, "month_start_local": 2, "month_boundaries": 1}
    assert where["CAISO"] == {"rows": 1, "hours": 1, "nodes": 1, "month_start_local": 0, "month_boundaries": 0}   # 01:00 PST
    assert where["ISONE"]["rows"] == where["MISO"]["rows"] == 0

    def never_calls_the_node_map(dirs=None, markets=None):
        return raw
    monkeypatch.setattr(data, "load_dev_raw", never_calls_the_node_map)
    with pytest.raises(RuntimeError, match="restrict_to_node_map"):
        load_dev_raw_with_dropped()

    def miscounts(dirs=None, markets=None):
        out = _fake_loader(_month_start_frame(), data)()
        out.attrs["duplicates_dropped"] = 99
        return out
    monkeypatch.setattr(data, "load_dev_raw", miscounts)
    with pytest.raises(RuntimeError, match="99"):
        load_dev_raw_with_dropped()


# --- the files written --------------------------------------------------------------------------

def test_main_with_data_keeps_the_json_and_csv_contract(tmp_path, monkeypatch):
    import json
    import es.data
    import es.impl_table as it
    import es.settlement
    raw, panel, fmm, nyi = _synthetic_world()
    month_starts = pd.DataFrame({"market": "NYISO", "node": ["A", "B"], "timestamp_utc": pd.to_datetime(["2023-12-01 05:00"] * 2, utc=True)})
    pre = pd.concat([panel, month_starts, month_starts], ignore_index=True)           # the two month-start rows are duplicated
    monkeypatch.setattr(it, "PAPER_OUT_DIR", tmp_path)
    monkeypatch.setattr(it, "ensure_paper_dirs", lambda: None)
    identity = lambda frame: frame                                                     # noqa: E731
    monkeypatch.setattr(es.data, "restrict_to_node_map", identity)
    monkeypatch.setattr(es.data, "load_dev_raw", _fake_loader(pre, es.data))
    monkeypatch.setattr(es.data, "build_panel", lambda r: panel)
    monkeypatch.setattr(es.settlement, "load_caiso_fmm", lambda root, dev_only=True: fmm)
    monkeypatch.setattr(es.settlement, "load_nyiso_integrated", lambda root, dev_only=True: nyi)
    assert it.main(["--with-data"]) == 0
    assert es.data.restrict_to_node_map is identity
    info = json.loads((tmp_path / "implementation.json").read_text())
    assert {"xgboost", "logistic", "scaler", "windows", "blocks", "rules", "versions", "coverage", "coverage_notes",
            "coverage_duplicates"} <= set(info)
    dup = info["coverage_duplicates"]
    assert (dup["rows"], dup["at_month_start"], dup["month_boundaries"], dup["nodes"], dup["markets"]) == (2, 2, 1, 2, ["NYISO"])
    assert dup["by_market"]["NYISO"] == {"rows": 2, "hours": 1, "nodes": 2, "month_start_local": 2, "month_boundaries": 1}
    assert dup["by_market"]["CAISO"]["rows"] == 0
    assert isinstance(info["coverage"], list) and {"series", "first_local_date"} <= set(info["coverage"][0])   # Task 4 reads records
    assert all(isinstance(n, str) for n in info["coverage_notes"])
    assert any(n.startswith("Duplicates: 2 duplicate rows, all at NYISO month boundaries (each at 00:00 local on the 1st")
               for n in info["coverage_notes"])
    assert list(pd.read_csv(tmp_path / "coverage.csv").columns) == [
        "series", "nodes", "hourly_rows", "first_local_date", "last_local_date", "first_utc", "last_utc", "duplicates_dropped_total"]


def test_a_static_run_keeps_the_stored_coverage(tmp_path, monkeypatch):
    import json
    import es.impl_table as it
    monkeypatch.setattr(it, "PAPER_OUT_DIR", tmp_path)
    monkeypatch.setattr(it, "ensure_paper_dirs", lambda: None)
    out = tmp_path / "implementation.json"
    out.write_text(json.dumps({"xgboost": {"stale": 1}, "coverage": [{"series": "NYISO"}], "coverage_notes": ["kept"],
                               "coverage_duplicates": {"rows": 390}, "other": 1}))
    assert it.main([]) == 0
    info = json.loads(out.read_text())
    assert info["coverage"] == [{"series": "NYISO"}] and info["coverage_notes"] == ["kept"]      # the data part survives
    assert info["coverage_duplicates"] == {"rows": 390}
    assert "other" not in info and info["xgboost"] == json.loads(json.dumps(it.xgb_params(), default=str))   # the static part is fresh
    assert list(info)[-3:] == ["coverage", "coverage_notes", "coverage_duplicates"]
    out.unlink()                                                         # with no earlier file there is nothing to keep
    assert it.main([]) == 0
    assert set(json.loads(out.read_text())) == {"xgboost", "logistic", "scaler", "windows", "blocks", "rules", "versions"}


def test_the_stored_static_part_equals_a_fresh_collect_static():
    """A drift alarm: if the code changes and implementation.json is not regenerated, this fails. The line numbers of
    citations are masked (an edit elsewhere in a cited file moves them without changing what the table says); the file
    paths and every value are compared."""
    import json
    from es.impl_table import PAPER_OUT_DIR, static_drift
    stored_path = PAPER_OUT_DIR / "implementation.json"
    if not stored_path.exists():
        pytest.skip("outputs/es/paper/implementation.json has not been generated (paper/build.sh impl)")
    drift = static_drift(json.loads(stored_path.read_text()))
    assert drift == [], f"the stored implementation.json differs from the code at {drift}: rerun paper/build.sh impl --with-data"


def test_the_drift_alarm_ignores_line_numbers_but_not_values_or_paths():
    import copy
    from es.impl_table import mask_line_references, static_drift
    fresh = {"xgboost": {"n_estimators": 300,
                         "scale_pos_weight": "n_negative / n_positive (src/transfer.py:205); every other parameter at the default"},
             "rules": {"P5_clip": "clipped to [0.001, 0.5] (es.policies.p5_self_correcting, es/policies.py:156)",
                       "span": "see es/run_legacy.py:526-533 and v2/ingest_nyiso.py:104", "n": [1, 2], "times": "01:00 and 02:00"}}
    # only the digits after '.py:' go (a range whole); other colons and every number stay
    masked = mask_line_references(fresh)
    assert masked["rules"]["span"] == "see es/run_legacy.py:# and v2/ingest_nyiso.py:#"
    assert masked["rules"]["times"] == "01:00 and 02:00" and masked["xgboost"]["n_estimators"] == 300
    assert masked["rules"]["P5_clip"] == "clipped to [0.001, 0.5] (es.policies.p5_self_correcting, es/policies.py:#)"
    # a line-number change alone passes
    shifted = copy.deepcopy(fresh)
    shifted["xgboost"]["scale_pos_weight"] = shifted["xgboost"]["scale_pos_weight"].replace(":205", ":230")
    shifted["rules"]["P5_clip"] = shifted["rules"]["P5_clip"].replace(":156", ":160")
    shifted["rules"]["span"] = "see es/run_legacy.py:530-537 and v2/ingest_nyiso.py:99"
    assert static_drift(shifted, fresh) == []
    assert static_drift({**shifted, "coverage": [{"series": "NYISO"}], "coverage_notes": ["x"]}, fresh) == []   # the data part is not compared
    # a value change fails, a number or a number inside a citation sentence
    changed = copy.deepcopy(shifted)
    changed["xgboost"]["n_estimators"] = 301
    assert static_drift(changed, fresh) == ["xgboost.n_estimators"]
    changed = copy.deepcopy(shifted)
    changed["rules"]["P5_clip"] = changed["rules"]["P5_clip"].replace("0.001", "0.01")
    assert static_drift(changed, fresh) == ["rules.P5_clip"]
    # a different file in a citation fails; so do a missing key and a changed list entry
    changed = copy.deepcopy(shifted)
    changed["xgboost"]["scale_pos_weight"] = changed["xgboost"]["scale_pos_weight"].replace("src/transfer.py", "src/other.py")
    assert static_drift(changed, fresh) == ["xgboost.scale_pos_weight"]
    assert static_drift({k: v for k, v in shifted.items() if k != "rules"}, fresh) == ["rules"]
    changed = copy.deepcopy(shifted)
    changed["rules"]["n"] = [1, 3]
    assert static_drift(changed, fresh) == ["rules.n.1"]
    changed["rules"]["n"] = [1, 2, 3]
    assert static_drift(changed, fresh) == ["rules.n"]


# ---------------------------------------------------------------------------------------------
# Controller Ruling 75: Rulings 68, 69, 73 and 83 in the table, the re-review's minors (every guard fired, the stress
# text in words, the merge keys) and the structured duplicates field.
# ---------------------------------------------------------------------------------------------

def test_strata_entries_follow_ruling_68(static_rules):
    import inspect
    from es import impl_table
    from es.measures import STRATA_DECILE, STRATA_PRIMARY
    assert static_rules["strata_primary"] == list(STRATA_PRIMARY) == ["market", "node", "settle_hour", "wclass", "month"]
    assert static_rules["strata_sensitivity"] == list(STRATA_DECILE) == ["market", "node", "cal_decile"]
    assert "STRATA_MONTH" not in inspect.getsource(impl_table)               # the alias is not read


def _rotation_window(days):
    import numpy as np
    from es.bootstrap import RowCube
    n = 2 * days                                                              # one stratum, two hourly cells a day
    rng = np.random.default_rng(days)
    return RowCube.from_arrays(np.repeat(np.arange(days), 2), days, np.tile([0, 1], days), np.zeros(n, dtype=int),
                               rng.normal(size=n), (np.arange(n) % 3 == 0).astype(float))


def test_bootstrap_entry_states_the_full_cyclic_reference_set(static_rules):
    from es.bootstrap import rotation_test
    b = static_rules["bootstrap"]
    assert b["rotation_min_shift_days"] == 1                                  # Ruling 69: read from rotation_test's default
    text = b["rotation_reference_set"]
    assert "the H-E reference set is the full cyclic group" in text and "every shift 0, 1, ..., D-1" in text
    assert "n_null = (product of the windows' D) - 1" in text
    # what the code does, probed independently: one window of D days has D - 1 other cells, two windows D1 x D2 - 1
    one = rotation_test([_rotation_window(6)])
    assert one["n_null"] == 5 and abs(one["p"] * 6 - round(one["p"] * 6)) < 1e-9          # p is k / D
    assert rotation_test([_rotation_window(4), _rotation_window(6)])["n_null"] == 23
    cites = _cited(text)
    assert [path for path, _, _ in cites] == ["es/bootstrap.py"] and cites[0][2].lstrip().startswith("def rotation_test(")


def test_the_reference_set_probe_refuses_a_restricted_shift_set(monkeypatch):
    from es import bootstrap
    from es.impl_table import rotation_reference_set
    real = bootstrap.rotation_test
    monkeypatch.setattr(bootstrap, "rotation_test", lambda cubes: real(cubes, min_shift=2))    # {0} u {2, ..., D - 2}: no group
    with pytest.raises(RuntimeError, match="full cyclic group"):
        rotation_reference_set()


# --- the domain classifier, the placebo columns and the documented level (Rulings 73 and 83) -------

def test_the_domain_classifier_probe_reads_the_splitter_and_refuses_what_the_text_would_misstate():
    import copy
    import numpy as np
    from es.impl_table import _check_domain_record, _run_recorded_domain_classifier
    seen, meta = _run_recorded_domain_classifier()
    assert _check_domain_record(seen, meta) == {
        "splitter": "StratifiedGroupKFold", "n_splits": 5, "classifier": {"max_iter": 2000, "class_weight": "balanced"}}
    assert len(seen["folds"]) == 5 and len(seen["classifiers"]) == 5 and len(np.unique(seen["groups"])) == 44

    def refused(mutate, match):
        broken = copy.deepcopy(seen)
        mutate(broken)
        with pytest.raises(RuntimeError, match=match):
            _check_domain_record(broken, meta)

    def one_class_test_part(s):
        train, _ = s["folds"][0]
        s["folds"][0] = (train, np.flatnonzero(s["y"] == 0)[:6])

    def shared_group(s):
        train, test = s["folds"][0]
        s["folds"][0] = (train, np.r_[test, train[:4]])

    refused(lambda s: s.pop("folds"), "never split the data")
    refused(lambda s: s.update(splitter_args=(5,)), "positional")
    refused(lambda s: s.update(splitter_kwargs={"n_splits": 3, "shuffle": False}), "n_splits")
    refused(lambda s: s.update(splitter_kwargs={"n_splits": 5, "shuffle": True}), "shuffles")
    refused(lambda s: s.update(folds=s["folds"][:3]), "3 folds")
    refused(lambda s: s.update(groups=s["groups"] // 7), "one group per local settlement day")
    refused(one_class_test_part, "class-stratified")
    refused(shared_group, "blocked by day")
    refused(lambda s: s.update(classifiers=s["classifiers"][:3]), "3 classifiers")
    refused(lambda s: s["classifiers"].__setitem__(1, ((), {"max_iter": 5})), "identical")


def test_the_probe_refuses_a_module_without_exactly_one_splitter(monkeypatch):
    from sklearn.model_selection import KFold
    from es import run_monitors
    from es.impl_table import _splitter_name
    assert _splitter_name() == "StratifiedGroupKFold"
    with monkeypatch.context() as m:
        m.setattr(run_monitors, "KFold", KFold, raising=False)               # a second splitter
        with pytest.raises(RuntimeError, match="expected exactly one"):
            _splitter_name()
    with monkeypatch.context() as m:
        m.delattr(run_monitors, "StratifiedGroupKFold")                      # none
        with pytest.raises(RuntimeError, match="expected exactly one"):
            _splitter_name()


def test_the_probe_refuses_cv_groups_that_let_a_calibration_and_a_deployment_day_share_a_group(monkeypatch):
    import numpy as np
    from es import run_monitors
    from es.impl_table import _check_cv_groups
    _check_cv_groups()
    monkeypatch.setattr(run_monitors, "cv_groups", lambda ids, n_cal: np.asarray(ids))
    with pytest.raises(RuntimeError, match="share a group"):
        _check_cv_groups()


def test_the_probe_refuses_a_domain_classifier_whose_guards_do_not_fire(monkeypatch):
    from es import run_monitors
    from es.impl_table import _check_domain_guards
    _check_domain_guards()
    real = run_monitors.domain_auc

    def lenient_on(call_number):
        calls = []

        def domain_auc(cal_X, dep_X, ids):
            calls.append(1)
            return 0.5 if len(calls) == call_number else real(cal_X, dep_X, ids)
        return domain_auc

    with monkeypatch.context() as m:
        m.setattr(run_monitors, "domain_auc", lenient_on(1))                  # accepts too few day groups
        with pytest.raises(RuntimeError, match="did not refuse fewer than 5 distinct day groups"):
            _check_domain_guards()
    with monkeypatch.context() as m:
        m.setattr(run_monitors, "domain_auc", lenient_on(2))                  # accepts a layout with one deployment day
        with pytest.raises(RuntimeError, match="did not refuse a layout with one deployment day"):
            _check_domain_guards()


def test_the_probe_checks_the_placebo_split_and_its_columns(monkeypatch):
    from es import run_monitors
    from es.impl_table import _check_placebo
    assert _check_placebo() == ["W1_norm_placebo", "psi_placebo", "domain_auc_placebo"]
    real = run_monitors.placebo_split
    with monkeypatch.context() as m:
        m.setattr(run_monitors, "placebo_split", lambda cal: real(cal)[::-1])            # the roles swapped
        with pytest.raises(RuntimeError, match="placebo_split"):
            _check_placebo()
    with monkeypatch.context() as m:
        m.setattr(run_monitors, "PLACEBO_COLUMNS", {"Z_norm": "z_placebo"})              # a placebo of something that is no monitor
        with pytest.raises(RuntimeError, match="mirrors Z_norm, which is not in MONITORS"):
            _check_placebo()
    with monkeypatch.context() as m:
        m.setattr(run_monitors, "MONITORS", (*run_monitors.MONITORS, "psi_placebo"))     # a placebo that is a monitor
        with pytest.raises(RuntimeError, match="is a monitor"):
            _check_placebo()
    with monkeypatch.context() as m:
        m.setattr(run_monitors, "ROW_COLUMNS", [c for c in run_monitors.ROW_COLUMNS if c != "psi_placebo"])
        with pytest.raises(RuntimeError, match="ROW_COLUMNS"):
            _check_placebo()
    with monkeypatch.context() as m:
        m.setattr(run_monitors, "EVAL_COLUMNS", [c for c in run_monitors.EVAL_COLUMNS if c != "placebo_median"])
        with pytest.raises(RuntimeError, match="EVAL_COLUMNS"):
            _check_placebo()


def test_the_monitors_entry_quotes_no_simulated_level_and_does_not_depend_on_the_docstring(monkeypatch):
    """Ruling 93: the table states implementation facts read from the code, not the null AUC levels that a docstring quotes, and
    the build neither reads nor pins that docstring."""
    from es import run_monitors
    from es.impl_table import monitor_rule
    before = monitor_rule()
    assert "no-shift" not in before and "optimistic" not in before and not re.search(r"\d\.\d", before)
    monkeypatch.setattr(run_monitors.domain_auc, "__doc__", "Out-of-fold AUROC of a logistic classifier.")    # says nothing of a level
    assert monitor_rule() == before
    monkeypatch.setattr(run_monitors.domain_auc, "__doc__", None)                                           # or has no docstring
    assert monitor_rule() == before


def test_monitor_rule_refuses_monitors_without_an_early_rate(monkeypatch):
    from es import run_monitors
    from es.impl_table import monitor_rule
    monkeypatch.setattr(run_monitors, "MONITORS", ("W1_norm", "PSI", "domain_auc"))
    with pytest.raises(RuntimeError, match="has no early_rate"):
        monitor_rule()


# --- the stress-fee text in words, and the merge keys ------------------------------------------

def test_stress_fee_text_reads_the_168_and_the_48_from_the_code(monkeypatch):
    import re
    from es import scores
    from es.impl_table import stress_fee_rule
    window, lag = re.fullmatch(r"dart_spread_roll_abs_mean_(\d+)h_shift_(\d+)h", scores._ABS_MEAN_SOURCE).groups()
    assert (window, lag) == ("168", "48")
    assert f"the trailing {window}-hour mean of |DART|, lagged {lag} h, at its first hour" in stress_fee_rule()
    for source, match in (("abs_mean_168h", "not of the form"),
                          ("dart_spread_roll_abs_mean_336h_shift_48h", "does not match"),
                          ("dart_spread_roll_abs_mean_168h_shift_24h", "does not match")):
        with monkeypatch.context() as m:
            m.setattr(scores, "_ABS_MEAN_SOURCE", source)
            with pytest.raises(RuntimeError, match=match):
                stress_fee_rule()


def test_stress_fee_rule_refuses_a_pipeline_that_does_not_double_exactly_the_stressed_days(monkeypatch):
    from es import positions
    from es.impl_table import stress_fee_rule
    real = positions.attach_pnl
    with monkeypatch.context() as m:                                           # never doubles
        m.setattr(positions, "attach_pnl", lambda rows, position, direction=None, fees="calibrated", stress=None:
                  real(rows, position, direction, fees, stress=None))
        with pytest.raises(RuntimeError, match="no longer doubles"):
            stress_fee_rule()
    with monkeypatch.context() as m:                                           # doubles every day, whatever its first hour
        m.setattr(positions, "attach_pnl", lambda rows, position, direction=None, fees="calibrated", stress=None:
                  real(rows.assign(abs_mean_168h=1e9), position, direction, fees, stress=stress))
        with pytest.raises(RuntimeError, match="no longer doubles"):
            stress_fee_rule()


def test_ingester_merge_keys_are_read_from_the_source_and_refused_when_unreadable(monkeypatch):
    import es.impl_table as it
    assert it._ingester_merges()[1] == "on timestamp and node (and node type for ISO-NE and MISO)"

    def stub(by_file):
        def find(where, needle):
            rel = str(where.relative_to(it.DEV_ROOT))
            return rel, 7, by_file[rel]
        return find

    merge = lambda keys, how='how="inner"': f'panel = da.merge(rt, on=[{keys}], {how})'              # noqa: E731
    plain = merge('"timestamp_utc", "node"')
    ok = {f"v2/ingest_{m}.py": plain for m in ("nyiso", "isone", "miso", "caiso")}
    with monkeypatch.context() as m:
        m.setattr(it, "_find", stub(ok))
        assert it._ingester_merges() == ("v2/ingest_nyiso.py:7, v2/ingest_isone.py:7, v2/ingest_miso.py:7, v2/ingest_caiso.py:7",
                                         "on timestamp and node")
        m.setattr(it, "_find", stub({**ok, "v2/ingest_isone.py": merge('"timestamp_utc", "node", "node_type"'),
                                     "v2/ingest_caiso.py": merge('"timestamp_utc", "node", "region"')}))
        assert it._ingester_merges()[1] == "on timestamp and node (and node type for ISO-NE; region for CAISO)"
    with monkeypatch.context() as m:
        m.setattr(it, "_find", stub({**ok, "v2/ingest_nyiso.py": 'panel = da.merge(rt, on=["timestamp_utc", "node"])'}))
        with pytest.raises(RuntimeError, match="no longer inner-merges"):
            it._ingester_merges()
    with monkeypatch.context() as m:
        m.setattr(it, "_find", stub({**ok, "v2/ingest_nyiso.py": 'panel = da.merge(rt, left_on="a", right_on="b", how="inner")'}))
        with pytest.raises(RuntimeError, match="cannot read"):
            it._ingester_merges()


# --- the structured duplicates field (Ruling 76a) ----------------------------------------------

def test_duplicates_field_is_derived_like_the_note():
    import numpy as np
    from es.impl_table import duplicates_field
    starts = pd.date_range("2023-02-01", periods=39, freq="MS", tz="America/New_York").tz_convert("UTC")
    dropped = pd.DataFrame([("NYISO", f"N{i}", t) for i in range(10) for t in starts], columns=["market", "node", "timestamp_utc"])
    f = duplicates_field(dropped)
    assert (f["rows"], f["month_boundaries"], f["nodes"], f["at_month_start"], f["markets"]) == (390, 39, 10, 390, ["NYISO"])
    assert f["by_market"]["NYISO"] == {"rows": 390, "hours": 39, "nodes": 10, "month_start_local": 390, "month_boundaries": 39}
    assert all(f["by_market"][m]["rows"] == 0 for m in ("ISONE", "MISO", "CAISO"))
    # rows that are not at a month start do not count as boundaries, and the flat numbers are over all markets
    mixed = pd.DataFrame({"market": ["NYISO"] * 4 + ["CAISO"], "node": ["A", "A", "B", "B", "X"],
                          "timestamp_utc": pd.to_datetime(["2023-12-01 05:00", "2024-01-01 05:00"] * 2 + ["2023-12-01 09:00"], utc=True)})
    f = duplicates_field(mixed)
    assert (f["rows"], f["at_month_start"], f["month_boundaries"], f["nodes"], f["markets"]) == (5, 4, 2, 3, ["NYISO", "CAISO"])
    assert f["by_market"]["CAISO"] == {"rows": 1, "hours": 1, "nodes": 1, "month_start_local": 0, "month_boundaries": 0}
    # nothing dropped
    none = duplicates_field(pd.DataFrame({"market": [], "node": [], "timestamp_utc": pd.to_datetime([], utc=True)}))
    assert (none["rows"], none["at_month_start"], none["month_boundaries"], none["nodes"], none["markets"]) == (0, 0, 0, 0, [])
    assert np.isscalar(f["rows"]) and isinstance(f["rows"], int)                  # plain ints: the JSON writer needs no default


def test_collect_coverage_carries_the_duplicates_field_and_refuses_one_that_disagrees_with_the_total():
    from es.impl_table import collect_coverage
    raw, panel, fmm, nyi = _synthetic_world()
    raw.attrs["duplicates_dropped"] = 390
    field = {"rows": 390, "at_month_start": 390, "month_boundaries": 39, "nodes": 10, "markets": ["NYISO"], "by_market": {"NYISO": {}}}
    raw.attrs["duplicates_dropped_field"] = field
    assert collect_coverage(raw, panel, fmm, nyi).attrs["duplicates"] == field
    del raw.attrs["duplicates_dropped_field"]                                   # not examined: null values, never a sentinel
    assert collect_coverage(raw, panel, fmm, nyi).attrs["duplicates"] == {
        "rows": 390, "at_month_start": None, "month_boundaries": None, "nodes": None, "markets": None, "by_market": None}
    raw.attrs["duplicates_dropped_field"] = {**field, "rows": 391}
    with pytest.raises(ValueError, match="does not match"):
        collect_coverage(raw, panel, fmm, nyi)


# --- the guards and branches the re-review found unexercised -----------------------------------

def test_calendar_guards_refuse_what_the_text_would_misstate(monkeypatch):
    import es.impl_table as it
    from v2.panel_prep import add_features
    with pytest.raises(RuntimeError, match="not a function of one clock field"):
        it._flag_keys(pd.Series([0, 1, 0, 0], name="calendar_is_peak_hour"), [1, 1, 2, 2])
    assert it._flag_keys(pd.Series([0, 1, 1, 0], name="flag"), [5, 6, 6, 5]) == [6]
    ts = pd.date_range("2023-01-01", periods=72, freq="h", tz="UTC")
    cal = add_features(pd.DataFrame({"market": "NYISO", "node": "n", "timestamp_utc": ts, "da_lmp": 1.0, "rt_lmp": 1.0}))
    hours = ts.tz_convert("America/New_York").hour
    assert it._period(cal, "hour", hours) == 24.0
    with pytest.raises(RuntimeError, match="calendar_hour is not the sin and cos"):
        it._period(cal.assign(calendar_hour_sin=cal["calendar_hour_sin"] * 0.5), "hour", hours)

    def odd_peak(panel):                                                       # peak hours 7 and 9 only: not a range
        out = add_features(panel)
        out["calendar_is_peak_hour"] = out["timestamp_utc"].dt.tz_convert("America/New_York").dt.hour.isin([7, 9]).astype(int)
        return out

    with monkeypatch.context() as m:
        m.setattr(it, "add_features", odd_peak)
        with pytest.raises(RuntimeError, match="not a contiguous range"):
            it.calendar_rule()
    with monkeypatch.context() as m:
        m.setattr(it, "MARKET_TZ", {**it.MARKET_TZ, "NYISO": "UTC"})
        with pytest.raises(RuntimeError, match="MARKET_TZ differs"):
            it.calendar_rule()


def test_seed_scope_guards_refuse_a_plan_that_the_sentence_would_misdescribe(monkeypatch):
    import es.impl_table as it
    from es import run_legacy
    real = run_legacy.run_plan
    with monkeypatch.context() as m:
        m.setattr(it, "SEEDS", (0, 1, 7, 13, 99))
        with pytest.raises(RuntimeError, match="es.paper_pins.SEEDS"):
            it.seed_scope()

    def one_run_short():
        plan = list(real())
        del plan[next(i for i, run in enumerate(plan) if run[3] != 42)]
        yield from plan

    with monkeypatch.context() as m:
        m.setattr(run_legacy, "run_plan", one_run_short)
        with pytest.raises(RuntimeError, match="full grid"):
            it.seed_scope()


def test_window_conventions_refuse_an_analysis_that_mixes_conventions(monkeypatch):
    import es.impl_table as it
    utc, local = "UTC half-open [start, end)", "local settlement dates [start, end]"
    row = lambda calib: [{"group": "legacy", "id": "X", "train": {"convention": utc}, "calib": {"convention": calib},   # noqa: E731
                          "deploy": {"convention": utc}, "label_train": {"convention": utc}}]
    monkeypatch.setattr(it, "windows", lambda: row(local))
    with pytest.raises(RuntimeError, match="mixes window conventions"):
        it.window_conventions()
    monkeypatch.setattr(it, "windows", lambda: row(utc))
    assert it.window_conventions().startswith(f"{utc}: legacy (1 analysis); the two conventions are never pooled")


def test_versions_report_a_package_that_is_not_installed(monkeypatch):
    import importlib.metadata as md
    import es.impl_table as it
    real = md.version

    def version(name):
        if name == "statsmodels":
            raise md.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(it.md, "version", version)
    v = it.versions()
    assert v["statsmodels"] == "not installed" and v["numpy"] == real("numpy")


def test_series_gaps_refuse_timestamps_off_the_hour_and_runs_of_no_hours():
    import es.impl_table as it
    frame = pd.DataFrame({"node": ["A", "A"], "timestamp_utc": pd.to_datetime(["2023-11-04 00:00", "2023-11-04 01:30"], utc=True)})
    with pytest.raises(ValueError, match="not on the hour"):
        it._series_gaps(frame, "America/New_York")
    assert it._runs(pd.DatetimeIndex([], tz="UTC")) == []


def test_coverage_notes_cover_two_fall_back_days_and_affected_nodes_missing_from_the_settlement_series():
    from es.impl_table import collect_coverage
    # a year with two fall-back days, both repeated hours dropped in both nodes: the plural sentence the real JSON prints
    fall = FALL_BACK_NY + ["2024-11-03 05:00", "2024-11-03 06:00"]
    year = _hourly("NYISO", ["A", "B"], "2023-11-04 00:00", "2024-11-04 23:00", drop={"*": fall})
    raw = year.copy()
    raw.attrs["duplicates_dropped"] = 0
    fb = _one(collect_coverage(raw, year, None, None).attrs["notes"], "NYISO panel: both rows")
    assert "absent on each of the 2 fall-back days (2023-11-05, 2024-11-03), 4 hours per node" in fb
    # both nodes lack 2023-11-06 10:00Z, but the settlement series has only one of them: say so, in the singular
    start, end = "2023-11-04 00:00", "2023-11-07 23:00"
    ny = _hourly("NYISO", ["A", "B"], start, end, drop={"*": FALL_BACK_NY + ["2023-11-06 10:00"]})
    settle = _hourly("NYISO", ["B", "C"], start, end, drop={"*": FALL_BACK_NY}).drop(columns="market")
    raw = ny.copy()
    raw.attrs["duplicates_dropped"] = 0
    gap = _one(collect_coverage(raw, ny, None, settle).attrs["notes"], "NYISO panel: 1 hour absent")
    assert "The NYISO integrated series has every one of these node-hours. (1 affected node not in that series.)" in gap
    # no shared node at all: no sentence about the settlement series
    other = _hourly("NYISO", ["C"], start, end).drop(columns="market")
    gap = _one(collect_coverage(raw, ny, None, other).attrs["notes"], "NYISO panel: 1 hour absent")
    assert "integrated series" not in gap


def test_running_the_module_as_a_script_writes_the_static_file(tmp_path, monkeypatch):
    """`python -m es.impl_table` is what paper/build.sh impl runs; its output directory is redirected here."""
    import json
    import runpy
    import sys
    import warnings
    import es.paper_paths
    monkeypatch.setattr(es.paper_paths, "PAPER_OUT_DIR", tmp_path)
    monkeypatch.setattr(es.paper_paths, "ensure_paper_dirs", lambda: None)
    monkeypatch.setattr(sys, "argv", ["es.impl_table"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                   # runpy: the module is already imported
        with pytest.raises(SystemExit) as stopped:
            runpy.run_module("es.impl_table", run_name="__main__")
    assert stopped.value.code == 0
    assert set(json.loads((tmp_path / "implementation.json").read_text())) == {
        "xgboost", "logistic", "scaler", "windows", "blocks", "rules", "versions"}


# ---------------------------------------------------------------------------------------------
# G3, item 14: the Online Resource prints these strings, and its reader has no design specification, no forensic report and no analysis log
# ---------------------------------------------------------------------------------------------
INTERNAL = re.compile(r"Ruling|FORENSICS|Seat |Opus|controller|spec v|spec \d|specification|\bI\d\b")


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def test_no_printed_rule_names_a_document_the_reader_does_not_have(static_rules):
    found = [s for s in _strings(static_rules) if INTERNAL.search(s)]
    assert not found, found


def test_no_coverage_note_names_a_document_the_reader_does_not_have():
    from es.impl_table import collect_coverage
    raw, panel, fmm, nyi = _synthetic_world()
    assert not [n for n in collect_coverage(raw, panel, fmm, nyi).attrs["notes"] if INTERNAL.search(n)]
    for total, where in ((390, None), (390, _dups(NYISO=(390, 39, 10, 390))), (390, _dups(NYISO=(390, 40, 10, 380))),
                         (390, _dups(NYISO=(300, 30, 10, 300), ISONE=(90, 10, 9, 90)))):
        raw.attrs["duplicates_dropped"] = total
        raw.attrs.pop("duplicates_dropped_where", None)
        if where is not None:
            raw.attrs["duplicates_dropped_where"] = where
        notes = collect_coverage(raw, panel, fmm, nyi).attrs["notes"]
        assert not [n for n in notes if INTERNAL.search(n)], notes


#: Plan 2C task T2b. The strings of implementation.json (a saved output that no task changes) and of coverage.csv still point at sections by the number that
#: they had when they were written ("§3.3", "§6.1.1"). The tables print none of these numbers: es.paper_tables.SECTION_POINTERS writes each pointer as the
#: title of the heading that it means, \\mainref{key} for a section of the paper (the tables are in Online Resource 1) and \\ref for a section of Online Resource 1.
#: What each pointer MEANS is fixed here, as a title; the numbering of the manuscript is not read, so that a section may be renumbered freely.
POINTER_MEANS = {"3.3": "Feature blocks", "3.4": "Analyses and windows", "3.7.1": "Positions and fees", "3.7.3": "Cutoff policies and comparators",
                 "6.1.1": "Windows, exclusions and daylight saving"}


def _paper_dir():
    from pathlib import Path
    paper = Path(__file__).resolve().parents[3] / "paper"
    if not (paper / "sections").exists():
        pytest.skip("the manuscript folder is not next to dart_dev")
    return paper


def _headings_with_labels(folder):
    """{title: label or None} of the \\section, \\subsection and \\subsubsection headings (starred or not) of the files of paper/<folder>/."""
    out = {}
    for path in sorted((_paper_dir() / folder).glob("*.tex")):
        text = re.sub(r"(?<!\\)%.*$", "", path.read_text(encoding="utf-8"), flags=re.M)
        for m in re.finditer(r"\\(?:sub){0,2}section\*?\s*\{([^{}]*)\}(\s*\\label\{([^{}]*)\})?", text):
            out[" ".join(m.group(1).split())] = m.group(3)
    return out


def _main_keys():
    """{key: title} of the \\mainkey lines of paper/crossref.tex."""
    text = re.sub(r"(?<!\\)%.*$", "", (_paper_dir() / "crossref.tex").read_text(encoding="utf-8"), flags=re.M)
    return {m.group(1): " ".join(m.group(2).split()) for m in re.finditer(r"^\\mainkey\{([^{}]*)\}\{(.*)\}\s*$", text, flags=re.M)}


def _what_a_pointer_prints(entry):
    """The title of the heading that an entry of SECTION_POINTERS names: \\mainref{key} through the key table (the title must be a heading of the paper),
    \\ref{label} through the label of an ESM heading."""
    m = re.fullmatch(r"\\mainref\{([^{}]*)\}", entry)
    if m:
        title = _main_keys()[m.group(1)]
        assert title in _headings_with_labels("sections"), f"{entry}: no heading of the paper reads {title!r}"
        return title
    m = re.fullmatch(r"Section~\\ref\{([^{}]*)\}", entry)
    assert m, f"{entry!r} is neither \\mainref{{key}} nor Section~\\ref{{label}}"
    titles = [title for title, label in _headings_with_labels("esm").items() if label == m.group(1)]
    assert len(titles) == 1, f"{entry}: {len(titles)} headings of Online Resource 1 carry the label {m.group(1)}"
    return titles[0]


def test_the_section_pointers_of_the_rules_name_the_headings_that_they_mean(static_rules):
    """The strings point at sections by number (§3.3, §6.1.1). Each number is mapped to the title of the heading it meant, and this test checks the map
    against the headings: every pointer of the strings has an entry, every entry is used, and the entry names (through crossref.tex or a label of
    Online Resource 1) the heading with the title that the pointer means. Nothing here reads the numbering of the manuscript."""
    from es.paper_tables import SECTION_POINTERS
    pointers = set(re.findall(r"§(\d+(?:\.\d+)*)", " ".join(_strings(static_rules))))
    assert pointers == set(POINTER_MEANS) == set(SECTION_POINTERS), (pointers, set(SECTION_POINTERS))      # every pointer is mapped and known here, and all are used
    for number, meant in POINTER_MEANS.items():
        assert _what_a_pointer_prints(SECTION_POINTERS[number]) == meant, number


def test_every_printed_section_pointer_names_a_section_that_exists():
    """Formerly an expected failure (strict xfail, plan 2C task T2a), kept visible until the pointers were fixed: §6.1.1 of the implementation table and
    of two footnotes of the coverage table named a section that the paper no longer has, and §3.3 to §3.7.3 held only while Section 3 kept its numbering.
    Task T2b maps them to titles. This builds the two tables from the saved outputs, as ./build.sh tables does, and reads what they print: no typed section
    number or section sign, and every pointer of the map that occurs names an existing heading."""
    from es.paper_macros import Sources
    from es.paper_tables import BUILDERS, SECTION_POINTERS
    src = Sources()
    printed = {name: BUILDERS[name][0](src) for name in ("tab_implementation", "tab_coverage")}
    for name, tex in printed.items():
        assert "§" not in tex and "\\S{}" not in tex, name                                        # no section sign, no typed number
        assert not re.search(r"Sections?\s*~?\s*\d", tex), name
    used = {entry for entry in SECTION_POINTERS.values() if any(entry in tex for tex in printed.values())}
    assert used == set(SECTION_POINTERS.values()), set(SECTION_POINTERS.values()) - used                  # the real outputs use every pointer of the map
    for entry in used:
        assert _what_a_pointer_prints(entry) in POINTER_MEANS.values()


def test_a_section_pointer_that_the_map_does_not_know_stops_the_table():
    """The table must not print a section number that nothing keeps right: a new pointer in the saved strings needs a line in SECTION_POINTERS."""
    from es.paper_tables import _esc
    assert _esc("set before the sealed windows were opened, §6.1.1").endswith("Section~\\ref{sec:sealed-windows}")
    assert _esc("a bare \u00a7 sign") == "a bare \\S{} sign"
    with pytest.raises(LookupError, match="9.9.9"):
        _esc("see §9.9.9")
