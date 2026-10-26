import datetime as dt
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_psi_zero_for_identical_and_positive_for_shift():
    from es.run_monitors import psi
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert psi(x, x) < 1e-6
    assert psi(x, x + 1.0) > 0.2


def test_domain_auc_detects_shift():
    from es.run_monitors import domain_auc
    rng = np.random.default_rng(1)
    cal, dep = rng.normal(size=(800, 3)), rng.normal(size=(800, 3)) + 1.0
    weeks = np.repeat(np.arange(32), 50)
    assert domain_auc(cal, dep, weeks) > 0.8
    assert abs(domain_auc(cal, rng.normal(size=(800, 3)), weeks) - 0.5) < 0.1


def test_cv_groups_use_each_sides_own_weeks():
    from es.run_monitors import cv_groups
    weeks = np.r_[np.repeat([1, 2], 3), np.repeat([7, 8, 9], 2)]
    assert cv_groups(weeks, 6).tolist() == [1, 1, 1, 2, 2, 2, 10007, 10007, 10008, 10008, 10009, 10009]


def test_domain_auc_rejects_misaligned_weeks():
    import pytest
    from es.run_monitors import domain_auc
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError):
        domain_auc(rng.normal(size=(10, 2)), rng.normal(size=(10, 2)), np.arange(19))


def test_evaluate_monitors_one_sided_p():
    import pandas as pd
    from es.run_monitors import evaluate_monitors
    n = 12
    base = pd.DataFrame({"analysis": "A", "block": "LEV6", "market": "M", "node": [f"n{i}" for i in range(n)],
                         "log_error": np.linspace(0, 1, n), "fail": [False] * 6 + [True] * 6,
                         "W1_norm": np.linspace(0, 1, n), "PSI": np.linspace(1, 0, n),
                         "domain_auc": np.linspace(0.5, 1, n), "early_rate": np.linspace(0.05, 0.2, n)})
    ev = evaluate_monitors(base).set_index("monitor")
    assert ev.loc["W1_norm", "spearman_p_one_sided"] < 0.01
    assert ev.loc["PSI", "spearman_p_one_sided"] > 0.99
    assert ev.loc["W1_norm", "auroc_fail"] == 1.0


# --------------------------------------------------------------------------------------------------
# Above: the brief's five tests, verbatim. Below: the pre-review supplement (controller Ruling 40) and
# its additions. All data are synthetic.
# --------------------------------------------------------------------------------------------------

@pytest.fixture
def rm():
    from es import run_monitors
    return run_monitors


@pytest.fixture(autouse=True)
def _single_threaded_blas():
    """These tests fit hundreds of tiny models. Multi-threaded BLAS only adds contention on matrices this small
    (a 20-fold slowdown when the machine is busy), so each test runs with one BLAS thread."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        yield


class _Window:
    """What monitor_rows needs of a window: mask(frame) -> boolean Series over inclusive local dates."""
    def __init__(self, start, end):
        self.start, self.end = dt.date.fromisoformat(start), dt.date.fromisoformat(end)

    def mask(self, frame):
        d = frame["settle_date"]
        return (d >= self.start) & (d <= self.end)


def _fake_analysis(calib, deploy, id="FAKE"):
    return SimpleNamespace(id=id, calib=_Window(*calib), deploy=_Window(*deploy))


# Unequal windows on different dates (calibration 3-30 March 2025, deployment 9-29 March 2026), so a
# mis-ordered or mis-aligned vector of day ids cannot give the same CV groups by accident.
FAKE = _fake_analysis(("2025-03-03", "2025-03-30"), ("2026-03-09", "2026-03-29"))
MARKET = {"N1": "NYISO", "I1": "ISONE"}
LONG = _fake_analysis(("2025-01-06", "2025-03-30"), ("2026-03-09", "2026-03-29"), id="LONG")     # 84 calibration days
LONG_SPANS = (("2025-01-06", 84), ("2026-03-09", 21))
SPANS = (("2025-03-03", 28), ("2026-03-09", 21))     # (start date, days): calibration, then deployment


def _panel(drift, seed=0, const_feature=None, spans=SPANS, hours=1, slow=None):
    """Rows every ``hours`` hours per node and span. The first span is calibration; the later spans
    carry the node's mean shift. With ``slow`` (a window in hours) every feature is a rolling mean of iid hourly
    noise, scaled to unit variance, so it moves slowly (the 168-hour case of the Ruling 83 study)."""
    from es.blocks import BLOCKS
    from es.guard import settlement_frame
    feats = sorted({f for b in BLOCKS for f in BLOCKS[b]})
    rng = np.random.default_rng(seed)
    parts = []
    for node, d in drift.items():
        for i, (start, n_days) in enumerate(spans):
            ts = pd.date_range(pd.Timestamp(start, tz="UTC") + pd.Timedelta(hours=5),
                               periods=24 * n_days // hours, freq=f"{hours}h")
            f = pd.DataFrame({"market": MARKET[node], "node": node, "timestamp_utc": ts})
            for c in feats:
                if slow:
                    x = pd.Series(rng.normal(size=len(ts) + slow)).rolling(slow).mean().to_numpy()[slow:] * np.sqrt(slow)
                else:
                    x = rng.normal(size=len(ts))
                f[c] = x + (d if i > 0 else 0.0)
            if const_feature:
                f[const_feature] = 5.0
            parts.append(f)
    p = pd.concat(parts, ignore_index=True)
    return pd.concat([p, settlement_frame(p["timestamp_utc"], p["market"]).reset_index(drop=True)], axis=1)


def _scores(panel, shift, seed=1, analysis=FAKE):
    """A fake score frame like es.scores.score_analysis: deployment scores shifted per node (a shift
    means more alerts, so R > 1)."""
    rng = np.random.default_rng(seed)
    sc = panel[["market", "node", "timestamp_utc", "settle_date"]].copy()
    cal, dep = analysis.calib.mask(panel).to_numpy(), analysis.deploy.mask(panel).to_numpy()
    sc["split"] = np.select([cal, dep], ["calib", "deploy"], default="other")
    sc["score"] = rng.normal(size=len(sc)) + np.where(dep, sc["node"].map(shift).to_numpy(float), 0.0)
    return sc


def _day_since_first(panel, analysis):
    """Local settlement days since each node's first deployment day, per panel row (NaN outside the deployment
    window), by plain date arithmetic. Day 0 is the first deployment day; days 0-13 are the 'early' days."""
    out = pd.Series(np.nan, index=panel.index)
    for _, g in panel[analysis.deploy.mask(panel).to_numpy()].groupby(["market", "node"]):
        first = min(g["settle_date"])
        out.loc[g.index] = [(d - first).days for d in g["settle_date"]]
    return out


def _drop_days(panel, node, dates):
    """The panel without the given settlement dates for one node (a data outage)."""
    return panel[~((panel["node"] == node) & panel["settle_date"].isin(dates))].reset_index(drop=True)


def _independent_outcome(scores, alpha=0.05, days=14):
    """Per node, from first principles: the calibration cutoff, the early alert rate, and R, |log10 R| and the
    failure flag on the days after the first ``days``."""
    out = {}
    for key, s in scores.groupby(["market", "node"]):
        tau = np.quantile(s.loc[s["split"] == "calib", "score"].to_numpy(float), 1 - alpha)
        dep = s[s["split"] == "deploy"]
        first = min(dep["settle_date"])
        day = np.array([(d - first).days for d in dep["settle_date"]])
        hit = dep["score"].to_numpy(float) >= tau
        n_late = int((day >= days).sum())
        R = hit[day >= days].mean() / alpha
        out[key] = {"early_rate": hit[day < days].mean(), "n_late": n_late,
                    "log_error": abs(np.log10(max(R, 1 / (2 * n_late * alpha)))), "fail": bool(R < 0.5 or R > 2)}
    return out


# ---- domain_auc: class-stratified CV blocked by local settlement day (Ruling 73) ----------------------
# Spec v3 §5.5 said "blocked by week". On the 14-day early window of Ruling 66b that leaves 2-3 deployment
# groups, a null AUC far below 0.5 (about 0.26 on AQ3's calendar) and fold crashes, so Ruling 73 blocks by
# settlement day and stratifies.

def _epoch_day(d):
    """Independent day id: days since 1970-01-01, from the standard library."""
    return (d - dt.date(1970, 1, 1)).days


def _ar1(n, p, rho, rng):
    """Persistent features (AR(1) per column), as in the Ruling 66b/73 study: neighbouring hours are alike."""
    x = np.zeros((n, p))
    e = rng.normal(size=(n, p))
    x[0] = e[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + np.sqrt(1 - rho ** 2) * e[t]
    return x


def _hourly_dates(start, n_days):
    """The local settlement date of each hourly row of ``n_days`` consecutive days from ``start`` (24 a day)."""
    first = dt.date.fromisoformat(start)
    return [first + dt.timedelta(days=k) for k in range(n_days) for _ in range(24)]


def _calendar_case(cal_start, cal_days, dep_start, rho, seed):
    """Hourly calibration rows for ``cal_days`` days and the first 14 deployment days from ``dep_start``, with 6
    features: iid when ``rho`` is 0, persistent AR(rho) otherwise. Returns (cal_X, dep_X, cal_dates, dep_dates)."""
    cal_dates, dep_dates = _hourly_dates(cal_start, cal_days), _hourly_dates(dep_start, 14)
    rng = np.random.default_rng(7000 + seed)
    make = (lambda n: _ar1(n, 6, rho, rng)) if rho else (lambda n: rng.normal(size=(n, 6)))
    return make(len(cal_dates)), make(len(dep_dates)), cal_dates, dep_dates


def _auc_on_calendar(rm, cal_start, cal_days, dep_start, rho, shift, seed):
    """domain_auc on a ``_calendar_case`` whose deployment features are shifted by ``shift`` sd."""
    cal, dep, cal_dates, dep_dates = _calendar_case(cal_start, cal_days, dep_start, rho, seed)
    return rm.domain_auc(cal, dep + shift, np.array([_epoch_day(d) for d in cal_dates + dep_dates]))


def _week_blocked_auc(rm, cal, dep, cal_dates, dep_dates):
    """The scheme Ruling 73 replaced: GroupKFold on ISO-week groups with a balanced logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    X, y = np.vstack([cal, dep]), np.r_[np.zeros(len(cal)), np.ones(len(dep))]
    weeks = np.array([d.isocalendar()[0] * 100 + d.isocalendar()[1] for d in cal_dates + dep_dates])
    pred = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=5).split(X, y, rm.cv_groups(weeks, len(cal))):
        pred[test] = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    return roc_auc_score(y, pred)


def test_domain_auc_uses_class_stratified_day_blocked_balanced_cv(rm, monkeypatch):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    seen = {"splitter": [], "groups": [], "lr": []}

    def spy_splitter(*args, **kw):
        seen["splitter"].append((args, kw))
        inner = StratifiedGroupKFold(*args, **kw)
        real_split = inner.split

        def split(X, y=None, groups=None):
            seen["groups"].append(np.asarray(groups).copy())
            return real_split(X, y, groups)

        inner.split = split
        return inner

    def spy_lr(*args, **kw):
        seen["lr"].append((args, kw))
        return LogisticRegression(*args, **kw)

    monkeypatch.setattr(rm, "StratifiedGroupKFold", spy_splitter)
    monkeypatch.setattr(rm, "LogisticRegression", spy_lr)
    rng = np.random.default_rng(3)
    days = np.r_[np.repeat(np.arange(1, 9), 3), np.repeat(np.arange(11, 19), 3)]    # 8 calibration days, then 8 deployment days
    rm.domain_auc(rng.normal(size=(24, 2)), rng.normal(size=(24, 2)), days)
    # stratified 5-fold, deterministic (no shuffling, so no random state), blocked by day, deployment ids offset
    assert seen["splitter"] == [((), {"n_splits": 5, "shuffle": False})]
    assert seen["groups"][0].tolist() == [d for d in range(1, 9) for _ in range(3)] + [d + 10_000 for d in range(11, 19) for _ in range(3)]
    # balanced class weights (Ruling 40), one model per fold
    assert seen["lr"] == [((), {"max_iter": 2000, "class_weight": "balanced"})] * 5


def test_domain_auc_misaligned_days_message_names_the_contract(rm):
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="day ids must cover the calibration rows, then the deployment rows"):
        rm.domain_auc(rng.normal(size=(10, 2)), rng.normal(size=(10, 2)), np.arange(19))


def test_domain_auc_is_calibrated_under_no_drift_and_still_detects_shift(rm):
    """Day groups are class-pure, so an unweighted fit's intercept follows the training class balance, which
    runs against each held-out fold's balance, and the pooled out-of-fold AUC falls below 0.5 under no drift.
    class_weight='balanced' removes that bias, and a real shift is still found."""
    days = np.r_[np.repeat(np.arange(13), 25), np.repeat(np.arange(13), 25)]    # 13 + 13 groups of 25 rows
    null, shifted = [], []
    for seed in range(10):
        rng = np.random.default_rng(1000 + seed)
        cal, dep = rng.normal(size=(325, 2)), rng.normal(size=(325, 2))
        null.append(rm.domain_auc(cal, dep, days))
        shifted.append(rm.domain_auc(cal, dep + 1.0, days))
    assert abs(np.mean(null) - 0.5) < 0.05
    assert np.mean(shifted) > 0.75


def test_the_day_blocked_null_is_far_from_the_week_blocked_value_on_a_quarter_plus_14_days(rm):
    # AQ3's calendar (a 92-day calibration quarter, the first 14 days of the next year's quarter), persistent
    # AR(0.9) features, no drift. The week-blocked scheme Ruling 73 replaced scored about 0.26 here (40 seeds):
    # 14 days leave only 2-3 deployment week groups. The day-blocked scheme is somewhat above 0.5 (about 0.54).
    day, week = [], []
    for seed in range(8):
        cal, dep, cal_dates, dep_dates = _calendar_case("2024-07-01", 92, "2025-07-01", 0.9, seed)
        day.append(rm.domain_auc(cal, dep, np.array([_epoch_day(d) for d in cal_dates + dep_dates])))
        week.append(_week_blocked_auc(rm, cal, dep, cal_dates, dep_dates))
    assert np.mean(week) < 0.40                          # the old scheme: far below 0.5
    assert np.mean(day) > 0.40                           # so its value (about 0.26) is clearly rejected ...
    assert np.mean(day) - np.mean(week) > 0.15           # ... by a wide margin on the same data


@pytest.mark.slow
def test_day_blocked_null_levels_over_many_seeds(rm):
    # Same calendar, 40 seeds, so the means are tight (standard error about 0.01 for AR, 0.004 for iid)
    iid, ar, ar_shifted = [], [], []
    for seed in range(40):
        cal, dep, cal_dates, dep_dates = _calendar_case("2024-07-01", 92, "2025-07-01", 0.0, seed)
        ids = np.array([_epoch_day(d) for d in cal_dates + dep_dates])
        iid.append(rm.domain_auc(cal, dep, ids))
        cal, dep, _, _ = _calendar_case("2024-07-01", 92, "2025-07-01", 0.9, seed)
        ar.append(rm.domain_auc(cal, dep, ids))
        ar_shifted.append(rm.domain_auc(cal, dep + 0.5, ids))
    assert abs(np.mean(iid) - 0.5) < 0.05                # iid features: 0.50
    assert 0.45 < np.mean(ar) < 0.62                     # persistent features: somewhat above 0.5 (about 0.54), not far
    assert np.mean(ar_shifted) > 0.65 and np.mean(ar_shifted) > np.mean(ar) + 0.12     # a +0.5 sd shift: about 0.79


def test_the_calendar_that_used_to_crash_now_runs(rm):
    from sklearn.model_selection import GroupKFold
    # SAF's group structure, in the development period: a quarter that starts on a Monday and a deployment that
    # starts on a Wednesday. Its deployment week groups are 120, 168 and 48 rows, and the week-blocked GroupKFold
    # left one training fold without a deployment row, so the fit raised.
    cal_dates, dep_dates = _hourly_dates("2024-07-01", 92), _hourly_dates("2025-07-02", 14)
    n_cal = len(cal_dates)
    y = np.r_[np.zeros(n_cal), np.ones(len(dep_dates))]
    week_groups = rm.cv_groups(np.array([d.isocalendar()[0] * 100 + d.isocalendar()[1] for d in cal_dates + dep_dates]), n_cal)
    old_folds = list(GroupKFold(n_splits=5).split(np.zeros((len(y), 1)), y, week_groups))
    assert any(len(np.unique(y[train])) < 2 for train, _ in old_folds)               # the old scheme's failure
    rng = np.random.default_rng(2)
    auc = rm.domain_auc(rng.normal(size=(n_cal, 6)), rng.normal(size=(len(dep_dates), 6)),
                        np.array([_epoch_day(d) for d in cal_dates + dep_dates]))
    assert 0.0 <= auc <= 1.0


def test_domain_auc_is_deterministic(rm):
    first = _auc_on_calendar(rm, "2024-07-01", 30, "2025-07-01", 0.9, 0.0, 0)
    assert _auc_on_calendar(rm, "2024-07-01", 30, "2025-07-01", 0.9, 0.0, 0) == first       # no random state anywhere
    panel = _panel({"N1": 1.0, "I1": 0.0}, seed=51)
    scores = {"LEV6": _scores(panel, {"N1": 1.5, "I1": 0.0}, seed=52)}
    pd.testing.assert_frame_equal(rm.monitor_rows(panel, FAKE, scores), rm.monitor_rows(panel, FAKE, scores))


def test_domain_auc_names_the_problem_with_fewer_than_five_day_groups(rm):
    rng = np.random.default_rng(6)
    four = np.r_[np.repeat([1, 2], 10), np.repeat([7, 8], 10)]                  # 2 + 2 day groups
    with pytest.raises(ValueError, match=r"4 distinct day group.*at least 5"):
        rm.domain_auc(rng.normal(size=(20, 2)), rng.normal(size=(20, 2)), four)
    # the two sides count separately: the same eight days on both sides are 16 groups, not 8
    shared = np.r_[np.repeat(np.arange(8), 10), np.repeat(np.arange(8), 10)]
    assert 0.0 <= rm.domain_auc(rng.normal(size=(80, 2)), rng.normal(size=(80, 2)), shared) <= 1.0


def test_domain_auc_guard_says_why_when_a_fold_lacks_a_class(rm):
    rng = np.random.default_rng(7)
    # 20 calibration days but only 3 deployment days: five folds cannot all test on a deployment day
    days = np.r_[np.repeat(np.arange(20), 24), np.repeat(np.arange(100, 103), 24)]
    with pytest.raises(ValueError, match=r"has no deployment rows.*at least 5 calibration days and 5 deployment days "
                                         r"\(here 20 and 3\)"):
        rm.domain_auc(rng.normal(size=(480, 2)), rng.normal(size=(72, 2)), days)
    # the mirror image: 3 calibration days
    days = np.r_[np.repeat(np.arange(3), 24), np.repeat(np.arange(100, 120), 24)]
    with pytest.raises(ValueError, match=r"has no calibration rows.*\(here 3 and 20\)"):
        rm.domain_auc(rng.normal(size=(72, 2)), rng.normal(size=(480, 2)), days)
    # the same three days on both sides are 6 groups, which passes the group count but not the class check
    days = np.r_[np.repeat(np.arange(3), 24), np.repeat(np.arange(3), 24)]
    with pytest.raises(ValueError, match=r"has no (calibration|deployment) rows.*\(here 3 and 3\)"):
        rm.domain_auc(rng.normal(size=(72, 2)), rng.normal(size=(72, 2)), days)
    # five deployment days are enough: stratification puts one in each fold's test part
    days = np.r_[np.repeat(np.arange(20), 24), np.repeat(np.arange(100, 105), 24)]
    assert 0.0 <= rm.domain_auc(rng.normal(size=(480, 2)), rng.normal(size=(120, 2)), days) <= 1.0


# ---- day ids ---------------------------------------------------------------------------------------

def test_day_ids_are_one_integer_per_local_settlement_day(rm):
    d = dt.date
    dates = [d(2024, 12, 31), d(2025, 1, 1), d(2025, 1, 1), d(2024, 1, 1), d(2024, 12, 30)]
    expected = [_epoch_day(x) for x in dates]
    assert rm.day_ids(dates).tolist() == expected                                   # days since 1970-01-01
    assert expected[1] == expected[2] and expected[1] - expected[0] == 1 and expected[0] - expected[4] == 1
    as_series = pd.Series(pd.to_datetime(dates), index=range(10, 15))               # a non-default index
    assert rm.day_ids(as_series).tolist() == expected
    same_day = pd.to_datetime(["2025-01-01 05:00", "2025-01-01 23:00", "2025-01-02 00:00"])   # the time of day is ignored
    assert rm.day_ids(same_day).tolist() == [expected[1], expected[1], expected[1] + 1]


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")    # 6-hourly rows: W1 needs 50 values a side
def test_monitor_rows_blocks_the_domain_classifier_by_settlement_day(rm, monkeypatch):
    seen = []
    monkeypatch.setattr(rm, "domain_auc", lambda cal_X, dep_X, day_ids: seen.append(np.asarray(day_ids).copy()) or 0.5)
    # both windows cross New Year, which a calendar-year or week key would get wrong; the classifier sees the
    # 21 calibration days and the first 14 deployment days (2025-12-22..2026-01-04), one id per day
    cross = _fake_analysis(("2024-12-23", "2025-01-12"), ("2025-12-22", "2026-01-11"), id="CROSS")
    panel = _panel({"N1": 1.0}, spans=(("2024-12-23", 21), ("2025-12-22", 21)), hours=6)
    rm.monitor_rows(panel, cross, {"LEV6": _scores(panel, {"N1": 1.0}, analysis=cross)})
    g = panel[panel["node"] == "N1"]
    early = (_day_since_first(panel, cross).loc[g.index] < 14).to_numpy()
    dates = [*g.loc[cross.calib.mask(g).to_numpy(), "settle_date"], *g.loc[early, "settle_date"]]
    expected = [_epoch_day(x) for x in dates]
    assert len(set(expected)) == 21 + 14
    assert len(seen) == 2 and seen[0].tolist() == expected          # the monitors' call, then the placebo's (Ruling 83)


def test_monitor_rows_names_the_node_when_its_domain_classifier_cannot_run(rm, monkeypatch):
    monkeypatch.setattr(rm, "MONITOR_DAYS", 3)            # a contrived early window of 3 deployment days
    panel = _panel({"N1": 1.0, "I1": 0.0}, seed=41)
    scores = {"LEV6": _scores(panel, {"N1": 1.0, "I1": 0.0}, seed=42)}
    with pytest.raises(ValueError, match=r"FAKE.*LEV6.*ISONE/I1.*has no deployment rows.*\(here 28 and 3\)") as err:
        rm.monitor_rows(panel, FAKE, scores)
    assert isinstance(err.value.__cause__, ValueError)


# ---- psi -------------------------------------------------------------------------------------------

def test_psi_matches_a_hand_computed_value_with_ten_bins_and_a_1e_4_floor(rm):
    ref = np.arange(100.0)                          # decile edges 0, 9.9, ..., 99: ten bins of 10 values each
    cur = np.full(50, 99.5)                         # all current mass in the top bin; nine bins are empty
    r, c_top, floor = 0.1, 1.0, 1e-4
    expected = 9 * (floor - r) * np.log(floor / r) + (c_top - r) * np.log(c_top / r)
    assert rm.psi(ref, cur) == pytest.approx(expected)


def test_psi_is_nan_when_the_reference_leaves_fewer_than_two_bins(rm):
    # A constant reference has one quantile edge, so there is nothing to bin by. 0.0 would say "no drift" even
    # though the deployment moved (Ruling 44).
    assert np.isnan(rm.psi(np.ones(100), np.full(100, 5.0)))
    assert np.isnan(rm.psi(np.ones(100), np.ones(100)))
    # Deciles that collapse onto two values leave a single bin, where PSI is 0 for every current sample.
    assert np.isnan(rm.psi(np.r_[np.zeros(95), np.ones(5)], np.full(50, 7.0)))
    # Two bins are enough: a 50/50 reference splits at 0.5.
    half = np.r_[np.zeros(50), np.ones(50)]
    assert rm.psi(half, half) == 0.0
    assert rm.psi(half, np.ones(100)) > 0.5


# ---- nwd: the July pipeline's 1-Wasserstein summary ------------------------------------------------

def _july_nwd_def():
    """The ``nwd`` FunctionDef parsed from the source text of v2/scripts/open_holdout.py. That script opens
    the sealed holdout when it runs, so it is never imported or executed."""
    import ast
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "v2" / "scripts" / "open_holdout.py"
    func = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "nwd")
    return func, str(path)


def _july_nwd():
    """The July pipeline's ``nwd`` as a callable: only the one function definition is compiled, in a namespace
    that holds just numpy and scipy's wasserstein_distance."""
    import ast
    from scipy.stats import wasserstein_distance
    func, path = _july_nwd_def()
    namespace = {"np": np, "wasserstein_distance": wasserstein_distance}
    exec(compile(ast.Module(body=[func], type_ignores=[]), path, "exec"), namespace)
    return namespace["nwd"]


def test_nwd_is_a_verbatim_copy_of_the_july_pipelines_function(rm):
    import ast
    import inspect
    import textwrap
    original, _ = _july_nwd_def()
    copy = ast.parse(textwrap.dedent(inspect.getsource(rm.nwd))).body[0]
    assert ast.dump(copy) == ast.dump(original)          # same structure and constants, whatever the layout


def test_nwd_is_the_july_pipelines_function(rm):
    july = _july_nwd()
    rng = np.random.default_rng(5)
    cases = [(rng.normal(size=n_a), rng.normal(0.3, 1.5, size=n_d))
             for n_a, n_d in ((200, 200), (60, 500), (50, 50), (49, 200), (200, 49), (10, 10), (1000, 80))]
    with_bad = rng.normal(size=120)
    with_bad[[3, 17]], with_bad[40] = np.nan, np.inf                       # non-finite values are dropped
    cases.append((with_bad, rng.normal(size=100)))
    cases.append((np.r_[np.full(10, np.nan), rng.normal(size=45)], rng.normal(size=100)))   # 55 values, 45 finite
    cases.append((np.r_[np.zeros(80), np.arange(20.0)], rng.normal(size=100)))              # zero IQR: 1e-9 is used
    cases.append((np.zeros(60), np.ones(60)))                                               # constant calibration
    mine, theirs = [rm.nwd(a, d) for a, d in cases], [july(a, d) for a, d in cases]
    for got, want in zip(mine, theirs):
        assert (np.isnan(got) and np.isnan(want)) or got == want
    # the cases exercise every branch, so agreement is not agreement on NaN everywhere
    assert sum(np.isnan(v) for v in theirs) == 4                              # 49 / 49 / 10 values, and 45 finite
    assert sum(np.isfinite(v) for v in theirs) == 7
    assert theirs[-2] > 1e6 and theirs[-1] > 1e6                              # divided by 1e-9


# ---- early_rate ------------------------------------------------------------------------------------

def test_early_rate_is_the_mean_alert_over_each_nodes_first_14_days(rm):
    d0 = dt.date(2025, 7, 1)

    def day(k):
        return d0 + dt.timedelta(days=k)

    alerts = pd.DataFrame({
        "market": ["M"] * 6 + ["M"] * 3 + ["X"] * 2, "node": ["a"] * 6 + ["b"] * 3 + ["a"] * 2,
        "settle_date": [day(k) for k in (0, 1, 13, 14, 20, 30)] + [day(k) for k in (5, 6, 30)] + [day(2), day(3)],
        "alert": [True, False, False, True, True, False, True, True, False, False, False]})
    er = rm.early_rate(alerts)
    assert er[("M", "a")] == pytest.approx(1 / 3)      # days 0, 1, 13 count; day 14 is outside (< 14)
    assert er[("M", "b")] == 1.0                       # measured from the node's own first day (5): days 5, 6 count
    assert er[("X", "a")] == 0.0                       # same node name in another market is a separate node
    assert rm.early_rate(alerts, days=15)[("M", "a")] == pytest.approx(2 / 4)


def test_early_rate_reads_monitor_days_when_it_is_called(rm, monkeypatch):
    d0 = dt.date(2025, 7, 1)

    def day(k):
        return d0 + dt.timedelta(days=k)

    alerts = pd.DataFrame({
        "market": ["M"] * 6 + ["M"] * 3 + ["X"] * 2, "node": ["a"] * 6 + ["b"] * 3 + ["a"] * 2,
        "settle_date": [day(k) for k in (0, 1, 13, 14, 20, 30)] + [day(k) for k in (5, 6, 30)] + [day(2), day(3)],
        "alert": [True, False, False, True, True, False, True, True, False, False, False]})
    assert rm.early_rate(alerts)[("M", "a")] == pytest.approx(1 / 3)                 # the default: days 0-13
    monkeypatch.setattr(rm, "MONITOR_DAYS", 3)                                       # the constant, changed at run time
    er = rm.early_rate(alerts)                                                       # now days 0-2
    assert er[("M", "a")] == 0.5 and er[("M", "b")] == 1.0 and er[("X", "a")] == 0.0
    assert rm.early_rate(alerts, days=15)[("M", "a")] == pytest.approx(2 / 4)        # an explicit argument still wins


def test_every_call_site_reads_monitor_days_when_called(rm, monkeypatch):
    monkeypatch.setattr(rm, "MONITOR_DAYS", 5)             # five early days: still enough for the day-blocked classifier
    panel = _panel({"N1": 1.0, "I1": 0.0}, seed=71)
    sc = _scores(panel, {"N1": 1.5, "I1": 0.0}, seed=72)
    rows = rm.monitor_rows(panel, FAKE, {"LEV6": sc}).set_index("node")
    day = _day_since_first(panel, FAKE)
    outcome = _independent_outcome(sc, days=5)
    for node, key in (("N1", ("NYISO", "N1")), ("I1", ("ISONE", "I1"))):
        g = panel[panel["node"] == node]
        assert rows.loc[node, "n_early"] == int((day.loc[g.index] < 5).sum())        # the panel split
        assert rows.loc[node, "early_rate"] == pytest.approx(outcome[key]["early_rate"])   # early_rate, too
        assert rows.loc[node, "n_late"] == outcome[key]["n_late"]                    # the late split
        assert rows.loc[node, "log_error"] == pytest.approx(outcome[key]["log_error"])


# ---- evaluate_monitors -----------------------------------------------------------------------------

def _eval_frame(**cols):
    n = len(next(iter(cols.values())))
    base = {"analysis": "A", "block": "B", "market": "M", "node": [f"n{i}" for i in range(n)],
            "log_error": np.linspace(0.1, 0.9, n), "fail": [False] * (n // 2) + [True] * (n - n // 2),
            "W1_norm": np.linspace(0, 1, n), "PSI": np.linspace(0, 1, n), "domain_auc": np.linspace(0.5, 1, n),
            "early_rate": np.full(n, 0.05)}
    base.update(cols)
    return pd.DataFrame(base)


def test_early_rate_enters_as_abs_log_ratio_to_alpha_and_zero_does_not_crash(rm):
    rates = np.array([0.05, 0.025, 0.1, 0.0125, 0.2, 0.0])                   # |log10(r/0.05)| = 0, .30, .30, .60, .60, 3
    err = np.abs(np.log10(np.maximum(rates / 0.05, 1e-3)))
    ev = rm.evaluate_monitors(_eval_frame(early_rate=rates, log_error=err)).set_index("monitor")
    assert ev.loc["early_rate", "spearman"] == pytest.approx(1.0)
    assert np.isfinite(ev.loc["early_rate", "auroc_fail"])                  # a zero rate must not give inf


@pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning")   # the default early_rate column is constant
def test_one_sided_p_equals_scipy_greater_for_positive_negative_and_null_rho(rm):
    from scipy.stats import spearmanr
    rng = np.random.default_rng(7)
    n = 15
    err = rng.normal(size=n)
    cols = {"W1_norm": err + rng.normal(scale=0.5, size=n), "PSI": -err + rng.normal(scale=0.5, size=n),
            "domain_auc": rng.normal(size=n)}
    ev = rm.evaluate_monitors(_eval_frame(log_error=err, **cols)).set_index("monitor")
    for mon, x in cols.items():
        assert ev.loc[mon, "spearman_p_one_sided"] == pytest.approx(spearmanr(x, err, alternative="greater").pvalue)
    assert ev.loc["W1_norm", "spearman"] > 0 > ev.loc["PSI", "spearman"]


def test_nan_conventions_constant_monitor_and_single_class_fail(rm):
    from scipy.stats import ConstantInputWarning
    with pytest.warns(ConstantInputWarning):
        ev = rm.evaluate_monitors(_eval_frame(W1_norm=np.ones(8), fail=[False] * 8)).set_index("monitor")
    assert np.isnan(ev.loc["W1_norm", "spearman"]) and np.isnan(ev.loc["W1_norm", "spearman_p_one_sided"])
    assert ev["auroc_fail"].isna().all()


def test_evaluate_monitors_drops_nan_values_per_monitor_and_reports_n_used(rm):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(8)
    n = 12
    err = rng.normal(size=n)
    w1 = err + rng.normal(scale=0.6, size=n)
    w1[[2, 8]] = np.nan                                  # two nodes have no W1 (a window under 50 values)
    psi = err + rng.normal(scale=0.6, size=n)
    frame = _eval_frame(W1_norm=w1, PSI=psi, log_error=err, fail=[False] * 6 + [True] * 6,
                        early_rate=np.linspace(0.05, 0.2, n))
    ev = rm.evaluate_monitors(frame).set_index("monitor")
    keep = ~np.isnan(w1)
    assert list(rm.evaluate_monitors(frame).columns) == rm.EVAL_COLUMNS and rm.EVAL_COLUMNS[-4:] == ["n_used", "calib_days", "placebo_median", "placebo_n"]
    assert ev.loc["W1_norm", "n_used"] == 10 and ev.loc["PSI", "n_used"] == n
    # Spearman and AUROC are computed on the ten nodes that have a W1 value; the other monitors are untouched
    assert ev.loc["W1_norm", "spearman"] == pytest.approx(spearmanr(w1[keep], err[keep]).statistic)
    assert ev.loc["W1_norm", "spearman_p_one_sided"] == pytest.approx(
        spearmanr(w1[keep], err[keep], alternative="greater").pvalue)
    assert ev.loc["W1_norm", "auroc_fail"] == pytest.approx(roc_auc_score(frame["fail"][keep], w1[keep]))
    assert ev.loc["PSI", "spearman"] == pytest.approx(spearmanr(psi, err).statistic)
    assert ev.loc["PSI", "auroc_fail"] == pytest.approx(roc_auc_score(frame["fail"], psi))


def test_evaluate_monitors_handles_a_monitor_with_too_few_values_left(rm):
    n = 8
    # no node has a W1 value; for PSI only one node does; for domain_auc the only nodes left are all non-failing
    psi = np.full(n, np.nan)
    psi[3] = 0.4
    auc = np.full(n, np.nan)
    auc[:3] = [0.5, 0.6, 0.7]
    frame = _eval_frame(W1_norm=np.full(n, np.nan), PSI=psi, domain_auc=auc, early_rate=np.linspace(0.05, 0.2, n))
    ev = rm.evaluate_monitors(frame).set_index("monitor")
    assert ev.loc["W1_norm", "n_used"] == 0 and ev.loc["PSI", "n_used"] == 1 and ev.loc["domain_auc", "n_used"] == 3
    for mon in ("W1_norm", "PSI"):
        assert ev.loc[mon, ["spearman", "spearman_p_one_sided", "auroc_fail"]].isna().all()
    assert np.isfinite(ev.loc["domain_auc", "spearman"]) and np.isnan(ev.loc["domain_auc", "auroc_fail"])  # one fail class
    assert ev.loc["early_rate", "n_used"] == n


def test_evaluate_monitors_reports_the_calibration_window_length_so_long_windows_can_be_flagged(rm):
    # a quarter analysis next to a full-year one (the legacy analyses are evaluated too). calib_days is descriptive:
    # it tells the reader which analyses have a long calibration window
    quarter = _eval_frame(early_rate=np.linspace(0.05, 0.2, 12), W1_norm=np.linspace(0, 1, 12)).assign(
        analysis="AQ3", calib_days=92)
    quarter.loc[0, "calib_days"] = 90                         # one node has two days of calibration data missing
    year = _eval_frame(early_rate=np.linspace(0.05, 0.2, 12), W1_norm=np.linspace(0, 1, 12)).assign(
        analysis="2024-2025", calib_days=367)
    ev = rm.evaluate_monitors(pd.concat([quarter, year], ignore_index=True))
    assert list(ev.columns) == rm.EVAL_COLUMNS and "calib_days" in ev.columns
    assert ev.loc[ev["analysis"] == "AQ3", "calib_days"].tolist() == [92] * 4        # the longest node's count
    assert ev.loc[ev["analysis"] == "2024-2025", "calib_days"].tolist() == [367] * 4
    flagged = ev[ev["calib_days"] > 100]
    assert set(flagged["analysis"]) == {"2024-2025"} and len(flagged) == 4            # one row per monitor


@pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning")
def test_evaluate_monitors_leaves_calib_days_missing_when_the_frame_has_no_such_column(rm):
    ev = rm.evaluate_monitors(_eval_frame(W1_norm=np.linspace(0, 1, 8)))
    assert ev["calib_days"].isna().all() and list(ev.columns) == rm.EVAL_COLUMNS


# ---- monitor_rows ----------------------------------------------------------------------------------

def test_monitor_rows_values_are_pinned_to_independent_formulas(rm, monkeypatch):
    from scipy.stats import wasserstein_distance
    from es.blocks import BLOCKS
    monkeypatch.setattr(rm, "cached_scores", lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache read")))
    panel = _panel({"N1": 1.0, "I1": 0.0}, seed=11)
    scores = {"LEV6": _scores(panel, {"N1": 1.5, "I1": 0.0}, seed=12), "MID3": _scores(panel, {"N1": 1.5, "I1": 0.0}, seed=13)}
    rows = rm.monitor_rows(panel, FAKE, scores)
    assert list(rows.columns) == rm.ROW_COLUMNS and len(rows) == 4
    day = _day_since_first(panel, FAKE)
    for block, sc in scores.items():
        outcome = _independent_outcome(sc)
        feats = BLOCKS[block]
        for node, g in panel.groupby("node"):
            r = rows[(rows.block == block) & (rows.node == node)].iloc[0]
            key = (g["market"].iloc[0], node)
            cal = g[FAKE.calib.mask(g).to_numpy()]
            early = g[(day.loc[g.index] < 14).to_numpy()]            # the first 14 deployment days: all the monitors read
            # W1 is the July pipeline's statistic (Ruling 44): per feature the Wasserstein distance between the raw
            # calibration and early-deployment values over the calibration IQR, then the median over the family
            iqr = {f: np.percentile(cal[f], 75) - np.percentile(cal[f], 25) for f in feats}
            w1 = np.median([wasserstein_distance(cal[f], early[f]) / iqr[f] for f in feats])
            assert r["W1_norm"] == pytest.approx(w1, rel=1e-9)
            assert r["PSI"] == pytest.approx(np.mean([rm.psi(cal[f].to_numpy(), early[f].to_numpy()) for f in feats]))
            # the early alert rate and the outcome (late days, calibration cutoff) are recomputed from the scores
            assert r["early_rate"] == pytest.approx(outcome[key]["early_rate"])
            assert r["log_error"] == pytest.approx(outcome[key]["log_error"])
            assert bool(r["fail"]) == outcome[key]["fail"]
            assert r["n_early"] == len(early) and r["n_late"] == outcome[key]["n_late"]
            assert r["calib_days"] == len({*cal["settle_date"]}) == 28            # distinct local settlement days
            # the placebo (Ruling 83): the same three statistics with the last 14 calibration days as the
            # pseudo-deployment and the earlier calibration days as the pseudo-calibration
            last14 = sorted({*cal["settle_date"]})[-14:]
            p_dep = cal[cal["settle_date"].isin(last14).to_numpy()]
            p_cal = cal[~cal["settle_date"].isin(last14).to_numpy()]
            p_iqr = {f: np.percentile(p_cal[f], 75) - np.percentile(p_cal[f], 25) for f in feats}
            assert r["W1_norm_placebo"] == pytest.approx(
                np.median([wasserstein_distance(p_cal[f], p_dep[f]) / p_iqr[f] for f in feats]), rel=1e-9)
            assert r["psi_placebo"] == pytest.approx(np.mean([rm.psi(p_cal[f].to_numpy(), p_dep[f].to_numpy()) for f in feats]))
            p_med, p_sd = p_cal[feats].median(), p_cal[feats].std()
            p_days = np.array([_epoch_day(x) for x in [*p_cal["settle_date"], *p_dep["settle_date"]]])
            assert r["domain_auc_placebo"] == pytest.approx(rm.domain_auc(
                ((p_cal[feats] - p_med) / p_sd).to_numpy(), ((p_dep[feats] - p_med) / p_sd).to_numpy(), p_days))
            # the domain classifier sees the calibration rows first, then the early rows, blocked by settlement day
            wk = np.array([_epoch_day(x) for x in [*cal["settle_date"], *early["settle_date"]]])
            med, s = cal[feats].median(), cal[feats].std()
            assert r["domain_auc"] == pytest.approx(
                rm.domain_auc(((cal[feats] - med) / s).to_numpy(), ((early[feats] - med) / s).to_numpy(), wk))
    stable, drifted = rows[rows.node == "I1"].set_index("block"), rows[rows.node == "N1"].set_index("block")
    for mon in ("W1_norm", "PSI", "domain_auc"):
        assert (drifted[mon] > stable[mon]).all()


def test_monitor_rows_averages_psi_over_the_features_that_have_one(rm):
    from es.blocks import BLOCKS
    const = "da_lmp_lag_48h"                         # constant in both windows: PSI is NaN for it (Ruling 44)
    panel = _panel({"N1": 1.0, "I1": 0.0}, seed=5, const_feature=const)
    rows = rm.monitor_rows(panel, FAKE, {"LEV6": _scores(panel, {"N1": 1.5, "I1": 0.0})})
    assert not rows[list(rm.MONITORS)].isna().any().any()
    day = _day_since_first(panel, FAKE)
    for node, g in panel.groupby("node"):
        cal, early = g[FAKE.calib.mask(g).to_numpy()], g[(day.loc[g.index] < 14).to_numpy()]
        assert np.isnan(rm.psi(cal[const].to_numpy(), early[const].to_numpy()))
        others = [f for f in BLOCKS["LEV6"] if f != const]
        expected = np.mean([rm.psi(cal[f].to_numpy(), early[f].to_numpy()) for f in others])
        assert rows.loc[rows.node == node, "PSI"].iloc[0] == pytest.approx(expected)


def test_monitor_rows_rejects_a_block_without_price_features(rm):
    panel = _panel({"N1": 1.0, "I1": 0.0})
    # the wording matters: without the guard, domain_auc fails on 0 features and the node-naming re-raise
    # would also mention 'G6', so matching the block name alone could not tell the two apart
    with pytest.raises(ValueError, match="'G6' has no price-history features"):
        rm.monitor_rows(panel, FAKE, {"G6": _scores(panel, {"N1": 0.0, "I1": 0.0})})


def test_monitor_rows_names_the_node_for_any_per_node_failure(rm, monkeypatch):
    panel = _panel({"N1": 1.0, "I1": 0.0})
    scores = {"LEV6": _scores(panel, {"N1": 1.5, "I1": 0.0})}

    def boom(reference, current, bins=10):
        raise RuntimeError("boom")

    monkeypatch.setattr(rm, "psi", boom)                  # not a ValueError, and not from domain_auc
    with pytest.raises(ValueError, match=r"monitor_rows\('FAKE', 'LEV6'\): node ISONE/I1: RuntimeError: boom") as err:
        rm.monitor_rows(panel, FAKE, scores)
    assert isinstance(err.value.__cause__, RuntimeError)


@pytest.mark.parametrize("side", ["calibration", "deployment"])
def test_monitor_rows_names_the_node_whose_window_is_empty_after_dropna(rm, side):
    panel = _panel({"N1": 1.0, "I1": 0.0})
    scores = {"LEV6": _scores(panel, {"N1": 1.5, "I1": 0.0})}             # scores exist for both windows
    window = FAKE.calib if side == "calibration" else FAKE.deploy
    panel.loc[(panel["node"] == "I1") & window.mask(panel), "da_lmp_lag_48h"] = np.nan   # no complete LEV6 row left
    with pytest.raises(ValueError, match=rf"FAKE.*LEV6.*ISONE/I1.*no {side} rows with complete LEV6 features"):
        rm.monitor_rows(panel, FAKE, scores)


@pytest.mark.parametrize("kind", ["date", "utc"])
def test_monitor_rows_runs_with_real_analysis_windows(rm, kind):
    from scipy.stats import wasserstein_distance
    from es.analyses import Analysis
    from es.blocks import BLOCKS
    from es.windows import DateWindow, UtcWindow
    if kind == "date":      # inclusive local settlement dates
        train, cal, dep = (DateWindow("2024-12-02", "2025-01-05"), DateWindow("2025-01-06", "2025-02-02"),
                           DateWindow("2026-01-05", "2026-01-25"))
    else:                   # half-open UTC intervals
        train, cal, dep = (UtcWindow("2024-12-02", "2025-01-06"), UtcWindow("2025-01-06", "2025-02-03"),
                           UtcWindow("2026-01-05", "2026-01-26"))
    analysis = Analysis("REAL", train, cal, dep, train, "rolling", "analog")
    panel = _panel({"N1": 1.0, "I1": 0.0}, spans=(("2025-01-06", 28), ("2026-01-05", 21)))
    scores = {"LEV6": _scores(panel, {"N1": 1.5, "I1": 0.0}, analysis=analysis)}
    rows = rm.monitor_rows(panel, analysis, scores)
    assert list(rows.columns) == rm.ROW_COLUMNS and len(rows) == 2 and set(rows["analysis"]) == {"REAL"}
    assert not rows[list(rm.MONITORS)].isna().any().any()
    stable, drifted = rows[rows.node == "I1"].iloc[0], rows[rows.node == "N1"].iloc[0]
    for mon in ("W1_norm", "PSI", "domain_auc"):
        assert drifted[mon] > stable[mon]
    # each kind of window selects the rows its own rule says; W1 over the calibration rows and the first 14
    # deployment days, computed independently, shows it. The outcome is scored on the days after those 14.
    outcome = _independent_outcome(scores["LEV6"])
    for node, g in panel.groupby("node"):
        ts = g["timestamp_utc"]
        if kind == "date":
            in_cal = (g["settle_date"] >= dt.date(2025, 1, 6)) & (g["settle_date"] <= dt.date(2025, 2, 2))
            in_dep = (g["settle_date"] >= dt.date(2026, 1, 5)) & (g["settle_date"] <= dt.date(2026, 1, 25))
        else:
            in_cal = (ts >= pd.Timestamp("2025-01-06", tz="UTC")) & (ts < pd.Timestamp("2025-02-03", tz="UTC"))
            in_dep = (ts >= pd.Timestamp("2026-01-05", tz="UTC")) & (ts < pd.Timestamp("2026-01-26", tz="UTC"))
        c, d = g[in_cal], g[in_dep]
        first = min(d["settle_date"])
        early = d[[(x - first).days < 14 for x in d["settle_date"]]]
        w1 = np.median([wasserstein_distance(c[f], early[f]) / (np.percentile(c[f], 75) - np.percentile(c[f], 25))
                        for f in BLOCKS["LEV6"]])
        r = rows.loc[rows.node == node].iloc[0]
        assert r["W1_norm"] == pytest.approx(w1, rel=1e-9)
        key = (g["market"].iloc[0], node)
        assert r["n_early"] == len(early) and r["n_late"] == outcome[key]["n_late"] == len(d) - len(early)
        assert r["calib_days"] == len({*c["settle_date"]})
        assert r["log_error"] == pytest.approx(outcome[key]["log_error"])


# ---- early and late deployment days (Ruling 66b) ---------------------------------------------------
# Every monitor is computable by deployment day 14, and the outcome (|log10 R| and the failure flag) is scored
# on the days after them, so a monitor cannot correlate with an outcome that contains its own data.

def test_deployment_day_counts_from_each_nodes_first_deployment_day(rm):
    d0 = dt.date(2026, 3, 9)

    def day(k):
        return d0 + dt.timedelta(days=k)

    frame = pd.DataFrame({"market": ["M"] * 6 + ["X"], "node": ["a"] * 4 + ["b"] * 2 + ["a"],
                          "settle_date": [day(0), day(1), day(13), day(14), day(5), day(19), day(2)]})
    assert rm.deployment_day(frame).tolist() == [0, 1, 13, 14, 0, 14, 0]     # b starts on day 5, X/a on day 2


def test_the_monitors_read_only_the_first_14_deployment_days(rm):
    from es.blocks import BLOCKS
    feats = BLOCKS["LEV6"]
    panel = _panel({"N1": 1.0, "I1": 0.0}, seed=31)
    scores = {"LEV6": _scores(panel, {"N1": 1.5, "I1": 0.0}, seed=32)}
    base = rm.monitor_rows(panel, FAKE, scores)
    day = _day_since_first(panel, FAKE)
    # extreme feature values on the days after the first 14 move neither W1, PSI nor the AUC (nor anything else)
    late_wild = panel.copy()
    late_wild.loc[(day >= 14).to_numpy(), feats] = 1e6
    pd.testing.assert_frame_equal(rm.monitor_rows(late_wild, FAKE, scores), base)
    # the check is not vacuous: values that extreme on the first 14 days do move all three
    early_wild = panel.copy()
    early_wild.loc[(day < 14).to_numpy(), feats] = 50.0
    moved = rm.monitor_rows(early_wild, FAKE, scores)
    for mon in ("W1_norm", "PSI", "domain_auc"):
        assert (moved[mon] > base[mon]).all()


def test_the_outcome_reads_only_the_days_after_the_first_14(rm):
    panel = _panel({"N1": 1.0, "I1": 0.0}, seed=33)
    sc = _scores(panel, {"N1": 1.5, "I1": 0.0}, seed=34)
    base = rm.monitor_rows(panel, FAKE, {"LEV6": sc})
    day = _day_since_first(panel, FAKE).to_numpy()
    # an alert on every row of the first 14 days: the early alert rate moves, the outcome and the monitors do not
    all_early = sc.copy()
    all_early.loc[day < 14, "score"] = 100.0
    r = rm.monitor_rows(panel, FAKE, {"LEV6": all_early})
    assert (r["early_rate"] == 1.0).all()
    pd.testing.assert_frame_equal(r.drop(columns="early_rate"), base.drop(columns="early_rate"))
    # no alert on any later day: the outcome moves, the early rate and the monitors do not
    quiet_late = sc.copy()
    quiet_late.loc[day >= 14, "score"] = -100.0
    r2 = rm.monitor_rows(panel, FAKE, {"LEV6": quiet_late})
    assert r2["fail"].all() and (r2["log_error"] > 1.0).all() and not (r2["log_error"] == base["log_error"]).any()
    pd.testing.assert_frame_equal(r2.drop(columns=["log_error", "fail"]), base.drop(columns=["log_error", "fail"]))


def test_each_node_counts_its_14_days_from_its_own_first_deployment_day(rm):
    # I1's deployment data starts three days after N1's: its first 14 days are 12-25 March, N1's are 9-22 March
    panel = _drop_days(_panel({"N1": 1.0, "I1": 0.0}, seed=35), "I1", [dt.date(2026, 3, d) for d in (9, 10, 11)])
    sc = _scores(panel, {"N1": 1.5, "I1": 0.0}, seed=36)
    rows = rm.monitor_rows(panel, FAKE, {"LEV6": sc}).set_index("node")
    day = _day_since_first(panel, FAKE)
    outcome = _independent_outcome(sc)
    for node, key in (("N1", ("NYISO", "N1")), ("I1", ("ISONE", "I1"))):
        g = panel[panel["node"] == node]
        assert rows.loc[node, "n_early"] == int((day.loc[g.index] < 14).sum())
        assert rows.loc[node, "n_late"] == outcome[key]["n_late"]
        assert rows.loc[node, "early_rate"] == pytest.approx(outcome[key]["early_rate"])
        assert rows.loc[node, "log_error"] == pytest.approx(outcome[key]["log_error"])
        assert bool(rows.loc[node, "fail"]) == outcome[key]["fail"]
    # N1 has 9-29 March (late days 23-29, seven of them); I1 has 12-29 March (late days 26-29, four of them)
    assert rows.loc["N1", "n_late"] == 7 * 24 and rows.loc["I1", "n_late"] == 4 * 24


@pytest.mark.parametrize("case", ["outage", "short window"])
def test_monitor_rows_names_the_node_with_fewer_than_14_early_days(rm, case):
    if case == "outage":            # N1 has no rows on 15 March, inside its first 14 days: 13 early days
        analysis = FAKE
        panel = _drop_days(_panel({"N1": 1.0, "I1": 0.0}, seed=37), "N1", [dt.date(2026, 3, 15)])
        found = r"FAKE.*LEV6.*NYISO/N1.*only 13 early deployment day"
    else:                           # the deployment window ends after 10 days
        analysis = _fake_analysis(("2025-03-03", "2025-03-30"), ("2026-03-09", "2026-03-18"), id="TEN")
        panel = _panel({"N1": 1.0, "I1": 0.0}, seed=37)
        found = r"TEN.*LEV6.*ISONE/I1.*only 10 early deployment day"
    scores = {"LEV6": _scores(panel, {"N1": 1.5, "I1": 0.0}, analysis=analysis)}
    with pytest.raises(ValueError, match=found) as err:
        rm.monitor_rows(panel, analysis, scores)
    assert isinstance(err.value.__cause__, ValueError)


@pytest.mark.parametrize("case", ["every node", "one node"])
def test_monitor_rows_names_the_node_with_no_day_after_the_first_14(rm, case):
    panel = _panel({"N1": 1.0, "I1": 0.0}, seed=38)
    if case == "every node":        # the deployment window is exactly 14 days long
        analysis = _fake_analysis(("2025-03-03", "2025-03-30"), ("2026-03-09", "2026-03-22"), id="NOLATE")
        found = r"NOLATE.*LEV6.*ISONE/I1.*no deployment day after the first 14"
    else:                           # only N1 stops after its 14th day; it must be reported, not skipped quietly
        analysis = FAKE
        panel = _drop_days(panel, "N1", [dt.date(2026, 3, d) for d in range(23, 30)])
        found = r"FAKE.*LEV6.*NYISO/N1.*no deployment day after the first 14"
    scores = {"LEV6": _scores(panel, {"N1": 1.5, "I1": 0.0}, analysis=analysis)}
    with pytest.raises(ValueError, match=found) as err:
        rm.monitor_rows(panel, analysis, scores)
    assert isinstance(err.value.__cause__, ValueError)


def test_a_node_the_scores_do_not_cover_is_skipped_not_reported(rm):
    panel = _panel({"N1": 1.0, "I1": 0.0}, seed=39)
    sc = _scores(panel, {"N1": 1.5, "I1": 0.0}, seed=40)
    rows = rm.monitor_rows(panel, FAKE, {"LEV6": sc[sc["node"] == "N1"]})
    assert rows["node"].tolist() == ["N1"]


# ---- strict=False (controller Ruling 104(d)) ----------------------------------------------------------
# The sealed run cannot be repeated, so one node with too little data must not cost it a whole family of tests: with
# strict=False such a node is skipped and counted. The default is untouched.

def _outage_case(seed=37):
    panel = _drop_days(_panel({"N1": 1.0, "I1": 0.0}, seed=seed), "N1", [dt.date(2026, 3, 15)])   # N1: 13 early days
    return panel, {"LEV6": _scores(panel, {"N1": 1.5, "I1": 0.0}, analysis=FAKE)}


def test_strict_is_the_default_and_strict_false_changes_nothing_when_every_node_can_be_monitored(rm):
    panel = _panel({"N1": 1.0, "I1": 0.0}, seed=51)
    scores = {"LEV6": _scores(panel, {"N1": 1.5, "I1": 0.0})}
    default, strict = rm.monitor_rows(panel, FAKE, scores), rm.monitor_rows(panel, FAKE, scores, strict=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)                      # nothing is skipped, so nothing is said
        lenient = rm.monitor_rows(panel, FAKE, scores, strict=False)
    pd.testing.assert_frame_equal(default, strict)
    pd.testing.assert_frame_equal(default, lenient)
    assert list(lenient.columns) == rm.ROW_COLUMNS and len(lenient) == 2
    assert "skipped_nodes" not in default.attrs and default.attrs == strict.attrs      # the default frame is as it was
    assert lenient.attrs["skipped_nodes"] == [] and lenient.attrs["placebo_unavailable"] == default.attrs["placebo_unavailable"]


def test_strict_false_counts_a_node_the_blocks_score_frame_does_not_cover(rm):
    """Controller Ruling 109, M3. The strict default passes over such a node without a word (the test above, unchanged);
    strict=False lists it in skipped_nodes and in the one warning, per block, and the rows it returns are the same."""
    panel = _panel({"N1": 1.0, "I1": 0.0}, seed=39)
    sc = _scores(panel, {"N1": 1.5, "I1": 0.0}, seed=40)
    scores = {"LEV6": sc, "DART3": sc[sc["node"] == "N1"]}                  # DART3's frame lacks ISONE/I1
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)                          # strict: nothing is said, nothing is listed
        strict = rm.monitor_rows(panel, FAKE, scores)
    assert "skipped_nodes" not in strict.attrs
    assert sorted(zip(strict["block"], strict["node"])) == [("DART3", "N1"), ("LEV6", "I1"), ("LEV6", "N1")]
    with pytest.warns(UserWarning, match=r"1 node-block\(s\) skipped.*DART3 ISONE/I1 \(absent from the DART3 score frame"):
        lenient = rm.monitor_rows(panel, FAKE, scores, strict=False)
    pd.testing.assert_frame_equal(lenient, strict)                           # skipping it is all that changes
    skipped = lenient.attrs["skipped_nodes"]
    assert [{k: d[k] for k in ("analysis", "block", "market", "node")} for d in skipped] == [
        {"analysis": "FAKE", "block": "DART3", "market": "ISONE", "node": "I1"}]
    assert "absent from the DART3 score frame" in skipped[0]["reason"]
    full = rm.monitor_rows(panel, FAKE, {"LEV6": sc, "DART3": sc}, strict=False)           # every node covered: nothing counted
    assert full.attrs["skipped_nodes"] == []


def test_strict_false_skips_and_counts_a_node_with_fewer_than_14_early_days(rm):
    panel, scores = _outage_case()
    with pytest.raises(ValueError, match=r"FAKE.*LEV6.*NYISO/N1.*only 13 early deployment day"):
        rm.monitor_rows(panel, FAKE, scores)                              # strict: raises, as before
    with pytest.warns(UserWarning, match=r"1 node-block\(s\) skipped.*LEV6 NYISO/N1 \(only 13 early deployment day"):
        rows = rm.monitor_rows(panel, FAKE, scores, strict=False)
    assert rows["node"].tolist() == ["I1"]
    skipped = rows.attrs["skipped_nodes"]
    assert len(skipped) == 1 and {k: skipped[0][k] for k in ("analysis", "block", "market", "node")} == {
        "analysis": "FAKE", "block": "LEV6", "market": "NYISO", "node": "N1"}
    assert "only 13 early deployment day" in skipped[0]["reason"]
    # the node that is monitored gets exactly the row it gets when it is the only node there is
    alone = rm.monitor_rows(panel[panel["node"] == "I1"], FAKE, {"LEV6": scores["LEV6"][scores["LEV6"]["node"] == "I1"]})
    pd.testing.assert_frame_equal(rows.reset_index(drop=True), alone.reset_index(drop=True))


def test_strict_false_skips_every_node_of_a_window_that_is_too_short_and_returns_the_empty_frame(rm):
    analysis = _fake_analysis(("2025-03-03", "2025-03-30"), ("2026-03-09", "2026-03-18"), id="TEN")     # 10 days
    panel = _panel({"N1": 1.0, "I1": 0.0}, seed=37)
    scores = {"LEV6": _scores(panel, {"N1": 1.5, "I1": 0.0}, analysis=analysis)}
    with pytest.warns(UserWarning, match=r"2 node-block\(s\) skipped"):
        rows = rm.monitor_rows(panel, analysis, scores, strict=False)
    assert rows.empty and list(rows.columns) == rm.ROW_COLUMNS
    assert sorted((d["market"], d["node"]) for d in rows.attrs["skipped_nodes"]) == [("ISONE", "I1"), ("NYISO", "N1")]
    assert all(d["analysis"] == "TEN" and "only 10 early deployment day" in d["reason"] for d in rows.attrs["skipped_nodes"])


@pytest.mark.parametrize("case", ["every node", "one node"])
def test_strict_false_skips_a_node_with_no_deployment_day_after_the_first_14(rm, case):
    panel = _panel({"N1": 1.0, "I1": 0.0}, seed=38)
    if case == "every node":                # the deployment window is exactly 14 days long
        analysis = _fake_analysis(("2025-03-03", "2025-03-30"), ("2026-03-09", "2026-03-22"), id="NOLATE")
        left = []
    else:                                   # only N1 stops after its 14th day
        analysis = FAKE
        panel = _drop_days(panel, "N1", [dt.date(2026, 3, d) for d in range(23, 30)])
        left = ["I1"]
    scores = {"LEV6": _scores(panel, {"N1": 1.5, "I1": 0.0}, analysis=analysis)}
    with pytest.warns(UserWarning, match="skipped"):
        rows = rm.monitor_rows(panel, analysis, scores, strict=False)
    assert rows["node"].tolist() == left
    skipped = rows.attrs["skipped_nodes"]
    assert len(skipped) == 2 - len(left) and all("no deployment day after the first 14" in d["reason"] for d in skipped)


@pytest.mark.parametrize("side", ["calibration", "deployment"])
def test_strict_false_skips_a_node_whose_window_is_empty_after_dropna(rm, side):
    panel = _panel({"N1": 1.0, "I1": 0.0})
    scores = {"LEV6": _scores(panel, {"N1": 1.5, "I1": 0.0})}
    window = FAKE.calib if side == "calibration" else FAKE.deploy
    panel.loc[(panel["node"] == "I1") & window.mask(panel), "da_lmp_lag_48h"] = np.nan
    with pytest.warns(UserWarning, match=rf"ISONE/I1 \(no {side} rows with complete LEV6 features"):
        rows = rm.monitor_rows(panel, FAKE, scores, strict=False)
    assert rows["node"].tolist() == ["N1"] and rows.attrs["skipped_nodes"][0]["node"] == "I1"


def test_strict_false_names_each_block_a_node_is_skipped_in(rm):
    panel, lev = _outage_case()
    scores = {"LEV6": lev["LEV6"], "DART3": lev["LEV6"]}                  # both blocks see the same thin node
    with pytest.warns(UserWarning, match="2 node-block"):
        rows = rm.monitor_rows(panel, FAKE, scores, strict=False)
    assert sorted((d["block"], d["node"]) for d in rows.attrs["skipped_nodes"]) == [("DART3", "N1"), ("LEV6", "N1")]
    assert sorted(zip(rows["block"], rows["node"])) == [("DART3", "I1"), ("LEV6", "I1")]


def test_strict_false_still_raises_for_every_other_failure(rm, monkeypatch):
    """Only the data-availability conditions are skipped; a bug or a classifier that cannot run still names its node."""
    panel, scores = _outage_case()
    monkeypatch.setattr(rm, "psi", lambda reference, current, bins=10: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(ValueError, match=r"monitor_rows\('FAKE', 'LEV6'\): node ISONE/I1: RuntimeError: boom") as err:
        rm.monitor_rows(panel, FAKE, scores, strict=False)
    assert isinstance(err.value.__cause__, RuntimeError)
    with pytest.raises(ValueError, match="monitor_rows: block 'G6' has no price-history features"):
        rm.monitor_rows(panel, FAKE, {"G6": scores["LEV6"]}, strict=False)


def test_strict_false_does_not_hide_the_domain_classifier_guard(rm, monkeypatch):
    panel = _panel({"N1": 1.0, "I1": 0.0}, seed=52)
    scores = {"LEV6": _scores(panel, {"N1": 1.5, "I1": 0.0})}
    real = rm.domain_auc

    def refuse(cal_X, dep_X, ids):
        raise ValueError("the classifier cannot run on these day groups")

    monkeypatch.setattr(rm, "domain_auc", refuse)
    with pytest.raises(ValueError, match=r"FAKE.*LEV6.*ISONE/I1.*the classifier cannot run"):
        rm.monitor_rows(panel, FAKE, scores, strict=False)
    monkeypatch.setattr(rm, "domain_auc", real)
    assert len(rm.monitor_rows(panel, FAKE, scores, strict=False)) == 2


# ---- placebo reference columns (Ruling 83) ----------------------------------------------------------
# The registered monitors compare 14 deployment days with the calibration window. A 14-day window of a slow
# feature (168-hour rolling means) is a localized excursion, so those statistics are high even without drift.
# The placebo computes the same three statistics inside the calibration window (the last 14 calibration days
# as a pseudo-deployment) to show that no-drift level per block. It is descriptive: never a monitor.

PLACEBO_COLS = ["W1_norm_placebo", "psi_placebo", "domain_auc_placebo"]


def _long_rows(rm, seeds, **panel_kwargs):
    """monitor_rows on a 12-week calibration window, for two nodes and each seed, stacked."""
    parts = []
    for seed in seeds:
        panel = _panel({"N1": 0.0, "I1": 0.0}, seed=seed, spans=LONG_SPANS, **panel_kwargs)
        parts.append(rm.monitor_rows(panel, LONG, {"LEV6": _scores(panel, {"N1": 0.0, "I1": 0.0}, analysis=LONG)}))
    return pd.concat(parts, ignore_index=True)


def test_placebo_split_takes_exactly_the_last_14_calibration_days(rm):
    days = [dt.date(2025, 3, 1) + dt.timedelta(days=k) for k in range(30)]
    cal = pd.DataFrame({"settle_date": np.repeat(days, 24), "x": np.arange(30 * 24)})
    pseudo_cal, pseudo_dep = rm.placebo_split(cal)
    assert sorted({*pseudo_dep["settle_date"]}) == days[-14:] and sorted({*pseudo_cal["settle_date"]}) == days[:16]
    assert len(pseudo_dep) == 14 * 24 and len(pseudo_cal) == 16 * 24
    assert not set(pseudo_cal.index) & set(pseudo_dep.index) and len(pseudo_cal) + len(pseudo_dep) == len(cal)
    # a day without rows is skipped: the last 14 days that have rows
    gap = cal[cal["settle_date"] != days[-3]]
    assert sorted({*rm.placebo_split(gap)[1]["settle_date"]}) == [d for d in days if d != days[-3]][-14:]
    # under 14 days: all of it is pseudo-deployment and nothing is left to calibrate against
    short = cal[cal["settle_date"].isin(days[:10]).to_numpy()]
    assert rm.placebo_split(short)[0].empty and len(rm.placebo_split(short)[1]) == len(short)


def test_the_placebo_uses_calibration_rows_only_with_the_last_14_days_as_pseudo_deployment(rm, monkeypatch):
    seen = []
    monkeypatch.setattr(rm, "domain_auc", lambda cal_X, dep_X, ids: seen.append((len(cal_X), len(dep_X), np.asarray(ids).copy())) or 0.5)
    panel = _panel({"N1": 1.0}, spans=LONG_SPANS)
    rm.monitor_rows(panel, LONG, {"LEV6": _scores(panel, {"N1": 1.0}, analysis=LONG)})
    g = panel[panel["node"] == "N1"]
    cal_dates = list(g.loc[LONG.calib.mask(g).to_numpy(), "settle_date"])
    last14 = set(sorted(set(cal_dates))[-14:])
    assert len(seen) == 2                                   # the monitors' call, then the placebo's
    n_cal, n_dep, ids = seen[1]
    expected = [_epoch_day(d) for d in cal_dates if d not in last14] + [_epoch_day(d) for d in cal_dates if d in last14]
    assert ids.tolist() == expected                         # earlier calibration days first, then the last 14
    assert n_dep == sum(d in last14 for d in cal_dates) and n_cal + n_dep == len(cal_dates)
    deployment_days = {_epoch_day(d) for d in g.loc[LONG.deploy.mask(g).to_numpy(), "settle_date"]}
    assert not deployment_days & set(ids.tolist())          # no deployment row anywhere in the placebo


def test_the_placebo_auc_is_about_one_half_for_iid_features(rm):
    rows = _long_rows(rm, seeds=(61, 62, 63))
    assert len(rows) == 6 and rows[PLACEBO_COLS].notna().all().all()
    assert abs(rows["domain_auc_placebo"].mean() - 0.5) < 0.07


def test_the_placebo_auc_is_high_for_slow_features_without_any_drift(rm):
    iid = _long_rows(rm, seeds=(61, 62, 63))
    slow = _long_rows(rm, seeds=(61, 62, 63), slow=168)            # 168-hour rolling means, no drift in the calibration window
    assert slow["domain_auc_placebo"].mean() > 0.8                 # about 0.9 on a 168-hour mean (30 seeds: 0.91)
    assert slow["domain_auc_placebo"].mean() > iid["domain_auc_placebo"].mean() + 0.25
    # the other two statistics show the same localized excursion
    assert slow["W1_norm_placebo"].mean() > 2 * iid["W1_norm_placebo"].mean()
    assert slow["psi_placebo"].mean() > 2 * iid["psi_placebo"].mean()


def test_the_placebo_is_nan_and_counted_when_fewer_than_five_pseudo_calibration_days_are_left(rm):
    def rows_for(cal_days):
        start = dt.date(2025, 3, 3)
        analysis = _fake_analysis((start.isoformat(), (start + dt.timedelta(days=cal_days - 1)).isoformat()),
                                  ("2026-03-09", "2026-03-29"), id="SMALL")
        panel = _panel({"N1": 1.0, "I1": 0.0}, spans=((start.isoformat(), cal_days), ("2026-03-09", 21)))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = rm.monitor_rows(panel, analysis, {"LEV6": _scores(panel, {"N1": 1.0, "I1": 0.0}, analysis=analysis)})
        return out, [str(w.message) for w in caught if "placebo" in str(w.message)]

    # 18 calibration days leave 4 after the last 14: no placebo, counted, reported, and the monitors are intact
    rows, notes = rows_for(18)
    assert rows[PLACEBO_COLS].isna().all().all() and rows[list(rm.MONITORS)].notna().all().all()
    assert rows.attrs["placebo_unavailable"] == {"cases": 2, "too_small": 2, "domain_guard": 0}
    assert len(notes) == 1 and "2 of 2" in notes[0]
    # 19 days leave exactly 5: the placebo is computed and nothing is reported
    rows, notes = rows_for(19)
    assert rows[PLACEBO_COLS].notna().all().all() and not notes
    assert rows.attrs["placebo_unavailable"] == {"cases": 2, "too_small": 0, "domain_guard": 0}


def test_a_placebo_domain_classifier_that_cannot_run_gives_nan_and_is_counted(rm, monkeypatch):
    calls = []

    def flaky(cal_X, dep_X, ids):
        calls.append(1)
        if len(calls) % 2 == 0:                              # every second call is the placebo's
            raise ValueError("domain_auc: contrived guard")
        return 0.5

    monkeypatch.setattr(rm, "domain_auc", flaky)
    panel = _panel({"N1": 1.0, "I1": 0.0}, spans=LONG_SPANS)
    with pytest.warns(UserWarning, match=r"placebo.*2 of 2"):
        rows = rm.monitor_rows(panel, LONG, {"LEV6": _scores(panel, {"N1": 1.0, "I1": 0.0}, analysis=LONG)})
    assert (rows["domain_auc"] == 0.5).all() and rows["domain_auc_placebo"].isna().all()
    assert rows["W1_norm_placebo"].notna().all() and rows["psi_placebo"].notna().all()
    assert rows.attrs["placebo_unavailable"] == {"cases": 2, "too_small": 0, "domain_guard": 2}


def test_input_statistics_is_strict_unless_asked_otherwise(rm):
    rng = np.random.default_rng(9)
    feats = ["a", "b"]

    def frame(first_day, n_days):
        days = [dt.date.fromisoformat(first_day) + dt.timedelta(days=k) for k in range(n_days) for _ in range(24)]
        return pd.DataFrame({"settle_date": days, "a": rng.normal(size=len(days)), "b": rng.normal(size=len(days))})

    cal, dep = frame("2025-03-01", 20), frame("2025-04-01", 3)      # 3 deployment days: the classifier cannot run
    with pytest.raises(ValueError, match="has no deployment rows"):
        rm.input_statistics(cal, dep, feats)
    w1, p, auc = rm.input_statistics(cal, dep, feats, strict=False)
    assert np.isnan(auc) and np.isfinite(w1) and np.isfinite(p)
    w1, p, auc = rm.input_statistics(cal, frame("2025-04-01", 14), feats)
    assert np.isfinite([w1, p, auc]).all()


def test_placebo_columns_never_enter_the_evaluation(rm):
    rng = np.random.default_rng(10)
    n = 12
    base = _eval_frame(W1_norm=rng.normal(size=n), PSI=rng.normal(size=n), domain_auc=rng.uniform(size=n),
                       early_rate=np.linspace(0.05, 0.2, n), log_error=rng.uniform(size=n))
    with_placebo = base.assign(W1_norm_placebo=rng.normal(size=n), psi_placebo=rng.uniform(size=n),
                               domain_auc_placebo=rng.uniform(size=n))
    garbage = with_placebo.assign(W1_norm_placebo=np.nan, psi_placebo=1e9, domain_auc_placebo=-1.0)
    evaluated = [rm.evaluate_monitors(f) for f in (base, with_placebo, garbage)]
    registered = ["analysis", "block", "monitor", "spearman", "spearman_p_one_sided", "auroc_fail", "n_used", "calib_days"]
    for ev in evaluated[1:]:
        pd.testing.assert_frame_equal(ev[registered], evaluated[0][registered])        # nothing registered moves
    assert rm.MONITORS == ("W1_norm", "PSI", "domain_auc", "early_rate")
    assert all(list(ev["monitor"]) == list(rm.MONITORS) for ev in evaluated)           # no row for a placebo


def test_evaluate_monitors_reports_the_placebo_median_per_monitor_and_block(rm):
    n = 9
    skewed = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, np.nan]                          # median 0.45 over 8 values, mean 0.45 -> skew it
    frame = _eval_frame(W1_norm=np.linspace(0, 1, n), early_rate=np.linspace(0.05, 0.2, n)).assign(
        W1_norm_placebo=[0.1, 0.1, 0.1, 0.2, 0.9, 5.0, 6.0, 7.0, np.nan],              # median 0.55, mean 2.4
        psi_placebo=skewed, domain_auc_placebo=[0.9] * 8 + [np.nan])
    other = frame.assign(block="B2", W1_norm_placebo=0.3, psi_placebo=np.nan, domain_auc_placebo=0.5)
    ev = rm.evaluate_monitors(pd.concat([frame, other], ignore_index=True))
    first = ev[ev["block"] == "B"].set_index("monitor")
    assert first.loc["W1_norm", "placebo_median"] == pytest.approx(0.55) and first.loc["W1_norm", "placebo_n"] == 8
    assert first.loc["PSI", "placebo_median"] == pytest.approx(0.45) and first.loc["PSI", "placebo_n"] == 8
    assert first.loc["domain_auc", "placebo_median"] == pytest.approx(0.9)
    assert np.isnan(first.loc["early_rate", "placebo_median"]) and np.isnan(first.loc["early_rate", "placebo_n"])
    second = ev[ev["block"] == "B2"].set_index("monitor")
    assert second.loc["W1_norm", "placebo_median"] == pytest.approx(0.3) and second.loc["W1_norm", "placebo_n"] == n
    assert np.isnan(second.loc["PSI", "placebo_median"]) and second.loc["PSI", "placebo_n"] == 0
    # no placebo columns at all (the brief's own frame has none): not available, not zero
    plain = rm.evaluate_monitors(_eval_frame(W1_norm=np.linspace(0, 1, n), early_rate=np.linspace(0.05, 0.2, n)))
    assert plain["placebo_median"].isna().all() and plain["placebo_n"].isna().all()


# ---- main ------------------------------------------------------------------------------------------

def test_main_writes_both_files_with_the_documented_columns(rm, monkeypatch, tmp_path):
    from es.blocks import BLOCKS
    panel = _panel({"N1": 1.0, "I1": 0.0}, seed=21)
    calls = []

    def fake_scores(p, a, block, estimator, seed, *args, **kw):
        calls.append((a.id, block, estimator, seed))
        return _scores(p, {"N1": 1.5, "I1": 0.0}, seed=len(calls))

    one, two, three = (SimpleNamespace(id=i, calib=FAKE.calib, deploy=FAKE.deploy) for i in ("S1", "S2", "S3"))
    monkeypatch.setattr(rm, "load_dev_raw", lambda: None)
    monkeypatch.setattr(rm, "build_panel", lambda raw: panel)
    monkeypatch.setattr(rm, "cached_scores", fake_scores)
    monkeypatch.setattr(rm, "ensure_dirs", lambda: None)
    monkeypatch.setattr(rm, "MON_DIR", tmp_path / "monitors")
    monkeypatch.setattr(rm, "ANALOGS", {"S1": one})
    monkeypatch.setattr(rm, "LEGACY", {"2024-2025": two, "25H1-26H1": three})
    monkeypatch.setattr(rm, "BLOCKS", {"G6": [], "LEV6": BLOCKS["LEV6"], "MID3": BLOCKS["MID3"]})
    rm.main()
    sc = pd.read_csv(tmp_path / "monitors" / "monitor_scores.csv")
    ev = pd.read_csv(tmp_path / "monitors" / "monitor_eval.csv")
    assert calls == [(a, b, "xgb", 42) for a in ("S1", "S2", "S3") for b in ("LEV6", "MID3")]
    assert list(sc.columns) == ["analysis", "block", "market", "node", "W1_norm", "PSI", "domain_auc",
                                "early_rate", "log_error", "fail", "n_early", "n_late", "calib_days",
                                "W1_norm_placebo", "psi_placebo", "domain_auc_placebo"]
    assert list(ev.columns) == ["analysis", "block", "monitor", "spearman", "spearman_p_one_sided", "auroc_fail",
                                "n_used", "calib_days", "placebo_median", "placebo_n"]
    assert len(sc) == 3 * 2 * 2 and len(ev) == 3 * 2 * 4
    assert (ev["n_used"] == 2).all()                 # two nodes per (analysis, block), none dropped
    assert (sc["calib_days"] == 28).all() and (ev["calib_days"] == 28).all()      # FAKE's calibration window: 3-30 March
