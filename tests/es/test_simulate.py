import datetime as dt
import math
import warnings

import numpy as np
import pandas as pd
import pytest

# WindowTemplate.build warns when the fit lacks a seasonal mean the window needs (the brief's own fixture, fitted
# on Jan-Jul and used for Jul-Sep, does). test_template_warns_... asserts the warning; elsewhere it is noise.
pytestmark = pytest.mark.filterwarnings("ignore:WindowTemplate.build")


def _days(start, n):
    return [start + dt.timedelta(days=i) for i in range(n)]


def _fake_node(phi=0.8, n_days=200, seed=0, pnl_phi=None):
    """The brief's fixture. Its P&L is iid noise; ``pnl_phi`` (not in the brief) adds a daily AR(1) component of that
    persistence and innovation SD 1 to the P&L, drawn after the score's so that the default draws are the brief's."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n_days)
    for t in range(1, n_days):
        x[t] = phi * x[t - 1] + rng.normal(0, 1)
    y = np.zeros(n_days)
    if pnl_phi is not None:
        for t in range(1, n_days):
            y[t] = pnl_phi * y[t - 1] + rng.normal(0, 1)
    rows = []
    for i, d in enumerate(_days(dt.date(2025, 1, 1), n_days)):
        for h in range(24):
            rows.append({"settle_date": d, "settle_hour": h, "score": x[i] + rng.normal(0, 0.5),
                         "pnl_open": y[i] + rng.normal(0, 2)})
    return pd.DataFrame(rows)


def test_fit_node_recovers_persistence():
    from es.simulate import fit_node
    p = fit_node(_fake_node(phi=0.8))
    assert 0.6 < p.phi_s < 0.95


def test_simulate_shapes_and_coupling_sign():
    from es.simulate import WindowTemplate, fit_node, simulate
    # The brief's fixture has iid P&L, which the fit used to read as a daily component (the hourly noise leaking into
    # the daily mean). With that leak corrected (Ruling 84a) there is no daily P&L for rho to couple to, so the
    # fixture carries one.
    params = fit_node(_fake_node(pnl_phi=0.5))
    tpl = WindowTemplate.build(params, _days(dt.date(2026, 7, 1), 92))
    rng = np.random.default_rng(1)
    s0, p0 = simulate(tpl, 0.0, rng)
    assert s0.shape == (92 * 24,) and p0.shape == (92 * 24,)
    corr = []
    for _ in range(30):
        s, p = simulate(tpl, 0.9, rng)
        corr.append(np.corrcoef(s.reshape(92, 24).mean(1), p.reshape(92, 24).mean(1))[0, 1])
    assert np.mean(corr) < -0.3          # skill: high score days are loss days


def test_he_size_close_to_nominal_under_no_skill():
    from es.simulate import fit_node, he_size_power
    params = {("M", f"N{k}"): fit_node(_fake_node(seed=k)) for k in range(4)}
    out = he_size_power(params, [_days(dt.date(2025, 7, 1), 92)], [_days(dt.date(2026, 7, 1), 92)],
                        rho=0.0, n_sim=60, alpha_tests=(0.05,), seed=3)
    assert out["reject_0.05"] <= 0.15


# ---------------------------------------------------------------------------------------------------------------------
# Beyond the brief's three tests. A node drawn from the model itself, with known parameters, is the reference.

def _true_mu(h, m, amp=3.0):
    return amp * np.sin(2 * np.pi * np.asarray(h) / 24) + 0.4 * (np.asarray(m) - 6.5)


def _true_nu(h, m):
    return -0.5 * np.cos(2 * np.pi * np.asarray(h) / 24) + 0.3 * np.sin(2 * np.pi * np.asarray(m) / 12)


def _hetero_sd(hours, wclass):
    """A known P&L noise SD by (hour, weekday class): 1.0 off-peak, up to 4.0 around hour 6 (where the seasonal score
    mean peaks, so the volatile hours are the high-propensity strata), and 40% lower on weekends."""
    hours = np.asarray(hours)
    return (1.0 + 3.0 * np.exp(-0.5 * ((hours - 6) / 2.5) ** 2)) * np.where(np.asarray(wclass) == 1, 0.6, 1.0)


def _weekend_mu(hours, wclass):
    """A known weekend effect on the score's seasonal mean: lower by 1.0 on average, with an hourly shape."""
    return np.asarray(wclass) * (-1.0 + 0.8 * np.cos(2 * np.pi * np.asarray(hours) / 24))


def _weekend_nu(hours, wclass):
    """A known weekend effect on the P&L's seasonal mean: lower (a larger loss) by 0.8 on average, with a shape."""
    return np.asarray(wclass) * (-0.8 + 0.4 * np.sin(2 * np.pi * np.asarray(hours) / 24))


def _known_node(seed, start=dt.date(2017, 1, 1), n_days=8 * 365, phi_s=0.8, sd_s=1.0, sd_eps=0.5,
                phi_p=0.5, sd_p=2.0, sd_eta=1.0, keep=None, eta=None, wk_mu=None, wk_nu=None):
    """One node drawn from the model with known parameters: seasonal mean (hour x month), daily AR(1) with innovation
    SD sd_s / sd_p, white hourly noise sd_eps / sd_eta. ``keep`` (a boolean mask over days) drops days. ``eta`` (a
    function of hour and weekday class) replaces the P&L noise SD sd_eta by a known heteroskedastic profile.
    ``wk_mu`` / ``wk_nu`` (functions of hour and weekday class) add a known weekday effect to the two seasonal means."""
    rng = np.random.default_rng(seed)

    def ar1(phi, sd):
        x = np.empty(n_days)
        x[0] = rng.normal(0, sd / np.sqrt(1 - phi ** 2))
        shocks = rng.normal(0, sd, n_days)
        for t in range(1, n_days):
            x[t] = phi * x[t - 1] + shocks[t]
        return x

    days = _days(start, n_days)
    months = np.repeat([d.month for d in days], 24)
    hours = np.tile(np.arange(24), n_days)
    wclass = np.repeat([int(d.weekday() >= 5) for d in days], 24)
    pnl_sd = sd_eta if eta is None else eta(hours, wclass)
    mean_s = _true_mu(hours, months) + (0.0 if wk_mu is None else wk_mu(hours, wclass))
    mean_p = _true_nu(hours, months) + (0.0 if wk_nu is None else wk_nu(hours, wclass))
    rows = pd.DataFrame({
        "settle_date": np.repeat(days, 24), "settle_hour": hours,
        "score": mean_s + np.repeat(ar1(phi_s, sd_s), 24) + rng.normal(0, sd_eps, n_days * 24),
        "pnl_open": mean_p + np.repeat(ar1(phi_p, sd_p), 24) + rng.normal(0, 1, n_days * 24) * pnl_sd})
    if keep is not None:
        rows = rows[np.repeat(np.asarray(keep), 24)].reset_index(drop=True)
    return rows


@pytest.fixture(scope="module")
def known_params():
    """Four nodes with distinct persistence, each fitted from a single year, as the real fit is. Their daily P&L
    component is the larger part of the P&L's day-level variance, so that rho has something to couple to."""
    from es.simulate import fit_node
    specs = [(0.6, 0.2), (0.7, 0.3), (0.8, 0.4), (0.85, 0.4)]
    return {("M", f"N{i}"): fit_node(_known_node(20 + i, dt.date(2025, 7, 1), 365, phi_s=a, phi_p=b,
                                                 sd_s=1.0, sd_eps=1.0, sd_p=1.5, sd_eta=3.0))
            for i, (a, b) in enumerate(specs)}


def test_fit_node_recovers_known_parameters():
    """Eight years, so that the (hour, month) cell means are estimated from ~240 days each; the daily P&L component is
    large against its hourly noise / sqrt(24), where the daily-mean series is an almost clean copy of the AR(1)."""
    from es.simulate import fit_node
    p = fit_node(_known_node(7))
    assert abs(p.phi_s - 0.8) < 0.06 and abs(p.phi_p - 0.5) < 0.06
    assert abs(p.sd_s / 1.0 - 1) < 0.06 and abs(p.sd_p / 2.0 - 1) < 0.06
    assert abs(p.sd_eps / 0.5 - 1) < 0.015 and abs(p.sd_eta / 1.0 - 1) < 0.015
    mu_err = [p.mu[(h, m)] - _true_mu(h, m) for (h, m) in p.mu]
    nu_err = [p.nu[(h, m)] - _true_nu(h, m) for (h, m) in p.nu]
    assert len(p.mu) == 24 * 12 and len(p.nu) == 24 * 12
    assert np.sqrt(np.mean(np.square(mu_err))) < 0.5      # sampling error of a cell mean: ~0.3 (persistent days)
    assert np.sqrt(np.mean(np.square(nu_err))) < 0.5
    assert np.corrcoef([p.mu[k] for k in sorted(p.mu)], [_true_mu(*k) for k in sorted(p.mu)])[0, 1] > 0.99


def test_fit_node_hourly_noise_uses_the_degrees_of_freedom_left_by_the_means_it_removed():
    """The hourly noise SD is taken from what is left after removing 288 cell means and 365 day means. Dividing by
    N - 1 instead reads about 4% low in a year of data, 50 standard errors off."""
    from es.simulate import fit_node
    est = [fit_node(_known_node(100 + i, dt.date(2025, 7, 1), 365, sd_eps=0.5, sd_eta=2.0)) for i in range(8)]
    for name, truth in (("sd_eps", 0.5), ("sd_eta", 2.0)):
        v = np.array([getattr(e, name) for e in est]) / truth
        assert abs(v.mean() - 1) < 0.01, (name, v.mean())


def test_fit_node_persistence_uses_only_consecutive_days():
    """30% of the days are missing at random. Pairing neighbours in the remaining list treats a two-day gap as one day
    and understates phi; pairing only calendar-adjacent days does not."""
    from es.simulate import fit_node
    rng = np.random.default_rng(3)
    keep = rng.random(8 * 365) > 0.3
    full, gappy = fit_node(_known_node(11)), fit_node(_known_node(11, keep=keep))
    assert abs(gappy.phi_s - full.phi_s) < 0.04, (gappy.phi_s, full.phi_s)
    assert abs(gappy.phi_p - full.phi_p) < 0.06, (gappy.phi_p, full.phi_p)


def test_fit_node_does_not_depend_on_row_order():
    from es.simulate import fit_node
    rows = _known_node(5, n_days=2 * 365)
    a, b = fit_node(rows), fit_node(rows.sample(frac=1.0, random_state=0))
    for name in ("phi_s", "sd_s", "sd_eps", "phi_p", "sd_p", "sd_eta"):
        assert getattr(a, name) == pytest.approx(getattr(b, name), rel=1e-9), name
    assert a.mu.keys() == b.mu.keys()
    assert all(a.mu[k] == pytest.approx(b.mu[k]) for k in a.mu)


@pytest.mark.parametrize("column", ["settle_date", "settle_hour", "score", "pnl_open"])
def test_fit_node_refuses_a_missing_column_and_a_nan(column):
    from es.simulate import fit_node
    rows = _fake_node(n_days=40)
    with pytest.raises(ValueError, match=column):
        fit_node(rows.drop(columns=column))
    if column in ("score", "pnl_open", "settle_hour"):
        rows = rows.astype({column: float})
        rows.loc[5, column] = np.nan
        with pytest.raises(ValueError, match=f"NaN.*{column}"):
            fit_node(rows)


def test_fit_node_refuses_an_empty_frame_and_more_than_one_node():
    from es.simulate import fit_node
    rows = _fake_node(n_days=40)
    with pytest.raises(ValueError, match="empty"):
        fit_node(rows.iloc[:0])
    two = pd.concat([rows.assign(market="M", node="A"), rows.assign(market="M", node="B")])
    with pytest.raises(ValueError, match="one node"):
        fit_node(two)
    one = rows.assign(market="M", node="A")
    assert fit_node(one).phi_s == fit_node(rows).phi_s          # market and node columns are ignored for one node


def test_fit_node_handles_a_constant_pnl_column():
    """The H-B fit has no P&L: pnl_open = 0.0 for every row."""
    from es.simulate import fit_node
    p = fit_node(_fake_node(n_days=60).assign(pnl_open=0.0))
    assert (p.phi_p, p.sd_p, p.sd_eta) == (0.0, 0.0, 0.0) and set(p.nu.values()) == {0.0}
    assert p.phi_status["pnl"] == "no_signal" and (p.phi_p_raw, p.sd_p_raw) == (0.0, 0.0)


def test_template_warns_when_the_fit_lacks_a_seasonal_mean_and_stays_silent_otherwise():
    from es.simulate import WindowTemplate, fit_node
    params = fit_node(_fake_node(n_days=200))                                  # January to mid-July
    with pytest.warns(UserWarning, match="WindowTemplate.build.*96 of 144"):
        tpl = WindowTemplate.build(params, _days(dt.date(2026, 7, 1), 92))     # July, August, September
    aug = np.repeat([d.month for d in _days(dt.date(2026, 7, 1), 92)], 24) == 8
    assert np.allclose(tpl.mu_arr[aug], np.mean(list(params.mu.values())))     # the documented fallback level
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        WindowTemplate.build(params, _days(dt.date(2026, 1, 1), 31))


def test_ar1_path_has_the_stationary_variance_and_lag_one_correlation():
    from es.simulate import _sim_ar1
    for phi, sd in ((0.7, 1.0), (-0.4, 2.0), (0.0, 1.5)):
        x = _sim_ar1(40000, phi, sd, np.random.default_rng(2))
        assert x.var() == pytest.approx(sd ** 2 / (1 - phi ** 2), rel=0.05)
        assert np.corrcoef(x[:-1], x[1:])[0, 1] == pytest.approx(phi, abs=0.02)


def test_simulate_reproduces_the_fitted_moments(known_params):
    """On a long window the daily score component has the fitted persistence and stationary SD, and the hourly noise
    the fitted SD. (The daily mean also carries noise / sqrt(24), allowed for.)"""
    from es.simulate import WindowTemplate, simulate
    p = known_params[("M", "N2")]
    n = 1500
    tpl = WindowTemplate.build(p, _days(dt.date(2026, 1, 1), n))
    rng = np.random.default_rng(4)
    ac, daily_sd, hourly_sd = [], [], []
    for _ in range(8):
        s, _ = simulate(tpl, 0.0, rng)
        r = (s - tpl.mu_arr).reshape(n, 24)
        x = r.mean(1) - r.mean()
        ac.append(np.corrcoef(x[:-1], x[1:])[0, 1])
        daily_sd.append(x.std())
        hourly_sd.append(np.sqrt(((r - r.mean(1, keepdims=True)) ** 2).sum() / (n * 23)))
    assert np.mean(hourly_sd) == pytest.approx(p.sd_eps, rel=0.02)
    assert np.mean(ac) == pytest.approx(p.phi_s, abs=0.03)
    assert np.mean(daily_sd) == pytest.approx(np.sqrt(p.sd_s ** 2 / (1 - p.phi_s ** 2) + p.sd_eps ** 2 / 24), rel=0.08)


def test_simulate_refuses_a_rho_outside_minus_one_to_one(known_params):
    from es.simulate import WindowTemplate, simulate
    tpl = WindowTemplate.build(known_params[("M", "N0")], _days(dt.date(2026, 7, 1), 31))
    for rho in (1.2, -1.01):
        with pytest.raises(ValueError, match="rho"):
            simulate(tpl, rho, np.random.default_rng(0))
    simulate(tpl, 1.0, np.random.default_rng(0))
    simulate(tpl, -1.0, np.random.default_rng(0))


# --- heteroskedastic P&L noise (Ruling 55g) ------------------------------------------------------------------------

def _wclass(day):
    return int(day.weekday() >= 5)


@pytest.fixture(scope="module")
def hetero_params():
    """Four nodes whose P&L noise is known to be heteroskedastic by (hour, weekday class): the volatile hours are the
    high-propensity ones, so the calendar strata differ in loss volatility. Each fitted from one year."""
    from es.simulate import fit_node
    return {("M", f"H{i}"): fit_node(_known_node(60 + i, dt.date(2025, 7, 1), 365, phi_s=a, phi_p=b, sd_s=1.0,
                                                  sd_eps=1.0, sd_p=1.0, eta=lambda h, w: 1.5 * _hetero_sd(h, w)))
            for i, (a, b) in enumerate([(0.6, 0.2), (0.7, 0.3), (0.8, 0.4), (0.85, 0.4)])}


def test_fit_node_recovers_known_heteroskedastic_pnl_noise():
    """Eight years, P&L noise SD known for each of the 48 (hour, weekday class) cells (1.0 to 4.0, 40% lower on
    weekends): each cell SD is recovered, the contrast between hours is not blurred by the day means the fit removes,
    and the cells average to the pooled noise variance."""
    from es.simulate import fit_node
    p = fit_node(_known_node(31, eta=_hetero_sd))
    truth = {(h, w): float(_hetero_sd(h, w)) for h in range(24) for w in (0, 1)}
    assert set(p.sd_eta_cell) == set(truth)
    rel = np.array([p.sd_eta_cell[k] / truth[k] - 1 for k in truth])
    assert np.abs(rel).max() < 0.10 and np.sqrt(np.mean(rel ** 2)) < 0.035 and abs(rel.mean()) < 0.015
    quiet, loud = p.sd_eta_cell[(20, 0)], p.sd_eta_cell[(6, 0)]
    assert loud / quiet == pytest.approx(truth[(6, 0)] / truth[(20, 0)], rel=0.08)       # day-mean mixing would shrink it
    assert p.sd_eta_cell[(6, 1)] / p.sd_eta_cell[(6, 0)] == pytest.approx(0.6, rel=0.06)   # weekend / weekday
    days = _days(dt.date(2017, 1, 1), 8 * 365)
    n_cell = {(h, w): sum(_wclass(d) == w for d in days) for h in range(24) for w in (0, 1)}
    pooled = np.sqrt(sum(n_cell[k] * p.sd_eta_cell[k] ** 2 for k in n_cell) / sum(n_cell.values()))
    assert pooled == pytest.approx(p.sd_eta, rel=0.02)


def test_fit_node_one_year_heteroskedastic_noise_is_unbiased_in_each_weekday_class():
    """A weekend cell's mean comes from about 9 days, a weekday cell's from about 22, so the share of variance the means
    absorb differs by class (11% against 4.5%). With one absorption factor for both the weekend SDs read 2.6% low and the
    weekday SDs 1.4% high; with none, 5.9% and 2.0% low. Separately by class the estimator is unbiased."""
    from es.simulate import fit_node
    err = {0: [], 1: []}
    for i in range(8):
        p = fit_node(_known_node(80 + i, dt.date(2025, 7, 1), 365, eta=_hetero_sd))
        for w in (0, 1):
            err[w].append(np.mean([p.sd_eta_cell[(h, w)] / float(_hetero_sd(h, w)) - 1 for h in range(24)]))
    assert abs(np.mean(err[0])) < 0.012 and abs(np.mean(err[1])) < 0.012, {w: np.mean(v) for w, v in err.items()}


def test_residual_df_is_the_rows_less_the_rank_of_the_day_and_cell_indicators():
    """Exact on unbalanced designs: missing hours, repeated rows (a 25-hour day), and some days whose cells are pooled
    with other groups' (the fallback to the (hour, month) mean), against the rank of the indicator matrix."""
    from es.simulate import _residual_df
    rng = np.random.default_rng(0)
    for _ in range(60):
        n_days, n_hours, n_groups = int(rng.integers(3, 14)), int(rng.integers(2, 6)), 3
        group = rng.integers(0, n_groups, n_days)
        pooled = rng.random(n_groups) < 0.5
        day, hour = np.repeat(np.arange(n_days), n_hours), np.tile(np.arange(n_hours), n_days)
        keep = rng.random(len(day)) > 0.15
        keep[np.arange(n_days) * n_hours] = True                              # every day has at least one row
        day, hour = np.append(day[keep], day[0]), np.append(hour[keep], hour[0])   # one repeated row
        own_cell = np.where(pooled[group[day]], hour, n_hours * (1 + group[day]) + hour)   # (group, hour) or pooled (hour)
        _, cell = np.unique(own_cell, return_inverse=True)
        _, dcode = np.unique(day, return_inverse=True)
        design = np.zeros((len(day), dcode.max() + 1 + cell.max() + 1))
        design[np.arange(len(day)), dcode] = 1
        design[np.arange(len(day)), dcode.max() + 1 + cell] = 1
        assert _residual_df(dcode, cell) == len(day) - np.linalg.matrix_rank(design)


def test_fit_node_cell_noise_falls_back_to_the_pooled_sd_below_30_residuals():
    """50 days from a Monday: 36 weekdays (>= 30 per hour) and 14 weekend days (< 30). Only the weekday cells are
    estimated; a weekend day in a template takes the pooled SD."""
    from es.simulate import WindowTemplate, fit_node
    start = dt.date(2025, 7, 7)                                                          # a Monday
    rows = _known_node(5, start, 50, eta=_hetero_sd)
    p = fit_node(rows)
    assert set(p.sd_eta_cell) == {(h, 0) for h in range(24)}
    tpl = WindowTemplate.build(p, _days(start, 14))                                      # Mon..Sun, Mon..Sun
    eta = tpl.eta_arr.reshape(14, 24)
    assert np.allclose(eta[5], p.sd_eta) and np.allclose(eta[6], p.sd_eta) and np.allclose(eta[12], p.sd_eta)
    assert np.allclose(eta[0], [p.sd_eta_cell[(h, 0)] for h in range(24)])
    assert np.allclose(eta[11], [p.sd_eta_cell[(h, 0)] for h in range(24)])


def test_cell_noise_needs_exactly_30_residuals_per_cell():
    """15 whole weeks from a Monday give 30 weekend days, so 30 residuals in every weekend cell: estimated. Dropping the
    last Sunday leaves 29: the weekend cells fall back to the pooled SD."""
    from es.simulate import fit_node
    start = dt.date(2025, 7, 7)                                                          # a Monday
    p30 = fit_node(_known_node(9, start, 105, eta=_hetero_sd))
    p29 = fit_node(_known_node(9, start, 104, eta=_hetero_sd))
    assert set(p30.sd_eta_cell) == {(h, w) for h in range(24) for w in (0, 1)}
    assert set(p29.sd_eta_cell) == {(h, 0) for h in range(24)}


def test_node_params_without_cell_noise_is_homoskedastic():
    """A NodeParams built the old way (eight fields) has no cell SDs and every cell takes the pooled SD."""
    from es.simulate import NodeParams, WindowTemplate, simulate
    p = NodeParams({(h, 7): 0.0 for h in range(24)}, {(h, 7): 0.0 for h in range(24)}, 0.5, 1.0, 1.0, 0.3, 1.0, 2.5)
    assert p.sd_eta_cell == {}
    tpl = WindowTemplate.build(p, _days(dt.date(2026, 7, 1), 31))
    assert np.allclose(tpl.eta_arr, 2.5)
    a = simulate(tpl, 0.3, np.random.default_rng(3))
    b = simulate(tpl, 0.3, np.random.default_rng(3), hetero=True)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_template_carries_the_cell_sd_of_each_hour_and_day_class(hetero_params):
    from es.simulate import WindowTemplate
    p = hetero_params[("M", "H1")]
    days = _days(dt.date(2026, 7, 1), 92)
    tpl = WindowTemplate.build(p, days)
    assert tpl.eta_arr.shape == (92 * 24,)
    for i in (0, 3, 4, 5, 6, 40):
        w = _wclass(days[i])
        assert np.allclose(tpl.eta_arr[i * 24:(i + 1) * 24], [p.sd_eta_cell[(h, w)] for h in range(24)])


def test_simulate_hetero_draws_each_cells_own_pnl_noise_and_leaves_the_score_alone(hetero_params):
    from es.simulate import WindowTemplate, simulate
    p = hetero_params[("M", "H2")]
    n = 700
    days = _days(dt.date(2026, 1, 5), n)                                                  # starts on a Monday
    tpl = WindowTemplate.build(p, days)
    cls = np.array([_wclass(d) for d in days])
    s_hom, y_hom = simulate(tpl, 0.0, np.random.default_rng(8))
    s_het, y_het = simulate(tpl, 0.0, np.random.default_rng(8), hetero=True)
    assert np.array_equal(s_hom, s_het)                          # the score, and every draw, are unchanged

    def hourly_sd(y, w):
        r = (y - tpl.nu_arr).reshape(n, 24)
        dev = r - r.mean(1, keepdims=True)                       # the daily component is out
        return np.sqrt((dev[cls == w] ** 2).mean(0) * 24 / 23)   # to first order; the day mean mixes a little

    for w in (0, 1):
        want = np.array([p.sd_eta_cell[(h, w)] for h in range(24)])
        got = hourly_sd(y_het, w)
        assert np.corrcoef(got, want)[0, 1] > 0.97
        loud = want > 1.5 * np.median(want)
        assert loud.any() and np.allclose(got[loud], want[loud], rtol=0.2)
        flat = hourly_sd(y_hom, w)
        assert flat.max() / flat.min() < 1.5                     # homoskedastic: no hour stands out


def test_hetero_switch_keeps_the_total_noise_variance_and_changes_how_it_is_shared_across_strata(hetero_params):
    """Same average P&L noise variance in both scenarios, but across the calendar strata it is very uneven under
    hetero and flat otherwise: the failure mode of a loss-weighted excess is a volatility that differs by stratum."""
    from es.simulate import WindowTemplate
    for p in hetero_params.values():
        tpl = WindowTemplate.build(p, _days(dt.date(2026, 7, 1), 92))
        assert np.mean(tpl.eta_arr ** 2) == pytest.approx(p.sd_eta ** 2, rel=0.05)
        by_stratum = pd.Series(tpl.eta_arr ** 2).groupby(tpl.strata_decile).mean()
        assert by_stratum.max() / by_stratum.min() > 3.0
        within = pd.Series(tpl.eta_arr).groupby(tpl.strata_primary).nunique()
        assert (within == 1).all()                  # inside a primary stratum the volatility is one number


def test_hetero_switch_can_show_the_failure_mode_of_a_rotation_null_centred_away_from_the_observed_excess():
    """Ruling 55g: the simulation must be able to show that a loss volatility that differs across the calendar strata
    breaks the rotation test. An extreme case, to make the mechanism visible: the seasonal score mean moves a lot from
    month to month, and the loud P&L hours are the low-propensity ones. Rotating the alerts moves them across months
    into decile strata whose average volatility is not that of their own hour, so with hetero the rotation null's mean
    sits about a null SD below the observed excess; with the pooled noise it sits within a fifth of an SD. With the
    primary strata (one (hour, weekday class, month) cell each, Ruling 68) the volatility is constant inside a
    stratum and the null stays on the observed excess."""
    from es.bootstrap import rotation_excess
    from es.simulate import NodeParams, WindowTemplate, _cube
    mu = {(h, m): 3.0 * np.sin(2 * np.pi * h / 24) + 3.0 * (m - 6.5) for h in range(24) for m in range(1, 13)}
    nu = {(h, m): -0.5 * np.cos(2 * np.pi * h / 24) + 0.3 * np.sin(2 * np.pi * m / 12) for h in range(24) for m in range(1, 13)}
    cell = {(h, w): float((0.5 + 4.5 * np.exp(-0.5 * (((h + 12) % 24 - 6) / 2.5) ** 2)) * (0.6 if w else 1.0))
            for h in range(24) for w in (0, 1)}
    pooled = float(np.sqrt(np.mean(np.square(list(cell.values())))))
    p = NodeParams(mu, nu, 0.4, 1.0, 1.0, 0.1, 1.5, pooled, cell)
    dep, cal = _days(dt.date(2026, 7, 1), 92), _days(dt.date(2025, 7, 1), 92)
    tpl = lambda days: [WindowTemplate.build(p, days) for _ in range(4)]
    shift = {}
    for hetero in (False, True):
        for strata in ("decile", "primary"):
            rng = np.random.default_rng(5)
            gap, spread = [], []
            for _ in range(24):
                cube = _cube(tpl(dep), tpl(cal), 0.0, rng, 0.05, 1.0, hetero, "fit", strata)
                _, null = rotation_excess(cube)                                   # the default reference set
                gap.append(null.mean() - cube.selc_parts()[0])
                spread.append(null.std())
            shift[(hetero, strata)] = np.mean(gap) / np.mean(spread)
    assert shift[(False, "decile")] > -0.4 and shift[(True, "decile")] < -0.8, shift
    assert shift[(True, "primary")] > -0.4 and shift[(False, "primary")] > -0.4, shift


# --- the amended rotation test: nothing here reads the rotation statistic as a SELC -----------------------------

def _rotation_floor():
    """The smallest shift of rotation_test's own default reference set (Ruling 69: 1, the full cyclic group)."""
    import inspect
    from es.bootstrap import rotation_test
    return inspect.signature(rotation_test).parameters["min_shift"].default


def _n_null(window_days):
    """The grid of Rulings 55g and 69: each window's shift set holds 0 and the n_w = D - 2 * min_shift + 1 admissible
    shifts, so n_null = prod(n_w + 1) - 1, with min_shift the default of rotation_test. For the full cyclic group
    (min_shift 1) that is prod(D_w) - 1."""
    m = _rotation_floor()
    return int(np.prod([d - 2 * m + 2 for d in window_days])) - 1


def test_the_rotation_default_is_the_full_cyclic_group_and_n_null_is_the_product_of_the_window_lengths():
    """Ruling 69, landed in es.bootstrap: min_shift 1, so n_null = prod(D_w) - 1."""
    assert _rotation_floor() == 1
    assert _n_null([92]) == 91 and _n_null([92, 92]) == 92 * 92 - 1 and _n_null([30, 31, 92]) == 30 * 31 * 92 - 1


def test_he_size_power_leaves_the_reference_set_to_rotation_test(known_params, monkeypatch):
    """No shift floor is set in es.simulate: rotation_test is called with its own default (Ruling 69)."""
    import es.simulate as sim
    calls, real = [], sim.rotation_test

    def spy(cubes, *args, **kwargs):
        calls.append((args, kwargs))
        return real(cubes, *args, **kwargs)

    monkeypatch.setattr(sim, "rotation_test", spy)
    cal, dep = [_days(dt.date(2025, 7, 1), 92)], [_days(dt.date(2026, 7, 1), 92)]
    sim.he_size_power(known_params, cal, dep, 0.0, 3)
    assert calls == [((), {})] * 3


def test_mean_selc_is_the_pooled_selc_of_the_cubes_whatever_the_rotation_statistic(known_params):
    """Task 11 matches mean_selc to a SELC target. rotation_test's t_obs becomes the studentized excess, so mean_selc is
    computed from the cubes: pooled excess over pooled loss, the same ratio the old t_obs was."""
    from es.bootstrap import rotation_test
    from es.simulate import WindowTemplate, _cube, he_size_power
    nodes = sorted(known_params)
    cal = [_days(dt.date(2025, 7, 1), 92), _days(dt.date(2025, 10, 1), 92)]
    dep = [_days(dt.date(2026, 7, 1), 92), _days(dt.date(2026, 10, 1), 92)]
    out = he_size_power(known_params, cal, dep, 0.5, 3, seed=21)
    rng = np.random.default_rng(21)
    tpl = lambda days: [WindowTemplate.build(known_params[k], days) for k in nodes]
    want, want_t = [], []
    for _ in range(3):
        cubes = [_cube(tpl(d), tpl(c), 0.5, rng, 0.05, 1.0, False) for c, d in zip(cal, dep)]
        parts = [c.selc_parts() for c in cubes]
        want.append(sum(e for e, _ in parts) / sum(t for _, t in parts))
        res = rotation_test(cubes)
        want_t.append(res["t_obs"])
        if "selc_obs" in res:                         # the amended rotation_test reports the pooled SELC next to T
            assert res["selc_obs"] == pytest.approx(want[-1], rel=1e-12)
            assert res["t_obs"] != pytest.approx(want[-1], rel=1e-3)      # and T is a different statistic
    assert out["mean_selc"] == pytest.approx(np.mean(want), rel=1e-12)
    assert out["mean_t_obs"] == pytest.approx(np.mean(want_t), rel=1e-12)        # whatever statistic rotation_test reports
    assert out["mean_selc"] > 0.003                                          # rho = 0.5 carries real skill


def test_p_values_and_null_size_come_from_rotation_test_only(known_params):
    """he_size_power reads rotation_test's p (and n_null here), so a single window cannot reject below 1/(n_null + 1)."""
    from es.bootstrap import rotation_test
    from es.simulate import WindowTemplate, _cube, he_size_power
    nodes = sorted(known_params)
    days_c, days_d = _days(dt.date(2025, 7, 1), 92), _days(dt.date(2026, 7, 1), 92)
    tpl = lambda days: [WindowTemplate.build(known_params[k], days) for k in nodes]
    cube = _cube(tpl(days_d), tpl(days_c), 0.0, np.random.default_rng(1), 0.05, 1.0, False)
    res = rotation_test([cube])
    assert res["n_null"] == _n_null([92])
    assert 1 / (1 + res["n_null"]) <= res["p"] <= 1.0
    cube2 = _cube(tpl(days_d), tpl(days_c), 0.0, np.random.default_rng(2), 0.05, 1.0, False)
    pooled = rotation_test([cube, cube2])                       # the full grid: prod(n_w + 1) - 1
    assert pooled["n_null"] == _n_null([92, 92])
    assert 1 / (1 + pooled["n_null"]) <= pooled["p"] <= 1.0
    floor = 1 / (1 + _n_null([92])) - 1e-9
    out = he_size_power(known_params, [days_c], [days_d], 0.0, 40, alpha_tests=(floor,), seed=1)
    assert out[f"reject_{floor:g}"] == 0.0                                    # needs p below 1 / (n_null + 1): impossible


# --- the null and the alternative ------------------------------------------------------------------------------------

def _draws(known_params, rho, n, seed=11, node=("M", "N2"), phi_scale=1.0):
    """n quarters of one node: calibration threshold, deployment score and P&L, and the template."""
    from es.simulate import WindowTemplate, simulate
    p = known_params[node]
    tpl_c = WindowTemplate.build(p, _days(dt.date(2025, 7, 1), 92))
    tpl_d = WindowTemplate.build(p, _days(dt.date(2026, 7, 1), 92))
    rng = np.random.default_rng(seed)
    for _ in range(n):
        cs, _ = simulate(tpl_c, 0.0, rng, phi_scale)
        s, y = simulate(tpl_d, rho, rng, phi_scale)
        yield np.quantile(cs, 0.95), s, y, tpl_d


def test_null_carries_no_information_beyond_the_seasonal_means(known_params):
    """rho = 0: the score and the P&L, each less its seasonal mean, are uncorrelated hourly and daily, and the alert is
    uncorrelated with the P&L less its seasonal mean. rho = 0.9 makes all three clearly negative."""
    n = 600
    out = {}
    for rho in (0.0, 0.9):
        hourly, daily, alert = [], [], []
        for tau, s, y, tpl in _draws(known_params, rho, n):
            rs, ry = s - tpl.mu_arr, y - tpl.nu_arr
            hourly.append(np.corrcoef(rs, ry)[0, 1])
            daily.append(np.corrcoef(rs.reshape(92, 24).mean(1), ry.reshape(92, 24).mean(1))[0, 1])
            if (s >= tau).any():                         # a persistent node has quarters without a single alert
                alert.append(np.corrcoef((s >= tau).astype(float), ry)[0, 1])
        out[rho] = [(np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v))) for v in (hourly, daily, alert)]
    for mean, se in out[0.0]:
        assert abs(mean) < 4 * se, (mean, se)                 # no information under the null
    for mean, se in out[0.9]:
        assert mean < -10 * se, (mean, se)
    assert out[0.9][1][0] < -0.5


def test_p_and_l_does_not_depend_on_the_score_model_at_rho_zero(known_params):
    """Structural form of the null. Both draws consume the generator identically, so with the score's own parameters
    changed (persistence, SD, noise, seasonal mean) a P&L at rho = 0 must be bit-for-bit the same; at rho > 0 it must
    not be, because the score's daily component then enters it."""
    import dataclasses
    from es.simulate import WindowTemplate, simulate
    days = _days(dt.date(2026, 7, 1), 92)
    p = known_params[("M", "N1")]
    q = dataclasses.replace(p, phi_s=0.2, sd_s=3.0, sd_eps=0.1, mu={k: v + 5.0 for k, v in p.mu.items()})
    tp, tq = WindowTemplate.build(p, days), WindowTemplate.build(q, days)
    s_p, y_p = simulate(tp, 0.0, np.random.default_rng(9))
    s_q, y_q = simulate(tq, 0.0, np.random.default_rng(9))
    assert np.array_equal(y_p, y_q) and not np.allclose(s_p, s_q)
    _, y_p = simulate(tp, 0.5, np.random.default_rng(9))
    _, y_q = simulate(tq, 0.5, np.random.default_rng(9))
    assert not np.allclose(y_p, y_q)


def test_phi_scale_still_works_but_is_deprecated(known_params):
    """The multiplicative phi_scale of the brief is kept for old callers only: it warns, it still raises the persistence
    (and, with the innovation SD fixed, the stationary SD), it is capped at 0.99, and it cannot be mixed with the
    persistence upper end."""
    from es.simulate import WindowTemplate, simulate
    tpl = WindowTemplate.build(known_params[("M", "N1")], _days(dt.date(2026, 7, 1), 92))

    def lag1(scale, n=300):
        rng = np.random.default_rng(6)
        out = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            for _ in range(n):
                s, _ = simulate(tpl, 0.0, rng, scale)
                x = (s - tpl.mu_arr).reshape(92, 24).mean(1)
                out.append(np.corrcoef(x[:-1] - x.mean(), x[1:] - x.mean())[0, 1])
        return np.mean(out)

    assert lag1(1.2) > lag1(1.0) + 0.05
    with pytest.warns(DeprecationWarning, match="phi_scale.*persistence='upper'"):
        s, y = simulate(tpl, 0.0, np.random.default_rng(1), phi_scale=50.0)   # capped at 0.99, never explosive
    assert np.isfinite(s).all() and np.isfinite(y).all()
    with pytest.raises(ValueError, match="phi_scale"):
        simulate(tpl, 0.0, np.random.default_rng(1), phi_scale=1.1, persistence="upper")
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                         # the default 1.0 is silent
        simulate(tpl, 0.0, np.random.default_rng(1))


# --- the cubes the H-E test sees -------------------------------------------------------------------------------------

def _node_templates(known_params, nodes, days):
    from es.simulate import WindowTemplate
    return [WindowTemplate.build(known_params[k], days) for k in nodes]


def test_cube_decile_strata_are_the_calendar_propensity_deciles_taken_by_value(known_params):
    """The sensitivity (strata="decile"). Every row of a (node, hour, weekday class, month) cell has the same calendar
    propensity, so it must share a stratum: the rule of es.measures.add_calendar_deciles. Splitting tied rows by their
    order would put the first days of a month in one stratum and the last days in the next."""
    from es.measures import _deciles_by_value
    from es.simulate import _cube
    nodes = [("M", "N0"), ("M", "N3")]
    dep, cal = _days(dt.date(2026, 7, 1), 92), _days(dt.date(2025, 7, 1), 92)
    tpl_d, tpl_c = _node_templates(known_params, nodes, dep), _node_templates(known_params, nodes, cal)
    cube = _cube(tpl_d, tpl_c, 0.0, np.random.default_rng(0), 0.05, 1.0, False, "fit", "decile")
    month = np.array([d.month for d in dep])[cube.day_idx]
    wk = np.array([int(d.weekday() >= 5) for d in dep])[cube.day_idx]
    cells = pd.DataFrame({"cell": cube.cell_idx, "month": month, "wclass": wk, "sid": cube.sid})
    assert (cells.groupby(["cell", "month", "wclass"])["sid"].nunique() == 1).all()
    for k, t in enumerate(tpl_d):
        sid_k = cube.sid[cube.cell_idx // 24 == k]
        assert np.array_equal(sid_k - sid_k.min(), _deciles_by_value(pd.Series(t.mu_arr)).to_numpy())
    assert len(np.unique(cube.sid)) == sum(len(np.unique(t.strata_decile)) for t in tpl_d)      # no node shares a stratum


def test_cube_primary_strata_are_the_hour_weekday_class_month_cells_of_each_node(known_params):
    """Ruling 68: the registered strata are node x local hour x weekday class x month, the default of the cube."""
    from es.simulate import _cube
    nodes = [("M", "N0"), ("M", "N3")]
    dep, cal = _days(dt.date(2026, 7, 1), 92), _days(dt.date(2025, 7, 1), 92)
    tpl_d, tpl_c = _node_templates(known_params, nodes, dep), _node_templates(known_params, nodes, cal)
    cube = _cube(tpl_d, tpl_c, 0.0, np.random.default_rng(0), 0.05, 1.0)
    key = pd.DataFrame({"node": cube.cell_idx // 24, "hour": cube.cell_idx % 24,
                        "wclass": np.array([int(d.weekday() >= 5) for d in dep])[cube.day_idx],
                        "month": np.array([d.month for d in dep])[cube.day_idx], "sid": cube.sid})
    cell = key.groupby(["node", "hour", "wclass", "month"]).ngroup()
    assert (key.groupby(cell)["sid"].nunique() == 1).all() and (key.groupby("sid")["node"].nunique() == 1).all()
    assert (pd.Series(cell).groupby(key["sid"]).nunique() == 1).all()          # the same partition, both ways
    assert len(np.unique(cube.sid)) == 2 * 24 * 2 * 3                           # 2 nodes x 24 hours x 2 classes x 3 months
    assert (key.groupby("sid")["wclass"].nunique() == 1).all()                  # a stratum never mixes weekday and weekend


def test_primary_strata_are_es_measures_strata_primary_and_deciles_are_strata_decile(known_params):
    """The simulation's strata are the pipeline's (Ruling 68): grouping the cube's rows by es.measures.STRATA_PRIMARY
    gives the partition of the default strata, and the decile option is STRATA_DECILE's, node x calendar decile. Until
    measures redefines STRATA_PRIMARY (it was the decile) there is nothing to compare."""
    from es import measures
    from es.simulate import _cube
    if "cal_decile" in measures.STRATA_PRIMARY:
        pytest.skip("es.measures.STRATA_PRIMARY is still the decile definition (Ruling 68 has not landed there)")
    assert sorted(measures.STRATA_PRIMARY) == sorted(["market", "node", "settle_hour", "wclass", "month"])
    assert sorted(measures.STRATA_DECILE) == sorted(["market", "node", "cal_decile"])
    nodes = [("M", "N0"), ("M", "N3")]
    dep, cal = _days(dt.date(2026, 7, 1), 92), _days(dt.date(2025, 7, 1), 92)
    tpl_d, tpl_c = _node_templates(known_params, nodes, dep), _node_templates(known_params, nodes, cal)
    for strata, columns in (("primary", measures.STRATA_PRIMARY), ("decile", measures.STRATA_DECILE)):
        cube = _cube(tpl_d, tpl_c, 0.0, np.random.default_rng(0), 0.05, 1.0, False, "fit", strata)
        node = cube.cell_idx // 24
        rows = pd.DataFrame({"market": [nodes[k][0] for k in node], "node": [nodes[k][1] for k in node],
                             "settle_hour": cube.cell_idx % 24,
                             "wclass": np.array([int(d.weekday() >= 5) for d in dep])[cube.day_idx],
                             "month": np.array([d.month for d in dep])[cube.day_idx]})
        rows["cal_decile"] = np.concatenate([t.strata_decile for t in tpl_d])
        ours = pd.Series(cube.sid)
        theirs = rows.groupby(list(columns)).ngroup()
        assert (ours.groupby(theirs).nunique() == 1).all() and (theirs.groupby(ours).nunique() == 1).all(), strata


def test_cube_is_a_valid_complete_grid_with_the_registered_alert_rate(known_params):
    from es.simulate import _cube
    nodes = sorted(known_params)
    dep, cal = _days(dt.date(2026, 7, 1), 92), _days(dt.date(2025, 7, 1), 92)
    tpl_d, tpl_c = _node_templates(known_params, nodes, dep), _node_templates(known_params, nodes, cal)
    rng = np.random.default_rng(2)
    rates = []
    for _ in range(40):
        cube = _cube(tpl_d, tpl_c, 0.0, rng, 0.05, 1.0, False)
        rates.append(cube.alert.mean())
    assert cube.n_days == 92 and cube.n_cells == 24 * len(nodes) and len(cube.pnl) == 92 * 24 * len(nodes)
    assert len(np.unique(cube.node_idx)) == len(nodes)            # nodes are told apart, for SNET's node-days
    assert cube.snet_parts()[1] == 92 * len(nodes)                # one node-day per node and day
    # tau is the 95th percentile of an independent quarter. Its sampling noise grows with the persistence, and the tail
    # probability is convex, so the mean realised rate sits above 5% (here about 7%), as it does for P1 in real data
    assert 0.045 < np.mean(rates) < 0.09
    assert np.std(rates) > 0.003                                  # so the realised alert rate varies, unlike P6's exact rate
    # the cube is the one rotation_test takes
    from es.bootstrap import rotation_test
    assert rotation_test([cube])["n_null"] == _n_null([92])


def test_he_size_power_output_and_defaults(known_params):
    from es.simulate import he_size_power
    cal = [_days(dt.date(2025, 7, 1), 92), _days(dt.date(2025, 10, 1), 92)]
    dep = [_days(dt.date(2026, 7, 1), 92), _days(dt.date(2026, 10, 1), 92)]
    out = he_size_power(known_params, cal, dep, 0.0, 3)
    assert set(out) == {"reject_0.0166667", "reject_0.05", "mean_selc", "mean_t_obs", "n_sim", "rho", "phi_scale",
                        "hetero", "persistence", "strata"}
    assert (out["n_sim"], out["rho"], out["phi_scale"], out["hetero"], out["persistence"], out["strata"]) == \
        (3, 0.0, 1.0, False, "fit", "primary")
    again = he_size_power(known_params, cal, dep, 0.0, 3)
    assert out == again                                               # the registered default seed is fixed


def test_he_size_power_is_reproducible_and_seed_dependent(known_params):
    from es.simulate import he_size_power
    cal, dep = [_days(dt.date(2025, 7, 1), 92)], [_days(dt.date(2026, 7, 1), 92)]
    a = he_size_power(known_params, cal, dep, 0.3, 8, seed=5)
    assert a == he_size_power(known_params, cal, dep, 0.3, 8, seed=5)
    assert a["mean_selc"] != he_size_power(known_params, cal, dep, 0.3, 8, seed=6)["mean_selc"]


def test_he_power_rises_with_rho(known_params):
    """Common random numbers across rho (same seed): mean SELC increases strictly, and so does the rejection rate."""
    from es.simulate import he_size_power
    cal, dep = [_days(dt.date(2025, 7, 1), 92)], [_days(dt.date(2026, 7, 1), 92)]
    out = [he_size_power(known_params, cal, dep, r, 60, seed=7) for r in (0.0, 0.15, 0.3, 0.6)]
    selc = [o["mean_selc"] for o in out]
    r05 = [o["reject_0.05"] for o in out]
    assert selc == sorted(selc) and len(set(selc)) == 4 and selc[0] == pytest.approx(0, abs=0.004)
    assert r05 == sorted(r05) and r05[-1] > r05[0] + 0.5
    r3 = [o["reject_0.0166667"] for o in out]
    assert r3 == sorted(r3)


def test_he_size_power_validates_its_windows(known_params):
    from es.simulate import he_size_power
    q = lambda y, m, n=92: _days(dt.date(y, m, 1), n)
    with pytest.raises(ValueError, match="one calibration"):
        he_size_power(known_params, [q(2025, 7), q(2025, 10)], [q(2026, 7)], 0.0, 2)
    with pytest.raises(ValueError, match="rotation_test"):                     # one deployment day: nothing to rotate
        he_size_power(known_params, [q(2025, 7, 1)], [q(2026, 7, 1)], 0.0, 2)
    with pytest.raises(ValueError, match="strata"):
        he_size_power(known_params, [q(2025, 7)], [q(2026, 7)], 0.0, 2, strata="cells")
    with pytest.raises(ValueError, match="window"):
        he_size_power(known_params, [], [], 0.0, 2)
    with pytest.raises(ValueError, match="node"):
        he_size_power({}, [q(2025, 7)], [q(2026, 7)], 0.0, 2)
    with pytest.raises(ValueError, match="n_sim"):
        he_size_power(known_params, [q(2025, 7)], [q(2026, 7)], 0.0, 0)


# --- H-B -------------------------------------------------------------------------------------------------------------

def _hb_setup(params, nodes):
    from es.simulate import _hb_templates
    cal, dep = _days(dt.date(2025, 7, 1), 92), _days(dt.date(2026, 7, 1), 92)
    return cal, dep, _hb_templates({k: params[k] for k in nodes}, cal, dep)


def test_hb_blocks_are_exchangeable_at_delta_zero_and_alt_drifts_when_shifted(known_params):
    """By construction the alternative and base blocks share parameters at delta = 0 (equal portability): the
    median-over-nodes paired |log10 R| difference averages 0. A shifted alternative averages clearly above it."""
    from es.bootstrap import block_contrast_bootstrap
    from es.simulate import _hb_frames
    nodes = sorted(known_params)
    cal, dep, templates = _hb_setup(known_params, nodes)
    out = {}
    for delta in (0.0, 0.5):
        rng = np.random.default_rng(1)
        t = []
        for _ in range(80):
            alt, base = _hb_frames(nodes, templates, cal, dep, rng, delta, 1.0)
            t.append(block_contrast_bootstrap(alt, base, n_rep=2, seed=1)["t_hat"])
        out[delta] = (np.mean(t), np.std(t, ddof=1) / np.sqrt(len(t)))
    assert abs(out[0.0][0]) < 4 * out[0.0][1], out[0.0]
    assert out[0.5][0] > 0.03 and out[0.5][0] > 8 * out[0.5][1], out[0.5]


def test_hb_frames_carry_what_the_block_bootstrap_needs(known_params):
    from es.simulate import _hb_frames
    nodes = sorted(known_params)[:2]
    cal, dep, templates = _hb_setup(known_params, nodes)
    alt, base = _hb_frames(nodes, templates, cal, dep, np.random.default_rng(0), 0.0, 1.0)
    for f in (alt, base):
        assert {"market", "node", "split", "settle_date", "score"} <= set(f.columns)
        assert not f[["market", "node", "split", "settle_date", "score"]].isna().any().any()
        assert len(f) == 2 * (92 + 92) * 24 and set(f["split"]) == {"calib", "deploy"}
        assert set(f[f.split == "calib"]["settle_date"]) == set(cal)
        assert set(f[f.split == "deploy"]["settle_date"]) == set(dep)
    assert not np.allclose(alt["score"].to_numpy(), base["score"].to_numpy())        # independent draws


def test_hb_size_output_reproducibility_and_the_simulated_gap(known_params):
    from es.simulate import hb_size
    nodes = sorted(known_params)
    cal, dep = _days(dt.date(2025, 7, 1), 92), _days(dt.date(2026, 7, 1), 92)
    out = hb_size(known_params, cal, dep, 4, 30, seed=2)
    assert {"reject_0.0166667", "reject_0.05", "n_sim", "delta", "phi_scale", "mean_t_hat", "mean_node_gap",
            "persistence", "common_shift",
            "nodrift_fail_q95_0.5_2", "nodrift_fail_q95_0.67_1.5", "nodrift_fail_q95_0.8_1.25"} == set(out)
    assert (out["persistence"], out["common_shift"], out["delta"]) == ("fit", 0.0, 0.0)
    assert out == hb_size(known_params, cal, dep, 4, 30, seed=2)
    # the failure counts are counts of nodes, nested by band
    assert 0 <= out["nodrift_fail_q95_0.5_2"] <= out["nodrift_fail_q95_0.67_1.5"] <= out["nodrift_fail_q95_0.8_1.25"] <= len(nodes)
    shifted = hb_size(known_params, cal, dep, 4, 30, seed=2, delta=0.5)
    assert shifted["mean_node_gap"] > out["mean_node_gap"] + 0.03 and shifted["mean_t_hat"] > out["mean_t_hat"] + 0.03


def test_hb_simulated_gap_is_the_paired_log_error_difference_the_test_uses(known_params):
    """mean_t_hat is the average of the block contrast's own statistic (median over nodes of the paired difference);
    with one node the median is the node's difference, so it equals mean_node_gap."""
    from es.simulate import hb_size
    one = {("M", "N1"): known_params[("M", "N1")]}
    out = hb_size(one, _days(dt.date(2025, 7, 1), 92), _days(dt.date(2026, 7, 1), 92), 6, 10, seed=4, delta=0.3)
    assert out["mean_t_hat"] == pytest.approx(out["mean_node_gap"], abs=1e-12)


def test_hb_power_rises_with_delta_and_size_is_modest(known_params):
    from es.simulate import hb_size
    cal, dep = _days(dt.date(2025, 7, 1), 92), _days(dt.date(2026, 7, 1), 92)
    null = hb_size(known_params, cal, dep, 16, 100, seed=3)
    alt = hb_size(known_params, cal, dep, 16, 100, seed=3, delta=2.0)
    assert null["reject_0.05"] <= 0.30                 # 16 replications: loose; the size study proper is separate
    assert alt["reject_0.05"] >= 0.9 and alt["reject_0.0166667"] >= 0.8


def test_hb_size_validates_its_inputs(known_params):
    from es.simulate import hb_size
    cal, dep = _days(dt.date(2025, 7, 1), 92), _days(dt.date(2026, 7, 1), 92)
    with pytest.raises(ValueError, match="node"):
        hb_size({}, cal, dep, 2, 10)
    with pytest.raises(ValueError, match="n_sim"):
        hb_size(known_params, cal, dep, 0, 10)
    with pytest.raises(ValueError, match="window"):
        hb_size(known_params, [], dep, 2, 10)
    with pytest.raises(ValueError, match="market, node"):
        hb_size({"A": known_params[("M", "N0")]}, cal, dep, 2, 10)


def test_registered_defaults_are_pinned():
    import inspect
    from es.simulate import hb_size, he_size_power
    for fn in (he_size_power, hb_size):
        d = {k: v.default for k, v in inspect.signature(fn).parameters.items() if v.default is not inspect.Parameter.empty}
        assert d["alpha_tests"] == (0.05 / 3, 0.05) and d["seed"] == 20261022 and d["alpha"] == 0.05
        assert d["phi_scale"] == 1.0
    assert inspect.signature(hb_size).parameters["n_rep"].default == 299            # (B + 1) * alpha is whole (M-7)
    assert inspect.signature(hb_size).parameters["delta"].default == 0.0
    assert inspect.signature(hb_size).parameters["common_shift"].default == 0.0
    for fn in (he_size_power, hb_size):
        assert inspect.signature(fn).parameters["persistence"].default == "fit"
    assert inspect.signature(he_size_power).parameters["strata"].default == "primary"
    assert f"reject_{0.05 / 3:g}" == "reject_0.0166667"           # the key Task 11's tables read


# --- Ruling 63 (a): the persistence upper end -----------------------------------------------------------------------

def _clean_params(phi_s=0.7, phi_p=0.5, n_fit_days=100, sd_s=1.0, sd_p=1.0):
    """Hourly noise negligible: the daily mean of (series - seasonal mean) is the daily AR(1) component itself."""
    from es.simulate import NodeParams
    zero = {(h, m): 0.0 for h in range(24) for m in range(1, 13)}
    return NodeParams(zero, dict(zero), phi_s, sd_s, 1e-9, phi_p, sd_p, 1e-9, n_fit_days=n_fit_days)


@pytest.mark.parametrize("phi", [-0.3, 0.0, 0.5, 0.8, 0.95, 0.99])
@pytest.mark.parametrize("n_pairs", [30, 364])
def test_persistence_upper_end_is_the_formula_and_keeps_the_stationary_sd(phi, n_pairs):
    from es.simulate import persistence_upper
    sd = 1.7
    phi_u, sd_u = persistence_upper(phi, sd, n_pairs)
    assert phi_u == pytest.approx(min(0.99, phi + 1.96 * math.sqrt((1 - phi ** 2) / n_pairs)), rel=1e-12)
    assert phi_u >= phi and phi_u <= 0.99
    assert sd_u / math.sqrt(1 - phi_u ** 2) == pytest.approx(sd / math.sqrt(1 - phi ** 2), rel=1e-12)
    assert sd_u == pytest.approx(sd * math.sqrt((1 - phi_u ** 2) / (1 - phi ** 2)), rel=1e-12)


def test_persistence_upper_end_refuses_what_it_cannot_compute():
    from es.simulate import persistence_upper
    for n_pairs in (0, -3):
        with pytest.raises(ValueError, match="n_fit_days"):
            persistence_upper(0.5, 1.0, n_pairs)
    for phi in (1.0, -1.2):
        with pytest.raises(ValueError, match="phi"):
            persistence_upper(phi, 1.0, 100)


def test_fit_node_stores_the_number_of_day_pairs_its_ar1_used():
    from es.simulate import fit_node
    assert fit_node(_fake_node(n_days=200)).n_fit_days == 199
    keep = np.random.default_rng(3).random(3 * 365) > 0.3
    gappy = fit_node(_known_node(11, n_days=3 * 365, keep=keep))
    assert gappy.n_fit_days == int(np.sum(keep[1:] & keep[:-1]))        # only calendar-adjacent days form a pair
    assert gappy.n_fit_days < keep.sum() - 1


def test_upper_persistence_needs_the_pair_count_and_an_unambiguous_switch(known_params):
    from es.simulate import NodeParams, WindowTemplate, he_size_power, hb_size, simulate
    hand_built = _clean_params(n_fit_days=0)
    tpl = WindowTemplate.build(hand_built, _days(dt.date(2026, 7, 1), 31))
    with pytest.raises(ValueError, match="n_fit_days"):
        simulate(tpl, 0.0, np.random.default_rng(0), persistence="upper")
    simulate(tpl, 0.0, np.random.default_rng(0))                                 # "fit" never needs it
    ok = WindowTemplate.build(known_params[("M", "N0")], _days(dt.date(2026, 7, 1), 31))
    for bad in ("lower", "Upper", None):
        with pytest.raises(ValueError, match="persistence"):
            simulate(ok, 0.0, np.random.default_rng(0), persistence=bad)
    cal, dep = [_days(dt.date(2025, 7, 1), 92)], [_days(dt.date(2026, 7, 1), 92)]
    with pytest.raises(ValueError, match="persistence"):
        he_size_power(known_params, cal, dep, 0.0, 2, persistence="lower")
    with pytest.raises(ValueError, match="persistence"):
        hb_size(known_params, cal[0], dep[0], 2, 5, persistence="lower")
    assert NodeParams({}, {}, 0, 1, 1, 0, 1, 1).n_fit_days == 0


def test_simulate_upper_persistence_raises_lag_one_and_keeps_the_stationary_sd_of_both_components():
    from es.simulate import WindowTemplate, persistence_upper, simulate
    p = _clean_params()
    n = 4000
    tpl = WindowTemplate.build(p, _days(dt.date(2026, 1, 1), n))
    out = {}
    for mode in ("fit", "upper"):
        rng = np.random.default_rng(12)
        acc = {"s_ac": [], "s_sd": [], "p_ac": [], "p_sd": []}
        for _ in range(4):
            s, y = simulate(tpl, 0.0, rng, persistence=mode)
            for tag, series in (("s", s - tpl.mu_arr), ("p", y - tpl.nu_arr)):
                x = series.reshape(n, 24).mean(1)
                x = x - x.mean()
                acc[tag + "_ac"].append(np.corrcoef(x[:-1], x[1:])[0, 1])
                acc[tag + "_sd"].append(x.std())
        out[mode] = {k: float(np.mean(v)) for k, v in acc.items()}
    up_s, up_p = persistence_upper(0.7, 1.0, 100)[0], persistence_upper(0.5, 1.0, 100)[0]
    assert (up_s, up_p) == (pytest.approx(0.7 + 1.96 * math.sqrt(0.51 / 100)), pytest.approx(0.5 + 1.96 * math.sqrt(0.75 / 100)))
    assert out["fit"]["s_ac"] == pytest.approx(0.7, abs=0.03) and out["upper"]["s_ac"] == pytest.approx(up_s, abs=0.03)
    assert out["fit"]["p_ac"] == pytest.approx(0.5, abs=0.03) and out["upper"]["p_ac"] == pytest.approx(up_p, abs=0.03)
    assert out["upper"]["s_ac"] > out["fit"]["s_ac"] + 0.08 and out["upper"]["p_ac"] > out["fit"]["p_ac"] + 0.08
    assert out["upper"]["s_sd"] == pytest.approx(out["fit"]["s_sd"], rel=0.08)          # the stationary SD is unchanged
    assert out["upper"]["p_sd"] == pytest.approx(out["fit"]["p_sd"], rel=0.08)
    assert out["fit"]["s_sd"] == pytest.approx(1 / math.sqrt(1 - 0.7 ** 2), rel=0.08)


def test_upper_persistence_leaves_the_draws_and_the_hetero_switch_alone(known_params):
    """The generator is consumed identically, so score and P&L differ only through phi and the innovation SD."""
    from es.simulate import WindowTemplate, simulate
    tpl = WindowTemplate.build(known_params[("M", "N2")], _days(dt.date(2026, 7, 1), 92))
    a = simulate(tpl, 0.3, np.random.default_rng(4))
    b = simulate(tpl, 0.3, np.random.default_rng(4), persistence="upper")
    assert not np.array_equal(a[0], b[0]) and not np.array_equal(a[1], b[1])
    c = simulate(tpl, 0.3, np.random.default_rng(4), persistence="upper", hetero=True)
    assert np.array_equal(b[0], c[0]) and not np.array_equal(b[1], c[1])      # hetero touches only the P&L noise


def test_drivers_take_and_report_the_persistence_and_the_deprecated_phi_scale_warns(known_params):
    from es.simulate import hb_size, he_size_power
    cal, dep = [_days(dt.date(2025, 7, 1), 92)], [_days(dt.date(2026, 7, 1), 92)]
    fit = he_size_power(known_params, cal, dep, 0.0, 4, seed=3)
    up = he_size_power(known_params, cal, dep, 0.0, 4, seed=3, persistence="upper")
    assert (fit["persistence"], up["persistence"]) == ("fit", "upper") and fit["mean_t_obs"] != up["mean_t_obs"]
    hb_fit = hb_size(known_params, cal[0], dep[0], 3, 10, seed=3)
    hb_up = hb_size(known_params, cal[0], dep[0], 3, 10, seed=3, persistence="upper")
    assert (hb_fit["persistence"], hb_up["persistence"]) == ("fit", "upper") and hb_fit["mean_t_hat"] != hb_up["mean_t_hat"]
    with pytest.warns(DeprecationWarning, match="phi_scale"):
        he_size_power(known_params, cal, dep, 0.0, 2, phi_scale=1.1)
    with pytest.warns(DeprecationWarning, match="phi_scale"):
        hb_size(known_params, cal[0], dep[0], 2, 5, phi_scale=1.1)
    with pytest.raises(ValueError, match="phi_scale"):
        he_size_power(known_params, cal, dep, 0.0, 2, phi_scale=1.1, persistence="upper")
    with pytest.raises(ValueError, match="phi_scale"):
        hb_size(known_params, cal[0], dep[0], 2, 5, phi_scale=1.1, persistence="upper")


def test_hb_blocks_have_higher_lag_one_under_upper_persistence(known_params):
    """The score blocks of the H-B draws are more persistent at the upper end. (That the stationary SD is unchanged is
    tested on 4000-day windows above: a 92-day sample SD understates it more the more persistent the series is.)"""
    from es.simulate import _hb_frames, _hb_templates
    nodes = sorted(known_params)
    cal, dep = _days(dt.date(2025, 7, 1), 92), _days(dt.date(2026, 7, 1), 92)
    templates = _hb_templates(known_params, cal, dep)
    ac = {}
    for mode in ("fit", "upper"):
        rng = np.random.default_rng(8)
        out = []
        for _ in range(60):
            alt, _ = _hb_frames(nodes, templates, cal, dep, rng, 0.0, 1.0, persistence=mode)
            for k, (_, td) in zip(nodes, templates):
                sc = alt[(alt["node"] == k[1]) & (alt["split"] == "deploy")]["score"].to_numpy()
                x = (sc - td.mu_arr).reshape(92, 24).mean(1)
                x = x - x.mean()
                out.append(np.corrcoef(x[:-1], x[1:])[0, 1])
        ac[mode] = np.mean(out)
    assert ac["upper"] > ac["fit"] + 0.02, ac


# --- Ruling 63 (b): equal portability under common drift ------------------------------------------------------------

def test_hb_common_shift_moves_both_blocks_and_keeps_them_exchangeable(known_params):
    """common_shift shifts BOTH blocks' deployment scores by that many score SDs. Both blocks then drift by the same
    amount (equal portability under drift), so the contrast is still centred on 0, while delta adds the alt-only
    shift on top."""
    from es.bootstrap import block_contrast_bootstrap
    from es.simulate import _abs_log_r, _hb_frames, _hb_templates
    nodes = sorted(known_params)
    cal, dep = _days(dt.date(2025, 7, 1), 92), _days(dt.date(2026, 7, 1), 92)
    templates = _hb_templates(known_params, cal, dep)
    n_rows = 92 * 24
    stats = {}
    for label, shift, delta in (("none", 0.0, 0.0), ("common", 0.5, 0.0), ("common+alt", 0.5, 0.3)):
        rng = np.random.default_rng(1)
        t, e_alt, e_base = [], [], []
        for _ in range(80):
            alt, base = _hb_frames(nodes, templates, cal, dep, rng, delta, 1.0, common_shift=shift)
            t.append(block_contrast_bootstrap(alt, base, n_rep=2, seed=1)["t_hat"])
            e_alt.append(np.mean(_abs_log_r(_frame_rates(alt, 0.05), n_rows, 0.05)))
            e_base.append(np.mean(_abs_log_r(_frame_rates(base, 0.05), n_rows, 0.05)))
        diff = np.array(e_alt) - np.array(e_base)
        stats[label] = dict(t=np.mean(t), t_se=np.std(t, ddof=1) / np.sqrt(len(t)), alt=np.mean(e_alt),
                            base=np.mean(e_base), diff=np.mean(diff), diff_se=np.std(diff, ddof=1) / np.sqrt(len(diff)))
    none, common, both = stats["none"], stats["common"], stats["common+alt"]
    assert abs(none["t"]) < 4 * none["t_se"]                                  # no drift: centred
    assert abs(common["t"]) < 4 * common["t_se"], common                      # drift on both blocks: centred near 0
    assert abs(common["diff"]) < 4 * common["diff_se"], common                # the two blocks drift alike ...
    # ... and they do drift: a shift of 0.5 score SDs takes the 95th percentile's tail share from 5% to about 12.6%,
    # R = 2.5 and |log10 R| = 0.4, on top of the (persistence-sized) error that the unshifted blocks already show
    assert common["alt"] > none["alt"] + 0.08 and common["base"] > none["base"] + 0.08, (none, common)
    assert both["t"] > 0.03 and both["t"] > 8 * both["t_se"] and both["alt"] > both["base"]  # delta: on top


def test_hb_size_reports_the_common_shift_and_zero_is_the_old_null(known_params):
    from es.simulate import hb_size
    cal, dep = _days(dt.date(2025, 7, 1), 92), _days(dt.date(2026, 7, 1), 92)
    base = hb_size(known_params, cal, dep, 4, 20, seed=2)
    assert base == hb_size(known_params, cal, dep, 4, 20, seed=2, common_shift=0.0)
    drift = hb_size(known_params, cal, dep, 4, 20, seed=2, common_shift=0.5)
    assert drift["common_shift"] == 0.5 and base["common_shift"] == 0.0 and drift["delta"] == 0.0
    assert abs(drift["mean_node_gap"]) < 0.15                                  # the blocks drift alike: no gap on average


# --- Ruling 63 (c): calibrating delta to a target gap ---------------------------------------------------------------

def test_gap_interpolation_is_linear_monotone_and_refuses_targets_outside_the_grid():
    from es.simulate import _interp_delta
    grid, gaps = (0.1, 0.2, 0.4), [0.0, 0.1, 0.5]
    assert _interp_delta(grid, gaps, 0.05) == pytest.approx(0.15)
    assert _interp_delta(grid, gaps, 0.1) == pytest.approx(0.2)
    assert _interp_delta(grid, gaps, 0.3) == pytest.approx(0.3)
    assert _interp_delta(grid, gaps, 0.0) == pytest.approx(0.1) and _interp_delta(grid, gaps, 0.5) == pytest.approx(0.4)
    for bad in (-0.01, 0.51):
        with pytest.raises(ValueError, match="outside"):
            _interp_delta(grid, gaps, bad)
    # a dip in a measured curve is flattened by its running maximum: the answer is the first crossing
    grid4, dip = (0.1, 0.2, 0.3, 0.4), [0.0, 0.2, 0.15, 0.4]
    assert _interp_delta(grid4, dip, 0.3) == pytest.approx(0.35)
    assert _interp_delta(grid4, dip, 0.1) == pytest.approx(0.15)
    assert _interp_delta(grid4, dip, 0.2) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="outside"):
        _interp_delta(grid4, dip, 0.41)


def test_calibrate_delta_interpolates_a_known_increasing_gap_and_equals_hb_size(known_params):
    from es.simulate import calibrate_delta, hb_size
    three = {k: known_params[k] for k in sorted(known_params)[:3]}
    cal, dep = _days(dt.date(2025, 7, 1), 92), _days(dt.date(2026, 7, 1), 92)
    grid = (0.1, 0.3, 0.6)
    gaps = [hb_size(three, cal, dep, 6, 3, seed=5, delta=d)["mean_t_hat"] for d in grid]
    assert gaps[0] < gaps[1] < gaps[2] and gaps[0] > -0.05                      # the setup: a gap increasing in delta
    target = 0.5 * (gaps[0] + gaps[1])
    out = calibrate_delta(three, cal, dep, target, grid=grid, n_sim=6, n_rep=3, seed=5)
    assert set(out) >= {"delta", "grid_gaps", "grid", "target_gap", "statistic"}
    assert out["grid"] == list(grid) and out["target_gap"] == target
    assert out["grid_gaps"] == pytest.approx(gaps, abs=1e-14, rel=1e-12)       # the numbers hb_size reports as mean_t_hat
    assert out["delta"] == pytest.approx(0.2)                                   # the midpoint of the gaps: of the deltas
    inside = calibrate_delta(three, cal, dep, gaps[2] - 1e-9, grid=grid, n_sim=6, seed=5)
    assert 0.3 < inside["delta"] <= 0.6
    for bad in (gaps[0] - 0.01, gaps[2] + 0.01):
        with pytest.raises(ValueError, match="outside"):
            calibrate_delta(three, cal, dep, bad, grid=grid, n_sim=6, seed=5)


def test_calibrate_delta_follows_the_scenario_it_is_asked_about(known_params):
    from es.simulate import calibrate_delta, hb_size
    three = {k: known_params[k] for k in sorted(known_params)[:3]}
    cal, dep = _days(dt.date(2025, 7, 1), 92), _days(dt.date(2026, 7, 1), 92)
    for kw in ({"common_shift": 0.5}, {"persistence": "upper"}):
        lo = hb_size(three, cal, dep, 4, 3, seed=9, delta=0.2, **kw)["mean_t_hat"]
        hi = hb_size(three, cal, dep, 4, 3, seed=9, delta=0.5, **kw)["mean_t_hat"]
        got = calibrate_delta(three, cal, dep, 0.5 * (lo + hi), grid=(0.2, 0.5), n_sim=4, seed=9, **kw)
        assert got["grid_gaps"] == pytest.approx([lo, hi], abs=1e-14, rel=1e-12) and got["delta"] == pytest.approx(0.35)


def test_calibrate_delta_validates_its_inputs(known_params):
    from es.simulate import calibrate_delta
    cal, dep = _days(dt.date(2025, 7, 1), 92), _days(dt.date(2026, 7, 1), 92)
    with pytest.raises(ValueError, match="grid"):
        calibrate_delta(known_params, cal, dep, 0.05, grid=(0.3, 0.1), n_sim=2)
    with pytest.raises(ValueError, match="grid"):
        calibrate_delta(known_params, cal, dep, 0.05, grid=(0.3,), n_sim=2)
    with pytest.raises(ValueError, match="target_gap"):
        calibrate_delta(known_params, cal, dep, float("nan"), n_sim=2)
    with pytest.raises(ValueError, match="n_sim"):
        calibrate_delta(known_params, cal, dep, 0.05, n_sim=0)
    with pytest.raises(ValueError, match="node"):
        calibrate_delta({}, cal, dep, 0.05, n_sim=2)


# --- Ruling 64: weekday-class dimension in the seasonal means -------------------------------------------------------

def test_fit_node_recovers_a_known_weekday_effect_in_both_seasonal_means():
    """Ten years with a known weekend effect on the score (lower by 1.0 on average, with an hourly shape) and on the
    P&L (lower by 0.8): the (hour, weekday class, month) cell means recover it, and it does not leak into the daily
    component or the noise."""
    from es.simulate import fit_node
    p = fit_node(_known_node(41, n_days=10 * 365, wk_mu=_weekend_mu, wk_nu=_weekend_nu))
    assert len(p.mu_w) == 24 * 2 * 12 and len(p.nu_w) == 24 * 2 * 12 and p.fallback_cells == ()
    for name, cells, truth, tol in (("score", p.mu_w, lambda h, w, m: _true_mu(h, m) + _weekend_mu(h, w), (0.3, 0.5)),
                                    ("P&L", p.nu_w, lambda h, w, m: _true_nu(h, m) + _weekend_nu(h, w), (0.5, 0.8))):
        for w in (0, 1):
            err = [v - truth(h, w, m) for (h, ww, m), v in cells.items() if ww == w]
            assert np.sqrt(np.mean(np.square(err))) < tol[w], (name, w)
    contrast = {h: np.mean([p.mu_w[(h, 1, m)] - p.mu_w[(h, 0, m)] for m in range(1, 13)]) for h in range(24)}
    assert np.mean(list(contrast.values())) == pytest.approx(-1.0, abs=0.2)
    assert np.corrcoef([contrast[h] for h in range(24)], [float(_weekend_mu(h, 1)) for h in range(24)])[0, 1] > 0.95
    nu_contrast = np.mean([p.nu_w[(h, 1, m)] - p.nu_w[(h, 0, m)] for h in range(24) for m in range(1, 13)])
    assert nu_contrast == pytest.approx(-0.8, abs=0.25)
    assert abs(p.phi_s - 0.8) < 0.06 and abs(p.sd_s - 1.0) < 0.07 and abs(p.sd_eps / 0.5 - 1) < 0.015
    assert abs(p.phi_p - 0.5) < 0.06 and abs(p.sd_p / 2.0 - 1) < 0.07 and abs(p.sd_eta / 1.0 - 1) < 0.015
    # the (hour, month) means stay, for the cells that fall back: a day-count weighted average of the two classes
    for h, m in ((6, 7), (18, 2)):
        n_wk = sum(1 for d in _days(dt.date(2017, 1, 1), 10 * 365) if d.month == m and _wclass(d) == 1)
        n_all = sum(1 for d in _days(dt.date(2017, 1, 1), 10 * 365) if d.month == m)
        want = ((n_all - n_wk) * p.mu_w[(h, 0, m)] + n_wk * p.mu_w[(h, 1, m)]) / n_all
        assert p.mu[(h, m)] == pytest.approx(want, abs=1e-9)


def test_hourly_noise_df_counts_the_weekday_cell_means_it_estimated():
    """One year with weekday cells: 576 cell means and 24 (class, month) groups, not 288 and 12. The hourly noise SD
    stays unbiased (dividing by the two-way df would read 1.7% low)."""
    from es.simulate import fit_node
    est = [fit_node(_known_node(120 + i, dt.date(2025, 7, 1), 365, sd_eps=0.5, sd_eta=2.0,
                                wk_mu=_weekend_mu, wk_nu=_weekend_nu)) for i in range(8)]
    for name, truth in (("sd_eps", 0.5), ("sd_eta", 2.0)):
        v = np.array([getattr(e, name) for e in est]) / truth
        assert abs(v.mean() - 1) < 0.006, (name, v.mean())


def test_weekday_cells_with_fewer_than_5_days_fall_back_to_the_hour_month_mean_and_say_so():
    """July 2025 from the 1st to the 19th holds 5 weekend days (5, 6, 12, 13, 19): the weekend cells are estimated. To the
    18th it holds 4: they take the (hour, month) mean, and NodeParams records the cells."""
    from es.simulate import WindowTemplate, fit_node
    days = _days(dt.date(2026, 7, 1), 31)                                    # July 2026: 4, 5, 11, 12, ... are weekend days
    weekend = np.repeat([_wclass(d) for d in days], 24) == 1
    for last, estimated in ((19, True), (18, False)):
        p = fit_node(_known_node(3, dt.date(2025, 7, 1), last, wk_mu=_weekend_mu, wk_nu=_weekend_nu))
        tpl = WindowTemplate.build(p, days)
        hours = np.tile(np.arange(24), 31)
        if estimated:
            assert p.fallback_cells == () and all((h, 1, 7) in p.mu_w for h in range(24))
            assert np.allclose(tpl.mu_arr[weekend], [p.mu_w[(h, 1, 7)] for h in hours[weekend]])
        else:
            assert p.fallback_cells == tuple((h, 1, 7) for h in range(24))
            assert all((h, 1, 7) not in p.mu_w and (h, 1, 7) not in p.nu_w for h in range(24))
            assert np.allclose(tpl.mu_arr[weekend], [p.mu[(h, 7)] for h in hours[weekend]])
            assert np.allclose(tpl.nu_arr[weekend], [p.nu[(h, 7)] for h in hours[weekend]])
        assert np.allclose(tpl.mu_arr[~weekend], [p.mu_w[(h, 0, 7)] for h in hours[~weekend]])   # weekdays: 13+ days


def test_template_takes_the_weekday_class_cell_mean_and_the_hour_month_mean_otherwise():
    from es.simulate import NodeParams, WindowTemplate
    mu = {(h, 7): 3.0 for h in range(24)}
    nu = {(h, 7): -2.0 for h in range(24)}
    mu_w = {(h, 0, 7): 1.0 for h in range(24)}
    mu_w.update({(h, 1, 7): 5.0 for h in range(12)})                           # hours 12..23 of weekend days: no cell
    nu_w = {(h, 0, 7): -1.0 for h in range(24)}
    p = NodeParams(mu, nu, 0.5, 1.0, 1.0, 0.3, 1.0, 1.0, mu_w=mu_w, nu_w=nu_w,
                   fallback_cells=tuple((h, 1, 7) for h in range(12, 24)))
    days = _days(dt.date(2026, 7, 6), 7)                                       # Monday 6 July .. Sunday 12 July
    tpl = WindowTemplate.build(p, days)
    mu_days, nu_days = tpl.mu_arr.reshape(7, 24), tpl.nu_arr.reshape(7, 24)
    assert np.all(mu_days[:5] == 1.0) and np.all(nu_days[:5] == -1.0)           # weekdays: their own cell
    assert np.all(mu_days[5:, :12] == 5.0) and np.all(mu_days[5:, 12:] == 3.0)  # weekend: own cell, else (hour, month)
    assert np.all(nu_days[5:] == -2.0)                                          # no weekend P&L cell at all: (hour, month)
    # a (hour, month) mean old-style NodeParams (no weekday cells) is the same mean on every day
    old = NodeParams(mu, nu, 0.5, 1.0, 1.0, 0.3, 1.0, 1.0)
    assert np.all(WindowTemplate.build(old, days).mu_arr == 3.0)


def test_strata_mix_weekday_and_weekend_rows_when_the_weekend_shifts_the_score_mean():
    """The strata are the deciles, by value, of the score's seasonal mean over the window's rows, as the G6 calendar
    propensity strata are: with the weekend shifted by a fraction of the hourly pattern, deciles straddle both classes."""
    from es.measures import _deciles_by_value
    from es.simulate import NodeParams, WindowTemplate
    mu = {(h, m): 3.0 * np.sin(2 * np.pi * h / 24) for h in range(24) for m in range(1, 13)}
    mu_w = {(h, w, m): mu[(h, m)] - w * 1.0 for h in range(24) for w in (0, 1) for m in range(1, 13)}
    nu = {k: 0.0 for k in mu}
    p = NodeParams(mu, nu, 0.5, 1.0, 1.0, 0.3, 1.0, 1.0, mu_w=mu_w, nu_w={k: 0.0 for k in mu_w})
    days = _days(dt.date(2026, 7, 1), 92)
    tpl = WindowTemplate.build(p, days)
    assert np.array_equal(tpl.strata_decile, _deciles_by_value(pd.Series(tpl.mu_arr)).to_numpy())
    cls = np.repeat([_wclass(d) for d in days], 24)
    mixed = [s for s in np.unique(tpl.strata_decile) if len(np.unique(cls[tpl.strata_decile == s])) == 2]
    assert len(mixed) >= 5                                                      # most deciles hold both classes
    assert all(len(np.unique(cls[tpl.strata_primary == s])) == 1 for s in np.unique(tpl.strata_primary))   # primary: never
    flat = NodeParams(mu, nu, 0.5, 1.0, 1.0, 0.3, 1.0, 1.0)                    # no weekday effect: strata are the hours
    flat_tpl = WindowTemplate.build(flat, days)
    assert all(len(np.unique(cls[flat_tpl.strata_decile == s])) == 2 for s in np.unique(flat_tpl.strata_decile))


def _weekday_design(d_mu=0.0, d_nu=0.0, sd_ratio=1.0):
    """Hand-built node: hourly pattern in the score (amplitude 3), a weekend shift d_mu of its mean, a weekend shift
    d_nu of the P&L mean and the weekend P&L noise SD at sd_ratio times the weekday 3.0. Weak persistence and no month
    terms, so that the control (all three neutral) is near nominal and any distortion comes from the weekday class."""
    from es.simulate import NodeParams
    mu = {(h, m): 3.0 * np.sin(2 * np.pi * h / 24) for h in range(24) for m in range(1, 13)}
    nu = {(h, m): -0.5 * np.cos(2 * np.pi * h / 24) for h in range(24) for m in range(1, 13)}
    mu_w = {(h, w, m): mu[(h, m)] + w * d_mu for h in range(24) for w in (0, 1) for m in range(1, 13)}
    nu_w = {(h, w, m): nu[(h, m)] + w * d_nu for h in range(24) for w in (0, 1) for m in range(1, 13)}
    cell = {(h, w): 3.0 * (sd_ratio if w else 1.0) for h in range(24) for w in (0, 1)}
    pooled = float(np.sqrt(np.mean(np.square(list(cell.values())))))
    return NodeParams(mu, nu, 0.4, 1.0, 1.0, 0.1, 1.5, pooled, cell, 364, mu_w, nu_w, ())


STRONG_WEEKDAY = dict(d_mu=-1.0, d_nu=1.0, sd_ratio=0.6)       # weekends: fewer alerts, calmer, less loss


def test_weekday_contrasts_shift_the_rotation_null_when_the_deciles_mix_the_classes():
    """Ruling 64. Deciles of the score's seasonal mean can hold weekday rows of one hour and weekend rows of another.
    Rotating the alerts changes the class they land on, so in a design where weekends differ in alert propensity, loss
    level and loss SD, the rotation null is centred away from the observed excess (here about 0.8 null SDs below it);
    with no weekday effect it is centred on it. Ruling 68: with the registered primary strata, one (hour, weekday class,
    month) cell each, no stratum mixes the classes and the null stays on the observed excess."""
    from es.bootstrap import rotation_excess
    from es.simulate import WindowTemplate, _cube
    dep, cal = _days(dt.date(2026, 7, 1), 92), _days(dt.date(2025, 7, 1), 92)
    shift = {}
    for label, kw in (("control", {}), ("strong", STRONG_WEEKDAY)):
        p = _weekday_design(**kw)
        tpl = lambda days: [WindowTemplate.build(p, days) for _ in range(4)]
        td, tc = tpl(dep), tpl(cal)
        for strata in ("decile", "primary"):
            rng = np.random.default_rng(5)
            gap, spread = [], []
            for _ in range(30):
                cube = _cube(td, tc, 0.0, rng, 0.05, 1.0, True, "fit", strata)
                _, null = rotation_excess(cube)                                   # the default reference set
                gap.append(null.mean() - cube.selc_parts()[0])
                spread.append(null.std())
            shift[(label, strata)] = np.mean(gap) / np.mean(spread)
    assert shift[("control", "decile")] > -0.3 and shift[("strong", "decile")] < -0.5, shift
    assert shift[("strong", "decile")] < shift[("control", "decile")] - 0.4, shift
    assert shift[("strong", "primary")] > -0.3 and shift[("control", "primary")] > -0.3, shift


def test_pooled_size_with_weekday_contrasts_is_above_the_control_for_deciles_and_not_for_primary_strata():
    """The size check of Rulings 64 and 68 at modest n_sim (pooled A+B, 4 nodes, 60 simulations on the same random
    numbers). With the decile strata the control is near nominal and the weekday design rejects several times as often;
    with the registered primary strata the weekday design is not above nominal. The milder regime of the Task 9 finding
    (about 1.5x nominal pooled) needs a few hundred simulations to resolve, so it is reported, not asserted."""
    from es.simulate import he_size_power
    cal = [_days(dt.date(2025, 7, 1), 92), _days(dt.date(2025, 10, 1), 92)]
    dep = [_days(dt.date(2026, 7, 1), 92), _days(dt.date(2026, 10, 1), 92)]
    rate = {}
    for label, kw in (("control", {}), ("strong", STRONG_WEEKDAY)):
        params = {("M", f"N{i}"): _weekday_design(**kw) for i in range(4)}
        for strata in ("decile", "primary"):
            rate[(label, strata)] = he_size_power(params, cal, dep, 0.0, 60, hetero=True, seed=3, strata=strata)["reject_0.05"]
    assert rate[("control", "decile")] <= 0.15 and rate[("strong", "decile")] >= 0.2, rate
    assert rate[("strong", "decile")] > rate[("control", "decile")] + 0.1, rate
    assert rate[("strong", "primary")] <= 0.15 and rate[("control", "primary")] <= 0.15, rate


# --- Ruling 84a, I-1: the fitted daily AR(1) is corrected by indirect inference ----------------------------------------

def _persistence_node(seed, phi, sd_eps=1.0, **kw):
    """One year (the SIMFIT calendar) whose daily score component has persistence phi and stationary SD 1."""
    return _known_node(seed, dt.date(2025, 7, 1), 365, phi_s=phi, sd_s=np.sqrt(1 - phi ** 2), sd_eps=sd_eps,
                       phi_p=0.3, sd_p=1.0, sd_eta=2.0, **kw)


@pytest.mark.parametrize("phi", [0.3, 0.5, 0.65, 0.8, 0.9])
def test_the_fitted_score_persistence_and_stationary_sd_are_unbiased_in_a_year_of_data(phi):
    """The review's fitbias.py. The 576 cell means take out each (weekday class, month) group's mean of the daily
    component, and the hourly noise (sd_eps^2 / 24) leaks into the daily means, so the plain AR(1) fit read 0.8 as
    0.68 and 0.9 as 0.74, with a stationary SD of 0.88 and 0.76. Corrected by indirect inference, ten one-year fits
    average to the truth within 0.05 and to the stationary SD within 5%."""
    from es.simulate import fit_node
    fits = [fit_node(_persistence_node(500 + r, phi)) for r in range(10)]
    corrected, raw = np.mean([p.phi_s for p in fits]), np.mean([p.phi_s_raw for p in fits])
    stationary = np.mean([p.sd_s / np.sqrt(1 - p.phi_s ** 2) for p in fits])
    assert abs(corrected - phi) <= 0.05, (phi, corrected, raw)
    assert abs(stationary - 1.0) <= 0.05, (phi, stationary)
    if phi >= 0.5:
        assert raw < phi - 0.03 and corrected > raw + 0.03, (phi, corrected, raw)      # the bias was real


@pytest.mark.parametrize("phi", [0.3, 0.65, 0.85])
def test_the_fitted_pnl_persistence_and_stationary_sd_are_unbiased_in_a_year_of_data(phi):
    """The same for the P&L, whose daily mean carries more hourly noise: 2.5^2 / 24 = 0.26 against a daily variance of 1."""
    from es.simulate import fit_node
    fits = [fit_node(_known_node(700 + r, dt.date(2025, 7, 1), 365, phi_s=0.5, sd_s=0.8, sd_eps=1.0, phi_p=phi,
                                 sd_p=np.sqrt(1 - phi ** 2), sd_eta=2.5)) for r in range(10)]
    corrected, raw = np.mean([p.phi_p for p in fits]), np.mean([p.phi_p_raw for p in fits])
    stationary = np.mean([p.sd_p / np.sqrt(1 - p.phi_p ** 2) for p in fits])
    assert abs(corrected - phi) <= 0.05, (phi, corrected, raw)
    assert abs(stationary - 1.0) <= 0.05, (phi, stationary)
    assert raw < phi                                                                  # attenuated before the correction


def test_a_pnl_without_a_daily_component_is_not_given_one_by_its_hourly_noise():
    """The brief's fixture has iid P&L (SD 2). Its daily mean has SD 2 / sqrt(24) = 0.41, which the plain fit read as a
    daily component (the documented leak); the corrected fit finds almost none."""
    from es.simulate import fit_node
    p = fit_node(_fake_node(n_days=365))
    before = p.sd_p_raw / np.sqrt(1 - p.phi_p_raw ** 2)
    after = p.sd_p / np.sqrt(1 - p.phi_p ** 2)
    assert before > 0.35 and after < 0.2, (before, after)


def test_the_persistence_correction_is_deterministic_and_keeps_the_uncorrected_fit():
    from es.simulate import fit_node
    rows = _persistence_node(3, 0.8)
    a, b = fit_node(rows), fit_node(rows.copy())
    assert (a.phi_s, a.sd_s, a.phi_p, a.sd_p) == (b.phi_s, b.sd_s, b.phi_p, b.sd_p)       # bit for bit
    assert a.phi_status == b.phi_status == {"score": "corrected", "pnl": "corrected"}
    # the uncorrected estimates are the plain AR(1) fit of the daily residuals, recomputed independently here
    r = rows.assign(month=pd.to_datetime(rows["settle_date"]).dt.month,
                    wclass=(pd.to_datetime(rows["settle_date"]).dt.dayofweek >= 5).astype(int))
    for col, raw_phi, raw_sd in (("score", a.phi_s_raw, a.sd_s_raw), ("pnl_open", a.phi_p_raw, a.sd_p_raw)):
        res = r[col] - r.groupby(["settle_hour", "wclass", "month"])[col].transform("mean")
        x = res.groupby(r["settle_date"]).mean().sort_index().to_numpy()
        x = x - x.mean()
        phi = np.corrcoef(x[:-1], x[1:])[0, 1]
        assert raw_phi == pytest.approx(phi, rel=1e-9) and raw_sd == pytest.approx(np.std(x[1:] - phi * x[:-1]), rel=1e-9)
    assert a.phi_s > a.phi_s_raw                                                           # and the correction moved it up


def test_a_series_outside_the_range_of_the_correction_is_clipped_and_recorded_not_raised():
    """A daily component alternating in sign has a lag-one correlation near -1; a slow cycle (30 days, so the monthly
    group means leave it), near +1. Neither lies in the range the AR(1) map can reach, so the persistence is clipped to
    the end of the search range and the node says so."""
    from es import simulate as sim
    base = _persistence_node(4, 0.5)
    day = (pd.to_datetime(base["settle_date"]) - pd.Timestamp("2025-07-01")).dt.days.to_numpy()
    for label, shift, status, bound in (("alternating", 30.0 * np.where(day % 2 == 0, 1.0, -1.0), "clipped_low", sim._PHI_SEARCH[0]),
                                        ("slow cycle", 30.0 * np.sin(2 * np.pi * day / 30), "clipped_high", sim._PHI_SEARCH[1])):
        p = sim.fit_node(base.assign(score=base["score"] + shift))
        assert p.phi_status["score"] == status and p.phi_s == bound, (label, p.phi_status, p.phi_s)
        assert np.isfinite(p.sd_s) and p.sd_s > 0 and abs(p.phi_s_raw) > 0.9              # the uncorrected one is kept
        assert p.phi_status["pnl"] == "corrected"                                          # the other component is its own


def test_the_upper_end_is_taken_from_the_corrected_persistence():
    from es import simulate as sim
    p = sim.fit_node(_persistence_node(7, 0.8))
    assert p.phi_s > p.phi_s_raw + 0.03
    phi_s, sd_s, phi_p, sd_p = sim._daily_params(p, 1.0, "upper")
    want = sim.persistence_upper(p.phi_s, p.sd_s, p.n_fit_days)
    assert (phi_s, sd_s) == want and phi_s > p.phi_s
    assert sd_s / np.sqrt(1 - phi_s ** 2) == pytest.approx(p.sd_s / np.sqrt(1 - p.phi_s ** 2), rel=1e-12)


# --- Ruling 84a, M-1 to M-7 ----------------------------------------------------------------------------------------------

def test_the_drift_is_scaled_by_one_deterministic_sd_per_node_shared_by_both_blocks(known_params):
    """M-1. Scaling by each block's own realised SD made the shift random and different in the two blocks, adding
    variance the bootstrap cannot see. Now one number per node and window: the fitted stationary SD of the score,
    sqrt(var of the seasonal pattern + daily variance + hourly noise variance)."""
    from es.simulate import _hb_frames, _hb_templates, _score_sd, _simulate
    nodes = sorted(known_params)
    cal, dep = _days(dt.date(2025, 7, 1), 92), _days(dt.date(2026, 7, 1), 92)
    templates = _hb_templates(known_params, cal, dep)
    ref_alt, ref_base = _hb_frames(nodes, templates, cal, dep, np.random.default_rng(1), 0.0, 1.0)
    alt, base = _hb_frames(nodes, templates, cal, dep, np.random.default_rng(1), 0.3, 1.0, common_shift=0.5)
    for k, (_, td) in zip(nodes, templates):
        p, sd = known_params[k], _score_sd(td)
        assert sd == pytest.approx(np.sqrt(np.var(td.mu_arr) + p.sd_s ** 2 / (1 - p.phi_s ** 2) + p.sd_eps ** 2), rel=1e-12)

        def pick(frame, split):
            return frame[(frame["node"] == k[1]) & (frame["split"] == split)]["score"].to_numpy()

        assert np.allclose(pick(alt, "deploy") - pick(ref_alt, "deploy"), 0.8 * sd, rtol=0, atol=1e-10)    # common + delta
        assert np.allclose(pick(base, "deploy") - pick(ref_base, "deploy"), 0.5 * sd, rtol=0, atol=1e-10)  # common only
        assert np.array_equal(pick(alt, "calib"), pick(ref_alt, "calib"))                                  # calibration untouched
    p, (_, td) = known_params[nodes[0]], templates[0]
    rng = np.random.default_rng(2)
    realised = np.mean([np.std(_simulate(td, 0.0, rng)[0]) for _ in range(150)])
    assert realised == pytest.approx(_score_sd(td), rel=0.07)                              # the unit is the right size


def test_the_no_drift_failure_counts_are_only_reported_for_the_no_drift_null(known_params):
    """M-2. They are the no-drift context of the registered bands; in a drift or delta row every node fails them."""
    from es.simulate import hb_size
    cal, dep = _days(dt.date(2025, 7, 1), 92), _days(dt.date(2026, 7, 1), 92)
    keys = ["nodrift_fail_q95_0.5_2", "nodrift_fail_q95_0.67_1.5", "nodrift_fail_q95_0.8_1.25"]
    null = hb_size(known_params, cal, dep, 3, 10, seed=2)
    assert all(np.isfinite(null[k]) for k in keys)
    for kw in ({"common_shift": 0.5}, {"delta": 0.3}, {"common_shift": 0.25, "delta": 0.1}):
        row = hb_size(known_params, cal, dep, 3, 10, seed=2, **kw)
        assert all(np.isnan(row[k]) for k in keys), kw
        assert np.isfinite(row["reject_0.05"]) and np.isfinite(row["mean_t_hat"])


def test_calibrate_delta_matches_the_median_over_nodes_of_the_paired_gap_not_the_mean():
    """M-4. The H-B statistic is the median over nodes of the paired |log10 R| difference, so that is what the effect
    size is matched on (Task 11 passes 0.5 x the median paired AQ3 difference). With nodes as unlike as these two quiet
    ones and a very persistent one, the mean over nodes is a different number."""
    from es.simulate import calibrate_delta, hb_size
    P = {("M", "A"): _clean_params(phi_s=0.2, n_fit_days=364), ("M", "B"): _clean_params(phi_s=0.2, n_fit_days=364),
         ("M", "C"): _clean_params(phi_s=0.97, n_fit_days=364)}
    cal, dep = _days(dt.date(2025, 7, 1), 92), _days(dt.date(2026, 7, 1), 92)
    grid = (0.2, 0.6)
    rows = [hb_size(P, cal, dep, 24, 3, seed=8, delta=d) for d in grid]
    median_gaps, mean_gaps = [r["mean_t_hat"] for r in rows], [r["mean_node_gap"] for r in rows]
    assert max(abs(a - b) for a, b in zip(median_gaps, mean_gaps)) > 0.01               # the two statistics differ here
    out = calibrate_delta(P, cal, dep, 0.5 * sum(median_gaps), grid=grid, n_sim=24, seed=8)
    assert out["grid_gaps"] == pytest.approx(median_gaps, abs=1e-14, rel=1e-12)
    assert out["delta"] == pytest.approx(0.4, abs=1e-9) and "median" in out["statistic"]


def test_hb_size_default_bootstrap_size_makes_both_registered_levels_attainable():
    """M-7. With B replicates the smallest p-values are (1 + k) / (B + 1). At B = 299, (B + 1) alpha is whole for
    alpha/3 (5) and for 0.05 (15), so a valid test rejects at exactly those levels; at B = 200 alpha/3 would read
    3/201 = 0.0149, 10% low."""
    import inspect
    from es.simulate import hb_size
    b = inspect.signature(hb_size).parameters["n_rep"].default
    assert b == 299
    assert (b + 1) * (0.05 / 3) == pytest.approx(5.0, abs=1e-12) and (b + 1) * 0.05 == pytest.approx(15.0, abs=1e-12)
    assert "299" in hb_size.__doc__ and "attainable" in hb_size.__doc__


# --- Ruling 92a: the fast path of the H-B bootstrap ---------------------------------------------------------------------
# ``block_contrast_bootstrap`` takes ~1.5 s per simulation at 33 nodes, almost all of it in ``np.quantile`` inside
# ``_log_error`` (one call per node, block and replicate). ``_hb_bootstrap_fast`` computes the same numbers from the
# arrays the simulation draws; the gate is that it equals the registered function bit for bit, on every grid hb_size builds.

def _frame_rates(frame, alpha):
    """Per (market, node), the share of the deployment rows at or above tau (the 1 - alpha quantile of the node's
    calibration scores) over alpha, read off a score frame with pandas: the independent reference of ``_block_rates``."""
    tau = frame[frame["split"] == "calib"].groupby(["market", "node"])["score"].quantile(1 - alpha).rename("tau")
    dep = frame[frame["split"] == "deploy"].join(tau, on=["market", "node"])
    return (dep["score"] >= dep["tau"]).groupby([dep["market"], dep["node"]]).mean() / alpha


def _bits(x):
    return np.asarray(x, dtype=float).tobytes()


def _assert_same_bootstrap(fast, registered):
    """Equal to the last bit: the statistic, every replicate, the p-value and both intervals."""
    assert set(fast) == set(registered) == {"t_hat", "t_star", "p", "ci", "ci_basic", "n_nodes"}
    assert fast["t_star"].shape == registered["t_star"].shape and fast["t_star"].dtype == registered["t_star"].dtype
    for key in ("t_hat", "t_star", "p", "ci", "ci_basic"):
        assert _bits(fast[key]) == _bits(registered[key]), key
    assert fast["n_nodes"] == registered["n_nodes"]


def _random_nodes(rng, n, known_params):
    """``n`` synthetic nodes: the fitted seasonal means of the four fixture nodes, with persistence and noise drawn at
    random (the bootstrap sees only the simulated scores)."""
    import dataclasses
    base = [known_params[k] for k in sorted(known_params)]
    return {("M", f"N{i}"): dataclasses.replace(base[i % len(base)], phi_s=float(rng.uniform(0.1, 0.9)),
                                                 sd_s=float(rng.uniform(0.5, 1.5)), sd_eps=float(rng.uniform(0.5, 1.5)))
            for i in range(n)}


#: (calibration days, deployment days): the registered 92 and shorter windows, down to a single day.
_FAST_WINDOWS = [(92, 92), (20, 14), (31, 30), (45, 60), (7, 5), (60, 31), (1, 1), (3, 2)]


def _fast_config(i):
    rng = np.random.default_rng(9200 + i)
    cal_n, dep_n = _FAST_WINDOWS[i % len(_FAST_WINDOWS)]
    return {"n_nodes": 1 + i % 7, "cal_n": cal_n, "dep_n": dep_n,
            "delta": (0.0, 0.25, 0.8, 1.5)[(i // 2) % 4], "common_shift": (0.0, 0.5, -0.4, 1.0)[(i // 3) % 4],
            "n_rep": 2000 if i % 5 == 1 else 299, "alpha": (0.05 / 3, 0.05)[i % 2],
            "persistence": ("fit", "upper")[(i // 5) % 2],
            "draw_seed": int(rng.integers(10 ** 6)), "boot_seed": int(rng.integers(10 ** 9))}


def test_the_fast_bootstrap_gate_covers_what_the_ruling_asks_for():
    cfgs = [_fast_config(i) for i in range(32)]
    assert len(cfgs) >= 30 and len({c["draw_seed"] for c in cfgs}) == len({c["boot_seed"] for c in cfgs}) == len(cfgs)
    assert {c["alpha"] for c in cfgs} == {0.05 / 3, 0.05} and {c["n_rep"] for c in cfgs} == {299, 2000}
    assert {c["n_nodes"] for c in cfgs} == set(range(1, 8))                       # odd and even: the median differs
    assert len({c["delta"] for c in cfgs}) >= 4 and len({c["common_shift"] for c in cfgs}) >= 4
    assert {c["persistence"] for c in cfgs} == {"fit", "upper"} and {(c["cal_n"], c["dep_n"]) for c in cfgs} == set(_FAST_WINDOWS)
    for alpha in (0.05 / 3, 0.05):
        assert {c["n_rep"] for c in cfgs if c["alpha"] == alpha} == {299, 2000}


@pytest.mark.parametrize("i", range(32))
def test_the_fast_bootstrap_equals_the_registered_one_bit_for_bit(known_params, i):
    """The gate (Ruling 92a): on random synthetic grids of the kind hb_size draws, over seeds, node counts, deltas,
    common shifts, window lengths, B = 299 and 2000 and alpha = 0.05 / 3 and 0.05."""
    from es.bootstrap import block_contrast_bootstrap
    from es.simulate import _hb_bootstrap_fast, _hb_draw, _hb_frames, _hb_templates
    c = _fast_config(i)
    params = _random_nodes(np.random.default_rng(c["draw_seed"]), c["n_nodes"], known_params)
    nodes = sorted(params)
    cal, dep = _days(dt.date(2025, 7, 1), c["cal_n"]), _days(dt.date(2026, 7, 1), c["dep_n"])
    templates = _hb_templates(params, cal, dep)
    blocks = _hb_draw(templates, np.random.default_rng(c["draw_seed"]), c["delta"], 1.0, c["persistence"],
                      c["common_shift"])
    alt, base = _hb_frames(nodes, templates, cal, dep, np.random.default_rng(c["draw_seed"]), c["delta"], 1.0,
                           c["persistence"], c["common_shift"])
    for k, pair in zip(nodes, blocks):                      # the frames the registered function reads carry these arrays
        for frame, (cs, ds) in zip((alt, base), pair):
            mine = frame[frame["node"] == k[1]]
            assert np.array_equal(mine[mine["split"] == "calib"]["score"].to_numpy(), cs)
            assert np.array_equal(mine[mine["split"] == "deploy"]["score"].to_numpy(), ds)
    registered = block_contrast_bootstrap(alt, base, alpha=c["alpha"], n_rep=c["n_rep"], seed=c["boot_seed"])
    fast = _hb_bootstrap_fast(blocks, alpha=c["alpha"], n_rep=c["n_rep"], seed=c["boot_seed"])
    assert np.isfinite(registered["t_star"]).all()
    if min(c["cal_n"], c["dep_n"]) > 1:                      # (a window of one day resamples to itself: t_star is constant)
        assert np.ptp(registered["t_star"]) > 0
    _assert_same_bootstrap(fast, registered)


def _hand_blocks(rng, n_nodes, n_cal, n_dep, grid=None, shift=0.0, persistence=0.7):
    """Hand-made complete grids (24 hours a day, days in order): a daily AR(1) plus hourly noise; ``grid`` rounds the
    scores to multiples of it, so that exact ties abound (among the sample, and between it and the threshold)."""
    blocks = []
    for _ in range(n_nodes):
        pair = []
        for b in range(2):
            def window(n, add=0.0):
                x = np.empty(n)
                x[0] = rng.normal()
                for t in range(1, n):
                    x[t] = persistence * x[t - 1] + rng.normal(0, 0.7)
                s = np.repeat(x, 24) + rng.normal(0, 1, n * 24) + add
                return s if grid is None else np.round(s / grid) * grid
            pair.append((window(n_cal), window(n_dep, shift if b == 0 else 0.0)))
        blocks.append(tuple(pair))
    return blocks


def _fast_and_registered(blocks, n_cal, n_dep, **kw):
    from es.bootstrap import block_contrast_bootstrap
    from es.simulate import _frames_of, _hb_bootstrap_fast
    nodes = [("M", f"N{i}") for i in range(len(blocks))]
    alt, base = _frames_of(nodes, blocks, _days(dt.date(2025, 7, 1), n_cal), _days(dt.date(2026, 7, 1), n_dep))
    return _hb_bootstrap_fast(blocks, **kw), block_contrast_bootstrap(alt, base, **kw)


@pytest.mark.parametrize("grid", [0.5, 2.0])
@pytest.mark.parametrize("alpha", [0.05, 0.05 / 3])
def test_the_fast_bootstrap_matches_the_registered_one_when_scores_are_tied(grid, alpha):
    """Rounded scores tie within the sample and with the threshold, so what the quantile interpolates between and what
    '>= tau' counts both depend on how ties are read."""
    rng = np.random.default_rng(4)
    blocks = _hand_blocks(rng, 3, 13, 9, grid=grid, shift=0.4)
    fast, registered = _fast_and_registered(blocks, 13, 9, alpha=alpha, n_rep=150, seed=12)
    _assert_same_bootstrap(fast, registered)


def test_the_fast_bootstrap_matches_the_registered_one_for_a_constant_calibration_window():
    rng = np.random.default_rng(5)
    blocks = []
    for _ in range(3):
        pair = []
        for _ in range(2):
            pair.append((np.full(10 * 24, 1.0), rng.choice([0.5, 1.0, 1.5], size=6 * 24)))
        blocks.append(tuple(pair))
    for alpha in (0.05, 0.05 / 3):
        fast, registered = _fast_and_registered(blocks, 10, 6, alpha=alpha, n_rep=120, seed=3)
        _assert_same_bootstrap(fast, registered)


def test_the_fast_bootstrap_matches_the_registered_one_when_the_top_scores_sit_in_a_few_days(monkeypatch):
    """The upper tail is read from the largest rows only; a replicate that draws none of the days that hold them
    needs the whole window, and gets it."""
    import es.simulate as S
    full_calls = []
    real = S._order_pair_full
    monkeypatch.setattr(S, "_order_pair_full", lambda *a, **k: (full_calls.append(1), real(*a, **k))[1])
    rng = np.random.default_rng(6)
    blocks = _hand_blocks(rng, 3, 31, 12)
    for pair in blocks:
        for cal, _ in pair:
            cal[:7 * 24] += 25.0                   # the 168 largest rows are those of the first seven days
    fast, registered = _fast_and_registered(blocks, 31, 12, alpha=0.05, n_rep=200, seed=21)
    assert full_calls                              # the fallback ran
    _assert_same_bootstrap(fast, registered)


@pytest.mark.parametrize("n_days", [1, 2, 5, 31, 92])
@pytest.mark.parametrize("q", [0.95, 1 - 0.05 / 3, 0.5, 0.02])
def test_replicate_thresholds_are_numpys_quantile_of_the_expanded_sample(n_days, q, monkeypatch):
    """np.quantile(np.repeat(scores, count of the row's day), q) for each replicate, to the last bit (a zero of either
    sign is the same threshold: nothing compares them apart), for the counts of the stationary bootstrap, of an iid
    bootstrap, of resamples that put nearly all their weight on one day, and of resamples that put all of it on a day
    holding none of the largest scores (the case in which the upper tail cannot be read from the largest rows); the
    scores are also rounded, so that they tie."""
    import es.simulate as S
    from es.bootstrap import stationary_counts
    rng = np.random.default_rng(n_days)
    scores = np.repeat(rng.normal(0, 1, n_days), 24) + rng.normal(0, 1, n_days * 24)
    scores[:24] -= 100.0                                  # day 0 holds none of the largest scores
    mostly_last = np.ones(n_days) * 1e-3
    mostly_last[-1] = 1.0
    only_day_0 = np.zeros((40, n_days), dtype=np.int64)
    only_day_0[:, 0] = n_days
    day = np.repeat(np.arange(n_days), 24)
    full_calls = []
    real = S._order_pair_full
    monkeypatch.setattr(S, "_order_pair_full", lambda *a, **k: (full_calls.append(1), real(*a, **k))[1])
    for cal in (scores, np.round(scores * 2) / 2):
        for counts in (stationary_counts(n_days, 7, 40, 5),
                       rng.multinomial(n_days, np.ones(n_days) / n_days, size=40),
                       rng.multinomial(n_days, mostly_last / mostly_last.sum(), size=40),
                       only_day_0):
            got = S._replicate_thresholds(cal, np.ascontiguousarray(counts.T, dtype=np.int32), q)
            want = np.array([np.quantile(np.repeat(cal, counts[b][day]), q) for b in range(len(counts))])
            assert got.shape == want.shape and (got + 0.0).tobytes() == (want + 0.0).tobytes()
    if n_days >= 5 and q in (0.95, 1 - 0.05 / 3):         # the largest rows are fewer than the window: the fallback is on
        assert full_calls


def test_replicate_thresholds_interpolate_like_numpy_at_exactly_one_half():
    """At the weight 1/2 numpy interpolates from the upper element, b - (b - a) (1 - 1/2), which for some pairs is not
    the bit pattern of the lower form a + (b - a) / 2."""
    import es.simulate as S
    a, c = np.sort(np.random.default_rng(0).normal(0, 1, 2))
    assert a + (c - a) * 0.5 != c - (c - a) * (1 - 0.5)                 # the two forms differ for this pair
    cal = np.concatenate([a - 1 - np.arange(11), [a, c], c + 1 + np.arange(11)])        # one day: the median is at 11.5
    got = S._replicate_thresholds(cal, np.ones((1, 3), dtype=np.int32), 0.5)
    assert got.tobytes() == np.full(3, np.quantile(cal, 0.5)).tobytes()


def test_replicate_thresholds_need_the_whole_window_when_the_largest_rows_hold_just_the_elements_above_the_quantile():
    """If the resample holds, among the largest rows, exactly as many elements as lie above the quantile, the element at
    the quantile is not one of them (it needs one more): reading the tail from the largest rows alone would put the
    quantile between the 96th largest score and itself instead of between -100 and 100."""
    import es.simulate as S
    n_days, q = 31, 1 - 0.05 / 3
    n = n_days * 24
    lo = int(np.floor((n - 1) * q))
    above = n - 1 - lo
    assert above == 13 and min(n, max(S._FAST_TOP_FACTOR * (n - lo), S._FAST_TOP_MIN)) < n         # the design needs both
    cal = np.random.default_rng(3).normal(0, 1, n)
    cal[7 * 24:9 * 24] = -100.0
    cal[7 * 24] = 100.0                              # day 7 holds one of the largest rows, day 8 none
    counts = np.zeros((3, n_days), dtype=np.int64)
    counts[:, 7], counts[:, 8] = above, n_days - above
    got = S._replicate_thresholds(cal, np.ascontiguousarray(counts.T, dtype=np.int32), q)
    want = np.array([np.quantile(np.repeat(cal, c[np.repeat(np.arange(n_days), 24)]), q) for c in counts])
    assert (got + 0.0).tobytes() == (want + 0.0).tobytes() and want[0] > 20


@pytest.mark.parametrize("n_days", [1, 3, 31, 92])
def test_replicate_alert_counts_are_the_weighted_number_of_rows_at_or_above_tau(n_days):
    from es.bootstrap import stationary_counts
    from es.simulate import _replicate_alert_counts
    rng = np.random.default_rng(100 + n_days)
    dep = np.repeat(rng.normal(0, 1, n_days), 24) + rng.normal(0, 1, n_days * 24)
    for scores in (dep, np.round(dep * 2) / 2):
        counts = stationary_counts(n_days, 7, 60, 9)
        tau = np.concatenate([scores[:20],                                       # exactly at data values
                              rng.normal(0, 1, 36), [scores.min() - 1, scores.min(), scores.max(), scores.max() + 1]])
        got = _replicate_alert_counts(scores, tau, counts)
        want = np.array([(np.repeat(counts[b], 24) * (scores >= tau[b])).sum() for b in range(len(counts))])
        assert np.array_equal(got, want) and got[-1] == 0 and got[-4] == counts[-4].sum() * 24


def test_the_fast_bootstrap_refuses_what_it_cannot_reproduce():
    from es.bootstrap import block_contrast_bootstrap
    from es.simulate import _FAST_MAX_DAYS, _frames_of, _hb_bootstrap_fast
    rng = np.random.default_rng(7)
    ok = _hand_blocks(rng, 2, 10, 8)
    _hb_bootstrap_fast(ok, n_rep=5)
    with pytest.raises(ValueError, match="no node"):
        _hb_bootstrap_fast([], n_rep=5)
    for bad in (0.0, 1.0, -0.1, float("nan")):
        with pytest.raises(ValueError, match="alpha"):
            _hb_bootstrap_fast(ok, alpha=bad, n_rep=5)
    with pytest.raises(ValueError, match="n_rep"):
        _hb_bootstrap_fast(ok, n_rep=0)
    # NaN: the same refusal, in the same words, as the registered function
    nan_blocks = [((np.where(np.arange(240) == 7, np.nan, cal), dep), other) for (cal, dep), other in ok]
    nodes = [("M", "N0"), ("M", "N1")]
    alt, base = _frames_of(nodes, nan_blocks, _days(dt.date(2025, 7, 1), 10), _days(dt.date(2026, 7, 1), 8))
    with pytest.raises(ValueError) as registered:
        block_contrast_bootstrap(alt, base, n_rep=5)
    with pytest.raises(ValueError) as fast:
        _hb_bootstrap_fast(nan_blocks, n_rep=5)
    assert str(fast.value) == str(registered.value) and "NaN" in str(fast.value)
    # grids that are not complete and equal across nodes
    with pytest.raises(ValueError, match="same number of days"):
        _hb_bootstrap_fast([ok[0], _hand_blocks(rng, 1, 11, 8)[0]], n_rep=5)
    with pytest.raises(ValueError, match="multiple of 24"):
        _hb_bootstrap_fast([((np.zeros(239), np.zeros(240)),) * 2], n_rep=5)
    long_cal = [((np.zeros(24 * (_FAST_MAX_DAYS + 1)), np.zeros(24)),) * 2]
    with pytest.raises(ValueError, match="days"):
        _hb_bootstrap_fast(long_cal, n_rep=5)
    tiny = [((np.zeros(24), np.zeros(24)),) * 2]
    with pytest.raises(ValueError, match="alpha"):                      # 1 - alpha rounds to 1: the maximum, not a quantile
        _hb_bootstrap_fast(tiny, alpha=1e-17, n_rep=5)


def _same_result(a, b):
    """hb_size's two result dicts are equal key by key, the NaN of a drift row counted as equal to itself."""
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], float) and math.isnan(a[k]):
            assert isinstance(b[k], float) and math.isnan(b[k]), k
        else:
            assert a[k] == b[k], k


@pytest.mark.parametrize("kw", [{}, {"delta": 0.5}, {"common_shift": 0.5, "alpha": 0.05 / 3},
                                {"persistence": "upper", "delta": 0.3, "common_shift": -0.25}])
def test_hb_size_with_the_fast_path_equals_hb_size_with_the_registered_function(known_params, kw):
    from es.simulate import hb_size
    three = {k: known_params[k] for k in sorted(known_params)[:3]}
    cal, dep = _days(dt.date(2025, 7, 1), 31), _days(dt.date(2026, 7, 1), 20)
    fast = hb_size(three, cal, dep, 4, 40, seed=6, **kw)
    registered = hb_size(three, cal, dep, 4, 40, seed=6, fast=False, **kw)
    _same_result(fast, registered)
    assert 0.0 <= fast["reject_0.05"] <= 1.0 and np.isfinite(fast["mean_t_hat"])


def test_hb_size_uses_the_fast_path_by_default_and_the_registered_function_when_asked_or_when_it_must(known_params, monkeypatch):
    import inspect
    import es.simulate as S
    assert inspect.signature(S.hb_size).parameters["fast"].default is True
    calls = []
    real = S.block_contrast_bootstrap
    monkeypatch.setattr(S, "block_contrast_bootstrap", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    two = {k: known_params[k] for k in sorted(known_params)[:2]}
    cal, dep = _days(dt.date(2025, 7, 1), 12), _days(dt.date(2026, 7, 1), 9)
    ref = S.hb_size(two, cal, dep, 3, 20, seed=1)
    assert calls == []                                                          # the default never calls it
    _same_result(S.hb_size(two, cal, dep, 3, 20, seed=1, fast=False), ref)
    assert len(calls) == 3                                                      # fast=False: once per simulation
    # grids the fast path is not built for go to the registered function, with the same answer as fast=False
    for label, c, d in (("days not increasing", cal[::-1], dep), ("a repeated day", cal[:1] + cal, dep),
                        ("a day out of place", cal + cal[:1], dep)):
        calls.clear()
        got = S.hb_size(two, c, d, 3, 20, seed=1)
        assert len(calls) == 3, label
        calls.clear()
        _same_result(got, S.hb_size(two, c, d, 3, 20, seed=1, fast=False))
    calls.clear()
    S.hb_size(two, cal, dep, 3, 20, seed=1, alpha=1e-17)                        # 1 - alpha is 1: no quantile to speak of
    assert len(calls) == 3


def test_the_array_rates_of_the_draws_equal_the_pandas_rates_of_the_frames(known_params):
    from es.simulate import _block_rates, _hb_draw, _hb_frames, _hb_templates
    nodes = sorted(known_params)
    cal, dep = _days(dt.date(2025, 7, 1), 40), _days(dt.date(2026, 7, 1), 30)
    templates = _hb_templates(known_params, cal, dep)
    for seed, delta, shift in ((1, 0.0, 0.0), (2, 0.6, 0.5)):
        blocks = _hb_draw(templates, np.random.default_rng(seed), delta, 1.0, "fit", shift)
        alt, base = _hb_frames(nodes, templates, cal, dep, np.random.default_rng(seed), delta, 1.0, "fit", shift)
        for alpha in (0.05, 0.05 / 3):
            rate_alt, rate_base = _block_rates(blocks, alpha)
            assert rate_alt.shape == rate_base.shape == (len(nodes),)
            assert rate_alt == pytest.approx(_frame_rates(alt, alpha).to_numpy(), rel=1e-12, abs=1e-12)
            assert rate_base == pytest.approx(_frame_rates(base, alpha).to_numpy(), rel=1e-12, abs=1e-12)
