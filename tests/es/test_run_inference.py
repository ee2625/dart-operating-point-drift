import datetime as dt
import re
import zlib
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest


def test_bh_and_holm():
    from es.run_inference import bh, holm
    p = [0.001, 0.02, 0.04, 0.3]
    assert bh(p, 0.05).tolist() == [True, True, False, False]
    assert holm(p, 0.05).tolist() == [True, False, False, False]


# --------------------------------------------------------------------------------------------------
# Above: the brief's test, verbatim. Below: the multiplicity helpers' input contract, ties and order
# (controller decision 1). The sealed entry point (plan 2A) imports `holm` and zips its booleans with a
# list of hypothesis names, so the order of the output is part of the contract.
# --------------------------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _one_blas_thread():
    """statsmodels' MixedLM does many small solves. With a BLAS thread per core and other jobs holding the cores they
    spend their time waiting: a 216-row fit took 14 s instead of 0.3 s on 23 Oct, with heavy simulations running."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(params=["bh", "holm"])
def procedure(request):
    import es.run_inference as ri
    return getattr(ri, request.param)


def test_empty_input_returns_an_empty_boolean_array(procedure):
    for empty in ([], (), np.array([]), pd.Series([], dtype=float)):
        out = procedure(empty)
        assert isinstance(out, np.ndarray) and out.dtype == bool and out.shape == (0,)


@pytest.mark.parametrize("bad", [[0.01, np.nan], [np.nan], [None, 0.2], [0.3, np.nan, 0.001]])
def test_a_nan_p_value_is_refused(procedure, bad):
    with pytest.raises(ValueError, match="NaN"):
        procedure(bad)


@pytest.mark.parametrize("bad", [[-0.01, 0.5], [0.5, 1.01], [np.inf], [-np.inf, 0.2], [2.0]])
def test_a_p_value_outside_zero_one_is_refused(procedure, bad):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        procedure(bad)


@pytest.mark.parametrize("bad", [0.03, np.float64(0.03), [[0.01, 0.02], [0.03, 0.04]], np.zeros((2, 2)),
                                 np.array(0.5), [[0.01, 0.02], [0.03]]])
def test_input_that_is_not_one_dimensional_is_refused(procedure, bad):
    with pytest.raises(ValueError):
        procedure(bad)


def test_zero_and_one_are_valid_p_values(procedure):
    assert procedure([0.0, 1.0]).tolist() == [True, False]
    assert procedure([1.0]).tolist() == [False]
    assert procedure([0.0]).tolist() == [True]


def test_containers_dtypes_and_inputs_are_handled_alike(procedure):
    p = [0.001, 0.02, 0.04, 0.3]
    want = procedure(p).tolist()
    for container in (tuple(p), np.array(p), pd.Series(p), pd.Series(p, index=[9, 3, 5, 1]), np.array(p, dtype=np.float32)):
        assert procedure(container).tolist() == want
    assert procedure([0, 1]).tolist() == [True, False]                 # integers are p-values too
    arr = np.array(p)
    procedure(arr)
    assert arr.tolist() == p                                           # the input is not sorted in place


@pytest.mark.parametrize("level", [0, 0.0, -0.05, 1.5, 5, np.nan])
def test_the_level_must_lie_in_zero_one(level):
    from es.run_inference import bh, holm
    with pytest.raises(ValueError, match="level"):
        bh([0.01, 0.2], q=level)
    with pytest.raises(ValueError, match="level"):
        holm([0.01, 0.2], alpha=level)


def test_level_one_is_allowed_and_rejects_everything():
    from es.run_inference import bh, holm
    assert bh([0.2, 0.9], q=1.0).tolist() == [True, True]
    assert holm([0.2, 0.9], alpha=1.0).tolist() == [True, True]


def test_holm_returns_booleans_in_input_order():
    from es.run_inference import holm
    # sorted these are 0.001, 0.02, 0.04, 0.3, and only the smallest is rejected (the brief's example)
    assert holm([0.3, 0.001, 0.04, 0.02]).tolist() == [False, True, False, False]
    assert holm([0.04, 0.3, 0.02, 0.001]).tolist() == [False, False, False, True]
    # the sealed entry point's use: three hypotheses in the order H-E, H-B1, H-B2
    pvals = {"H-E": 0.2, "H-B1": 0.004, "H-B2": 0.011}
    out = dict(zip(pvals, map(bool, holm(list(pvals.values()), 0.05))))
    assert out == {"H-E": False, "H-B1": True, "H-B2": True}


def test_bh_returns_booleans_in_input_order():
    from es.run_inference import bh
    assert bh([0.3, 0.001, 0.04, 0.02]).tolist() == [False, True, False, True]
    assert bh([0.02, 0.3, 0.001, 0.04]).tolist() == [True, False, True, False]


def test_holm_is_a_step_down_procedure_not_bonferroni():
    from es.run_inference import holm
    # Bonferroni's threshold is 0.05 / 3 = 0.0167, which 0.02 misses; Holm's second step is 0.05 / 2 = 0.025
    assert holm([0.012, 0.02, 0.9]).tolist() == [True, True, False]
    # and it stops at the first failure: 0.02 > 0.05 / 3 ends the procedure although 0.02 <= 0.05 / 1
    assert holm([0.001, 0.02, 0.02, 0.02]).tolist() == [True, False, False, False]


def test_bh_is_a_step_up_procedure():
    from es.run_inference import bh
    # 0.04 * 3 / 1 and 0.04 * 3 / 2 are above 0.05, but 0.04 * 3 / 3 is not: the largest passing rank rejects all below it
    assert bh([0.04, 0.04, 0.04]).tolist() == [True, True, True]
    assert bh([0.01, 0.045, 0.049]).tolist() == [True, True, True]
    assert bh([0.01, 0.03, 0.2]).tolist() == [True, True, False]


def test_p_values_exactly_at_the_threshold_are_rejected():
    from es.run_inference import bh, holm
    # 0.025 = 0.05 / 2 and 0.05 are exact in binary arithmetic when halved or doubled, so this is not a rounding test
    assert holm([0.025, 0.05], 0.05).tolist() == [True, True]
    assert bh([0.025, 0.05], 0.05).tolist() == [True, True]
    assert holm([0.0625, 0.125], 0.125).tolist() == [True, True]          # powers of two: 0.125 / 2 and 0.125 / 1 exactly
    assert holm([0.0625, 0.1251], 0.125).tolist() == [True, False]
    assert bh([0.0625, 0.125], 0.125).tolist() == [True, True]
    assert bh([0.0625, 0.1251], 0.125).tolist() == [True, False]


def _exact_bh(ps, q):
    """BH in exact rational arithmetic (Fractions): reject the k smallest, k the largest rank r with p_(r) * m <= q * r."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    k = max((r for r, i in enumerate(order, 1) if ps[i] * m <= q * r), default=0)
    out = [False] * m
    for r, i in enumerate(order, 1):
        out[i] = r <= k
    return out


def _exact_holm(ps, alpha):
    """Holm in exact rational arithmetic: reject in order while p_(r) * (m - r + 1) <= alpha, and stop at the first miss."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    out = [False] * m
    for rank, i in enumerate(order):
        if ps[i] * (m - rank) > alpha:
            break
        out[i] = True
    return out


def test_decimal_p_values_on_a_threshold_are_rejected_whatever_the_rounding():
    from es.run_inference import bh, holm
    # p * m / i is 0.05 * 3 / 3 = 0.05000000000000001 in binary floating point, but the third test sits exactly on its BH
    # threshold 3 * 0.05 / 3 and exact arithmetic rejects it (the reviewer's case; statsmodels gets this one right)
    assert bh([0.01, 0.02, 0.05], 0.05).tolist() == [True, True, True]
    # 0.15 / 3 is 0.049999999999999996 < 0.05 in floating point, but 0.05 is exactly Holm's first threshold, and the next
    # two (0.068 <= 0.075, 0.1 <= 0.15) follow. statsmodels' own comparison fails here too: it rejects nothing.
    assert holm([0.05, 0.068, 0.1], 0.15).tolist() == [True, True, True]
    ps = [Fraction("0.05"), Fraction("0.068"), Fraction("0.1")]
    assert _exact_holm(ps, Fraction("0.15")) == [True, True, True]
    assert _exact_bh([Fraction("0.01"), Fraction("0.02"), Fraction("0.05")], Fraction("0.05")) == [True, True, True]
    # the slack is a relative 1e-12 and no more: a p-value a hair above a threshold is still not rejected
    assert bh([0.01, 0.02, 0.0501], 0.05).tolist() == [True, True, False]
    assert bh([0.01, 0.02, 0.05 * (1 + 1e-9)], 0.05).tolist() == [True, True, False]
    assert holm([0.0501, 0.068, 0.1], 0.15).tolist() == [False, False, False]
    assert holm([0.05 * (1 + 1e-9), 0.068, 0.1], 0.15).tolist() == [False, False, False]


def _decimal_vectors(rng, n):
    """(level, p-values) as Fractions: decimals, many of them exactly on a BH threshold q * i / m or a Holm threshold
    q / (m - i + 1) (kept when they are decimals with at most four places), the rest on a grid of 0.001."""
    levels = [Fraction(5, 100), Fraction(1, 10), Fraction(15, 100), Fraction(1, 100), Fraction(25, 1000),
              Fraction(2, 10), Fraction(3, 10)]
    for _ in range(n):
        q = levels[int(rng.integers(len(levels)))]
        m = int(rng.integers(1, 13))
        ps = []
        for _ in range(m):
            if rng.random() < 0.6:
                i = int(rng.integers(1, m + 1))
                threshold = q * i / m if rng.random() < 0.5 else q / i
                if (threshold * 10 ** 4).denominator == 1 and threshold <= 1:
                    ps.append(threshold)
                    continue
            ps.append(Fraction(int(rng.integers(0, 501)), 1000))
        yield q, ps


def test_bh_and_holm_match_exact_rational_arithmetic_at_decimal_thresholds():
    """The reference here is fractions.Fraction, not statsmodels: on exact decimal thresholds statsmodels' own comparison
    is off by one ulp in the same places as the plain floating-point rule (on 20,000 such vectors, 23 Oct: statsmodels
    348 wrong BH decisions and 712 wrong Holm decisions, the plain rule 464 and 712, the 1e-12 slack 0 and 0). The
    random continuous vectors above stay compared with statsmodels."""
    from es.run_inference import bh, holm

    def plain_bh(p, q):                                  # the rule without the slack: p * m / i <= q
        order = np.argsort(p, kind="stable")
        below = np.nonzero(p[order] * len(p) / (np.arange(len(p)) + 1) <= q)[0]
        out = np.zeros(len(p), dtype=bool)
        if below.size:
            out[order[: below.max() + 1]] = True
        return out.tolist()

    def plain_holm(p, alpha):                            # the rule without the slack: p <= alpha / (m - rank)
        out = np.zeros(len(p), dtype=bool)
        for rank, idx in enumerate(np.argsort(p, kind="stable")):
            if p[idx] > alpha / (len(p) - rank):
                break
            out[idx] = True
        return out.tolist()

    rng = np.random.default_rng(7)
    on_a_threshold = plain_bh_wrong = plain_holm_wrong = 0
    for q, ps in _decimal_vectors(rng, 3000):
        p, level = np.array([float(x) for x in ps]), float(q)         # the doubles nearest to the decimals 0.05, 0.068, ...
        want_bh, want_holm = _exact_bh(ps, q), _exact_holm(ps, q)
        assert bh(p, level).tolist() == want_bh, (q, ps)
        assert holm(p, level).tolist() == want_holm, (q, ps)
        order = sorted(range(len(ps)), key=lambda i: ps[i])
        on_a_threshold += (any(ps[i] * len(ps) == q * r for r, i in enumerate(order, 1))
                           or any(ps[i] * (len(ps) - r) == q for r, i in enumerate(order)))
        plain_bh_wrong += plain_bh(p, level) != want_bh
        plain_holm_wrong += plain_holm(p, level) != want_holm
    assert on_a_threshold > 1000                         # many vectors sit exactly on a threshold
    assert plain_bh_wrong > 0 and plain_holm_wrong > 0   # and the rule without the slack gets some of them wrong


def test_tied_p_values_are_decided_together_whatever_their_order():
    from es.run_inference import bh, holm
    assert holm([0.012, 0.012, 0.9]).tolist() == [True, True, False]
    assert holm([0.02, 0.02, 0.02]).tolist() == [False, False, False]
    assert holm([0.9, 0.012, 0.012]).tolist() == [False, True, True]
    assert bh([0.02, 0.02, 0.5, 0.5]).tolist() == [True, True, False, False]
    assert bh([0.5, 0.02, 0.5, 0.02]).tolist() == [False, True, False, True]
    # 0.02 is tied at ranks 1 and 2 of 4: rank 1 alone (0.02 * 4 / 1 = 0.08) would fail, rank 2 (0.04) passes for both
    assert bh([0.02, 0.5, 0.02, 0.5]).tolist() == [True, False, True, False]


def _random_p_vectors(rng, n=300):
    for _ in range(n):
        m = int(rng.integers(1, 13))
        p = rng.uniform(size=m) ** 3                                    # skewed towards 0, so that there are rejections
        if rng.random() < 0.5 and m > 1:                                 # a tie group of 2-3 equal p-values
            k = int(rng.integers(2, min(m, 3) + 1))
            p[:k] = p[0]
        yield p


def test_outputs_follow_the_inputs_under_any_permutation():
    from es.run_inference import bh, holm
    rng = np.random.default_rng(0)
    for p in _random_p_vectors(rng):
        perm = rng.permutation(len(p))
        for f in (bh, holm):
            assert f(p[perm]).tolist() == f(p)[perm].tolist()


def test_bh_and_holm_agree_with_statsmodels_multipletests():
    from statsmodels.stats.multitest import multipletests
    from es.run_inference import bh, holm
    rng = np.random.default_rng(1)
    n_rejecting = 0
    for p in _random_p_vectors(rng, n=120):
        want_holm = multipletests(p, alpha=0.05, method="holm")[0]
        want_bh = multipletests(p, alpha=0.05, method="fdr_bh")[0]
        assert holm(p, 0.05).tolist() == want_holm.tolist(), p
        assert bh(p, 0.05).tolist() == want_bh.tolist(), p
        n_rejecting += int(want_bh.any())
    assert n_rejecting > 40                                             # the comparison is not vacuous


def test_bh_rejects_a_superset_of_holm():
    from es.run_inference import bh, holm
    rng = np.random.default_rng(2)
    strictly_more = 0
    for p in _random_p_vectors(rng):
        h, b = holm(p), bh(p)
        assert not (h & ~b).any()
        strictly_more += int((b & ~h).any())
    assert strictly_more > 0


# --------------------------------------------------------------------------------------------------
# The descriptive mixed model (spec v3 §5.1, review M6). All data are synthetic.
# --------------------------------------------------------------------------------------------------

#: planted mean log-error of each block, so that a fit can be checked against what generated the data
_BLOCK_EFFECT = {"G6": 0.0, "LEV6": 0.1, "DART3": 0.2}


def _cells(seed=0, markets=("NYISO", "ISONE", "MISO", "CAISO"), nodes_per=3, analyses=("AQ3", "AQ4")):
    """P1 / xgb activity cells in the shape of outputs/es/analogs/activity_all.csv (one row per node, analysis and
    block): log_error = block effect + operator + operator x analysis + node effects + noise. The random effects are
    large enough for the REML optimum to be interior, so the fit raises no boundary warning. 72 rows, about 0.05 s."""
    rng = np.random.default_rng(seed)
    operator = {m: rng.normal(0, 0.4) for m in markets}
    operator_analysis = {(m, a): rng.normal(0, 0.3) for m in markets for a in analyses}
    rows = []
    for m in markets:
        for k in range(nodes_per):
            node_effect = rng.normal(0, 0.3)
            for a in analyses:
                for block, effect in _BLOCK_EFFECT.items():
                    rows.append({"market": m, "node": f"{m}_{k}", "analysis": a, "block": block,
                                 "log_error": 0.3 + effect + operator[m] + operator_analysis[(m, a)] + node_effect
                                 + rng.normal(0, 0.1)})
    return pd.DataFrame(rows)


def test_fit_mixed_fits_a_small_frame_without_a_warning():
    import warnings
    from es.run_inference import fit_mixed
    cells = _cells()
    with warnings.catch_warnings():
        warnings.simplefilter("error")                       # silence nothing: a convergence or boundary warning fails
        res = fit_mixed(cells)
    assert res.converged and res.nobs == len(cells) == 72
    text = str(res.summary())
    assert "Mixed Linear Model Regression Results" in text and "REML" in text
    # (1|operator), (1|operator:node), (1|operator:analysis) of spec v3 §5.1, the first labelled by its grouping column
    assert "market Var" in text and "node Var" in text and "operator_analysis Var" in text and res.model.k_re == 1
    assert "Group Var" not in text
    assert list(res.model.group_labels) == ["CAISO", "ISONE", "MISO", "NYISO"]      # grouped by operator (market)
    vc = res.model.exog_vc                                   # within each operator: a column per node, and one per analysis
    assert list(vc.names) == ["node", "operator_analysis"]
    assert vc.colnames[0] == [[f"C(node)[{m}_{k}]" for k in range(3)] for m in ("CAISO", "ISONE", "MISO", "NYISO")]
    assert vc.colnames[1] == [["C(analysis)[AQ3]", "C(analysis)[AQ4]"]] * 4
    # block x analysis fixed effects, DART3 and AQ3 being the reference levels
    assert list(res.fe_params.index) == ["Intercept", "C(block)[T.G6]", "C(block)[T.LEV6]", "C(analysis)[T.AQ4]",
                                         "C(block)[T.G6]:C(analysis)[T.AQ4]", "C(block)[T.LEV6]:C(analysis)[T.AQ4]"]
    assert abs(res.fe_params["C(block)[T.G6]"] - (0.0 - 0.2)) < 0.1
    assert abs(res.fe_params["C(block)[T.LEV6]"] - (0.1 - 0.2)) < 0.1


def test_fit_mixed_does_not_depend_on_the_row_order_or_index_of_the_frame():
    from es.run_inference import fit_mixed
    cells = _cells()
    shuffled = cells.sample(frac=1.0, random_state=3)
    shuffled.index = np.arange(len(shuffled)) * 3 + 7        # a filtered frame, like main()'s `acts[...]`
    a, b = fit_mixed(cells), fit_mixed(shuffled)
    assert np.allclose(a.fe_params.sort_index(), b.fe_params.sort_index(), atol=1e-6)
    assert np.allclose(a.vcomp, b.vcomp, atol=1e-6)
    assert np.allclose(a.cov_re, b.cov_re, atol=1e-6)


def test_fit_mixed_has_the_operator_random_intercept_of_the_spec():
    """Spec v3 §5.1 lists (1|operator). With a vc_formula and no re_formula statsmodels adds no group random intercept
    (model.k_re == 0), so fit_mixed passes re_formula (controller Ruling 80) and groups="market", the column name, so that
    the intercept prints as "market Var" in the Appendix C table and not as "Group Var" (Ruling 89, M6)."""
    from es.run_inference import fit_mixed
    res = fit_mixed(_cells())
    assert res.model.k_re == 1 and list(res.cov_re.index) == ["market"] and float(res.cov_re.iloc[0, 0]) >= 0.0
    assert "market Var" in str(res.summary())


def test_fit_mixed_is_the_model_that_the_group_variable_call_fitted():
    """groups="market" with re_formula="~1" changes the label of the operator intercept, nothing else: the fit equals the
    one with groups=cells["market"] and re_formula="1" (llf, fixed effects, standard errors, variances)."""
    import statsmodels.formula.api as smf
    from es.run_inference import fit_mixed
    for cells in (_cells(), _cells(seed=1), _cells(seed=2).sample(frac=1.0, random_state=5)):
        old = smf.mixedlm("log_error ~ C(block) * C(analysis)", cells, groups=cells["market"], re_formula="1",
                          vc_formula={"node": "0 + C(node)", "operator_analysis": "0 + C(analysis)"}).fit(reml=True)
        new = fit_mixed(cells)
        assert list(old.cov_re.index) == ["Group"] and list(new.cov_re.index) == ["market"]
        assert abs(new.llf - old.llf) < 1e-9
        assert np.allclose(new.fe_params, old.fe_params, rtol=1e-9, atol=1e-12)
        assert np.allclose(new.bse_fe, old.bse_fe, rtol=1e-9, atol=1e-12)
        assert np.allclose(new.cov_re, old.cov_re, rtol=1e-9, atol=1e-12)
        assert np.allclose(new.vcomp, old.vcomp, rtol=1e-9, atol=1e-12) and abs(new.scale - old.scale) < 1e-12


# --------------------------------------------------------------------------------------------------
# main(): the mixed model, the leave-one-operator-out table and the BH rehearsal on AQ3. Nothing here reads or
# writes the real outputs/ or the score cache: OUT_DIR is a temporary folder, the panel and cached_scores are stubbed,
# and `sandbox` fails any write under the real outputs/ or cache/ (so a main() that ignored OUT_DIR would fail the
# test instead of writing there).
# --------------------------------------------------------------------------------------------------

_SIX_BLOCKS = ["G6", "LEV6", "DART3", "ROLL_DISP9", "DART18", "G1_24"]
_MARKETS = ("NYISO", "ISONE", "MISO", "CAISO")
_LOG2 = 0.30103                                    # log10(2): the failure band is [0.5, 2]


def _activity_all(seed=4):
    """A small outputs/es/analogs/activity_all.csv (Task 11): activity()'s columns plus analysis, block, estimator and
    policy. The 216 cells the mixed model is fitted on are the P1 / xgb / six-block rows (`decoy` False, a column
    main() does not read); the decoys are other policies, the logit estimator and a seventh block, all with huge errors.
    Seed 4 is a draw whose operator variance is clearly above zero: with only four operators many draws put it on the
    boundary (seed 0 does), and statsmodels then warns, which would hide in the output of every main() test."""
    rng = np.random.default_rng(seed)
    analyses = ("AQ3", "AQ4", "AQ3F")
    operator = {m: rng.normal(0, 0.4) for m in _MARKETS}
    operator_analysis = {(m, a): rng.normal(0, 0.3) for m in _MARKETS for a in analyses}
    rows = []
    for m in _MARKETS:
        for k in range(3):
            node_effect = rng.normal(0, 0.3)
            for a in analyses:
                for b in _SIX_BLOCKS + ["MID3"]:
                    for estimator in ("xgb", "logit"):
                        for policy in ("P1", "P4_56"):
                            decoy = estimator != "xgb" or policy != "P1" or b == "MID3"
                            err = 5.0 + rng.random() if decoy else abs(
                                0.45 + 0.05 * _SIX_BLOCKS.index(b) + operator[m] + operator_analysis[(m, a)]
                                + node_effect + rng.normal(0, 0.1))
                            rows.append({"market": m, "node": f"{m}_{k}", "n": 2200, "log_error": err,
                                         "fail_0.5_2": err > _LOG2, "analysis": a, "block": b,
                                         "estimator": estimator, "policy": policy, "decoy": decoy})
    return pd.DataFrame(rows)


#: (position, fees) of the AQ3 P1 rows in the fake economics summary, and their rotation p-values. The last is NaN.
_ECON_P = [0.001, 0.004, 0.012, 0.02, 0.03, 0.035, 0.04, 0.045, 0.06, 0.2, 0.4, 0.6, 0.8, np.nan]


def _economics_summary():
    """Rows in the shape of outputs/es/economics/summary.csv (Task 12) for AQ3, with decoys that economic_family must
    leave out: other policies, comparators, and AQ1 rows whose tiny p-values would loosen BH's thresholds if they
    were in the AQ3 family."""
    cells = [(pos, fee) for pos in ("premium", "inc", "dec") for fee in ("calibrated", "0.0", "0.25", "stress")]
    cells += [("premium[high]", "calibrated"), ("premium[low]", "calibrated")]
    assert len(cells) == len(_ECON_P)

    def row(analysis, policy, comparator, position, fees, selc, p):
        return {"analysis": analysis, "policy": policy, "comparator": comparator, "position": position,
                "fees": fees, "SELC": selc, "NET": 1.0, "rotation_p": p}

    rows = [row("AQ3", "P1", "none", pos, fee, 0.01 * i, p) for i, ((pos, fee), p) in enumerate(zip(cells, _ECON_P))]
    rows += [row("AQ3", "P4", "none", "premium", "calibrated", 0.5, 0.0001),
             row("AQ3", "comparator", "VOL@count", "premium", "calibrated", 0.5, 0.0001),
             row("AQ3", "P1", "VOL@count", "premium", "0.25", 0.5, 0.0001)]
    rows += [row("AQ1", "P1", "none", "premium", "calibrated", 0.5, 1e-6) for _ in range(12)]
    return pd.DataFrame(rows)


_MONITOR_P = [0.001, 0.004, 0.012, 0.02, 0.03, 0.035, 0.2, 0.4, 0.6, 0.8, 0.9, np.nan]


def _monitor_eval():
    """Rows in the shape of outputs/es/monitors/monitor_eval.csv (Task 14): AQ3 over three blocks and four monitors,
    plus AQ1 rows with tiny p-values that must stay out of the AQ3 family."""
    keys = [(b, m) for b in ("LEV6", "DART3", "ROLL_DISP9") for m in ("W1_norm", "PSI", "domain_auc", "early_rate")]
    rows = [{"analysis": "AQ3", "block": b, "monitor": m, "spearman": 0.1 * i, "spearman_p_one_sided": p,
             "auroc_fail": 0.6, "n_used": 33} for i, ((b, m), p) in enumerate(zip(keys, _MONITOR_P))]
    rows += [{"analysis": "AQ1", "block": "LEV6", "monitor": "PSI", "spearman": 0.9, "spearman_p_one_sided": 1e-6,
              "auroc_fail": 0.9, "n_used": 33}] * 12
    return pd.DataFrame(rows)


def _small_score(kind, block, estimator, seed):
    """A score frame of the shape es.bootstrap reads, 3 nodes with 16 calibration and 16 deployment days of 6 hours.
    LEV6 drifts at deployment. Deterministic in its arguments."""
    rng = np.random.default_rng(zlib.crc32(repr((kind, block, estimator, seed)).encode()))
    rows = [{"market": "M", "node": node, "split": split, "settle_date": start + dt.timedelta(days=d),
             "score": rng.normal(2.0 if (split == "deploy" and block == "LEV6") else 0.0, 1.0)}
            for node in ("A", "B", "C")
            for split, start in (("calib", dt.date(2024, 7, 1)), ("deploy", dt.date(2025, 7, 1)))
            for d in range(16) for _ in range(6)]
    return pd.DataFrame(rows)


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """es.run_inference pointed at tmp_path with every input stubbed. Returns (module, calls, activity_all, panel),
    where `calls` collects what main() asks cached_scores for as (panel, analysis id, block, estimator, seed) and
    `panel` is the sentinel that the stubbed build_panel returns."""
    import pathlib
    import es.config as config
    import es.run_inference as ri
    import es.secondary as sec

    guarded = [config.DEV_ROOT / "outputs", config.DEV_ROOT / "cache"]

    def check(target):
        try:
            resolved = pathlib.Path(target).resolve()
        except TypeError:                                  # to_csv() to a buffer or to nothing
            return
        assert not any(g.resolve() in (resolved, *resolved.parents) for g in guarded), (
            f"a test wrote to {resolved}, under the real outputs/ or cache/")

    original = {"write_text": pathlib.Path.write_text, "mkdir": pathlib.Path.mkdir, "to_csv": pd.DataFrame.to_csv}

    def write_text(self, *a, **k):
        check(self)
        return original["write_text"](self, *a, **k)

    def mkdir(self, *a, **k):
        check(self)
        return original["mkdir"](self, *a, **k)

    def to_csv(self, path_or_buf=None, *a, **k):
        check(path_or_buf)
        return original["to_csv"](self, path_or_buf, *a, **k)

    monkeypatch.setattr(pathlib.Path, "write_text", write_text)
    monkeypatch.setattr(pathlib.Path, "mkdir", mkdir)
    monkeypatch.setattr(pd.DataFrame, "to_csv", to_csv)
    monkeypatch.setattr(ri, "OUT_DIR", tmp_path)
    monkeypatch.setattr(ri, "ensure_dirs", lambda: None)
    monkeypatch.chdir(tmp_path)

    calls = []
    panel = object()
    monkeypatch.setattr(ri, "load_dev_raw", lambda: "raw")
    monkeypatch.setattr(ri, "build_panel", lambda raw: panel if raw == "raw" else None)

    def cached_scores(p, analysis, block, estimator, seed):
        calls.append((p, analysis.id, block, estimator, seed))
        return _small_score(analysis.id, block, estimator, seed)

    monkeypatch.setattr(ri, "cached_scores", cached_scores)
    real_bootstrap = sec.block_contrast_bootstrap                 # the same bootstrap, with 30 replicates instead of 2000
    monkeypatch.setattr(sec, "block_contrast_bootstrap", lambda a, b, **kw: real_bootstrap(a, b, **{**kw, "n_rep": 30}))

    acts = _activity_all()
    for name, frame in (("analogs/activity_all.csv", acts), ("economics/summary.csv", _economics_summary()),
                        ("monitors/monitor_eval.csv", _monitor_eval())):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(tmp_path / name, index=False)
    return ri, calls, acts, panel


def test_main_writes_the_mixed_model_looo_and_bh_into_out_dir_only(sandbox, tmp_path, capsys):
    ri, calls, acts, panel = sandbox
    before = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")}
    ri.main()
    after = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")}
    assert after - before == {"inference", "inference/mixed_model.txt", "inference/looo.csv", "inference/bh.csv"}
    out = capsys.readouterr().out
    assert "missing" not in out
    # the tests counted per family are the ones that entered BH (13 of the 14 economic rows and 11 of the 12 monitor rows)
    assert "audit robustness 18 tests" in out and "economic robustness 13 tests" in out and "drift monitors 11 tests" in out
    assert not list(tmp_path.glob("**/*.parquet"))                                  # no score cache


def test_main_fits_the_mixed_model_on_the_p1_xgb_cells_of_the_six_blocks(sandbox, tmp_path):
    ri, calls, acts, panel = sandbox
    ri.main()
    text = (tmp_path / "inference" / "mixed_model.txt").read_text()
    assert "Mixed Linear Model Regression Results" in text and "Converged:" in text
    assert re.search(r"No\. Observations:\s+216\b", text)                          # the 216 non-decoy rows, no others
    assert int((~acts["decoy"]).sum()) == 216
    assert "C(block)[T.G6]:C(analysis)[T.AQ4]" in text
    assert "market Var" in text and "node Var" in text and "operator_analysis Var" in text      # the spec's three components


def test_main_looo_drops_one_operator_at_a_time(sandbox, tmp_path):
    ri, calls, acts, panel = sandbox
    ri.main()
    looo = pd.read_csv(tmp_path / "inference" / "looo.csv", float_precision="round_trip")
    assert set(looo.columns) == {"dropped", "analysis", "block", "failures", "median_log_error"}
    cells = acts[~acts["decoy"]]
    want = []
    for dropped in sorted(_MARKETS):
        for analysis in sorted(cells["analysis"].unique()):
            for block in sorted(_SIX_BLOCKS):
                kept = cells[(cells["market"] != dropped) & (cells["analysis"] == analysis) & (cells["block"] == block)]
                assert len(kept) == 9                                               # 3 other operators x 3 nodes
                want.append((dropped, analysis, block, int(kept["fail_0.5_2"].sum()), float(np.median(kept["log_error"]))))
    got = looo.sort_values(["dropped", "analysis", "block"])[["dropped", "analysis", "block", "failures",
                                                              "median_log_error"]]
    assert len(got) == len(want) == 4 * 3 * 6
    assert [r[:4] for r in got.itertuples(index=False)] == [w[:4] for w in want]
    assert got["median_log_error"].tolist() == [w[4] for w in want]                # bit for bit: the CSVs round-trip
    assert looo["failures"].between(0, 9).all() and looo["failures"].nunique() > 1          # not all the same count


def test_main_audit_asks_for_aq3_scores_and_aq3f_for_the_frozen_arm(sandbox):
    ri, calls, acts, panel = sandbox
    ri.main()
    # 36 requests (18 tests x alt and DART3), 21 distinct frames, each asked for once: DART3's 7 are not asked for 18 times
    assert len(calls) == 21 and all(c[0] is panel for c in calls)                  # build_panel(load_dev_raw()), once
    assert len({c[1:] for c in calls}) == 21
    frozen = {(a, b, e, s) for _, a, b, e, s in calls if a == "AQ3F"}
    assert frozen == {("AQ3F", b, "xgb", 42) for b in ("LEV6", "ROLL_DISP9", "DART3")}
    assert sum(1 for c in calls if c[1] == "AQ3F") == 3
    assert {c[1] for c in calls} == {"AQ3", "AQ3F"}
    roll = {(b, e, s) for _, a, b, e, s in calls if a == "AQ3"}
    assert roll == {(b, e, s) for b in ("LEV6", "ROLL_DISP9", "DART3")
                    for (e, s) in (("logit", 42), ("xgb", 42), ("xgb", 0), ("xgb", 1), ("xgb", 7), ("xgb", 13))}


def test_main_bh_csv_holds_the_three_aq3_families_adjusted_one_by_one(sandbox, tmp_path):
    from statsmodels.stats.multitest import multipletests
    ri, calls, acts, panel = sandbox
    ri.main()
    bh = pd.read_csv(tmp_path / "inference" / "bh.csv")
    assert list(bh.columns) == ["family", "test", "estimate", "p", "bh_reject", "included"]
    assert bh["family"].tolist() == (["audit robustness"] * 18 + ["economic robustness"] * 14 + ["drift monitors"] * 12)
    assert bh["included"].tolist() == bh["p"].notna().tolist()                  # the NaN-p rows are not in BH's m
    assert bh.groupby("family", sort=False)["included"].sum().tolist() == [18, 13, 11]
    audit = bh[bh.family == "audit robustness"]
    assert audit["test"].tolist() == [f"{h} {n}" for h in ("H-B1", "H-B2")
                                      for n in ("logit", "frozen", "alpha=0.01", "alpha=0.025", "alpha=0.1",
                                                "seed=0", "seed=1", "seed=7", "seed=13")]
    econ_src, mon_src = _economics_summary(), _monitor_eval()
    econ = bh[bh.family == "economic robustness"]
    aq3 = econ_src[(econ_src.analysis == "AQ3") & (econ_src.policy == "P1") & (econ_src.comparator == "none")]
    assert econ["test"].tolist() == [f"P1 {r.position} fees={r.fees}" for r in aq3.itertuples()]
    assert np.allclose(econ["estimate"], aq3["SELC"]) and np.allclose(econ["p"], aq3["rotation_p"], equal_nan=True)
    mon = bh[bh.family == "drift monitors"]
    aq3m = mon_src[mon_src.analysis == "AQ3"]
    assert mon["test"].tolist() == [f"{r.block} {r.monitor}" for r in aq3m.itertuples()]
    assert np.allclose(mon["p"], aq3m["spearman_p_one_sided"], equal_nan=True)
    for family, g in bh.groupby("family"):
        ok = g["p"].notna().to_numpy()
        want = np.zeros(len(g), dtype=bool)
        want[ok] = multipletests(g["p"].to_numpy()[ok], alpha=0.05, method="fdr_bh")[0]
        assert g["bh_reject"].tolist() == want.tolist(), family
    # the decoys would have changed the decisions: with the AQ1 rows in the family BH rejects more
    polluted = econ_src[(econ_src.policy == "P1") & (econ_src.comparator == "none")]["rotation_p"].dropna()
    n_polluted = int(multipletests(polluted, alpha=0.05, method="fdr_bh")[0].sum())
    assert n_polluted > int(econ["bh_reject"].sum()) >= 1
    assert int(mon["bh_reject"].sum()) >= 1
    assert int(audit["bh_reject"].sum()) == 0          # with 30 replicates the smallest p is 1 / 31 > 0.05 / 18
    assert not mon.loc[mon["p"].isna(), "bh_reject"].any() and not econ.loc[econ["p"].isna(), "bh_reject"].any()


def test_main_writes_the_families_that_exist_and_says_which_are_missing(sandbox, tmp_path, capsys):
    ri, calls, acts, panel = sandbox
    (tmp_path / "economics" / "summary.csv").unlink()
    ri.main()
    out = capsys.readouterr().out
    bh = pd.read_csv(tmp_path / "inference" / "bh.csv")
    assert bh["family"].unique().tolist() == ["audit robustness", "drift monitors"] and len(bh) == 18 + 12
    assert "missing: economic robustness" in out and "summary.csv" in out and "missing: drift monitors" not in out
    (tmp_path / "monitors" / "monitor_eval.csv").unlink()
    ri.main()
    out = capsys.readouterr().out
    bh = pd.read_csv(tmp_path / "inference" / "bh.csv")
    assert bh["family"].unique().tolist() == ["audit robustness"] and len(bh) == 18
    assert "missing: economic robustness" in out and "missing: drift monitors" in out and "monitor_eval.csv" in out
    # the mixed model and the leave-one-operator-out table do not depend on those two outputs
    assert (tmp_path / "inference" / "mixed_model.txt").exists() and (tmp_path / "inference" / "looo.csv").exists()


def test_main_refuses_an_economics_or_monitor_file_it_cannot_use(sandbox, tmp_path):
    ri, calls, acts, panel = sandbox
    summary = _economics_summary()
    summary.drop(columns=["rotation_p"]).to_csv(tmp_path / "economics" / "summary.csv", index=False)
    with pytest.raises(ValueError, match="rotation_p"):                 # a stale or wrong file is an error, not "missing"
        ri.main()
    summary.assign(analysis="AQ1").to_csv(tmp_path / "economics" / "summary.csv", index=False)
    with pytest.raises(ValueError, match="AQ3"):
        ri.main()
    summary.to_csv(tmp_path / "economics" / "summary.csv", index=False)
    _monitor_eval().drop(columns=["spearman_p_one_sided"]).to_csv(tmp_path / "monitors" / "monitor_eval.csv", index=False)
    with pytest.raises(ValueError, match="spearman_p_one_sided"):
        ri.main()


def test_text_columns_of_the_summary_are_read_verbatim_and_the_numbers_stay_numbers(tmp_path):
    import math
    import es.run_inference as ri
    csv = tmp_path / "summary.csv"
    csv.write_text("analysis,policy,comparator,position,fees,SELC,rotation_p\n"
                   "AQ3,P1,None,NA,N/A,0.1,0.2\n"
                   "AQ3,comparator,null,nan,NaN,-0.5,\n"
                   "AQ3,P4,none,premium,0.50,2,0.75\n"
                   "AQ3,NA,none,premium,0.25,3,0.5\n"
                   "AQ1,P1,none,premium,0.50,0.3,0.4\n")
    default = pd.read_csv(csv)
    assert default["comparator"].isna().sum() == 2 and default["position"].isna().sum() == 2        # the trap
    assert default["policy"].isna().sum() == 1 and default["fees"].isna().sum() == 2
    rows = ri._aq3_records(csv, ri._ECON_COLUMNS, ri._ECON_TEXT)
    assert [r["policy"] for r in rows] == ["P1", "comparator", "P4", "NA"]
    assert [r["comparator"] for r in rows] == ["None", "null", "none", "none"]
    assert [r["position"] for r in rows] == ["NA", "nan", "premium", "premium"]
    assert [r["fees"] for r in rows] == ["N/A", "NaN", "0.50", "0.25"]
    assert [r["SELC"] for r in rows] == [0.1, -0.5, 2.0, 3.0] and all(type(r["SELC"]) is float for r in rows)
    assert [rows[0]["rotation_p"], rows[2]["rotation_p"], rows[3]["rotation_p"]] == [0.2, 0.75, 0.5]
    assert math.isnan(rows[1]["rotation_p"])                                  # an empty cell is the one missing value
    assert ri._ECON_TEXT == ("policy", "comparator", "position", "fees")


def test_main_reads_the_summary_labels_as_written(sandbox, tmp_path):
    ri, calls, acts, panel = sandbox
    s = _economics_summary()
    aq3_p1 = (s.analysis == "AQ3") & (s.policy == "P1") & (s.comparator == "none")
    s.loc[aq3_p1 & (s.position == "inc") & (s.fees == "0.0"), "fees"] = "None"        # labels pandas takes for missing
    s.loc[aq3_p1 & (s.position == "dec") & (s.fees == "0.25"), "position"] = "NA"
    extra = [{"analysis": "AQ3", "policy": "P1", "comparator": "None", "position": "premium", "fees": "0.25",
              "SELC": 0.9, "NET": 1.0, "rotation_p": 1e-5},                       # not a P1 / "none" row, and not NaN
             {"analysis": "AQ3", "policy": "comparator", "comparator": "None", "position": "NA", "fees": "NA",
              "SELC": 0.9, "NET": 1.0, "rotation_p": 1e-5}]
    pd.concat([s, pd.DataFrame(extra)], ignore_index=True).to_csv(tmp_path / "economics" / "summary.csv", index=False)
    ri.main()
    bh = pd.read_csv(tmp_path / "inference" / "bh.csv")
    econ = bh[bh.family == "economic robustness"]
    assert len(econ) == 14 and econ["included"].sum() == 13                      # none lost, none added, the NaN p out
    assert "P1 inc fees=None" in econ["test"].tolist() and "P1 NA fees=0.25" in econ["test"].tolist()
    assert not econ["test"].str.contains("nan").any()
    assert econ["p"].dtype == float and econ.loc[econ["test"] == "P1 inc fees=None", "p"].iloc[0] == 0.035


def test_main_refuses_a_summary_without_a_p1_row_of_comparator_none(sandbox, tmp_path):
    ri, calls, acts, panel = sandbox
    s = _economics_summary()
    s.loc[s.comparator == "none", "comparator"] = "None"                          # the label is kept, so it no longer matches
    s.to_csv(tmp_path / "economics" / "summary.csv", index=False)
    with pytest.raises(ValueError, match=r"P1.*comparator 'none'.*carry the comparators \['None', 'VOL@count'\]"):
        ri.main()
    assert not (tmp_path / "inference" / "bh.csv").exists() and not calls          # refused before any score was asked for


def test_main_keeps_the_fee_labels_of_the_summary_as_text(sandbox, tmp_path):
    ri, calls, acts, panel = sandbox
    summary = _economics_summary()
    summary["fees"] = summary["fees"].replace({"calibrated": "0.50", "stress": "1.0"})      # numeric-looking labels
    summary.to_csv(tmp_path / "economics" / "summary.csv", index=False)
    ri.main()
    bh = pd.read_csv(tmp_path / "inference" / "bh.csv")
    assert "P1 premium fees=0.50" in bh["test"].tolist() and "P1 dec fees=1.0" in bh["test"].tolist()


def test_main_writes_the_mixed_model_and_looo_before_the_bh_rehearsal(sandbox, tmp_path, monkeypatch):
    ri, calls, acts, panel = sandbox

    def boom(raw):
        raise RuntimeError("no raw prices here")

    monkeypatch.setattr(ri, "build_panel", boom)
    with pytest.raises(RuntimeError, match="no raw prices"):
        ri.main()
    assert (tmp_path / "inference" / "mixed_model.txt").exists() and (tmp_path / "inference" / "looo.csv").exists()
    assert not (tmp_path / "inference" / "bh.csv").exists()


def test_main_needs_the_analogue_activity_table(sandbox, tmp_path):
    ri, calls, acts, panel = sandbox
    (tmp_path / "analogs" / "activity_all.csv").unlink()
    with pytest.raises(FileNotFoundError, match="activity_all.csv"):
        ri.main()
    assert not (tmp_path / "inference").exists()                                     # not even an empty folder


def test_main_leaves_no_inference_folder_when_the_fit_fails(sandbox, tmp_path, monkeypatch):
    ri, calls, acts, panel = sandbox

    def boom(cells):
        raise RuntimeError("singular")

    monkeypatch.setattr(ri, "fit_mixed", boom)
    with pytest.raises(RuntimeError, match="singular"):
        ri.main()
    assert not (tmp_path / "inference").exists()


def test_the_audit_hands_the_bootstrap_only_the_columns_it_reads(sandbox, tmp_path, monkeypatch):
    """The memo behind the audit keeps a slim copy of each frame (21 full frames would not fit in memory: plan 2A's probe
    gives 338 MB for a 990,597-row frame), and the audit numbers are those of the full frames."""
    from types import SimpleNamespace
    import es.secondary as sec
    ri, calls, acts, panel = sandbox
    thin = ri.cached_scores                                       # the sandbox's stub: a score frame with its five columns
    extra = {"timestamp_utc": 0, "da_lmp": 1.0, "rt_lmp": 2.0, "dart_spread": 3.0, "label": 0, "abs_mean_168h": 4.0}

    def fat(p, analysis, block, estimator, seed):
        return thin(p, analysis, block, estimator, seed).assign(**extra)

    monkeypatch.setattr(ri, "cached_scores", fat)
    wrapped, seen = sec.block_contrast_bootstrap, []              # the sandbox's bootstrap, with 30 replicates
    columns = ["market", "node", "split", "settle_date", "score"]

    def spy(alt, base, **kwargs):
        seen.append((list(alt.columns), list(base.columns)))
        return wrapped(alt, base, **kwargs)

    monkeypatch.setattr(sec, "block_contrast_bootstrap", spy)
    ri.main()
    assert len(seen) == 18 and all(cols == columns for pair in seen for cols in pair)
    bh = pd.read_csv(tmp_path / "inference" / "bh.csv", float_precision="round_trip")
    direct = sec.audit_family(lambda kind, b, e, s: fat(panel, SimpleNamespace(id="AQ3F" if kind == "frozen" else "AQ3"),
                                                       b, e, s))
    assert [list(c) for c in seen[-1]] == [columns + list(extra)] * 2                # the direct audit sees the full frames
    assert bh.loc[bh.family == "audit robustness", "estimate"].tolist() == [r["estimate"] for r in direct]
    assert bh.loc[bh.family == "audit robustness", "p"].tolist() == [r["p"] for r in direct]


@pytest.mark.parametrize("failure", ["panel", "economics file"])
def test_a_failed_run_leaves_no_stale_bh_csv_beside_fresh_outputs(sandbox, tmp_path, monkeypatch, failure):
    ri, calls, acts, panel = sandbox
    inf = tmp_path / "inference"
    inf.mkdir()
    for name in ("bh.csv", "mixed_model.txt", "looo.csv"):
        (inf / name).write_text("stale\n")                        # the outputs of an earlier run
    if failure == "panel":
        def boom(raw):
            raise RuntimeError("no raw prices here")
        monkeypatch.setattr(ri, "build_panel", boom)
        expected = RuntimeError
    else:
        _economics_summary().drop(columns=["rotation_p"]).to_csv(tmp_path / "economics" / "summary.csv", index=False)
        expected = ValueError
    with pytest.raises(expected):
        ri.main()
    assert not (inf / "bh.csv").exists()                                        # no stale file beside the fresh ones
    assert "Mixed Linear Model" in (inf / "mixed_model.txt").read_text()
    assert (inf / "looo.csv").read_text().startswith("analysis,block,dropped,failures")


def test_bh_csv_is_written_through_a_temp_file_and_renamed(sandbox, tmp_path, monkeypatch):
    from types import SimpleNamespace
    ri, calls, acts, panel = sandbox
    inf = tmp_path / "inference"
    seen = []

    def watch_replace(src, dst):                         # the file is complete and under its temp name when it is renamed
        seen.append((src.name, dst.name, pd.read_csv(src).shape))
        return os_replace(src, dst)

    import os
    os_replace = os.replace
    monkeypatch.setattr(ri, "os", SimpleNamespace(replace=watch_replace))
    ri.main()
    assert seen == [("bh.csv.tmp", "bh.csv", (44, 6))]                           # 18 + 14 + 12 rows, six columns
    assert sorted(p.name for p in inf.iterdir()) == ["bh.csv", "looo.csv", "mixed_model.txt"]


def test_a_crash_before_the_rename_leaves_neither_bh_csv_nor_the_temp_file(sandbox, tmp_path, monkeypatch):
    from types import SimpleNamespace
    ri, calls, acts, panel = sandbox
    inf = tmp_path / "inference"
    inf.mkdir()
    (inf / "bh.csv").write_text("stale\n")

    def no_replace(src, dst):
        raise OSError("crash before the rename")

    monkeypatch.setattr(ri, "os", SimpleNamespace(replace=no_replace))
    with pytest.raises(OSError, match="crash before the rename"):
        ri.main()
    assert sorted(p.name for p in inf.iterdir()) == ["looo.csv", "mixed_model.txt"]


def test_a_crash_while_bh_csv_is_written_leaves_no_partial_file(sandbox, tmp_path, monkeypatch):
    ri, calls, acts, panel = sandbox
    inf = tmp_path / "inference"
    guarded = pd.DataFrame.to_csv                        # the sandbox's guard around the real writer

    def half_written(self, path_or_buf=None, *a, **k):
        if str(path_or_buf).endswith("bh.csv.tmp"):
            with open(path_or_buf, "w") as handle:
                handle.write("family,test\npartial")
            raise OSError("disk full")
        return guarded(self, path_or_buf, *a, **k)

    monkeypatch.setattr(pd.DataFrame, "to_csv", half_written)
    with pytest.raises(OSError, match="disk full"):
        ri.main()
    assert sorted(p.name for p in inf.iterdir()) == ["looo.csv", "mixed_model.txt"]
    monkeypatch.setattr(pd.DataFrame, "to_csv", guarded)
    ri.main()                                            # the next healthy run writes it
    assert sorted(p.name for p in inf.iterdir()) == ["bh.csv", "looo.csv", "mixed_model.txt"]


def test_p_values_are_read_back_exactly(tmp_path):
    import es.run_inference as ri
    p = [j / 92 for j in range(1, 92)]                   # pandas' default float parser changes 36 of these 91 on a round trip
    path = tmp_path / "summary.csv"
    pd.DataFrame({"analysis": "AQ3", "policy": "P1", "comparator": "none", "position": "premium", "fees": "0.25",
                  "SELC": p, "rotation_p": p}).to_csv(path, index=False)
    for text in (ri._ECON_TEXT, ()):                     # the economics reader, and the monitor reader (pandas' defaults)
        rows = ri._aq3_records(path, ri._ECON_COLUMNS, text)
        assert [r["rotation_p"] for r in rows] == p and [r["SELC"] for r in rows] == p        # bit for bit


def test_the_sandbox_stops_a_write_under_the_real_outputs(sandbox):
    import es.config as config
    with pytest.raises(AssertionError, match="real outputs"):
        (config.DEV_ROOT / "outputs" / "es" / "inference" / "never_written.txt").write_text("x")
    with pytest.raises(AssertionError, match="real outputs"):
        pd.DataFrame({"a": [1]}).to_csv(config.DEV_ROOT / "outputs" / "es" / "never_written.csv")
    assert not (config.DEV_ROOT / "outputs" / "es" / "inference" / "never_written.txt").exists()
