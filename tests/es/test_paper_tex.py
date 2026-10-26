import re

import pytest


def test_rounding_is_half_up_and_prints_a_real_minus():
    from es.paper_tex import MINUS, fmt_dec, fmt_int, fmt_pct, fmt_usd, round_half_up
    assert round_half_up(0.125, 2) == "0.13"          # banker's rounding would give 0.12
    assert round_half_up(2.5, 0) == "3"
    assert fmt_dec(-0.04, 2) == MINUS + "0.04"
    assert fmt_dec(-0.001, 2) == "0.00"                # no minus on a rounded zero
    assert fmt_int(29837) == "29{,}837"
    assert fmt_pct(0.1234, 1) == "12.3\\%"
    assert fmt_usd(-1234.5, 0) == MINUS + "\\$1{,}235"
    assert fmt_usd(2.345, 2) == "\\$2.35"


def test_pvalue_floor_fraction_and_ci():
    from es.paper_tex import FORMATS, PAIR_FORMATS, fmt_pval
    assert fmt_pval(0.0004) == "$<$0.001"
    assert fmt_pval(0.0166) == "0.017"
    assert PAIR_FORMATS["frac"](12, 33) == "12/33"
    assert PAIR_FORMATS["ci2"](-0.1, 0.25) == "[" + FORMATS["dec2"](-0.1) + ", 0.25]"
    assert PAIR_FORMATS["ciusd0"](-10, 2000) == "[\\textminus{}\\$10, \\$2{,}000]"


def test_missing_values_raise_and_names_are_letters():
    from es.paper_tex import fmt_dec, macro_name_ok, tex_escape
    with pytest.raises(ValueError):
        fmt_dec(float("nan"), 2)
    assert macro_name_ok("failLevYY") and not macro_name_ok("fail2024") and not macro_name_ok("fail_lev")
    assert tex_escape("P4@r* & 5%") == "P4@r* \\& 5\\%"


def test_tabular_has_booktabs_rules():
    from es.paper_tex import tabular
    t = tabular("lrr", ["block", "2024", "2026"], [["cal", "0", "0"], ["lev", "13", "1"]], midrules_before={1})
    assert t.startswith("\\begin{tabular}{lrr}\n\\toprule\n") and t.count("\\midrule") == 2
    assert "cal & 0 & 0 \\\\" in t and t.rstrip().endswith("\\end{tabular}")


def test_pct_and_g_round_half_up_on_the_decimal_string():
    from es.paper_tex import MINUS, fmt_g, fmt_pct
    assert fmt_pct(0.0045, 1) == "0.5\\%"              # 0.0045 * 100 is 0.44999999999999996 in binary: it must shift in decimal
    assert fmt_pct(0.0515, 1) == "5.2\\%"              # binary 5.1499999999999995
    assert fmt_pct(0.145, 0) == "15\\%"                # binary 14.499999999999998
    assert fmt_pct(-0.0045, 1) == MINUS + "0.5\\%" and fmt_pct(-0.00001, 1) == "0.0\\%"   # no minus on a rounded zero
    assert fmt_g(0.1235, 3) == "0.124"                 # f"{0.1235:.3g}" gives 0.123
    assert fmt_g(5.0, 3) == "5" and fmt_g(0.5, 3) == "0.5" and fmt_g(9.995, 3) == "10" and fmt_g(99.95, 3) == "100"
    assert fmt_g(-0.04, 3) == MINUS + "0.04" and fmt_g(0, 3) == "0"


def test_fmt_g_never_prints_scientific_notation():
    from es.paper_tex import fmt_g
    assert fmt_g(123456, 3) == "123000" and fmt_g(1.2e-16, 3) == "0.00000000000000012"
    assert fmt_g(1e-5, 3) == "0.00001" and fmt_g(2.5e10, 3) == "25000000000"
    for v in (1e-9, 1.5e-7, 123456789.0, 1e22, 0.000123456, 12.0, 100.0):
        s = fmt_g(v, 3)
        assert re.fullmatch(r"\d+(\.\d+)?", s), (v, s)           # plain digits only: no e, no exponent


def test_non_finite_and_out_of_range_values_raise_and_big_values_do_not_overflow():
    from es.paper_tex import fmt_dec, fmt_g, fmt_int, fmt_pct, fmt_pval, fmt_usd
    for bad in (float("inf"), float("-inf"), float("nan")):
        for fmt in (lambda v: fmt_dec(v, 2), fmt_int, lambda v: fmt_pct(v, 1), fmt_usd, fmt_g, fmt_pval):
            with pytest.raises(ValueError):
                fmt(bad)
    for p in (-0.1, 1.5, float("-inf"), float("inf")):
        with pytest.raises(ValueError):
            fmt_pval(p)                                              # -inf used to print "<0.001"
    assert fmt_pval(0.0) == "$<$0.001" and fmt_pval(1.0) == "1.000" and fmt_pval(0.001) == "0.001"
    assert fmt_dec(1e30, 2) == "1000000000000000000000000000000.00"  # no decimal.InvalidOperation on a big value
    assert fmt_usd(1e25, 0) == "\\$10" + "{,}000" * 8                   # 1e25 has 26 digits: no float noise, no overflow


def test_yesno_raises_on_unknown_strings():
    from es.paper_tex import FORMATS
    yesno = FORMATS["yesno"]
    assert [yesno(v) for v in (True, "true", "Yes", 1, "1")] == ["yes"] * 5
    assert [yesno(v) for v in (False, "false", "No", 0, "0")] == ["no"] * 5
    for bad in ("maybe", "", None, 2, "yes!"):
        with pytest.raises(ValueError):
            yesno(bad)


# ---------------------------------------------------------------------------------------------- Ruling 65
def test_tex_escape_maps_bar_less_and_greater_to_text_commands():
    from es.paper_tex import tex_escape
    # under pdflatex's OT1 encoding the raw characters print as an em dash, an inverted ! and an inverted ?
    assert tex_escape("spike = |DART| >= q") == "spike = \\textbar{}DART\\textbar{} \\textgreater{}= q"
    assert tex_escape("MAD <= 1e-6") == "MAD \\textless{}= 1e-6"
    assert tex_escape("<|>") == "\\textless{}\\textbar{}\\textgreater{}"


def test_tex_escape_is_one_pass_and_leaves_strings_without_the_new_characters_unchanged():
    from es.paper_tex import tex_escape
    before = {"&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", "_": "\\_", "{": "\\{", "}": "\\}",
              "~": "\\textasciitilde{}", "^": "\\textasciicircum{}", "\\": "\\textbackslash{}"}     # the escapes before Ruling 65
    for s in ("P4@r* & 5%", "cal_decile x_1", "E2: SNET (\\$/MW-day)", "market $\\times$ node", "a~b^c{d}#", "plain text"):
        assert tex_escape(s) == "".join(before.get(ch, ch) for ch in s), s      # no |, < or >: exactly as before
    out = tex_escape("a|b<c>d")
    assert out == "a\\textbar{}b\\textless{}c\\textgreater{}d" and "textbackslash" not in out   # a replacement is never escaped again


def test_formatters_that_emit_latex_are_not_routed_through_tex_escape():
    from es.paper_registry import CODE_CONSTANTS
    from es.paper_tex import FORMATS, PAIR_FORMATS
    assert FORMATS["pval"](0.0004) == "$<$0.001"                       # the "<" of a math-mode p-value floor stays as it is
    assert FORMATS["pct1"](0.1234) == "12.3\\%" and FORMATS["usd0"](-1234.5) == "\\textminus{}\\$1{,}235"
    assert PAIR_FORMATS["ciusd0"](-10, 2000) == "[\\textminus{}\\$10, \\$2{,}000]"
    assert CODE_CONSTANTS["strataDecile"] == "market $\\times$ node $\\times$ cal\\_decile"   # parts escaped, math joined after
    assert "$\\times$ settle\\_hour $\\times$" in CODE_CONSTANTS["strataPrimary"]


def test_row_status_and_reasons_formats():
    """Ruling 101: a row's labels in one text, the registered status, and the reasons of a demotion."""
    from es.paper_tex import FORMATS, ROW_FORMATS
    where = ROW_FORMATS["where"]
    assert where(["LEV6", "common_drift_0.5", "upper"]) == "LEV6, common\\_drift\\_0.5, upper"
    assert where([float("nan"), "heteroskedastic", "fit"]) == "heteroskedastic, fit"      # an H-E row has no law
    assert where([None, "no_drift", ""]) == "no\\_drift"
    with pytest.raises(ValueError):
        where([float("nan"), None, ""])                                                  # a row with no label at all
    assert FORMATS["registered"](True) == "confirmatory" and FORMATS["registered"]("false") == "estimand"
    assert FORMATS["registered"](0) == "estimand" and FORMATS["registered"]("yes") == "confirmatory"
    with pytest.raises(ValueError):
        FORMATS["registered"]("perhaps")                                                 # never a silent "estimand"
    assert FORMATS["reasons"]([]) == "none"
    assert FORMATS["reasons"](["size 0.130 > 1.5 x 0.0167 (LEV6, common_drift_0.5)", "power 0.28 < 0.5"]) == (
        "size 0.130 \\textgreater{} 1.5 x 0.0167 (LEV6, common\\_drift\\_0.5); power 0.28 \\textless{} 0.5")
    assert FORMATS["reasons"]("one reason") == "one reason" and FORMATS["reasons"](("a", "b")) == "a; b"
    for bad in (7, [1, 2], None, {"a": 1}):
        with pytest.raises(ValueError):
            FORMATS["reasons"](bad)
