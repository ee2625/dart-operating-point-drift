import re
import shutil
import subprocess
import warnings
import zlib
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

PDFFONTS = shutil.which("pdffonts") or "/opt/homebrew/bin/pdffonts"


#: Failing nodes of the six core blocks (calendar, plain DART lags, absolute level, rolling dispersion, DART-relative, combined) in each
#: column of Table 3 that FigFail draws: the numbers of the real audit, so that the fixture looks like the real frame.
FAIL_FIXTURE = {("2024-2025", "xgb"): [0, 0, 13, 5, 7, 12], ("25H1-26H1", "xgb"): [0, 0, 1, 15, 12, 8],
                ("2024-2025", "logit"): [0, 0, 31, 19, 20, 29], ("25H1-26H1", "logit"): [0, 0, 13, 4, 3, 7],
                ("frozen-26H1", "xgb"): [0, 9, 12, 10, 9, 23]}


def _write_summary(out: Path, n_nodes=33, fixture=None):
    """legacy/summary.csv as es.run_legacy writes it (the columns that es.paper_tables._summary needs, and n_nodes), with the primary rows of
    FAIL_FIXTURE and, next to them, rows that must not be read: another seed, another policy, another cutoff target, another block."""
    rows = []
    common = {"median_R": 1.0, "median_log_error": 0.1, "median_auprc": 0.1, "prevalence": 0.05, "n_nodes": n_nodes}
    for (analysis, est), counts in (fixture or FAIL_FIXTURE).items():
        for block, k in zip(BLOCKS, counts):
            rows.append({"analysis": analysis, "block": block, "estimator": est, "seed": 42, "policy": "P1", "alpha": 0.05, "failures": k, **common})
            rows.append({"analysis": analysis, "block": block, "estimator": est, "seed": 0, "policy": "P1", "alpha": 0.05, "failures": 99, **common})
            rows.append({"analysis": analysis, "block": block, "estimator": est, "seed": 42, "policy": "P2_all_train", "alpha": 0.05, "failures": 98, **common})
            rows.append({"analysis": analysis, "block": block, "estimator": est, "seed": 42, "policy": "P1", "alpha": 0.01, "failures": 97, **common})
        rows.append({"analysis": analysis, "block": "MID3", "estimator": est, "seed": 42, "policy": "P1", "alpha": 0.05, "failures": 96, **common})
    pd.DataFrame(rows).to_csv(out / "legacy" / "summary.csv", index=False)


def _write_drift(out: Path, weeks=8, analyses=None):
    """paper/drift_path.csv in the form es.run_driftpath writes it: two analyses, P1 and P4, five nodes, `weeks` Monday-start weeks, the first and the
    last of which are partial. A node's weekly alerts follow a fixed pattern, so that the medians and quartiles are order statistics of five values."""
    analyses = analyses or {"2024-2025": "2025-01-06", "25H1-26H1": "2026-01-05"}
    rows = []
    for analysis, first in analyses.items():
        for policy, shift in (("P1", 3), ("P4", 1)):
            for market, node in (("NYISO", "N0"), ("NYISO", "N1"), ("ISONE", "N2"), ("MISO", "N3"), ("CAISO", "N4")):
                for w in range(weeks):
                    alerts = (int(node[1]) * shift + w) % 17
                    full = 0 < w < weeks - 1
                    n_hours = 168 if full else 96
                    rows.append({"analysis": analysis, "policy": policy, "market": market, "node": node,
                                 "week_start": (pd.Timestamp(first) + pd.Timedelta(days=7 * w)).strftime("%Y-%m-%d"), "n_hours": n_hours, "alerts": alerts,
                                 "rate": alerts / n_hours, "R": alerts / n_hours / 0.05, "full_week": full})
    (out / "paper").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "paper" / "drift_path.csv", index=False)


def _sources(tmp_path: Path):
    from es.paper_macros import Sources
    out = tmp_path / "out"
    (out / "legacy").mkdir(parents=True)
    (out / "analogs").mkdir()
    (out / "economics").mkdir()
    _write_summary(out)
    _write_drift(out)
    rng = np.random.default_rng(0)
    nodes = [(m, f"{m}{i}") for m, k in (("NYISO", 10), ("ISONE", 9), ("MISO", 8), ("CAISO", 6)) for i in range(k)]
    for aid in ("2024-2025", "25H1-26H1"):
        for b in ("G6", "DART3", "LEV6", "ROLL_DISP9", "DART18", "G1_24"):
            pd.DataFrame({"market": [m for m, _ in nodes], "node": [n for _, n in nodes], "R": rng.lognormal(0, 0.3, 33)}) \
                .to_csv(out / "legacy" / f"{aid}__{b}__xgb__s42__P1__a0.05.csv", index=False)
    act = pd.DataFrame([{"analysis": a, "block": b, "estimator": "xgb", "policy": "P1", "market": m, "node": n,
                         "R": rng.lognormal(0, 0.3), "log_error": abs(rng.normal(0, 0.2)), "fail_0.5_2": False}
                        for a in ("AQ3", "AQ4", "AQ2") for b in ("G6", "DART3", "LEV6", "ROLL_DISP9", "DART18", "G1_24") for m, n in nodes])
    act.to_csv(out / "analogs" / "activity_all.csv", index=False)
    stats = ["SELC", "NET", "E3:ESred", "E4:SELC_decile", "E4:SELC|calendar@count", "E4:SELC|VOL@count",
             "E4:dSELC|calendar@count", "E4:dSELC|VOL@count",
             "E4:ESred|calendar@count", "E4:ESred|VOL@count", "E4:dESred|calendar@count", "E4:dESred|VOL@count"] + \
            [f"E1:{k}:{p}" for k in ("NET", "ESred", "J_0.5") for p in ("timing", "level", "total")]     # Ruling 81: the dSELC pair; Ruling 118: ESred
    pd.DataFrame([{"analysis": a, "stat": s, "estimate": 10.0, "lo": 5.0, "hi": 15.0, "approximate": False, "n_windows": 1}
                  for a in ("AQ3", "AQ4", "AQ1", "AQ2") for s in stats]).to_csv(out / "economics" / "bootstrap_ci.csv", index=False)
    return Sources(out, tmp_path / "paper", tmp_path / "docs")


def test_all_seven_figures_build_without_sealed_results(tmp_path):
    from es.paper_figures import build_all
    src = _sources(tmp_path)
    figdir = tmp_path / "paper" / "generated" / "figures"
    status = build_all(src, figdir)
    for n in ("Fig1", "Fig2", "Fig3", "Fig4"):
        assert status[n] == "built", status          # (T1b) these four print nothing of Test A while it has not run, and the index no longer says "pending" either
        assert (figdir / f"{n}.pdf").exists() and (figdir / f"{n}.eps").exists()
    for n in ("FigFail", "FigDrift", "FigContrast"):
        assert status[n] == "built", status                           # these three never have a Test A panel
        assert (figdir / f"{n}.pdf").exists() and (figdir / f"{n}.eps").exists()
    assert "pending" not in (figdir / "INDEX.md").read_text()


def test_pdf_fonts_are_embedded_arial_without_type3_or_transparency(tmp_path):
    from es.paper_figures import build_all
    src = _sources(tmp_path)
    figdir = tmp_path / "paper" / "generated" / "figures"
    build_all(src, figdir)
    for n in ALL_FIGS:
        data = (figdir / f"{n}.pdf").read_bytes()
        assert not re.search(rb"/(?:ca|CA) 0\.\d", data), f"{n}: transparency found; EPS cannot carry it"
        if Path(PDFFONTS).exists():
            fonts = subprocess.run([PDFFONTS, str(figdir / f"{n}.pdf")], capture_output=True, text=True).stdout
            assert "Arial" in fonts and "Type 3" not in fonts, fonts
            for line in fonts.splitlines()[2:]:
                assert " yes " in line, f"{n}: font not embedded: {line}"


def test_eps_is_vector_arial_and_narrower_than_174mm(tmp_path):
    from es.paper_figures import build_all
    src = _sources(tmp_path)
    figdir = tmp_path / "paper" / "generated" / "figures"
    build_all(src, figdir)
    for n in ALL_FIGS:
        eps = (figdir / f"{n}.eps").read_bytes()
        assert eps.startswith(b"%!PS-Adobe-3.0 EPSF-3.0") and b"/FontName /ArialMT" in eps
        m = re.search(rb"%%BoundingBox: 0 0 (\d+) (\d+)", eps)
        assert m and int(m.group(1)) <= 380, f"{n}: wider than the sn-jnl text width (372 pt; the journal limit is 493 pt)"


def test_sealed_panels_appear_when_test_a_exists(tmp_path):
    from es.paper_figures import build_all
    src = _sources(tmp_path)
    sealed = src.out_dir / "sealed" / "A" / "tables"
    sealed.mkdir(parents=True)
    (src.out_dir / "sealed" / "A" / "results.json").write_text("{}")
    act = src.frame("analogs/activity_all.csv")
    act[act.analysis == "AQ3"].assign(maintenance="roll", seed=42).to_csv(sealed / "audit_activity.csv", index=False)
    ci = src.frame("economics/bootstrap_ci.csv")
    ci[ci.analysis == "AQ3"].drop(columns="analysis").to_csv(sealed / "estimands.csv", index=False)
    pd.DataFrame({"block": ["G6", "DART3", "LEV6", "ROLL_DISP9", "DART18", "G1_24"], "median_log_error": 0.1, "lo": 0.05, "hi": 0.2,
                  "failures": 3, "failures_lo": 1, "failures_hi": 6, "n_nodes": 33}).to_csv(sealed / "E5_blocks.csv", index=False)
    status = build_all(src, tmp_path / "paper" / "generated" / "figures")
    assert all(v == "built" for v in status.values()), status


# ---------------------------------------------------------------------------------------------------------------------
# Checks beyond the four above: what the brief states about the figures but its first tests do not measure (lettering size,
# width at \textwidth, hatching, Arial only), and the defects found when the figures were drawn and looked at.
# ---------------------------------------------------------------------------------------------------------------------
PDFTOPPM = shutil.which("pdftoppm") or "/opt/homebrew/bin/pdftoppm"
FIGS = ("Fig1", "Fig2", "Fig3", "Fig4")
NEW_FIGS = ("FigFail", "FigDrift", "FigContrast")        # the three figures of plan 2C: one finding each, never a Test A panel
ALL_FIGS = FIGS + NEW_FIGS
TEXT_WIDTH_BP = 372 / 72.27 * 72          # the sn-jnl \textwidth (372 TeX pt) in PostScript points: what \includegraphics scales to
BLOCKS = ("G6", "DART3", "LEV6", "ROLL_DISP9", "DART18", "G1_24")


def _add_test_a(src):
    """Write the Test A tables the way test_sealed_panels_appear_when_test_a_exists does."""
    sealed = src.out_dir / "sealed" / "A" / "tables"
    sealed.mkdir(parents=True)
    (src.out_dir / "sealed" / "A" / "results.json").write_text("{}")
    act = src.frame("analogs/activity_all.csv")
    act[act.analysis == "AQ3"].assign(maintenance="roll", seed=42).to_csv(sealed / "audit_activity.csv", index=False)
    ci = src.frame("economics/bootstrap_ci.csv")
    ci[ci.analysis == "AQ3"].drop(columns="analysis").to_csv(sealed / "estimands.csv", index=False)
    pd.DataFrame({"block": list(BLOCKS), "median_log_error": 0.1, "lo": 0.05, "hi": 0.2, "failures": 3, "failures_lo": 1,
                  "failures_hi": 6, "n_nodes": 33}).to_csv(sealed / "E5_blocks.csv", index=False)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """{state: (figures_dir, src, status)} for the pending and the sealed state, built once. out["warnings"] holds what
    drawing raised: the UserWarnings (a missing glyph, tight_layout not applied) and matplotlib's own log warnings (font fallback)."""
    import logging

    from es.paper_figures import build_all
    logged = []

    class Collect(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                logged.append(record.getMessage())

    handler, mpl_log = Collect(), logging.getLogger("matplotlib")
    mpl_log.addHandler(handler)
    out = {}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for state in ("pending", "sealed"):
                root = tmp_path_factory.mktemp(state)
                src = _sources(root)
                if state == "sealed":
                    _add_test_a(src)
                figdir = root / "paper" / "generated" / "figures"
                out[state] = (figdir, src, build_all(src, figdir))
    finally:
        mpl_log.removeHandler(handler)
    out["warnings"] = [str(w.message) for w in caught if issubclass(w.category, UserWarning)] + logged
    return out


def _pdf_font_sizes(path: Path) -> set[float]:
    """Every size given to a Tf operator in the content streams of a matplotlib PDF."""
    data = path.read_bytes()
    sizes = set()
    for m in re.finditer(rb"stream\r?\n", data):
        raw = data[m.end():data.find(b"endstream", m.end())]
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            pass
        sizes |= {float(s) for s in re.findall(rb"/F\d+(?:-\d+)? ([0-9.]+) Tf", raw)}
    return sizes


def test_no_lettering_is_smaller_than_8_pt_in_the_pdf_or_the_eps(built):
    """The brief: no font is smaller than 8 pt. mathtext sets sub- and superscripts at 70 % of the size (6.3 pt at 9 pt)."""
    for state in ("pending", "sealed"):
        figdir = built[state][0]
        for n in ALL_FIGS:
            pdf_sizes = _pdf_font_sizes(figdir / f"{n}.pdf")
            assert pdf_sizes and min(pdf_sizes) >= 8.0, f"{state} {n}.pdf: font sizes {sorted(pdf_sizes)}"
            eps_sizes = {float(s) for s in re.findall(rb"/\S+ ([0-9.]+) selectfont", (figdir / f"{n}.eps").read_bytes())}
            assert eps_sizes and min(eps_sizes) >= 8.0, f"{state} {n}.eps: font sizes {sorted(eps_sizes)}"


def test_figures_are_not_wider_than_the_text_width_so_inclusion_never_shrinks_the_lettering(built):
    """Included at \\textwidth (372 TeX pt = 370.6 bp), a file wider than that would be scaled down and its 8 pt lettering with it;
    a file narrower than 75 % of it would be scaled up until its 9 pt labels print above the journal's 12 pt."""
    for state in ("pending", "sealed"):
        figdir = built[state][0]
        for n in ALL_FIGS:
            media = re.search(rb"/MediaBox \[\s*0 0 ([0-9.]+) ([0-9.]+)\s*\]", (figdir / f"{n}.pdf").read_bytes())
            assert media and float(media.group(1)) <= TEXT_WIDTH_BP + 0.01, f"{state} {n}.pdf is {media.group(1)} bp wide"
            scale = TEXT_WIDTH_BP / float(media.group(1))
            sizes = _pdf_font_sizes(figdir / f"{n}.pdf")
            assert 8.0 <= min(sizes) * scale and max(sizes) * scale <= 12.0, f"{state} {n}: prints at {min(sizes) * scale:.2f}-{max(sizes) * scale:.2f} pt"
            assert float(media.group(2)) <= 234 / 25.4 * 72, f"{n}: taller than the journal's 234 mm"
            hires = re.search(rb"%%HiResBoundingBox: 0\.0+ 0\.0+ ([0-9.]+) ([0-9.]+)", (figdir / f"{n}.eps").read_bytes())
            assert hires and float(hires.group(1)) <= TEXT_WIDTH_BP + 0.01, f"{state} {n}.eps is {hires.group(1)} bp wide"


def test_every_embedded_font_is_an_arial_face(built):
    if not Path(PDFFONTS).exists():
        pytest.skip("pdffonts not installed")
    for state in ("pending", "sealed"):
        for n in ALL_FIGS:
            lines = subprocess.run([PDFFONTS, str(built[state][0] / f"{n}.pdf")], capture_output=True, text=True).stdout.splitlines()[2:]
            assert lines and all("Arial" in line.split()[0] for line in lines), f"{state} {n}: {lines}"


def test_drawing_raises_no_missing_glyph_font_fallback_or_layout_warning(built):
    assert built["warnings"] == []


def test_the_fig3_level_bars_are_hatched_and_the_timing_bars_are_not(tmp_path):
    from es.paper_figures import e1_frame, fig3_decomposition
    fig = fig3_decomposition(e1_frame(_sources(tmp_path)), False)
    hatches = {}                                              # bar offset from its row (timing -0.2, level +0.2) -> hatches
    for p in fig.axes[0].patches:
        centre = p.get_y() + p.get_height() / 2
        hatches.setdefault(round(centre - round(centre), 1), set()).add(p.get_hatch())
    legend_hatches = {p.get_label(): p.get_hatch() for p in fig.legends[0].get_patches()}
    plt.close(fig)
    assert hatches == {-0.2: {None}, 0.2: {"//"}}, hatches
    assert legend_hatches == {"timing": None, "level": "//"}, legend_hatches


def test_a_row_without_estimates_gets_no_bars(tmp_path):
    """The Test A row is empty while the test is pending; a zero-length bar would still print its black edge on the zero line."""
    from es.paper_figures import e1_frame, fig3_decomposition
    fig = fig3_decomposition(e1_frame(_sources(tmp_path)), False)          # rows AQ3, AQ4, AQ1, AQ2 have data; SA is pending
    n_bars = [len(ax.patches) for ax in fig.axes]
    plt.close(fig)
    assert n_bars == [8, 8, 8], n_bars


def test_figures_with_missing_sources_are_skipped_and_the_index_says_why(tmp_path):
    from es.paper_figures import build_all
    src = _sources(tmp_path)
    (src.out_dir / "analogs" / "activity_all.csv").unlink()
    figdir = tmp_path / "paper" / "generated" / "figures"
    status = build_all(src, figdir)
    assert status["Fig1"] == "skipped: missing analogs/activity_all.csv" and status["Fig4"] == status["Fig1"]
    assert status["Fig2"] == status["Fig3"] == "built"
    assert [status[n] for n in NEW_FIGS] == ["built"] * 3                    # their sources are other files
    assert not (figdir / "Fig1.pdf").exists() and not (figdir / "Fig4.eps").exists() and (figdir / "Fig2.pdf").exists()
    index = (figdir / "INDEX.md").read_text()
    assert "| Fig1 |" in index and "skipped: missing analogs/activity_all.csv" in index
    (src.out_dir / "economics" / "bootstrap_ci.csv").unlink()
    status = build_all(src, tmp_path / "again")
    assert {status[n] for n in FIGS} == {"skipped: missing analogs/activity_all.csv", "skipped: missing economics/bootstrap_ci.csv"}
    assert status["FigContrast"] == "skipped: missing economics/bootstrap_ci.csv" and status["FigFail"] == status["FigDrift"] == "built"


def test_a_placeholder_test_a_panel_keeps_the_y_tick_labels_of_the_shared_axis(built):
    """_pending() used set_yticks([]) on an axis shared with every other panel, which removed all the y tick labels. The placeholder is drawn only
    when Test A did not run ('Test A: did not run'); while it has not run there is no panel at all (see test_fig1_draws_five_panels_...)."""
    from es.paper_figures import blocks_frame, fig1_r_drift, fig4_blocks, r_drift_frame
    src = built["pending"][1]
    act, _ = blocks_frame(src)
    for fig in (fig1_r_drift(r_drift_frame(src), False, pending="did not run"), fig4_blocks(act, None, False, pending="did not run")):
        fig.canvas.draw()
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels() if t.get_text()]
        plt.close(fig)
        assert len(labels) >= 3, labels


def test_an_estimate_outside_its_interval_still_draws(tmp_path):
    """A bootstrap percentile interval need not contain the estimate (the ESred interval is only approximate)."""
    from es.paper_figures import build_all
    src = _sources(tmp_path)
    p = src.out_dir / "economics" / "bootstrap_ci.csv"
    ci = pd.read_csv(p)
    ci.loc[ci.stat.isin(["SELC", "E3:ESred", "E1:NET:timing", "E1:J_0.5:level"]), "estimate"] = 25.0     # lo 5, hi 15
    ci.to_csv(p, index=False)
    status = build_all(src, tmp_path / "figs")
    assert status["Fig2"] == status["Fig3"] == "built"


def test_no_legend_sits_on_a_plotting_area(built):
    from es.paper_figures import blocks_frame, e1_frame, fig1_r_drift, fig2_selc_esred, fig3_decomposition, fig4_blocks, r_drift_frame
    for state, sealed in (("pending", False), ("sealed", True)):
        src = built[state][1]
        act, e5 = blocks_frame(src)
        figs = {"Fig1": fig1_r_drift(r_drift_frame(src), sealed), "Fig2": fig2_selc_esred(e1_frame(src), sealed),
                "Fig3": fig3_decomposition(e1_frame(src), sealed), "Fig4": fig4_blocks(act, e5, sealed)}
        for name, fig in figs.items():
            fig.canvas.draw()
            r = fig.canvas.get_renderer()
            legends = list(fig.legends) + [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
            assert legends or name == "Fig4", f"{name}: no legend"
            for leg in legends:
                box = leg.get_window_extent(r)
                for ax in fig.axes:
                    assert not box.overlaps(ax.get_window_extent(r)), f"{state} {name}: a legend covers a plotting area"
            plt.close(fig)


def test_no_ink_touches_the_edge_of_the_saved_page(built, tmp_path):
    """bbox_inches='tight' ignores the x-extent of an x label that is wider than its axes (matplotlib 3.9), so such a label
    is cut off by the page. Rasterise each PDF and look at the outermost pixel ring: with the 0.02 in pad it is white."""
    from PIL import Image
    if not Path(PDFTOPPM).exists():
        pytest.skip("pdftoppm not installed")
    for state in ("pending", "sealed"):
        for n in ALL_FIGS:
            stem = tmp_path / f"{state}_{n}"
            subprocess.run([PDFTOPPM, "-r", "100", "-png", "-singlefile", str(built[state][0] / f"{n}.pdf"), str(stem)], check=True)
            img = np.asarray(Image.open(f"{stem}.png").convert("L"), dtype=float) / 255
            ring = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
            assert (ring < 0.9).sum() == 0, f"{state} {n}: ink on the page edge (text or a mark is cut off)"


def test_the_test_a_failure_counts_in_fig4_do_not_overlap_each_other(built):
    from es.paper_figures import blocks_frame, fig4_blocks
    act, e5 = blocks_frame(built["sealed"][1])
    fig = fig4_blocks(act, e5, True)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(r) for t in fig.axes[2].texts]
    plt.close(fig)
    assert len(boxes) == 6
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert not a.overlaps(b), "two failure counts overlap"


def test_the_failure_counts_stay_inside_the_axes_when_the_intervals_are_high(built):
    """The rotated count sits above its interval; with an interval near the top of the shared y range the limit must grow."""
    from es.paper_figures import blocks_frame, fig4_blocks
    act, e5 = blocks_frame(built["sealed"][1])
    fig = fig4_blocks(act, e5.assign(median_log_error=0.45, lo=0.3, hi=0.62), True)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    top = fig.axes[2].get_window_extent(r).y1
    heights = [t.get_window_extent(r).y1 for t in fig.axes[2].texts]
    plt.close(fig)
    assert len(heights) == 6 and max(heights) <= top + 0.5, (max(heights), top)


def test_a_missing_column_is_a_lookup_error_that_names_the_file_and_the_column(tmp_path):
    from es.paper_figures import build_all
    src = _sources(tmp_path)
    p = src.out_dir / "analogs" / "activity_all.csv"
    pd.read_csv(p).drop(columns="policy").to_csv(p, index=False)
    with pytest.raises(LookupError, match=r"activity_all.*policy"):
        build_all(src, tmp_path / "figs")


def test_arial_missing_stops_the_build_instead_of_substituting_a_font(tmp_path, monkeypatch):
    from matplotlib import font_manager
    from es.paper_figures import build_all

    def no_arial(*args, **kwargs):
        raise ValueError("Failed to find font Arial, and fallback to the default font was disabled")

    monkeypatch.setattr(font_manager, "findfont", no_arial)
    with pytest.raises(RuntimeError, match="Arial"):
        build_all(_sources(tmp_path), tmp_path / "figs")
    assert not (tmp_path / "figs" / "Fig1.pdf").exists()


def test_the_ex_caiso_twin_of_the_pooled_rehearsal_does_not_replace_it(tmp_path):
    """Task 12 appends '<analysis>|ex-CAISO' rows; the pooled one starts with 'AQ3+AQ4' too. Fig2 and Fig3 show the full panel."""
    from es.paper_figures import _ci_rows, e1_frame
    src = _sources(tmp_path)
    p = src.out_dir / "economics" / "bootstrap_ci.csv"
    ci = pd.read_csv(p)
    pooled = ci[ci.analysis == "AQ3"].assign(analysis="AQ3+AQ4 (rehearsal of the pooled estimands)")
    twin = pooled.assign(analysis="AQ3+AQ4 (rehearsal of the pooled estimands)|ex-CAISO", estimate=99.0)
    ex = ci[ci.analysis == "AQ3"].assign(analysis="AQ3|ex-CAISO", estimate=77.0)
    pd.concat([twin, ex, ci, pooled], ignore_index=True).to_csv(p, index=False)           # the twin comes first on purpose
    frame = e1_frame(src)
    est, _, _ = _ci_rows(frame, ["AQ3", "POOL"], "SELC")
    assert list(est) == [10.0, 10.0]
    assert not frame.analysis.astype(str).str.contains("ex-CAISO").any()


def test_two_rows_for_one_analysis_and_statistic_are_refused(tmp_path):
    from es.paper_figures import _ci_rows
    ci = pd.DataFrame([{"analysis": "AQ3", "stat": "SELC", "estimate": e, "lo": 0.0, "hi": 1.0} for e in (1.0, 2.0)])
    with pytest.raises(LookupError, match="AQ3.*SELC"):
        _ci_rows(ci, ["AQ3"], "SELC")


def test_each_figure_function_and_save_apply_the_style_themselves(tmp_path, built):
    """A caller that does not go through build_all (a notebook, a later task) must still get embedded Arial, not Type 3."""
    from es.paper_figures import e1_frame, fig3_decomposition, save
    src = built["pending"][1]
    with plt.rc_context(matplotlib.rcParamsDefault):
        fig = fig3_decomposition(e1_frame(src), False)
        pdf, eps = save(fig, "Fig3", tmp_path)
    if Path(PDFFONTS).exists():
        fonts = subprocess.run([PDFFONTS, str(pdf)], capture_output=True, text=True).stdout
        assert "Arial" in fonts and "DejaVu" not in fonts and "Type 3" not in fonts, fonts
    assert b"/FontName /ArialMT" in eps.read_bytes()


def test_an_interval_with_lo_above_hi_is_refused(tmp_path):
    from es.paper_figures import _ci_rows
    ci = pd.DataFrame([{"analysis": "AQ3", "stat": "SELC", "estimate": 1.0, "lo": 2.0, "hi": 1.0}])
    with pytest.raises(LookupError, match="AQ3.*SELC.*lo"):
        _ci_rows(ci, ["AQ3"], "SELC")


def test_the_figure_functions_and_save_leave_the_callers_rcparams_alone(tmp_path, built):
    from es.paper_figures import STYLE, e1_frame, fig3_decomposition, save
    before = {k: matplotlib.rcParams[k] for k in STYLE}
    save(fig3_decomposition(e1_frame(built["pending"][1]), False), "Fig3", tmp_path)
    assert {k: matplotlib.rcParams[k] for k in STYLE} == before


def test_only_the_primary_pipeline_rows_reach_the_figures(tmp_path):
    """The real activity_all.csv holds every block, both estimators and four policies, and the sealed audit_activity.csv
    adds frozen maintenance and other seeds. Rows outside xgb / P1 / the six blocks (/ rolling, seed 42) must not be drawn."""
    from es.paper_figures import BLOCK_ORDER, blocks_frame, r_drift_frame
    src = _sources(tmp_path)
    p = src.out_dir / "analogs" / "activity_all.csv"
    act = pd.read_csv(p)
    stray = pd.concat([act.assign(policy="P6", R=50.0, log_error=9.0), act.assign(estimator="logit", R=50.0, log_error=9.0),
                       act.assign(block="MID3", R=50.0, log_error=9.0), act.assign(analysis="AQ1", R=50.0, log_error=9.0)])
    pd.concat([act, stray], ignore_index=True).to_csv(p, index=False)
    df = r_drift_frame(src)
    assert df.R.max() < 10 and set(df.block) == set(BLOCK_ORDER) and "AQ1" not in set(df.analysis)
    assert len(df[df.analysis == "AQ3"]) == 6 * 33
    a, _ = blocks_frame(src)                              # fig4_blocks picks AQ3 and AQ4 from it, so the stray AQ1 rows are inert there
    assert set(a.block) == set(BLOCK_ORDER) and set(a.policy) == {"P1"} and set(a.estimator) == {"xgb"}
    assert a[a.analysis.isin(["AQ3", "AQ4"])].log_error.max() < 5
    _add_test_a(src)
    s = src.out_dir / "sealed" / "A" / "tables" / "audit_activity.csv"
    audit = pd.read_csv(s)
    pd.concat([audit, audit.assign(maintenance="frozen", R=50.0), audit.assign(seed=7, R=50.0), audit.assign(policy="P4_56", R=50.0)],
              ignore_index=True).to_csv(s, index=False)
    sa = r_drift_frame(src)
    sa = sa[sa.analysis == "SA"]
    assert len(sa) == 6 * 33 and sa.R.max() < 10


def test_main_builds_into_paper_generated_figures_and_prints_each_status(tmp_path, capsys):
    from es.paper_figures import main
    src = _sources(tmp_path)
    assert main(["--out-dir", str(src.out_dir), "--paper-dir", str(src.paper_dir)]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed == [f"Fig{i}: built" for i in (1, 2, 3, 4)] + [f"{n}: built" for n in NEW_FIGS]
    assert (src.paper_dir / "generated" / "figures" / "Fig4.eps").exists()
    (src.out_dir / "economics" / "bootstrap_ci.csv").unlink()
    assert main(["--out-dir", str(src.out_dir), "--paper-dir", str(src.paper_dir)]) == 0      # a skip is not a failure: build.sh all goes on
    out = capsys.readouterr().out
    assert "Fig2: skipped: missing economics/bootstrap_ci.csv" in out and "FigContrast: skipped: missing economics/bootstrap_ci.csv" in out


# ---------------------------------------------------------------------------------------------------------------------
# Fix round 1 (review-task-5.md, controller Ruling 85) and controller Ruling 81 (paired E4 contrasts in Fig2).
# ---------------------------------------------------------------------------------------------------------------------
def _nodes():
    return [(m, f"{m}{i}") for m, k in (("NYISO", 10), ("ISONE", 9), ("MISO", 8), ("CAISO", 6)) for i in range(k)]


def _activity_fixture(src, n_hours=600, alpha=0.05, zero=("AQ3", "LEV6", "NYISO0")):
    """analogs/activity_all.csv as es.policies.activity writes it (R_floored, zero_activity, log_error, fail_0.5_2), with one
    node that never alerts: what an operating-point collapse looks like. Call before anything reads the file."""
    from es.policies import activity
    rng = np.random.default_rng(5)
    parts = []
    for a in ("AQ3", "AQ4", "AQ2"):
        for b in BLOCKS:
            recs = []
            for m, node in _nodes():
                p = 0.0 if (a, b, node) == zero else alpha * rng.lognormal(0, 0.3)
                recs.append(pd.DataFrame({"market": m, "node": node, "alert": rng.random(n_hours) < p}))
            parts.append(activity(pd.concat(recs, ignore_index=True), alpha).assign(analysis=a, block=b, estimator="xgb", policy="P1"))
    full = pd.concat(parts, ignore_index=True)
    full.to_csv(src.out_dir / "analogs" / "activity_all.csv", index=False)
    return full


def _scatters(ax):
    """The scatter collections of an axes (ax.collections also holds the LineCollection of the median bars)."""
    from matplotlib.collections import PathCollection
    return [c for c in ax.collections if isinstance(c, PathCollection) and len(c.get_offsets())]


def _is_open(c):
    return (np.asarray(c.get_facecolor())[:, :3] == 1).all()


def _edit_ci(src, edit):
    p = src.out_dir / "economics" / "bootstrap_ci.csv"
    ci = pd.read_csv(p)
    edit(ci)
    ci.to_csv(p, index=False)


# ---- I1: the floor of R -------------------------------------------------------------------------------------------------
def test_fig1_takes_log10_of_the_pipelines_floored_r(tmp_path):
    """es.policies.activity floors R at 1/(2 n alpha), which is what log_error (Fig4, E5) is built from; Fig1 used 1e-3 for every n."""
    from es.paper_figures import r_drift_frame
    src = _sources(tmp_path)
    full = _activity_fixture(src)
    z = full[(full.analysis == "AQ3") & (full.block == "LEV6") & (full.node == "NYISO0")].iloc[0]
    assert z.zero_activity and z.R == 0 and z.R_floored == pytest.approx(1 / (2 * 600 * 0.05))
    df = r_drift_frame(src)
    row = df[(df.analysis == "AQ3") & (df.block == "LEV6") & (df.node == "NYISO0")].iloc[0]
    assert row.log10R == pytest.approx(np.log10(z.R_floored)) and bool(row.zero_activity)           # not log10(1e-3) = -3
    assert int(df.zero_activity.sum()) == 1


def test_without_the_floored_column_fig1_falls_back_to_r(tmp_path):
    """The legacy fixture of the brief has only R: log10 of R, floored at 1e-3 so that log10 exists."""
    from es.paper_figures import r_drift_frame
    src = _sources(tmp_path)
    p = src.out_dir / "legacy" / "2024-2025__G1_24__xgb__s42__P1__a0.05.csv"
    d = pd.read_csv(p)
    d.loc[0, "R"] = 0.0
    d.to_csv(p, index=False)
    df = r_drift_frame(src)
    first = df[(df.analysis == "2024-2025") & (df.block == "G1_24")].iloc[0]
    assert first.log10R == pytest.approx(-3.0) and bool(first.zero_activity)
    assert df[~df.zero_activity].log10R.between(-1.5, 1.5).all()


def test_a_node_with_no_alerts_is_an_open_marker_and_never_stretches_the_shared_axis(tmp_path):
    from es.paper_figures import fig1_r_drift, r_drift_frame
    src = _sources(tmp_path)
    _activity_fixture(src)
    df = r_drift_frame(src)
    fig = fig1_r_drift(df, False)
    fig.canvas.draw()
    body_min = df[~df.zero_activity].log10R.min()
    ylim = fig.axes[0].get_ylim()
    assert ylim[0] > body_min - 0.6 and ylim[0] > -1.0, (ylim, body_min)         # the node's floor is -1.78; the 1e-3 floor gave -3.17
    ax = fig.axes[2]                                                              # AQ3
    opens = [c for c in _scatters(ax) if _is_open(c)]
    assert len(opens) == 1 and len(opens[0].get_offsets()) == 1
    x, y = opens[0].get_offsets()[0]
    assert round(x) == 2 and ylim[0] < y < body_min                               # LEV6 is block 2; the marker sits in the strip under the data
    assert any("no alerts" in t.get_text() for t in fig.legends[0].get_texts())
    plt.close(fig)


def test_without_a_node_that_never_alerts_there_is_no_open_marker_and_no_note(built):
    from es.paper_figures import fig1_r_drift, r_drift_frame
    fig = fig1_r_drift(r_drift_frame(built["pending"][1]), False)
    texts = [t.get_text() for t in fig.legends[0].get_texts()]
    opens = [c for ax in fig.axes for c in _scatters(ax) if _is_open(c)]
    plt.close(fig)
    assert not any("no alerts" in t for t in texts) and not opens


# ---- I2: Fig4 shows the tail --------------------------------------------------------------------------------------------
def test_fig4_whiskers_run_to_the_minimum_and_maximum(tmp_path):
    """With showfliers=False and whiskers at 1.5 IQR the failing nodes (the tail the figure is about) were not drawn."""
    from es.paper_figures import blocks_frame, fig4_blocks
    src = _sources(tmp_path)
    full = _activity_fixture(src)
    act, _ = blocks_frame(src)
    fig = fig4_blocks(act, None, False)
    top = max(float(np.max(line.get_ydata())) for line in fig.axes[0].lines)
    plt.close(fig)
    assert top == pytest.approx(full[full.analysis == "AQ3"].log_error.max())       # the node with no alerts: |log10 R_floored| = 1.78


def test_the_fig4_aq_panels_carry_the_failure_count_like_the_test_a_panel(tmp_path):
    from es.paper_figures import BLOCK_ORDER, blocks_frame, fig4_blocks
    src = _sources(tmp_path)
    full = _activity_fixture(src)
    act, _ = blocks_frame(src)
    fig = fig4_blocks(act, None, False)
    for ax, aid in zip(fig.axes[:2], ("AQ3", "AQ4")):
        want = [int(full[(full.analysis == aid) & (full.block == b)]["fail_0.5_2"].sum()) for b in BLOCK_ORDER]
        got = [t.get_text() for t in sorted(ax.texts, key=lambda t: t.xy[0])]
        assert got == [f"{n} fail" for n in want], (aid, got, want)
    plt.close(fig)


def test_fig4_needs_the_failure_column(tmp_path):
    from es.paper_figures import blocks_frame
    src = _sources(tmp_path)
    p = src.out_dir / "analogs" / "activity_all.csv"
    pd.read_csv(p).drop(columns="fail_0.5_2").to_csv(p, index=False)
    with pytest.raises(LookupError, match=r"activity_all.*fail_0.5_2"):
        blocks_frame(src)


# ---- I3: a missing statistic is an error ---------------------------------------------------------------------------------
def test_a_statistic_the_figure_draws_but_the_file_lacks_is_an_error_not_a_silent_gap(tmp_path):
    """A bootstrap_ci.csv from before Ruling 68 has E4:SELC_month where Fig2 draws E4:SELC_decile; the legend promised it."""
    from es.paper_figures import build_all
    src = _sources(tmp_path)
    _edit_ci(src, lambda ci: ci.__setitem__("stat", ci["stat"].replace({"E4:SELC_decile": "E4:SELC_month"})))
    with pytest.raises(LookupError, match=r"E4:SELC_decile"):
        build_all(src, tmp_path / "figs")


def test_a_nan_estimate_is_an_error(tmp_path):
    from es.paper_figures import build_all
    src = _sources(tmp_path)
    _edit_ci(src, lambda ci: ci.__setitem__("estimate", ci["estimate"].mask((ci.analysis == "AQ4") & (ci.stat == "E1:J_0.5:level"))))
    with pytest.raises(LookupError, match=r"AQ4.*E1:J_0.5:level"):
        build_all(src, tmp_path / "figs")


def test_only_the_test_a_row_may_be_empty_and_only_while_the_test_is_pending():
    from es.paper_figures import _ci_rows
    ci = pd.DataFrame([{"analysis": "AQ3", "stat": "SELC", "estimate": 1.0, "lo": 0.0, "hi": 2.0}])
    est, _, _ = _ci_rows(ci, ["AQ3", "SA"], "SELC")                      # pending: the Test A row is not there yet
    assert est[0] == 1.0 and np.isnan(est[1])
    with pytest.raises(LookupError, match="SA"):
        _ci_rows(ci, ["AQ3", "SA"], "SELC", sealed=True)                 # Test A has run: its row is required like any other
    with pytest.raises(LookupError, match="AQ4"):
        _ci_rows(ci, ["AQ3", "AQ4"], "SELC")


# ---- I4: Test A content reaches the figures ----------------------------------------------------------------------------
def _row_lines(ax, row):
    return [line for line in ax.lines if np.any(np.abs(np.asarray(line.get_ydata(), float) - row) < 0.45)]


def test_test_a_content_reaches_all_four_figures_and_status_follows_what_was_drawn(built):
    from es.paper_figures import _ci_rows, blocks_frame, e1_frame, fig1_r_drift, fig2_selc_esred, fig3_decomposition, fig4_blocks, r_drift_frame
    for state, sealed in (("pending", False), ("sealed", True)):
        src = built[state][1]
        ci = e1_frame(src)
        act, e5 = blocks_frame(src)
        f1, f2 = fig1_r_drift(r_drift_frame(src), sealed), fig2_selc_esred(ci, sealed)
        f3, f4 = fig3_decomposition(ci, sealed), fig4_blocks(act, e5, sealed)
        sa = 4                                                           # AQ3, AQ4, AQ1, AQ2, then the Test A row
        have = [bool(_row_lines(f2.axes[0], sa)) and bool(_row_lines(f2.axes[1], sa)),
                all(any(abs(p.get_y() + p.get_height() / 2 - sa) < 0.45 for p in ax.patches) for ax in f3.axes)]
        if sealed:
            offs = [c.get_offsets() for c in _scatters(f1.axes[5])]
            blocks_drawn = set(np.round(np.vstack(offs)[:, 0]).astype(int)) if offs else set()
            have += [blocks_drawn == set(range(6)) and not f1.axes[5].texts, len(f4.axes[2].lines) > 0 and len(f4.axes[2].texts) == 6]
        else:                                                            # no Test A panel at all: five panels in Fig1, two in Fig4
            have += [len(f1.axes) > 5, len(f4.axes) > 2]
        assert have == [sealed] * 4, (state, have)
        assert [f.test_a_drawn for f in (f1, f2, f3, f4)] == [sealed] * 4
        if sealed:
            est, lo, hi = _ci_rows(ci, ["SA"], "SELC")
            assert np.isfinite([est[0], lo[0], hi[0]]).all()
        for f in (f1, f2, f3, f4):
            plt.close(f)
    assert {v for v in built["sealed"][2].values()} == {"built"}
    assert {v for v in built["pending"][2].values()} == {"built"}          # (T1b) the three new figures have no Test A to wait for, and the other four print none of it
    assert {built["pending"][2][n] for n in FIGS} == {"built"}


def test_sealed_rows_that_never_reach_a_figure_stop_the_build(tmp_path, monkeypatch):
    """The mutation of the review: Test A tables exist, yet the frames drop the Test A rows. 'built' came from file existence."""
    import es.paper_figures as pf
    for name, frame in (("e1_frame", "SA"), ("r_drift_frame", "SA")):
        src = _sources(tmp_path / name)
        _add_test_a(src)
        original = getattr(pf, name)
        monkeypatch.setattr(pf, name, lambda s, original=original: original(s)[lambda d: d.analysis != "SA"])
        with pytest.raises(LookupError, match="SA|Test A"):
            pf.build_all(src, tmp_path / name / "figs")
        monkeypatch.setattr(pf, name, original)


# ---- Ruling 81: Fig2 draws the paired contrasts, the comparator levels become open markers ------------------------------
def test_fig2_draws_the_paired_contrasts_with_intervals_and_the_comparator_levels_as_open_markers(built):
    from es.paper_figures import e1_frame, fig2_selc_esred
    fig = fig2_selc_esred(e1_frame(built["pending"][1]), False)
    ax = fig.axes[0]
    assert [t.get_text() for t in fig.legends[0].get_texts()] == [
        "alerts (P1)", "P1 − calendar-only (paired)", "P1 − VOL (paired)", "alerts, decile strata",
        "calendar-only (level)", "VOL (level)"]
    marked = [line for line in ax.lines if line.get_marker() in ("o", "s", "^", "D")]
    filled = sorted(line.get_marker() for line in marked if line.get_markerfacecolor() != "white")
    opened = sorted(line.get_marker() for line in marked if line.get_markerfacecolor() == "white")
    assert filled == ["D", "^", "o", "s"] and opened == ["^", "s"], (filled, opened)
    assert len(ax.collections) == 4                                       # interval bars: P1, the two contrasts, the decile strata; none for the levels
    for line in marked:
        assert len(line.get_xdata()) == 4                                  # AQ3, AQ4, AQ1, AQ2 (Test A is pending)
    plt.close(fig)


def test_fig2_needs_the_paired_contrasts(tmp_path):
    """A bootstrap_ci.csv from before Ruling 81 has no E4:dSELC rows."""
    from es.paper_figures import build_all
    src = _sources(tmp_path)
    _edit_ci(src, lambda ci: ci.drop(ci[ci.stat.str.startswith("E4:dSELC")].index, inplace=True))
    with pytest.raises(LookupError, match=r"E4:dSELC\|calendar@count"):
        build_all(src, tmp_path / "figs")


# ---- M1 .. M9 ------------------------------------------------------------------------------------------------------------
def test_money_axes_have_no_offset_text_even_at_millions(tmp_path):
    """Matplotlib puts '1e6' at the axis end for |x| >= 1e6; it collided with the x label of the J panel."""
    from es.paper_figures import e1_frame, fig2_selc_esred, fig3_decomposition
    src = _sources(tmp_path)

    def millions(ci):
        ci.loc[ci.stat.str.contains("NET|ESred|J_"), ["estimate", "lo", "hi"]] = [3.0e6, 2.0e6, 4.0e6]

    _edit_ci(src, millions)
    for fig in (fig3_decomposition(e1_frame(src), False), fig2_selc_esred(e1_frame(src), False)):
        fig.canvas.draw()
        offsets = [ax.xaxis.get_offset_text().get_text() for ax in fig.axes[-3:] if ax.get_visible()]
        plt.close(fig)
        assert not any(offsets), offsets


def test_the_approximate_note_follows_the_approximate_column(tmp_path):
    from es.paper_figures import e1_frame, fig2_selc_esred, fig3_decomposition
    src = _sources(tmp_path)
    f2, f3 = fig2_selc_esred(e1_frame(src), False), fig3_decomposition(e1_frame(src), False)
    assert not any("approximate" in ax.get_xlabel() for ax in f2.axes + f3.axes)          # the brief's fixture flags nothing
    plt.close(f2)
    plt.close(f3)
    _edit_ci(src, lambda ci: ci.__setitem__("approximate", ci.stat.str.contains("ESred|E1:J_")))   # what es.run_economics._approximate flags
    src._frames.clear()
    f2, f3 = fig2_selc_esred(e1_frame(src), False), fig3_decomposition(e1_frame(src), False)
    assert ["approximate" in ax.get_xlabel() for ax in f2.axes] == [False, True]
    assert ["approximate" in ax.get_xlabel() for ax in f3.axes] == [False, True, True]       # NET, ESred, J
    plt.close(f2)
    plt.close(f3)


def test_the_pooled_row_says_sum_of_two_quarters_where_it_is_one(tmp_path):
    from es.paper_figures import e1_frame, fig2_selc_esred, fig3_decomposition
    src = _sources(tmp_path)
    f3 = fig3_decomposition(e1_frame(src), False)
    assert not any("sum of two quarters" in t.get_text() for t in f3.texts)               # no pooled row, no note
    plt.close(f3)
    ci = pd.read_csv(src.out_dir / "economics" / "bootstrap_ci.csv")
    pooled = ci[ci.analysis == "AQ3"].assign(analysis="AQ3+AQ4 (rehearsal of the pooled estimands)")
    pd.concat([ci, pooled], ignore_index=True).to_csv(src.out_dir / "economics" / "bootstrap_ci.csv", index=False)
    src._frames.clear()
    f3, f2 = fig3_decomposition(e1_frame(src), False), fig2_selc_esred(e1_frame(src), False)
    notes = [t.get_text() for t in f3.texts if "sum of two quarters" in t.get_text()]
    assert len(notes) == 1 and "NET and J" in notes[0] and "ESred" not in notes[0]
    assert not any("sum of two quarters" in t.get_text() for t in f2.texts)               # SELC is a ratio of sums, ESred a mean
    plt.close(f3)
    plt.close(f2)


def test_the_lambda_label_and_the_failure_band_come_from_the_code(tmp_path, monkeypatch):
    import es.paper_figures as pf
    src = _sources(tmp_path)
    f1 = pf.fig1_r_drift(pf.r_drift_frame(src), False)
    legend = [t.get_text() for t in f1.legends[0].get_texts()]
    assert "R within [0.5, 2]" in legend and "R = 1" in legend                              # the band and the zero line have entries
    plt.close(f1)
    assert "0.5" in pf.fig3_decomposition(pf.e1_frame(src), False).axes[2].get_xlabel()
    _edit_ci(src, lambda ci: ci.__setitem__("stat", ci["stat"].str.replace("J_0.5", "J_0.25")))
    src._frames.clear()
    monkeypatch.setattr(pf, "J_LAMBDA", 0.25)
    monkeypatch.setattr(pf, "J_KEY", "J_0.25")
    monkeypatch.setattr(pf, "BANDS", {"0.5_2": (2 / 3, 1.5)})
    f3 = pf.fig3_decomposition(pf.e1_frame(src), False)
    assert "0.25" in f3.axes[2].get_xlabel()
    f1 = pf.fig1_r_drift(pf.r_drift_frame(src), False)
    band = f1.axes[0].patches[0]
    assert band.get_y() == pytest.approx(np.log10(2 / 3)) and band.get_y() + band.get_height() == pytest.approx(np.log10(1.5))
    assert "R within [0.67, 1.5]" in [t.get_text() for t in f1.legends[0].get_texts()]
    plt.close(f3)
    plt.close(f1)


def test_a_skipped_figure_leaves_no_stale_file_behind(tmp_path):
    from es.paper_figures import build_all
    src = _sources(tmp_path)
    figdir = tmp_path / "paper" / "generated" / "figures"
    build_all(src, figdir)
    assert (figdir / "Fig1.pdf").exists() and (figdir / "Fig4.eps").exists()
    (src.out_dir / "analogs" / "activity_all.csv").unlink()
    status = build_all(src, figdir)
    assert status["Fig1"].startswith("skipped") and status["Fig4"].startswith("skipped")
    assert not any((figdir / f"{n}.{e}").exists() for n in ("Fig1", "Fig4") for e in ("pdf", "eps"))
    assert (figdir / "Fig2.pdf").exists() and (figdir / "Fig3.eps").exists()


def test_a_test_a_that_failed_its_coverage_rule_is_said_to_have_not_run(tmp_path):
    from es.paper_figures import blocks_frame, build_all, e1_frame, fig1_r_drift, fig2_selc_esred, fig3_decomposition, fig4_blocks, r_drift_frame
    src = _sources(tmp_path)
    (src.out_dir / "sealed" / "A").mkdir(parents=True)
    (src.out_dir / "sealed" / "A" / "results.json").write_text('{"coverage": {"ran": false, "usable_nodes": 12}}')
    figdir = tmp_path / "figs"
    status = build_all(src, figdir)
    assert {status[n] for n in FIGS} == {"built (Test A did not run)"} and [status[n] for n in NEW_FIGS] == ["built"] * 3, status
    assert "pending" not in (figdir / "INDEX.md").read_text() and "did not run" in (figdir / "INDEX.md").read_text()
    act, _ = blocks_frame(src)
    figs = [fig1_r_drift(r_drift_frame(src), False, pending="did not run"), fig2_selc_esred(e1_frame(src), False, pending="did not run"),
            fig3_decomposition(e1_frame(src), False, pending="did not run"), fig4_blocks(act, None, False, pending="did not run")]
    texts = [t.get_text() for f in figs for ax in f.axes for t in ax.texts + ax.get_yticklabels()]
    for f in figs:
        plt.close(f)
    assert "Test A: did not run" in texts and "Test A: 2026-Q3: did not run" in texts and not any("pending" in t for t in texts)


def test_fig1_markers_are_large_enough_to_tell_apart_in_greyscale(built):
    from es.paper_figures import fig1_r_drift, r_drift_frame
    fig = fig1_r_drift(r_drift_frame(built["pending"][1]), False)
    sizes = [float(c.get_sizes()[0]) for c in _scatters(fig.axes[0])]
    legend_ms = [h.get_markersize() for h in fig.legends[0].legend_handles[:4]]
    plt.close(fig)
    assert min(sizes) >= 16 and legend_ms == pytest.approx([np.sqrt(sizes[0])] * 4)        # 4 pt across, the legend draws the same size


def test_log10_is_a_9_pt_log_and_a_lowered_8_pt_10_in_separate_text_objects(built):
    from matplotlib.text import Text
    from es.paper_figures import blocks_frame, fig1_r_drift, fig4_blocks, r_drift_frame
    src = built["pending"][1]
    act, _ = blocks_frame(src)
    for fig, columns in ((fig1_r_drift(r_drift_frame(src), False), 2), (fig4_blocks(act, None, False), 1)):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        shown = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
        logs = [t for t in shown if t.get_text().endswith("log")]
        subs = [t for t in shown if t.get_text() == "10"]
        assert len(logs) == len(subs) == columns and {t.get_fontsize() for t in logs} == {9.0} and {t.get_fontsize() for t in subs} == {8.0}
        for lg, sub in zip(logs, subs):
            a, b = lg.get_window_extent(r), sub.get_window_extent(r)
            assert b.x1 > a.x1 + 1.0 and abs(b.y0 - a.y1) < 0.5            # lowered towards the axis (rotated text: to the right), stacked after "log"
        assert not any(ax.yaxis.label.get_visible() and ax.yaxis.label.get_text() for ax in fig.axes if ax.yaxis.label.get_text().startswith(("log", "|log")))
        plt.close(fig)


def test_the_index_lists_every_source_file_and_states_the_whisker_rule(built):
    for state in ("pending", "sealed"):
        index = (built[state][0] / "INDEX.md").read_text()
        for aid in ("2024-2025", "25H1-26H1"):
            for b in BLOCKS:
                assert f"legacy/{aid}__{b}__xgb__s42__P1__a0.05.csv" in index, (state, aid, b)
        assert "analogs/activity_all.csv" in index and "sealed/A/tables/audit_activity.csv" in index
        assert "whiskers" in index and "minimum and maximum" in index


def test_two_builds_are_byte_identical(built, tmp_path):
    """The PDF has no CreationDate (metadata) and the EPS has a fixed one (SOURCE_DATE_EPOCH): rebuilding changes no byte.
    time.ctime() has a one-second resolution, so wait that long or the test could pass with the dates still in."""
    import time

    from es.paper_figures import build_all
    time.sleep(1.1)
    for state in ("pending", "sealed"):
        figdir, src, _ = built[state]
        again = tmp_path / state
        build_all(src, again)
        for n in ALL_FIGS:
            for ext in ("pdf", "eps"):
                assert (figdir / f"{n}.{ext}").read_bytes() == (again / f"{n}.{ext}").read_bytes(), f"{state} {n}.{ext} differs"
        assert (figdir / "INDEX.md").read_bytes() == (again / "INDEX.md").read_bytes()
    assert b"CreationDate" not in (built["pending"][0] / "Fig1.pdf").read_bytes()


def test_a_long_legend_never_widens_the_saved_figure(tmp_path):
    """With a node that never alerts, Fig1's legend gets an eighth, long entry. Found by looking at the render: at four columns it was
    wider than the figure and the saved page grew to 388 bp, so inclusion at \\textwidth would have scaled the lettering below 8 pt."""
    from es.paper_figures import build_all
    src = _sources(tmp_path)
    _activity_fixture(src)
    figdir = tmp_path / "figs"
    status = build_all(src, figdir)
    assert status["Fig1"] == "built"
    media = re.search(rb"/MediaBox \[\s*0 0 ([0-9.]+) ([0-9.]+)\s*\]", (figdir / "Fig1.pdf").read_bytes())
    assert media and float(media.group(1)) <= TEXT_WIDTH_BP + 0.01, f"Fig1.pdf is {media.group(1)} bp wide"


def test_a_legend_wider_than_the_figure_drops_columns_until_it_fits():
    from matplotlib.lines import Line2D
    from es.paper_figures import STYLE, _legend_below
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(5.1, 2.0))
        handles = [Line2D([], [], marker="o", ls="", label=f"a rather long legend entry, number {i} of eight") for i in range(8)]
        _legend_below(fig, handles, 4)
        width, limit, n = fig.legends[0].get_window_extent(fig.canvas.get_renderer()).width, fig.bbox.width, len(fig.legends)
        plt.close(fig)
    assert n == 1 and width <= limit, (n, width, limit)


def test_a_contrast_marker_is_drawn_above_the_open_level_marker_on_the_same_line(built):
    """The open markers have a white fill. Where a comparator's level and the P1 contrast coincide, the estimate with its interval
    is what must stay visible, so the filled marker has the higher zorder."""
    from es.paper_figures import e1_frame, fig2_selc_esred
    fig = fig2_selc_esred(e1_frame(built["pending"][1]), False)
    squares = [line for line in fig.axes[0].lines if line.get_marker() == "s"]
    filled = [line.get_zorder() for line in squares if line.get_markerfacecolor() != "white"]
    opened = [line.get_zorder() for line in squares if line.get_markerfacecolor() == "white"]
    plt.close(fig)
    assert len(filled) == len(opened) == 1 and filled[0] > opened[0]


# ---------------------------------------------------------------------------------------------------------------------
# Plan 2C. Three figures that show one finding each (FigFail, FigDrift, FigContrast) replace the two of the main text, and no figure carries a
# Test A panel that is empty. The expected numbers below are worked out by hand from the fixtures above.
# ---------------------------------------------------------------------------------------------------------------------
PDFTOTEXT = shutil.which("pdftotext") or "/opt/homebrew/bin/pdftotext"
CORE_NAMES = ["calendar", "plain DART lags", "absolute level", "rolling dispersion", "DART-relative", "combined"]     # BLOCKS, top to bottom
_EPS_GLYPH = {"space": " ", "colon": ":", "period": ".", "comma": ",", "hyphen": "-", "slash": "/", "parenleft": "(", "parenright": ")",
              "percent": "%", "bracketleft": "[", "bracketright": "]", "arrowleft": "←", "arrowright": "→", "equal": "=", "plus": "+"}


def _eps_strings(path: Path) -> list[str]:
    """The text of an EPS that matplotlib wrote with Type 42 fonts: each string is a run of '<x> 0 m /<glyph> glyphshow' lines whose x starts again at 0.
    Letters are single-letter glyph names; the other glyphs are mapped for the ones the figures use, and any glyph name left over is '?'."""
    strings = []
    for line in path.read_text(errors="replace").splitlines():
        m = re.fullmatch(r"(-?[0-9.]+) 0 m /(\S+) glyphshow", line)
        if not m:
            continue
        glyph = m.group(2)
        char = glyph if len(glyph) == 1 else _EPS_GLYPH.get(glyph, "?")
        if float(m.group(1)) == 0.0 or not strings:
            strings.append("")
        strings[-1] += char
    return strings


def _texts(fig) -> list[str]:
    """Every visible text of a figure: titles, tick labels, annotations, legend entries and the figure's own texts."""
    from matplotlib.text import Text
    fig.canvas.draw()
    return [t.get_text() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]


def _cell_texts(ax):
    """The texts printed in the cells of a heatmap axes ({(column, row): text}); the group headings are Annotations and are left out."""
    from matplotlib.text import Text
    return {(int(t.get_position()[0]), int(t.get_position()[1])): t for t in ax.texts if type(t) is Text}


def _luminance(rgba):
    r, g, b = rgba[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


# ---- no panel, row or word "pending" while Test A has not run ----------------------------------------------------------------
def test_no_figure_prints_the_word_pending_while_test_a_is_absent(built):
    figdir = built["pending"][0]
    for n in ALL_FIGS:
        strings = _eps_strings(figdir / f"{n}.eps")
        assert len(strings) > 5, f"{n}.eps: the text of the EPS was not read ({len(strings)} strings)"
        assert not [s for s in strings if "pending" in s], f"{n}.eps prints the word pending"
        if Path(PDFTOTEXT).exists():
            text = subprocess.run([PDFTOTEXT, str(figdir / f"{n}.pdf"), "-"], capture_output=True, text=True).stdout
            assert text.strip() and "pending" not in text, f"{n}.pdf prints the word pending"


def test_fig1_draws_five_panels_while_test_a_has_not_run_and_six_when_it_has_run_or_did_not(built):
    from es.paper_figures import fig1_r_drift, r_drift_frame
    pending = fig1_r_drift(r_drift_frame(built["pending"][1]), False)
    rows = sorted({round(ax.get_position().y0, 3) for ax in pending.axes}, reverse=True)
    per_row = [sum(1 for ax in pending.axes if round(ax.get_position().y0, 3) == y) for y in rows]
    assert len(pending.axes) == 5 and per_row == [3, 2], per_row                 # the sixth cell of the 2 x 3 grid is gone
    assert not [s for s in _texts(pending) if "pending" in s or "Test A" in s]
    assert not pending.test_a_drawn
    did_not_run = fig1_r_drift(r_drift_frame(built["pending"][1]), False, pending="did not run")
    assert len(did_not_run.axes) == 6 and "Test A: did not run" in _texts(did_not_run)
    ran = fig1_r_drift(r_drift_frame(built["sealed"][1]), True)
    assert len(ran.axes) == 6 and ran.test_a_drawn and not [s for s in _texts(ran) if "pending" in s]
    for f in (pending, did_not_run, ran):
        plt.close(f)


def test_fig2_and_fig3_have_no_test_a_row_while_it_has_not_run(built):
    from es.paper_figures import e1_frame, fig2_selc_esred, fig3_decomposition
    ci = e1_frame(built["pending"][1])
    for make in (fig2_selc_esred, fig3_decomposition):
        pending = make(ci, False)
        labels = [t.get_text() for t in pending.axes[0].get_yticklabels()]
        assert labels == ["AQ3: 2025-Q3", "AQ4: 2025-Q4", "AQ1: 2026-Q1", "AQ2: 2026-Q2"], (make.__name__, labels)
        assert not [s for s in _texts(pending) if "pending" in s or "Test A" in s]
        did_not_run = make(ci, False, pending="did not run")
        assert [t.get_text() for t in did_not_run.axes[0].get_yticklabels()][-1] == "Test A: 2026-Q3: did not run"
        ran = make(e1_frame(built["sealed"][1]), True)
        assert [t.get_text() for t in ran.axes[0].get_yticklabels()][-1] == "Test A: 2026-Q3"
        for f in (pending, did_not_run, ran):
            plt.close(f)


def test_fig4_has_two_panels_while_test_a_has_not_run_and_three_when_it_has_run_or_did_not(built):
    from es.paper_figures import blocks_frame, fig4_blocks
    act, _ = blocks_frame(built["pending"][1])
    pending = fig4_blocks(act, None, False)
    assert len(pending.axes) == 2 and not pending.test_a_drawn
    assert not [s for s in _texts(pending) if "pending" in s or "Test A" in s]
    did_not_run = fig4_blocks(act, None, False, pending="did not run")
    assert len(did_not_run.axes) == 3 and "Test A: did not run" in _texts(did_not_run)
    act, e5 = blocks_frame(built["sealed"][1])
    ran = fig4_blocks(act, e5, True)
    assert len(ran.axes) == 3 and ran.test_a_drawn
    for f in (pending, did_not_run, ran):
        plt.close(f)


def test_the_two_panels_of_fig4_without_test_a_stay_inside_the_text_width(built):
    media = re.search(rb"/MediaBox \[\s*0 0 ([0-9.]+) ([0-9.]+)\s*\]", (built["pending"][0] / "Fig4.pdf").read_bytes())
    assert media and float(media.group(1)) <= TEXT_WIDTH_BP + 0.01


# ---- FigFail: the heatmap of operating-point failures ---------------------------------------------------------------------------
def _fail_axes(fig):
    """{panel: {(column, row): printed count}} of a FigFail figure, read from the texts in the cells."""
    return [{pos: t.get_text() for pos, t in _cell_texts(ax).items()} for ax in fig.axes]


def test_fail_frame_is_the_cells_of_table_3_and_reads_no_other_row(tmp_path):
    from es.paper_figures import fail_frame
    cells = fail_frame(_sources(tmp_path))
    assert len(cells) == 5 * 6 and set(cells.n_nodes) == {33}
    for (analysis, est), counts in FAIL_FIXTURE.items():                      # the rows of another seed, policy, target or block are 99, 98, 97, 96
        got = cells[(cells.analysis == analysis) & (cells.estimator == est)].set_index("block").failures.reindex(BLOCKS)
        assert list(got) == counts, (analysis, est)


def test_a_cell_of_table_3_that_the_summary_lacks_is_an_error_that_names_it(tmp_path):
    from es.paper_figures import fail_frame
    src = _sources(tmp_path)
    s = pd.read_csv(src.out_dir / "legacy" / "summary.csv")
    s[~((s.analysis == "frozen-26H1") & (s.block == "LEV6") & (s.seed == 42) & (s.policy == "P1") & (s.alpha == 0.05))].to_csv(src.out_dir / "legacy" / "summary.csv", index=False)
    src._frames.clear()
    with pytest.raises(LookupError, match=r"frozen-26H1.*LEV6|LEV6.*frozen-26H1"):
        fail_frame(src)


def test_fig_fail_prints_each_count_in_its_cell_with_the_rows_of_the_brief_top_to_bottom(tmp_path):
    from es.paper_figures import fail_frame, fig_fail
    fig = fig_fail(fail_frame(_sources(tmp_path)))
    left, right = _fail_axes(fig)
    columns_left = [("2024-2025", "xgb"), ("25H1-26H1", "xgb")]
    columns_right = [("2024-2025", "logit"), ("25H1-26H1", "logit"), ("frozen-26H1", "xgb")]
    for printed, columns in ((left, columns_left), (right, columns_right)):
        assert len(printed) == 6 * len(columns)
        for c, column in enumerate(columns):
            for r in range(6):
                assert printed[(c, r)] == str(FAIL_FIXTURE[column][r]), (column, CORE_NAMES[r])
    fig.canvas.draw()
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == CORE_NAMES
    ys = [t.get_position()[1] for t in fig.axes[0].get_yticklabels()]
    assert ys == sorted(ys)                                                   # calendar at the top: the axis is inverted, so the first row is the smallest y
    assert fig.axes[0].get_ylim()[0] > fig.axes[0].get_ylim()[1]
    plt.close(fig)


def test_fig_fail_has_the_two_panels_and_the_columns_of_the_brief(tmp_path):
    from es.paper_figures import fail_frame, fig_fail
    fig = fig_fail(fail_frame(_sources(tmp_path)))
    texts = _texts(fig)
    assert len(fig.axes) == 2
    assert "XGBoost, rolling origin" in [ax.get_title(loc="left") for ax in fig.axes] and "robustness" in [ax.get_title(loc="left") for ax in fig.axes]
    for needed in ("logistic regression", "2024 to\n2025", "2025-H1 to\n2026-H1", "2026-H1"):
        assert needed in texts, (needed, texts)
    assert any(s.startswith("XGBoost,\nfrozen origin") for s in texts)
    plt.close(fig)


def test_cells_of_five_or_more_nodes_are_bold_with_a_heavy_outline_and_the_others_are_not(tmp_path):
    """The figure must read in greyscale: bold and the outline do not depend on the colour. 5 nodes (rolling dispersion, 2024 to 2025, XGBoost) is heavy,
    4 nodes (rolling dispersion, 2025-H1 to 2026-H1, logistic regression) is not."""
    from es.paper_figures import fail_frame, fig_fail
    fig = fig_fail(fail_frame(_sources(tmp_path)))
    for ax, columns in zip(fig.axes, ([("2024-2025", "xgb"), ("25H1-26H1", "xgb")], [("2024-2025", "logit"), ("25H1-26H1", "logit"), ("frozen-26H1", "xgb")])):
        weights = {pos: t.get_fontweight() for pos, t in _cell_texts(ax).items()}
        outlines = {(int(p.get_x()), int(p.get_y())): p.get_linewidth() for p in ax.patches}
        for c, column in enumerate(columns):
            for r in range(6):
                heavy = FAIL_FIXTURE[column][r] >= 5
                assert (weights[(c, r)] == "bold") == heavy, (column, r)
                assert (outlines[(c, r)] >= 1.2) == heavy, (column, r, outlines[(c, r)])
                assert (outlines[(c, r)] <= 1.0) == (not heavy)
    plt.close(fig)


def test_the_fill_grows_with_the_count_on_one_scale_shared_by_both_panels(tmp_path):
    from es.paper_figures import fail_frame, fig_fail
    fig = fig_fail(fail_frame(_sources(tmp_path)))
    fills = []
    for ax in fig.axes:
        fills += [(int(_cell_texts(ax)[(int(p.get_x()), int(p.get_y()))].get_text()), _luminance(p.get_facecolor())) for p in ax.patches]
    by_count = {}
    for count, lum in fills:
        by_count.setdefault(count, set()).add(round(lum, 6))
    assert all(len(v) == 1 for v in by_count.values()), by_count               # the same count has the same fill in both panels (13 and 12 occur in both)
    counts = sorted(by_count)
    lums = [next(iter(by_count[c])) for c in counts]
    assert lums == sorted(lums, reverse=True) and len(set(lums)) == len(lums), list(zip(counts, lums))        # more nodes, darker, strictly
    assert max(lums) == pytest.approx(1.0)                                     # a count of zero is white: the scale starts at zero
    plt.close(fig)


def test_the_text_colour_follows_the_fill_so_that_every_count_can_be_read(tmp_path):
    from matplotlib.colors import to_rgb
    from es.paper_figures import fail_frame, fig_fail
    fig = fig_fail(fail_frame(_sources(tmp_path)))
    for ax in fig.axes:
        texts = _cell_texts(ax)
        for p in ax.patches:
            fill, ink = _luminance(p.get_facecolor()), _luminance(to_rgb(texts[(int(p.get_x()), int(p.get_y()))].get_color()))
            assert abs(fill - ink) >= 0.45, (fill, ink)                         # black on light cells, white on dark ones
    plt.close(fig)


def test_the_number_of_nodes_comes_from_the_data_and_is_never_typed(tmp_path):
    from es.paper_figures import fail_frame, fig_fail
    src = _sources(tmp_path)
    _write_summary(src.out_dir, n_nodes=20)
    src._frames.clear()
    fig = fig_fail(fail_frame(src))
    texts = _texts(fig)
    assert any("of 20" in s for s in texts) and not any("33" in s for s in texts), texts
    plt.close(fig)


def test_cells_with_different_numbers_of_nodes_print_the_count_over_their_own_denominator(tmp_path):
    from es.paper_figures import fail_frame, fig_fail
    src = _sources(tmp_path)
    s = pd.read_csv(src.out_dir / "legacy" / "summary.csv")
    s.loc[(s.analysis == "frozen-26H1") & (s.seed == 42) & (s.policy == "P1") & (s.alpha == 0.05), "n_nodes"] = 31
    s.to_csv(src.out_dir / "legacy" / "summary.csv", index=False)
    src._frames.clear()
    fig = fig_fail(fail_frame(src))
    left, right = _fail_axes(fig)
    assert left[(0, 2)] == "13/33" and right[(2, 5)] == "23/31", (left[(0, 2)], right[(2, 5)])
    plt.close(fig)


def test_the_note_says_what_a_failure_is_with_the_band_of_the_code_and_what_the_outline_means(tmp_path):
    from es.paper_figures import fail_frame, fig_fail
    fig = fig_fail(fail_frame(_sources(tmp_path)))
    text = " ".join(_texts(fig))
    assert "realised alert rate outside 0.5 to 2 times the target" in text and "5 or more" in text
    assert not any(w in text.lower() for w in ("better", "pending"))
    plt.close(fig)


# ---- FigDrift: the alert rate over the target, in trailing windows of eight weeks (T1b) ----------------------------------------
# The fixture (_write_drift) has five nodes N0 to N4 and eight Monday-start weeks w0 to w7 from Monday 6 January 2025 (6 January 2026 minus one day for the second panel:
# Monday 5 January 2026): w0 and w7 are partial (96 hours, four days), w1 to w6 have 168 hours. A node's alerts in week w are (3 k + w) % 17 under P1 and (k + w) % 17
# under P4, for node Nk. The windows of the figure end with weeks w3 to w7 (the first has four weeks) and, with eight weeks at most, all start at w0:
#   the hours of the window ending with week i: 600, 768, 936, 1104, 1200 for i = 3 to 7 (96 + 168 (i - 1), and 96 more at i = 7); alpha x hours: 30, 38.4, 46.8, 55.2, 60.
#   P1: the alerts of the five nodes over the window, sorted (node k = 0..4: 0 + 1 + 2 + 3 = 6 at i = 3, ...):
#       i = 3: 6, 18, 30, 42, 54   i = 4: 10, 25, 40, 55, 70   i = 5: 15, 33, 51, 69, 70   i = 6: 21, 42, 63, 71, 84   i = 7: 28, 52, 73, 76, 100
#       (N4 wraps: 12, 13, 14, 15, 16, 0, 1, 2); the first quartile, median and third quartile of five values are the second, third and fourth.
#   P4: (i + 1) k + i (i + 1) / 2 for node Nk: i = 3: 6, 10, 14, 18, 22   i = 4: 10, 15, 20, 25, 30   i = 5: 15, 21, 27, 33, 39   i = 6: 21, 28, 35, 42, 49   i = 7: 28, 36, 44, 52, 60.
P1_WINDOWS = {3: [6, 18, 30, 42, 54], 4: [10, 25, 40, 55, 70], 5: [15, 33, 51, 69, 70], 6: [21, 42, 63, 71, 84], 7: [28, 52, 73, 76, 100]}
P4_WINDOWS = {3: [6, 10, 14, 18, 22], 4: [10, 15, 20, 25, 30], 5: [15, 21, 27, 33, 39], 6: [21, 28, 35, 42, 49], 7: [28, 36, 44, 52, 60]}
ALPHA_HOURS = {3: 30.0, 4: 38.4, 5: 46.8, 6: 55.2, 7: 60.0}              # 0.05 x the hours of the window ending with week i
#: The ends of those windows in the first panel: the Monday after week i (w3 starts on 27 January, w7 on 24 February), and for the last week of four days (24 February
#: plus four days) 28 February, not Monday 3 March.
WINDOW_ENDS_2025 = [pd.Timestamp(d) for d in ("2025-02-03", "2025-02-10", "2025-02-17", "2025-02-24", "2025-02-28")]
WINDOW_ENDS_2026 = [pd.Timestamp(d) for d in ("2026-02-02", "2026-02-09", "2026-02-16", "2026-02-23", "2026-02-27")]


def _quartile_values(windows):
    """[q1, median, q3] of the window R of each window i = 3..7: the second, third and fourth of the five sorted alert sums over alpha x hours."""
    return np.array([[windows[i][1], windows[i][2], windows[i][3]] for i in range(3, 8)]) / np.array([[ALPHA_HOURS[i]] for i in range(3, 8)])


def _drift(tmp_path, **kw):
    from es.paper_figures import drift_frame
    src = _sources(tmp_path)
    if kw:
        _write_drift(src.out_dir, **kw)
        src._frames.clear()
    return src, drift_frame(src)


def test_the_summary_is_the_median_and_the_quartiles_over_nodes_of_the_window_rates(tmp_path):
    from es.paper_figures import drift_summary
    _, frame = _drift(tmp_path)
    for policy, windows in (("P1", P1_WINDOWS), ("P4", P4_WINDOWS)):
        s = drift_summary(frame, "2024-2025", policy)
        assert list(s.columns) == ["week", "q1", "median", "q3", "nodes"] and len(s) == 5 and set(s["nodes"]) == {5}
        np.testing.assert_allclose(s[["q1", "median", "q3"]].to_numpy(), _quartile_values(windows))
        assert list(s.week) == WINDOW_ENDS_2025
    p1 = drift_summary(frame, "2024-2025", "P1")
    assert list(p1["median"]) == pytest.approx([30 / 30, 40 / 38.4, 51 / 46.8, 63 / 55.2, 73 / 60])           # 1.0, 1.0417, 1.0897, 1.1413, 1.2167
    assert list(p1["q1"]) == pytest.approx([18 / 30, 25 / 38.4, 33 / 46.8, 42 / 55.2, 52 / 60]) and list(p1["q3"]) == pytest.approx([42 / 30, 55 / 38.4, 69 / 46.8, 71 / 55.2, 76 / 60])
    p4 = drift_summary(frame, "2024-2025", "P4")
    assert list(p4["median"]) == pytest.approx([14 / 30, 20 / 38.4, 27 / 46.8, 35 / 55.2, 44 / 60])           # 0.4667, 0.5208, 0.5769, 0.6341, 0.7333
    assert list(drift_summary(frame, "25H1-26H1", "P1").week) == WINDOW_ENDS_2026


def test_no_point_is_before_the_end_of_the_fourth_week_and_the_x_positions_are_the_ends_of_the_windows(tmp_path):
    """The first window has the four weeks so far (w0 to w3: 6 to 27 January), so its point is on Monday 3 February; nothing is drawn on 13, 20 or 27 January
    (the ends of the windows of one to three weeks). The last point is the day after the last settlement day: 28 February, 24 February plus four days. The lines of
    the figure have their points at those days, in both panels."""
    from es.paper_figures import BLACK, BLUE, drift_summary, fig_drift
    _, frame = _drift(tmp_path)
    s = drift_summary(frame, "2024-2025", "P1")
    assert s.week.min() == pd.Timestamp("2025-02-03") and s.week.max() == pd.Timestamp("2025-02-28")
    assert not {pd.Timestamp(d) for d in ("2025-01-13", "2025-01-20", "2025-01-27")} & set(s.week)
    fig = fig_drift(frame)
    for ax, ends in zip(fig.axes, (WINDOW_ENDS_2025, WINDOW_ENDS_2026)):
        for color in (BLACK, BLUE):
            lines = _lines_of(ax, color)
            assert len(lines) == 3                                                          # the median and the two quartiles
            for line in lines:
                assert list(pd.DatetimeIndex(line.get_xdata())) == ends
    plt.close(fig)


def test_a_node_whose_window_has_too_few_hours_is_left_out_of_it_and_the_count_says_so(tmp_path):
    """N4 gets 5 hours in every week of P1 in 2025, so each of its windows (at most 8 x 5 = 40 hours, against half of 4 x 168 = 336 or more) is left out and the four
    others are summarised: for the window of w3 their alert sums are 6, 18, 30, 42, so R = 0.2, 0.6, 1.0, 1.4 (over 30); median (0.6 + 1.0) / 2 = 0.8,
    first quartile 0.2 + 0.75 (0.6 - 0.2) = 0.5, third quartile 1.0 + 0.25 (1.4 - 1.0) = 1.1. P4 is untouched (five nodes)."""
    from es.paper_figures import drift_summary
    _, frame = _drift(tmp_path)
    frame = frame.copy()
    frame.loc[(frame.analysis == "2024-2025") & (frame.policy == "P1") & (frame.node == "N4"), "n_hours"] = 5
    s = drift_summary(frame, "2024-2025", "P1")
    assert set(s["nodes"]) == {4} and len(s) == 5
    assert list(s.iloc[0][["q1", "median", "q3"]]) == pytest.approx([0.5, 0.8, 1.1])
    assert set(drift_summary(frame, "2024-2025", "P4")["nodes"]) == {5}


def test_a_window_in_which_no_node_has_enough_hours_is_not_drawn(tmp_path):
    """Every node of P1 has 20 hours in weeks w0 to w3: the windows ending with w3, w4, w5 and w6 have 80, 248, 416 and 584 hours, against 336, 420, 504 and 588 (half
    the nominal hours of 4, 5, 6 and 7 weeks): all left out. The window ending with w7 has 80 + 168 x 3 + 96 = 680 hours, against 672: kept, with the alert
    sums of the fixture, 28, 52, 73, 76, 100, over 0.05 x 680 = 34: the median is 73 / 34."""
    from es.paper_figures import drift_summary
    _, frame = _drift(tmp_path)
    frame = frame.copy()
    first_four = {pd.Timestamp("2025-01-06") + pd.Timedelta(days=7 * w) for w in range(4)}
    frame.loc[(frame.analysis == "2024-2025") & (frame.policy == "P1") & frame.week_start.isin(first_four), "n_hours"] = 20
    s = drift_summary(frame, "2024-2025", "P1")
    assert list(s.week) == [pd.Timestamp("2025-02-28")] and list(s["nodes"]) == [5]
    assert list(s["median"]) == pytest.approx([73 / 34]) and list(s["q1"]) == pytest.approx([52 / 34]) and list(s["q3"]) == pytest.approx([76 / 34])


def test_the_summary_reads_the_alerts_and_the_hours_and_neither_the_weekly_rate_nor_the_flag_of_a_full_week(tmp_path):
    from es.paper_figures import drift_summary
    _, frame = _drift(tmp_path)
    changed = frame.assign(full_week=False, R=99.0, rate=99.0)
    for policy in ("P1", "P4"):
        pd.testing.assert_frame_equal(drift_summary(changed, "2024-2025", policy), drift_summary(frame, "2024-2025", policy))


def test_an_analysis_without_rows_or_without_a_window_is_an_error_that_names_it(tmp_path):
    from es.paper_figures import drift_summary
    _, frame = _drift(tmp_path)
    with pytest.raises(LookupError, match="2025-H2"):
        drift_summary(frame, "2025-H2", "P1")
    short = frame[frame.week_start < pd.Timestamp("2025-01-27")]                  # only w0 to w2 of 2025 (three weeks): the first window needs four
    with pytest.raises(LookupError, match=r"2024-2025.*P1.*no window"):
        drift_summary(short, "2024-2025", "P1")


def _drift_fig(tmp_path, **kw):
    from es.paper_figures import fig_drift
    return fig_drift(_drift(tmp_path, **kw)[1])


def _lines_of(ax, color, lw_min=0.0, lw_max=99.0):
    from matplotlib.colors import to_rgb
    return [line for line in ax.lines if len(line.get_xdata()) > 2 and to_rgb(line.get_color()) == to_rgb(color) and lw_min <= line.get_linewidth() <= lw_max]


def test_fig_drift_has_two_panels_the_deployment_year_and_the_first_half_of_2026(tmp_path):
    fig = _drift_fig(tmp_path)
    assert len(fig.axes) == 2
    assert [ax.get_title(loc="left") for ax in fig.axes] == ["(a) Deployment year 2025", "(b) First half of 2026"]
    assert not fig._suptitle                                                    # no title inside the figure
    plt.close(fig)


def test_each_policy_is_a_thick_median_and_two_thin_quartiles_of_one_colour_and_dash(tmp_path):
    from es.paper_figures import BLACK, BLUE
    fig = _drift_fig(tmp_path)
    ax = fig.axes[0]
    for color, dashed, windows in ((BLACK, False, P1_WINDOWS), (BLUE, True, P4_WINDOWS)):
        thick, thin = _lines_of(ax, color, 1.5), _lines_of(ax, color, 0.0, 0.9)
        assert len(thick) == 1 and len(thin) == 2, (color, len(thick), len(thin))
        assert (thick[0].get_linestyle() != "-") == dashed
        assert all(ln.get_linestyle() == thick[0].get_linestyle() for ln in thin)          # the same style: the thin lines are the quartiles of the thick one
        want = _quartile_values(windows)
        assert list(thick[0].get_ydata()) == pytest.approx(list(want[:, 1]))
        q1_line, q3_line = sorted(thin, key=lambda ln: sum(ln.get_ydata()))
        assert list(q1_line.get_ydata()) == pytest.approx(list(want[:, 0])) and list(q3_line.get_ydata()) == pytest.approx(list(want[:, 2]))
    plt.close(fig)


def test_the_two_policies_differ_by_colour_and_by_line_style_not_by_colour_alone(tmp_path):
    """distinguishable in black and white: P1 is solid, P4 is dashed"""
    from es.paper_figures import BLACK, BLUE
    fig = _drift_fig(tmp_path)
    p1 = _lines_of(fig.axes[0], BLACK, 1.5)[0]
    p4 = _lines_of(fig.axes[0], BLUE, 1.5)[0]
    assert p1.get_linestyle() == "-" and p4.get_linestyle() != "-"                      # solid against a dash pattern
    plt.close(fig)


def test_the_x_limits_are_the_deployment_window_of_each_analysis(tmp_path):
    import matplotlib.dates as mdates
    fig = _drift_fig(tmp_path)
    got = [tuple(mdates.num2date(v).replace(tzinfo=None).date().isoformat() for v in ax.get_xlim()) for ax in fig.axes]
    assert got == [("2025-01-01", "2026-01-01"), ("2026-01-01", "2026-07-01")]
    plt.close(fig)


def test_the_y_axis_is_log2_with_the_ticks_of_the_brief_and_a_zero_that_can_be_drawn(tmp_path):
    fig = _drift_fig(tmp_path)
    fig.canvas.draw()
    for ax in fig.axes:
        assert [t.get_text() for t in ax.get_yticklabels()] == ["0", "0.25", "0.5", "1", "2", "4"]
        y = ax.transData.transform([(0, v) for v in (0.25, 0.5, 1, 2, 4)])[:, 1]
        steps = np.diff(y)
        assert steps == pytest.approx([steps[0]] * 4) and steps[0] > 8, steps         # one octave is one step: a log2 axis (more than 8 pixels)
        zero = ax.transData.transform((0, 0))[1]
        assert np.isfinite(zero) and zero < y[0] - 8                                  # zero is drawn, below 0.25
    assert [ax.get_ylabel() for ax in fig.axes] == ["Alert rate / target"] * 2
    plt.close(fig)


def test_the_upper_limit_is_the_highest_third_quartile_plus_five_percent_and_never_below_the_top_tick(tmp_path):
    """The fixture's highest third quartile is 69 / 46.8 = 1.474 (P1, the window of w5), below the top tick 4: the limit is 4 x 1.05 = 4.2. With the
    last two nodes of P1 in 2025 at 100 alerts in every week, the sums of the window of w3 are 6, 18, 30, 400, 400 (over 30): the third quartile is 400 / 30 = 13.33, and the
    limit 14.0; the other windows have 500 / 38.4 = 13.02, 600 / 46.8 = 12.82, 700 / 55.2 = 12.68 and 800 / 60 = 13.33."""
    from es.paper_figures import drift_frame, fig_drift
    src = _sources(tmp_path)
    fig = fig_drift(drift_frame(src))
    assert [ax.get_ylim()[1] for ax in fig.axes] == pytest.approx([4.2, 4.2])
    plt.close(fig)
    frame = drift_frame(src)
    frame.loc[(frame.analysis == "2024-2025") & (frame.policy == "P1") & frame.node.isin(["N3", "N4"]), "alerts"] = 100
    fig = fig_drift(frame)
    for ax in fig.axes:
        top = ax.get_ylim()[1]
        assert top == pytest.approx(400 / 30 * 1.05)
        assert all(max(line.get_ydata()) <= top for line in ax.lines if len(line.get_xdata()) > 2)      # nothing is clipped
    plt.close(fig)


def test_the_band_is_shaded_the_target_is_dashed_and_nothing_is_transparent(tmp_path):
    from matplotlib.colors import to_rgb
    from es.paper_figures import BAND
    from matplotlib.patches import Rectangle
    fig = _drift_fig(tmp_path)
    ax = fig.axes[0]
    bands = [p for p in ax.patches if isinstance(p, Rectangle) and to_rgb(p.get_facecolor()) == to_rgb(BAND)]
    assert len(bands) == 1 and bands[0].get_y() == pytest.approx(0.5) and bands[0].get_y() + bands[0].get_height() == pytest.approx(2.0)
    one = [ln for ln in ax.lines if list(ln.get_ydata()) == [1, 1] and ln.get_linestyle() != "-"]
    assert len(one) == 1                                                        # R = 1, dashed
    fig.canvas.draw()
    from matplotlib.artist import Artist
    assert [a for a in fig.findobj(Artist) if a.get_visible() and a.get_alpha() not in (None, 1)] == []     # (the legend's frame has alpha 0.8 and is not drawn)
    plt.close(fig)


def test_month_ticks_and_the_names_of_the_months(tmp_path):
    import matplotlib.dates as mdates
    fig = _drift_fig(tmp_path)
    fig.canvas.draw()
    for ax, first, n in ((fig.axes[0], "2025-01-01", 12), (fig.axes[1], "2026-01-01", 6)):
        start = pd.Timestamp(first)
        majors = [mdates.num2date(v).replace(tzinfo=None) for v in ax.xaxis.get_majorticklocs()]
        assert all(m.day == 1 for m in majors) and majors[0] == start.to_pydatetime() and len(majors) == n + 1     # first of each month, both ends
        names = [t.get_text() for t in ax.xaxis.get_minorticklabels() if t.get_text()]
        assert names == ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:n]
    plt.close(fig)


def test_no_month_label_touches_another_and_none_is_cut_off(tmp_path):
    fig = _drift_fig(tmp_path)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        boxes = [t.get_window_extent(r) for t in ax.xaxis.get_minorticklabels() if t.get_text()]
        for a, b in zip(boxes, boxes[1:]):
            assert a.x1 < b.x0, "two month labels touch"
        assert boxes[0].x0 >= ax.get_window_extent(r).x0 - 1 and boxes[-1].x1 <= ax.get_window_extent(r).x1 + 1
    plt.close(fig)


def test_the_legend_is_below_the_axes_and_names_both_policies_the_band_and_the_target(tmp_path):
    fig = _drift_fig(tmp_path)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    assert len(fig.legends) == 1
    leg = fig.legends[0].get_window_extent(r)
    for ax in fig.axes:
        tight = ax.get_tightbbox(r)
        assert leg.y1 <= tight.y0 + 0.5, "the legend sits on an axis or its labels"
    entries = [t.get_text() for t in fig.legends[0].get_texts()]
    assert any("P1" in e for e in entries) and any("P4" in e for e in entries) and any("[0.5, 2]" in e for e in entries) and any("R = 1" in e for e in entries)
    plt.close(fig)


def test_the_note_under_the_legend_says_what_the_lines_are(tmp_path):
    """The brief's words: the median over nodes of the alert rate over the last eight weeks (the weeks so far at the start), and its quartiles. No word of the weekly
    figure is left ('full weeks only'), and the note fits the figure (it is two lines, below the legend)."""
    fig = _drift_fig(tmp_path)
    texts = _texts(fig)
    note = [t for t in texts if t.startswith("thick line")]
    assert len(note) == 1 and " ".join(note[0].split()) == ("thick line: median over nodes of the alert rate over the last eight weeks "
                                                          "(the weeks so far at the start); thin lines: its quartiles")
    assert not [t for t in texts if "full week" in t or "weekly" in t.lower()]
    r = fig.canvas.get_renderer()
    note_box = [t for t in fig.texts if t.get_text().startswith("thick line")][0].get_window_extent(r)
    assert note_box.x0 >= 0 and note_box.x1 <= fig.bbox.width and note_box.y1 <= fig.legends[0].get_window_extent(r).y0 + 0.5
    plt.close(fig)


def test_fig_drift_never_says_better_worse_or_pending(tmp_path):
    fig = _drift_fig(tmp_path)
    assert not [s for s in _texts(fig) if any(w in s.lower() for w in ("better", "worse", "pending"))]
    plt.close(fig)


def test_a_median_of_zero_is_drawn_on_the_zero_line_and_not_dropped(tmp_path):
    """Three of the five nodes of P1 in 2026-H1 issue no alert in any week: in every window their R is 0, so the median and the first quartile are 0 (the
    sorted values are 0, 0, 0 and two more), and the line is drawn at zero."""
    from es.paper_figures import BLACK, drift_frame, drift_summary, fig_drift
    src = _sources(tmp_path)
    frame = drift_frame(src)
    frame.loc[(frame.analysis == "25H1-26H1") & (frame.policy == "P1") & frame.node.isin(["N0", "N1", "N2"]), "alerts"] = 0
    s = drift_summary(frame, "25H1-26H1", "P1")
    assert list(s["median"]) == [0.0] * 5 and list(s["q1"]) == [0.0] * 5 and (s["q3"] > 0).all()
    fig = fig_drift(frame)
    thick = _lines_of(fig.axes[1], BLACK, 1.5)[0]
    assert list(thick.get_ydata()) == [0.0] * 5 and not np.isnan(thick.get_ydata()).any()
    plt.close(fig)


def test_drift_frame_needs_its_columns_and_both_analyses(tmp_path):
    from es.paper_figures import drift_frame, fig_drift
    src = _sources(tmp_path)
    d = pd.read_csv(src.out_dir / "paper" / "drift_path.csv")
    d.drop(columns="full_week").to_csv(src.out_dir / "paper" / "drift_path.csv", index=False)
    src._frames.clear()
    with pytest.raises(LookupError, match=r"drift_path.*full_week"):
        drift_frame(src)
    d[d.analysis == "2024-2025"].to_csv(src.out_dir / "paper" / "drift_path.csv", index=False)
    src._frames.clear()
    with pytest.raises(LookupError, match="25H1-26H1"):
        fig_drift(drift_frame(src))


# ---- FigContrast: the matched-activity contrasts with the calendar-only rule ---------------------------------------------------
CONTRAST_ROWS = ["AQ3: 2025-Q3", "AQ4: 2025-Q4", "AQ1: 2026-Q1", "AQ2: 2026-Q2"]
#: (estimate, lo, hi) of the paired contrast of each quarter. dSELC is a fraction (the figure draws percentage points), dESred is in USD.
D_SELC = {"AQ3": (-0.020, -0.066, 0.030), "AQ4": (-0.001, -0.015, 0.020), "AQ1": (0.036, -0.007, 0.075), "AQ2": (0.031, -0.010, 0.062)}
D_ESRED = {"AQ3": (-2538.0, -9935.0, 2328.0), "AQ4": (-482.0, -1137.0, -36.0), "AQ1": (-54.0, -1020.0, 661.0), "AQ2": (194.0, -936.0, 2047.0)}


def _contrast_sources(tmp_path, pooled=False):
    src = _sources(tmp_path)
    p = src.out_dir / "economics" / "bootstrap_ci.csv"
    ci = pd.read_csv(p)
    for stat, values in (("E4:dSELC|calendar@count", D_SELC), ("E4:dESred|calendar@count", D_ESRED)):
        for a, (est, lo, hi) in values.items():
            ci.loc[(ci.analysis == a) & (ci.stat == stat), ["estimate", "lo", "hi"]] = [est, lo, hi]
    if pooled:
        row = ci[(ci.analysis == "AQ3") & ci.stat.isin(["E4:dSELC|calendar@count", "E4:dESred|calendar@count"])]
        ci = pd.concat([ci, row.assign(analysis="AQ3+AQ4 (rehearsal of the pooled estimands)", estimate=123456.0, lo=-1.0, hi=1.0)], ignore_index=True)
    ci.to_csv(p, index=False)
    src._frames.clear()
    return src


def _contrast_fig(tmp_path, **kw):
    from es.paper_figures import e1_frame, fig_contrast
    return fig_contrast(e1_frame(_contrast_sources(tmp_path, **kw)))


def _contrast_marks(ax):
    """(marker x values, interval segments) of one axes: the filled markers and the bars of the intervals."""
    markers = [ln for ln in ax.lines if ln.get_marker() == "o" and ln.get_linestyle() == "None"]
    assert len(markers) == 1
    bars = [seg for c in ax.collections for seg in c.get_segments()]
    return list(markers[0].get_xdata()), sorted(bars, key=lambda s: s[0][1])


def test_the_rows_are_the_four_quarters_in_chronological_order_and_there_is_no_pooled_row(tmp_path):
    fig = _contrast_fig(tmp_path, pooled=True)                                  # the file also holds the pooled rehearsal: it is not drawn
    fig.canvas.draw()
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert len(labels) == 4 and [ln.split("\n")[0] for ln in labels] == CONTRAST_ROWS
    assert not [s for s in _texts(fig) if "AQ3+AQ4" in s or "pool" in s.lower() or "rehearsal" in s.lower()]
    ys = [t.get_position()[1] for t in fig.axes[0].get_yticklabels()]
    assert ys == sorted(ys) and fig.axes[0].get_ylim()[0] > fig.axes[0].get_ylim()[1]       # AQ3 at the top
    for ax in fig.axes:
        x, bars = _contrast_marks(ax)
        assert len(x) == 4 and len(bars) == 4 and 123456.0 not in x
    plt.close(fig)


def test_the_two_panels_share_the_rows_and_draw_the_paired_contrasts_with_their_intervals(tmp_path):
    fig = _contrast_fig(tmp_path)
    assert len(fig.axes) == 2
    for ax, values, scale in ((fig.axes[0], D_SELC, 100), (fig.axes[1], D_ESRED, 1)):
        x, bars = _contrast_marks(ax)
        rows = ["AQ3", "AQ4", "AQ1", "AQ2"]
        assert x == pytest.approx([scale * values[a][0] for a in rows])
        np.testing.assert_allclose([(s[0][0], s[1][0]) for s in bars], [(scale * values[a][1], scale * values[a][2]) for a in rows])
        assert [s[0][1] for s in bars] == [0, 1, 2, 3]                            # one bar per row, at the row's position
    plt.close(fig)


def test_each_panel_has_a_vertical_zero_line_and_the_notes_on_which_side_is_higher(tmp_path):
    from matplotlib.text import Annotation
    fig = _contrast_fig(tmp_path)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        zero = [ln for ln in ax.lines if list(ln.get_xdata()) == [0, 0]]
        assert len(zero) == 1
        z = ax.transData.transform((0, 0))[0]
        notes = {" ".join(a.get_text().split()): a.get_window_extent(r) for a in ax.texts if isinstance(a, Annotation)}     # two lines each
        right = [b for t, b in notes.items() if "alerts higher" in t]
        left = [b for t, b in notes.items() if "calendar-only higher" in t]
        assert len(right) == 1 and len(left) == 1
        assert right[0].x0 >= z - 0.5 and left[0].x1 <= z + 0.5                  # 'alerts higher' right of zero, 'calendar-only higher' left of zero
        box = ax.get_window_extent(r)
        assert right[0].x1 <= box.x1 + 0.5 and left[0].x0 >= box.x0 - 0.5         # inside the axes: the limits leave room for the notes
        assert right[0].y1 <= box.y0 and left[0].y1 <= box.y0                     # under the axis
    plt.close(fig)


def _assert_no_text_overlap(fig):
    """No two visible texts of a figure (tick labels, titles, notes, axis labels, legend entries, cell numbers) overlap one another. A tick label
    counts only if its tick is inside the view: the locator also makes ticks beyond the limits, whose labels are never drawn."""
    from matplotlib.text import Text
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    undrawn = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not (lo - 1e-9 <= tick.get_loc() <= hi + 1e-9):
                    undrawn.update({id(tick.label1), id(tick.label2)})
    shown = [(t.get_text(), t.get_window_extent(r)) for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and id(t) not in undrawn]
    for i, (a, box_a) in enumerate(shown):
        for b, box_b in shown[i + 1:]:
            assert not box_a.overlaps(box_b), f"{a!r} overlaps {b!r}"


def test_no_two_texts_of_a_new_figure_overlap(tmp_path):
    from es.paper_figures import drift_frame, e1_frame, fail_frame, fig_contrast, fig_drift, fig_fail
    src = _contrast_sources(tmp_path)
    for fig in (fig_fail(fail_frame(src)), fig_drift(drift_frame(src)), fig_contrast(e1_frame(src))):
        _assert_no_text_overlap(fig)
        plt.close(fig)


def test_the_x_labels_carry_their_units_and_the_figure_never_says_better(tmp_path):
    fig = _contrast_fig(tmp_path)
    labels = [ax.get_xlabel() for ax in fig.axes]
    assert "percentage points" in labels[0] and "SELC" in labels[0]
    assert "USD" in labels[1] and "ESred" in labels[1]
    assert any("calendar-only rule" in s and "same number of alerts" in s for s in _texts(fig))     # what is subtracted
    assert not [s for s in _texts(fig) if "better" in s.lower() or "worse" in s.lower() or "pending" in s.lower()]
    plt.close(fig)


def test_the_figure_says_that_the_intervals_are_approximate_and_the_rows_exploratory(tmp_path):
    fig = _contrast_fig(tmp_path)
    text = " ".join(_texts(fig))
    assert "approximate" in text and "95 %" in text and "exploratory" in text.lower()
    assert "opened" in [t.get_text() for t in fig.axes[0].get_yticklabels()][2] and "opened" in [t.get_text() for t in fig.axes[0].get_yticklabels()][3]
    assert "opened" not in [t.get_text() for t in fig.axes[0].get_yticklabels()][0]                  # the rows that use the opened 2026-H1 data say so
    plt.close(fig)


def test_a_contrast_row_the_file_lacks_is_an_error_that_names_it(tmp_path):
    from es.paper_figures import e1_frame, fig_contrast
    src = _contrast_sources(tmp_path)
    p = src.out_dir / "economics" / "bootstrap_ci.csv"
    ci = pd.read_csv(p)
    ci[~((ci.analysis == "AQ2") & (ci.stat == "E4:dESred|calendar@count"))].to_csv(p, index=False)
    src._frames.clear()
    with pytest.raises(LookupError, match=r"AQ2.*E4:dESred\|calendar@count"):
        fig_contrast(e1_frame(src))


# ---- the index ---------------------------------------------------------------------------------------------------------------
def test_the_index_lists_the_sources_and_the_caption_notes_of_the_new_figures(built):
    for state in ("pending", "sealed"):
        index = (built[state][0] / "INDEX.md").read_text()
        for name, source in (("FigFail", "legacy/summary.csv"), ("FigDrift", "paper/drift_path.csv"), ("FigContrast", "economics/bootstrap_ci.csv")):
            row = [ln for ln in index.splitlines() if ln.startswith(f"| {name} |")]
            assert len(row) == 1 and source in row[0] and row[0].endswith("| built |"), row
            assert "sealed/A" not in row[0]                                       # no Test A panel for them
            assert any(ln.startswith(f"- {name}:") for ln in index.splitlines()), name
        assert "realised alert rate outside" in index                            # what a failure is: the caption note of FigFail
        assert "pending" not in index                                           # (T1b) INDEX.md says plain "built" while Test A has not run


def test_the_index_status_is_plain_built_while_test_a_has_not_run_and_is_what_it_was_when_it_has_run_or_not(tmp_path):
    """(T1b) Fig1 to Fig4 print nothing of Test A while it has not run, so their status is "built" and not "built (Test A pending)". A Test A that has run (its
    results.json is there) but left a figure's own table out, and a Test A that did not run, keep the status they had: "built (Test A pending)" and
    "built (Test A did not run)"; a Test A with all its tables is "built" as it was (test_sealed_panels_appear_when_test_a_exists)."""
    from es.paper_figures import build_all
    src = _sources(tmp_path)
    figdir = tmp_path / "paper" / "generated" / "figures"
    status = build_all(src, figdir)
    assert set(status) == set(ALL_FIGS) and set(status.values()) == {"built"}
    rows = [line for line in (figdir / "INDEX.md").read_text().splitlines() if line.startswith("| Fig")]
    assert len(rows) == 7 and all(line.endswith("| built |") for line in rows), rows
    (src.out_dir / "sealed" / "A").mkdir(parents=True)
    (src.out_dir / "sealed" / "A" / "results.json").write_text("{}")                      # Test A has run, and no table of it is in the folder
    status = build_all(src, figdir)
    assert {status[n] for n in FIGS} == {"built (Test A pending)"} and [status[n] for n in NEW_FIGS] == ["built"] * 3
    (src.out_dir / "sealed" / "A" / "results.json").write_text('{"coverage": {"ran": false}}')           # Test A did not run
    src._json.clear()                                                                                       # Sources has read the old results.json
    status = build_all(src, figdir)
    assert {status[n] for n in FIGS} == {"built (Test A did not run)"} and [status[n] for n in NEW_FIGS] == ["built"] * 3


def test_the_caption_note_of_figdrift_describes_the_windows_and_not_the_weeks(built):
    for state in ("pending", "sealed"):
        index = (built[state][0] / "INDEX.md").read_text()
        note = " ".join([ln for ln in index.splitlines() if ln.startswith("- FigDrift:")][0].split())
        assert "over trailing windows of eight Monday-start weeks" in note and "the weeks so far from the fourth week on" in note
        assert "each point is drawn at the end of its window" in note and "partial weeks and weeks with missing hours count with the hours they have" in note
        assert "fewer than half its nominal hours is left out" in note and "outputs/es/paper/drift_summary.csv" in note and "outputs/es/paper/drift_path.csv" in note
        assert "weekly realised" not in note and "full weeks only" not in note
        assert "FigDrift | paper/drift_path.csv + no Test A panel | built |" in index


def test_a_figure_whose_source_is_missing_is_skipped_and_its_stale_files_are_removed(tmp_path):
    from es.paper_figures import build_all
    src = _sources(tmp_path)
    figdir = tmp_path / "paper" / "generated" / "figures"
    build_all(src, figdir)
    for n, source in (("FigFail", "legacy/summary.csv"), ("FigDrift", "paper/drift_path.csv")):
        assert (figdir / f"{n}.pdf").exists()
        (src.out_dir / source).unlink()
        status = build_all(src, figdir)
        assert status[n] == f"skipped: missing {source}"
        assert not (figdir / f"{n}.pdf").exists() and not (figdir / f"{n}.eps").exists()
        assert f"| {n} |" in (figdir / "INDEX.md").read_text() and f"skipped: missing {source}" in (figdir / "INDEX.md").read_text()


def test_the_new_figures_leave_the_callers_rcparams_alone(tmp_path, built):
    from es.paper_figures import STYLE, drift_frame, e1_frame, fail_frame, fig_contrast, fig_drift, fig_fail, save
    src = built["pending"][1]
    before = {k: matplotlib.rcParams[k] for k in STYLE}
    for fig, name in ((fig_fail(fail_frame(src)), "FigFail"), (fig_drift(drift_frame(src)), "FigDrift"), (fig_contrast(e1_frame(src)), "FigContrast")):
        save(fig, name, tmp_path)
    assert {k: matplotlib.rcParams[k] for k in STYLE} == before


def test_the_new_figures_style_themselves_so_a_caller_outside_build_all_gets_embedded_arial(tmp_path, built):
    from es.paper_figures import drift_frame, e1_frame, fail_frame, fig_contrast, fig_drift, fig_fail, save
    src = built["pending"][1]
    with plt.rc_context(matplotlib.rcParamsDefault):
        made = ((fig_fail(fail_frame(src)), "FigFail"), (fig_drift(drift_frame(src)), "FigDrift"), (fig_contrast(e1_frame(src)), "FigContrast"))
        files = [save(fig, name, tmp_path) for fig, name in made]
    for pdf, eps in files:
        if Path(PDFFONTS).exists():
            fonts = subprocess.run([PDFFONTS, str(pdf)], capture_output=True, text=True).stdout
            assert "Arial" in fonts and "DejaVu" not in fonts and "Type 3" not in fonts, fonts
        assert b"/FontName /ArialMT" in eps.read_bytes()


# ---- the real tree (read only) ---------------------------------------------------------------------------------------------------
REAL_PAPER = Path(__file__).resolve().parents[3] / "paper"
REAL_OUTPUTS = Path(__file__).resolve().parents[2] / "outputs" / "es"
needs_real_tree = pytest.mark.skipif(
    not all((REAL_OUTPUTS / f).exists() for f in ("legacy/summary.csv", "economics/bootstrap_ci.csv", "paper/drift_path.csv", "analogs/activity_all.csv"))
    or not (REAL_PAPER / "generated" / "tables" / "tab_block_audit.tex").exists() or not (REAL_PAPER / "generated" / "macros.tex").exists(),
    reason="the outputs of the pipeline or the generated tables are not there (python -m es.run_driftpath, paper/build.sh all)")


def _table_3_cells():
    """{block name: [XGB 2024-25, logit 2024-25, XGB 25H1-26H1, logit 25H1-26H1, XGB frozen]} read from the LaTeX of tab_block_audit."""
    out = {}
    for line in (REAL_PAPER / "generated" / "tables" / "tab_block_audit.tex").read_text().splitlines():
        cells = [c.strip().rstrip("\\").strip() for c in line.split("&")]
        if len(cells) == 7 and cells[0] in CORE_NAMES:
            out[cells[0]] = [int(c) for c in cells[2:]]
    return out


def _macro_numbers(name):
    """(estimate, lo, hi) of a macro pair ('desredCalCountAqFour' and its CI macro) from generated/macros.tex, as numbers."""
    text = (REAL_PAPER / "generated" / "macros.tex").read_text()

    def value(macro):
        m = re.search(r"\\newcommand\{\\" + macro + r"\}\{(.*)\}\s*$", text, re.M)
        assert m, macro
        return m.group(1).replace("\\textminus{}", "-").replace("\\$", "").replace("\\%", "").replace("{,}", "")

    lo, hi = value(name + "CI").strip("[]").split(", ")
    return float(value(name)), float(lo), float(hi)


@needs_real_tree
def test_the_real_tree_builds_all_seven_figures_and_none_prints_pending(tmp_path):
    from es.paper_figures import build_all
    from es.paper_macros import Sources
    src = Sources()
    status = build_all(src, tmp_path)
    assert {status[n] for n in NEW_FIGS} == {"built"}, status
    if src.exists("sealed/A/results.json"):
        pytest.skip("Test A has run: its panels are drawn")
    for n in ALL_FIGS:
        assert not [s for s in _eps_strings(tmp_path / f"{n}.eps") if "pending" in s], n


@needs_real_tree
def test_fig_fail_of_the_real_tree_prints_the_cells_of_table_3():
    from es.paper_figures import fail_frame, fig_fail
    from es.paper_macros import Sources
    table = _table_3_cells()
    assert sorted(table) == sorted(CORE_NAMES), table                             # the six rows that the figure draws are in the table
    fig = fig_fail(fail_frame(Sources()))
    left, right = _fail_axes(fig)
    for r, name in enumerate(CORE_NAMES):
        xgb1, logit1, xgb2, logit2, frozen = table[name]
        assert [left[(0, r)], left[(1, r)]] == [str(xgb1), str(xgb2)], name
        assert [right[(0, r)], right[(1, r)], right[(2, r)]] == [str(logit1), str(logit2), str(frozen)], name
    assert "of 33" in " ".join(_texts(fig))                                       # read from n_nodes, which is 33 in every cell
    plt.close(fig)


@needs_real_tree
def test_fig_contrast_of_the_real_tree_has_exactly_four_rows_and_no_pooled_row_and_agrees_with_the_macros():
    from es.paper_figures import e1_frame, fig_contrast
    from es.paper_macros import Sources
    fig = fig_contrast(e1_frame(Sources()))
    fig.canvas.draw()
    labels = [t.get_text().split("\n")[0] for t in fig.axes[0].get_yticklabels()]
    assert labels == CONTRAST_ROWS and not [s for s in _texts(fig) if "pool" in s.lower() or "AQ3+AQ4" in s]
    (x_selc, bars_selc), (x_esred, bars_esred) = (_contrast_marks(ax) for ax in fig.axes)
    assert len(x_selc) == len(x_esred) == 4
    est, lo, hi = _macro_numbers("desredCalCountAqFour")                          # -482 [-1137, -36]: the second row of the right panel
    assert (round(x_esred[1]), round(bars_esred[1][0][0]), round(bars_esred[1][1][0])) == (est, lo, hi) == (-482, -1137, -36)
    est, lo, hi = _macro_numbers("dselcCalCountAqOne")                            # 3.6 [-0.7, 7.5] percent: the third row of the left panel
    assert (round(x_selc[2], 1), round(bars_selc[2][0][0], 1), round(bars_selc[2][1][0], 1)) == (est, lo, hi) == (3.6, -0.7, 7.5)
    for name, row, ax_x, ax_bars, digits in (("dselcCalCountAqThree", 0, x_selc, bars_selc, 1), ("dselcCalCountAqFour", 1, x_selc, bars_selc, 1),
                                             ("dselcCalCountAqTwo", 3, x_selc, bars_selc, 1), ("desredCalCountAqThree", 0, x_esred, bars_esred, 0),
                                             ("desredCalCountAqOne", 2, x_esred, bars_esred, 0), ("desredCalCountAqTwo", 3, x_esred, bars_esred, 0)):
        est, lo, hi = _macro_numbers(name)
        got = (round(ax_x[row], digits), round(ax_bars[row][0][0], digits), round(ax_bars[row][1][0], digits))
        assert got == pytest.approx((est, lo, hi), abs=10 ** -digits), name
    plt.close(fig)
