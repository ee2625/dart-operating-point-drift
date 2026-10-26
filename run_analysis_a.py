"""Analysis A — magnitude score as a risk filter on a virtual carry position.

**v2.** Executes the protocol sealed in ``PREREGISTRATION_ANALYSIS_A.md`` (v1,
sha256 ``0cfc9c8f...``), as corrected by
``analysis_a_preregistration_v2_amendment.md`` and
``PREREG_ZERO_LABEL_TARGET_CALIBRATION.md``. Thresholds come from F1/F2 only and
are frozen before any F3/F4 realized DART value is read.

**Corrects a defect in v1's transfer arm.** v1's single ``zero_shot`` arm
calibrated its threshold on the *target* market's F1/F2 scores while training on
source markets — see ``DEFECT_2026-07-20_zero_shot_threshold_mislabelled.md``.
That arm was actually ``zero_label_target_calibrated``. Pure zero-shot — source
threshold applied unchanged to an unseen target — was never run. This version
implements three arms with explicit threshold provenance on every row, so the
mistake cannot recur silently:

    within                          — target model,   target F1/F2 threshold
    zero_shot_source_threshold      — transfer model,  source F1/F2 threshold
    zero_label_target_calibrated    — transfer model,  target F1/F2 threshold

The two transfer arms share one fitted model per (specification, target
definition, fold, held-out market) — never refit — so the only difference
between them is where the threshold came from. That isolates ranking transfer
from probability-scale transfer:

    Does the ranking transfer?              -> zero_shot_source_threshold's AUPRC
                                                (unaffected by threshold policy)
    Does the probability scale transfer?    -> zero_shot_source_threshold's
                                                realized_target_abstention vs budget
    Can unlabeled target scores repair it?  -> zero_label_target_calibrated
                                                vs zero_shot_source_threshold
    Does any of that improve tail risk?     -> the P&L comparisons below

Prior Analysis A transfer results measured zero-label target calibration, not
pure zero-shot deployment. This is stated once here, frozen, rather than
re-derived narratively per result.

Returns the result table without a narrative verdict, per the protocol.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src import calibration, transfer
from src.paths import DEV, assert_writable

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

RANDOM_SEED = 42
MARKETS = ("NYISO", "ISONE", "CAISO")
VALIDATION_FOLDS = ("F1", "F2")     # thresholds only
REPORTING_FOLDS = ("F3", "F4")      # evaluation only
ALERT_BUDGETS = (0.05, 0.025)       # primary, robustness
COST_GRID = (0.00, 0.05, 0.10, 0.25, 0.50)
REPETITIONS = 400

SPECIFICATIONS = {
    "G6": ["G6"],
    "G6+G3-R": ["G6", "G3-R"],
    "G6+G4": ["G6", "G4"],
    "all_shared": ["G6", "G1", "G4", "G2-R", "G3-R", "G5-P"],
}

#: Target definition -> (cache file, label column). Raw is primary; the
#: season/hour-standardized target is the prespecified robustness check.
TARGET_DEFINITIONS = {
    "raw": ("common_panel_v1.parquet", "label_high_spread"),
    "standardized": ("common_panel_conditional_v1.parquet", "label_high_spread_conditional"),
}

PREREG_FILES = {
    "v1": "PREREGISTRATION_ANALYSIS_A.md",
    "v2_amendment": "analysis_a_preregistration_v2_amendment.md",
    "zero_label_prereg": "PREREG_ZERO_LABEL_TARGET_CALIBRATION.md",
}


def model_factory(scale_pos_weight: float):
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=2.0, scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr", random_state=RANDOM_SEED, n_jobs=4,
    )


def loss_episodes(frame: pd.DataFrame, pnl_column: str) -> pd.Series:
    """Label contiguous runs of loss-making hours (the largest-loss-episode unit)."""
    ordered = frame.sort_values("timestamp_utc")
    losing = ordered[pnl_column] < 0
    episode = (losing & ~losing.shift(1, fill_value=False)).cumsum()
    return episode.where(losing)


def strategy_metrics(frame: pd.DataFrame, pnl_column: str) -> dict:
    pnl = frame[pnl_column].dropna()
    if pnl.empty:
        return {}

    daily = frame.set_index("timestamp_utc")[pnl_column].resample("D").sum()
    cumulative = pnl.cumsum()
    drawdown = (cumulative.cummax() - cumulative).max()

    tail_5 = pnl.quantile(0.05)
    tail_1 = pnl.quantile(0.01)

    episodes = loss_episodes(frame, pnl_column)
    episode_totals = (
        frame.assign(_ep=episodes).dropna(subset=["_ep"])
        .groupby("_ep")[pnl_column].sum()
    )

    return {
        "hours": int(len(pnl)),
        "mean_pnl_per_mwh": float(pnl.mean()),
        "total_pnl": float(pnl.sum()),
        "median_daily_pnl": float(daily.median()),
        "daily_volatility": float(daily.std()),
        "sharpe_like": float(daily.mean() / daily.std() * np.sqrt(365))
        if daily.std() > 0 else np.nan,
        "maximum_drawdown": float(drawdown),
        "expected_shortfall_5pct": float(pnl[pnl <= tail_5].mean()),
        "expected_shortfall_1pct": float(pnl[pnl <= tail_1].mean()),
        "worst_day": float(daily.min()),
        "worst_episode": float(episode_totals.min()) if len(episode_totals) else np.nan,
    }


def paired_bootstrap(work: pd.DataFrame, filtered_column: str, baseline_column: str):
    """Paired 168h-block bootstrap on the mean-P&L and ES5 differences."""
    rng = np.random.default_rng(RANDOM_SEED)
    block, n = 168, len(work)
    es_diffs: list[float] = []
    mean_diffs: list[float] = []
    if n >= block * 2:
        blocks = int(np.ceil(n / block))
        for _ in range(REPETITIONS):
            starts = rng.integers(0, n - block + 1, size=blocks)
            idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
            sample = work.iloc[idx]
            b = sample[baseline_column]
            f = sample[filtered_column]
            bq, fq = b.quantile(0.05), f.quantile(0.05)
            es_diffs.append(f[f <= fq].mean() - b[b <= bq].mean())
            mean_diffs.append(f.mean() - b.mean())
    return es_diffs, mean_diffs


def build_position_record(work: pd.DataFrame, keep: pd.Series, threshold_row: dict,
                          spec_name: str, target_definition: str, market: str,
                          fold: str, budget: float, prereg_hashes: dict) -> dict:
    """P&L, decomposition, success classification, and provenance for one cell."""
    work = work.copy()
    work["pnl_filtered_inc"] = work["pnl_always_inc"].where(keep, 0.0)

    base = strategy_metrics(work, "pnl_always_inc")
    filt = strategy_metrics(work, "pnl_filtered_inc")

    positive = work.loc[work["pnl_always_inc"] > 0, "pnl_always_inc"].sum()
    negative = work.loc[work["pnl_always_inc"] < 0, "pnl_always_inc"].sum()
    positive_kept = work.loc[keep & (work["pnl_always_inc"] > 0), "pnl_always_inc"].sum()
    negative_kept = work.loc[keep & (work["pnl_always_inc"] < 0), "pnl_always_inc"].sum()

    episodes = loss_episodes(work, "pnl_always_inc")
    totals = (
        work.assign(_ep=episodes).dropna(subset=["_ep"])
        .groupby("_ep")["pnl_always_inc"].sum().sort_values()
    )
    worst_five = totals.head(5).index
    avoided_five = 0.0
    if len(worst_five):
        in_worst = work.assign(_ep=episodes)["_ep"].isin(worst_five)
        avoided_five = float((~keep & in_worst).sum() / max(in_worst.sum(), 1))

    es_diffs, mean_diffs = paired_bootstrap(work, "pnl_filtered_inc", "pnl_always_inc")

    baseline_mean = base["mean_pnl_per_mwh"]
    filtered_mean = filt["mean_pnl_per_mwh"]
    baseline_sign = "positive_carry" if baseline_mean > 0 else "nonpositive_baseline"

    # "arm" (which strategy this row is) and "threshold_origin_arm" (where its
    # threshold came from) are expected to be identical in every row a correct
    # pipeline produces. Keeping them as two separately-sourced columns rather
    # than one is deliberate: it is exactly the comparison that would have
    # caught the v1 defect, where a threshold's real origin silently diverged
    # from the arm name it was filed under.
    record = {
        "specification": spec_name, "target_definition": target_definition,
        "market": market, "fold": fold, "alert_budget": budget,
        "arm": threshold_row["threshold_origin_arm"],
        **threshold_row,
        "realized_target_abstention": float((~keep).mean()),

        "baseline_sign": baseline_sign,
        "baseline_mean_pnl": baseline_mean,
        "filtered_mean_pnl": filtered_mean,
        "mean_pnl_uplift": filtered_mean - baseline_mean,
        # v2 amendment 3A/3B: retention is meaningful only when baseline > 0.
        "carry_retention": (filtered_mean / baseline_mean
                            if baseline_sign == "positive_carry" else np.nan),

        "baseline_median_daily_pnl": base["median_daily_pnl"],
        "filtered_median_daily_pnl": filt["median_daily_pnl"],
        "baseline_daily_volatility": base["daily_volatility"],
        "filtered_daily_volatility": filt["daily_volatility"],
        "baseline_sharpe": base["sharpe_like"],
        "filtered_sharpe": filt["sharpe_like"],
        "baseline_max_drawdown": base["maximum_drawdown"],
        "filtered_max_drawdown": filt["maximum_drawdown"],
        "baseline_es5": base["expected_shortfall_5pct"],
        "filtered_es5": filt["expected_shortfall_5pct"],
        "es5_improvement": filt["expected_shortfall_5pct"] - base["expected_shortfall_5pct"],
        "baseline_es1": base["expected_shortfall_1pct"],
        "filtered_es1": filt["expected_shortfall_1pct"],
        "es1_improvement": filt["expected_shortfall_1pct"] - base["expected_shortfall_1pct"],
        "baseline_worst_day": base["worst_day"],
        "filtered_worst_day": filt["worst_day"],
        "baseline_worst_episode": base["worst_episode"],
        "filtered_worst_episode": filt["worst_episode"],

        "positive_carry_sacrificed": (1 - positive_kept / positive) if positive else np.nan,
        # v2 amendment 3B: this fraction IS "percentage reduction in total losses".
        "negative_pnl_avoided": (1 - negative_kept / negative) if negative else np.nan,
        "worst_five_loss_episode_hours_avoided": avoided_five,
    }

    if es_diffs:
        es_lo, es_hi = float(np.quantile(es_diffs, 0.025)), float(np.quantile(es_diffs, 0.975))
        mean_lo, mean_hi = float(np.quantile(mean_diffs, 0.025)), float(np.quantile(mean_diffs, 0.975))
        record.update({
            "es5_diff_lower": es_lo, "es5_diff_upper": es_hi,
            "es5_diff_excludes_zero": bool(es_lo > 0 or es_hi < 0),
            "mean_diff_lower": mean_lo, "mean_diff_upper": mean_hi,
            "mean_diff_excludes_zero": bool(mean_lo > 0 or mean_hi < 0),
        })
    else:
        record.update({
            "es5_diff_lower": np.nan, "es5_diff_upper": np.nan,
            "es5_diff_excludes_zero": False,
            "mean_diff_lower": np.nan, "mean_diff_upper": np.nan,
            "mean_diff_excludes_zero": False,
        })

    # v2 amendment §3 — the two success types, kept in separate columns,
    # each applicable only on its own domain, never pooled (§3C).
    if baseline_sign == "positive_carry":
        record["positive_carry_success"] = bool(
            record["es5_improvement"] > 0
            and record["es5_diff_excludes_zero"]
            and not np.isnan(record["carry_retention"])
            and record["carry_retention"] >= 0.80
        )
        record["loss_mitigation"] = np.nan
    else:
        record["positive_carry_success"] = np.nan
        record["loss_mitigation"] = bool(
            record["mean_diff_excludes_zero"] and record["es5_diff_excludes_zero"]
        )

    for cost in COST_GRID:
        traded = keep.sum()
        record[f"filtered_mean_pnl_cost_{cost:.2f}"] = (
            filtered_mean - cost * traded / len(work)
        )

    record.update(prereg_hashes)
    return record


def calibration_pass(frame: pd.DataFrame, groups: dict, prereg_hashes: dict) -> tuple[dict, list]:
    """Fit calibration models on F1/F2 and freeze thresholds with provenance.

    Returns ``(frozen_thresholds, diagnostics)`` where ``frozen_thresholds`` maps
    ``(spec, target_definition, arm, market)`` to a
    :class:`~src.calibration.ThresholdProvenance`, and ``diagnostics`` records
    every model-identifier / arm pairing for the shared-model audit.
    """
    pooled_scores: dict[tuple, list[pd.Series]] = {}
    diagnostics: list[dict] = []

    for target_definition, (_, label_column) in TARGET_DEFINITIONS.items():
        for spec_name, codes in SPECIFICATIONS.items():
            names = sorted({n for code in codes for n in groups[code]})

            for fold in VALIDATION_FOLDS:
                fold_frame = frame[
                    (frame["fold_id"] == fold) & (frame["target_definition"] == target_definition)
                ]
                working = transfer.normalize_within_market(fold_frame, names)

                for market in MARKETS:
                    target_train = working[
                        (working["iso"] == market) & (working["split_role"] == "train")
                    ]
                    target_eval = working[
                        (working["iso"] == market) & (working["split_role"] == "evaluation")
                    ]
                    source_train = working[
                        (working["iso"] != market) & (working["split_role"] == "train")
                    ]
                    source_eval = working[
                        (working["iso"] != market) & (working["split_role"] == "evaluation")
                    ]

                    # ---- within: own model, own threshold ----------------
                    within_model = transfer.fit_model(model_factory, target_train, names, label_column)
                    if within_model is not None:
                        scored = transfer.score_with_model(within_model, target_eval, names)
                        if scored is not None:
                            identifier = f"{target_definition}|{spec_name}|within|{market}|{fold}"
                            pooled_scores.setdefault(
                                (spec_name, target_definition, calibration.WITHIN, market), []
                            ).append(scored["score"])
                            diagnostics.append({
                                "target_definition": target_definition, "specification": spec_name,
                                "fold": fold, "market": market, "arm": calibration.WITHIN,
                                "model_identifier": identifier, "scored_rows": len(scored),
                            })

                    # ---- transfer model: fit ONCE, score TWICE ------------
                    transfer_model = transfer.fit_model(model_factory, source_train, names, label_column)
                    if transfer_model is None:
                        continue
                    identifier = calibration.transfer_model_identifier(
                        target_definition, spec_name, market, fold
                    )

                    source_scored = transfer.score_with_model(transfer_model, source_eval, names)
                    if source_scored is not None:
                        calibration.assert_no_target_rows(
                            source_eval, market, context=f"{identifier}/zero_shot_source"
                        )
                        pooled_scores.setdefault(
                            (spec_name, target_definition, calibration.ZERO_SHOT_SOURCE, market), []
                        ).append(source_scored["score"])
                        diagnostics.append({
                            "target_definition": target_definition, "specification": spec_name,
                            "fold": fold, "market": market,
                            "arm": calibration.ZERO_SHOT_SOURCE,
                            "model_identifier": identifier, "scored_rows": len(source_scored),
                        })

                    target_scored = transfer.score_with_model(transfer_model, target_eval, names)
                    if target_scored is not None:
                        # The frame a zero-label calibration routine would actually
                        # receive: target features plus the model's own scores,
                        # nothing else. Asserted here rather than on the full
                        # ``target_scored`` slice, which inherits every original
                        # panel column (labels, DART, prices included) from the
                        # copy and would fail this check unconditionally.
                        calibration.assert_no_outcome_columns(
                            target_scored[names + ["score"]],
                            context=f"{identifier}/zero_label_target",
                        )
                        pooled_scores.setdefault(
                            (spec_name, target_definition, calibration.ZERO_LABEL_TARGET, market), []
                        ).append(target_scored["score"])
                        diagnostics.append({
                            "target_definition": target_definition, "specification": spec_name,
                            "fold": fold, "market": market,
                            "arm": calibration.ZERO_LABEL_TARGET,
                            "model_identifier": identifier, "scored_rows": len(target_scored),
                        })

    frozen: dict[tuple, calibration.ThresholdProvenance] = {}
    for (spec_name, target_definition, arm, market), score_list in pooled_scores.items():
        pooled = pd.concat(score_list)
        if arm == calibration.ZERO_SHOT_SOURCE:
            origin_market = "+".join(m for m in MARKETS if m != market)
        else:
            origin_market = market
        for budget in ALERT_BUDGETS:
            frozen[(spec_name, target_definition, arm, market, budget)] = calibration.calibrate_threshold(
                pooled, budget, origin_arm=arm, origin_market=origin_market,
                calibration_folds="+".join(VALIDATION_FOLDS),
                model_identifier=f"{target_definition}|{spec_name}|{arm}|{market}",
            )

    return frozen, diagnostics


def reporting_pass(frame: pd.DataFrame, groups: dict, frozen: dict,
                   prereg_hashes: dict) -> list[dict]:
    """Apply frozen thresholds to F3/F4. First contact with F3/F4 P&L."""
    records: list[dict] = []

    for target_definition, (_, label_column) in TARGET_DEFINITIONS.items():
        for spec_name, codes in SPECIFICATIONS.items():
            names = sorted({n for code in codes for n in groups[code]})

            for fold in REPORTING_FOLDS:
                fold_frame = frame[
                    (frame["fold_id"] == fold) & (frame["target_definition"] == target_definition)
                ]
                working = transfer.normalize_within_market(fold_frame, names)

                for market in MARKETS:
                    target_train = working[
                        (working["iso"] == market) & (working["split_role"] == "train")
                    ]
                    target_eval = working[
                        (working["iso"] == market) & (working["split_role"] == "evaluation")
                    ]
                    source_train = working[
                        (working["iso"] != market) & (working["split_role"] == "train")
                    ]
                    if target_eval.empty:
                        continue

                    # ---- within ---------------------------------------
                    within_model = transfer.fit_model(model_factory, target_train, names, label_column)
                    within_scored = (
                        transfer.score_with_model(within_model, target_eval, names)
                        if within_model is not None else None
                    )

                    # ---- transfer model: fit ONCE, reused for BOTH arms ----
                    transfer_model = transfer.fit_model(model_factory, source_train, names, label_column)
                    transfer_scored = (
                        transfer.score_with_model(transfer_model, target_eval, names)
                        if transfer_model is not None else None
                    )

                    for budget in ALERT_BUDGETS:
                        for arm, scored in (
                            (calibration.WITHIN, within_scored),
                            (calibration.ZERO_SHOT_SOURCE, transfer_scored),
                            (calibration.ZERO_LABEL_TARGET, transfer_scored),
                        ):
                            if scored is None:
                                continue
                            key = (spec_name, target_definition, arm, market, budget)
                            if key not in frozen:
                                continue
                            provenance = frozen[key]

                            work = scored.sort_values("timestamp_utc").copy()
                            work["pnl_always_inc"] = work["dart_spread"]
                            keep = work["score"] < provenance.value

                            records.append(build_position_record(
                                work, keep, provenance.as_record(),
                                spec_name, target_definition, market, fold, budget,
                                prereg_hashes,
                            ))

    return records


def main() -> None:
    DEV.ensure()

    prereg_dir = Path(__file__).resolve().parent
    prereg_hashes = {
        f"{key}_sha256": hashlib.sha256((prereg_dir / filename).read_bytes()).hexdigest()
        for key, filename in PREREG_FILES.items()
    }
    for key, digest in prereg_hashes.items():
        print(f"{key}: {digest[:16]}...")
    print()

    groups = json.load(open(DEV.cache / "common_panel_groups.json"))
    all_features = sorted({n for v in groups.values() for n in v})

    frames = []
    for target_definition, (filename, label_column) in TARGET_DEFINITIONS.items():
        part = pd.read_parquet(DEV.cache / filename)
        part = part.dropna(subset=all_features).copy()
        part["timestamp_utc"] = pd.to_datetime(part["timestamp_utc"], utc=True)
        part["target_definition"] = target_definition
        part["label_high_spread"] = part[label_column]   # unify the modelling column
        frames.append(part)
    frame = pd.concat(frames, ignore_index=True)

    print("=" * 100)
    print("STEP 1 — calibration pass on F1/F2 (three arms, explicit provenance)")
    print("=" * 100)
    frozen, diagnostics = calibration_pass(frame, groups, prereg_hashes)
    print(f"frozen {len(frozen)} thresholds across specs x target-definitions x arms x markets x budgets\n")

    diag = pd.DataFrame(diagnostics)
    shared = (
        diag[diag["arm"].isin([calibration.ZERO_SHOT_SOURCE, calibration.ZERO_LABEL_TARGET])]
        .groupby(["target_definition", "specification", "fold", "market"])["model_identifier"]
        .nunique()
    )
    assert (shared == 1).all(), "transfer arms diverged onto different fitted models"
    print("verified: zero_shot_source_threshold and zero_label_target_calibrated "
          "share one model_identifier in every (spec, target, fold, market) cell\n")

    print("=" * 100)
    print("STEP 2 — reporting pass on F3/F4. First contact with F3/F4 P&L.")
    print("=" * 100)
    records = reporting_pass(frame, groups, frozen, prereg_hashes)
    results = pd.DataFrame.from_records(records)
    results.to_csv(assert_writable(DEV.tables / "analysis_a_risk_filter.csv"), index=False)
    print(f"{len(results)} result rows written\n")

    primary = results[
        (results["alert_budget"] == 0.05) & (results["target_definition"] == "raw")
    ]
    print("=" * 100)
    print("PRIMARY VIEW — raw target, 5% alert budget, F3/F4")
    print("=" * 100)
    columns = [
        "specification", "arm", "market", "fold", "threshold_origin_market",
        "realized_target_abstention", "baseline_sign",
        "baseline_mean_pnl", "filtered_mean_pnl", "mean_pnl_uplift", "carry_retention",
        "es5_improvement", "es5_diff_excludes_zero",
        "positive_carry_success", "loss_mitigation",
    ]
    print(primary[columns].round(4).to_string(index=False))

    print(f"\nSaved to {DEV.tables / 'analysis_a_risk_filter.csv'}")
    print("No automated narrative verdict is generated, per protocol.")


if __name__ == "__main__":
    main()
