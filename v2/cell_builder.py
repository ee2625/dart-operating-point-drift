"""v2 cell_builder: a prepared panel -> canonical operating-point cells.

For every ordered (source unit -> target unit) at the primary config (G6, raw,
5%), fit the source model, place the cutoff on the source calibration scores,
deploy on the target, and record the operating-point error. Each cell is tagged:

    temporal       source unit == target unit         (same market+node, later period)
    spatial        same market, different node
    institutional  different market, matching node_type (unless match_node_type=False)

Reuses v1's exact machinery (``src.transfer`` + ``analysis_a.model_factory`` + the
G6 feature group + ``label_high_spread``), so running it on the v1 panel reproduces
v1. Output feeds ``v2/decomposition.py``.

A "unit" is a (market, node) pair. Calibration folds F1/F2 set the cutoff;
reporting folds F3/F4 are deployed (v1 convention). The panel must carry
``market, node, node_type, fold_id, split_role``, the G6 feature columns, and the
label column.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import run_analysis_a as analysis_a
from run_operating_point_analysis import log_error
from src import transfer
from src.paths import DEV

ALPHA = 0.05
CALIB_FOLDS = ("F1", "F2")
DEPLOY_FOLDS = ("F3", "F4")


def g6_features() -> list[str]:
    groups = json.load(open(DEV.cache / "common_panel_groups.json"))
    return sorted(groups["G6"])


def build_cells(panel: pd.DataFrame, features: list[str],
                label_col: str = "label_high_spread", alpha: float = ALPHA,
                match_node_type: bool = True) -> pd.DataFrame:
    panel = panel.dropna(subset=features).copy()
    units = (panel[["market", "node", "node_type"]].drop_duplicates()
             .itertuples(index=False))
    units = list(units)
    rows: list[dict] = []

    for deploy_fold in DEPLOY_FOLDS:
        rep = panel[panel["fold_id"] == deploy_fold]
        cal = panel[panel["fold_id"].isin(CALIB_FOLDS)]
        for su in units:
            s_tr = rep[(rep["market"] == su.market) & (rep["node"] == su.node)
                       & (rep["split_role"] == "train")]
            centre, spread = transfer.fit_scaler(s_tr, features)
            model = transfer.fit_model(
                analysis_a.model_factory,
                transfer.apply_scaler(s_tr, features, centre, spread),
                features, label_col)
            if model is None:
                continue
            s_cal = cal[(cal["market"] == su.market) & (cal["node"] == su.node)
                        & (cal["split_role"] == "evaluation")]
            thr_scored = transfer.score_with_model(
                model, transfer.apply_scaler(s_cal, features, centre, spread), features)
            if thr_scored is None:
                continue
            tau = float(np.quantile(thr_scored["score"].to_numpy(float), 1 - alpha))

            for tu in units:
                if su.market == tu.market and su.node == tu.node:
                    contrast = "temporal"
                elif su.market == tu.market:
                    contrast = "spatial"
                elif (not match_node_type) or su.node_type == tu.node_type:
                    contrast = "institutional"
                else:
                    continue
                t_ev = rep[(rep["market"] == tu.market) & (rep["node"] == tu.node)
                           & (rep["split_role"] == "evaluation")]
                if t_ev.empty:
                    continue
                dep = transfer.score_with_model(
                    model, transfer.apply_scaler(t_ev, features, centre, spread), features)
                if dep is None:
                    continue
                r = float((dep["score"].to_numpy(float) >= tau).mean()) / alpha
                rows.append({
                    "target_market": tu.market, "target_node": tu.node,
                    "period": deploy_fold, "contrast": contrast,
                    "source_market": su.market, "source_node": su.node,
                    "R": r, "log_error": float(log_error(np.array([r]))[0]),
                })
    return pd.DataFrame(rows)


def build_cells_v2(panel: pd.DataFrame, features: list[str],
                   label_col: str = "label_high_spread", alpha: float = ALPHA,
                   calib_years: tuple[int, ...] = (2022, 2023),
                   deploy_years: tuple[int, ...] = (2024, 2025),
                   match_node_type: bool = True) -> pd.DataFrame:
    """v2-native cell builder driven by calendar years (era-period design) rather
    than v1's fold_id. Fit each source unit on all rows *before* the deploy year,
    place the cutoff on the source's calibration-year scores, deploy on the target
    unit's deploy-year rows. Awaits the multi-node backfill to run at scale;
    validated in miniature by the fold-based `build_cells` G-REPRO above.
    """
    panel = panel.dropna(subset=features).copy()
    panel["year"] = pd.to_datetime(panel["timestamp_utc"], utc=True).dt.year
    units = list(panel[["market", "node", "node_type"]].drop_duplicates()
                 .itertuples(index=False))
    rows: list[dict] = []
    for dy in deploy_years:
        for su in units:
            s_tr = panel[(panel["market"] == su.market) & (panel["node"] == su.node)
                         & (panel["year"] < dy)]
            centre, spread = transfer.fit_scaler(s_tr, features)
            model = transfer.fit_model(
                analysis_a.model_factory,
                transfer.apply_scaler(s_tr, features, centre, spread), features, label_col)
            if model is None:
                continue
            s_cal = panel[(panel["market"] == su.market) & (panel["node"] == su.node)
                          & (panel["year"].isin(calib_years))]
            thr = transfer.score_with_model(
                model, transfer.apply_scaler(s_cal, features, centre, spread), features)
            if thr is None:
                continue
            tau = float(np.quantile(thr["score"].to_numpy(float), 1 - alpha))
            for tu in units:
                if su.market == tu.market and su.node == tu.node:
                    contrast = "temporal"
                elif su.market == tu.market:
                    contrast = "spatial"
                elif (not match_node_type) or su.node_type == tu.node_type:
                    contrast = "institutional"
                else:
                    continue
                t_ev = panel[(panel["market"] == tu.market) & (panel["node"] == tu.node)
                             & (panel["year"] == dy)]
                dep = (transfer.score_with_model(
                    model, transfer.apply_scaler(t_ev, features, centre, spread), features)
                    if not t_ev.empty else None)
                if dep is None:
                    continue
                r = float((dep["score"].to_numpy(float) >= tau).mean()) / alpha
                rows.append({"target_market": tu.market, "target_node": tu.node,
                             "period": str(dy), "contrast": contrast,
                             "source_market": su.market, "source_node": su.node,
                             "R": r, "log_error": float(log_error(np.array([r]))[0])})
    return pd.DataFrame(rows)


def build_cells_windows(panel: pd.DataFrame, features: list[str],
                        train: tuple[str, str], calib: tuple[str, str],
                        deploy: tuple[str, str], label_col: str = "label_high_spread",
                        alpha: float = ALPHA, match_node_type: bool = True) -> pd.DataFrame:
    """Cell builder with explicit, separable train / calibrate / deploy windows
    (UTC [start, end), end-exclusive), addressing the in-sample-calibration confound:
    when ``calib`` is disjoint from ``train`` the cutoff sits on genuinely held-out
    scores. Records signed and absolute log-error, the realized ratio, a zero-activity
    flag, and deploy-set AUPRC + prevalence so ranking retention travels with every
    operating-point cell.
    """
    from sklearn.metrics import average_precision_score
    panel = panel.dropna(subset=features).copy()
    ts = pd.to_datetime(panel["timestamp_utc"], utc=True)
    panel = panel.assign(_ts=ts)

    def win(df, w):
        return df[(df["_ts"] >= pd.Timestamp(w[0], tz="UTC"))
                  & (df["_ts"] < pd.Timestamp(w[1], tz="UTC"))]

    units = list(panel[["market", "node", "node_type"]].drop_duplicates().itertuples(index=False))
    rows: list[dict] = []
    for su in units:
        su_rows = panel[(panel["market"] == su.market) & (panel["node"] == su.node)]
        s_tr = win(su_rows, train)
        centre, spread = transfer.fit_scaler(s_tr, features)
        model = transfer.fit_model(
            analysis_a.model_factory,
            transfer.apply_scaler(s_tr, features, centre, spread), features, label_col)
        if model is None:
            continue
        thr = transfer.score_with_model(
            model, transfer.apply_scaler(win(su_rows, calib), features, centre, spread), features)
        if thr is None:
            continue
        tau = float(np.quantile(thr["score"].to_numpy(float), 1 - alpha))
        for tu in units:
            if su.market == tu.market and su.node == tu.node:
                contrast = "temporal"
            elif su.market == tu.market and su.node_type == tu.node_type:
                contrast = "spatial"          # node-type matched, else it re-adds the hub/zone confound
            elif su.market != tu.market and ((not match_node_type) or su.node_type == tu.node_type):
                contrast = "institutional"
            else:
                continue
            tu_rows = panel[(panel["market"] == tu.market) & (panel["node"] == tu.node)]
            dep = transfer.score_with_model(
                model, transfer.apply_scaler(win(tu_rows, deploy), features, centre, spread), features)
            if dep is None:
                continue
            scores = dep["score"].to_numpy(float)
            y = dep[label_col].to_numpy(int)
            r = float((scores >= tau).mean()) / alpha
            auprc = (float(average_precision_score(y, scores))
                     if y.min() != y.max() else float("nan"))
            rows.append({
                "target_market": tu.market, "target_node": tu.node, "contrast": contrast,
                "source_market": su.market, "source_node": su.node,
                "R": r, "signed_logR": (float(np.log10(r)) if r > 0 else float("nan")),
                "log_error": (abs(float(np.log10(r))) if r > 0 else float("nan")),
                "zero_activity": bool(r == 0.0), "auprc": auprc,
                "prevalence": float(y.mean()), "n": int(len(y)), "n_pos": int(y.sum()),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # G-REPRO on the cached v1 panel (single node per market; cross-market
    # unmatched to mirror v1's design). Should reproduce v1's -0.001 excess.
    from v2.decomposition import decompose
    panel = pd.read_parquet(DEV.cache / "common_panel_v1.parquet")
    panel["market"] = panel["iso"]
    panel["node"] = panel["iso"]
    panel["node_type"] = "reference"
    cells = build_cells(panel, g6_features(), match_node_type=False)
    print(f"cells: {len(cells)} ({cells['contrast'].value_counts().to_dict()})\n")
    print(decompose(cells, reps=10000).round(3).to_string(index=False))
