"""Run the first multi-node three-way decomposition on the v2 backfill.

Loads the backfilled market parquets, restricts to the frozen NODE_MAP nodes,
runs panel_prep -> build_cells_v2 -> decompose, plus leave-one-market-out
sensitivity. Writes the cells and the summary to the local data root.

    python3.12 v2/run_decomposition.py --root ~/dart-spread-data/backfill_2023_2024 \
        --calib 2023 --deploy 2024
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from v2.cell_builder import build_cells_v2, g6_features
from v2.decomposition import decompose
from v2.panel_prep import add_features, add_label

HERE = Path(__file__).resolve().parent


def load_backfill(root: Path, markets: list[str]) -> pd.DataFrame:
    frames = []
    for m in markets:
        for f in glob.glob(str(root / f"{m.lower()}_da_rt_*.parquet")):
            frames.append(pd.read_parquet(f))
    if not frames:
        raise SystemExit(f"no parquet files under {root}")
    return pd.concat(frames, ignore_index=True)


def restrict_to_node_map(raw: pd.DataFrame) -> pd.DataFrame:
    nm = pd.read_csv(HERE / "NODE_MAP.csv")
    keep = set(zip(nm["market"], nm["node_id"]))
    ntype = dict(zip(zip(nm["market"], nm["node_id"]), nm["node_type"]))
    mask = [(m, n) in keep for m, n in zip(raw["market"], raw["node"])]
    out = raw[mask].copy()
    out["node_type"] = [ntype[(m, n)] for m, n in zip(out["market"], out["node"])]  # authoritative
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--calib", type=int, nargs="+", default=[2023])
    ap.add_argument("--deploy", type=int, nargs="+", default=[2024])
    ap.add_argument("--markets", nargs="+", default=["NYISO", "MISO", "ISONE", "CAISO"])
    a = ap.parse_args()
    root = Path(a.root).expanduser()

    raw = restrict_to_node_map(load_backfill(root, a.markets))
    print("panel:", raw.groupby("market")["node"].nunique().to_dict(),
          "rows", len(raw), flush=True)

    feat = add_features(raw)
    lab = add_label(feat, train_end=f"{min(a.deploy)}-01-01")
    cells = build_cells_v2(lab, g6_features(), calib_years=tuple(a.calib),
                           deploy_years=tuple(a.deploy))
    cells.to_csv(root / "v2_cells.csv", index=False)
    print(f"\ncells: {len(cells)}  {cells['contrast'].value_counts().to_dict()}")

    print("\n=== FULL (all markets) ===")
    full = decompose(cells, reps=10000)
    print(full.round(3).to_string(index=False))
    full.to_csv(root / "v2_decomposition.csv", index=False)

    print("\n=== leave-one-market-out (does the ordering survive?) ===")
    for m in sorted(cells["target_market"].unique()):
        sub = cells[(cells["target_market"] != m) & (cells["source_market"] != m)]
        d = decompose(sub, reps=4000)
        row = {c: d.loc[d["contrast"] == c, "mean_excess"].iloc[0]
               for c in ("spatial", "institutional")}
        print(f"  drop {m:6s}: spatial {row['spatial']:+.3f}  institutional {row['institutional']:+.3f}")


if __name__ == "__main__":
    main()
