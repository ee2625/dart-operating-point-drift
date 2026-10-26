"""Development raw prices and the feature panel. Rows in sealed windows are filtered at
read time (pyarrow predicate) and then asserted absent, on the raw read (before the node-map
restriction) and again on the final frame."""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from v2.panel_prep import add_features, add_g1_features
from v2.run_decomposition import restrict_to_node_map

from .config import DEV_DATA_DIRS, MARKETS
from .guard import assert_development_safe, first_sealed_utc, settlement_frame


def raw_files(dirs=DEV_DATA_DIRS, markets=MARKETS) -> list[Path]:
    files: list[Path] = []
    for directory in dirs:
        for market in markets:
            pattern = str(Path(directory) / f"{market.lower()}_da_rt_*.parquet")
            files += [Path(f) for f in sorted(glob.glob(pattern))]
    return files


def _market_of(path: Path) -> str:
    return path.name.split("_")[0].upper()


def load_dev_raw(dirs=DEV_DATA_DIRS, markets=MARKETS) -> pd.DataFrame:
    frames = []
    for path in raw_files(dirs, markets):
        cut = first_sealed_utc(_market_of(path)).to_pydatetime()
        # The files mix tz-aware (timestamp[ns, tz=UTC]) and tz-naive (MISO, UTC wall-clock)
        # timestamp columns, and arrow will not compare across the two, so type the cut like
        # the file's own column.
        ts_type = pq.ParquetFile(path).schema_arrow.field("timestamp_utc").type
        cut_scalar = pa.scalar(cut, type=ts_type)
        frames.append(pq.read_table(path, filters=[("timestamp_utc", "<", cut_scalar)]).to_pandas())
    raw = pd.concat(frames, ignore_index=True)
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True)
    # Guard the raw read itself. restrict_to_node_map would silently drop sealed rows on nodes
    # outside the node map, so a filter that failed must be caught before it.
    assert_development_safe(raw)
    raw = restrict_to_node_map(raw)
    n_before = len(raw)
    raw = (raw.drop_duplicates(["market", "node", "timestamp_utc"], keep="first")
              .sort_values(["market", "node", "timestamp_utc"]).reset_index(drop=True))
    assert_development_safe(raw)
    raw.attrs["duplicates_dropped"] = n_before - len(raw)
    return raw


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    panel = add_g1_features(add_features(raw))
    for h in (48, 72, 168):
        panel[f"mid_lag_{h}h"] = (panel[f"da_lmp_lag_{h}h"] + panel[f"rt_lmp_lag_{h}h"]) / 2
    panel = panel.reset_index(drop=True)
    s = settlement_frame(panel["timestamp_utc"], panel["market"]).reset_index(drop=True)
    panel = pd.concat([panel, s], axis=1)
    panel["wclass"] = (panel["weekday"] >= 5).astype(int)
    return panel
