"""Pin the read-time seal filter of es.data.load_dev_raw on synthetic files with fake prices.

The real development files end exactly at the seal, so tests on them hold with or without the
pyarrow predicate, and they cannot tell whether the cut is right for the tz-naive MISO files.
The tests here write files that straddle the seal by two days, switch off the post-hoc guard
(es.data.assert_development_safe) so that only the predicate can remove rows, and check every
market's last development hour. Two column layouts are covered: the real mix (MISO tz-naive,
the other markets tz-aware) and all tz-naive.

The second group checks the guard itself. It must see a sealed row even when the filter has
failed, including a row on a node outside v2/NODE_MAP.csv, which restrict_to_node_map drops.

The last test is the only one that reads the real files, and it goes through the filtered loader.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

START = pd.Timestamp("2026-06-29 00:00", tz="UTC")
END = pd.Timestamp("2026-07-03 00:00", tz="UTC")           # inclusive; two days past the seal
HOURS = pd.date_range(START, END, freq="h")

#: Last development hour per market, i.e. the hour before its first sealed settlement hour.
#: Written out here, not derived from es.guard, so the tests do not share the code under test.
LAST_DEV_HOUR = {
    "NYISO": pd.Timestamp("2026-07-01 03:00", tz="UTC"),
    "ISONE": pd.Timestamp("2026-07-01 03:00", tz="UTC"),
    "MISO": pd.Timestamp("2026-07-01 04:00", tz="UTC"),
    "CAISO": pd.Timestamp("2026-07-01 06:00", tz="UTC"),
}
MARKETS = tuple(LAST_DEV_HOUR)

#: layout -> markets whose timestamp column is tz-aware, timestamp[ns, tz=UTC]. The others are
#: tz-naive timestamp[ns] holding UTC wall-clock, as the real MISO files do.
LAYOUTS = {
    "real_mix": {"NYISO", "ISONE", "CAISO"},
    "all_naive": set(),
}

UNMAPPED_NODE = "NOT_IN_NODE_MAP"
NODE_MAP = pd.read_csv(Path(__file__).resolve().parents[2] / "v2" / "NODE_MAP.csv")


def _mapped_nodes(market):
    nodes = NODE_MAP[NODE_MAP["market"] == market]
    return list(zip(nodes["node_id"], nodes["node_type"]))


def _node_frame(market, node, node_type, hours, tz_aware, k):
    index = hours if tz_aware else hours.tz_localize(None)
    i = np.arange(len(hours))
    return pd.DataFrame({
        "market": market, "node": node, "node_type": node_type, "timestamp_utc": index,
        "da_lmp": 30.0 + k + (i % 24),                       # fake prices
        "rt_lmp": 31.0 + k + ((i * 7) % 24)})


def _write_market(directory, market, tz_aware, mapped_hours=HOURS, unmapped_hours=None):
    """Write '<market>_da_rt_test.parquet': every node of NODE_MAP.csv for the market on
    ``mapped_hours`` and, if given, one node that is not in the node map on ``unmapped_hours``."""
    assert UNMAPPED_NODE not in set(NODE_MAP["node_id"])
    parts = [_node_frame(market, node, node_type, mapped_hours, tz_aware, k)
             for k, (node, node_type) in enumerate(_mapped_nodes(market))]
    if unmapped_hours is not None:
        parts.append(_node_frame(market, UNMAPPED_NODE, "load_zone", unmapped_hours, tz_aware, 99))
    path = Path(directory) / f"{market.lower()}_da_rt_test.parquet"
    pq.write_table(pa.Table.from_pandas(pd.concat(parts, ignore_index=True), preserve_index=False), path)
    # The layout is the point of these files, so make sure the file really has it.
    expected = pa.timestamp("ns", tz="UTC") if tz_aware else pa.timestamp("ns")
    written = pq.ParquetFile(path).schema_arrow.field("timestamp_utc").type
    assert written == expected, f"{path.name}: wrote {written}, wanted {expected}"
    return path


def _write_layout(directory, layout):
    for market in MARKETS:
        _write_market(directory, market, market in LAYOUTS[layout])


def _load_predicate_only(directory, monkeypatch, layout):
    """Load the straddling files with the post-hoc guard off, so only the arrow predicate acts."""
    from es import data
    _write_layout(directory, layout)
    monkeypatch.setattr("es.data.assert_development_safe", lambda frame: frame)
    return data.load_dev_raw(dirs=[directory])


# ---- the read-time predicate ----------------------------------------------------------------

@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_predicate_stops_each_market_at_its_last_development_hour(tmp_path, monkeypatch, layout):
    raw = _load_predicate_only(tmp_path, monkeypatch, layout)
    assert raw.groupby("market")["timestamp_utc"].max().to_dict() == LAST_DEV_HOUR


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_predicate_keeps_every_development_row(tmp_path, monkeypatch, layout):
    raw = _load_predicate_only(tmp_path, monkeypatch, layout)
    rows_per_node = raw.groupby(["market", "node"]).size()
    for market in MARKETS:
        n_hours = int((LAST_DEV_HOUR[market] - START) / pd.Timedelta(hours=1)) + 1
        assert (rows_per_node[market] == n_hours).all(), market
        assert raw.loc[raw["market"] == market, "timestamp_utc"].min() == START, market


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_guards_leave_the_filtered_frame_alone(tmp_path, monkeypatch, layout):
    """With both guards on, the straddling files load without error and give the same frame as
    the predicate alone, so the last development hour of every market passes the guard."""
    from es import data
    _write_layout(tmp_path, layout)
    guarded = data.load_dev_raw(dirs=[tmp_path])
    monkeypatch.setattr("es.data.assert_development_safe", lambda frame: frame)
    pd.testing.assert_frame_equal(guarded, data.load_dev_raw(dirs=[tmp_path]))


# ---- the guard behind the predicate ---------------------------------------------------------

@pytest.mark.parametrize("sealed_on", ["mapped_nodes", "node_outside_node_map"])
@pytest.mark.parametrize("market", MARKETS)
def test_guard_trips_when_the_filter_fails(tmp_path, monkeypatch, market, sealed_on):
    """If the arrow filter silently stopped working, load_dev_raw must still refuse sealed rows,
    including rows on a node that restrict_to_node_map would drop before a late guard saw them."""
    from es import data
    from es.guard import SealedDataError
    tz_aware = market in LAYOUTS["real_mix"]
    if sealed_on == "mapped_nodes":
        _write_market(tmp_path, market, tz_aware)
    else:   # sealed rows only on the node outside the node map
        dev_hours = HOURS[HOURS <= LAST_DEV_HOUR[market]]
        _write_market(tmp_path, market, tz_aware, mapped_hours=dev_hours, unmapped_hours=HOURS)
    real_read_table = pq.read_table                      # simulate a filter that stopped working
    monkeypatch.setattr(
        pq, "read_table", lambda path, *args, filters=None, **kwargs: real_read_table(path, *args, **kwargs))
    with pytest.raises(SealedDataError):
        data.load_dev_raw(dirs=[tmp_path], markets=(market,))


# ---- one pin on the real files --------------------------------------------------------------

@pytest.fixture(scope="module")
def real_raw():
    from es.data import load_dev_raw
    return load_dev_raw()


def test_real_miso_files_start_at_est_midnight_in_utc(real_raw):
    """The real MISO files hold tz-naive timestamps. They are UTC wall-clock: MISO settles in
    fixed EST, so its first hour of 2023 begins at 05:00Z. This locks the convention that makes
    the cut for the naive columns right."""
    miso = real_raw[real_raw["market"] == "MISO"]
    assert miso["timestamp_utc"].min() == pd.Timestamp("2023-01-01 05:00", tz="UTC")
