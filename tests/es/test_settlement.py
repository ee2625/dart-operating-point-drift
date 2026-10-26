import io
import zipfile

import pandas as pd
import pytest

NODE = "TH_NP15_GEN-APND"
OASIS_COLUMNS = ["INTERVALSTARTTIME_GMT", "INTERVALENDTIME_GMT", "OPR_DT", "OPR_HR", "NODE_ID_XML", "NODE_ID", "NODE",
                 "MARKET_RUN_ID", "LMP_TYPE", "XML_DATA_ITEM", "PNODE_RESMRID", "GRP_TYPE", "POS", "PRC",
                 "OPR_INTERVAL", "GROUP"]
OASIS_ITEM = {"LMP": "LMP_PRC", "MCC": "LMP_CONG_PRC", "MCE": "LMP_ENE_PRC", "MCL": "LMP_LOSS_PRC"}
NYISO_HEADER = ('"Time Stamp","Name","PTID","LBMP ($/MWHr)","Marginal Cost Losses ($/MWHr)",'
                '"Marginal Cost Congestion ($/MWHr)"')


# ----------------------------------------------------------------------------- hermetic by construction
@pytest.fixture(autouse=True)
def _no_network_no_waiting(monkeypatch):
    """Every test runs with the network blocked and sleeping disabled. A test that needs a response installs its
    own `es.settlement._fetch` stub (see `_serve`); a test that wants to see the pauses uses the `sleeps` fixture."""
    import urllib.request

    import es.settlement as S

    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted in a unit test")

    monkeypatch.setattr(S, "_fetch", blocked)
    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    monkeypatch.setattr(S.time, "sleep", lambda seconds: None)


@pytest.fixture
def sleeps(monkeypatch):
    """Records the pauses the pullers ask for (OASIS needs more than 5 s between requests) instead of waiting."""
    import es.settlement as S

    calls = []
    monkeypatch.setattr(S.time, "sleep", calls.append)
    return calls


def _serve(monkeypatch, responder):
    """Install an `es.settlement._fetch` stub that answers `responder(url)`; returns the list of requested URLs."""
    import es.settlement as S

    urls = []

    def fake_fetch(url, attempts=5):
        urls.append(url)
        return responder(url)

    monkeypatch.setattr(S, "_fetch", fake_fetch)
    return urls


def _caiso_url(start, end, node=NODE):
    """The verified OASIS request, spelled out rather than built from es.settlement.CAISO_URL, so that a change
    to the template itself is caught. The window is [start 07:00 GMT, end 07:00 GMT), i.e. whole Pacific days in PDT."""
    return ("https://oasis.caiso.com/oasisapi/SingleZip?queryname=PRC_RTPD_LMP"
            f"&startdatetime={start}T07:00-0000&enddatetime={end}T07:00-0000&market_run_id=RTPD"
            f"&node={node}&version=1&resultformat=6")


def _nyiso_url(year_month):
    return f"http://mis.nyiso.com/public/csv/rtlbmp/{year_month}01rtlbmp_zone_csv.zip"


def _utc(stamp):
    return pd.Timestamp(stamp, tz="UTC")


def _by_time(frame, column):
    return dict(zip(frame.timestamp_utc, frame[column]))


def _oasis_zip(node, quarter_hours):
    """A synthetic OASIS PRC_RTPD_LMP answer (resultformat=6): a zipped CSV with the real column set.

    quarter_hours: [(interval start 'YYYY-MM-DD HH:MM' UTC, {lmp_type: price})]; one CSV row per lmp_type.
    """
    rows = []
    for start, prices in quarter_hours:
        t0 = _utc(start)
        local = t0.tz_convert("America/Los_Angeles")
        for lmp_type, price in prices.items():
            rows.append({"INTERVALSTARTTIME_GMT": t0.strftime("%Y-%m-%dT%H:%M:%S-00:00"),
                         "INTERVALENDTIME_GMT": (t0 + pd.Timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%S-00:00"),
                         "OPR_DT": local.strftime("%Y-%m-%d"), "OPR_HR": local.hour + 1, "NODE_ID_XML": node,
                         "NODE_ID": node, "NODE": node, "MARKET_RUN_ID": "RTPD", "LMP_TYPE": lmp_type,
                         "XML_DATA_ITEM": OASIS_ITEM[lmp_type], "PNODE_RESMRID": node, "GRP_TYPE": "ALL_APNODES",
                         "POS": 0, "PRC": price, "OPR_INTERVAL": 1, "GROUP": 1})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("20250701_20250702_PRC_RTPD_LMP_RTPD_v1.csv",
                   pd.DataFrame(rows, columns=OASIS_COLUMNS).to_csv(index=False))
    return buf.getvalue()


def _components(lmp, k):
    """All four OASIS price components for one interval; the three non-LMP ones are far from the LMP on purpose."""
    return {"LMP": lmp, "MCC": 900.0 + k, "MCE": 800.0 + k, "MCL": 700.0 + k}


def _quarter_hours(prices_by_start):
    return [(start, _components(price, k)) for k, (start, price) in enumerate(prices_by_start)]


def _nyiso_zip(days):
    """A synthetic monthly NYISO 'rtlbmp' zip of daily CSVs. days: {'YYYYMMDD': [(stamp, name, lbmp)]}."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for day, records in days.items():
            lines = [NYISO_HEADER] + [f'"{stamp}","{name}",61761,{lbmp},1.0,2.0' for stamp, name, lbmp in records]
            z.writestr(f"{day}rtlbmp_zone.csv", "\n".join(lines) + "\n")
    return buf.getvalue()


# ----------------------------------------------------------------------------- the sealed window
def test_pullers_refuse_sealed_dates(tmp_path, monkeypatch):
    import es.settlement as S
    from es.guard import SealedDataError

    def no_network(*args, **kwargs):
        raise AssertionError("a sealed pull must not reach the network")

    monkeypatch.setattr(S, "_fetch", no_network)
    monkeypatch.setattr(S.time, "sleep", lambda seconds: None)
    with pytest.raises(SealedDataError):
        S.pull_caiso_fmm("2026-06-01", "2026-07-02", tmp_path / "a")
    with pytest.raises(SealedDataError):
        S.pull_nyiso_integrated("2026-06", "2026-07", tmp_path / "b")
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()      # refused before creating anything


def test_seal_boundary_end_2026_07_01_is_refused_and_2026_06_30_is_allowed(tmp_path, monkeypatch):
    import es.settlement as S
    from es.guard import SealedDataError

    def forbidden(url, attempts=5):
        raise AssertionError(f"a sealed pull reached the network: {url}")

    monkeypatch.setattr(S, "_fetch", forbidden)
    with pytest.raises(SealedDataError):
        S.pull_caiso_fmm("2026-06-29", "2026-07-01", tmp_path / "ca_sealed")
    with pytest.raises(SealedDataError):
        S.pull_nyiso_integrated("2026-06", "2026-07-01", tmp_path / "ny_sealed")
    assert not (tmp_path / "ca_sealed").exists() and not (tmp_path / "ny_sealed").exists()

    # The last development day passes the guard, and no request reaches past its end.
    # 06:00-06:45 UTC on 1 July is 23:00-23:45 PDT on 30 June.
    last_hour = _quarter_hours([(f"2026-07-01 06:{m:02d}", 10.0 + m) for m in (0, 15, 30, 45)])
    caiso_urls = _serve(monkeypatch, lambda url: _oasis_zip(NODE, last_hour))
    path = S.pull_caiso_fmm("2026-06-29", "2026-06-30", tmp_path / "ca_ok", nodes=(NODE,))
    assert path.exists()
    assert caiso_urls == [_caiso_url("20260629", "20260701")]
    assert "enddatetime=20260701T07:00-0000" in caiso_urls[0]          # = first_sealed_utc('CAISO'), exclusive
    assert S.load_caiso_fmm(tmp_path / "ca_ok").timestamp_utc.tolist() == [_utc("2026-07-01 06:00")]

    ny_urls = _serve(monkeypatch, lambda url: _nyiso_zip({"20260630": [("06/30/2026 23:00", "N.Y.C.", 41.0)]}))
    path = S.pull_nyiso_integrated("2026-06", "2026-06-30", tmp_path / "ny_ok")
    assert path.exists()
    assert ny_urls == [_nyiso_url("202606")]
    assert S.load_nyiso_integrated(tmp_path / "ny_ok").timestamp_utc.tolist() == [_utc("2026-07-01 03:00")]


# ----------------------------------------------------------------------------- NYISO time stamps
def test_parse_nyiso_hour_beginning_eastern():
    from es.settlement import _parse_nyiso
    csv = ('"Time Stamp","Name","PTID","LBMP ($/MWHr)","Marginal Cost Losses ($/MWHr)",'
           '"Marginal Cost Congestion ($/MWHr)"\n"07/01/2025 00:00","N.Y.C.",61761,42.5,1.0,2.0\n')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("20250701rtlbmp_zone.csv", csv)
    out = _parse_nyiso(buf.getvalue())
    assert out.loc[0, "timestamp_utc"] == pd.Timestamp("2025-07-01 04:00", tz="UTC")
    assert out.loc[0, "rt_int_lmp"] == 42.5 and out.loc[0, "node"] == "N.Y.C."


def test_parse_nyiso_winter_and_summer_offsets():
    from es.settlement import _parse_nyiso
    out = _parse_nyiso(_nyiso_zip({"20250115": [("01/15/2025 00:00", "N.Y.C.", 31.0)],
                                   "20250715": [("07/15/2025 00:00", "N.Y.C.", 45.0)]}))
    assert _by_time(out, "rt_int_lmp") == {
        _utc("2025-01-15 05:00"): 31.0,        # winter, EST = UTC-5
        _utc("2025-07-15 04:00"): 45.0,        # summer, EDT = UTC-4
    }
    assert out.attrs["dropped_nat"] == 0


def test_parse_nyiso_drops_and_counts_the_repeated_fall_back_hour():
    from es.settlement import _parse_nyiso
    # 2025-11-02: the 01:00 hour happens twice (EDT, then EST) under one and the same time stamp,
    # so neither occurrence can be placed.
    records = []
    for name, base in (("N.Y.C.", 30.0), ("WEST", 20.0)):
        records += [("11/02/2025 00:00", name, base), ("11/02/2025 01:00", name, base + 1),
                    ("11/02/2025 01:00", name, base + 2), ("11/02/2025 02:00", name, base + 3)]
    out = _parse_nyiso(_nyiso_zip({"20251102": records}))
    assert out.attrs["dropped_nat"] == 4                                # 2 repeated rows x 2 names
    assert len(out) == 4
    assert _by_time(out[out.node == "N.Y.C."], "rt_int_lmp") == {
        _utc("2025-11-02 04:00"): 30.0,        # 00:00 EDT
        _utc("2025-11-02 07:00"): 33.0,        # 02:00 EST
    }
    assert not out.timestamp_utc.isin([_utc("2025-11-02 05:00"), _utc("2025-11-02 06:00")]).any()


def test_parse_nyiso_drops_and_counts_a_nonexistent_spring_forward_hour():
    from es.settlement import _parse_nyiso
    out = _parse_nyiso(_nyiso_zip({"20250309": [("03/09/2025 01:00", "N.Y.C.", 10.0),
                                                ("03/09/2025 02:00", "N.Y.C.", 11.0),     # does not exist on 9 March
                                                ("03/09/2025 03:00", "N.Y.C.", 12.0)]}))
    assert out.attrs["dropped_nat"] == 1
    assert _by_time(out, "rt_int_lmp") == {
        _utc("2025-03-09 06:00"): 10.0,        # 01:00 EST
        _utc("2025-03-09 07:00"): 12.0,        # 03:00 EDT
    }


def test_pull_nyiso_integrated_logs_every_drop_in_coverage(tmp_path, monkeypatch, sleeps):
    import es.settlement as S
    nyc, west = "N.Y.C.", "WEST"
    month_zips = {
        # 31 October, plus the first hour of 1 November, which the next month's zip lists again (for both names)
        "202510": _nyiso_zip({"20251031": [("10/31/2025 22:00", nyc, 10.0), ("10/31/2025 23:00", nyc, 11.0),
                                           ("11/01/2025 00:00", nyc, 12.0), ("11/01/2025 00:00", west, 22.0)]}),
        "202511": _nyiso_zip({"20251101": [("11/01/2025 00:00", nyc, 12.0), ("11/01/2025 00:00", west, 22.0)],
                              # 2 November: the 01:00 hour happens twice and cannot be placed
                              "20251102": [("11/02/2025 00:00", nyc, 13.0), ("11/02/2025 01:00", nyc, 14.0),
                                           ("11/02/2025 01:00", nyc, 15.0), ("11/02/2025 02:00", nyc, 16.0)]}),
    }
    urls = _serve(monkeypatch, lambda url: month_zips[url.rsplit("/", 1)[1][:6]])
    root = tmp_path / "nyiso"

    path = S.pull_nyiso_integrated("2025-10", "2025-11", root)

    assert urls == [_nyiso_url("202510"), _nyiso_url("202511")]
    assert sleeps == [2.0, 2.0]
    out = pd.read_parquet(path)
    assert set(zip(out.node, out.timestamp_utc)) == {
        (nyc, _utc("2025-11-01 02:00")), (nyc, _utc("2025-11-01 03:00")), (nyc, _utc("2025-11-01 04:00")),
        (nyc, _utc("2025-11-02 04:00")), (nyc, _utc("2025-11-02 07:00")), (west, _utc("2025-11-01 04:00"))}
    assert len(out) == 6                                                  # no duplicate (node, hour) is left
    assert _by_time(out[out.node == west], "rt_int_lmp") == {_utc("2025-11-01 04:00"): 22.0}

    cov = pd.read_csv(root / "coverage.csv")
    assert cov.columns.tolist() == ["month", "raw_rows", "dropped_nat", "rows", "dropped_duplicates"]
    assert cov.to_dict("records") == [
        {"month": "2025-10", "raw_rows": 4, "dropped_nat": 0, "rows": 4, "dropped_duplicates": 0},
        # November: the repeated 01:00 hour of the change day is dropped
        {"month": "2025-11", "raw_rows": 6, "dropped_nat": 2, "rows": 4, "dropped_duplicates": 0},
        # the last row is drop_duplicates: 8 rows in, 6 kept, 2 removed (the 1 November rows listed twice)
        {"month": "drop_duplicates", "raw_rows": 8, "dropped_nat": 0, "rows": 6, "dropped_duplicates": 2},
    ]
    assert (cov.raw_rows - cov.dropped_nat - cov.dropped_duplicates == cov.rows).all()    # every row reconciles
    assert cov.iloc[-1]["rows"] == len(out)


# ----------------------------------------------------------------------------- attach_settlement
def _rows():
    ts = pd.to_datetime(["2025-07-01 07:00", "2025-07-01 07:00", "2025-07-01 07:00"], utc=True)
    return pd.DataFrame({"market": ["CAISO", "NYISO", "ISONE"],
                         "node": ["TH_SP15_GEN-APND", "N.Y.C.", ".Z.MAINE"],
                         "timestamp_utc": ts, "da_lmp": [50.0, 40.0, 30.0], "rt_lmp": [45.0, 30.0, 20.0]})


def test_attach_settlement_uses_settlement_series():
    from es.settlement import attach_settlement
    r = _rows()
    fmm = pd.DataFrame({"node": ["TH_SP15_GEN-APND"], "timestamp_utc": r.timestamp_utc[:1], "fmm_lmp": [60.0]})
    nyi = pd.DataFrame({"node": ["N.Y.C."], "timestamp_utc": r.timestamp_utc[:1], "rt_int_lmp": [35.0]})
    out = attach_settlement(r, fmm, nyi)
    assert out["dart_settle"].tolist() == [-10.0, 5.0, 10.0]
    assert out["settle_approx"].tolist() == [False, False, False]


def test_attach_settlement_flags_missing_series():
    from es.settlement import attach_settlement
    out = attach_settlement(_rows(), None, None)
    assert out["dart_settle"].tolist() == [5.0, 10.0, 10.0]
    assert out["settle_approx"].tolist() == [True, True, False]


H0, H1, H2 = pd.to_datetime(["2025-07-01 07:00", "2025-07-01 08:00", "2025-07-01 09:00"], utc=True)
SP15, NP15 = "TH_SP15_GEN-APND", "TH_NP15_GEN-APND"


def test_attach_settlement_caiso_joins_on_node_and_hour():
    from es.settlement import attach_settlement
    rows = pd.DataFrame({"market": "CAISO",
                         "node": [SP15, NP15, NP15, SP15],
                         "timestamp_utc": [H1, H0, H1, H0],           # shuffled: neither node nor hour is in order
                         "da_lmp": [80.0, 55.0, 65.0, 75.0],
                         "rt_lmp": [70.0, 50.0, 60.0, 66.0]}, index=[10, 11, 12, 13])
    fmm = pd.DataFrame({"node": [SP15, NP15, SP15],                   # (NP15, H1) is not in the FMM table
                        "timestamp_utc": [H0, H0, H1],
                        "fmm_lmp": [101.0, 202.0, 303.0]})            # a different value for every node-hour

    out = attach_settlement(rows, fmm, None)

    assert out.index.tolist() == [10, 11, 12, 13]                     # row order and labels are kept
    # each row takes its own node's value for its own hour; the missing one keeps rt_lmp
    assert out["rt_settle"].tolist() == [303.0, 202.0, 60.0, 101.0]
    assert out.loc[12, "rt_settle"] == out.loc[12, "rt_lmp"] == 60.0
    assert out["settle_approx"].tolist() == [False, False, True, False]
    assert out["dart_settle"].tolist() == [80.0 - 303.0, 55.0 - 202.0, 65.0 - 60.0, 75.0 - 101.0]
    assert "rt_settle" not in rows.columns and "settle_approx" not in rows.columns      # the input is not modified


def test_attach_settlement_nyiso_joins_on_node_and_hour_and_leaves_other_markets_alone():
    from es.settlement import attach_settlement
    nyc, west = "N.Y.C.", "WEST"
    rows = pd.DataFrame({"market": ["NYISO", "ISONE", "NYISO", "MISO", "NYISO", "NYISO"],
                         "node": [west, ".Z.MAINE", nyc, "AECI", nyc, west],
                         "timestamp_utc": [H1, H0, H0, H0, H1, H0],
                         "da_lmp": [60.0, 30.0, 40.0, 25.0, 45.0, 41.0],
                         "rt_lmp": [50.0, 20.0, 30.0, 28.0, 33.0, 31.0]}, index=[5, 3, 9, 1, 7, 2])
    nyi = pd.DataFrame({"node": [nyc, west, west, ".Z.MAINE", "AECI"],   # (NYC, H1) is missing; the last two are decoys
                        "timestamp_utc": [H0, H0, H1, H0, H0],
                        "rt_int_lmp": [35.0, 36.0, 37.0, 999.0, 998.0]})

    out = attach_settlement(rows, None, nyi)

    assert out["rt_settle"].tolist() == [37.0, 20.0, 35.0, 28.0, 33.0, 36.0]
    assert out["settle_approx"].tolist() == [False, False, False, False, True, False]
    assert out["dart_settle"].tolist() == [60.0 - 37.0, 30.0 - 20.0, 40.0 - 35.0, 25.0 - 28.0, 45.0 - 33.0, 41.0 - 36.0]
    other = out["market"].isin(["ISONE", "MISO"])
    assert (out.loc[other, "rt_settle"] == out.loc[other, "rt_lmp"]).all()     # untouched ...
    assert not out.loc[other, "settle_approx"].any()                           # ... and unflagged
    pd.testing.assert_frame_equal(out[rows.columns], rows)                     # the original columns come back as given


def test_attach_settlement_applies_each_table_to_its_own_market_only():
    from es.settlement import attach_settlement
    rows = pd.DataFrame({"market": ["CAISO", "NYISO", "ISONE", "MISO", "CAISO", "NYISO"],
                         "node": [SP15, "N.Y.C.", ".Z.MAINE", "AECI", NP15, "WEST"],
                         "timestamp_utc": [H0, H0, H0, H0, H1, H1],
                         "da_lmp": [75.0, 40.0, 30.0, 25.0, 65.0, 60.0],
                         "rt_lmp": [66.0, 30.0, 20.0, 28.0, 60.0, 50.0]})
    fmm = pd.DataFrame({"node": [SP15, ".Z.MAINE", "N.Y.C."], "timestamp_utc": [H0, H0, H0],
                        "fmm_lmp": [101.0, 999.0, 997.0]})                # decoys named like other markets' nodes
    nyi = pd.DataFrame({"node": ["N.Y.C.", "AECI", SP15], "timestamp_utc": [H0, H0, H0],
                        "rt_int_lmp": [35.0, 998.0, 996.0]})

    out = attach_settlement(rows, fmm, nyi)

    assert out["rt_settle"].tolist() == [101.0, 35.0, 20.0, 28.0, 60.0, 50.0]
    # (NP15, H1) and (WEST, H1) are the two node-hours missing from their tables
    assert out["settle_approx"].tolist() == [False, False, False, False, True, True]
    assert out["dart_settle"].tolist() == [75.0 - 101.0, 40.0 - 35.0, 30.0 - 20.0, 25.0 - 28.0, 65.0 - 60.0,
                                           60.0 - 50.0]


def test_attach_settlement_requires_a_unique_index():
    from es.settlement import attach_settlement
    duplicated = pd.concat([_rows(), _rows()])                            # index 0, 1, 2, 0, 1, 2
    assert not duplicated.index.is_unique
    with pytest.raises(ValueError, match="rows must have a unique index"):
        attach_settlement(duplicated, None, None)


def test_attach_settlement_uses_only_complete_fmm_hours_when_n15_is_given():
    from es.settlement import attach_settlement
    rows = pd.DataFrame({"market": "CAISO", "node": SP15, "timestamp_utc": [H0, H1, H2],
                         "da_lmp": 50.0, "rt_lmp": [20.0, 21.0, 22.0]})
    fmm = pd.DataFrame({"node": SP15, "timestamp_utc": [H0, H1, H2], "fmm_lmp": [101.0, 202.0, 303.0],
                        "n15": [4, 3, 4]})                                # H1 has only 3 of its 4 intervals

    out = attach_settlement(rows, fmm, None)

    assert out["rt_settle"].tolist() == [101.0, 21.0, 303.0]
    assert out["settle_approx"].tolist() == [False, True, False]
    assert fmm["n15"].tolist() == [4, 3, 4] and len(fmm) == 3             # the caller's table is not modified


# ----------------------------------------------------------------------------- the CAISO puller and loader
def _caiso_answers(offset_by_node):
    """Responder for the stubbed `_fetch`: the same seven intervals for every node (one full hour and one with only
    3 of its 4 intervals), each price shifted by the node's offset so that the nodes can be told apart."""
    seven = [("2025-07-01 07:00", 10.0), ("2025-07-01 07:15", 20.0), ("2025-07-01 07:30", 30.0),
             ("2025-07-01 07:45", 40.0),
             ("2025-07-01 08:00", 1.0), ("2025-07-01 08:15", 2.0), ("2025-07-01 08:30", 6.0)]    # 08:45 is missing

    def respond(url):
        node = url.split("&node=")[1].split("&")[0]
        return _oasis_zip(node, _quarter_hours([(t, p + offset_by_node[node]) for t, p in seven]))

    return respond


def test_pull_caiso_fmm_hourly_mean_of_lmp_rows_only_and_partial_hours_stay_visible(tmp_path, monkeypatch, sleeps):
    import es.settlement as S
    urls = _serve(monkeypatch, _caiso_answers({NP15: 0.0, SP15: 100.0}))
    root = tmp_path / "caiso"

    path = S.pull_caiso_fmm("2025-07-01", "2025-07-01", root, nodes=(NP15, SP15))

    assert urls == [_caiso_url("20250701", "20250702", NP15), _caiso_url("20250701", "20250702", SP15)]
    assert sleeps == [5.5, 5.5]                                           # the OASIS rate-limit pause is kept
    assert path == root / "caiso_fmm_20250701_20250701.parquet"
    hourly = pd.read_parquet(path)
    assert hourly.columns.tolist() == ["node", "timestamp_utc", "fmm_lmp", "n15"]
    assert len(hourly) == 4
    hourly = hourly.set_index(["node", "timestamp_utc"])
    # only the LMP rows count: the MCC/MCE/MCL rows would make n15 16 and move the means
    assert hourly.loc[(NP15, H0)].tolist() == [25.0, 4]                   # mean of 10, 20, 30, 40
    assert hourly.loc[(NP15, H1)].tolist() == [3.0, 3]                    # mean of 1, 2, 6
    assert hourly.loc[(SP15, H0)].tolist() == [125.0, 4]
    assert hourly.loc[(SP15, H1)].tolist() == [103.0, 3]

    loaded = S.load_caiso_fmm(root)
    assert loaded.columns.tolist() == ["node", "timestamp_utc", "fmm_lmp"]
    # the 3-interval hours are dropped
    assert list(zip(loaded.node, loaded.timestamp_utc, loaded.fmm_lmp)) == [(NP15, H0, 25.0), (SP15, H0, 125.0)]

    rows = pd.DataFrame({"market": "CAISO", "node": [NP15, NP15, NP15, SP15, SP15],
                         "timestamp_utc": [H0, H1, H2, H0, H1], "da_lmp": 50.0,
                         "rt_lmp": [20.0, 21.0, 22.0, 23.0, 24.0]})
    out = S.attach_settlement(rows, loaded)
    assert out["rt_settle"].tolist() == [25.0, 21.0, 22.0, 125.0, 24.0]   # partial and absent hours keep rt_lmp ...
    assert out["settle_approx"].tolist() == [False, True, True, False, True]          # ... and are flagged
    raw_out = S.attach_settlement(rows, pd.read_parquet(path))            # the unfiltered hourly table says the same
    assert raw_out["rt_settle"].tolist() == out["rt_settle"].tolist()
    assert raw_out["settle_approx"].tolist() == out["settle_approx"].tolist()


def test_pull_caiso_fmm_removes_duplicate_intervals_and_counts_them(tmp_path, monkeypatch, sleeps):
    import es.settlement as S
    urls = _serve(monkeypatch, _caiso_answers({NP15: 0.0, SP15: 100.0}))      # each node's two chunks overlap fully
    root = tmp_path / "caiso"

    path = S.pull_caiso_fmm("2025-07-01", "2025-08-05", root, nodes=(NP15, SP15))

    assert urls == [_caiso_url("20250701", "20250731", NP15), _caiso_url("20250731", "20250806", NP15),
                    _caiso_url("20250701", "20250731", SP15), _caiso_url("20250731", "20250806", SP15)]
    assert sleeps == [5.5] * 4
    hourly = pd.read_parquet(path)
    assert hourly.n15.tolist() == [4, 3, 4, 3]                            # not doubled, and the nodes are not merged
    assert hourly.fmm_lmp.tolist() == [25.0, 3.0, 125.0, 103.0]
    cov = pd.read_csv(root / "coverage.csv")
    assert cov.columns.tolist() == ["node", "start", "rows15", "dropped_duplicates"]
    assert cov.to_dict("records") == [
        {"node": NP15, "start": "2025-07-01", "rows15": 7, "dropped_duplicates": 0},
        {"node": NP15, "start": "2025-07-31", "rows15": 7, "dropped_duplicates": 0},
        {"node": SP15, "start": "2025-07-01", "rows15": 7, "dropped_duplicates": 0},
        {"node": SP15, "start": "2025-07-31", "rows15": 7, "dropped_duplicates": 0},
        # the last row is drop_duplicates: 14 intervals kept (7 per node), 14 removed (each node's second chunk)
        {"node": "ALL", "start": "drop_duplicates", "rows15": 14, "dropped_duplicates": 14},
    ]
    chunks, final = cov.iloc[:-1], cov.iloc[-1]
    assert chunks.rows15.sum() == final["rows15"] + final["dropped_duplicates"]      # 28 parsed = 14 kept + 14 removed
    assert hourly.n15.sum() == final["rows15"]                            # the hourly table uses exactly the kept rows


# ----------------------------------------------------------------------------- the loaders and the seal
MARKET_CASES = [("CAISO", "load_caiso_fmm", "caiso_fmm", "fmm_lmp"),
                ("NYISO", "load_nyiso_integrated", "nyiso_rtint", "rt_int_lmp")]


def _write_settlement(directory, prefix, price_col, timestamps, market):
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"node": "A", "timestamp_utc": pd.to_datetime(timestamps, utc=True),
                          price_col: [float(k + 1) for k in range(len(timestamps))]})
    if market == "CAISO":
        frame["n15"] = 4
    frame.to_parquet(directory / f"{prefix}_20250101_20260630.parquet", index=False)
    return directory


@pytest.mark.parametrize("market, loader, prefix, price_col", MARKET_CASES)
def test_loaders_refuse_sealed_rows_unless_dev_only_is_false(tmp_path, market, loader, prefix, price_col):
    import es.settlement as S
    from es.guard import SealedDataError, first_sealed_utc
    load = getattr(S, loader)
    cut = first_sealed_utc(market)
    hour = pd.Timedelta(hours=1)

    dev = _write_settlement(tmp_path / "dev", prefix, price_col, [cut - 2 * hour, cut - hour], market)
    for kwargs in ({}, {"dev_only": True}, {"dev_only": False}):
        got = load(dev, **kwargs)
        assert got.columns.tolist() == ["node", "timestamp_utc", price_col] and len(got) == 2

    # one row exactly at the first sealed instant
    sealed = _write_settlement(tmp_path / "sealed", prefix, price_col, [cut - hour, cut], market)
    with pytest.raises(SealedDataError, match=market):
        load(sealed)                                                      # dev_only defaults to True
    with pytest.raises(SealedDataError, match=market):
        load(sealed, dev_only=True)
    assert len(load(sealed, dev_only=False)) == 2                         # reserved for the sealed entry point

    # a clean file next to a file with a sealed row
    mixed = _write_settlement(tmp_path / "mixed", prefix, price_col, [cut - hour], market)
    _write_settlement(tmp_path / "mixed", prefix + "_later", price_col, [cut + 5 * hour], market)
    with pytest.raises(SealedDataError):
        load(mixed)


def test_load_caiso_fmm_checks_the_seal_before_dropping_partial_hours(tmp_path):
    import es.settlement as S
    from es.guard import SealedDataError, first_sealed_utc
    cut = first_sealed_utc("CAISO")
    directory = tmp_path / "caiso"
    directory.mkdir()
    pd.DataFrame({"node": "A", "timestamp_utc": [cut - pd.Timedelta(hours=1), cut + pd.Timedelta(hours=3)],
                  "fmm_lmp": [1.0, 2.0], "n15": [4, 2]}).to_parquet(directory / "caiso_fmm_x.parquet", index=False)

    with pytest.raises(SealedDataError):                    # the sealed hour has n15 == 2, but it is still in the file
        S.load_caiso_fmm(directory)
    only_complete = S.load_caiso_fmm(directory, dev_only=False)
    assert only_complete.timestamp_utc.tolist() == [cut - pd.Timedelta(hours=1)]    # the n15 == 4 filter still applies


@pytest.mark.parametrize("market, loader, prefix, price_col", MARKET_CASES)
def test_loaders_return_none_without_files(tmp_path, market, loader, prefix, price_col):
    import es.settlement as S
    load = getattr(S, loader)
    (tmp_path / "empty").mkdir()
    for kwargs in ({}, {"dev_only": True}, {"dev_only": False}):
        assert load(tmp_path / "empty", **kwargs) is None
        assert load(tmp_path / "missing", **kwargs) is None
