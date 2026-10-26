import pytest


def test_project_root_resolves_locally():
    from es import config
    from src import paths
    assert config.REVISION_ROOT.name == "EnergySystems_Revision"
    assert paths.PROJECT_ROOT.resolve() == config.REVISION_ROOT.resolve()
    assert "CloudStorage" not in str(paths.DEV_ROOT)


def test_refuses_a_foreign_root(monkeypatch):
    from es import config
    monkeypatch.setenv("DART_PROJECT_ROOT", "/tmp")
    with pytest.raises(RuntimeError, match="DART_PROJECT_ROOT"):
        config.require_local_project_root()


def test_settlement_timezones_match_panel_prep():
    from es.config import SETTLE_TZ
    from v2.panel_prep import MARKET_TZ
    for market, tz in SETTLE_TZ.items():
        assert MARKET_TZ[market] == tz
