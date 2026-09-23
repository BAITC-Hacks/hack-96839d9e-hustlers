import json

import pytest

from windops.ui.configuration import load_configuration


def config():
    return {"timezone": "UTC", "sites": [{"site_id": "real-id-from-config"}],
            "normalization": "0_1", "default_origin": "2026-01-31T23:00:00+00:00",
            "timestamp_convention": "target_time = issued_at + horizon_step hours"}


def test_absent_configuration_does_not_guess_zone_or_sites(tmp_path, monkeypatch):
    monkeypatch.setenv("WINDOPS_CONFIG", str(tmp_path / "absent.json"))
    result = load_configuration()
    assert result.issues
    assert result.timezone is None
    assert not result.sites


def test_real_site_ids_preserved(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config()), encoding="utf-8")
    monkeypatch.setenv("WINDOPS_CONFIG", str(path))
    result = load_configuration()
    assert not result.issues
    assert list(result.sites) == ["real-id-from-config"]
    assert result.default_origin.utcoffset().total_seconds() == 0


@pytest.mark.parametrize("field,value", [("timezone", "NoSuch/Zone"), ("sites", []),
    ("sites", [{"site_id": "x"}, {"site_id": "x"}]), ("sites", [{"site_id": ""}]),
    ("default_origin", "2026-01-31T23:00:00"), ("normalization", "MW"),
    ("timestamp_convention", "unknown")])
def test_malformed_configuration_is_reported(tmp_path, monkeypatch, field, value):
    payload = config()
    payload[field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("WINDOPS_CONFIG", str(path))
    assert load_configuration().issues
