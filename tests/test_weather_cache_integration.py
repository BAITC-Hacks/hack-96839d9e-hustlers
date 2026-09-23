"""Actual GFS points, read offline when the private local archive is present."""
import pytest

from windops.core import data_root, stamp
from windops.weather import cache_path, load_step, weather_for_run, validate_weather


def test_real_noaa_48_hour_weather_and_point_evidence():
    root = data_root()
    run = stamp("2026-01-31T00:00:00Z")
    if not all(cache_path(run, lead, root).exists() for lead in range(19, 67)):
        pytest.skip("Реальный GFS не собран: запустите windops.cli weather для 31.01.2026 23:00 +05:00.")
    bundle = weather_for_run("2026-01-31T23:00:00+05:00", root=root, offline=True)
    assert validate_weather(bundle)["valid"]
    assert len(bundle["rows"]) == 96
    assert {row["gfs_lead_hours"] for row in bundle["rows"]} == set(range(19, 67))
    assert min(row["target_time"] for row in bundle["rows"]) == "2026-01-31T19:00:00Z"
    assert max(row["target_time"] for row in bundle["rows"]) == "2026-02-02T18:00:00Z"
    for row in bundle["rows"]:
        assert row["provider"] == "NOAA"
        assert (row["grid_latitude"], row["grid_longitude"]) == (43.75, 78.5)
        assert 10 < row["distance_km"] < 15
    first = load_step(run, 19, root=root, offline=True)
    assert first["source_url"].startswith("https://noaa-gfs-bdp-pds.s3.amazonaws.com/")
    assert len(first["fields"]) == 5
    assert all(len(field["sha256"]) == 64 for field in first["fields"])
    assert first["points"]["turbine_1"]["temperature_2m_c"] == pytest.approx(-1.94181640625)
    again = weather_for_run("2026-01-31T23:00:00+05:00", root=root, offline=True)
    assert again["weather_version"] == bundle["weather_version"]
    assert again["retrieved_at"] == bundle["retrieved_at"]


def test_real_archive_rejects_unpublished_cycle_and_accepts_update():
    root = data_root()
    new_run = stamp("2026-02-01T00:00:00Z")
    future_run = stamp("2026-02-01T06:00:00Z")
    if not cache_path(future_run, 1, root).exists() or not all(cache_path(new_run, lead, root).exists() for lead in range(7, 55)):
        pytest.skip("Реальные данные сценария обновления не собраны.")
    from windops.core import BackendError
    with pytest.raises(BackendError, match="FUTURE_WEATHER"):
        weather_for_run("2026-02-01T06:00:00Z", run=future_run, root=root, offline=True)
    update = weather_for_run("2026-02-01T06:00:00Z", run=new_run, root=root, offline=True)
    assert len(update["rows"]) == 96
    assert stamp(update["available_at"]) <= stamp(update["forecast_origin"])
    assert {r["gfs_lead_hours"] for r in update["rows"]} == set(range(7, 55))
