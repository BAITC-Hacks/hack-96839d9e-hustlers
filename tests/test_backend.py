"""Backend failure boundaries and UI handoff using explicit test-only fixtures.

No fixture is saved to data/. A separately marked cache test reads genuine GFS.
"""
from datetime import date, datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from windops.agent import TOOLS, execute_tool, run_agent
from windops.core import BackendError, SITES, WEATHER_SCHEMA, atomic_json, digest, iso, stamp, daily_origins
from windops.ml_bridge import MLPredictor, validate_metadata, validate_predictions
from windops.pipeline import ForecastService, deterministic, compare_forecasts
from windops.ui.adapter import BackendAdapter, _bundle, validate_bundle
from windops.ui.charts import weather_chart
from windops.ui.exports import selected_csv
from windops.weather import FIELDS, _get, cache_path, index_ranges, load_step, object_url, validate_weather, weather_for_run

ORIGIN = stamp("2026-01-31T18:00:00Z")


def fixture_weather(origin=ORIGIN, horizon=48, **kwargs):
    origin = stamp(origin)
    run = kwargs.get("run") or origin.replace(hour=0)
    available = run + timedelta(hours=5)
    version = "test-weather-" + iso(run) + "-" + iso(origin)
    rows = []
    for site in SITES:
        for step in range(1, horizon + 1):
            rows.append({"site_id": site, "forecast_origin": iso(origin), "target_time": iso(origin + timedelta(hours=step)),
                         "run_initialized_at": iso(run), "availability_upper_bound": iso(available),
                         "availability_basis": "test fixture only", "lead_hours": step,
                         "gfs_lead_hours": int((origin + timedelta(hours=step) - run).total_seconds() / 3600),
                         "provider": "test-only", "provider_model": "gfs_pgrb2.0p25", "weather_version": version,
                         "temperature_2m_c": -5., "wind_u_10m_ms": 0., "wind_v_10m_ms": 0.,
                         "wind_u_100m_ms": 0., "wind_v_100m_ms": 0., "wind_speed_10m_ms": 0.,
                         "wind_speed_100m_ms": 0., "wind_direction_10m_deg": 0., "wind_direction_100m_deg": 0.})
    return {"weather_version": version, "schema_version": WEATHER_SCHEMA, "forecast_origin": iso(origin),
            "run_initialized_at": iso(run), "horizon_hours": horizon, "available_at": iso(available),
            "availability_basis": "test only", "retrieved_at": "2026-09-23T00:00:00Z", "rows": rows, "evidence": []}


class FixturePredictor:
    def metadata(self, site_id, origin):
        return {"site_id": site_id, "version": "test-only-model", "training_cutoff": "2025-12-31T00:00:00Z",
                "normalization": "0_1", "weather_schema": WEATHER_SCHEMA, "provider_model": "gfs_pgrb2.0p25",
                "labels_available_by_cutoff": True}

    def predict(self, site_id, weather, metadata):
        return [{"target_time": row["target_time"], "prediction": 0.0} for row in weather["rows"] if row["site_id"] == site_id]


def service(tmp_path, origin=ORIGIN, **kwargs):
    return ForecastService("turbine_1", origin, root=tmp_path, predictor=FixturePredictor(), weather_loader=fixture_weather, **kwargs)


def test_local_midnight_and_gfs_lead_are_distinct():
    origin = next(daily_origins(date(2026, 1, 31), date(2026, 1, 31)))
    assert origin == ORIGIN
    first = origin + timedelta(hours=1)
    assert first == stamp("2026-02-01T00:00:00+05:00")
    assert (first - origin.replace(hour=0)).total_seconds() / 3600 == 19
    assert object_url(origin.replace(hour=0), 19).endswith("gfs.t00z.pgrb2.0p25.f019")


def test_naive_time_rejected():
    with pytest.raises(BackendError, match="INVALID_TIME"):
        stamp("2026-01-31T23:00:00")


def test_empty_environment_keeps_data_in_ignored_folder(monkeypatch):
    from windops.core import ROOT, data_root
    monkeypatch.setenv("WINDOPS_DATA_DIR", "")
    assert data_root() == ROOT / "data"


def test_zero_wind_valid_and_future_rejected():
    weather = fixture_weather()
    assert validate_weather(weather)["valid"]
    weather["rows"][0]["availability_upper_bound"] = iso(ORIGIN + timedelta(seconds=1))
    weather["available_at"] = weather["rows"][0]["availability_upper_bound"]
    with pytest.raises(BackendError, match="FUTURE_WEATHER"):
        validate_weather(weather)


@pytest.mark.parametrize("change", ["missing", "duplicate", "nan", "provider", "origin", "direction"])
def test_bad_weather_fails(change):
    weather = fixture_weather()
    if change == "missing":
        weather["rows"].pop()
    elif change == "duplicate":
        weather["rows"][-1] = weather["rows"][0]
    elif change == "nan":
        weather["rows"][0]["temperature_2m_c"] = float("nan")
    elif change == "provider":
        weather["rows"][0]["provider_model"] = "ecmwf"
    elif change == "direction":
        weather["rows"][0]["wind_direction_10m_deg"] = 360.
    else:
        weather["rows"][0]["forecast_origin"] = "2026-02-01T18:00:00Z"
    with pytest.raises(BackendError):
        validate_weather(weather)


@pytest.mark.parametrize("field,value", [("training_cutoff", "2026-02-01T00:00:00Z"),
                                         ("weather_schema", "ecmwf"), ("provider_model", "ecmwf"),
                                         ("labels_available_by_cutoff", False)])
def test_bad_model_metadata_rejected(field, value):
    metadata = FixturePredictor().metadata("turbine_1", ORIGIN)
    metadata[field] = value
    with pytest.raises(BackendError):
        validate_metadata(metadata, "turbine_1", ORIGIN)


@pytest.mark.parametrize("value", [True, None, -0.1, 1.01, float("nan"), float("inf")])
def test_invalid_power_never_clipped(value):
    weather = [{"target_time": "2026-01-31T19:00:00Z"}]
    with pytest.raises(BackendError, match="INVALID_PREDICTION"):
        validate_predictions([{**weather[0], "prediction": value}], weather)


def test_prediction_time_and_quantile_checks():
    weather = [{"target_time": "2026-01-31T19:00:00Z"}]
    with pytest.raises(BackendError, match="PREDICTION_GRID"):
        validate_predictions([{"target_time": [], "prediction": 0.2}], weather)
    with pytest.raises(BackendError, match="PREDICTION_GRID"):
        validate_predictions([{"target_time": "2026-01-31T20:00:00Z", "prediction": 0.2}], weather)
    with pytest.raises(BackendError, match="INVALID_PREDICTION"):
        validate_predictions([{**weather[0], "prediction": 0.2, "p10": 0.7, "p90": 0.1}], weather)


def test_missing_model_does_not_use_fake(monkeypatch):
    monkeypatch.delenv("WINDOPS_ML_MODULE", raising=False)
    with pytest.raises(BackendError, match="MODEL_NOT_CONFIGURED"):
        MLPredictor()


def test_fixture_pipeline_ui_handoff_and_idempotency(tmp_path):
    first = deterministic(service(tmp_path))
    again = deterministic(service(tmp_path))
    assert first["forecast_id"] == again["forecast_id"]
    assert len(list((tmp_path / "forecasts").glob("*/bundle.json"))) == 1
    bundle = _bundle(first, trusted=True)
    validation = validate_bundle(bundle)
    assert validation.exportable, validation.issues
    assert len(bundle.rows) == 48
    assert selected_csv(bundle, final=True)
    assert len(weather_chart(bundle.weather, "Asia/Almaty", {"turbine_1": "Турбина 1"}).data) == 3
    assert all("recorded_at" in event for event in first["events"])


def test_agent_rejects_extra_arguments_and_invented_ids(tmp_path):
    engine = service(tmp_path)
    result = execute_tool(engine, "fetch_weather", '{"candidate_index":0,"shell":"echo unsafe"}')
    assert result["error"]["code"] == "INVALID_ARGUMENTS"
    assert not engine.weather
    result = execute_tool(engine, "predict_power", '{"weather_id":"invented"}')
    assert result["error"]["code"] == "UNKNOWN_WEATHER"
    assert execute_tool(engine, "exec", '{}')["error"]["code"] == "UNKNOWN_TOOL"


def test_bool_candidate_rejected(tmp_path):
    assert execute_tool(service(tmp_path), "fetch_weather", '{"candidate_index":true}')["status"] == "error"


def test_no_key_has_explicit_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = run_agent(service(tmp_path), mode="auto")
    assert result["executor"] == "deterministic_fallback"
    assert any(e["status"] == "unavailable" for e in result["events"])
    with pytest.raises(BackendError, match="MISSING_API_KEY"):
        run_agent(service(tmp_path), mode="agent")


def test_responses_loop_uses_tool_ids_and_preserves_output(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "test-only")
    engine = service(tmp_path)
    requests = []
    reasoning = SimpleNamespace(type="reasoning")

    def create(**kwargs):
        requests.append(kwargs)
        assert kwargs["store"] is False and kwargs["parallel_tool_calls"] is False
        step = len(requests)
        if step == 1:
            name, args = "fetch_weather", {"candidate_index": 0}
        elif step == 2:
            assert reasoning in kwargs["input"]
            name, args = "predict_power", {"weather_id": next(iter(engine.weather))}
        else:
            name, args = "publish_forecast", {"prediction_id": next(iter(engine.predictions))}
        return SimpleNamespace(output=[reasoning, SimpleNamespace(type="function_call", name=name, arguments=json.dumps(args), call_id=str(step))])

    result = run_agent(engine, client=SimpleNamespace(responses=SimpleNamespace(create=create)), mode="agent")
    assert result["executor"] == "openai_responses"
    assert len(requests) == 3
    assert "reasoning" not in json.dumps(result["events"])


def test_agent_cannot_claim_success_without_publishing(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "test-only")
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(output=[])))
    with pytest.raises(BackendError, match="AGENT_INCOMPLETE"):
        run_agent(service(tmp_path), client=client, mode="agent")
    assert not list(tmp_path.glob("forecasts/*/bundle.json"))


def test_agent_loop_is_bounded_and_reports_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "test-only")
    calls = []
    def create(**kwargs):
        calls.append(1)
        return SimpleNamespace(output=[SimpleNamespace(type="function_call", name="predict_power",
                                  arguments='{"weather_id":"invented"}', call_id=str(len(calls)))])
    engine = service(tmp_path)
    with pytest.raises(BackendError, match="AGENT_INCOMPLETE"):
        run_agent(engine, client=SimpleNamespace(responses=SimpleNamespace(create=create)), mode="agent")
    assert len(calls) == 8
    assert engine.bundle is None
    assert any(event.get("error_code") == "UNKNOWN_WEATHER" for event in engine.events)


def test_modified_saved_forecast_is_not_trusted(tmp_path):
    bundle = deterministic(service(tmp_path))
    path = tmp_path / "forecasts" / bundle["forecast_id"] / "bundle.json"
    bundle["rows"][0]["prediction"] = 0.9
    path.write_text(json.dumps(bundle))
    with pytest.raises(BackendError, match="CORRUPT_FORECAST"):
        deterministic(service(tmp_path))


def test_full_body_rejected_before_reading(monkeypatch):
    import windops.weather as weather
    class Response:
        status_code = 200
        headers = {"Content-Length": "500000000"}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_content(self, size): raise AssertionError("must not read whole globe")
    monkeypatch.setattr(weather, "session", lambda: SimpleNamespace(get=lambda *args, **kwargs: Response()))
    with pytest.raises(BackendError, match="HTTP_STATUS"):
        _get("https://example.invalid", expected_status=206)


def test_index_requires_exact_fields_and_hour():
    run = ORIGIN.replace(hour=0)
    lines = [f"{i+1}:{i*100}:d=2026013100:{var}:{level}:19 hour fcst:" for i, (var, level) in enumerate(FIELDS)]
    lines.append("6:500:d=2026013100:OTHER:surface:19 hour fcst:")
    assert len(index_ranges("\n".join(lines), run, 19)) == 5
    with pytest.raises(BackendError, match="WRONG_GFS_TIME"):
        index_ranges("\n".join(lines), run, 20)
    with pytest.raises(BackendError, match="MISSING_FIELDS"):
        index_ranges("\n".join(lines[1:]), run, 19)


def test_tampered_cache_rejected_without_network(tmp_path):
    run = ORIGIN.replace(hour=0)
    path = cache_path(run, 19, tmp_path)
    atomic_json(path, {"sha256": "bad", "payload": {}})
    with pytest.raises(BackendError, match="CORRUPT_CACHE"):
        load_step(run, 19, root=tmp_path, offline=True)


def test_update_keeps_old_forecast_and_compares_shared_hours(tmp_path, monkeypatch):
    from windops import backend
    monkeypatch.setenv("WINDOPS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("WINDOPS_EXECUTION_MODE", "deterministic")
    old = deterministic(service(tmp_path))
    monkeypatch.setattr(backend, "ForecastService", lambda site_id, origin, horizon, **kw: service(tmp_path, origin))
    updates = backend.check_updates(old["forecast_id"], as_of="2026-02-01T06:00:00Z")
    assert len(updates) == 1
    new = updates[0]
    assert new["forecast_id"] != old["forecast_id"]
    assert new["revises_forecast_id"] == old["forecast_id"]
    assert new["comparison"]["overlap_hours"] == 36
    assert json.loads((tmp_path / "forecasts" / old["forecast_id"] / "bundle.json").read_text())["rows"] == old["rows"]


def test_same_cycle_does_not_create_update(tmp_path, monkeypatch):
    from windops import backend
    monkeypatch.setenv("WINDOPS_DATA_DIR", str(tmp_path))
    old = deterministic(service(tmp_path))
    monkeypatch.setattr(backend, "ForecastService", lambda site_id, origin, horizon, **kw: service(tmp_path, origin))
    assert backend.check_updates(old["forecast_id"], as_of="2026-01-31T19:00:00Z") == []


def test_real_backend_missing_model_is_actionable_in_ui(tmp_path, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("WINDOPS_BACKEND_MODULE", "windops.backend")
    monkeypatch.setenv("WINDOPS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("WINDOPS_ML_MODULE", raising=False)
    config = tmp_path / "config.json"
    atomic_json(config, {"timezone": "Asia/Almaty", "sites": [{"site_id": "turbine_1"}],
                         "normalization": "0_1", "timestamp_convention": "target_time = issued_at + horizon_step hours",
                         "default_origin": "2026-01-31T23:00:00+05:00"})
    monkeypatch.setenv("WINDOPS_CONFIG", str(config))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
    assert not app.exception
    app.button(key="run_forecast").click().run()
    assert not app.exception
    assert any("MODEL_NOT_CONFIGURED" in item.value for item in app.error)
    assert not app.session_state["history"]
    error = json.loads((tmp_path / "data/reports/last_forecast_error.json").read_text())
    assert error["error"]["code"] == "MODEL_NOT_CONFIGURED"
