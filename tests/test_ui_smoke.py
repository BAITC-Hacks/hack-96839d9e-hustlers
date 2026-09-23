import json
from datetime import date
from io import BytesIO
import os
from pathlib import Path
import sys
import types

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from windops.ui.adapter import load_bundle_json, validate_bundle
from windops.ui.exports import ExportBlocked, selected_csv


APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("WINDOPS_BACKEND_MODULE", raising=False)
    monkeypatch.setenv("WINDOPS_CONFIG", str(tmp_path / "missing-config.json"))


def test_app_without_backend_is_empty_and_demo_off(isolated_environment):
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    assert app.toggle(key="demo_mode").value is False
    assert app.button(key="run_forecast").disabled
    assert not app.session_state["history"]
    assert len(app.tabs) == 3
    assert any("Фактическая мощность за февраль" in item.value for item in app.info)
    assert any("backtest пока не загружен" in item.value for item in app.info)


def test_development_comparison_is_not_presented_as_independent(isolated_environment):
    from test_ui_charts_quality import backtest_payload
    from windops.ui.quality import load_backtest
    payload = backtest_payload()
    payload["provenance"]["evaluation_independent"] = False
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.session_state["quality_artifact"] = load_backtest(payload)
    app.run()
    assert not app.exception
    assert any("Это не новая независимая проверка" in item.value for item in app.warning)


def test_explicit_demo_load_and_draft_parameters_do_not_replace_result(isolated_environment):
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.toggle(key="demo_mode").set_value(True).run()
    app.button(key="load_demo").click().run()
    assert not app.exception
    selected = app.session_state["selected_id"]
    bundle = app.session_state["history"][selected]
    assert bundle.source_mode == "demo"
    # The persistent banner is checked in the browser acceptance script. AppTest
    # verifies the actual trust gate independently of the rendered component type.
    with pytest.raises(ExportBlocked):
        selected_csv(bundle, final=True)
    assert len(bundle.rows) == 96
    app.radio(key="horizon").set_value(24).run()
    assert not app.exception
    assert app.session_state["selected_id"] == selected
    assert app.session_state["history"][selected].horizon_hours == 48
    app.toggle(key="demo_mode").set_value(False).run()
    assert not app.exception
    assert any("Подготовьте первый выпуск" in item.value for item in app.markdown)


def test_second_demo_turbine_keeps_label_and_color(isolated_environment):
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.toggle(key="demo_mode").set_value(True).run()
    app.selectbox(key="site_choice").set_value("demo-2")
    app.button(key="load_demo").click().run()
    assert not app.exception
    spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert spec["data"][0]["name"] == "Турбина 2 · Прогноз"
    assert spec["data"][0]["line"]["color"] == "#2DD4BF"


def test_backend_calls_only_on_submit_and_failure_is_visible(monkeypatch, tmp_path, make_bundle):
    calls = []
    bridge = types.ModuleType("windops_test_bridge")

    def run_forecast(*, site_id, forecast_origin, horizon_hours):
        calls.append((site_id, forecast_origin, horizon_hours))
        if len(calls) > 1:
            raise RuntimeError("Backend test failure")
        return make_bundle(sites=(site_id,), origin=forecast_origin, horizon=horizon_hours)

    bridge.run_forecast = run_forecast
    monkeypatch.setitem(sys.modules, bridge.__name__, bridge)
    monkeypatch.setenv("WINDOPS_BACKEND_MODULE", bridge.__name__)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"timezone": "UTC", "sites": [{"site_id": "site-a"}],
        "normalization": "0_1", "timestamp_convention": "target_time = issued_at + horizon_step hours",
        "default_origin": "2026-01-31T23:00:00Z"}), encoding="utf-8")
    monkeypatch.setenv("WINDOPS_CONFIG", str(path))
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    assert not calls
    app.button(key="run_forecast").click().run()
    assert not app.exception
    assert len(calls) == 1
    selected = app.session_state["selected_id"]
    selected_csv(app.session_state["history"][selected], final=True)
    app.run()
    assert not app.exception
    assert len(calls) == 1
    app.radio(key="horizon").set_value(24).run()
    assert len(calls) == 1
    app.button(key="run_forecast").click().run()
    assert not app.exception
    assert len(calls) == 2
    assert app.session_state["status"] == "error"
    assert app.error
    assert any("Предыдущий успешный выпуск" in item.value for item in app.warning)


def test_malformed_saved_result_shows_repairable_state(isolated_environment, make_bundle):
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    bundle = make_bundle()
    bundle.rows = bundle.rows.drop(columns=["site_id", "prediction"])
    bundle.verification_origin = "uploaded_unverified"
    app.session_state["history"] = {bundle.forecast_id: bundle}
    app.session_state["selected_id"] = bundle.forecast_id
    app.run()
    assert not app.exception
    assert app.error or app.warning


def configure_bridge(monkeypatch, tmp_path, bridge, sites):
    monkeypatch.setitem(sys.modules, bridge.__name__, bridge)
    monkeypatch.setenv("WINDOPS_BACKEND_MODULE", bridge.__name__)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"timezone": "UTC", "sites": [{"site_id": site} for site in sites],
        "normalization": "0_1", "timestamp_convention": "target_time = issued_at + horizon_step hours",
        "default_origin": "2026-01-31T23:00:00Z"}), encoding="utf-8")
    monkeypatch.setenv("WINDOPS_CONFIG", str(path))


def test_both_turbines_preserve_backend_ids_and_unchanged_updates(monkeypatch, tmp_path, make_bundle):
    calls, update_calls = [], []
    bridge = types.ModuleType("windops_two_sites_test_bridge")

    def run_forecast(*, site_id, forecast_origin, horizon_hours):
        calls.append((site_id, forecast_origin, horizon_hours))
        return make_bundle(sites=(site_id,), origin=forecast_origin, horizon=horizon_hours,
                           forecast_id=f"backend-{site_id}")

    def check_updates(*, forecast_id):
        update_calls.append(forecast_id)
        return []

    bridge.run_forecast, bridge.check_updates = run_forecast, check_updates
    configure_bridge(monkeypatch, tmp_path, bridge, ["site-a", "site-b"])
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    assert app.selectbox(key="site_choice").value == "__both__"
    assert calls == update_calls == []
    app.button(key="run_forecast").click().run()
    assert not app.exception
    assert [call[0] for call in calls] == ["site-a", "site-b"]
    selected = app.session_state["selected_id"]
    bundle = app.session_state["history"][selected]
    assert len(bundle.rows) == 96
    assert set(bundle.site_ids) == {"site-a", "site-b"}
    assert {member.forecast_id for member in bundle.members} == {"backend-site-a", "backend-site-b"}
    exported = pd.read_csv(BytesIO(selected_csv(bundle, final=True)))
    assert set(exported.forecast_id) == {"backend-site-a", "backend-site-b"}
    assert set(exported.run_id) == {"backend-site-a-run", "backend-site-b-run"}
    assert len(exported) == 96
    versions = set(app.session_state["history"])
    app.run()
    assert not app.exception
    assert len(calls) == 2
    assert not update_calls
    app.button(key="check_updates").click().run()
    assert not app.exception
    assert set(update_calls) == {"backend-site-a", "backend-site-b"}
    assert len(update_calls) == 2
    assert len(calls) == 2
    assert set(app.session_state["history"]) == versions
    assert app.session_state["selected_id"] == selected
    assert any("Новых входных данных нет" in element.value for element in app.info)
    app.run()
    assert len(calls) == len(update_calls) == 2


def test_partial_backend_replay_shows_errors_and_blocks_final_export(monkeypatch, tmp_path, make_bundle):
    calls = []
    bridge = types.ModuleType("windops_partial_replay_test_bridge")

    def run_replay(*, start_date, end_date):
        calls.append((start_date, end_date))
        return {"bundles": [make_bundle(sites=("site-a",), forecast_id="replay-success")],
                "errors": [{"issued_at": "2026-02-02T23:00:00Z", "error": "Тестовая недоступность архива"}]}

    bridge.run_replay = run_replay
    configure_bridge(monkeypatch, tmp_path, bridge, ["site-a"])
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    assert not calls
    button = next(button for button in app.button if button.label == "Запустить replay")
    button.click().run()
    assert not app.exception
    # The first February targets require the January 31 release at 23:00 local.
    assert calls == [(date(2026, 1, 31), date(2026, 2, 28))]
    replay = app.session_state["replay_result"]
    assert len(replay.bundles) == len(replay.errors) == 1
    assert any("Ошибок: 1" in item.value for item in app.markdown)
    assert any("Выпуск не выполнен" in item.value and "2026-02-02T23:00:00Z" in item.value for item in app.text)
    assert not any("Тестовая недоступность архива" in item.value for item in app.text)
    downloads = app.get("download_button")
    final = next(item for item in downloads if item.proto.label == "Все версии replay · итоговый CSV")
    assert final.proto.disabled
    app.checkbox(key="february_rule").set_value(True).run()
    assert any("replay содержит ошибки" in item.value for item in app.warning)
    assert not any(item.proto.label == "Февраль · один прогноз на час" for item in app.get("download_button"))
    app.run()
    assert not app.exception
    assert len(calls) == 1


def test_real_saved_artifact_when_provided():
    artifact = os.environ.get("WINDOPS_SMOKE_BUNDLE")
    if not artifact:
        pytest.skip("Реальный backend/проверенный сохранённый выпуск не предоставлен; WINDOPS_SMOKE_BUNDLE не задан.")
    bundle = load_bundle_json(Path(artifact).read_bytes())
    assert bundle.source_mode != "demo"
    result = validate_bundle(bundle)
    assert result.valid, result.issues
    for name in ("weather_available", "training_cutoff", "archived_weather", "inputs_verified", "model_verified"):
        assert result.checks.get(name), result.issues
    # A local artifact can be inspected but cannot grant itself backend trust.
    assert bundle.verification_origin == "uploaded_unverified"
    assert not result.exportable
