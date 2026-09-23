"""Explicit fixtures in pytest temporary directories; never production training data."""
from datetime import date, timedelta
import json

import numpy as np
import pandas as pd
import pytest

from windops.core import BackendError, SITES, digest, iso, stamp
from windops.ml.data import (CUTOFFS, LabelPolicy, SCADA_POWER, SCADA_TIME, aggregate_scada,
                             assert_disjoint, evaluation_rows, load_prepared, prepare, training_rows)
from windops.ml.features import FEATURE_SETS, bounded_predictions, prepare_features
from windops.ml.models import (baseline_prediction, fit_baselines, load_model, predict_model,
                               read_passport, train_artifact)
from windops.ml.plugin import get_model_metadata, predict_power
from windops.ml.training import comparison_metrics
from windops.ml.integration import february_table
from windops.ui.quality import load_backtest


def weather_rows(origin="2025-09-30T23:00:00+05:00", site="turbine_1", horizon=48):
    origin = stamp(origin)
    run = origin.replace(hour=0)
    return [{"site_id": site, "forecast_origin": iso(origin), "target_time": iso(origin + timedelta(hours=step)),
             "run_initialized_at": iso(run), "availability_upper_bound": iso(run + timedelta(hours=5)),
             "lead_hours": step, "gfs_lead_hours": int((origin + timedelta(hours=step) - run).total_seconds() / 3600),
             "provider": "NOAA", "provider_model": "gfs_pgrb2.0p25", "weather_version": "explicit-unit-test-only",
             "wind_speed_10m_ms": float(step % 10), "wind_speed_100m_ms": float(step % 10 + 1),
             "wind_u_10m_ms": float(step % 10), "wind_v_10m_ms": 0.,
             "wind_u_100m_ms": float(step % 10 + 1), "wind_v_100m_ms": 0., "temperature_2m_c": -5.}
            for step in range(1, horizon + 1)]


def raw_hour(start="2025-10-01 00:00", values=None):
    return pd.DataFrame({SCADA_TIME: pd.date_range(start, periods=6, freq="10min").astype(str),
                         SCADA_POWER: values if values is not None else [0, .1, .2, .3, .4, .5]})


def one_hour(raw, policy=LabelPolicy()):
    start = pd.Timestamp("2025-10-01T00:00:00+05:00").tz_convert("UTC")
    return aggregate_scada(raw, "turbine_1", policy, start=start, end=start + pd.Timedelta(hours=1))


def test_scada_local_start_and_delay_are_explicit():
    hourly, audit = one_hour(raw_hour(), LabelPolicy(scada_delay_minutes=17))
    row = hourly.iloc[0]
    assert row.target_time == pd.Timestamp("2025-09-30T19:00:00Z")
    assert row.actual == pytest.approx(.25)
    assert row.n_observations == row.n_valid_unique == 6 and row.complete
    assert row.interval_end == pd.Timestamp("2025-09-30T20:00:00Z")
    assert row.label_available_at == pd.Timestamp("2025-09-30T20:17:00Z")
    assert audit["zero_power_rows_retained"] == 1


def test_end_interval_labels_shift_ten_minutes():
    start, _ = one_hour(raw_hour())
    end, _ = one_hour(raw_hour("2025-10-01 00:10"), LabelPolicy(interval_label="end"))
    pd.testing.assert_frame_equal(start, end)


@pytest.mark.parametrize("case", ["missing", "duplicate", "out_of_range", "nan", "off_grid", "empty"])
def test_bad_or_incomplete_hour_has_no_answer(case):
    raw = raw_hour()
    if case == "missing":
        raw = raw.iloc[:5]
    elif case == "duplicate":
        raw = pd.concat([raw, raw.iloc[:1]])
    elif case == "out_of_range":
        raw.loc[1, SCADA_POWER] = 1.1
    elif case == "nan":
        raw.loc[1, SCADA_POWER] = np.nan
    elif case == "off_grid":
        raw.loc[1, SCADA_TIME] = "2025-10-01 00:11:00"
    else:
        raw = raw.iloc[:0]
    hourly, _ = one_hour(raw)
    assert not hourly.iloc[0].complete
    assert pd.isna(hourly.iloc[0].actual)


def test_zero_hour_retained_and_old_time_not_localized():
    raw = pd.concat([raw_hour(values=[0] * 6), pd.DataFrame({SCADA_TIME: ["2023-01-01 00:00:00", "bad"], SCADA_POWER: [1, 1]})])
    hourly, audit = one_hour(raw)
    assert hourly.iloc[0].actual == 0 and hourly.iloc[0].complete
    assert audit["outside_main_window_rows"] == audit["invalid_timestamp_rows"] == 1


@pytest.mark.parametrize("feature_set", ["weather_hour_lead", "weather_hour_lead_geometry"])
def test_features_ignore_future_measurements_and_keep_column_order(feature_set):
    rows = weather_rows()
    expected = prepare_features(rows, feature_set)
    augmented = [{**row, "actual": np.nan, "Средняя скорость ветра(m/s)": 1e20,
                  "retrieved_at": "2099-01-01", "ID": 234} for row in rows]
    pd.testing.assert_frame_equal(expected, prepare_features(augmented, feature_set))
    assert list(expected.columns) == list(FEATURE_SETS[feature_set])
    assert all(dtype == "float64" for dtype in expected.dtypes)
    assert expected.iloc[0].local_hour_sin == 0  # local midnight
    assert expected.iloc[0].lead_hours == 1 and expected.iloc[0].gfs_lead_hours == 19


@pytest.mark.parametrize("case", ["naive", "future", "lead", "duplicate", "infinite"])
def test_feature_contract_rejects_unusable_weather(case):
    rows = weather_rows()
    if case == "naive":
        rows[0]["target_time"] = "2025-10-01T00:00:00"
    elif case == "future":
        rows[0]["availability_upper_bound"] = "2026-01-01T00:00:00Z"
    elif case == "lead":
        rows[0]["gfs_lead_hours"] = 1
    elif case == "duplicate":
        rows[1] = rows[0]
    else:
        rows[0]["temperature_2m_c"] = float("inf")
    with pytest.raises(BackendError):
        prepare_features(rows)


def test_wind_table_empty_bins_rule_and_median():
    features = pd.DataFrame({"wind_speed_100m_ms": [1., 1.9, 3.]})
    baseline = fit_baselines(features, [0., .2, .8])
    assert baseline["constant_median"]["value"] == .2
    prediction = baseline_prediction("wind_table", baseline["wind_table"], pd.DataFrame({"wind_speed_100m_ms": [0., 2., 50.]}))
    np.testing.assert_equal(prediction, [.1, .1, .8])


def test_clip_is_identical_and_rejects_nonfinite_first():
    values, count = bounded_predictions([-.1, .4, 1.2])
    np.testing.assert_equal(values, [0, .4, 1])
    assert count == 2
    with pytest.raises(BackendError):
        bounded_predictions([float("inf")])


@pytest.fixture
def prepared_fixture(tmp_path):
    # Intentionally sparse test data, not genuine NOAA/SCADA and never data/.
    origins = ["2025-09-30", "2025-11-29", "2025-11-30", "2025-12-29", "2025-12-31", "2026-01-28", "2026-01-30"]
    weather = [row for site in SITES for origin in origins for row in weather_rows(origin + "T23:00:00+05:00", site)]
    (tmp_path / "scada").mkdir()
    (tmp_path / "weather").mkdir()
    pd.DataFrame(weather).to_csv(tmp_path / "weather/weather_for_ml.csv", index=False)
    for site in SITES:
        targets = sorted({r["target_time"] for r in weather if r["site_id"] == site})
        raw = []
        for target in targets:
            local = pd.Timestamp(target).tz_convert("Asia/Almaty").tz_localize(None)
            # Deliberately leave one January hour incomplete to test null export.
            points = 5 if str(local) == "2026-01-01 00:00:00" else 6
            for step in range(points):
                raw.append({SCADA_TIME: str(local + pd.Timedelta(minutes=10 * step)), SCADA_POWER: (local.hour % 10) / 10})
        pd.DataFrame(raw).to_csv(tmp_path / "scada" / f"{site}.csv", index=False)
    checksums = {str(p): digest(p.read_bytes()) for p in tmp_path.glob("scada/*.csv")}
    manifest = prepare(tmp_path, LabelPolicy())
    assert checksums == {str(p): digest(p.read_bytes()) for p in tmp_path.glob("scada/*.csv")}
    frame, _ = load_prepared(tmp_path)
    return tmp_path, frame, manifest


def test_split_boundaries_cutoff_and_overlapping_origins(prepared_fixture):
    _, frame, _ = prepared_fixture
    december = evaluation_rows(frame, "december")
    january = evaluation_rows(frame, "january")
    assert set(december.target_time.dt.tz_convert("Asia/Almaty").dt.month) == {12}
    assert set(january.target_time.dt.tz_convert("Asia/Almaty").dt.month) == {1}
    assert january.forecast_origin.min() == pd.Timestamp("2025-12-31T18:00:00Z")
    for stage, evaluation in (("selection_train", december), ("january", january)):
        train = training_rows(frame, stage)
        assert_disjoint(train, evaluation)
        assert (train.label_available_at <= CUTOFFS[stage]).all()
    train = training_rows(frame, "selection_train")
    assert train.target_time.max() == pd.Timestamp("2025-11-30T21:00:00+05:00")
    delayed = frame.copy()
    delayed["label_available_at"] += pd.Timedelta(minutes=1)
    assert training_rows(delayed, "selection_train").target_time.max() == pd.Timestamp("2025-11-30T20:00:00+05:00")
    assert frame.duplicated(["site_id", "target_time"]).any()  # overlapping releases retained
    assert len(frame) == 7 * 48 * 2  # no many-to-many multiplication
    with pytest.raises(BackendError, match="ML_TARGET_LEAKAGE"):
        assert_disjoint(train, train)


@pytest.mark.parametrize("kind", ["wind_table", "catboost"])
@pytest.mark.parametrize("feature_set", ["weather_hour_lead", "weather_hour_lead_geometry"])
def test_real_estimator_save_load_plugin_contract_on_fixtures(prepared_fixture, monkeypatch, kind, feature_set):
    root, frame, manifest = prepared_fixture
    models = root / "test_models"
    monkeypatch.setenv("WINDOPS_MODEL_DIR", str(models))
    monkeypatch.setenv("WINDOPS_DATA_DIR", str(root))
    monkeypatch.setenv("WINDOPS_ML_MODULE", "windops.ml.plugin")
    monkeypatch.setenv("WINDOPS_EXECUTION_MODE", "deterministic")
    config = {"kind": kind, "feature_set": feature_set}
    context = {"january_independent": False} if feature_set.endswith("_geometry") else None
    if kind == "catboost":
        config["params"] = {"iterations": 5, "depth": 2, "thread_count": 2, "random_seed": 2026,
                            "verbose": False, "allow_writing_files": False, "task_type": "CPU", "loss_function": "MAE"}
    train = training_rows(frame, "january")
    train = train.loc[train.site_id == "turbine_1"]
    card = train_artifact(train, config, site_id="turbine_1", purpose="explicit_test_fixture",
                          cutoff=CUTOFFS["january"], prepared=manifest, models=models, experiment_context=context)
    folder = models / "turbine_1" / card["version"]
    assert read_passport(folder)["save_load_predictions_identical"]
    repeated = train_artifact(train, config, site_id="turbine_1", purpose="explicit_test_fixture",
                              cutoff=CUTOFFS["january"], prepared=manifest, models=models, experiment_context=context)
    assert repeated == card
    with pytest.raises(BackendError, match="ML_MODEL_MISSING"):
        get_model_metadata(site_id="turbine_1", forecast_origin="2025-12-01T00:00:00Z")
    with pytest.raises(BackendError, match="FUTURE_MODEL"):
        predict_power(site_id="turbine_1", weather_rows=weather_rows(), model_version=card["version"])
    for horizon in (24, 48):
        weather = weather_rows("2026-01-31T23:00:00+05:00", horizon=horizon)
        metadata = get_model_metadata(site_id="turbine_1", forecast_origin=weather[0]["forecast_origin"])
        assert metadata["version"] == card["version"]
        if context:
            assert metadata["january_comparison_independent"] is False
            assert metadata["feature_version"] == "gfs-power-geometry-v2"
        else:
            assert metadata["feature_version"] == "gfs-power-v1"
        result = predict_power(site_id="turbine_1", weather_rows=weather, model_version=card["version"])
        expected, _ = predict_model(load_model(folder), config, weather)
        np.testing.assert_array_equal([r["prediction"] for r in result], expected)
        assert all(type(r["prediction"]) is float for r in result)
        reverse = predict_power(site_id="turbine_1", weather_rows=list(reversed(weather)), model_version=card["version"])
        assert reverse == list(reversed(result))
    # Actual trained estimator through participant 2 + BackendAdapter, with
    # explicitly fixture-only weather. This does not certify real NOAA replay.
    from test_backend import fixture_weather
    from windops.ui.adapter import BackendAdapter, load_bundle_json, validate_bundle
    from windops.ui.exports import ExportBlocked, selected_csv
    monkeypatch.setattr("windops.pipeline.weather_for_run", fixture_weather)
    adapter = BackendAdapter("windops.backend")
    for horizon in (24, 48):
        bundle = adapter.run_forecast(["turbine_1"], "2026-01-31T23:00:00+05:00", horizon)[0]
        assert validate_bundle(bundle).exportable
        assert selected_csv(bundle, final=True)
        again = adapter.run_forecast(["turbine_1"], "2026-01-31T23:00:00+05:00", horizon)[0]
        assert again.forecast_id == bundle.forecast_id
        uploaded = load_bundle_json((root / "forecasts" / bundle.forecast_id / "bundle.json").read_bytes())
        with pytest.raises(ExportBlocked):
            selected_csv(uploaded, final=True)
    artifact = folder / ("model.json" if kind == "wind_table" else "model.cbm")
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(BackendError, match="ML_CORRUPT_MODEL"):
        read_passport(folder)


def test_complete_training_workflow_is_frozen_and_ui_compatible(prepared_fixture, monkeypatch):
    from windops.ml import training
    root, _, _ = prepared_fixture
    models = root / "test_models"
    monkeypatch.setenv("WINDOPS_MODEL_DIR", str(models))
    monkeypatch.setenv("WINDOPS_DATA_DIR", str(root))
    # Bounded compatibility test of eval_set/early stopping, not a quality experiment.
    candidate = {**training.CATBOOST_CONFIGS[0], "params": {**training.CATBOOST_CONFIGS[0]["params"], "iterations": 8, "depth": 2}}
    monkeypatch.setattr(training, "CATBOOST_CONFIGS", [candidate])
    baseline = training.baseline_stage(root)
    assert set(baseline) == set(SITES)
    selection = training.select(root)
    assert selection == training.select(root)
    checksum = digest((root / "ml/selection.json").read_bytes())
    report = training.january_backtest(root)
    artifact = load_backtest((root / "ml/january_backtest.json").read_bytes())
    assert artifact.rows.actual.isna().sum() == 2
    assert len(artifact.rows) == 240  # two full Jan origins plus one truncated, per site
    assert all(value["mae"] is None or np.isfinite(value["mae"]) for value in report["metrics"])
    final = training.train_final(root)
    for site in SITES:
        assert final[site]["version"] != report["models"][site]
        assert get_model_metadata(site_id=site, forecast_origin="2025-12-01T00:00:00+05:00")["version"] == baseline[site]
        assert get_model_metadata(site_id=site, forecast_origin="2026-01-01T00:00:00+05:00")["version"] == report["models"][site]
        assert get_model_metadata(site_id=site, forecast_origin="2026-02-01T00:00:00+05:00")["version"] == final[site]["version"]
    assert digest((root / "ml/selection.json").read_bytes()) == checksum
    assert json.loads((root / "ml/january_metrics.json").read_text())["evaluation_models"] == report["models"]


def test_missing_inputs_are_explicit_and_do_not_create_artifacts(tmp_path):
    with pytest.raises(BackendError, match="scada/turbine_1.csv.*scada/turbine_2.csv.*weather/weather_for_ml.csv"):
        prepare(tmp_path, LabelPolicy())
    assert list(tmp_path.iterdir()) == []


def test_missing_backend_model_is_structured_cli_error(tmp_path, monkeypatch, capsys):
    from windops.ml.cli import main
    monkeypatch.setenv("WINDOPS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("WINDOPS_MODEL_DIR", str(tmp_path / "no_models"))
    monkeypatch.setenv("WINDOPS_ML_MODULE", "windops.ml.plugin")
    monkeypatch.setenv("WINDOPS_EXECUTION_MODE", "deterministic")
    monkeypatch.setenv("WINDOPS_OFFLINE", "1")
    assert main(["verify-backend"]) == 1
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "ML_BACKEND_VERIFICATION"
    assert "ML_MODEL_MISSING" in error["error"]["message"]


def test_baseline_comparisons_have_same_pairs_and_zero_denominator():
    rows = [{"site_id": "turbine_1", "horizon_step": step, "actual": actual, "prediction": .1,
             "actual_valid": actual is not None, "prediction_valid": True, "actual_reason": "missing_hour",
             "baselines": {"constant_median": 0., "wind_table": 0.}} for step, actual in ((1, 0.), (2, None), (25, 0.))]
    metrics = comparison_metrics(rows)
    overall = [m for m in metrics if m["horizon_band"] == "all"]
    assert {m["n_used"] for m in overall} == {2}
    assert {m["n_excluded"] for m in overall} == {1}
    assert all(m["model_mae_improvement_pct"] is None for m in overall)


def export_fixture(make_bundle):
    result = []
    for day in pd.date_range("2026-01-31", "2026-02-28", freq="D"):
        for site in SITES:
            item = make_bundle(sites=(site,), origin=f"{day:%Y-%m-%d}T23:00:00+05:00", forecast_id=f"fixture-{site}-{day:%d}")
            result.append({"rows": item.rows.to_dict("records"), "forecast_id": item.forecast_id, "horizon_hours": 48})
    return result


def test_february_export_exact_local_grid_and_full_march_hours(make_bundle):
    table, full = february_table(export_fixture(make_bundle))
    assert len(table) == 1344 and len(full) == 2784
    assert table.target_time_local.min() == "2026-02-01T00:00:00+05:00"
    assert table.target_time_local.max() == "2026-02-28T23:00:00+05:00"
    assert full.target_time_local.max() == "2026-03-02T23:00:00+05:00"


@pytest.mark.parametrize("case", ["duplicate", "missing", "shift", "update"])
def test_february_export_rejects_wrong_grid_and_updates(make_bundle, case):
    bundles = export_fixture(make_bundle)
    if case == "duplicate":
        bundles[0] = bundles[1]
    elif case == "missing":
        bundles.pop()
    elif case == "shift":
        bundles[0]["rows"][0]["target_time"] = bundles[0]["rows"][1]["target_time"]
    else:
        bundles[0]["revises_forecast_id"] = "old"
    with pytest.raises(BackendError):
        february_table(bundles)


def test_real_ml_backend_when_private_data_are_available():
    from windops.core import data_root
    from windops.ml.models import model_root
    import os
    if os.environ.get("WINDOPS_RUN_REAL_ML_TEST") != "1":
        pytest.skip("Реальная ML-проверка требует данных/моделей и WINDOPS_RUN_REAL_ML_TEST=1; fixtures не заменяют её.")
    assert (data_root() / "ml/january_backtest.json").exists()
    assert list(model_root().glob("*/ml-*/passport.json"))
    from windops.ml.integration import verify_backend
    result = verify_backend(data_root())
    assert result["replay_releases"] == 58
    assert result["february_export"]["hourly_rows"] == 1344
