"""Exercise participant 2's actual backend and export its canonical daily releases."""
from __future__ import annotations

from datetime import date
import json

import pandas as pd

from windops.core import BackendError, SITES, atomic_json, digest, load_forecast, read_json, stamp
from .data import write_csv


def february_table(bundles):
    """D uses steps 1–24 of original daily D−1 23:00, never an update."""
    selected, all_rows = [], []
    seen = set()
    expected_origins = set(pd.date_range("2026-01-31 23:00", "2026-02-28 23:00", freq="D", tz="Asia/Almaty").tz_convert("UTC"))
    for bundle in bundles:
        if bundle.get("revises_forecast_id"):
            raise BackendError("ML_EXPORT_UPDATE", "В ежедневный экспорт передано внутридневное обновление.")
        if bundle.get("horizon_hours") != 48 or len(bundle["rows"]) != 48:
            raise BackendError("ML_EXPORT_GRID", "Экспорт требует полные исходные 48-часовые выпуски одной турбины.")
        frame = pd.DataFrame(bundle["rows"])
        origin = pd.Timestamp(stamp(frame.issued_at.iloc[0]))
        sites = set(frame.site_id)
        if len(sites) != 1 or not sites.issubset(SITES) or origin not in expected_origins:
            raise BackendError("ML_EXPORT_GRID", "Неожиданный выпуск или турбина.")
        site = next(iter(sites))
        key = (site, origin)
        if key in seen:
            raise BackendError("ML_EXPORT_GRID", "Повтор ежедневного выпуска.")
        seen.add(key)
        targets = pd.to_datetime(frame.target_time, utc=True)
        steps = pd.to_numeric(frame.horizon_step)
        if (set(steps) != set(range(1, 49)) or
                not targets.eq(origin + pd.to_timedelta(steps, unit="h")).all() or
                not pd.to_datetime(frame.issued_at, utc=True).eq(origin).all()):
            raise BackendError("ML_EXPORT_GRID", "Пропуск, дубль или неверная привязка часов/шага.")
        frame["forecast_id"] = bundle["forecast_id"]
        frame["forecast_origin"] = frame.issued_at
        frame["target_time_local"] = targets.dt.tz_convert("Asia/Almaty").map(lambda t: t.isoformat())
        all_rows.append(frame)
        february = targets.dt.tz_convert("Asia/Almaty").dt.month.eq(2)
        selected.append(frame.loc[steps.le(24) & february])
    if seen != {(site, origin) for site in SITES for origin in expected_origins}:
        raise BackendError("ML_EXPORT_GRID", "Нужны все 58 исходных выпусков 31 января–28 февраля.")
    table = pd.concat(selected, ignore_index=True)
    targets = pd.to_datetime(table.target_time, utc=True)
    expected_targets = pd.date_range("2026-02-01", "2026-03-01", freq="h", inclusive="left", tz="Asia/Almaty").tz_convert("UTC")
    actual_keys = set(zip(table.site_id, targets))
    if len(table) != 1344 or actual_keys != {(site, time) for site in SITES for time in expected_targets}:
        raise BackendError("ML_EXPORT_GRID", "Февральская сетка должна содержать каждый из 1344 уникальных часов.")
    if table.duplicated(["site_id", "target_time"]).any():
        raise BackendError("ML_EXPORT_GRID", "Повтор февральского часа.")
    return table.sort_values(["site_id", "target_time"]), pd.concat(all_rows, ignore_index=True).sort_values(["site_id", "issued_at", "target_time"])


def export_february(root):
    report = read_json(root / "reports" / "replay.json")
    if report.get("errors"):
        raise BackendError("ML_REPLAY_INCOMPLETE", "Replay содержит ошибки; итоговый экспорт запрещён.")
    bundles = [load_forecast(root / "forecasts" / fid) for fid in report["forecast_ids"]]
    from windops.ui.adapter import load_bundle_json, validate_bundle
    for bundle in bundles:
        validation = validate_bundle(load_bundle_json(json.dumps(bundle, allow_nan=False)))
        if not validation.valid:
            raise BackendError("ML_EXPORT_INVALID", "; ".join(validation.issues))
    table, full = february_table(bundles)
    write_csv(root / "ml" / "february_hourly.csv", table)
    write_csv(root / "ml" / "february_full_releases.csv", full)
    result = {"hourly_rows": len(table), "full_release_rows": len(full), "original_releases": len(bundles),
              "rule": "For local day D: steps 1–24 of original daily D−1 23:00 Asia/Almaty; no intraday updates",
              "target_start_local": "2026-02-01T00:00:00+05:00", "target_end_exclusive_local": "2026-03-01T00:00:00+05:00",
              "february_metrics": None, "reason": "No February actual power provided",
              "checksums": {name: digest((root / "ml" / name).read_bytes()) for name in ("february_hourly.csv", "february_full_releases.csv")}}
    atomic_json(root / "ml" / "february_export.json", result)
    return result


def verify_backend(root, *, baseline_only=False):
    import os
    from windops import backend
    from windops.ui.adapter import BackendAdapter, load_bundle_json, validate_bundle
    from windops.ui.exports import ExportBlocked, selected_csv
    from .models import model_root
    from .plugin import get_model_metadata
    if (os.environ.get("WINDOPS_OFFLINE") != "1" or os.environ.get("WINDOPS_EXECUTION_MODE") != "deterministic" or
            os.environ.get("WINDOPS_ML_MODULE") != "windops.ml.plugin"):
        raise BackendError("ML_VERIFY_ENV", "Нужны WINDOPS_OFFLINE=1, WINDOPS_EXECUTION_MODE=deterministic, WINDOPS_ML_MODULE=windops.ml.plugin.")
    adapter = BackendAdapter("windops.backend")
    first_id, checked = None, []
    origin = "2026-01-31T23:00:00+05:00"
    before_models = {str(p): digest(p.read_bytes()) for p in model_root().glob("*/ml-*/*") if p.is_file()}
    for site in SITES:
        for horizon in (24, 48):
            bundle = adapter.run_forecast([site], origin, horizon)[0]
            validation = validate_bundle(bundle)
            if not validation.exportable or bundle.executor != "deterministic":
                raise BackendError("ML_VERIFY_UI", str(validation.issues))
            selected_csv(bundle, final=True)
            path = root / "forecasts" / bundle.forecast_id / "bundle.json"
            original_checksum = digest(path.read_bytes())
            saved_ids = {p.name for p in (root / "forecasts").iterdir() if p.is_dir()}
            repeated = adapter.run_forecast([site], origin, horizon)[0]
            if (repeated.forecast_id != bundle.forecast_id or digest(path.read_bytes()) != original_checksum or
                    saved_ids != {p.name for p in (root / "forecasts").iterdir() if p.is_dir()}):
                raise BackendError("ML_VERIFY_IDEMPOTENCY", "Повтор изменил сохранённый выпуск.")
            uploaded = load_bundle_json(path.read_bytes())
            if validate_bundle(uploaded).exportable:
                raise BackendError("ML_VERIFY_TRUST", "Загрузка JSON ошибочно дала доверие backend.")
            try:
                selected_csv(uploaded, final=True)
            except ExportBlocked:
                pass
            else:
                raise BackendError("ML_VERIFY_TRUST", "Непроверенный JSON разрешил итоговый экспорт.")
            # Contract alignment under permutation, using the actual selected weights.
            from .plugin import predict_power
            raw = load_forecast(path.parent)
            expected_purpose = "baseline_smoke" if baseline_only else "february_final"
            if raw["provenance"]["model"].get("purpose") != expected_purpose:
                raise BackendError("ML_VERIFY_MODEL", f"Для этой проверки требуется модель {expected_purpose}.")
            weather = raw["weather"]
            model_version = raw["provenance"]["model"]["version"]
            reversed_prediction = predict_power(site_id=site, weather_rows=list(reversed(weather)), model_version=model_version)
            expected = {r["target_time"]: r["prediction"] for r in raw["rows"]}
            if {r["target_time"]: r["prediction"] for r in reversed_prediction} != expected:
                raise BackendError("ML_VERIFY_ORDER", "Перестановка входа нарушила привязку прогноза ко времени.")
            checked.append({"site_id": site, "horizon": horizon, "forecast_id": bundle.forecast_id, "model_version": model_version})
            if site == "turbine_1" and horizon == 48:
                first_id = bundle.forecast_id
    try:
        get_model_metadata(site_id="turbine_1", forecast_origin="2025-10-01T23:00:00+05:00")
    except BackendError as exc:
        if exc.code != "ML_MODEL_MISSING":
            raise
    else:
        raise BackendError("ML_VERIFY_FUTURE", "Ранний origin получил будущую модель.")
    result = {"mode": "real_offline_deterministic", "live_llm_tested": False, "forecasts": checked,
              "adapter_final_export": True, "uploaded_json_final_export": False, "idempotent": True,
              "permutation_verified": True, "future_model_rejected": True,
              "smoke_bundle": str(root / "forecasts" / first_id / "bundle.json")}
    if not baseline_only:
        from windops.ui.quality import load_backtest
        artifact = load_backtest((root / "ml" / "january_backtest.json").read_bytes())
        result["january_rows_loaded"] = len(artifact.rows)
        replay = adapter.run_replay(date(2026, 1, 31), date(2026, 2, 28))
        if replay.errors or len(replay.bundles) != 58:
            raise BackendError("ML_VERIFY_REPLAY", f"Replay: {len(replay.bundles)} выпусков, {len(replay.errors)} ошибок.")
        for bundle in replay.bundles:
            if not validate_bundle(bundle).exportable:
                raise BackendError("ML_VERIFY_REPLAY", "UI отклонил выпуск replay.")
            selected_csv(bundle, final=True)
        result["replay_releases"] = len(replay.bundles)
        result["february_export"] = export_february(root)
        old_path = root / "forecasts" / first_id / "bundle.json"
        old_checksum = digest(old_path.read_bytes())
        updated = backend.check_updates(first_id, as_of="2026-02-01T06:00:00Z")
        if len(updated) != 1 or updated[0]["revises_forecast_id"] != first_id or updated[0]["forecast_id"] == first_id:
            raise BackendError("ML_VERIFY_UPDATE", "Не создана связанная новая версия выпуска.")
        if digest(old_path.read_bytes()) != old_checksum:
            raise BackendError("ML_VERIFY_UPDATE", "Обновление изменило старый выпуск.")
        old = load_forecast(old_path.parent)
        if (updated[0]["provenance"]["model"]["version"] != old["provenance"]["model"]["version"] or
                updated[0]["provenance"]["weather"]["version"] == old["provenance"]["weather"]["version"]):
            raise BackendError("ML_VERIFY_UPDATE", "Обновление должно использовать новую погоду и ту же модель.")
        adapter_update = adapter.check_updates(first_id)
        if len(adapter_update) != 1 or adapter_update[0].forecast_id != updated[0]["forecast_id"]:
            raise BackendError("ML_VERIFY_UPDATE", "Повтор обновления создал другую версию.")
        if not validate_bundle(adapter_update[0]).exportable:
            raise BackendError("ML_VERIFY_UPDATE", "Обновлённый выпуск не принят UI.")
        selected_csv(adapter_update[0], final=True)
        result["update"] = {"old_forecast_id": first_id, "new_forecast_id": updated[0]["forecast_id"], "as_of": "2026-02-01T06:00:00Z"}
    after_models = {str(p): digest(p.read_bytes()) for p in model_root().glob("*/ml-*/*") if p.is_file()}
    if before_models != after_models:
        raise BackendError("ML_VERIFY_RETRAIN", "Артефакты моделей изменились при inference/replay.")
    result["no_retraining"] = True
    atomic_json(root / "ml" / ("baseline_verification.json" if baseline_only else "backend_verification.json"), result)
    return result
