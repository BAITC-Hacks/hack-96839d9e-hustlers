"""Deterministic domain operations used by both the LLM and CLI."""
from __future__ import annotations

import csv
import json
import time

from .core import (BackendError, TIMESTAMP_CONVENTION, atomic_json, candidate_runs,
                   data_root, digest, iso, now, stamp, validate_request,
                   load_forecast, save_forecast)
from .ml_bridge import MLPredictor, validate_metadata, validate_predictions
from .weather import validate_weather, weather_for_run


class ForecastService:
    def __init__(self, site_id, origin, horizon=48, *, root=None, predictor=None, weather_loader=None, offline=False, runs=None):
        self.origin = validate_request(site_id, origin, horizon)
        self.site_id, self.horizon = site_id, horizon
        self.root = root or data_root()
        self.predictor = predictor or MLPredictor()
        self.metadata = self.predictor.metadata(site_id, iso(self.origin))
        validate_metadata(self.metadata, site_id, self.origin)
        self.weather_loader = weather_loader or weather_for_run
        self.offline = offline
        self.runs = runs or candidate_runs(self.origin)
        if len(self.runs) != 2:
            raise BackendError("INVALID_CANDIDATES", "Нужно два погодных кандидата.")
        self.weather, self.predictions, self.events = {}, {}, []
        self.bundle = None
        self.executor = "deterministic"
        self.revises_forecast_id = None

    def event(self, name, status, **details):
        self.events.append({"recorded_at": now(), "forecast_origin": iso(self.origin),
                            "tool": name, "status": status, **details})

    def list_weather_candidates(self):
        return {"candidates": [{"candidate_index": i, "run_initialized_at": iso(run),
                                "status": "availability_must_be_checked"}
                               for i, run in enumerate(self.runs)]}

    def fetch_weather(self, candidate_index):
        if type(candidate_index) is not int or candidate_index not in (0, 1):
            raise BackendError("INVALID_ARGUMENTS", "Недопустимый индекс погодного кандидата.")
        weather = self.weather_loader(self.origin, self.horizon, run=self.runs[candidate_index], root=self.root, offline=self.offline)
        validate_weather(weather)
        if stamp(weather["forecast_origin"]) != self.origin or weather["horizon_hours"] != self.horizon:
            raise BackendError("WEATHER_REQUEST_MISMATCH", "Погодный набор не соответствует текущему запросу.")
        self.weather[weather["weather_version"]] = weather
        return {"weather_id": weather["weather_version"], "hours": self.horizon, "available_at_upper_bound": weather["available_at"]}

    def validate_weather(self, weather_id):
        return validate_weather(self._weather(weather_id))

    def _weather(self, weather_id):
        if weather_id not in self.weather:
            raise BackendError("UNKNOWN_WEATHER", "Погодный набор не был получен в этом запросе.")
        return self.weather[weather_id]

    def predict_power(self, weather_id):
        weather = self._weather(weather_id)
        validate_weather(weather)
        validate_metadata(self.metadata, self.site_id, self.origin)
        rows = self.predictor.predict(self.site_id, weather, self.metadata)
        rows = validate_predictions(rows, [r for r in weather["rows"] if r["site_id"] == self.site_id])
        prediction_id = "prediction-" + digest({"weather": weather_id, "metadata": self.metadata, "site": self.site_id, "rows": rows})[:24]
        self.predictions[prediction_id] = {"rows": rows, "weather_id": weather_id}
        return {"prediction_id": prediction_id, "hours": len(rows), "model_version": self.metadata["version"]}

    def _prediction(self, prediction_id):
        if prediction_id not in self.predictions:
            raise BackendError("UNKNOWN_PREDICTION", "Прогноз не был получен в этом запросе.")
        return self.predictions[prediction_id]

    def analyse_forecast(self, prediction_id):
        item = self._prediction(prediction_id)
        values = [r["prediction"] for r in item["rows"]]
        return {"mean_normalized_power": sum(values) / len(values), "min_normalized_power": min(values),
                "max_normalized_power": max(values), "hours": len(values), "units": "normalized_0_1"}

    def publish_forecast(self, prediction_id):
        prediction = self._prediction(prediction_id)
        weather = self._weather(prediction["weather_id"])
        validate_weather(weather)
        validate_metadata(self.metadata, self.site_id, self.origin)
        weather_rows = [row for row in weather["rows"] if row["site_id"] == self.site_id]
        values = validate_predictions(prediction["rows"], weather_rows)
        identity = {"site_id": self.site_id, "origin": iso(self.origin), "horizon": self.horizon,
                    "weather_version": weather["weather_version"], "model_metadata": self.metadata}
        forecast_id = "forecast-" + digest(identity)[:24]
        folder = self.root / "forecasts" / forecast_id
        path = folder / "bundle.json"
        if path.exists():
            self.bundle = load_forecast(folder)
            return {"forecast_id": forecast_id, "status": "already_saved"}
        rows = [{"run_id": forecast_id, "site_id": self.site_id, "issued_at": iso(self.origin),
                 "target_time": row["target_time"], "horizon_step": i, "lead_hours": i,
                 "model_version": self.metadata["version"], "weather_version": weather["weather_version"],
                 "quality_flag": "verified_object_version",
                 **{key: row[key] for key in ("prediction", "p10", "p50", "p90") if key in row}}
                for i, row in enumerate(values, 1)]
        summary = self.analyse_forecast(prediction_id)
        bundle = {"forecast_id": forecast_id, "rows": rows, "horizon_hours": self.horizon,
                  "site_ids": [self.site_id], "timezone": "Asia/Almaty", "source_mode": "real", "executor": self.executor,
                  "validation_summary": {"inputs_verified": True, "model_verified": True, "hour_count": len(rows)},
                  "provenance": {"timestamp_convention": TIMESTAMP_CONVENTION,
                                 "weather": {"source": "NOAA GFS public S3", "version": weather["weather_version"],
                                             "data_kind": "archived_forecast", "synthetic": False,
                                             "run_initialized_at": weather["run_initialized_at"],
                                             "available_at": weather["available_at"], "availability_upper_bound": weather["available_at"],
                                             "availability_basis": weather["availability_basis"], "retrieved_at": weather["retrieved_at"]},
                                 "model": {**self.metadata, "historically_eligible": True}},
                  "weather": [{**r, "data_kind": "archived_forecast", "temperature_c": r["temperature_2m_c"],
                               "wind_speed_10m": r["wind_speed_10m_ms"], "wind_speed_100m": r["wind_speed_100m_ms"]}
                              for r in weather_rows],
                  "report_facts": [f"{len(rows)} часов; средняя нормализованная мощность {summary['mean_normalized_power']:.4f}."],
                  "events": self.events.copy(), "revises_forecast_id": self.revises_forecast_id, "created_at": now()}
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder / "forecast_manifest.json", {**identity, "forecast_id": forecast_id, "created_at": bundle["created_at"]})
        atomic_json(folder / "weather_manifest.json", {key: value for key, value in weather.items() if key != "rows"})
        atomic_json(folder / "validation_report.json", bundle["validation_summary"])
        with (folder / "forecast.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        # bundle is the completion marker, written last.
        save_forecast(folder, bundle)
        self.bundle = bundle
        return {"forecast_id": forecast_id, "status": "saved", "hours": len(rows)}

    def finish(self):
        if self.bundle is None:
            raise BackendError("INCOMPLETE_FORECAST", "Не получен проверенный сохранённый прогноз.")
        folder = self.root / "forecasts" / self.bundle["forecast_id"]
        # Preserve the original audit on idempotent re-use.
        if not (folder / "events.jsonl").exists():
            self.bundle["events"] = self.events.copy()
            save_forecast(folder, self.bundle)
            (folder / "events.jsonl").write_text("".join(json.dumps(event, ensure_ascii=False) + "\n" for event in self.events), encoding="utf-8")
        return self.bundle


def deterministic(service: ForecastService):
    weather_id = None
    for index in (0, 1):
        try:
            start = time.monotonic()
            weather_id = service.fetch_weather(index)["weather_id"]
            service.event("fetch_weather", "success", candidate_index=index, duration_seconds=round(time.monotonic() - start, 3))
            break
        except BackendError as exc:
            service.event("fetch_weather", "error", error_code=exc.code)
    if weather_id is None:
        raise BackendError("NO_ELIGIBLE_WEATHER", "Нет полного допустимого погодного выпуска.")
    for name, args in (("validate_weather", {"weather_id": weather_id}), ("predict_power", {"weather_id": weather_id})):
        result = getattr(service, name)(**args)
        service.event(name, "success")
    prediction_id = result["prediction_id"]
    service.analyse_forecast(prediction_id)
    service.event("analyse_forecast", "success")
    service.publish_forecast(prediction_id)
    service.event("publish_forecast", "success")
    return service.finish()


def compare_forecasts(old, new):
    left = {(r["site_id"], r["target_time"]): r["prediction"] for r in old["rows"]}
    diffs = [r["prediction"] - left[(r["site_id"], r["target_time"])] for r in new["rows"] if (r["site_id"], r["target_time"]) in left]
    return {"overlap_hours": len(diffs), "mean_absolute_change": sum(map(abs, diffs)) / len(diffs) if diffs else None,
            "max_absolute_change": max(map(abs, diffs)) if diffs else None}
