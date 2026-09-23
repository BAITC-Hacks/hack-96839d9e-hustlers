"""Public participant 2 API matching the existing UI adapter."""
from __future__ import annotations

from datetime import date, timedelta
import os
import re

from .agent import run_agent
from .core import (BackendError, SITES, atomic_json, daily_origins,
                   data_root, iso, now, stamp, load_forecast, save_forecast)
from .pipeline import ForecastService, compare_forecasts


def run_forecast(site_id, forecast_origin, horizon_hours):
    service = None
    try:
        service = ForecastService(site_id, forecast_origin, horizon_hours,
                                  offline=os.environ.get("WINDOPS_OFFLINE") == "1")
        return run_agent(service)
    except BackendError as exc:
        atomic_json(data_root() / "reports" / "last_forecast_error.json",
                    {"recorded_at": now(), "site_id": site_id, "forecast_origin": str(forecast_origin),
                     "error": exc.as_dict(), "events": service.events if service else []})
        raise


def run_replay(start_date: date, end_date: date):
    """Dates refer to LOCAL release dates, at 23:00 Asia/Almaty, inclusive."""
    bundles, errors = [], []
    for origin in daily_origins(start_date, end_date):
        for site in SITES:
            try:
                bundles.append(run_forecast(site, origin, 48))
            except BackendError as exc:
                errors.append({"issued_at": iso(origin), "site_id": site, "error_code": exc.code})
                if exc.code == "MODEL_NOT_CONFIGURED":
                    atomic_json(data_root() / "reports" / "replay.json", {"forecast_ids": [], "errors": errors})
                    return {"bundles": bundles, "errors": errors}
    atomic_json(data_root() / "reports" / "replay.json", {"forecast_ids": [b["forecast_id"] for b in bundles], "errors": errors})
    return {"bundles": bundles, "errors": errors}


def check_updates(forecast_id, *, as_of=None):
    """Historical UI checks advance virtual time by twelve hours, never to today.

    Only a newly eligible GFS cycle triggers an automatic recomputation. Call
    with as_of to choose an explicit historical clock from CLI or integration.
    """
    if not re.fullmatch(r"forecast-[0-9a-f]{24}", forecast_id):
        raise BackendError("INVALID_ID", "Неверный идентификатор прогноза.")
    old = load_forecast(data_root() / "forecasts" / forecast_id)
    previous_origin = stamp(old["rows"][0]["issued_at"])
    origin = stamp(as_of) if as_of else previous_origin + timedelta(hours=12)
    if origin <= previous_origin:
        raise BackendError("INVALID_UPDATE_TIME", "Обновление должно иметь более поздний forecast_origin.")
    latest = origin.replace(hour=origin.hour // 6 * 6)
    service = ForecastService(old["site_ids"][0], origin, old["horizon_hours"],
                              offline=os.environ.get("WINDOPS_OFFLINE") == "1",
                              runs=[latest, latest - timedelta(hours=6)])
    failures = []
    for index in (0, 1):
        try:
            result = service.fetch_weather(index)
            weather = service.weather[result["weather_id"]]
            if stamp(weather["run_initialized_at"]) <= stamp(old["provenance"]["weather"]["run_initialized_at"]):
                return []
            service.revises_forecast_id = forecast_id
            service.event("weather_updated", "success", old_weather_version=old["provenance"]["weather"]["version"], new_weather_version=weather["weather_version"])
            new = run_agent(service)
            new["comparison"] = compare_forecasts(old, new)
            save_forecast(data_root() / "forecasts" / new["forecast_id"], new)
            return [new]
        except BackendError as exc:
            if exc.code in ("CACHE_MISS", "HTTP_STATUS", "FUTURE_WEATHER", "WEATHER_NETWORK"):
                failures.append(exc.code)
                continue
            raise
    raise BackendError("NO_ELIGIBLE_WEATHER", "Проверка обновлений не завершена: " + ", ".join(failures))
