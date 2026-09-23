"""Explicitly synthetic UI data. No model, weather archive, or event log claims."""

import math
from uuid import uuid4

import pandas as pd

from .adapter import ForecastBundle, TIMESTAMP_CONVENTION


def synthetic_bundle(site_ids=("demo-1", "demo-2"), horizon_hours=48,
                     forecast_origin="2026-02-01T00:00:00+00:00") -> ForecastBundle:
    if horizon_hours not in (24, 48) or not site_ids:
        raise ValueError("Fixture требует турбины и горизонт 24/48 часов.")
    origin = pd.Timestamp(forecast_origin)
    if origin.tzinfo is None:
        raise ValueError("У fixture время должно содержать явную зону.")
    origin = origin.tz_convert("UTC")
    forecast_id = "demo-" + uuid4().hex[:12]
    rows = []
    weather = []
    for site_index, site in enumerate(site_ids):
        site_index = {"demo-1": 0, "demo-2": 1}.get(site, site_index)
        for hour in range(1, horizon_hours + 1):
            value = 0.47 + 0.24 * math.sin(hour / 5 + site_index * 1.3) + 0.08 * math.cos(hour / 2.8)
            target = (origin + pd.Timedelta(hours=hour)).isoformat()
            rows.append({"run_id": forecast_id, "site_id": site, "issued_at": origin.isoformat(),
                         "target_time": target, "horizon_step": hour, "lead_hours": hour,
                         "prediction": value, "model_version": "synthetic-ui-v1",
                         "weather_version": "synthetic-weather-v1", "quality_flag": "synthetic"})
            weather.append({"site_id": site, "target_time": target,
                            "wind_speed": 7.5 + 2.5 * math.sin(hour / 5 + site_index * 1.3),
                            "temperature": -5 + 3 * math.cos(hour / 7),
                            "data_kind": "synthetic", "wind_height_m": None})
    return ForecastBundle(
        forecast_id=forecast_id, rows=pd.DataFrame(rows), weather=pd.DataFrame(weather),
        validation_summary={"inputs_verified": False, "model_verified": False},
        provenance={"timestamp_convention": TIMESTAMP_CONVENTION,
                    "weather": {"source": "СИНТЕТИЧЕСКИЕ ДАННЫЕ — ТЕСТ ИНТЕРФЕЙСА",
                                "version": "synthetic-weather-v1", "data_kind": "synthetic", "synthetic": True},
                    "model": {"version": "synthetic-ui-v1", "normalization": "0_1",
                              "historically_eligible": False}},
        report_facts=[], events=[], horizon_hours=horizon_hours, site_ids=list(site_ids),
        timezone="UTC", source_mode="demo", executor="Детерминированный fixture интерфейса",
        verification_origin="synthetic",
    )
