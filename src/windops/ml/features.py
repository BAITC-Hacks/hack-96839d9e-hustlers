"""One explicit, target-free feature contract for training and inference."""
from __future__ import annotations

import numpy as np
import pandas as pd

from windops.core import BackendError, SITES, stamp

FEATURE_VERSION = "gfs-power-v1"
WEATHER_FIELDS = (
    "wind_speed_10m_ms", "wind_speed_100m_ms",
    "wind_u_10m_ms", "wind_v_10m_ms", "wind_u_100m_ms", "wind_v_100m_ms",
    "temperature_2m_c",
)
FEATURE_SETS = {
    "weather_hour": (*WEATHER_FIELDS, "local_hour_sin", "local_hour_cos"),
    "weather_hour_lead": (*WEATHER_FIELDS, "local_hour_sin", "local_hour_cos", "lead_hours", "gfs_lead_hours"),
}
TIME_FIELDS = ("forecast_origin", "target_time", "run_initialized_at", "availability_upper_bound")


def utc_series(values):
    # stamp rejects naive dates, including when the host timezone differs.
    return pd.to_datetime(values.map(stamp), utc=True)


def validate_rows(rows):
    frame = pd.DataFrame(rows).copy()
    required = {*TIME_FIELDS, *WEATHER_FIELDS, "site_id", "provider_model", "lead_hours", "gfs_lead_hours"}
    missing = required - set(frame)
    if missing or frame.empty:
        raise BackendError("ML_WEATHER_SCHEMA", f"Пустая погода или отсутствуют поля: {sorted(missing)}")
    if not frame.site_id.isin(SITES).all() or not frame.provider_model.eq("gfs_pgrb2.0p25").all():
        raise BackendError("ML_WEATHER_SCHEMA", "Неизвестная турбина или несовместимый provider_model.")
    for name in TIME_FIELDS:
        frame[name] = utc_series(frame[name])
    if frame.duplicated(["site_id", "forecast_origin", "target_time"]).any():
        raise BackendError("ML_WEATHER_DUPLICATE", "Повтор погодного ключа site/origin/target.")
    if not ((frame.run_initialized_at <= frame.availability_upper_bound) &
            (frame.availability_upper_bound <= frame.forecast_origin)).all():
        raise BackendError("FUTURE_WEATHER", "Нарушена историческая доступность погодных строк.")
    for name in (*WEATHER_FIELDS, "lead_hours", "gfs_lead_hours"):
        if frame[name].map(lambda x: isinstance(x, (bool, np.bool_))).any():
            raise BackendError("ML_FEATURE_VALUE", f"Логическое значение вместо числа: {name}.")
        frame[name] = pd.to_numeric(frame[name], errors="coerce").astype("float64")
        if not np.isfinite(frame[name]).all():
            raise BackendError("ML_FEATURE_VALUE", f"Нет конечного числового признака {name}.")
    for name, reference in (("lead_hours", "forecast_origin"), ("gfs_lead_hours", "run_initialized_at")):
        expected = (frame.target_time - frame[reference]).dt.total_seconds() / 3600
        if not (frame[name].eq(expected) & frame[name].gt(0) & frame[name].mod(1).eq(0)).all():
            raise BackendError("ML_LEAD_TIME", f"Неверный {name}.")
    if not frame.lead_hours.between(1, 48).all() or not frame.target_time.eq(frame.target_time.dt.floor("h")).all():
        raise BackendError("ML_LEAD_TIME", "Целевые часы должны лежать на сетке 1–48.")
    if (frame[["wind_speed_10m_ms", "wind_speed_100m_ms"]] < 0).any().any():
        raise BackendError("INVALID_WIND", "Отрицательная скорость ветра.")
    return frame


def prepare_features(rows, feature_set="weather_hour"):
    if feature_set not in FEATURE_SETS:
        raise BackendError("ML_FEATURE_SCHEMA", "Неизвестная версия набора признаков.")
    frame = validate_rows(rows)
    hour = frame.target_time.dt.tz_convert("Asia/Almaty").dt.hour
    frame["local_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["local_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return frame.loc[:, FEATURE_SETS[feature_set]].astype("float64")


def bounded_predictions(values):
    values = np.asarray(values, dtype="float64")
    if values.ndim != 1 or not np.isfinite(values).all():
        raise BackendError("INVALID_PREDICTION", "ML вернула NaN/inf или неверную размерность.")
    count = int(((values < 0) | (values > 1)).sum())
    return np.clip(values, 0, 1), count
