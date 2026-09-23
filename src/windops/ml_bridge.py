"""Participant 1's trusted Python plugin; no training or synthetic fallback."""
from __future__ import annotations

import importlib
import math
import os
import re

from .core import BackendError, WEATHER_SCHEMA, stamp
from .weather import validate_weather


class MLPredictor:
    def __init__(self, module=None):
        name = module or os.environ.get("WINDOPS_ML_MODULE")
        if not name:
            raise BackendError("MODEL_NOT_CONFIGURED", "Укажите WINDOPS_ML_MODULE: модуль участника №1 с get_model_metadata и predict_power.")
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", name):
            raise BackendError("MODEL_NOT_CONFIGURED", "Неверное имя ML-модуля.")
        try:
            self.module = importlib.import_module(name)
        except ImportError:
            raise BackendError("MODEL_NOT_CONFIGURED", "ML-модуль не установлен или отсутствуют его зависимости.") from None
        if not all(callable(getattr(self.module, key, None)) for key in ("get_model_metadata", "predict_power")):
            raise BackendError("MODEL_CONTRACT", "Нужны get_model_metadata и predict_power; см. docs/BACKEND.md.")

    def metadata(self, site_id, origin):
        try:
            metadata = self.module.get_model_metadata(site_id=site_id, forecast_origin=origin)
        except BackendError:
            raise
        except Exception:
            raise BackendError("MODEL_METADATA_FAILED", "Модуль №1 не вернул метаданные модели.") from None
        validate_metadata(metadata, site_id, origin)
        return metadata

    def predict(self, site_id, weather, metadata):
        validate_weather(weather)
        validate_metadata(metadata, site_id, weather["forecast_origin"])
        rows = [dict(row) for row in weather["rows"] if row["site_id"] == site_id]
        try:
            result = self.module.predict_power(site_id=site_id, weather_rows=rows, model_version=metadata["version"])
        except BackendError:
            raise
        except Exception:
            raise BackendError("MODEL_INFERENCE_FAILED", "Модуль №1 завершился ошибкой; прогноз не опубликован.") from None
        return validate_predictions(result, rows)


def validate_metadata(metadata, site_id, origin):
    if not isinstance(metadata, dict) or metadata.get("site_id") != site_id:
        raise BackendError("MODEL_CONTRACT", "Модель должна указать соответствующую турбину.")
    if not isinstance(metadata.get("version"), str) or not metadata["version"].strip():
        raise BackendError("MODEL_CONTRACT", "Нет версии модели.")
    if stamp(metadata.get("training_cutoff")) > stamp(origin):
        raise BackendError("FUTURE_MODEL", "Модель обучена на результатах после момента прогноза.")
    if metadata.get("normalization") != "0_1" or metadata.get("weather_schema") != WEATHER_SCHEMA:
        raise BackendError("MODEL_INCOMPATIBLE", "Модель несовместима с нормировкой или схемой GFS.")
    if metadata.get("provider_model") != "gfs_pgrb2.0p25":
        raise BackendError("MODEL_INCOMPATIBLE", "Модель должна быть обучена на выбранном GFS.")
    # The provider must attest that labels (entire target intervals) were known
    # at cutoff. The bridge cannot infer this from an opaque trained model.
    if metadata.get("labels_available_by_cutoff") is not True:
        raise BackendError("MODEL_CUTOFF_UNCONFIRMED", "ML-модуль не подтвердил доступность обучающих ответов на cutoff.")
    return metadata


def validate_predictions(result, weather_rows):
    if not isinstance(result, list) or len(result) != len(weather_rows):
        raise BackendError("PREDICTION_GRID", "Модель должна вернуть одну строку на каждый погодный час.")
    expected = {row["target_time"] for row in weather_rows}
    if any(not isinstance(row, dict) for row in result):
        raise BackendError("PREDICTION_GRID", "Нужен список записей с target_time и prediction.")
    times = [row.get("target_time") for row in result]
    if any(not isinstance(value, str) for value in times):
        raise BackendError("PREDICTION_GRID", "target_time модели должен быть строкой ISO UTC.")
    if len(set(times)) != len(times) or set(times) != expected:
        raise BackendError("PREDICTION_GRID", "Пропуск, дубль или неверный целевой час модели.")
    for row in result:
        for name in ("prediction", "p10", "p50", "p90"):
            if name not in row and name != "prediction":
                continue
            value = row.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise BackendError("INVALID_PREDICTION", "Модель вернула NaN/inf, неверный тип или значение вне 0–1.")
        quantiles = [row[name] for name in ("p10", "p50", "p90") if name in row]
        if quantiles != sorted(quantiles):
            raise BackendError("INVALID_PREDICTION", "Квантили модели не упорядочены.")
    for name in ("p10", "p50", "p90"):
        if any(name in row for row in result) and not all(name in row for row in result):
            raise BackendError("INVALID_PREDICTION", "Квантиль предоставлен только для части горизонта.")
    return sorted(result, key=lambda row: row["target_time"])
