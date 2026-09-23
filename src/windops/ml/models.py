"""Small learned baselines, CPU CatBoost, immutable artifacts and passports."""
from __future__ import annotations

from importlib.metadata import version
import os
from pathlib import Path
import platform
import re
import shutil
import tempfile
import time

import numpy as np

from windops.core import BackendError, ROOT, WEATHER_SCHEMA, atomic_json, digest, iso, now, read_json, stamp
from .features import FEATURE_SETS, FEATURE_VERSION, bounded_predictions, prepare_features

BASELINES = ("constant_median", "wind_table")


def model_root():
    return Path(os.environ.get("WINDOPS_MODEL_DIR") or ROOT / "models").resolve()


def library_versions():
    return {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "pandas", "catboost")}}


def implementation_hash():
    return digest({p.name: digest(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))})


def fit_baselines(features, labels):
    labels = np.asarray(labels, dtype=float)
    if not len(labels) or not np.isfinite(labels).all() or ((labels < 0) | (labels > 1)).any():
        raise BackendError("ML_TRAINING_LABELS", "Baseline требует валидные ответы 0–1.")
    bins = np.floor(features["wind_speed_100m_ms"].to_numpy()).astype(int)
    occupied = np.unique(bins)
    return {"constant_median": {"value": float(np.median(labels))},
            "wind_table": {"width_ms": 1, "wind_field": "wind_speed_100m_ms",
                           "empty_bin_rule": "nearest occupied bin, lower bin on ties; edge bins outside range",
                           "bins": occupied.tolist(),
                           "values": [float(np.median(labels[bins == i])) for i in occupied]}}


def baseline_prediction(kind, state, features):
    if kind == "constant_median":
        return np.full(len(features), state["value"], dtype=float)
    if kind != "wind_table":
        raise BackendError("ML_MODEL_TYPE", "Неизвестный baseline.")
    bins = np.asarray(state["bins"])
    requested = np.floor(features[state["wind_field"]].to_numpy()).astype(int)
    nearest = np.abs(requested[:, None] - bins[None, :]).argmin(axis=1)
    return np.asarray(state["values"], dtype=float)[nearest]


def fit_model(config, train, evaluation=None):
    features = prepare_features(train, config["feature_set"])
    baselines = fit_baselines(features, train.actual)
    if config["kind"] in BASELINES:
        return baselines[config["kind"]], baselines, config.copy()
    if config["kind"] != "catboost":
        raise BackendError("ML_MODEL_TYPE", "Неизвестный тип модели.")
    from catboost import CatBoostRegressor
    estimator = CatBoostRegressor(**config["params"])
    kwargs = {}
    if evaluation is not None:
        kwargs = {"eval_set": (prepare_features(evaluation, config["feature_set"]), evaluation.actual),
                  "early_stopping_rounds": 40, "use_best_model": True}
    estimator.fit(features, train.actual, **kwargs)
    fixed = {**config, "params": {**config["params"], "iterations": int(estimator.tree_count_)}}
    return estimator, baselines, fixed


def predict_model(model, config, rows):
    features = prepare_features(rows, config["feature_set"])
    raw = baseline_prediction(config["kind"], model, features) if config["kind"] in BASELINES else model.predict(features)
    return bounded_predictions(raw)


def read_passport(folder):
    card = read_json(folder / "passport.json")
    if read_json(folder / "passport_checksum.json").get("sha256") != digest(card):
        raise BackendError("ML_CORRUPT_MODEL", "Повреждён паспорт модели.")
    expected_version = "ml-" + digest(card["identity"])[:24]
    if card["version"] != expected_version or folder.name != expected_version:
        raise BackendError("ML_CORRUPT_MODEL", "Версия не соответствует содержимому паспорта.")
    for name, expected in card["artifact_checksums"].items():
        if Path(name).name != name or not (folder / name).is_file() or digest((folder / name).read_bytes()) != expected:
            raise BackendError("ML_CORRUPT_MODEL", "Повреждён артефакт модели.")
    if card["feature_version"] != FEATURE_VERSION or card["features"] != list(FEATURE_SETS[card["config"]["feature_set"]]):
        raise BackendError("ML_FEATURE_SCHEMA", "Несовместимый порядок/версия признаков.")
    if (card["labels_available_by_cutoff"] is not True or
            stamp(card["training_summary"]["max_label_available_at"]) > stamp(card["training_cutoff"])):
        raise BackendError("ML_TRAINING_LABELS", "Не подтверждена доступность обучающих ответов.")
    return card


def load_model(folder, card=None):
    card = card or read_passport(folder)
    if card["config"]["kind"] in BASELINES:
        return read_json(folder / "model.json")
    from catboost import CatBoostRegressor
    estimator = CatBoostRegressor()
    estimator.load_model(str(folder / "model.cbm"))
    if estimator.feature_names_ != card["features"]:
        raise BackendError("ML_FEATURE_SCHEMA", "Признаки сохранённого CatBoost отличаются от паспорта.")
    return estimator


def train_artifact(train, config, *, site_id, purpose, cutoff, prepared, models=None, selection_digest=None):
    models = models or model_root()
    if train.empty or set(train.site_id) != {site_id}:
        raise BackendError("ML_TRAINING_LABELS", f"Нет допустимых данных одной турбины: {site_id}.")
    cutoff = stamp(cutoff)
    if not (train.complete.eq(True) & train.actual.between(0, 1) &
            (train.interval_end <= train.label_available_at) & (train.label_available_at <= cutoff) &
            (train.forecast_origin < cutoff)).all():
        raise BackendError("ML_TRAINING_LABELS", "Обучающие ответы не прошли проверку cutoff/полноты.")
    features = prepare_features(train, config["feature_set"])
    identity = {"site_id": site_id, "purpose": purpose, "cutoff": iso(cutoff), "config": config,
                "prepared": digest(prepared), "selection_digest": selection_digest,
                "training_data_sha256": digest(train.to_csv(index=False).encode()),
                "feature_version": FEATURE_VERSION, "code_sha256": implementation_hash(), "libraries": library_versions()}
    model_version = "ml-" + digest(identity)[:24]
    folder = models / site_id / model_version
    if folder.exists():
        return read_passport(folder)
    started = time.monotonic()
    estimator, baselines, fixed = fit_model(config, train)
    if fixed != config:
        raise BackendError("ML_ITERATIONS", "При финальном обучении изменилось фиксированное число итераций.")
    probe, clipped = predict_model(estimator, config, train)
    folder.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".training-", dir=folder.parent))
    try:
        model_file = "model.json" if config["kind"] in BASELINES else "model.cbm"
        if model_file == "model.json":
            atomic_json(temporary / model_file, estimator)
        else:
            estimator.save_model(str(temporary / model_file))
        atomic_json(temporary / "baselines.json", baselines)
        card = {"schema_version": 1, "site_id": site_id, "version": model_version, "purpose": purpose,
                "identity": identity, "training_cutoff": iso(cutoff), "config": config,
                "normalization": "0_1", "weather_schema": WEATHER_SCHEMA, "provider_model": "gfs_pgrb2.0p25",
                "weather_source": "data/weather/weather_for_ml.csv; NOAA GFS archived forecasts",
                "labels_available_by_cutoff": True, "feature_version": FEATURE_VERSION,
                "features": list(features.columns), "feature_dtype": "float64",
                "feature_ranges": {name: {"min": float(features[name].min()), "max": float(features[name].max())} for name in features},
                "training_weather_lead_ranges": {name: {"min": float(train[name].min()), "max": float(train[name].max())}
                                                  for name in ("lead_hours", "gfs_lead_hours")},
                "label_policy": prepared["label_policy"], "input_checksums": prepared["inputs"],
                "training_summary": {"rows": len(train), "unique_target_hours": int(train.target_time.nunique()),
                                     "target_start": iso(train.target_time.min()), "target_end": iso(train.target_time.max()),
                                     "max_label_available_at": iso(train.label_available_at.max())},
                "output_policy": "reject nonfinite predictions; fixed clip to [0,1] in all paths",
                "training_prediction_clipped": clipped, "trained_at": now(),
                "training_seconds": round(time.monotonic() - started, 6),
                "limitations": ["SCADA interval semantics and arrival delay unconfirmed",
                                "Update origins/leads may differ from training; update accuracy not evaluated",
                                "Normalized power, rated MW unknown; February actuals unavailable"],
                "artifact_checksums": {name: digest((temporary / name).read_bytes()) for name in (model_file, "baselines.json")}}
        loaded = load_model(temporary, card)
        reloaded, _ = predict_model(loaded, config, train)
        if not np.array_equal(probe, reloaded):
            raise BackendError("ML_ROUNDTRIP", "Сохранённая и исходная модели дают разные прогнозы.")
        card["save_load_predictions_identical"] = True
        atomic_json(temporary / "passport.json", card)
        atomic_json(temporary / "passport_checksum.json", {"sha256": digest(card)})
        temporary.rename(folder)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return read_passport(folder)


def find_version(site_id, model_version, models=None):
    if not re.fullmatch(r"ml-[a-f0-9]{24}", model_version):
        raise BackendError("ML_MODEL_VERSION", "Неверный идентификатор модели.")
    folder = (models or model_root()) / site_id / model_version
    if not folder.is_dir():
        raise BackendError("ML_MODEL_MISSING", f"Не найден сохранённый артефакт {model_version}.")
    card = read_passport(folder)
    if card["site_id"] != site_id:
        raise BackendError("ML_MODEL_SITE", "Модель принадлежит другой турбине.")
    return folder, card
