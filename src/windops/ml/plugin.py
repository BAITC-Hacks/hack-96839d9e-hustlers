"""Backend contract. Saved weights only; no training, weather download or SCADA I/O."""
from __future__ import annotations

from windops.core import BackendError, SITES, atomic_json, data_root, digest, iso, stamp
from .features import prepare_features, validate_rows
from .models import find_version, load_model, model_root, predict_model, read_passport


def get_model_metadata(*, site_id, forecast_origin):
    if site_id not in SITES:
        raise BackendError("UNKNOWN_SITE", "Неизвестная турбина.")
    origin = stamp(forecast_origin)
    candidates = []
    for path in sorted((model_root() / site_id).glob("ml-*/passport.json")):
        card = read_passport(path.parent)
        if card["site_id"] == site_id and stamp(card["training_cutoff"]) < origin:
            candidates.append(card)
    if not candidates:
        raise BackendError("ML_MODEL_MISSING", f"Нет сохранённой модели {site_id} с cutoff раньше {iso(origin)}; выполните ML CLI.")
    newest_cutoff = max(stamp(card["training_cutoff"]) for card in candidates)
    candidates = [card for card in candidates if stamp(card["training_cutoff"]) == newest_cutoff]
    if len(candidates) != 1:
        raise BackendError("ML_MODEL_AMBIGUOUS", "Несколько моделей с одинаковым cutoff; используйте отдельный WINDOPS_MODEL_DIR для экспериментов.")
    card = candidates[0]
    metadata = {key: card[key] for key in ("site_id", "version", "training_cutoff", "normalization", "weather_schema",
                                         "provider_model", "labels_available_by_cutoff", "feature_version", "purpose")}
    context = card.get("experiment_context", {})
    if context.get("january_independent") is False:
        metadata["january_comparison_independent"] = False
        metadata["evaluation_status"] = "Разработочное сравнение на январе; январь уже использовался при разработке."
    return metadata


def predict_power(*, site_id, weather_rows, model_version):
    if site_id not in SITES:
        raise BackendError("UNKNOWN_SITE", "Неизвестная турбина.")
    frame = validate_rows(weather_rows)
    if set(frame.site_id) != {site_id} or frame.forecast_origin.nunique() != 1:
        raise BackendError("ML_WEATHER_GRID", "Нужен один выпуск одной турбины.")
    if len(frame) not in (24, 48) or set(frame.lead_hours) != set(range(1, len(frame) + 1)):
        raise BackendError("ML_WEATHER_GRID", "Нужна полная сетка из 24 или 48 часов.")
    folder, card = find_version(site_id, model_version)
    if not stamp(card["training_cutoff"]) < frame.forecast_origin.min():
        raise BackendError("FUTURE_MODEL", "Переданная версия обучена после допустимого cutoff.")
    predictions, clipped = predict_model(load_model(folder, card), card["config"], frame)
    features = prepare_features(frame, card["config"]["feature_set"])
    outside = {name: int((~features[name].between(bounds["min"], bounds["max"])).sum())
               for name, bounds in card["feature_ranges"].items()}
    lead_outside = {name: int((~frame[name].between(bounds["min"], bounds["max"])).sum())
                    for name, bounds in card["training_weather_lead_ranges"].items()}
    diagnostic = {"site_id": site_id, "model_version": model_version,
                  "forecast_origin": iso(frame.forecast_origin.iloc[0]), "rows": len(frame),
                  "clipped_predictions": clipped, "outside_training_range": outside,
                  "leads_outside_training_range": lead_outside,
                  "update_accuracy_evaluated": False}
    key = digest({"version": model_version, "features": features.to_dict("records"),
                  "times": [row["target_time"] for row in weather_rows]})[:24]
    atomic_json(data_root() / "ml" / "inference" / f"{key}.json", diagnostic)
    return [{"target_time": row["target_time"], "prediction": float(value)} for row, value in zip(weather_rows, predictions, strict=True)]
