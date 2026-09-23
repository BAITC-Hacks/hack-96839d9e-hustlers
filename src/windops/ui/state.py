"""Session-local results; navigation and download never call the backend."""

import hashlib
import json

from .adapter import AdapterError, sanitize_metadata, validate_bundle


def initialize_state(mapping):
    for key, value in {"status": "idle", "error": None, "history": {}, "selected_id": None,
                       "request_key": None, "previous_success": False}.items():
        if key not in mapping:
            mapping[key] = value


def begin_run(mapping, parameters) -> bool:
    initialize_state(mapping)
    key = hashlib.sha256(json.dumps(parameters, sort_keys=True, default=str).encode()).hexdigest()
    if mapping["status"] == "running":
        return False
    mapping["request_key"] = key
    mapping["status"] = "running"
    mapping["error"] = None
    mapping["previous_success"] = bool(mapping["history"])
    return True


def finish_run(mapping, bundles):
    initialize_state(mapping)
    if not bundles:
        fail_run(mapping, "Backend не вернул выпуск. Предыдущие результаты доступны в истории.")
        return
    incoming = {}
    for bundle in bundles:
        existing = mapping["history"].get(bundle.forecast_id) or incoming.get(bundle.forecast_id)
        if existing is not None and not existing.rows.equals(bundle.rows):
            raise AdapterError("forecast_id уже существует с другими строками. Для пересчёта нужен новый forecast_id; предыдущий выпуск сохранён.")
        incoming[bundle.forecast_id] = bundle
    history = dict(mapping["history"])
    history.update(incoming)
    mapping["history"] = history
    mapping["selected_id"] = bundles[-1].forecast_id
    mapping["status"] = "success" if all(validate_bundle(bundle).valid for bundle in bundles) else "partial"
    mapping["error"] = None
    mapping["request_key"] = None
    mapping["previous_success"] = False


def fail_run(mapping, message):
    initialize_state(mapping)
    mapping["status"] = "error"
    mapping["error"] = sanitize_metadata(str(message))
    mapping["request_key"] = None
    mapping["previous_success"] = bool(mapping["history"])


def select_bundle(mapping, forecast_id):
    initialize_state(mapping)
    if forecast_id not in mapping["history"]:
        raise AdapterError("Выбранный выпуск отсутствует в истории сессии.")
    mapping["selected_id"] = forecast_id


def selected_bundle(mapping):
    initialize_state(mapping)
    return mapping["history"].get(mapping["selected_id"])
