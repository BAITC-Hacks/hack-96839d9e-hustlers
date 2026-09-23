"""Exports preserve original prediction precision and recheck final eligibility."""

import json

import pandas as pd

from .adapter import REQUIRED_COLUMNS, sanitize_metadata, validate_bundle


class ExportBlocked(ValueError):
    pass


def _require_final(bundle):
    result = validate_bundle(bundle)
    if not result.exportable:
        raise ExportBlocked("Итоговый экспорт запрещён: " + " | ".join(result.issues))


def _rows(bundle, final=False):
    if final:
        _require_final(bundle)
    rows = bundle.rows.copy(deep=True)
    # Restrict exports to the agreed forecast columns; arbitrary provider payloads
    # and source datasets are not part of a submission or diagnostic download.
    columns = [c for c in (*REQUIRED_COLUMNS, "p10", "p50", "p90", "forecast_id")
               if c in rows and (c not in ("p10", "p50", "p90") or rows[c].notna().any())]
    rows = rows[columns]
    if "forecast_id" not in rows:
        rows["forecast_id"] = bundle.forecast_id
    for column in ("issued_at", "target_time"):
        if column in rows:
            def iso(value):
                try:
                    stamp = pd.Timestamp(value)
                    return stamp.isoformat() if not pd.isna(stamp) else value
                except (ValueError, TypeError):
                    return value
            rows[column] = rows[column].map(iso)
    if not final:
        rows["source_mode"] = bundle.source_mode
        rows["export_kind"] = "diagnostic_invalid"
        rows["synthetic"] = bundle.source_mode == "demo"
    return rows


def _csv(rows):
    return rows.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def selected_csv(bundle, final=False) -> bytes:
    return _csv(_rows(bundle, final))


def passport_json(bundle) -> bytes:
    result = validate_bundle(bundle)
    payload = {"forecast_id": bundle.forecast_id, "site_ids": bundle.site_ids,
               "horizon_hours": bundle.horizon_hours, "timezone": bundle.timezone,
               "source_mode": bundle.source_mode, "executor": bundle.executor,
               "verification_origin": bundle.verification_origin,
               "provenance": bundle.provenance, "validation_summary": bundle.validation_summary,
               "ui_validation": {"checks": result.checks, "issues": result.issues,
                                 "exportable": result.exportable},
               "report_facts": bundle.report_facts, "events": bundle.events,
               "rows": _rows(bundle).to_dict(orient="records"),
               "notice": "Формат интеграции команды; не утверждённый официальный формат сдачи."}
    return json.dumps(sanitize_metadata(payload), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def replay_csv(bundles, final=False) -> bytes:
    if not bundles:
        raise ExportBlocked("Replay не содержит выпусков.")
    rows = pd.concat([_rows(bundle, final) for bundle in bundles], ignore_index=True)
    if final:
        keys = rows[["run_id", "site_id", "target_time"]].copy()
        keys["target_time"] = pd.to_datetime(keys["target_time"], utc=True)
        if keys.duplicated().any():
            raise ExportBlocked("Replay содержит повторяющиеся run_id + site_id + target_time.")
    return _csv(rows)


def february_csv(bundles, site_ids, timezone, rule="latest_available_before_target", final=False) -> bytes:
    """Explicit team rule: latest issuance strictly before target; no tie guessing.

    Multiple recalculations at the same historical issuance need a separate
    version-availability contract. Until then an ambiguous tie is blocked.
    """
    if rule != "latest_available_before_target":
        raise ExportBlocked("Правило выбора выпуска не согласовано.")
    if not timezone or not site_ids or len(site_ids) != len(set(site_ids)):
        raise ExportBlocked("Для сетки февраля нужны явная зона и уникальные site_id.")
    if not bundles:
        raise ExportBlocked("Нет выпусков для таблицы февраля.")
    if any(bundle.timezone != timezone for bundle in bundles):
        raise ExportBlocked("Часовые пояса выпусков не совпадают с зоной таблицы февраля.")
    rows = pd.concat([_rows(bundle, final) for bundle in bundles], ignore_index=True)
    if not {"issued_at", "target_time", "site_id"}.issubset(rows):
        raise ExportBlocked("Не хватает полей времени или site_id.")
    try:
        start = pd.Timestamp("2026-02-01", tz=timezone)
        end = pd.Timestamp("2026-03-01", tz=timezone)
        target = pd.to_datetime(rows["target_time"], utc=True, errors="raise")
        issued = pd.to_datetime(rows["issued_at"], utc=True, errors="raise")
    except (ValueError, TypeError):
        raise ExportBlocked("Время выпуска или часовой пояс некорректны.") from None
    rows = rows.assign(_target=target, _issued=issued)
    rows = rows.loc[(target >= start) & (target < end) & (issued < target)]
    if set(rows["site_id"]) - set(site_ids):
        raise ExportBlocked("В replay есть неожиданные турбины.")
    latest = rows.groupby(["site_id", "_target"])["_issued"].transform("max")
    chosen = rows.loc[rows["_issued"] == latest].copy()
    if chosen.duplicated(["site_id", "_target"]).any():
        raise ExportBlocked("Неоднозначная версия: несколько выпусков имеют один issued_at. Нужны сведения о доступности версий.")
    expected = {(site, timestamp) for site in site_ids for timestamp in pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")}
    actual = set(zip(chosen["site_id"], chosen["_target"]))
    if actual != expected:
        raise ExportBlocked(f"Неполная сетка февраля: отсутствует {len(expected - actual)} часов, лишних {len(actual - expected)}.")
    chosen = chosen.sort_values(["site_id", "_target"]).drop(columns=["_target", "_issued"])
    chosen["selection_rule"] = rule
    return _csv(chosen)
