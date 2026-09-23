"""UI boundary: checked Python calls and local, untrusted JSON artifacts.

No weather download, model, or agent is implemented here. A configured module is
trusted application code; uploaded artifacts cannot grant themselves this trust.
"""

from dataclasses import dataclass, field
from datetime import date
import importlib
import inspect
import hashlib
import json
import math
import re
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd


REQUIRED_COLUMNS = (
    "run_id", "site_id", "issued_at", "target_time", "horizon_step",
    "lead_hours", "prediction", "model_version", "weather_version", "quality_flag",
)
TIMESTAMP_CONVENTION = "target_time = issued_at + horizon_step hours"


@dataclass
class ForecastBundle:
    forecast_id: str
    rows: pd.DataFrame
    validation_summary: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    report_facts: list = field(default_factory=list)
    events: list = field(default_factory=list)
    weather: pd.DataFrame = field(default_factory=pd.DataFrame)
    horizon_hours: int = 48
    site_ids: list[str] = field(default_factory=list)
    timezone: str | None = None
    source_mode: str = "cached"
    executor: str = "unknown"
    verification_origin: str = "uploaded_unverified"
    members: list["ForecastBundle"] = field(default_factory=list)


@dataclass
class ValidationResult:
    issues: list[str]
    checks: dict[str, bool]
    valid: bool
    exportable: bool


@dataclass
class ReplayResult:
    bundles: list[ForecastBundle]
    errors: list


class AdapterError(ValueError):
    """An actionable, sanitized error safe for display."""


def sanitize_metadata(value: Any, _depth: int = 0) -> Any:
    """Redact credentials recursively without converting numeric evidence to text."""
    if _depth > 15:
        return "[ограничена глубина]"
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            label = str(key)
            if re.search(r"token|secret|password|credential|authorization|api.?key|cookie|private.?key", label, re.I):
                result[label] = "[СКРЫТО]"
            elif re.search(r"traceback|stack.?trace|raw.?error|error.?message|exception|headers|^error$", label, re.I):
                result[label] = "[технические детали скрыты]"
            else:
                result[label] = sanitize_metadata(item, _depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [sanitize_metadata(item, _depth + 1) for item in value]
    if isinstance(value, str):
        # Free-form provider errors are not useful evidence and may carry secrets.
        if re.search(r"Traceback \(most recent|Authorization\s*:|-----BEGIN .*PRIVATE KEY", value, re.I):
            return "[технические детали скрыты]"
        value = re.sub(r"(?i)\b(?:Bearer|Basic)\s+[A-Za-z0-9_./+~=-]+", "[СКРЫТО]", value)
        value = re.sub(r"(?i)(\b(?:api[_-]?key|token|secret|password|credential)\s*[=:]\s*)[^\s,;&]+", r"\1[СКРЫТО]", value)
        value = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "[СКРЫТО]", value)
        value = re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1[СКРЫТО]@", value)
        return value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()
    return sanitize_metadata(str(value), _depth + 1)


def _aware(value: Any) -> pd.Timestamp | None:
    try:
        stamp = pd.Timestamp(value)
        return stamp.tz_convert("UTC") if stamp.tzinfo is not None and not pd.isna(stamp) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _safe_errors(errors: list) -> list[dict]:
    """Provider error text is arbitrary and is never published by the UI."""
    result = []
    for error in errors:
        safe = {"message": "Выпуск не выполнен. Подробности доступны в локальной диагностике backend."}
        if isinstance(error, dict):
            safe.update({key: sanitize_metadata(error[key]) for key in ("issued_at", "site_id", "forecast_id", "error_code") if key in error})
        result.append(safe)
    return result


def validate_bundle(bundle: ForecastBundle) -> ValidationResult:
    """Recheck original values. Never fill, clip, drop, or repair forecast rows."""
    checks: dict[str, bool] = {}
    issues: list[str] = []
    structural: list[bool] = []

    def check(name: str, passed: bool, message: str, data: bool = True) -> None:
        checks[name] = bool(passed)
        if data:
            structural.append(bool(passed))
        if not passed:
            issues.append(message)

    rows = bundle.rows
    if not isinstance(rows, pd.DataFrame) or not isinstance(bundle.provenance, dict) or not isinstance(bundle.validation_summary, dict):
        return ValidationResult(["Строки должны быть таблицей, provenance и validation_summary — объектами."], {"types": False}, False, False)
    if not isinstance(bundle.horizon_hours, int) or isinstance(bundle.horizon_hours, bool):
        return ValidationResult(["horizon_hours должен быть целым числом 24/48."], {"horizon": False}, False, False)
    if not isinstance(bundle.site_ids, (list, tuple)) or not all(isinstance(site, str) for site in bundle.site_ids):
        return ValidationResult(["site_ids должен быть списком строк."], {"expected_sites": False}, False, False)
    nonscalar = [name for name in REQUIRED_COLUMNS if name in rows and rows[name].map(lambda value: isinstance(value, (dict, list, tuple, set))).any()]
    if nonscalar:
        return ValidationResult(["Поля строк должны содержать скалярные значения: " + ", ".join(nonscalar)], {"row_scalars": False}, False, False)
    check("forecast_id", isinstance(bundle.forecast_id, str) and bool(bundle.forecast_id.strip()), "Не задан forecast_id выпуска.")
    check("horizon", bundle.horizon_hours in (24, 48), "Горизонт должен быть 24 или 48 часов.")
    expected_sites = bundle.site_ids
    check("expected_sites", 1 <= len(expected_sites) <= 2 and len(expected_sites) == len(set(expected_sites)) and all(isinstance(s, str) and s.strip() for s in expected_sites), "Нужны одна или две уникальные турбины site_id.")
    try:
        ZoneInfo(bundle.timezone or "")
        known_zone = True
    except (ValueError, TypeError, ZoneInfoNotFoundError):
        known_zone = False
    check("timezone", known_zone, "Не указан известный часовой пояс выпуска.")
    check("timestamp_convention", bundle.provenance.get("timestamp_convention") == TIMESTAMP_CONVENTION, "Не подтверждено соглашение: target_time = issued_at + horizon_step hours.")
    missing = [name for name in REQUIRED_COLUMNS if name not in rows.columns]
    check("required_fields", not missing, "Отсутствуют обязательные поля: " + ", ".join(missing))
    check("nonempty", not rows.empty, "Выпуск не содержит прогнозных строк.")
    for name in ("run_id", "site_id", "model_version", "weather_version", "quality_flag"):
        if name in rows:
            check(name, bool(rows[name].map(lambda v: isinstance(v, str) and bool(v.strip())).all()), f"Поле {name} содержит пустые или неверные идентификаторы.")
    if "site_id" in rows:
        actual_sites = set(rows["site_id"].dropna().astype(str))
        check("site_ids", actual_sites == set(expected_sites), "Набор турбин в строках не совпадает с ожидаемыми site_id.")
    if "forecast_id" in rows and not bundle.members:
        check("row_forecast_ids", bool(rows["forecast_id"].eq(bundle.forecast_id).all()), "forecast_id строк не совпадает с идентификатором выбранного выпуска.")
    if {"site_id", "target_time"}.issubset(rows):
        check("flat_unique", not rows.duplicated(["site_id", "target_time"]).any(), "Обнаружены дубли site_id + target_time в выбранной версии.")
    if {"run_id", "site_id", "target_time"}.issubset(rows):
        check("run_unique", not rows.duplicated(["run_id", "site_id", "target_time"]).any(), "Обнаружены дубли run_id + site_id + target_time.")
        check("single_run_per_site", bool(rows.groupby("site_id")["run_id"].nunique().le(1).all()), "В одной турбине смешаны строки разных run_id.")
    numeric = {}
    supplied = {}
    for name in ("prediction", "p10", "p50", "p90"):
        if name not in rows:
            continue
        if name != "prediction" and rows[name].isna().all():
            continue  # Optional schema fields serialized as null are absent data.
        active = pd.Series(True, index=rows.index)
        if name != "prediction" and "site_id" in rows:
            active = rows.groupby("site_id")[name].transform(lambda values: values.notna().any()).fillna(False).astype(bool)
        supplied[name] = active
        check(name + "_numeric_type", not bool(rows[name].map(pd.api.types.is_bool).any()), f"Поле {name} не должно содержать логические значения true/false.")
        numeric[name] = pd.to_numeric(rows[name], errors="coerce")
        finite = numeric[name].map(lambda v: pd.notna(v) and math.isfinite(v))
        check(name + "_finite", bool(finite[active].all()), f"Поле {name} содержит NaN/inf или нечисловые значения внутри предоставленного ряда турбины.")
        check(name + "_range", bool(numeric[name][active].between(0, 1).all()), f"Поле {name} выходит за диапазон 0–1.")
    for left, right in (("p10", "p50"), ("p50", "p90"), ("p10", "p90")):
        if left in numeric and right in numeric:
            common = supplied[left] & supplied[right]
            check(left + "_" + right, bool((numeric[left][common] <= numeric[right][common]).all()), f"Нарушен порядок квантилей: {left} > {right}.")
    # A point prediction need not equal p50 and need not lie between quantiles.
    issued = rows["issued_at"].map(_aware) if "issued_at" in rows else pd.Series(dtype=object)
    targets = rows["target_time"].map(_aware) if "target_time" in rows else pd.Series(dtype=object)
    times_ok = bool(len(issued) and len(targets) and issued.notna().all() and targets.notna().all())
    check("aware_times", times_ok, "issued_at и target_time должны быть корректными ISO-временами с UTC/смещением.")
    origin = issued.iloc[0] if times_ok else None
    if times_ok:
        check("single_origin", issued.nunique() == 1, "В выбранной версии смешаны разные issued_at.")
        if "site_id" in rows:
            normalized_keys = pd.DataFrame({"site_id": rows["site_id"], "target_time": targets})
            check("normalized_unique", not normalized_keys.duplicated().any(), "Обнаружены дубли целевого часа, в том числе с разными UTC-смещениями.")
        check("hour_alignment", all(t.minute == 0 and t.second == 0 and t.microsecond == 0 for t in [*issued, *targets]), "Временные метки должны лежать на почасовой сетке.")
    for name in ("horizon_step", "lead_hours"):
        if name in rows:
            values = pd.to_numeric(rows[name], errors="coerce")
            check(name + "_numeric_type", not bool(rows[name].map(pd.api.types.is_bool).any()), f"Поле {name} не должно содержать логические значения true/false.")
            check(name + "_integer", bool((values.notna() & (values % 1 == 0) & values.between(1, bundle.horizon_hours)).all()), f"Поле {name} должно содержать целые шаги 1…{bundle.horizon_hours}.")
            if times_ok:
                elapsed = pd.Series([(target - release).total_seconds() / 3600 for target, release in zip(targets, issued)], index=rows.index)
                check(name + "_time", bool(values.eq(elapsed).all()), f"Поле {name} не совпадает с target_time − issued_at.")
    if times_ok and "site_id" in rows and bundle.horizon_hours in (24, 48):
        expected = set(pd.date_range(origin + pd.Timedelta(hours=1), periods=bundle.horizon_hours, freq="h"))
        grid_ok = all(set(targets[rows["site_id"] == site]) == expected and int((rows["site_id"] == site).sum()) == bundle.horizon_hours for site in expected_sites)
        check("hourly_grid", grid_ok, "Прогноз не совпадает с непрерывной ожидаемой сеткой часов для каждой турбины.")
    if bundle.members:
        flat_members = all(isinstance(member, ForecastBundle) and not member.members for member in bundle.members)
        check("flat_members", flat_members, "Группа должна содержать только исходные выпуски backend.")
        if not flat_members:
            return ValidationResult(issues, checks, False, False)
        member_rows = []
        for member in bundle.members:
            part = member.rows.copy(deep=True)
            part["forecast_id"] = member.forecast_id
            member_rows.append(part)
        expected_rows = pd.concat(member_rows, ignore_index=True)
        check("member_rows_unchanged", rows.reset_index(drop=True).equals(expected_rows), "Строки группы изменены относительно исходных выпусков.")
        check("member_parameters", all(member.horizon_hours == bundle.horizon_hours and member.timezone == bundle.timezone and member.source_mode == bundle.source_mode for member in bundle.members), "Параметры группы не совпадают с исходными выпусками.")
        check("member_sites", set(expected_sites) == {site for member in bundle.members for site in member.site_ids}, "Турбины группы не совпадают с исходными выпусками.")
        for index, member in enumerate(bundle.members, 1):
            result = validate_bundle(member)
            structural.append(result.valid)
            checks[f"member_{index}_exportable"] = result.exportable
            issues.extend(f"Турбина {index}: {issue}" for issue in result.issues)
        return ValidationResult(issues, checks, all(structural), all(checks.values()))
    provenance = bundle.provenance
    weather = provenance.get("weather", {})
    model = provenance.get("model", {})
    weather = weather if isinstance(weather, dict) else {}
    model = model if isinstance(model, dict) else {}
    check("real_source", bundle.source_mode in ("real", "cached"), "Синтетический или неизвестный источник: итоговый экспорт запрещён.", False)
    if "quality_flag" in rows:
        synthetic_rows = rows["quality_flag"].astype(str).str.contains(r"synthetic|demo|синтет", case=False, regex=True)
        check("no_synthetic_rows", not bool(synthetic_rows.any()), "Строки помечены как синтетические; итоговый экспорт запрещён.", False)
    check("verified_origin", bundle.verification_origin == "backend", "Происхождение не подтверждено доверенным backend; загруженный JSON содержит только заявления автора.", False)
    check("inputs_verified", bundle.validation_summary.get("inputs_verified") is True, "Backend не подтвердил допустимость входных данных.", False)
    check("model_verified", bundle.validation_summary.get("model_verified") is True, "Backend не подтвердил допустимость модели.", False)
    check("weather_source", isinstance(weather.get("source"), str) and bool(weather.get("source")) and isinstance(weather.get("version"), str) and bool(weather.get("version")), "Не указаны источник и версия погоды.", False)
    check("archived_weather", weather.get("data_kind") == "archived_forecast" and weather.get("synthetic") is False, "Нужен реальный архивный прогноз погоды; наблюдения и синтетика недопустимы.", False)
    available = _aware(weather.get("available_at"))
    check("weather_available", available is not None and origin is not None and available <= origin, "Погодный выпуск не подтверждён как доступный на issued_at (проверяется available_at, не retrieved_at).", False)
    weather_issue = _aware(weather.get("issued_at", weather.get("run_initialized_at")))
    if "issued_at" in weather or "run_initialized_at" in weather:
        check("weather_issue", weather_issue is not None and available is not None and weather_issue <= available, "Время погодного выпуска некорректно или позже заявленной доступности.", False)
    cutoff = _aware(model.get("training_cutoff"))
    check("training_cutoff", cutoff is not None and origin is not None and cutoff <= origin, "Тренировочный cutoff отсутствует или находится после issued_at.", False)
    check("historical_model", model.get("historically_eligible") is True, "Историческая допустимость модели не подтверждена.", False)
    check("normalization", model.get("normalization") == "0_1", "Не подтверждена нормировка мощности 0–1.", False)
    for column, metadata in (("model_version", model), ("weather_version", weather)):
        if column in rows:
            version = metadata.get("version")
            check(column + "_provenance", isinstance(version, str) and bool(version) and set(rows[column].dropna()) == {version}, f"{column} не совпадает с версией в паспорте.", False)
    return ValidationResult(issues, checks, all(structural), all(checks.values()))


def _bundle(value: Any, trusted: bool = False) -> ForecastBundle:
    if isinstance(value, ForecastBundle):
        # Own copies avoid changes leaking into a backend cache or other sessions.
        value = {name: getattr(value, name) for name in ForecastBundle.__dataclass_fields__}
    elif hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, dict):
        raise AdapterError("Backend должен вернуть ForecastBundle или объект JSON выпуска.")
    if "rows" not in value or not isinstance(value.get("forecast_id"), str):
        raise AdapterError("В JSON нужны forecast_id и массив rows; см. docs/INTEGRATION.md.")
    try:
        rows = pd.DataFrame(value["rows"]).copy(deep=True)
        weather = pd.DataFrame(value.get("weather", [])).copy(deep=True)
    except (ValueError, TypeError) as exc:
        raise AdapterError("rows и weather должны быть таблицами записей.") from None
    for name in ("provenance", "validation_summary"):
        if not isinstance(value.get(name, {}), dict):
            raise AdapterError(f"Поле {name} должно быть объектом JSON.")
    sites = value.get("site_ids", [])
    if not isinstance(sites, (list, tuple)) or any(not isinstance(site, str) for site in sites):
        raise AdapterError("site_ids должен быть списком строк.")
    horizon = value.get("horizon_hours", 48)
    if not isinstance(horizon, int) or isinstance(horizon, bool):
        raise AdapterError("horizon_hours должен быть целым числом.")
    for name in ("report_facts", "events"):
        if not isinstance(value.get(name, []), list):
            raise AdapterError(f"Поле {name} должно быть массивом.")
    if value.get("timezone") is not None and not isinstance(value["timezone"], str):
        raise AdapterError("timezone должен быть строкой с именем часового пояса.")
    mode = str(value.get("source_mode", "cached"))
    return ForecastBundle(
        forecast_id=value["forecast_id"], rows=rows,
        validation_summary=sanitize_metadata(value.get("validation_summary", {})),
        provenance=sanitize_metadata(value.get("provenance", {})),
        report_facts=sanitize_metadata(value.get("report_facts", [])),
        events=sanitize_metadata(value.get("events", [])), weather=weather,
        horizon_hours=horizon, site_ids=list(sites), timezone=value.get("timezone"),
        source_mode=mode if trusted or mode == "demo" else "cached",
        executor=str(sanitize_metadata(value.get("executor", "unknown"))),
        verification_origin="backend" if trusted else "uploaded_unverified",
    )


def combine_bundles(bundles: list[ForecastBundle]) -> ForecastBundle:
    """Group coherent per-site releases for display, preserving every original ID."""
    if not bundles:
        raise AdapterError("Нет выпусков для объединения.")
    if len(bundles) == 1:
        return bundles[0]
    if len(bundles) > 2 or any(bundle.members for bundle in bundles):
        raise AdapterError("Для общего представления нужны один или два исходных выпуска.")
    first = bundles[0]
    origins = []
    rows = []
    sites = []
    for bundle in bundles:
        if bundle.horizon_hours != first.horizon_hours or bundle.timezone != first.timezone or bundle.source_mode != first.source_mode:
            raise AdapterError("Объединение требует одинаковых горизонта, зоны и режима данных.")
        if "issued_at" not in bundle.rows or bundle.rows.empty:
            raise AdapterError("Выпуски без issued_at нельзя объединить.")
        releases = bundle.rows["issued_at"].map(_aware)
        if releases.isna().any() or releases.nunique() != 1:
            raise AdapterError("Для объединения нужен один корректный момент выпуска.")
        origins.append(releases.iloc[0])
        part = bundle.rows.copy(deep=True)
        part["forecast_id"] = bundle.forecast_id
        rows.append(part)
        sites.extend(bundle.site_ids)
    if len(set(origins)) != 1 or len(set(sites)) != len(sites) or len(sites) > 2:
        raise AdapterError("Объединение требует общего issued_at и разных турбин.")
    digest = hashlib.sha256(json.dumps([bundle.forecast_id for bundle in bundles]).encode()).hexdigest()[:16]
    weather = [bundle.weather for bundle in bundles if not bundle.weather.empty]
    return ForecastBundle(
        forecast_id="ui-group-" + digest, rows=pd.concat(rows, ignore_index=True),
        provenance={"timestamp_convention": TIMESTAMP_CONVENTION, "ui_group": True,
                    "per_site": {site: bundle.provenance for bundle in bundles for site in bundle.site_ids},
                    "member_forecast_ids": [bundle.forecast_id for bundle in bundles]},
        report_facts=[fact for bundle in bundles for fact in bundle.report_facts],
        events=[event for bundle in bundles for event in bundle.events],
        weather=pd.concat(weather, ignore_index=True) if weather else pd.DataFrame(),
        horizon_hours=first.horizon_hours, site_ids=sites, timezone=first.timezone,
        source_mode=first.source_mode,
        executor=", ".join(dict.fromkeys(bundle.executor for bundle in bundles)),
        verification_origin="ui_group", members=list(bundles),
    )


def load_bundle_json(payload: bytes | str) -> ForecastBundle:
    if len(payload) > 20_000_000:
        raise AdapterError("JSON превышает ограничение 20 МБ.")
    try:
        value = json.loads(payload)
    except (ValueError, UnicodeError, TypeError):
        raise AdapterError("Не удалось прочитать JSON. Проверьте UTF-8 и структуру выпуска.") from None
    return _bundle(value, trusted=False)


class BackendAdapter:
    """Only this module imports and calls a explicitly configured backend module."""

    def __init__(self, module_name: str | None = None):
        self.capabilities = dict.fromkeys(("run_forecast", "run_replay", "check_updates"), False)
        self.status = "Backend не подключён: задайте WINDOPS_BACKEND_MODULE после согласования контракта."
        self._module = None
        if not module_name:
            return
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module_name):
            self.status = "Некорректное имя Python-модуля в WINDOPS_BACKEND_MODULE."
            return
        try:
            self._module = importlib.import_module(module_name)
        except Exception:
            self.status = "Модуль backend не загрузился. Проверьте установку и его зависимости; синтетика не включена."
            return
        expected = {
            "run_forecast": {"site_id": "site", "forecast_origin": "2026-02-01T00:00:00+00:00", "horizon_hours": 24},
            "run_replay": {"start_date": date(2026, 2, 1), "end_date": date(2026, 2, 28)},
            "check_updates": {"forecast_id": "forecast"},
        }
        for name, kwargs in expected.items():
            fn = getattr(self._module, name, None)
            if callable(fn):
                try:
                    inspect.signature(fn).bind(**kwargs)
                    self.capabilities[name] = True
                except (ValueError, TypeError):
                    pass
        self.status = "Python-модуль подключён; доступные вызовы проверены." if any(self.capabilities.values()) else "Модуль загружен, но совместимые публичные функции не найдены."

    def _call(self, name: str, **kwargs):
        if not self.capabilities.get(name):
            raise AdapterError(f"Вызов {name} не подключён. Проверьте контракт backend.")
        try:
            return getattr(self._module, name)(**kwargs)
        except Exception:
            raise AdapterError(f"Backend не выполнил {name}. Сохранённые выпуски доступны; проверьте конфигурацию и архивы.") from None

    def run_forecast(self, site_ids: list[str], forecast_origin: Any, horizon_hours: int) -> list[ForecastBundle]:
        origin = _aware(forecast_origin)
        if not isinstance(site_ids, (list, tuple)) or not 1 <= len(site_ids) <= 2 or not all(isinstance(site, str) and site for site in site_ids) or len(site_ids) != len(set(site_ids)) or origin is None or horizon_hours not in (24, 48):
            raise AdapterError("Для запуска нужны уникальные site_id, время с зоной и горизонт 24/48.")
        if origin.minute or origin.second or origin.microsecond:
            raise AdapterError("Момент выпуска должен лежать на почасовой сетке.")
        bundles = []
        for site in site_ids:
            bundle = _bundle(self._call("run_forecast", site_id=site, forecast_origin=forecast_origin, horizon_hours=horizon_hours), trusted=True)
            row_sites = set(bundle.rows.get("site_id", pd.Series(dtype=str)).dropna())
            releases = bundle.rows.get("issued_at", pd.Series(dtype=str)).map(_aware)
            if bundle.site_ids != [site] or row_sites != {site} or bundle.horizon_hours != horizon_hours or releases.empty or not releases.eq(origin).all():
                raise AdapterError("Backend вернул выпуск с параметрами, не совпадающими с запросом: site_id, issued_at или горизонт.")
            bundles.append(bundle)
        return bundles

    def run_replay(self, start_date: date, end_date: date) -> ReplayResult:
        if not isinstance(start_date, date) or not isinstance(end_date, date) or end_date < start_date:
            raise AdapterError("Конец replay должен быть не раньше начала.")
        raw = self._call("run_replay", start_date=start_date, end_date=end_date)
        if isinstance(raw, ReplayResult):
            return ReplayResult([_bundle(value, True) for value in raw.bundles], _safe_errors(raw.errors))
        if not isinstance(raw, dict) or not isinstance(raw.get("bundles"), list) or not isinstance(raw.get("errors", []), list):
            raise AdapterError("Replay должен вернуть {bundles: [...], errors: [...]}.")
        return ReplayResult([_bundle(value, True) for value in raw["bundles"]], _safe_errors(raw.get("errors", [])))

    def check_updates(self, forecast_id: str) -> list[ForecastBundle]:
        raw = self._call("check_updates", forecast_id=forecast_id)
        if raw is None:
            return []
        if isinstance(raw, dict) and "bundles" in raw:
            raw = raw["bundles"]
        if not isinstance(raw, list):
            raw = [raw]
        bundles = [_bundle(value, True) for value in raw]
        if any(bundle.forecast_id == forecast_id for bundle in bundles):
            raise AdapterError("Обновлённый выпуск обязан иметь новый forecast_id; backend вернул прежний идентификатор.")
        return bundles
