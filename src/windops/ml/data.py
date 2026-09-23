"""SCADA interval semantics, quality audit and chronological, label-aware splits."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import io

import numpy as np
import pandas as pd

from windops.core import BackendError, SITES, WEATHER_SCHEMA, atomic_json, digest, iso, read_json
from .features import TIME_FIELDS, utc_series, validate_rows

SCADA_TIME = "Статистическое время"
SCADA_POWER = "Нормализованная активная мощность"
START = pd.Timestamp("2025-10-01", tz="Asia/Almaty").tz_convert("UTC")
END = pd.Timestamp("2026-02-01", tz="Asia/Almaty").tz_convert("UTC")
CUTOFFS = {name: pd.Timestamp(value).tz_convert("UTC") for name, value in {
    "selection_train": "2025-11-30T22:00:00+05:00",
    "january": "2025-12-31T22:00:00+05:00",
    "final": "2026-01-31T22:00:00+05:00",
}.items()}


@dataclass(frozen=True)
class LabelPolicy:
    interval_label: str = "start"
    scada_delay_minutes: int = 0
    timestamp_format: str = "mixed"

    def __post_init__(self):
        if self.interval_label not in ("start", "end"):
            raise ValueError("interval_label must be start or end")
        if type(self.scada_delay_minutes) is not int or self.scada_delay_minutes < 0:
            raise ValueError("SCADA delay must be a nonnegative integer")

    def description(self):
        return {**asdict(self), "timezone": "Asia/Almaty", "target_time": "start of hour",
                "semantics_confirmed": False, "delay_confirmed": False,
                "assumption": "Unconfirmed interval labels and SCADA arrival delay; available at hour end plus configured delay.",
                "quality_rule": "six valid unique on-grid observations; any duplicate/invalid/off-grid row rejects hour"}


def aggregate_scada(raw, site_id, policy, *, start=START, end=END):
    if site_id not in SITES or not {SCADA_TIME, SCADA_POWER}.issubset(raw):
        raise BackendError("SCADA_SCHEMA", "Нет требуемых полей SCADA или неверная турбина.")
    parsed = pd.to_datetime(raw[SCADA_TIME], format=policy.timestamp_format, dayfirst=True, errors="coerce")
    if parsed.dt.tz is not None:
        raise BackendError("SCADA_TIME", "Ожидались исходные местные метки без смещения.")
    # Exclude the unconfirmed pre-2024 transition before localizing timestamps.
    local_start = start.tz_convert("Asia/Almaty").tz_localize(None)
    local_end = end.tz_convert("Asia/Almaty").tz_localize(None)
    shift = pd.Timedelta(minutes=10 if policy.interval_label == "end" else 0)
    in_window = (parsed - shift >= local_start) & (parsed - shift < local_end)
    times = parsed.loc[in_window].dt.tz_localize("Asia/Almaty", ambiguous="raise", nonexistent="raise").dt.tz_convert("UTC") - shift
    power = pd.to_numeric(raw.loc[in_window, SCADA_POWER], errors="coerce")
    on_grid = times.eq(times.dt.floor("10min"))
    valid = power.between(0, 1) & np.isfinite(power)
    valid &= ~raw.loc[in_window, SCADA_POWER].map(lambda value: isinstance(value, (bool, np.bool_)))
    duplicate = times.duplicated(keep=False)
    work = pd.DataFrame({"target_time": times.dt.floor("h"), "time": times, "power": power,
                         "valid": valid, "on_grid": on_grid, "duplicate": duplicate})
    work["usable"] = work.valid & work.on_grid & ~work.duplicate
    work["valid_power"] = work.power.where(work.usable)
    groups = work.groupby("target_time")
    hourly = groups.agg(n_observations=("time", "size"), n_valid_unique=("usable", "sum"),
                        n_invalid=("valid", lambda x: int((~x).sum())),
                        n_off_grid=("on_grid", lambda x: int((~x).sum())),
                        n_duplicate_rows=("duplicate", "sum"), observed_mean=("valid_power", "mean"))
    hourly = hourly.reindex(pd.date_range(start, end, freq="h", inclusive="left"))
    counts = [name for name in hourly if name.startswith("n_")]
    hourly[counts] = hourly[counts].astype("float64").fillna(0).astype(int)
    hourly["complete"] = hourly.n_observations.eq(6) & hourly.n_valid_unique.eq(6)
    hourly["actual"] = hourly.observed_mean.where(hourly.complete)
    hourly["quality_reason"] = np.select(
        [hourly.complete, hourly.n_observations.eq(0), hourly.n_duplicate_rows.gt(0),
         hourly.n_off_grid.gt(0), hourly.n_invalid.gt(0)],
        ["complete", "missing_hour", "duplicate_timestamp", "off_grid", "invalid_power"], default="incomplete_hour")
    hourly.index.name = "target_time"
    hourly = hourly.reset_index()
    hourly.insert(0, "site_id", site_id)
    hourly["interval_start"] = hourly.target_time
    hourly["interval_end"] = hourly.target_time + pd.Timedelta(hours=1)
    hourly["label_available_at"] = hourly.interval_end + pd.Timedelta(minutes=policy.scada_delay_minutes)
    report = {"source_rows": len(raw), "invalid_timestamp_rows": int(parsed.isna().sum()),
              "outside_main_window_rows": int((parsed.notna() & ~in_window).sum()),
              "rows_in_window": len(work), "complete_hours": int(hourly.complete.sum()),
              "hours": len(hourly), "zero_power_rows_retained": int((valid & power.eq(0)).sum()),
              "quality_reasons": {str(k): int(v) for k, v in hourly.quality_reason.value_counts().items()},
              "invalid_power_rows": int((~valid).sum()), "duplicate_rows": int(duplicate.sum()),
              "off_grid_rows": int((~on_grid).sum())}
    return hourly, report


def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    for name in output:
        if isinstance(output[name].dtype, pd.DatetimeTZDtype):
            output[name] = output[name].map(lambda x: iso(x) if pd.notna(x) else None)
    temporary = path.with_suffix(".tmp")
    output.to_csv(temporary, index=False)
    temporary.replace(path)


def prepare(root, policy):
    paths = [root / "scada" / f"{site}.csv" for site in SITES] + [root / "weather" / "weather_for_ml.csv"]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise BackendError("ML_DATA_MISSING", "Отсутствуют исходные файлы: " + ", ".join(missing))
    contents = {str(p.relative_to(root)): p.read_bytes() for p in paths}
    inputs = {name: digest(value) for name, value in contents.items()}
    hourly, reports = [], {}
    for site in SITES:
        table, reports[site] = aggregate_scada(pd.read_csv(io.BytesIO(contents[f"scada/{site}.csv"])), site, policy)
        hourly.append(table)
    hourly = pd.concat(hourly, ignore_index=True)
    weather = validate_rows(pd.read_csv(io.BytesIO(contents["weather/weather_for_ml.csv"])))
    if not weather.get("provider", pd.Series(dtype=str)).eq("NOAA").all() or "provider" not in weather:
        raise BackendError("ML_WEATHER_SCHEMA", "Основная таблица должна содержать настоящий NOAA GFS.")
    for (_, _), part in weather.groupby(["site_id", "forecast_origin"]):
        if len(part) != 48 or set(part.lead_hours) != set(range(1, 49)):
            raise BackendError("ML_WEATHER_GRID", "Нужны полные 48-часовые ежедневные выпуски.")
    local_origins = weather.forecast_origin.dt.tz_convert("Asia/Almaty")
    if not (local_origins.dt.hour.eq(23) & local_origins.dt.minute.eq(0) & local_origins.dt.second.eq(0)).all():
        raise BackendError("ML_WEATHER_GRID", "В основной таблице ожидаются ежедневные выпуски в 23:00.")
    if hourly.duplicated(["site_id", "target_time"]).any():
        raise BackendError("ML_LABEL_DUPLICATE", "Почасовой факт не уникален.")
    joined = weather.merge(hourly, on=["site_id", "target_time"], how="left", validate="many_to_one")
    if len(joined) != len(weather):
        raise BackendError("ML_JOIN", "Соединение изменило число погодных примеров.")
    folder = root / "ml"
    write_csv(folder / "hourly_scada.csv", hourly)
    write_csv(folder / "examples.csv", joined)
    manifest = {"schema_version": 1, "weather_schema": WEATHER_SCHEMA, "inputs": inputs,
                "label_policy": policy.description(), "scada": reports,
                "weather_rows": len(weather), "origins": int(weather.forecast_origin.nunique()),
                "joined_complete_pairs": int(joined.complete.eq(True).sum()),
                "target_window": [iso(START), iso(END)],
                "outputs": {name: digest((folder / name).read_bytes()) for name in ("hourly_scada.csv", "examples.csv")}}
    atomic_json(folder / "prepared.json", manifest)
    return manifest


def load_prepared(root):
    folder = root / "ml"
    if not (folder / "prepared.json").exists():
        raise BackendError("ML_DATA_MISSING", "Сначала выполните python -m windops.ml.cli prepare.")
    manifest = read_json(folder / "prepared.json")
    for name, expected in manifest["outputs"].items():
        if digest((folder / name).read_bytes()) != expected:
            raise BackendError("ML_DATA_CHANGED", "Изменён подготовленный набор; повторите prepare.")
    for name, expected in manifest["inputs"].items():
        if not (root / name).exists() or digest((root / name).read_bytes()) != expected:
            raise BackendError("ML_DATA_CHANGED", "Изменены исходные данные; повторите prepare.")
    frame = pd.read_csv(folder / "examples.csv")
    for name in (*TIME_FIELDS, "interval_start", "interval_end", "label_available_at"):
        if name in TIME_FIELDS:
            frame[name] = utc_series(frame[name])
        else:
            frame[name] = pd.to_datetime(frame[name], utc=True)
    return frame, manifest


def training_rows(frame, stage):
    cutoff = CUTOFFS[stage]
    selected = frame.loc[(frame.target_time >= START) & (frame.target_time < cutoff) &
                         frame.complete.eq(True) & frame.actual.between(0, 1) &
                         (frame.label_available_at <= cutoff)].copy()
    if selected.empty or not (selected.label_available_at >= selected.interval_end).all():
        raise BackendError("ML_TRAINING_LABELS", "Нет допустимых ответов или нарушена доступность SCADA.")
    if not (selected.interval_end <= cutoff).all() or not (selected.forecast_origin < cutoff).all():
        raise BackendError("ML_TRAINING_LABELS", "Обнаружены будущие обучающие ответы/выпуски.")
    return selected.sort_values(["site_id", "forecast_origin", "target_time"]).reset_index(drop=True)


def evaluation_rows(frame, stage, *, available_labels_only=True):
    if stage not in ("december", "january"):
        raise ValueError(stage)
    month_start = pd.Timestamp("2025-12-01" if stage == "december" else "2026-01-01", tz="Asia/Almaty").tz_convert("UTC")
    month_end = pd.Timestamp("2026-01-01" if stage == "december" else "2026-02-01", tz="Asia/Almaty").tz_convert("UTC")
    mask = (frame.target_time >= month_start) & (frame.target_time < month_end)
    mask &= (frame.forecast_origin >= month_start - pd.Timedelta(hours=1)) & (frame.forecast_origin < month_end - pd.Timedelta(days=1))
    result = frame.loc[mask].copy()
    if stage == "december" and available_labels_only:
        result = result.loc[result.complete.eq(True) & result.actual.between(0, 1) & (result.label_available_at <= CUTOFFS["january"])]
    if result.empty:
        raise BackendError("ML_EVALUATION_EMPTY", f"Нет допустимых пар для {stage}.")
    return result.sort_values(["site_id", "forecast_origin", "target_time"]).reset_index(drop=True)


def assert_disjoint(train, evaluation):
    a = set(zip(train.site_id, train.target_time))
    b = set(zip(evaluation.site_id, evaluation.target_time))
    if a & b:
        raise BackendError("ML_TARGET_LEAKAGE", "Пересечение целевых часов train и evaluation.")
