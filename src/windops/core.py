"""Shared backend conventions. No UI, network, or model side effects on import."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

UTC = timezone.utc
ZONE = ZoneInfo("Asia/Almaty")
ROOT = Path(__file__).resolve().parents[2]
SITES = {
    "turbine_1": {"latitude": 43.645150, "longitude": 78.535604},
    "turbine_2": {"latitude": 43.643198, "longitude": 78.538828},
}
WEATHER_SCHEMA = "gfs-0p25-nearest-v1"
TIMESTAMP_CONVENTION = "target_time = issued_at + horizon_step hours"


class BackendError(ValueError):
    def __init__(self, code: str, message: str, retryable: bool = False):
        self.code, self.message, self.retryable = code, message, retryable
        super().__init__(f"{code}: {message}")

    def as_dict(self):
        return {"code": self.code, "message": self.message, "retryable": self.retryable}


def data_root() -> Path:
    return Path(os.environ.get("WINDOPS_DATA_DIR") or str(ROOT / "data")).resolve()


def stamp(value) -> datetime:
    try:
        result = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError("timezone missing")
        return result.astimezone(UTC)
    except (ValueError, TypeError):
        raise BackendError("INVALID_TIME", "Нужно ISO-время с UTC/смещением.") from None


def iso(value: datetime) -> str:
    return stamp(value).isoformat().replace("+00:00", "Z")


def now() -> str:
    return iso(datetime.now(UTC))


def digest(value) -> str:
    payload = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise BackendError("INVALID_ARTIFACT", f"Не удалось прочитать {path.name}.") from None


def validate_request(site_id: str, origin, horizon: int) -> datetime:
    if site_id not in SITES:
        raise BackendError("UNKNOWN_SITE", "Неизвестная турбина.")
    if type(horizon) is not int or horizon not in (24, 48):
        raise BackendError("INVALID_HORIZON", "Допустимы 24 или 48 часов.")
    origin = stamp(origin)
    if origin.minute or origin.second or origin.microsecond:
        raise BackendError("INVALID_TIME", "Момент выпуска должен лежать на почасовой сетке.")
    return origin


def daily_origins(start: date, end: date):
    if end < start or (end - start).days > 1500:
        raise BackendError("INVALID_DATES", "Неверный диапазон дат (максимум 1501 день).")
    for offset in range((end - start).days + 1):
        day = start + timedelta(days=offset)
        yield datetime.combine(day, datetime.min.time(), ZONE).replace(hour=23).astimezone(UTC)


def candidate_runs(origin: datetime):
    # Fixed 00 UTC cycle makes training and replay reproducible. Availability is
    # checked against object evidence, never inferred from an arbitrary +6 hours.
    base = stamp(origin).replace(hour=0, minute=0, second=0, microsecond=0)
    return [base, base - timedelta(days=1)]


def load_forecast(folder: Path):
    payload = read_json(folder / "bundle.json")
    checksum = read_json(folder / "bundle_checksum.json")
    if checksum.get("sha256") != digest(payload):
        raise BackendError("CORRUPT_FORECAST", "Контрольная сумма сохранённого прогноза неверна.")
    return payload


def save_forecast(folder: Path, payload):
    atomic_json(folder / "bundle_checksum.json", {"sha256": digest(payload)})
    atomic_json(folder / "bundle.json", payload)
