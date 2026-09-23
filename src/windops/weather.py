"""Original NOAA GFS: bounded HTTP Range downloads and ecCodes point extraction.

The cache contains both sites, five fields, the original index and HTTP/GRIB
evidence. It does not contain whole global grids. No observational substitutes.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
import math
import re
import threading
import time

import requests

from .core import (BackendError, SITES, WEATHER_SCHEMA, atomic_json, candidate_runs,
                   data_root, digest, iso, now, read_json, stamp, validate_request)

BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
FIELDS = {
    ("TMP", "2 m above ground"): "temperature_2m_c",
    ("UGRD", "10 m above ground"): "wind_u_10m_ms",
    ("VGRD", "10 m above ground"): "wind_v_10m_ms",
    ("UGRD", "100 m above ground"): "wind_u_100m_ms",
    ("VGRD", "100 m above ground"): "wind_v_100m_ms",
}
_local = threading.local()


def session():
    if not hasattr(_local, "http"):
        _local.http = requests.Session()
        _local.http.headers["User-Agent"] = "WindOps-HackAlem/1.0"
    return _local.http


def object_url(run: datetime, lead: int) -> str:
    run = stamp(run)
    if run.hour not in (0, 6, 12, 18) or run.minute or run.second or run.microsecond or type(lead) is not int or not 1 <= lead <= 120:
        raise BackendError("INVALID_GFS_STEP", "Нужен цикл GFS 00/06/12/18 UTC и почасовой шаг 1–120.")
    return f"{BASE}/gfs.{run:%Y%m%d}/{run:%H}/atmos/gfs.t{run:%H}z.pgrb2.0p25.f{lead:03d}"


def _get(url, *, headers=None, max_bytes=2_000_000, expected_status=200):
    for attempt in range(3):
        try:
            with session().get(url, headers=headers or {}, stream=True, timeout=(15, 60), allow_redirects=False) as response:
                if response.status_code in (429, 500, 502, 503, 504):
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                if response.status_code != expected_status:
                    raise BackendError("HTTP_STATUS", f"NOAA вернул HTTP {response.status_code}; ожидается {expected_status}.", response.status_code >= 500)
                if int(response.headers.get("Content-Length", 0)) > max_bytes:
                    raise BackendError("DOWNLOAD_LIMIT", "Ответ превышает разрешённый размер.")
                pieces, size = [], 0
                for block in response.iter_content(128 * 1024):
                    size += len(block)
                    if size > max_bytes:
                        raise BackendError("DOWNLOAD_LIMIT", "Ответ превышает разрешённый размер.")
                    pieces.append(block)
                return b"".join(pieces), dict(response.headers)
        except requests.RequestException:
            if attempt == 2:
                raise BackendError("WEATHER_NETWORK", "Не удалось прочитать NOAA после трёх попыток.", True) from None
            time.sleep(2 ** attempt)
    raise BackendError("WEATHER_NETWORK", "NOAA временно недоступен.", True)


def index_ranges(payload: str, run: datetime, lead: int):
    entries = []
    for line in payload.splitlines():
        parts = line.split(":")
        if len(parts) < 6:
            raise BackendError("INVALID_INDEX", "Некорректная строка индекса GRIB.")
        try:
            entries.append((int(parts[1]), parts))
        except ValueError:
            raise BackendError("INVALID_INDEX", "Некорректное смещение GRIB.") from None
    if any(right[0] <= left[0] for left, right in zip(entries, entries[1:])):
        raise BackendError("INVALID_INDEX", "Смещения индекса не возрастают.")
    result = []
    for i, (start, parts) in enumerate(entries):
        key = (parts[3], parts[4])
        if key not in FIELDS:
            continue
        if parts[2] != f"d={run:%Y%m%d%H}" or parts[5] != f"{lead} hour fcst" or i + 1 == len(entries):
            raise BackendError("WRONG_GFS_TIME", "Время или шаг в индексе не совпадает с запросом.")
        end = entries[i + 1][0] - 1
        if end - start + 1 > 8_000_000:
            raise BackendError("DOWNLOAD_LIMIT", "Слишком большое GRIB-сообщение.")
        result.append({"start": start, "end": end, "field": FIELDS[key], "variable": key[0], "height_m": int(key[1].split()[0])})
    if len(result) != len(FIELDS) or {r["field"] for r in result} != set(FIELDS.values()):
        raise BackendError("MISSING_FIELDS", "В выпуске отсутствуют необходимые поля GFS.")
    return result


def _modified(headers) -> datetime:
    try:
        return stamp(parsedate_to_datetime(headers["Last-Modified"]))
    except (KeyError, TypeError, ValueError):
        raise BackendError("NO_AVAILABILITY", "Нет корректного Last-Modified архивного объекта.") from None


def decode_field(payload: bytes, spec, run, lead):
    import eccodes as ec
    if len(payload) < 20 or payload[:4] != b"GRIB" or payload[7] != 2 or payload[-4:] != b"7777" or int.from_bytes(payload[8:16], "big") != len(payload):
        raise BackendError("INVALID_GRIB", "Повреждённое GRIB2-сообщение.")
    try:
        handle = ec.codes_new_from_message(payload)
    except Exception:
        raise BackendError("INVALID_GRIB", "ecCodes не смог открыть сообщение GRIB2.") from None
    try:
        expected_units = "K" if spec["variable"] == "TMP" else "m s**-1"
        if (ec.codes_get(handle, "dataDate") != int(run.strftime("%Y%m%d"))
                or ec.codes_get(handle, "dataTime") != run.hour * 100
                or int(ec.codes_get(handle, "endStep")) != lead
                or ec.codes_get(handle, "typeOfLevel") != "heightAboveGround"
                or int(ec.codes_get(handle, "level")) != spec["height_m"]
                or ec.codes_get(handle, "units") != expected_units):
            raise BackendError("WRONG_GRIB_METADATA", "Время, высота или единицы GRIB не совпадают с запросом.")
        # parameterNumber is stable across ecCodes changes to shortName.
        category = ec.codes_get(handle, "parameterCategory")
        number = ec.codes_get(handle, "parameterNumber")
        if (category, number) != {"TMP": (0, 0), "UGRD": (2, 2), "VGRD": (2, 3)}[spec["variable"]]:
            raise BackendError("WRONG_GRIB_VARIABLE", "Индекс и параметр GRIB не совпадают.")
        points = {}
        for site, coords in SITES.items():
            point = ec.codes_grib_find_nearest(handle, coords["latitude"], coords["longitude"], npoints=1)[0]
            value = float(point["value"])
            if not math.isfinite(value) or value == ec.CODES_MISSING_DOUBLE:
                raise BackendError("MISSING_VALUE", "Нет значения в ближайшем узле GFS.")
            if spec["variable"] == "TMP":
                value -= 273.15
            points[site] = {"value": value, "grid_latitude": float(point["lat"]),
                            "grid_longitude": float(point["lon"]), "distance_km": float(point["distance"])}
        return points
    except BackendError:
        raise
    except Exception:
        raise BackendError("GRIB_DECODE_FAILED", "ecCodes не смог прочитать необходимые поля или точки GRIB2.") from None
    finally:
        ec.codes_release(handle)


def cache_path(run, lead, root=None):
    return (root or data_root()) / "weather" / "gfs" / f"{run:%Y%m%dT%H}" / f"f{lead:03d}.json"


def load_step(run, lead, *, root=None, offline=False):
    run = stamp(run)
    url = object_url(run, lead)
    path = cache_path(run, lead, root)
    if path.exists():
        document = read_json(path)
        payload = document.get("payload", {})
        if document.get("sha256") != digest(payload) or payload.get("run_initialized_at") != iso(run) or payload.get("gfs_lead_hours") != lead or payload.get("schema_version") != WEATHER_SCHEMA:
            raise BackendError("CORRUPT_CACHE", "Контрольная сумма или схема погодного кэша неверна.")
        return payload
    if offline:
        raise BackendError("CACHE_MISS", f"Нет GFS {run:%Y-%m-%d %H} f{lead:03d} в кэше.")
    index, idx_headers = _get(url + ".idx")
    try:
        index_text = index.decode("ascii")
    except UnicodeError:
        raise BackendError("INVALID_INDEX", "Индекс NOAA не является текстом ASCII.") from None
    ranges = index_ranges(index_text, run, lead)
    points = {site: dict(coords) for site, coords in SITES.items()}
    evidence, times, etags = [], [], set()
    for spec in ranges:
        start, end = spec["start"], spec["end"]
        headers = {"Range": f"bytes={start}-{end}"}
        if etags:
            headers["If-Match"] = next(iter(etags))
        raw, meta = _get(url, headers=headers, max_bytes=end - start + 1, expected_status=206)
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", meta.get("Content-Range", ""))
        if not match or (int(match[1]), int(match[2])) != (start, end) or len(raw) != end - start + 1:
            raise BackendError("INVALID_RANGE", "Сервер не подтвердил точный диапазон байтов.")
        if not meta.get("ETag"):
            raise BackendError("NO_OBJECT_VERSION", "Нет ETag архивного объекта.")
        etags.add(meta["ETag"])
        if len(etags) != 1:
            raise BackendError("OBJECT_CHANGED", "Объект изменился во время загрузки.")
        modified = _modified(meta)
        if modified < run:
            raise BackendError("NO_AVAILABILITY", "Время объекта предшествует инициализации GFS.")
        times.append(modified)
        decoded = decode_field(raw, spec, run, lead)
        for site, item in decoded.items():
            point = points[site]
            if "grid_latitude" in point and (point["grid_latitude"], point["grid_longitude"]) != (item["grid_latitude"], item["grid_longitude"]):
                raise BackendError("GRID_MISMATCH", "Поля извлечены из разных узлов.")
            point[spec["field"]] = item.pop("value")
            point.update(item)
        evidence.append({**spec, "sha256": digest(raw), "byte_count": len(raw), "last_modified": iso(modified), "etag": meta["ETag"]})
    for point in points.values():
        for height in (10, 100):
            u, v = point[f"wind_u_{height}m_ms"], point[f"wind_v_{height}m_ms"]
            point[f"wind_speed_{height}m_ms"] = math.hypot(u, v)
            point[f"wind_direction_{height}m_deg"] = (math.degrees(math.atan2(-u, -v)) + 360) % 360 if u or v else 0.0
    payload = {"schema_version": WEATHER_SCHEMA, "provider": "NOAA", "provider_model": "gfs_pgrb2.0p25",
               "source_url": url, "run_initialized_at": iso(run), "gfs_lead_hours": lead,
               "target_time": iso(run + timedelta(hours=lead)), "retrieved_at": now(),
               "availability_upper_bound": iso(max(times)),
               "availability_basis": "NOAA public S3 object Last-Modified: upper bound for this version, not initial publication time",
               "index_sha256": digest(index), "index": index_text, "index_last_modified": idx_headers.get("Last-Modified"),
               "fields": evidence, "points": points}
    atomic_json(path, {"sha256": digest(payload), "payload": payload})
    return payload


def weather_for_run(origin, horizon=48, *, run=None, root=None, offline=False, workers=4):
    origin = validate_request("turbine_1", origin, horizon)
    run = stamp(run) if run is not None else candidate_runs(origin)[0]
    leads = [int((origin + timedelta(hours=step) - run).total_seconds() / 3600) for step in range(1, horizon + 1)]
    if run > origin:
        raise BackendError("FUTURE_WEATHER", "Инициализация GFS позже момента прогноза.")
    for lead in leads:
        object_url(run, lead)  # Validate the whole requested range before I/O.
    first = load_step(run, leads[0], root=root, offline=offline)
    if not run <= stamp(first["availability_upper_bound"]) <= origin:
        raise BackendError("FUTURE_WEATHER", "Первый объект выпуска ещё не был доступен на момент прогноза.")
    steps = {leads[0]: first}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 24))) as pool:
        pending = {pool.submit(load_step, run, lead, root=root, offline=offline): lead for lead in leads[1:]}
        for future in as_completed(pending):
            steps[pending[future]] = future.result()
    rows, evidence = [], []
    for step, lead in enumerate(leads, 1):
        item = steps[lead]
        upper = stamp(item["availability_upper_bound"])
        if upper > origin or upper < run:
            raise BackendError("FUTURE_WEATHER", "Версия архивного объекта не подтверждена как доступная на момент прогноза.")
        if stamp(item["target_time"]) != origin + timedelta(hours=step):
            raise BackendError("WRONG_GFS_TIME", "Целевой час кэша не совпадает с запросом.")
        evidence.append({"gfs_lead_hours": lead, "fields": item["fields"], "source_url": item["source_url"]})
        for site, point in item["points"].items():
            rows.append({"site_id": site, "forecast_origin": iso(origin), "target_time": item["target_time"],
                         "run_initialized_at": iso(run), "availability_upper_bound": iso(upper),
                         "availability_basis": item["availability_basis"], "lead_hours": step, "gfs_lead_hours": lead,
                         "provider": item["provider"], "provider_model": item["provider_model"],
                         "quality_flag": "verified_object_version", **point})
    identity = {"schema": WEATHER_SCHEMA, "origin": iso(origin), "run": iso(run), "rows": rows, "evidence": evidence}
    version = "gfs-" + digest(identity)[:24]
    for row in rows:
        row["weather_version"] = version
    result = {"weather_version": version, "schema_version": WEATHER_SCHEMA, "forecast_origin": iso(origin),
              "run_initialized_at": iso(run), "horizon_hours": horizon, "rows": rows, "evidence": evidence,
              "available_at": max(row["availability_upper_bound"] for row in rows),
              "availability_basis": "upper bound from original NOAA public S3 objects",
              "retrieved_at": max(item["retrieved_at"] for item in steps.values()), "assembled_at": now()}
    validate_weather(result)
    folder = (root or data_root()) / "weather" / "bundles"
    atomic_json(folder / f"{version}.json", {"sha256": digest(result), "payload": result})
    return result


def validate_weather(bundle):
    origin = stamp(bundle["forecast_origin"])
    horizon = bundle["horizon_hours"]
    validate_request("turbine_1", origin, horizon)
    if bundle.get("schema_version") != WEATHER_SCHEMA:
        raise BackendError("WEATHER_SCHEMA", "Несовместимая схема погоды.")
    rows = bundle["rows"]
    expected = {(site, iso(origin + timedelta(hours=step))) for site in SITES for step in range(1, horizon + 1)}
    if len(rows) != len(expected) or {(r["site_id"], r["target_time"]) for r in rows} != expected:
        raise BackendError("WEATHER_GRID", "Неполная сетка или повторяющиеся часы погоды.")
    if stamp(bundle["available_at"]) != max(stamp(r["availability_upper_bound"]) for r in rows):
        raise BackendError("NO_AVAILABILITY", "Доступность набора не совпадает со свидетельствами всех часов.")
    for row in rows:
        if not stamp(row["run_initialized_at"]) <= stamp(row["availability_upper_bound"]) <= origin:
            raise BackendError("FUTURE_WEATHER", "Погода не была доступна к моменту прогноза.")
        if row["forecast_origin"] != iso(origin) or row["provider_model"] != "gfs_pgrb2.0p25" or row["weather_version"] != bundle["weather_version"]:
            raise BackendError("WEATHER_SCHEMA", "Смешаны выпуски или источники погоды.")
        if stamp(row["run_initialized_at"]) != stamp(bundle["run_initialized_at"]):
            raise BackendError("WEATHER_SCHEMA", "В одном наборе смешаны разные циклы GFS.")
        if row["lead_hours"] != (stamp(row["target_time"]) - origin).total_seconds() / 3600:
            raise BackendError("WRONG_GFS_TIME", "Шаг прогноза не совпадает с целевым часом.")
        if row["gfs_lead_hours"] != (stamp(row["target_time"]) - stamp(row["run_initialized_at"])).total_seconds() / 3600:
            raise BackendError("WRONG_GFS_TIME", "Шаг GFS не совпадает с целевым часом.")
        for field in (*FIELDS.values(), "wind_speed_10m_ms", "wind_speed_100m_ms", "wind_direction_10m_deg", "wind_direction_100m_deg"):
            value = row.get(field)
            if type(value) not in (float, int) or not math.isfinite(value):
                raise BackendError("MISSING_VALUE", f"Нет конечного числового значения {field}.")
        if min(row["wind_speed_10m_ms"], row["wind_speed_100m_ms"]) < 0:
            raise BackendError("INVALID_WIND", "Скорость ветра не может быть отрицательной.")
        if any(not 0 <= row[f"wind_direction_{height}m_deg"] < 360 for height in (10, 100)):
            raise BackendError("INVALID_WIND", "Направление ветра должно быть в диапазоне [0, 360) градусов.")
    return {"valid": True, "rows": len(rows), "weather_version": bundle["weather_version"]}


def fetch_weather(origin, horizon=48, *, root=None, offline=False):
    failures = []
    for run in candidate_runs(stamp(origin)):
        try:
            return weather_for_run(origin, horizon, run=run, root=root, offline=offline)
        except BackendError as exc:
            failures.append(exc.code)
    raise BackendError("NO_ELIGIBLE_WEATHER", "Нет полного допустимого выпуска GFS: " + ", ".join(failures))
