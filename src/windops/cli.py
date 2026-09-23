"""Participant 2 commands: audit, resumable archive, forecast, replay, updates."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import date
import json
import sys
import time
import uuid

from .core import (BackendError, SITES, WEATHER_SCHEMA, ZONE, atomic_json, daily_origins,
                   data_root, digest, iso, now, read_json, stamp)
from .weather import load_step, validate_weather, weather_for_run


def audit_scada(root):
    """Raw files remain byte-identical; this audit does not choose interval semantics."""
    import pandas as pd
    reports = {}
    for site in SITES:
        path = root / "scada" / f"{site}.csv"
        table = pd.read_csv(path)
        times = pd.to_datetime(table["Статистическое время"])
        power = pd.to_numeric(table["Нормализованная активная мощность"], errors="coerce")
        counts = times.dt.floor("h").value_counts()
        reports[site] = {"file": str(path.relative_to(root)), "sha256": digest(path.read_bytes()),
                         "rows": len(table), "first_timestamp": str(times.min()), "last_timestamp": str(times.max()),
                         "duplicate_timestamps": int(times.duplicated().sum()),
                         "missing_10minute_timestamps": len(pd.date_range(times.min(), times.max(), freq="10min").difference(times)),
                         "complete_hours": int((counts == 6).sum()), "power_min": float(power.min()), "power_max": float(power.max()),
                         "invalid_power_values": int((~power.between(0, 1)).sum()), "coordinates": SITES[site],
                         "time_basis": "Kazakhstan local time, per user; pre-March-2024 normalization not confirmed",
                         "interval_label": "not confirmed; participant 1 must document aggregation convention"}
    result = {"created_at": now(), "files": reports, "february_observations_provided": False}
    atomic_json(root / "reports" / "scada_audit.json", result)
    return result


def export_weather(root, start=None, end=None):
    """Export only complete canonical 00 UTC daily releases, excluding updates."""
    bundles = []
    for path in sorted((root / "weather" / "bundles").glob("*.json")):
        record = read_json(path)
        payload = record.get("payload", {})
        if record.get("sha256") != digest(payload):
            raise BackendError("CORRUPT_CACHE", "Контрольная сумма погодного набора неверна.")
        origin = stamp(payload["forecast_origin"])
        if start and origin < next(daily_origins(start, start)):
            continue
        if end and origin > next(daily_origins(end, end)):
            continue
        local_day = origin.astimezone(ZONE).date()
        canonical_origin = next(daily_origins(local_day, local_day))
        if origin != canonical_origin or payload["horizon_hours"] != 48 or stamp(payload["run_initialized_at"]) != origin.replace(hour=0):
            continue
        validate_weather(payload)
        bundles.append(payload)
    rows = [row for bundle in bundles for row in bundle["rows"]]
    unique = {(r["site_id"], r["forecast_origin"], r["target_time"]): r for r in rows}
    if len(unique) != len(rows):
        raise BackendError("DUPLICATE_RELEASE", "В экспорте несколько версий одного выпуска; выберите версию явно.")
    output = root / "weather" / "weather_for_ml.csv"
    if rows:
        temporary = output.with_suffix(f".{uuid.uuid4().hex}.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: (r["forecast_origin"], r["site_id"], r["target_time"])))
        temporary.replace(output)
    result = {"created_at": now(), "schema": WEATHER_SCHEMA, "complete_releases": len(bundles), "rows": len(rows),
              "path": str(output), "first_origin": min((b["forecast_origin"] for b in bundles), default=None),
              "last_origin": max((b["forecast_origin"] for b in bundles), default=None)}
    atomic_json(root / "reports" / "weather_coverage.json", result)
    return result


def collect(start, end, *, root, workers=8, offline=False, newest_first=False):
    """Download one original 00 UTC cycle/day, retaining both sites per object.

    Completion is checkpointed after each day. A repeated command resumes from
    individual hourly cache files; an error never turns into a zero weather row.
    """
    if type(workers) is not int or not 1 <= workers <= 24:
        raise BackendError("INVALID_WORKERS", "Число потоков должно быть от 1 до 24.")
    origins = list(daily_origins(start, end))
    if newest_first:
        origins.reverse()
    errors, complete = [], []
    started = time.monotonic()
    remaining = {origin: 48 for origin in origins}
    failed = set()
    # One bounded pool spans days: a slow object does not idle every other
    # worker at a day boundary. No extra parallelism or duplicate site downloads.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        for origin in origins:
            run = origin.replace(hour=0)
            first_lead = int((origin - run).total_seconds() / 3600) + 1
            for lead in range(first_lead, first_lead + 48):
                pending[pool.submit(load_step, run, lead, root=root, offline=offline)] = origin
        for future in as_completed(pending):
            origin = pending[future]
            try:
                future.result()
                remaining[origin] -= 1
                if remaining[origin] == 0 and origin not in failed:
                    bundle = weather_for_run(origin, 48, root=root, offline=True, workers=1)
                    complete.append(bundle["forecast_origin"])
                    print(json.dumps({"status": "complete", "origin": iso(origin), "releases": len(complete), "total": len(origins),
                                      "elapsed_seconds": round(time.monotonic() - started), "rows": len(bundle["rows"]) }), flush=True)
            except BackendError as exc:
                if origin not in failed:
                    failed.add(origin)
                    errors.append({"origin": iso(origin), **exc.as_dict()})
                    print(json.dumps({"status": "error", **errors[-1]}, ensure_ascii=False), flush=True)
            if remaining[origin] == 0 or origin in failed:
                atomic_json(root / "reports" / f"collection_{start}_{end}.json",
                            {"start": str(start), "end": str(end), "complete_origins": sorted(complete), "errors": errors, "updated_at": now()})
    coverage = export_weather(root)
    return {**coverage, "requested_releases": len(origins), "completed_this_request": len(complete), "errors": errors}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit")
    probe = commands.add_parser("probe")
    probe.add_argument("--run", required=True)
    probe.add_argument("--lead", required=True, type=int)
    weather = commands.add_parser("weather")
    weather.add_argument("--origin", required=True)
    weather.add_argument("--horizon", type=int, default=48, choices=(24, 48))
    weather.add_argument("--run")
    weather.add_argument("--offline", action="store_true")
    weather.add_argument("--workers", type=int, default=8)
    download = commands.add_parser("collect")
    download.add_argument("--start", required=True, type=date.fromisoformat)
    download.add_argument("--end", required=True, type=date.fromisoformat)
    download.add_argument("--workers", type=int, default=8)
    download.add_argument("--offline", action="store_true")
    download.add_argument("--newest-first", action="store_true", help="Сначала последние даты (сбор не является прогнозированием).")
    commands.add_parser("export-weather")
    forecast = commands.add_parser("forecast")
    forecast.add_argument("--site", required=True, choices=list(SITES))
    forecast.add_argument("--origin", required=True)
    forecast.add_argument("--horizon", type=int, default=48, choices=(24, 48))
    replay = commands.add_parser("replay")
    replay.add_argument("--start", required=True, type=date.fromisoformat)
    replay.add_argument("--end", required=True, type=date.fromisoformat)
    update = commands.add_parser("check-updates")
    update.add_argument("forecast_id")
    update.add_argument("--as-of")
    args = parser.parse_args(argv)
    root = data_root()
    try:
        if args.command == "audit":
            result = audit_scada(root)
        elif args.command == "probe":
            item = load_step(stamp(args.run), args.lead, root=root)
            result = {key: value for key, value in item.items() if key != "index"}
        elif args.command == "weather":
            item = weather_for_run(args.origin, args.horizon, run=args.run, root=root, offline=args.offline, workers=args.workers)
            result = {key: value for key, value in item.items() if key not in ("rows", "evidence")}
            result["rows"] = len(item["rows"])
        elif args.command == "collect":
            result = collect(args.start, args.end, root=root, workers=args.workers, offline=args.offline, newest_first=args.newest_first)
        elif args.command == "export-weather":
            result = export_weather(root)
        else:
            from . import backend
            if args.command == "forecast":
                item = backend.run_forecast(args.site, args.origin, args.horizon)
                result = {"forecast_id": item["forecast_id"], "rows": len(item["rows"]), "executor": item["executor"]}
            elif args.command == "replay":
                item = backend.run_replay(args.start, args.end)
                result = {"forecasts": len(item["bundles"]), "errors": item["errors"]}
            else:
                result = {"updated": [item["forecast_id"] for item in backend.check_updates(args.forecast_id, as_of=args.as_of)]}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result.get("errors") else 0
    except BackendError as exc:
        print(json.dumps({"status": "error", "error": exc.as_dict()}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
