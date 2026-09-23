"""Local UI configuration; an absent time convention is never guessed."""

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass
class Configuration:
    timezone: str | None = None
    sites: dict[str, str] = field(default_factory=dict)
    default_origin: datetime | None = None
    issues: list[str] = field(default_factory=list)


def load_configuration() -> Configuration:
    path = Path(os.environ.get("WINDOPS_CONFIG") or "config.local.json")
    result = Configuration()
    if not path.is_file():
        result.issues.append("Нет конфигурации площадок. Заполните config.local.json по config.example.json.")
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError
        zone = payload.get("timezone")
        if not isinstance(zone, str) or not zone:
            raise ValueError
        ZoneInfo(zone)
        result.timezone = zone
        sites = payload.get("sites")
        if not isinstance(sites, list) or not 1 <= len(sites) <= 2:
            result.issues.append("Укажите одну или две реальные турбины в sites.")
        else:
            for index, site in enumerate(sites, 1):
                if not isinstance(site, dict) or not isinstance(site.get("site_id"), str) or not site["site_id"].strip():
                    result.issues.append("У каждой турбины должен быть непустой site_id из backend.")
                    continue
                site_id = site["site_id"]
                if site_id in result.sites:
                    result.issues.append("Идентификаторы турбин в конфигурации повторяются.")
                result.sites[site_id] = f"Турбина {index}"
        if payload.get("normalization") != "0_1":
            result.issues.append("Не подтверждена нормировка мощности: требуется normalization = 0_1.")
        if payload.get("timestamp_convention") != "target_time = issued_at + horizon_step hours":
            result.issues.append("Не подтверждено соглашение: target_time = issued_at + horizon_step hours.")
        origin = payload.get("default_origin")
        if origin:
            parsed = datetime.fromisoformat(origin)
            if parsed.tzinfo is None:
                raise ValueError
            result.default_origin = parsed.astimezone(ZoneInfo(zone))
    except (ValueError, TypeError, OSError, ZoneInfoNotFoundError):
        result.issues.append("Конфигурация не прочитана: проверьте JSON, часовой пояс IANA и дату с UTC-смещением.")
    return result
