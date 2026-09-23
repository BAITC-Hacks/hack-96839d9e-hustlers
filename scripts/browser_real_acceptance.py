"""Exercise the existing Streamlit UI against real, offline backend artifacts.

Start app.py with WINDOPS_ML_MODULE=windops.ml.plugin, deterministic execution,
WINDOPS_OFFLINE=1 and the real local site configuration before running this.
No training is performed. Browser output stays in test-results/browser-real/.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from playwright.sync_api import expect, sync_playwright


def matching_rows(actual, expected, keys):
    columns = [*keys, "issued_at", "prediction", "model_version", "weather_version"]
    actual, expected = actual.copy(), expected.copy()
    for frame in (actual, expected):
        for name in ("issued_at", "target_time"):
            frame[name] = pd.to_datetime(frame[name], utc=True)
    actual = actual[columns].sort_values(keys).reset_index(drop=True)
    expected = expected[columns].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual.drop(columns="prediction"), expected.drop(columns="prediction"))
    np.testing.assert_allclose(actual.prediction, expected.prediction, rtol=0, atol=1e-14)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8502")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(root / ".playwright"))
    data_dir = (args.data_dir or root / "data").resolve()
    models_dir = (args.models_dir or root / "models").resolve()
    output = args.output_dir or root / "test-results" / "browser-real"
    output.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(data_dir / "ml/february_full_releases.csv")
    hourly = pd.read_csv(data_dir / "ml/february_hourly.csv")
    january = json.loads((data_dir / "ml/january_backtest.json").read_text())
    january_frame = pd.DataFrame(january["rows"])
    first_site = january_frame.loc[january_frame.site_id.eq(january_frame.site_id.iloc[0])]
    error = first_site.prediction - first_site.actual
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in models_dir.rglob("*") if p.is_file()}
    results = {"mode": "real_offline_deterministic", "live_llm_tested": False}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1366, "height": 768}, locale="ru-RU", accept_downloads=True)
        page.set_default_timeout(30000)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def download(label, filename):
            with page.expect_download() as info:
                page.get_by_role("button", name=label, exact=True).click()
            destination = output / filename
            info.value.save_as(str(destination))
            return pd.read_csv(destination)

        page.goto(args.url)
        expect(page.get_by_role("button", name="Сформировать прогноз", exact=True)).to_be_enabled()
        expect(page.get_by_role("switch", name="Тест интерфейса · синтетика")).not_to_be_checked()
        page.get_by_role("button", name="Сформировать прогноз", exact=True).click()
        expect(page.get_by_role("button", name="CSV выбранного выпуска", exact=True)).to_be_enabled(timeout=60000)
        expect(page.get_by_text("Допустимость входов: подтверждена", exact=True)).to_be_visible()
        selected = download("CSV выбранного выпуска", "forecast.csv")
        original = full[pd.to_datetime(full.issued_at, utc=True).eq(pd.Timestamp("2026-01-31T18:00:00Z"))]
        matching_rows(selected, original, ["site_id", "target_time"])
        assert len(selected) == 96
        results["selected_forecast_rows"] = len(selected)
        results["viewports"] = []
        for width, height in ((1366, 768), (1920, 1080), (800, 900)):
            page.set_viewport_size({"width": width, "height": height})
            page.get_by_role("tab", name="Прогноз", exact=True).scroll_into_view_if_needed()
            expect(page.locator(".js-plotly-plot").first).to_be_visible()
            assert not page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
            page.screenshot(path=str(output / f"forecast-{width}.png"), full_page=True)
            results["viewports"].append({"width": width, "height": height, "horizontal_overflow": False})
        page.set_viewport_size({"width": 1366, "height": 768})
        print("Real 96-row forecast exported and matched; three viewport checks passed", flush=True)

        page.get_by_role("tab", name="Проверка качества", exact=True).click()
        quality = page.get_by_role("tabpanel", name="Проверка качества", exact=True)
        quality.locator('input[type="file"]').set_input_files(str(data_dir / "ml/january_backtest.json"))
        page.get_by_role("button", name="Проверить артефакт", exact=True).click()
        expect(quality.get_by_text(f"{error.abs().mean():.4f}", exact=True)).to_be_visible()
        expect(quality.get_by_text(f"{error.pow(2).mean() ** .5:.4f}", exact=True)).to_be_visible()
        expect(quality.get_by_text(str(len(first_site)), exact=True)).to_be_visible()
        independent = january["provenance"].get("evaluation_independent") is not False
        if not independent:
            expect(quality.get_by_text("Разработочное сравнение: январь уже просматривался при разработке этой версии. Это не новая независимая проверка.", exact=True)).to_be_visible()
        quality.get_by_text(f"{error.abs().mean():.4f}", exact=True).scroll_into_view_if_needed()
        page.screenshot(path=str(output / "january-quality.png"), full_page=True)
        results["january_artifact_loaded"] = True
        results["january_evaluation_independent"] = independent
        print("January artifact loaded; MAE/RMSE and evaluation status checked", flush=True)

        page.get_by_role("tab", name="Агент и данные", exact=True).click()
        page.get_by_role("button", name="Проверить обновления", exact=True).click()
        expect(page.get_by_text("Новый выпуск добавлен в историю. Выберите его в боковой панели.", exact=True)).to_be_visible(timeout=60000)
        results["update_via_ui"] = True
        page.screenshot(path=str(output / "updated.png"), full_page=True)

        page.get_by_role("tab", name="Прогноз", exact=True).click()
        page.get_by_text("Исторический replay февраля", exact=True).click()
        expect(page.locator('[data-testid="stDateInput"]').filter(has_text="Дата первого выпуска").locator("input")).to_have_value("2026-01-31")
        expect(page.locator('[data-testid="stDateInput"]').filter(has_text="Дата последнего выпуска").locator("input")).to_have_value("2026-02-28")
        page.get_by_role("button", name="Запустить replay", exact=True).click()
        expect(page.get_by_text("Завершено выпусков: 58. Ошибок: 0.", exact=True)).to_be_visible(timeout=60000)
        replay = download("Все версии replay · итоговый CSV", "replay_all_versions.csv")
        matching_rows(replay, full.copy(), ["forecast_id", "site_id", "target_time"])
        assert len(replay) == 2784
        page.get_by_text("Для каждого дня беру часы 1–24 исходного выпуска предыдущего дня в 23:00", exact=True).click()
        month = download("Февраль · один прогноз на час", "february_selected.csv")
        matching_rows(month, hourly, ["site_id", "target_time"])
        assert len(month) == 1344
        assert set(month.selection_rule) == {"daily_previous_day_23"}
        assert not month.duplicated(["site_id", "target_time"]).any()
        results.update(replay_releases=58, replay_errors=0, full_replay_rows=len(replay), february_rows=len(month), downloads_match_saved_predictions=True)
        page.screenshot(path=str(output / "replay-export.png"), full_page=True)
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
        assert not errors, errors
        results["javascript_errors"] = errors
        browser.close()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == checksum for p, checksum in frozen.items())
    assert set(frozen) == {str(p) for p in models_dir.rglob("*") if p.is_file()}
    results.update(status="passed", model_artifacts_unchanged=True)
    (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
