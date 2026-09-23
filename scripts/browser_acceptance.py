"""Local visual acceptance with synthetic data only. No external services."""

import argparse
import json
import os
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(root / ".playwright"))
    output = root / "test-results" / "browser"
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1366, "height": 768}, locale="ru-RU")
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        expect(page.get_by_text("Подготовьте первый выпуск", exact=True)).to_be_visible(timeout=30000)
        expect(page.get_by_role("button", name="Сформировать прогноз", exact=True)).to_be_disabled()
        page.screenshot(path=str(output / "empty-1366.png"), full_page=True)
        page.get_by_text("Тест интерфейса · синтетика", exact=True).click()
        page.get_by_role("button", name="Загрузить тестовый выпуск", exact=True).click()
        expect(page.get_by_text("Почасовой прогноз", exact=True)).to_be_visible(timeout=30000)
        plot = page.locator(".js-plotly-plot").first
        expect(plot).to_be_visible(timeout=30000)
        expect(page.get_by_text("СИНТЕТИЧЕСКИЕ ДАННЫЕ — ТЕСТ ИНТЕРФЕЙСА.", exact=False)).to_be_visible()
        expect(page.get_by_role("button", name="CSV выбранного выпуска", exact=True)).to_be_disabled()
        for width, height in [(1366, 768), (1920, 1080), (800, 900)]:
            page.set_viewport_size({"width": width, "height": height})
            page.evaluate("window.scrollTo(0,0)")
            expect(plot).to_be_visible()
            page.screenshot(path=str(output / f"forecast-{width}.png"), full_page=True)
            overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
            assert not overflow, f"Page overflows at {width}px"
            bounds = plot.bounding_box()
            fits = bounds["y"] + bounds["height"] <= height
            if width >= 1366:
                assert fits, f"Main chart below the first viewport: {bounds}"
            results.append({"viewport": [width, height], "page_overflow": overflow, "chart_fits_first_screen": fits})
        page.set_viewport_size({"width": 1366, "height": 768})
        # Plotly's pointer interaction is exercised after scrolling into view.
        plot.scroll_into_view_if_needed()
        box = plot.bounding_box()
        page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
        expect(plot.locator(".hoverlayer")).to_be_visible()
        expect(plot.locator(".hoverlayer")).to_contain_text("Прогноз:")
        page.screenshot(path=str(output / "forecast-hover.png"))
        before_zoom = plot.evaluate("el => el._fullLayout.xaxis.range")
        plot.get_by_role("button", name="Приблизить", exact=True).click()
        page.wait_for_function("old => JSON.stringify(document.querySelector('.js-plotly-plot')._fullLayout.xaxis.range) !== JSON.stringify(old)", arg=before_zoom)
        plot.get_by_role("button", name="Сбросить масштаб", exact=True).click()
        page.wait_for_function("old => JSON.stringify(document.querySelector('.js-plotly-plot')._fullLayout.xaxis.range) === JSON.stringify(old)", arg=before_zoom)
        page.get_by_text("Какая погода использована", exact=True).click()
        expect(page.locator('[data-testid="stText"]:visible').filter(has_text="Источник:").first).to_be_visible()
        page.get_by_text("Технические поля выпуска", exact=True).click()
        table = page.get_by_role("tabpanel", name="Прогноз", exact=True).locator('[data-testid="stDataFrame"]:visible').nth(1)
        table.scroll_into_view_if_needed()
        table.hover()
        page.mouse.wheel(0, 900)
        page.wait_for_function("el => [...el.querySelectorAll('*')].some(node => node.scrollTop > 0)", arg=table.element_handle())
        page.screenshot(path=str(output / "table-scrolled.png"))
        with page.expect_download() as download_info:
            page.get_by_role("button", name="Диагностический CSV", exact=True).click()
        download = download_info.value
        assert "diagnostic_invalid" in download.suggested_filename
        download.save_as(str(output / download.suggested_filename))
        page.get_by_role("tab", name="Проверка качества", exact=True).click()
        page.screenshot(path=str(output / "quality-empty.png"), full_page=True)
        page.get_by_role("tab", name="Агент и данные", exact=True).click()
        expect(page.get_by_text("Паспорт прогноза", exact=True)).to_be_visible()
        expect(page.get_by_role("button", name="Проверить обновления", exact=True)).to_be_disabled()
        page.get_by_text("Полный JSON-паспорт", exact=True).click()
        page.screenshot(path=str(output / "passport.png"), full_page=True)
        page.get_by_role("tab", name="Прогноз", exact=True).click()
        page.get_by_role("region", name="Загрузить JSON-выпуск", exact=True).locator('input[type="file"]').set_input_files({
            "name": "invalid.json", "mimeType": "application/json", "buffer": b"{invalid-json"
        })
        page.get_by_role("button", name="Открыть выпуск", exact=True).click()
        expect(page.get_by_text("Предыдущий успешный выпуск — результат до последней неудачной операции.", exact=True)).to_be_visible()
        page.screenshot(path=str(output / "invalid-upload.png"), full_page=True)
        assert not errors, errors
        (output / "results.json").write_text(json.dumps({"checks": results, "page_errors": errors, "data": "synthetic only"}, ensure_ascii=False, indent=2), encoding="utf-8")
        browser.close()
    print("Browser acceptance passed: 1366x768, 1920x1080, 800x900, hover, zoom/reset, details, download, tabs, invalid upload.")


if __name__ == "__main__":
    main()
