"""WindOps AI: a Streamlit UI with no model or weather side effects on rerun."""

from datetime import date, datetime
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import streamlit as st

from windops.ui.adapter import AdapterError, BackendAdapter, load_bundle_json, validate_bundle, sanitize_metadata, combine_bundles
from windops.ui.charts import forecast_chart, weather_chart
from windops.ui.components import render_header, render_metric_cards, render_forecast_overview
from windops.ui.configuration import load_configuration
from windops.ui.exports import selected_csv, passport_json, replay_csv, february_csv
from windops.ui.fixtures import synthetic_bundle
from windops.ui.quality import render_quality
from windops.ui.state import initialize_state, begin_run, finish_run, fail_run, select_bundle, selected_bundle
from windops.ui.theme import apply_theme, PLOT_CONFIG


FEBRUARY_NOTICE = "Фактическая выработка за февраль 2026 не предоставлена. Показан прогноз; ошибка на этом периоде не рассчитана"
SOURCE_LABELS = {"real": "Архивные данные · новый расчёт", "cached": "Сохранённый выпуск", "demo": "Синтетические данные"}


def safe_text(value):
    """Untrusted descriptions are displayed as text, never interpolated HTML."""
    return str(sanitize_metadata(value))


def display_time(value, timezone=None):
    try:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is not None and timezone:
            stamp = stamp.tz_convert(timezone)
        return stamp.strftime("%d.%m.%Y %H:%M %z")
    except (ValueError, TypeError, KeyError):
        return "Нет данных"


def labels_for(bundle, configured):
    labels = dict(configured)
    if bundle.source_mode == "demo":
        labels.update({"demo-1": "Турбина 1", "demo-2": "Турбина 2"})
    for site_id in bundle.site_ids:
        labels.setdefault(site_id, site_id)
    return labels


def issue_details(result):
    with st.expander(f"Проверки и ограничения · {len(result.issues)}", expanded=False):
        if not result.issues:
            st.success("Проверки пройдены.")
        for issue in result.issues:
            st.text(safe_text(issue))


def render_forecast(bundle, configured):
    if bundle is None:
        st.markdown("### Подготовьте первый выпуск")
        st.info("Загрузите сохранённый JSON-выпуск в боковой панели или подключите модуль команды для расчёта.")
        render_metric_cards([
            {"label": "Средний прогноз", "value": "Нет данных", "detail": "Нормализованная мощность"},
            {"label": "Пиковый прогноз", "value": "Нет данных", "detail": "Почасовой горизонт"},
            {"label": "Полнота", "value": "Нет выпуска", "detail": "24 или 48 часов"},
            {"label": "Входные данные", "value": "Не проверены", "detail": "Нужен паспорт источника", "tone": "warning"},
        ])
        st.markdown("#### От данных к проверяемому прогнозу")
        st.write("Архив погоды → проверка входов → модель → почасовой прогноз → экспорт")
        st.caption("После загрузки здесь появятся график, погодные входы, таблица и паспорт выбранного выпуска.")
        st.info(FEBRUARY_NOTICE)
        return

    result = validate_bundle(bundle)
    rows = bundle.rows.copy()
    labels = labels_for(bundle, configured)
    demo = bundle.source_mode == "demo"
    if rows.empty:
        st.warning("В выбранном выпуске нет прогнозных строк.")
        issue_details(result)
        return
    if not {"site_id", "issued_at", "target_time", "prediction"}.issubset(rows.columns):
        st.error("Выпуск не содержит обязательных полей для отображения. Исправьте контракт JSON; паспорт доступен на вкладке «Агент и данные».")
        issue_details(result)
        return

    origin = rows["issued_at"].iloc[0] if "issued_at" in rows else None
    targets = pd.to_datetime(rows.get("target_time", pd.Series(dtype=str)), errors="coerce", utc=True)
    model = ", ".join(rows["model_version"].astype(str).unique()) if "model_version" in rows else "Нет данных"
    context_lines = [" · ".join([", ".join(labels.get(s, s) for s in bundle.site_ids), f"Момент выпуска: {display_time(origin, bundle.timezone)}", f"{bundle.horizon_hours} ч", f"Зона: {bundle.timezone or 'не задана'}"]),
                     f"Целевые часы: {display_time(targets.min(), bundle.timezone)} — {display_time(targets.max(), bundle.timezone)} · Модель: {safe_text(model)} · {SOURCE_LABELS.get(bundle.source_mode, 'Режим не подтверждён')}"]
    cards = []
    for site_id in bundle.site_ids:
        part = rows[rows["site_id"] == site_id] if "site_id" in rows else pd.DataFrame()
        values = pd.to_numeric(part.get("prediction", pd.Series(dtype=float)), errors="coerce")
        valid = values[values.notna() & values.between(0, 1)]
        complete_values = len(valid) == len(part) and len(valid) > 0
        peak_time = display_time(part.loc[valid.idxmax(), "target_time"], bundle.timezone) if complete_values and "target_time" in part else "Значения не прошли проверку"
        unique_hours = 0
        if "target_time" in part and origin is not None:
            try:
                expected = pd.date_range(pd.Timestamp(origin) + pd.Timedelta(hours=1), periods=bundle.horizon_hours, freq="h")
                observed = pd.to_datetime(part.loc[values.notna() & values.between(0, 1), "target_time"], utc=True, errors="coerce")
                unique_hours = len(set(observed.dropna()) & set(expected))
            except (ValueError, TypeError):
                pass
        prefix = labels.get(site_id, site_id).replace("Турбина ", "Т") + " · " if len(bundle.site_ids) > 1 else ""
        cards.extend([
            {"label": prefix + "Среднее", "value": f"{valid.mean():.3f}" if complete_values else "Нет данных", "detail": "Прогноз норм. мощности"},
            {"label": prefix + "Пик", "value": f"{valid.max():.3f}" if complete_values else "Нет данных", "detail": peak_time},
            {"label": prefix + "Полнота", "value": f"{unique_hours} из {bundle.horizon_hours}", "detail": "Целевых часов"},
        ])
    if len(bundle.site_ids) == 1:
        cards.append({"label": "Допустимость входов", "value": "Подтверждена" if result.exportable else "Не подтверждена", "detail": "См. паспорт и проверки", "tone": "success" if result.exportable else "warning"})
    render_forecast_overview(context_lines, cards, False,
                             "Допустимость входов: " + ("подтверждена" if result.exportable else "не подтверждена; см. паспорт и проверки"))
    if bundle.timezone and {"site_id", "target_time", "prediction"}.issubset(rows.columns):
        try:
            st.plotly_chart(forecast_chart(rows, bundle.timezone, labels, demo), width="stretch", config=PLOT_CONFIG, key="forecast_plot")
        except (ValueError, KeyError, TypeError):
            st.error("График недоступен: проверьте время, идентификаторы и уникальность строк в паспорте выпуска.")
    else:
        st.warning("Для графика нужны корректные прогнозные строки и подтверждённый часовой пояс.")
    if all(q in rows and rows[q].notna().any() for q in ("p10", "p90")):
        st.caption("P10–P90 — заявленный моделью прогнозный диапазон, не гарантия. Независимое покрытие не оценено на этом экране.")
    st.info(FEBRUARY_NOTICE)
    issue_details(result)

    with st.expander("Какая погода использована", expanded=False):
        for member in getattr(bundle, "members", []) or [bundle]:
            weather_meta = member.provenance.get("weather", {})
            weather_meta = weather_meta if isinstance(weather_meta, dict) else {}
            st.caption(", ".join(labels.get(s, s) for s in member.site_ids))
            st.text("Источник: " + safe_text(weather_meta.get("source", "Не указан")))
            st.text("Выпуск погоды: " + safe_text(weather_meta.get("run_initialized_at", weather_meta.get("issued_at", "Не указан"))))
            st.text("Высота ветра: " + safe_text(weather_meta.get("wind_height_m", "Не указана")))
        if bundle.weather.empty:
            st.info("Почасовые погодные входы не предоставлены в этом выпуске.")
        elif bundle.timezone:
            try:
                st.plotly_chart(weather_chart(bundle.weather, bundle.timezone, labels, demo), width="stretch", config=PLOT_CONFIG)
            except (ValueError, KeyError, TypeError):
                st.warning("Погодные входы не соответствуют формату графика. Доступна исходная таблица.")
            st.dataframe(bundle.weather, hide_index=True, width="stretch")

    with st.expander("Что важно в этом прогнозе", expanded=False):
        for site_id in bundle.site_ids:
            part = rows.loc[rows["site_id"] == site_id, "prediction"]
            numeric = pd.to_numeric(part, errors="coerce")
            if len(numeric) and numeric.notna().all() and numeric.between(0, 1).all():
                st.text(f"{labels.get(site_id, site_id)}: от {numeric.min():.3f} до {numeric.max():.3f} нормализованной мощности.")
        st.text("Квантили предоставлены моделью." if all(q in rows and rows[q].notna().any() for q in ("p10", "p90")) else "Прогнозный диапазон не предоставлен: показан точечный прогноз.")
        if bundle.report_facts:
            st.caption("Факты, переданные backend")
            for fact in bundle.report_facts:
                st.text(safe_text(fact))
        if bundle.provenance.get("llm_analysis"):
            st.caption("Комментарий LLM из backend")
            st.text(safe_text(bundle.provenance["llm_analysis"]))

    st.markdown("#### Почасовая таблица и экспорт")
    column_names = {"site_id": "Турбина", "target_time": "Целевой час", "horizon_step": "Шаг, ч", "prediction": "Прогноз", "p10": "P10", "p50": "P50", "p90": "P90", "quality_flag": "Качество"}
    visible = [c for c in column_names if c in rows and (c not in ("p10", "p50", "p90") or rows[c].notna().any())]
    table = rows[visible].copy()
    if "target_time" in table:
        table = table.sort_values(["site_id", "target_time"])
        table["target_time"] = table["target_time"].map(lambda v: display_time(v, bundle.timezone))
    if "site_id" in table:
        table["site_id"] = table["site_id"].map(lambda s: labels.get(s, s))
    if "quality_flag" in table:
        quality_labels = {"ok": "Без замечаний", "valid": "Валидно", "verified": "Подтверждено", "unverified": "Не подтверждено", "synthetic": "Синтетика", "demo": "Синтетика", "missing": "Пропуск", "warning": "Ограничения", "error": "Ошибка"}
        table["quality_flag"] = table["quality_flag"].map(lambda value: quality_labels.get(str(value), str(value)))
    if not result.exportable:
        table["Статус выгрузки"] = "СИНТЕТИКА · НЕ ДЛЯ СДАЧИ" if demo else "НЕПОДТВЕРЖДЁННЫЕ ДАННЫЕ · НЕ ДЛЯ СДАЧИ"
    st.dataframe(table.rename(columns=column_names), hide_index=True, width="stretch", height=340,
                 column_config={column_names[c]: st.column_config.NumberColumn(format="%.4f") for c in ("prediction", "p10", "p50", "p90") if c in rows})
    with st.expander("Технические поля выпуска"):
        technical = rows.copy()
        technical["export_status"] = "validated" if result.exportable else "diagnostic_invalid"
        technical["source_mode"] = bundle.source_mode
        st.dataframe(technical, hide_index=True, width="stretch")
    export_columns = st.columns(3)
    with export_columns[0]:
        final_data = selected_csv(bundle, final=True) if result.exportable else b""
        st.download_button("CSV выбранного выпуска", final_data, "forecast.csv", "text/csv", disabled=not result.exportable, on_click="ignore", width="stretch")
    with export_columns[1]:
        st.download_button("JSON-паспорт", passport_json(bundle), "forecast_passport.json", "application/json", on_click="ignore", width="stretch")
    with export_columns[2]:
        st.download_button("Диагностический CSV", selected_csv(bundle, final=False), "diagnostic_invalid_forecast.csv", "text/csv", on_click="ignore", width="stretch")
    if not result.exportable:
        st.caption("Итоговый CSV заблокирован до подтверждения происхождения и всех проверок. Диагностический файл не предназначен для сдачи.")
    st.caption("Формат CSV — формат интеграции команды; официальный формат итоговой таблицы ещё не предоставлен.")


def render_agent(bundle, backend, configured):
    st.markdown("### Паспорт прогноза")
    if bundle is None:
        st.info("Нет выбранного выпуска. После расчёта или загрузки здесь появятся происхождение данных и события выполнения.")
        st.caption(backend.status)
        return
    result = validate_bundle(bundle)
    provenance = sanitize_metadata(bundle.provenance)
    st.caption("Просмотр сохранённого результата. Чтение журнала не запускает агента.")
    st.text("Исполнитель исходного расчёта: " + safe_text(bundle.executor))
    st.text("Источник результата: " + SOURCE_LABELS.get(bundle.source_mode, "Не подтверждён"))
    st.text("Идентификатор прогноза: " + safe_text(bundle.forecast_id))
    if getattr(bundle, "members", []):
        st.caption("Совместное представление турбин. Исходные идентификаторы и паспорта сохранены отдельно; это не новый расчёт backend.")
    st.text("Часовой пояс: " + (bundle.timezone or "Не указан"))
    for member in getattr(bundle, "members", []) or [bundle]:
        if member is not bundle:
            st.text("Выпуск турбины: " + safe_text(member.forecast_id))
        member_provenance = sanitize_metadata(member.provenance)
        for area, name in (("weather", "Погода"), ("model", "Модель")):
            with st.container(border=True):
                st.markdown(f"**{name}**")
                data = member_provenance.get(area, {})
                data = data if isinstance(data, dict) else {}
                labels = {"source": "Источник", "version": "Версия", "run_initialized_at": "Инициализация погодной модели", "available_at": "Исторически доступно с", "availability_upper_bound": "Верхняя оценка доступности", "availability_basis": "Основание оценки доступности", "retrieved_at": "Время скачивания", "training_cutoff": "Последнее обучающее наблюдение", "data_kind": "Тип источника", "normalization": "Нормировка"}
                for key, title in labels.items():
                    if key in data:
                        st.text(f"{title}: {data[key]}")
                if not data:
                    st.warning("Паспорт не предоставлен.")
    check_labels = {"weather_available": "Погода доступна на момент выпуска", "training_cutoff": "Обучающие данные не выходят за момент выпуска", "hourly_grid": "Полная сетка целевых часов", "prediction_finite": "Все прогнозы — конечные числа", "prediction_range": "Все прогнозы в диапазоне 0–1", "real_source": "Источник не синтетический", "verified_origin": "Происхождение подтверждено backend"}
    check_rows = []
    for member in getattr(bundle, "members", []) or [bundle]:
        member_checks = validate_bundle(member).checks
        for key, label in check_labels.items():
            check_rows.append({"Турбина": ", ".join(labels_for(bundle, configured).get(s, s) for s in member.site_ids),
                               "Проверка": label, "Результат": "Пройдена" if member_checks.get(key) is True else "Не пройдена" if key in member_checks else "Не проверено"})
    st.dataframe(pd.DataFrame(check_rows), hide_index=True, width="stretch")
    st.caption("Историческую допустимость определяет доступность погоды на момент выпуска, а не время её сегодняшнего скачивания.")
    issue_details(result)
    with st.expander("Полный JSON-паспорт"):
        st.json(json.loads(passport_json(bundle)))
    st.markdown("#### Журнал выполнения")
    st.caption("Этапы: погода → подготовка → модель → проверка результата → сохранение → анализ")
    if bundle.events:
        st.dataframe(pd.DataFrame(sanitize_metadata(bundle.events)), hide_index=True, width="stretch")
        st.download_button("Очищенный журнал JSON", json.dumps(sanitize_metadata(bundle.events), ensure_ascii=False, default=str), "events.json", "application/json", on_click="ignore")
    else:
        st.info("Backend не предоставил журнал. Этапы и длительности не симулируются.")
    if st.button("Проверить обновления", disabled=not backend.capabilities.get("check_updates") or bundle.source_mode == "demo", key="check_updates"):
        previous = bundle
        try:
            members = getattr(bundle, "members", [])
            if members:
                candidate_members = []
                changed = False
                for member in members:
                    replacements = backend.check_updates(member.forecast_id)
                    changed = changed or bool(replacements)
                    candidate_members.append(replacements[-1] if replacements else member)
                updated = [combine_bundles(candidate_members)] if changed else []
            else:
                updated = backend.check_updates(bundle.forecast_id)
            if updated:
                finish_run(st.session_state, updated)
                new = updated[-1]
                left, right = previous.rows.copy(), new.rows.copy()
                left["target_time"] = pd.to_datetime(left["target_time"], utc=True)
                right["target_time"] = pd.to_datetime(right["target_time"], utc=True)
                common = left.merge(right, on=["site_id", "target_time"], suffixes=("_previous", "_new"))
                if common.empty:
                    st.warning("Новый выпуск получен, общего диапазона для сравнения нет.")
                else:
                    common["Изменение"] = common["prediction_new"] - common["prediction_previous"]
                    st.dataframe(common[["site_id", "target_time", "prediction_previous", "prediction_new", "Изменение"]], hide_index=True)
                st.success("Новый выпуск добавлен в историю. Выберите его в боковой панели.")
            else:
                st.info("Новых входных данных нет.")
        except AdapterError as exc:
            st.error(safe_text(exc))
        except Exception:
            st.error("Проверить обновления не удалось. Предыдущий выпуск доступен; проверьте подключение backend и повторите действие.")
    if not backend.capabilities.get("check_updates"):
        st.caption("Проверка обновлений не подключена: ожидается функция модуля команды.")


def render_replay(backend, config):
    with st.expander("Исторический replay февраля", expanded=False):
        st.caption("Даты относятся к выпускам в 23:00 Asia/Almaty. Для полного февраля начните с 31 января: этот выпуск покрывает первые часы 1 февраля. Открытие страницы не запускает replay.")
        with st.form("replay_form"):
            dates = st.columns(2)
            start = dates[0].date_input("Дата первого выпуска", date(2026, 1, 31), min_value=date(2026, 1, 31), max_value=date(2026, 2, 28))
            end = dates[1].date_input("Дата последнего выпуска", date(2026, 2, 28), min_value=date(2026, 1, 31), max_value=date(2026, 2, 28))
            submit = st.form_submit_button("Запустить replay", disabled=not backend.capabilities.get("run_replay") or bool(config.issues))
        if not backend.capabilities.get("run_replay"):
            st.info("Replay не подключён: модуль участника 2 пока отсутствует.")
        if submit:
            if end < start:
                st.error("Последний день должен быть не раньше первого.")
            else:
                try:
                    with st.spinner("Backend выполняет исторические выпуски…"):
                        replay = backend.run_replay(start, end)
                    st.session_state["replay_result"] = replay
                    if replay.bundles:
                        finish_run(st.session_state, replay.bundles)
                except AdapterError as exc:
                    st.session_state.pop("replay_result", None)
                    st.error(safe_text(exc))
                except Exception:
                    st.session_state.pop("replay_result", None)
                    st.error("Replay не выполнен. Проверьте backend и доступность архивов; сохранённые выпуски остаются в истории.")
        replay = st.session_state.get("replay_result")
        if replay:
            st.write(f"Завершено выпусков: {len(replay.bundles)}. Ошибок: {len(replay.errors)}.")
            for error in replay.errors:
                st.text(safe_text(error))
            if replay.bundles:
                st.dataframe(pd.DataFrame([{"Прогноз": b.forecast_id, "Турбины": ", ".join(b.site_ids), "Горизонт": b.horizon_hours, "Экспорт": "допустим" if validate_bundle(b).exportable else "заблокирован"} for b in replay.bundles]), hide_index=True)
                st.download_button("Все версии replay · диагностический CSV", replay_csv(replay.bundles), "diagnostic_invalid_replay.csv", "text/csv", on_click="ignore")
                final_ok = all(validate_bundle(b).exportable for b in replay.bundles) and not replay.errors
                st.download_button("Все версии replay · итоговый CSV", replay_csv(replay.bundles, final=True) if final_ok else b"", "replay_all_versions.csv", "text/csv", disabled=not final_ok, on_click="ignore")
                agreed = st.checkbox("Для таблицы февраля выбираю последний доступный выпуск до целевого часа", key="february_rule")
                st.caption("Это выбранное правило команды, не утверждённый формат организаторов. Полные выпуски сохраняют часы марта.")
                if agreed:
                    try:
                        data = february_csv(replay.bundles, list(config.sites), config.timezone, final=True)
                        st.download_button("Февраль · один прогноз на час", data, "february_selected.csv", "text/csv", on_click="ignore")
                    except ValueError:
                        st.warning("Таблица февраля неполна или не прошла проверку происхождения. Нужна полная сетка 28 × 24 часа для каждой настроенной турбины.")


def main():
    st.set_page_config(page_title="WindOps AI · Прогноз ВЭС", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    apply_theme()
    initialize_state(st.session_state)
    config = load_configuration()
    backend = BackendAdapter(os.environ.get("WINDOPS_BACKEND_MODULE") or "windops.backend")
    with st.sidebar:
        st.markdown("### Параметры выпуска")
        st.caption("Исторический прогноз · февраль 2026")
        demo_mode = st.toggle("Тест интерфейса · синтетика", value=False, key="demo_mode")
        sites = {"demo-1": "Турбина 1", "demo-2": "Турбина 2"} if demo_mode else config.sites
        timezone = "UTC" if demo_mode else config.timezone
        st.caption(f"Часовой пояс: {timezone or 'НЕ НАСТРОЕН'}" + (" · только для тестовых данных" if demo_mode else ""))
        options = list(sites)
        if len(options) > 1:
            options.append("__both__")
        if not options:
            options = ["__missing__"]
        default = config.default_origin or datetime(2026, 1, 31, 23, 0)
        if not date(2026, 1, 31) <= default.date() <= date(2026, 2, 28):
            default = datetime(2026, 1, 31, 23, 0)
        with st.form("forecast_form"):
            chosen = st.selectbox("Турбина", options, index=len(options) - 1, format_func=lambda s: "Обе турбины" if s == "__both__" else sites.get(s, "Турбины не настроены"), key="site_choice")
            issued_date = st.date_input("Дата выпуска", default.date(), min_value=date(2026, 1, 31), max_value=date(2026, 2, 28), key="issued_date")
            issued_time = st.time_input("Время выпуска", default.time().replace(tzinfo=None), step=3600, key="issued_time")
            horizon = st.radio("Горизонт", [24, 48], index=1, format_func=lambda h: f"{h} часов", horizontal=True, key="horizon")
            submitted = st.form_submit_button("Загрузить тестовый выпуск" if demo_mode else "Сформировать прогноз", type="primary", width="stretch", disabled=not demo_mode and (bool(config.issues) or not backend.capabilities.get("run_forecast")), key="load_demo" if demo_mode else "run_forecast")
        st.caption("Первый целевой час следует за моментом выпуска. Январская проверка — на вкладке «Проверка качества».")
        if not demo_mode:
            for issue in config.issues:
                st.caption(issue)
            if not backend.capabilities.get("run_forecast"):
                st.caption(backend.status)
        if submitted:
            selected_sites = list(sites) if chosen == "__both__" else [chosen]
            origin = datetime.combine(issued_date, issued_time, tzinfo=ZoneInfo(timezone))
            params = {"site_ids": selected_sites, "forecast_origin": origin.isoformat(), "horizon_hours": horizon, "demo": demo_mode}
            if begin_run(st.session_state, params):
                try:
                    with st.spinner("Подготовка тестового выпуска…" if demo_mode else "Backend формирует прогноз…"):
                        bundles = [synthetic_bundle(selected_sites, horizon, origin.isoformat())] if demo_mode else backend.run_forecast(selected_sites, origin.isoformat(), horizon)
                    if len(bundles) > 1:
                        bundles = [*bundles, combine_bundles(bundles)]
                    finish_run(st.session_state, bundles)
                    if bundles:
                        st.session_state["history_choice"] = bundles[-1].forecast_id
                except AdapterError as exc:
                    fail_run(st.session_state, "Новый выпуск не сформирован. " + safe_text(exc))
                except Exception:
                    fail_run(st.session_state, "Новый выпуск не сформирован. Предыдущий успешный выпуск сохранён. Проверьте backend, конфигурацию и доступность архивных данных.")
        st.divider()
        st.markdown("#### Сохранённые выпуски")
        upload = st.file_uploader("Загрузить JSON-выпуск", type=["json"], key="forecast_upload", help="Локальный файл не отправляется LLM. Самостоятельно загруженный паспорт не считается подтверждением для итогового экспорта.")
        if upload and st.button("Открыть выпуск", key="load_artifact", width="stretch"):
            try:
                bundle = load_bundle_json(upload.getvalue())
                finish_run(st.session_state, [bundle])
                st.session_state["history_choice"] = bundle.forecast_id
            except Exception:
                fail_run(st.session_state, "Выпуск не загружен. Проверьте JSON и обязательные поля по docs/INTEGRATION.md. История сохранена.")
        history = st.session_state["history"]
        if history:
            ids = list(history)
            current = selected_bundle(st.session_state)
            active = current.forecast_id if current else ids[-1]
            selection = st.selectbox("История выпусков", ids, index=ids.index(active) if active in ids else len(ids) - 1,
                                     format_func=lambda fid: history_label(history[fid]), key="history_choice")
            if current is None or selection != current.forecast_id:
                select_bundle(st.session_state, selection)
        else:
            st.caption("Сохранённых выпусков пока нет.")

    bundle = selected_bundle(st.session_state)
    if bundle is not None and bundle.source_mode == "demo" and not demo_mode:
        bundle = None
    render_header(status_label="Тест интерфейса" if bundle is not None and bundle.source_mode == "demo" else "Выпуск выбран" if bundle is not None else "Ожидаются данные", demo=bundle is not None and bundle.source_mode == "demo")
    if st.session_state["status"] == "error":
        st.error(st.session_state.get("error") or "Последняя операция завершилась ошибкой.")
        if bundle is not None:
            st.warning("Предыдущий успешный выпуск — результат до последней неудачной операции.")
    forecast_tab, quality_tab, agent_tab = st.tabs(["Прогноз", "Проверка качества", "Агент и данные"])
    with forecast_tab:
        render_forecast(bundle, config.sites)
        render_replay(backend, config)
    with quality_tab:
        render_quality(config.timezone, config.sites)
    with agent_tab:
        render_agent(bundle, backend, config.sites)


def history_label(bundle):
    rows = bundle.rows
    issued = display_time(rows["issued_at"].iloc[0], bundle.timezone) if not rows.empty and "issued_at" in rows else "Нет даты"
    version = str(rows["model_version"].iloc[0]) if not rows.empty and "model_version" in rows else "Нет версии"
    status = "проверен" if validate_bundle(bundle).exportable else "ограничения"
    return f"{issued} · {', '.join(bundle.site_ids)} · {bundle.horizon_hours} ч · {version} · {status} · {bundle.forecast_id}"


if __name__ == "__main__":
    main()
