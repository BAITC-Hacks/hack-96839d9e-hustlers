"""Validate a saved January backtest and evaluate matched observations locally.

This is a UI artifact format, not a training implementation or an assertion
that the declared provenance is authentic. No file is generated here.
"""

from dataclasses import dataclass
import json
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from .charts import forecast_chart
from .components import POWER_UNIT_HELP, render_metric_cards
from .adapter import sanitize_metadata
from .theme import PLOT_CONFIG


@dataclass(frozen=True)
class BacktestArtifact:
    rows: pd.DataFrame
    timezone: str
    methodology: str
    provenance: dict
    baseline_names: tuple[str, ...]


def _timestamps(values: pd.Series, field: str) -> pd.Series:
    try:
        if any(pd.isna(value) or pd.Timestamp(value).tzinfo is None for value in values):
            raise ValueError
        return pd.to_datetime(values, utc=True, errors="raise")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Поле {field}: нужны ISO 8601 даты с явным часовым поясом.") from exc


def load_backtest(payload: bytes | str | dict) -> BacktestArtifact:
    """Check shape, chronology and leakage; numeric missing values stay missing."""
    try:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8-sig")
        data = json.loads(payload) if isinstance(payload, str) else payload
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Артефакт должен быть JSON в UTF-8.") from exc
    if not isinstance(data, dict):
        raise ValueError("Ожидается JSON-объект проверочного артефакта.")
    if type(data.get("schema_version")) is not int or data.get("schema_version") != 1 or data.get("kind") != "january_backtest":
        raise ValueError("Нужен january_backtest с schema_version=1.")
    if data.get("data_mode") != "real_saved":
        raise ValueError("Синтетические данные не являются независимой проверкой качества.")
    timezone = data.get("timezone")
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError, TypeError) as exc:
        raise ValueError("Укажите известный часовой пояс IANA в артефакте.") from exc
    methodology = data.get("methodology")
    provenance = data.get("provenance")
    if not isinstance(methodology, str) or not methodology.strip():
        raise ValueError("Не описана методика независимой январской проверки.")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("data_source"), str) or not provenance["data_source"].strip():
        raise ValueError("Не указан provenance.data_source для фактических наблюдений.")
    raw_rows = data.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows or not all(isinstance(row, dict) for row in raw_rows):
        raise ValueError("Нужен непустой список rows с проверочными наблюдениями.")
    if len(raw_rows) > 100_000:
        raise ValueError("Артефакт слишком большой: не более 100 000 строк.")
    rows = pd.DataFrame(raw_rows).copy()
    required = {"run_id", "site_id", "issued_at", "target_time", "horizon_step", "prediction", "actual", "model_version", "training_cutoff"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError("Не предоставлены обязательные поля: " + ", ".join(missing))
    for name in ("run_id", "site_id", "model_version"):
        if not rows[name].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Поле {name} должно содержать непустые идентификаторы.")
    for name in ("issued_at", "target_time", "training_cutoff"):
        rows[name] = _timestamps(rows[name], name)
    local = rows["target_time"].dt.tz_convert(timezone)
    if not ((local.dt.year == 2026) & (local.dt.month == 1)).all():
        raise ValueError("Независимая проверка принимает только целевые часы января 2026.")
    if not (rows["training_cutoff"] < rows["issued_at"]).all():
        raise ValueError("Утечка данных: training_cutoff должен быть раньше момента проверочного выпуска.")
    step = pd.to_numeric(rows["horizon_step"], errors="coerce")
    if not (step.between(1, 48) & step.mod(1).eq(0)).all():
        raise ValueError("Шаг проверочного горизонта должен быть целым числом от 1 до 48.")
    rows["horizon_step"] = step.astype(int)
    expected = rows["issued_at"] + pd.to_timedelta(rows["horizon_step"], unit="h")
    if not rows["target_time"].eq(expected).all():
        raise ValueError("Целевой час должен соответствовать моменту выпуска и шагу горизонта.")
    if rows.duplicated(["run_id", "site_id", "target_time"]).any():
        raise ValueError("Дубли в проверке: run_id + site_id + target_time должны быть уникальны.")
    consistency = rows.groupby(["run_id", "site_id"])[["issued_at", "model_version", "training_cutoff"]].nunique()
    if consistency.gt(1).any().any():
        raise ValueError("Параметры одного проверочного выпуска противоречат друг другу.")
    names: set[str] = set()
    baselines = []
    for raw in raw_rows:
        values = raw.get("baselines", {})
        if not isinstance(values, dict) or not all(isinstance(key, str) and key.strip() for key in values):
            raise ValueError("baselines должен быть словарём названий и прогнозных значений.")
        names.update(values)
        baselines.append(values)
    for name in sorted(names):
        rows[f"baseline__{name}"] = [values.get(name) for values in baselines]
    for name in ["prediction", "actual", *[f"baseline__{name}" for name in sorted(names)]]:
        # Booleans are flags, not normalized power values.
        values = rows[name].map(lambda value: None if isinstance(value, bool) else value)
        rows[name] = pd.to_numeric(values, errors="coerce")
    for name in ("actual_valid", "prediction_valid"):
        if name in rows and not rows[name].map(lambda value: isinstance(value, bool)).all():
            raise ValueError(f"Необязательный флаг {name} должен быть true/false в каждой строке.")
    return BacktestArtifact(rows, timezone, sanitize_metadata(methodology.strip()), sanitize_metadata(provenance), tuple(sorted(names)))


def _valid_pairs(rows: pd.DataFrame) -> pd.Series:
    valid = rows["prediction"].between(0, 1) & rows["actual"].between(0, 1)
    for name in ("actual_valid", "prediction_valid"):
        if name in rows:
            valid &= rows[name]
    return valid


def evaluate_backtest(artifact: BacktestArtifact) -> pd.DataFrame:
    """Micro-average release/target pairs; each baseline uses its own matched cohort."""
    rows = artifact.rows.copy()
    rows["horizon_band"] = rows["horizon_step"].map(lambda step: "1–24" if step <= 24 else "25–48")
    results = []
    for (site, model, band), part in rows.groupby(["site_id", "model_version", "horizon_band"], sort=True):
        for baseline in (None, *artifact.baseline_names):
            mask = _valid_pairs(part)
            if baseline is not None:
                mask &= part[f"baseline__{baseline}"].between(0, 1)
            cohort = part.loc[mask]
            error = cohort["prediction"] - cohort["actual"]
            model_mae = float(error.abs().mean()) if len(cohort) else None
            model_rmse = float((error.pow(2).mean()) ** .5) if len(cohort) else None
            baseline_mae = baseline_rmse = improvement = None
            if baseline is not None and len(cohort):
                baseline_error = cohort[f"baseline__{baseline}"] - cohort["actual"]
                baseline_mae = float(baseline_error.abs().mean())
                baseline_rmse = float(baseline_error.pow(2).mean() ** .5)
                if baseline_mae > 0:
                    improvement = (baseline_mae - model_mae) / baseline_mae * 100
            results.append({"site_id": site, "model_version": model, "horizon_band": band,
                "comparison": baseline or "model", "n_used": len(cohort), "n_excluded": len(part) - len(cohort),
                "model_mae": model_mae, "model_rmse": model_rmse, "baseline_mae": baseline_mae,
                "baseline_rmse": baseline_rmse, "mae_improvement_pct": improvement})
    return pd.DataFrame(results)


def render_quality(timezone: str | None = None, site_labels: dict[str, str] | None = None) -> None:
    import streamlit as st

    st.markdown("### Январская проверка качества")
    st.caption("MAE и RMSE в шкале нормализованной мощности. Фактических значений февраля нет.")
    st.caption(POWER_UNIT_HELP)
    uploaded = st.file_uploader("Загрузить январский backtest JSON", type=["json"], key="quality_upload",
                                help="Формат локального артефакта описан в docs/INTEGRATION.md. Вкладка не обучает модель.")
    if uploaded is not None and st.button("Проверить артефакт", key="check_quality"):
        try:
            st.session_state["quality_artifact"] = load_backtest(uploaded.getvalue())
            st.session_state.pop("quality_error", None)
        except (ValueError, TypeError, KeyError) as exc:
            st.session_state.pop("quality_artifact", None)
            st.session_state["quality_error"] = sanitize_metadata(str(exc))
    if st.session_state.get("quality_error"):
        st.error("Артефакт не принят. Исправьте JSON по контракту интеграции и повторите загрузку.")
        st.text(st.session_state["quality_error"])
    artifact = st.session_state.get("quality_artifact")
    if artifact is None:
        st.info("Январский backtest пока не загружен в эту сессию. Метрики и график факта появятся после загрузки проверочного артефакта.")
        st.caption("Загрузите сохранённый январский JSON и нажмите «Проверить артефакт».")
        return
    rows = artifact.rows
    if artifact.provenance.get("evaluation_independent") is False:
        st.warning("Разработочное сравнение: январь уже просматривался при разработке этой версии. Это не новая независимая проверка.")
    labels = site_labels or {}
    if timezone and artifact.timezone != timezone:
        st.warning(f"Часовой пояс артефакта: {artifact.timezone}; конфигурация UI: {timezone}. Для проверки используется зона артефакта.")
    st.warning("Проверены структура, хронология и совпадение пар. Достоверность заявленного источника и независимость подготовки признаков должен подтвердить владелец backtest.")
    local_targets = rows["target_time"].dt.tz_convert(artifact.timezone)
    st.caption(f"Период оценки: {local_targets.min():%d.%m.%Y %H:%M} — {local_targets.max():%d.%m.%Y %H:%M} · {artifact.timezone}")
    st.text("Источник факта: " + artifact.provenance["data_source"])
    st.text("Методика артефакта: " + artifact.methodology)
    st.caption("Агрегация UI: каждое совпадение выпуска, турбины и целевого часа — отдельная пара с одинаковым весом. Перекрывающиеся выпуски учитываются отдельно; пропуски, некорректные значения и флаги невалидности исключаются.")
    site = st.selectbox("Турбина для проверки", list(rows["site_id"].unique()), format_func=lambda value: labels.get(value, value), key="quality_site")
    model_options = list(rows.loc[rows["site_id"] == site, "model_version"].unique())
    model = st.selectbox("Проверочная модель", model_options, key="quality_model")
    part = rows[(rows["site_id"] == site) & (rows["model_version"] == model)]
    valid = _valid_pairs(part)
    differences = part.loc[valid, "prediction"] - part.loc[valid, "actual"]
    render_metric_cards([
        {"label": "MAE", "value": f"{differences.abs().mean():.4f}" if valid.any() else "Нет данных", "detail": "Средняя абсолютная ошибка"},
        {"label": "RMSE", "value": f"{differences.pow(2).mean() ** .5:.4f}" if valid.any() else "Нет данных", "detail": "Корень средней квадратичной ошибки"},
        {"label": "Использовано пар", "value": str(int(valid.sum())), "detail": f"Исключено: {int((~valid).sum())}"},
    ])
    metrics = evaluate_backtest(artifact)
    metrics = metrics[(metrics["site_id"] == site) & (metrics["model_version"] == model)].drop(columns=["site_id", "model_version"])
    metrics["comparison"] = metrics["comparison"].replace({"model": "Все валидные пары модели"})
    st.dataframe(metrics.rename(columns={"horizon_band": "Горизонт, ч", "comparison": "Сравнение",
        "n_used": "Использовано пар", "n_excluded": "Исключено", "model_mae": "MAE модели",
        "model_rmse": "RMSE модели", "baseline_mae": "MAE baseline", "baseline_rmse": "RMSE baseline",
        "mae_improvement_pct": "Изменение MAE, %"}), hide_index=True, width="stretch")
    st.caption("Каждая строка сравнения использует одинаковые целевые пары модели и baseline. Положительное изменение MAE означает улучшение, отрицательное — ухудшение; при нулевой ошибке baseline процент не рассчитывается.")
    if not artifact.baseline_names:
        st.info("Baseline-прогнозы не предоставлены. Сравнение не рассчитано.")
    run_options = list(part["run_id"].unique())
    def run_label(run_id):
        origin = part.loc[part["run_id"] == run_id, "issued_at"].iloc[0].tz_convert(artifact.timezone)
        return f"{origin:%d.%m.%Y %H:%M} · {run_id}"
    run_id = st.selectbox("Проверочный выпуск на графике", run_options, format_func=run_label, key="quality_run")
    selected = part[part["run_id"] == run_id].copy()
    cutoff = selected["training_cutoff"].iloc[0].tz_convert(artifact.timezone)
    st.caption(f"Граница доступности обучающих ответов: {cutoff:%d.%m.%Y %H:%M %z} · модель {model}")
    plotted = selected.copy()
    plotted.loc[~_valid_pairs(plotted), ["prediction", "actual"]] = float("nan")
    st.plotly_chart(forecast_chart(plotted, artifact.timezone, labels, actuals=True), width="stretch",
                    config=PLOT_CONFIG, key="quality_chart")
    with st.expander("Проверочные наблюдения и паспорт"):
        st.dataframe(selected, hide_index=True, width="stretch")
        st.text("Часовой пояс: " + artifact.timezone)
        st.text("Сведения в JSON заявлены поставщиком; интерфейс не подтверждает подлинность источника.")
