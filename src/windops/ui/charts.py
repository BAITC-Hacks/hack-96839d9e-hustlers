"""Plot only supplied values, with an explicit timezone and honest missing hours."""

from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import BACKGROUND, SURFACE, BORDER, TEXT, MUTED, BLUE, TEAL, FONT

COLORS = (BLUE, TEAL, "#C4B5FD", "#FBBF24")


def _prepare(rows: pd.DataFrame, timezone: str) -> pd.DataFrame:
    ZoneInfo(timezone)
    if not {"site_id", "target_time"}.issubset(rows.columns):
        raise ValueError("Для графика нужны site_id и target_time.")
    data = rows.copy()
    for value in data["target_time"]:
        if pd.isna(value) or pd.Timestamp(value).tzinfo is None:
            raise ValueError("Целевые часы должны содержать часовой пояс.")
    data["target_time"] = pd.to_datetime(data["target_time"], utc=True).dt.tz_convert(timezone)
    if data.duplicated(["site_id", "target_time"]).any():
        raise ValueError("На графике должен быть только один выпуск для каждой турбины.")
    for key in ("run_id", "issued_at"):
        if key in data and data.groupby("site_id")[key].nunique().gt(1).any():
            raise ValueError("Нельзя соединять разные выпуски в один временной ряд.")
    return data


def _hourly(part: pd.DataFrame) -> pd.DataFrame:
    part = part.sort_values("target_time").set_index("target_time")
    if part.empty:
        return part
    grid = pd.date_range(part.index.min(), part.index.max(), freq="h")
    return part.reindex(grid)


def _color(site_id: str, site_labels: dict[str, str], site_ids: list[str]) -> str:
    order = list(dict.fromkeys([*site_labels, *sorted(site_ids)]))
    return COLORS[order.index(site_id) % len(COLORS)]


def _numbers(part: pd.DataFrame, name: str, *, normalized: bool = False) -> pd.Series:
    values = pd.to_numeric(part[name], errors="coerce").replace([float("inf"), -float("inf")], float("nan"))
    if normalized:
        values = values.where(values.between(0, 1))
    return values


def _layout(fig: go.Figure, timezone: str, demo: bool, height: int = 280) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BACKGROUND, plot_bgcolor=SURFACE,
        font={"family": FONT, "color": TEXT, "size": 13},
        margin={"l": 8, "r": 16, "t": 40 if demo else 25, "b": 12}, height=height,
        hovermode="x unified", dragmode="zoom",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
        hoverlabel={"bgcolor": SURFACE, "font_color": TEXT},
    )
    fig.update_xaxes(title_text=f"Целевой час · {timezone}", gridcolor=BORDER, zeroline=False,
                     tickformat="%d.%m\n%H:%M", title_font_color=MUTED)
    fig.update_yaxes(gridcolor=BORDER, zeroline=False, title_font_color=MUTED)
    if demo:
        fig.add_annotation(text="СИНТЕТИЧЕСКИЕ ДАННЫЕ · ТЕСТ ИНТЕРФЕЙСА", x=.99, y=.98,
                           xref="paper", yref="paper", xanchor="right", yanchor="top", showarrow=False,
                           bgcolor=SURFACE, borderpad=3, font={"size": 10, "color": "#FBBF24"})
    return fig


def forecast_chart(rows: pd.DataFrame, timezone: str, site_labels: dict[str, str], demo: bool = False,
                   *, actuals: bool = False) -> go.Figure:
    """Display one release; factual output is permitted only for January 2026."""
    data = _prepare(rows, timezone)
    if "prediction" not in data:
        raise ValueError("Не предоставлен прогноз.")
    fig = go.Figure()
    sites = list(data["site_id"].unique())
    for site_id in sites:
        part = _hourly(data[data["site_id"] == site_id])
        color = _color(site_id, site_labels, sites)
        label = site_labels.get(site_id, site_id)
        hover_times = part.index.strftime("%d.%m.%Y %H:%M %z")
        if {"p10", "p90"}.issubset(part.columns):
            lower = _numbers(part, "p10", normalized=True)
            upper = _numbers(part, "p90", normalized=True)
            valid = lower.notna() & upper.notna() & lower.le(upper)
            if "p50" in part:
                median = _numbers(part, "p50", normalized=True)
                valid &= median.isna() | (median.ge(lower) & median.le(upper))
            if valid.any():
                rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                # Separate polygons prevent a filled band from bridging missing hours.
                segments = valid.ne(valid.shift(fill_value=False)).cumsum()
                first_segment = True
                for _, segment in part.loc[valid].groupby(segments[valid]):
                    fig.add_trace(go.Scatter(x=segment.index, y=lower.loc[segment.index], mode="lines",
                        line={"width": 0}, name=f"{label} · P10", legendgroup=site_id,
                        showlegend=False, hoverinfo="skip", connectgaps=False))
                    fig.add_trace(go.Scatter(x=segment.index, y=upper.loc[segment.index], mode="lines",
                        line={"width": 0}, fill="tonexty", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.12)",
                        name=f"{label} · P10–P90", legendgroup=site_id, hoverinfo="skip",
                        showlegend=first_segment, connectgaps=False))
                    first_segment = False
        fig.add_trace(go.Scatter(x=part.index, y=_numbers(part, "prediction", normalized=True),
            mode="lines", line={"color": color, "width": 2.5, "shape": "linear"},
            name=f"{label} · Прогноз", legendgroup=site_id, connectgaps=False,
            customdata=hover_times, hovertemplate="%{customdata}<br>Прогноз: %{y:.4f}<extra>%{fullData.name}</extra>"))
        if "p50" in part and _numbers(part, "p50", normalized=True).notna().any():
            fig.add_trace(go.Scatter(x=part.index, y=_numbers(part, "p50", normalized=True),
                mode="lines", line={"color": color, "width": 1, "dash": "dot"},
                name=f"{label} · P50", legendgroup=site_id, connectgaps=False,
                customdata=hover_times, hovertemplate="%{customdata}<br>P50: %{y:.4f}<extra>%{fullData.name}</extra>"))
        if actuals and "actual" in part:
            january = (part.index.year == 2026) & (part.index.month == 1)
            values = _numbers(part, "actual", normalized=True).where(january)
            if values.notna().any():
                fig.add_trace(go.Scatter(x=part.index, y=values, mode="lines",
                    line={"color": color, "width": 2, "dash": "dash"}, name=f"{label} · Факт",
                    legendgroup=site_id, connectgaps=False, customdata=hover_times,
                    hovertemplate="%{customdata}<br>Факт: %{y:.4f}<extra>%{fullData.name}</extra>"))
    _layout(fig, timezone, demo)
    fig.update_yaxes(range=[0, 1], title_text="Нормализованная мощность", tickformat=".1f")
    return fig


def weather_chart(rows: pd.DataFrame, timezone: str, site_labels: dict[str, str], demo: bool = False) -> go.Figure:
    """Separate wind and temperature scales; no unit or height is guessed."""
    data = _prepare(rows, timezone)
    if "data_kind" in data:
        kinds = set(data["data_kind"].dropna().astype(str))
        if len(kinds) > 1 or kinds.intersection({"actual", "observed", "reanalysis"}):
            raise ValueError("Погодной график принимает один тип прогнозных входов, без смешивания с фактом.")
        if "synthetic" in kinds and not demo:
            raise ValueError("Синтетическая погода требует явной маркировки тестового режима.")
    wind_columns = [name for name in data if name in {"wind_speed", "wind_speed_mps", "wind_speed_10m", "wind_speed_100m"}]
    temperature = next((name for name in ("temperature_c", "temperature", "temperature_2m") if name in data), None)
    if not wind_columns and not temperature:
        raise ValueError("Нет поддерживаемых почасовых полей погоды.")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.16)
    sites = list(data["site_id"].unique())
    for site_id in sites:
        part = _hourly(data[data["site_id"] == site_id])
        color = _color(site_id, site_labels, sites)
        label = site_labels.get(site_id, site_id)
        for index, column in enumerate(wind_columns):
            height = "10 м" if column.endswith("_10m") else "100 м" if column.endswith("_100m") else "высота в паспорте"
            if column in {"wind_speed", "wind_speed_mps"} and "wind_height_m" in part:
                heights = part["wind_height_m"].dropna().unique()
                if len(heights) > 1:
                    raise ValueError("Нельзя объединять ветер разных высот в одну линию.")
                if len(heights) == 1:
                    height = f"{heights[0]} м"
            fig.add_trace(go.Scatter(x=part.index, y=_numbers(part, column), mode="lines",
                line={"color": color, "dash": "solid" if index == 0 else "dot"},
                name=f"{label} · ветер {height}", connectgaps=False,
                hovertemplate="%{y:.2f} м/с<extra>%{fullData.name}</extra>"), row=1, col=1)
        if temperature:
            fig.add_trace(go.Scatter(x=part.index, y=_numbers(part, temperature), mode="lines",
                line={"color": color}, name=f"{label} · температура", connectgaps=False,
                hovertemplate="%{y:.2f} °C<extra>%{fullData.name}</extra>"), row=2, col=1)
    _layout(fig, timezone, demo, height=350)
    fig.update_xaxes(title_text=None, row=1, col=1)
    fig.update_yaxes(title_text="Ветер, м/с", row=1, col=1)
    fig.update_yaxes(title_text="Температура, °C", row=2, col=1)
    return fig
