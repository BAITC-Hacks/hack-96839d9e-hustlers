"""Small responsive UI components; dynamic HTML is always escaped."""

from html import escape
from collections.abc import Mapping, Sequence


def render_header(mode_label: str = "Исторический запуск", status_label: str = "Нет данных", demo: bool = False) -> None:
    import streamlit as st

    st.html(
        '<header class="windops-header"><div><h1 class="windops-brand">WindOps <span>AI</span></h1>'
        '<p class="windops-subtitle">Почасовой прогноз выработки ВЭС</p></div>'
        '<div class="windops-badges"><span class="windops-badge">'
        + escape(str(mode_label)) + '</span><span class="windops-badge">'
        + escape(str(status_label)) + '</span></div></header>'
        + ('<div class="windops-demo-notice" role="note">СИНТЕТИЧЕСКИЕ ДАННЫЕ — ТЕСТ ИНТЕРФЕЙСА. Итоговый экспорт запрещён.</div>' if demo else '')
    )


def _metric_cards_html(cards: Sequence[Mapping]) -> str:
    parts = ['<div class="windops-cards">']
    for card in cards:
        tone = card.get("tone", "neutral")
        tone = tone if tone in {"neutral", "warning", "error", "success"} else "neutral"
        parts.append(f'<section class="windops-card windops-card-{tone}">')
        for field, css in (("label", "label"), ("value", "value"), ("detail", "detail")):
            value = card.get(field, "")
            parts.append(f'<div class="windops-card-{css}">{escape(str(value))}</div>')
        parts.append("</section>")
    parts.append("</div>")
    return "".join(parts)


def render_metric_cards(cards: Sequence[Mapping]) -> None:
    import streamlit as st

    st.html(_metric_cards_html(cards))


def render_forecast_overview(lines: list[str], cards: list[dict], demo: bool, admissibility: str) -> None:
    """Keep release context and summary compact without styling native UI internals."""
    import streamlit as st

    parts = ['<section class="windops-overview">']
    if demo:
        parts.append('<div class="windops-demo-notice" role="note">'
                     'СИНТЕТИЧЕСКИЕ ДАННЫЕ — ТЕСТ ИНТЕРФЕЙСА. Итоговый экспорт запрещён.</div>')
    parts.append('<h2 class="windops-overview-title">Почасовой прогноз</h2>')
    for line in lines:
        parts.append('<div class="windops-overview-context">' + escape(str(line)) + '</div>')
    parts.append(_metric_cards_html(cards))
    if admissibility:
        parts.append('<div class="windops-overview-footer">' + escape(str(admissibility)) + '</div>')
    parts.append('</section>')
    st.html(''.join(parts))
