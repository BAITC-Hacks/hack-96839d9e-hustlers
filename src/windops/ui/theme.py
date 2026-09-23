"""Shared native theme tokens and CSS limited to our own HTML components."""

BACKGROUND = "#0B1220"
SURFACE = "#121C2D"
BORDER = "#29364A"
TEXT = "#F1F5F9"
MUTED = "#B3C0D1"
BLUE = "#60A5FA"
TEAL = "#2DD4BF"
FONT = "Segoe UI, Arial, sans-serif"

# Inline locale avoids fetching a JavaScript locale or a font from a CDN.
PLOT_CONFIG = {
    "displaylogo": False, "scrollZoom": False, "locale": "ru",
    "locales": {"ru": {"dictionary": {
        "Download plot as a PNG": "Скачать график PNG",
        "Zoom": "Выделить область", "Pan": "Переместить",
        "Zoom in": "Приблизить", "Zoom out": "Отдалить",
        "Autoscale": "Масштаб по данным", "Reset axes": "Сбросить масштаб",
        "Double-click to zoom back out": "Двойной щелчок — сброс масштаба",
        "Double-click on legend to isolate one trace": "Двойной щелчок — показать только этот ряд",
        "Preparing image - this may take a few seconds": "Подготовка изображения…",
        "Image download succeeded": "Изображение сохранено",
    }, "format": {"date": "%d.%m.%Y", "decimal": ".", "thousands": " "}}},
}


def apply_theme() -> None:
    import streamlit as st

    st.html("""<style>
    .windops-header { display:flex; justify-content:space-between; align-items:center;
        gap:12px; flex-wrap:wrap; padding:0 0 4px; font-family:Segoe UI,Arial,sans-serif; }
    .windops-brand { margin:0; color:#F1F5F9; font-size:30px; font-weight:700; letter-spacing:-.7px; }
    .windops-header > div:first-child { display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }
    .windops-brand span { color:#60A5FA; }
    .windops-subtitle { margin:3px 0 0; color:#B3C0D1; font-size:15px; }
    .windops-badges { display:flex; gap:8px; flex-wrap:wrap; }
    .windops-badge { border:1px solid #29364A; background:#121C2D; color:#B3C0D1;
        padding:6px 10px; border-radius:7px; font-size:13px; }
    .windops-cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
        gap:10px; margin:4px 0 8px; font-family:Segoe UI,Arial,sans-serif; }
    .windops-card { background:#121C2D; border:1px solid #29364A; border-radius:10px;
        padding:7px 12px; min-width:0; }
    .windops-card-label { color:#B3C0D1; font-size:12px; line-height:1.4; }
    .windops-card-value { color:#F1F5F9; font-size:26px; font-weight:650;
        line-height:1.25; margin:6px 0; overflow-wrap:anywhere; }
    .windops-card-detail { color:#B3C0D1; font-size:11px; line-height:1.5; }
    .windops-card-warning .windops-card-value { color:#FBBF24; }
    .windops-card-error .windops-card-value { color:#F87171; }
    .windops-card-success .windops-card-value { color:#4ADE80; }
    .windops-overview { font-family:Segoe UI,Arial,sans-serif; }
    .windops-demo-notice { color:#FBBF24; background:#332913; border:1px solid #655124;
        border-radius:7px; padding:5px 10px; font-size:12px; line-height:18px; margin:0 0 8px; }
    .windops-overview-title { color:#F1F5F9; font-size:20px; font-weight:650;
        line-height:26px; margin:0 0 5px; padding:0; }
    .windops-overview-context { color:#B3C0D1; font-size:13px; line-height:18px;
        overflow-wrap:anywhere; }
    .windops-overview .windops-cards { margin:9px 0 6px; }
    .windops-overview-footer { color:#B3C0D1; font-size:13px; line-height:18px; }
    @media(max-width:600px) { .windops-cards { grid-template-columns:repeat(2,minmax(0,1fr)); }
        .windops-card { padding:12px; } .windops-card-value { font-size:23px; }
        .windops-brand { font-size:28px; } }
    </style>""")
