# V1619O_TIMEFRAME_SELECTOR
# V1619L_UI_MARKER
from __future__ import annotations

import html
import json
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# GIAO DIỆN STOCK RADAR — trắng nổi khối mềm (Bento)
# Mọi màu/bóng lấy từ bảng THEME để biểu đồ, AgGrid và HTML tự dựng dùng chung.
# ---------------------------------------------------------------------------
THEME = {
    "bg": "#eceff6",          # nền trang
    "bg_hi": "#ffffff",       # mặt trên của khối
    "bg_lo": "#f8f9fd",       # mặt dưới của khối
    "ink": "#0f172a",
    "ink2": "#475569",
    "ink3": "#94a3b8",
    "grid": "#eef1f6",
    "line": "#e3e8f0",
    "up": "#059669",
    "down": "#e11d48",
    "wait": "#d97706",
    "accent": "#4f46e5",
    "accent2": "#7c3aed",
}

# Bộ icon vector (nét 2px, bo tròn) dùng chung cho HTML tự dựng. Màu theo currentColor.
_ICON_PATHS = {
    "up": '<path d="M6 15l6-6 6 6"/>',
    "down": '<path d="M6 9l6 6 6-6"/>',
    "flat": '<path d="M6 12h12"/>',
    "trend-up": '<path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
    "trend-down": '<path d="M3 7l6 6 4-4 8 8"/><path d="M15 17h6v-6"/>',
    "radar": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4.5"/><path d="M12 12l6-6"/><circle cx="12" cy="12" r="1" fill="currentColor"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 8h.01"/>',
    "check": '<path d="M5 12l4 4 10-10"/>',
    "alert": '<path d="M12 4l9 16H3z"/><path d="M12 10v4"/><path d="M12 17h.01"/>',
}


def icon(name: str, size: int = 14, extra_style: str = "") -> str:
    """Trả về chuỗi <svg> inline. Dùng được trong st.markdown(unsafe_allow_html=True) và HTML nhúng."""
    body = _ICON_PATHS.get(name, "")
    return (
        f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="currentColor" '
        f'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" '
        f'style="vertical-align:-0.15em;{extra_style}" aria-hidden="true">{body}</svg>'
    )


APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --bg:#eceff6; --card-hi:#ffffff; --card-lo:#f8f9fd;
  --ink:#0f172a; --ink2:#475569; --ink3:#94a3b8;
  --line:rgba(15,23,42,.09);
  --up:#059669; --down:#e11d48; --wait:#d97706; --accent:#4f46e5;
  /* Nổi khối mềm: vệt sáng ở mép trên + viền mảnh + nhiều lớp bóng loãng */
  --hl:inset 0 1px 0 rgba(255,255,255,1);
  --ring:0 0 0 1px rgba(15,23,42,.06);
  --sh-sm:var(--hl),var(--ring),0 1px 2px rgba(15,23,42,.06),0 8px 14px -6px rgba(15,23,42,.16);
  --sh-md:var(--hl),var(--ring),0 2px 4px rgba(15,23,42,.05),0 14px 24px -8px rgba(15,23,42,.18),0 28px 44px -24px rgba(30,41,90,.26);
  --sh-lg:var(--hl),var(--ring),0 3px 6px rgba(15,23,42,.06),0 20px 34px -10px rgba(15,23,42,.2),0 44px 70px -32px rgba(30,41,90,.34);
  --well:inset 0 2px 4px rgba(15,23,42,.1),inset 0 0 0 1px rgba(15,23,42,.06);
  --surface:linear-gradient(180deg,var(--card-hi),var(--card-lo));
  --font:'Inter','Segoe UI',Arial,sans-serif;
}
html {font-size:105%;}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] p,
.stButton button, input, textarea, [data-baseweb="select"] * {font-family:var(--font) !important;}
[data-testid="stAppViewContainer"], .stApp {
  background:radial-gradient(900px 420px at 0% 0%,#e4e8ff 0,transparent 62%),radial-gradient(800px 400px at 100% 6%,#dcf3ec 0,transparent 58%),#f2f5fb !important;
  background-attachment:fixed !important; color:var(--ink);
}
[data-testid="stHeader"] {background:transparent !important;}
.block-container {padding-top:1.1rem; padding-bottom:2rem; max-width:min(99vw,1880px); padding-left:1.6rem; padding-right:1.6rem;}

/* Thanh bên */
[data-testid="stSidebar"] {background:#e5e9f2 !important; border-right:1px solid var(--line) !important;}
[data-testid="stSidebar"] h1 {font-size:.95rem !important; font-weight:800 !important; letter-spacing:.08em; color:var(--ink2) !important;}

/* Chữ */
h1, h2, h3 {color:var(--ink) !important; font-weight:800 !important; letter-spacing:-.02em;}
[data-testid="stCaptionContainer"] {color:var(--ink2) !important; font-size:.9em;}
[data-testid="stWidgetLabel"] p {color:var(--ink2) !important; font-weight:600; font-size:.92em;}

/* Nút phụ: khối trắng nổi nhẹ, rê chuột nổi cao hơn, bấm xuống thì chìm */
.stButton button, [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"],
[data-testid="stLinkButton"] a, [data-testid="stPopover"] button {
  background:var(--surface) !important; color:var(--ink) !important; border:none !important;
  border-radius:12px !important; box-shadow:var(--sh-sm) !important; font-weight:600 !important;
  transition:box-shadow .18s ease, transform .18s ease, color .18s ease;
}
.stButton button:not([kind="primary"]):hover, [data-testid="stLinkButton"] a:hover, [data-testid="stPopover"] button:hover {
  box-shadow:var(--sh-md) !important; transform:translateY(-2px); color:var(--accent) !important;
}
.stButton button:not([kind="primary"]):active, [data-testid="stPopover"] button:active {box-shadow:var(--well) !important; transform:translateY(0);}
/* Nút chính: chàm, bóng màu */
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind*="primary"],
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
  background:linear-gradient(180deg,#273449,#0f172a) !important; color:#fff !important; border:none !important;
  border-radius:12px !important; font-weight:700 !important; letter-spacing:.01em;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.22),0 1px 2px rgba(15,23,42,.4),0 12px 20px -8px rgba(15,23,42,.65) !important;
}
.stButton button[kind="primary"] * {color:#fff !important;}
.stButton button, .stButton button p {white-space:nowrap !important;}
.stButton button[kind="primary"]:hover {transform:translateY(-2px); filter:brightness(1.05);}
.stButton button[kind="primary"]:active {transform:translateY(0); box-shadow:inset 0 2px 6px rgba(2,6,23,.6) !important;}

/* Ô nhập liệu / chọn: lõm nhẹ */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div, [data-baseweb="textarea"],
.stNumberInput [data-baseweb="input"] {
  background:#f3f5fa !important; border:none !important; border-radius:11px !important; box-shadow:var(--well) !important;
}
[data-baseweb="input"] input, [data-baseweb="base-input"] input, textarea {background:transparent !important; color:var(--ink) !important;}
[data-baseweb="tag"] {background:#fff !important; color:var(--ink) !important; box-shadow:var(--sh-sm) !important; border-radius:8px !important;}
[data-baseweb="tag"] span {color:var(--ink) !important;}
[data-baseweb="popover"] [role="listbox"], [data-testid="stPopoverBody"] {
  background:#fff !important; border-radius:16px !important; box-shadow:var(--sh-lg) !important; border:none !important;
}

/* Khối nổi chung */
[data-testid="stExpander"] details, [data-testid="stForm"] {
  background:var(--surface) !important; border:none !important; border-radius:18px !important; box-shadow:var(--sh-md) !important;
}
[data-testid="stExpander"] summary {border-radius:18px !important; font-weight:700; color:var(--ink) !important;}
[data-testid="stMetric"] {background:var(--surface); border-radius:18px; padding:1.05rem 1.2rem; box-shadow:var(--sh-md);}
[data-testid="stMetricLabel"] p {color:var(--ink2) !important; font-weight:600 !important; font-size:.8rem !important; letter-spacing:.04em;}
[data-testid="stMetricValue"] {color:var(--ink) !important; font-weight:800 !important; font-size:1.5rem !important; letter-spacing:-.02em;}
[data-testid="stAlert"] {background:var(--surface) !important; border:none !important; border-radius:14px !important; box-shadow:var(--sh-sm) !important; color:var(--ink) !important;}
[data-testid="stAlert"] * {color:var(--ink) !important;}

/* Tab chọn chế độ: rãnh lõm, mục đang chọn nổi lên */
[data-testid="stButtonGroup"] [role="radiogroup"] {background:#e1e5ee; border-radius:14px; padding:5px; box-shadow:var(--well); display:inline-flex !important; width:auto !important; gap:3px;}
[data-testid="stButtonGroup"] button {background:transparent !important; border:none !important; border-radius:10px !important; color:var(--ink2) !important; font-weight:600 !important; box-shadow:none !important; padding:.45rem 1.05rem !important;}
[data-testid="stButtonGroup"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[kind*="Active"], [data-testid="stButtonGroup"] button[data-testid*="Active"] {
  background:linear-gradient(180deg,#273449,#0f172a) !important; color:#fff !important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.2),0 8px 14px -6px rgba(15,23,42,.55) !important;
}
[data-testid="stButtonGroup"] button[aria-checked="true"] * {color:#fff !important;}

/* Iframe: bảng, biểu đồ, đồ thị đặt trong khối nổi */
iframe[title="st_aggrid.AgGrid.agGrid"] {border-radius:18px; box-shadow:var(--sh-md); background:#fff;}
iframe[title="st.iframe"], iframe[title="st.components.v1.html"] {border-radius:18px;}
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {border-radius:22px; box-shadow:var(--sh-md); background:#fff; overflow:hidden; padding:0; box-sizing:border-box;}
[data-testid="stPlotlyChart"] > div {border-radius:22px; overflow:hidden;}

/* Bảng: vùng chuyển mờ dần ở đáy + sóng nước chảy nhẹ, chữ không dính mép khung */
[data-testid="stElementContainer"]:has(iframe[title="st_aggrid.AgGrid.agGrid"]) {position:relative;}
[data-testid="stElementContainer"]:has(iframe[title="st_aggrid.AgGrid.agGrid"])::after {
  content:""; position:absolute; left:0; right:14px; bottom:0; height:74px; pointer-events:none; z-index:3; border-radius:0 0 22px 22px;
  background:
    url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='320' height='22' viewBox='0 0 320 22'><path d='M0 11 C40 0 80 22 120 11 S200 0 240 11 S300 22 320 11 V22 H0Z' fill='%236366f1' fill-opacity='.10'/></svg>") repeat-x 0 100% / 320px 22px,
    url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='240' height='18' viewBox='0 0 240 18'><path d='M0 9 C30 0 60 18 90 9 S150 0 180 9 S210 18 240 9 V18 H0Z' fill='%2310b981' fill-opacity='.09'/></svg>") repeat-x 0 100% / 240px 18px,
    linear-gradient(to top,#ffffff 6%,rgba(255,255,255,.88) 34%,rgba(255,255,255,0) 100%);
  animation:sr-flow 9s linear infinite;
}
@keyframes sr-flow {
  from {background-position:0 100%,0 100%,0 0;}
  to {background-position:320px 100%,-240px 100%,0 0;}
}
@media (prefers-reduced-motion:reduce){[data-testid="stElementContainer"]:has(iframe[title="st_aggrid.AgGrid.agGrid"])::after{animation:none;}}

/* Header trang */
.nm-header {display:flex; flex-direction:column; gap:0; padding:14px 20px 6px; border-radius:20px; background:var(--surface); box-shadow:var(--sh-lg);}
.nm-head-top {display:flex; align-items:center; gap:16px; padding-bottom:12px;}
.nm-status {position:relative; padding:.7rem .4rem .7rem 1.5rem; border-top:1px solid #e9edf4; font-size:.88rem; line-height:1.5; color:var(--ink);}
.nm-status::before {content:""; position:absolute; left:.2rem; top:1.05rem; width:9px; height:9px; border-radius:50%; box-shadow:0 0 0 4px rgba(15,23,42,.06); background:var(--accent);}
.nm-ok::before {background:var(--up);} .nm-bad::before {background:var(--down);} .nm-wait::before {background:var(--wait);}
.nm-ok b {color:var(--up);} .nm-bad b {color:var(--down);}
.nm-logo {width:46px; height:46px; border-radius:14px; display:grid; place-items:center; font-weight:800; color:#fff; line-height:0;
  background:linear-gradient(145deg,#6366f1,#8b5cf6); box-shadow:inset 0 1px 0 rgba(255,255,255,.4),0 10px 18px -8px rgba(79,70,229,.8);}
.nm-title {font-size:1.4rem; font-weight:800; color:var(--ink); letter-spacing:-.025em; line-height:1.15;}
.nm-sub {font-size:.8rem; color:var(--ink2); margin-top:3px;}

/* Thông báo trạng thái dữ liệu */
.demo-banner, .live-banner, .warn-banner, .info-banner, .dev-banner {
  position:relative; padding:.8rem 1rem .8rem 2.2rem; border-radius:16px; margin:.9rem 0 .6rem; font-size:.9rem; line-height:1.5;
  color:var(--ink); background:var(--surface); box-shadow:var(--sh-sm);
}
.demo-banner::before, .live-banner::before, .warn-banner::before, .info-banner::before, .dev-banner::before {
  content:""; position:absolute; left:1rem; top:1.1rem; width:9px; height:9px; border-radius:50%; box-shadow:0 0 0 4px rgba(15,23,42,.06);
}
.live-banner::before {background:var(--up);} .warn-banner::before {background:var(--down);}
.demo-banner::before {background:var(--wait);} .info-banner::before, .dev-banner::before {background:var(--accent);}
.warn-banner b {color:var(--down);} .live-banner b {color:var(--up);}

.info-dot {font-size:.83rem; color:var(--ink3); cursor:help; margin-left:.2rem;}
.section-vn {font-weight:800; font-size:1.1rem; margin:.5rem 0 .7rem; color:var(--ink);}
.price-up {color:var(--up); font-weight:800;} .price-down {color:var(--down); font-weight:800;} .price-flat {color:var(--ink2); font-weight:800;}

/* 3 thẻ chỉ số sàn */
.market-compact-grid {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1.5rem; margin:1.1rem 0 1.4rem;}
.market-card-compact {padding:1.05rem 1.25rem; border-radius:20px; background:var(--surface); box-shadow:var(--sh-lg); min-height:108px; transition:transform .2s ease, box-shadow .2s ease;}
.market-card-compact:hover {transform:translateY(-3px);}
.market-card-compact.tone-up {background:linear-gradient(160deg,#e6f9f0,#ffffff 62%);}
.market-card-compact.tone-down {background:linear-gradient(160deg,#ffecef,#ffffff 62%);}
.market-card-compact.tone-flat {background:linear-gradient(160deg,#eaeeff,#ffffff 62%);}
.market-card-head {display:flex; justify-content:space-between; align-items:center; gap:.55rem; font-size:.78rem;}
.market-card-head b {color:var(--ink2); letter-spacing:.05em; font-weight:700;}
.market-change {font-size:.76rem; font-weight:700; white-space:nowrap; padding:.2rem .6rem; border-radius:999px; background:#f1f4f9; box-shadow:var(--well);}
.market-index-line {display:flex; align-items:baseline; gap:.6rem; margin:.45rem 0 .35rem;}
.market-index-value {font-size:1.75rem; font-weight:800; color:var(--ink); letter-spacing:-.03em;}
.market-status {font-size:.74rem; color:var(--ink2); font-weight:600;}
.market-card-foot {display:flex; justify-content:space-between; gap:.5rem; font-size:.72rem; color:var(--ink2); flex-wrap:wrap;}
.market-up, .market-down, .market-flat {display:inline-flex; align-items:center; gap:2px; margin-right:10px; font-weight:700;}
.market-up {color:var(--up);} .market-down {color:var(--down);} .market-flat {color:var(--ink2);}
.market-change svg {margin-right:2px;}

/* Tổng quan danh mục */
.pf-ov {display:grid; grid-template-columns:1.25fr .95fr 1.55fr; gap:1.5rem; margin:1.1rem 0 1.4rem;}
.pf-card {padding:1.05rem 1.2rem; border-radius:20px; background:var(--surface); box-shadow:var(--sh-lg);}
.pf-k {font-size:.7rem; font-weight:700; letter-spacing:.08em; color:var(--ink3); margin-bottom:.45rem;}
.pf-state {font-size:1.1rem; font-weight:800; margin-bottom:.35rem;}
.pf-desc {font-size:.84rem; line-height:1.55; color:var(--ink);}
.pf-foot {margin-top:.7rem; padding:.5rem .8rem; border-radius:12px; font-size:.76rem; color:var(--ink2); background:#f3f5fa; box-shadow:var(--well);}
.pf-stats {display:grid; grid-template-columns:repeat(3,1fr); gap:.7rem; margin-top:.3rem;}
.pf-stat {padding:.6rem .75rem; border-radius:14px; background:#fff; box-shadow:var(--sh-sm);}
.pf-stat .n {font-size:1.45rem; font-weight:800; line-height:1.1; letter-spacing:-.02em;}
.pf-stat .t {font-size:.72rem; color:var(--ink2); font-weight:600;}
.pf-line {margin:.3rem 0;}
.pf-tag {display:inline-block; padding:.18rem .6rem; border-radius:999px; font-size:.68rem; font-weight:700; margin-right:.4rem; letter-spacing:.03em; background:#fff; box-shadow:var(--sh-sm);}
.pf-tag.down {color:var(--down);} .pf-tag.up {color:var(--up);} .pf-tag.wait {color:var(--wait);}
@media (max-width:1100px){.pf-ov{grid-template-columns:1fr}}

/* Treemap danh mục */
.treemap-card-head {color:var(--ink2); font-size:.8rem; font-weight:700; letter-spacing:.06em; padding:.2rem .3rem .55rem;}
.treemap-card-wrap {border-radius:18px; overflow:hidden;}

/* Thanh cuộn */
::-webkit-scrollbar {width:10px; height:10px;}
::-webkit-scrollbar-track {background:transparent;}
::-webkit-scrollbar-thumb {background:#c3cad8; border-radius:8px; border:2px solid var(--bg);}
@media (max-width:900px){.market-compact-grid{grid-template-columns:1fr}}
</style>
"""

# CSS bơm vào iframe AgGrid (st_aggrid custom_css) để bảng cùng phong cách nổi khối.
AGGRID_CSS = {
    ".ag-root-wrapper": {"border": "none !important", "border-radius": "18px !important", "background": "#ffffff !important"},
    ".ag-theme-streamlit, .ag-root-wrapper, .ag-header, .ag-row, .ag-cell": {"font-family": "'Inter','Segoe UI',Arial,sans-serif !important"},
    ".ag-header": {"background": "#f7f8fc !important", "border-bottom": "1px solid #e9edf4 !important"},
    ".ag-header-cell-text": {"color": "#64748b !important", "font-weight": "600 !important", "font-size": "12px !important", "letter-spacing": ".02em"},
    ".ag-row": {"background": "#ffffff !important", "border-bottom": "1px solid #f0f3f8 !important", "color": "#0f172a !important"},
    ".ag-row-hover": {"background": "#f7f8ff !important"},
    ".ag-row-selected": {"background": "#eef0ff !important", "box-shadow": "inset 3px 0 0 #4f46e5 !important"},
    ".ag-row-selected::before": {"background": "transparent !important"},
    ".ag-cell": {"border": "none !important", "display": "flex", "align-items": "center", "padding-left": "14px !important", "padding-right": "10px !important"},
    ".ag-cell-focus": {"border": "none !important", "outline": "none !important"},
    ".ag-body-viewport, .ag-center-cols-viewport, .ag-body": {"background": "#ffffff !important"},
    ".ag-pinned-left-cols-container, .ag-pinned-left-header": {"border-right": "1px solid #eef1f6 !important", "background": "#ffffff !important"},
    ".ag-row .ag-cell[col-id='Mã']": {"font-weight": "800 !important"},
    ".ag-center-cols-container, .ag-pinned-left-cols-container": {"margin-bottom": "64px !important"},
    ".ag-header-cell": {"padding-left": "14px !important"},
}

# Khung ngoài của giao diện mới: bỏ thanh Deploy, gom thanh trên thành một hàng, chip trạng thái.
RADAR_CSS = """
<style>
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {display:none !important;}
[data-testid="stHeader"] {height:0 !important; min-height:0 !important; background:transparent !important;}
.block-container {padding-top:.9rem !important; padding-bottom:.4rem !important; max-width:min(99vw,1760px) !important;}
.nm-brand {display:flex; align-items:center; gap:12px;}
.nm-brand .nm-logo {width:46px; height:46px;}
.nm-brand .nm-title {font-size:1.35rem;}
.nm-chips {display:flex; gap:10px; flex-wrap:wrap; margin:.1rem 0 .55rem;}
.nm-chip {display:inline-flex; align-items:center; gap:9px; padding:7px 14px; border-radius:999px; background:var(--surface); box-shadow:var(--sh-sm); font-size:.8rem; color:var(--ink2); font-weight:600; max-width:100%;}
.nm-chip i {width:9px; height:9px; border-radius:50%; flex:none; background:var(--accent); box-shadow:0 0 0 4px rgba(79,70,229,.14);}
.nm-chip.nm-ok i {background:var(--up); box-shadow:0 0 0 4px rgba(5,150,105,.15);} .nm-chip.nm-bad i {background:var(--down); box-shadow:0 0 0 4px rgba(225,29,72,.14);}
.nm-chip.nm-wait i {background:var(--wait); box-shadow:0 0 0 4px rgba(217,119,6,.15);}
.nm-chip b {color:var(--ink);}
[data-testid="stHorizontalBlock"] [data-baseweb="select"] > div {min-height:44px;}
[data-testid="stPopover"] button, .stButton button {min-height:44px;}
iframe[title="st.iframe"] {height:calc(100vh - 150px) !important; min-height:560px; border-radius:22px;}
</style>
"""

# Giao diện tối: chỉ ghi đè biến/màu; chèn SAU APP_CSS nên thắng về thứ tự.
DARK_CSS = """
<style>
:root{--bg:#0b1020;--card-hi:#1a2138;--card-lo:#141a2e;--ink:#e8ecf8;--ink2:#a8b3cc;--ink3:#6f7b99;--line:rgba(255,255,255,.08);
  --hl:inset 0 1px 0 rgba(255,255,255,.07);--ring:0 0 0 1px rgba(255,255,255,.06);
  --sh-sm:var(--hl),var(--ring),0 1px 2px rgba(0,0,0,.4),0 8px 14px -6px rgba(0,0,0,.55);
  --sh-md:var(--hl),var(--ring),0 2px 4px rgba(0,0,0,.35),0 14px 24px -8px rgba(0,0,0,.6),0 28px 44px -24px rgba(0,0,0,.6);
  --sh-lg:var(--hl),var(--ring),0 3px 6px rgba(0,0,0,.4),0 20px 34px -10px rgba(0,0,0,.65),0 44px 70px -32px rgba(0,0,0,.7);
  --well:inset 0 2px 5px rgba(0,0,0,.55),inset 0 0 0 1px rgba(255,255,255,.05);}
[data-testid="stAppViewContainer"], .stApp {
  background:radial-gradient(900px 420px at 0% 0%,#1b2150 0,transparent 62%),radial-gradient(800px 400px at 100% 6%,#0f3a33 0,transparent 58%),#0b1020 !important; color:#e8ecf8 !important;}
[data-testid="stSidebar"] {background:#0f1527 !important;}
h1,h2,h3,h4,p,label,span,li,[data-testid="stMarkdownContainer"],[data-testid="stCaptionContainer"],[data-testid="stWidgetLabel"] p {color:#e8ecf8;}
[data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] p {color:#a8b3cc !important;}
.nm-header {background:linear-gradient(180deg,#1a2138,#141a2e) !important;}
.nm-status {border-top-color:rgba(255,255,255,.08) !important; color:#e8ecf8 !important;}
.nm-title {color:#e8ecf8 !important;} .nm-sub {color:#a8b3cc !important;}
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div, [data-baseweb="textarea"], .stNumberInput [data-baseweb="input"] {background:#121829 !important;}
[data-baseweb="input"] input, [data-baseweb="base-input"] input, textarea, [data-baseweb="select"] * {color:#e8ecf8 !important;}
.stButton button:not([kind="primary"]), [data-testid="stBaseButton-secondary"], [data-testid="stPopover"] button, [data-testid="stLinkButton"] a {
  background:linear-gradient(180deg,#232c47,#1b2339) !important; color:#e8ecf8 !important;}
[data-testid="stPopoverBody"], [data-baseweb="popover"] [role="listbox"], [data-testid="stExpander"] details, [data-testid="stForm"], [data-testid="stAlert"] {background:#1a2138 !important; color:#e8ecf8 !important;}
[data-testid="stAlert"] * {color:#e8ecf8 !important;}
[data-testid="stMetric"] {background:linear-gradient(180deg,#1a2138,#141a2e) !important;}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] p {color:#e8ecf8 !important;}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {background:transparent !important; color:#e8ecf8 !important;}
input::placeholder, textarea::placeholder {color:#6f7b99 !important;}
::-webkit-scrollbar-thumb {background:#3a4466 !important; border-color:#0b1020 !important;}
iframe[title="st.iframe"] {background:transparent;}
</style>
"""

HELP = {
    "vnindex": "VN-Index phản ánh biến động chung của cổ phiếu niêm yết trên HOSE. Giá trị và % thay đổi được lấy từ nguồn dữ liệu đang sử dụng.",
    "hnxindex": "HNX-Index phản ánh biến động chung của cổ phiếu niêm yết trên HNX.",
    "upcomindex": "UPCoM-Index phản ánh biến động chung của thị trường UPCoM.",
    "regime": "Trạng thái thị trường là điểm 0–100 tổng hợp từ xu hướng VN-Index, RSI, lợi suất 20 phiên và độ rộng. Điểm cao hơn nghĩa là bối cảnh thuận lợi hơn cho vị thế mua, nhưng không đảm bảo giá sẽ tăng.",
    "breadth": "Độ rộng = tỷ lệ mã tăng giá trong danh sách đang quét. Trên 50% nghĩa là số mã tăng chiếm ưu thế.",
    "risk": "Rủi ro thị trường được suy ra từ điểm trạng thái thị trường. Dùng để điều chỉnh mức thận trọng và tỷ trọng, không phải dự báo chắc chắn.",
    "score": "Điểm kỹ thuật 0–100 thể hiện mức độ tích cực/tiêu cực của tín hiệu. Đây không phải Độ rõ tín hiệu.",
    "confidence": "Độ rõ tín hiệu 50–95 = 52 + 0,75 × |điểm − 50|: điểm càng xa mức trung tính thì càng rõ, theo cả hai phía. Nó KHÔNG phải xác suất khuyến nghị đúng (đã kiểm chứng ngược: gần như không dự báo được gì). Muốn biết hiệu quả thực tế, xem số đo lịch sử của mức điểm.",
    "support": "Hỗ trợ 20 phiên: vùng giá thấp đáng chú ý trong 20 phiên gần đây. Nếu giá thủng vùng này, rủi ro thường tăng.",
    "resistance": "Kháng cự 20 phiên: vùng giá cao đáng chú ý trong 20 phiên gần đây. Khi giá tiến gần vùng này, áp lực bán có thể tăng.",
    "rsi": "RSI(14) đo động lượng trong 14 phiên. Khoảng 50–70 thường cho thấy lực giá khá tích cực; quá cao có thể đi kèm rủi ro mua đuổi.",
    "volume_ratio": "KL/TB20 = khối lượng hiện tại chia cho trung bình 20 phiên. Ví dụ 1,5 lần nghĩa là giao dịch cao hơn khoảng 50% so với bình thường.",
    "total_cost": "Tổng giá vốn = tổng số tiền đã bỏ ra cho số cổ phiếu đang nắm giữ.",
    "market_value": "Tổng giá hiện tại = giá mới nhất × số lượng. Chỉ hiện khi đang dùng dữ liệu thật.",
    "pnl": "Lãi/Lỗ tạm tính = Tổng giá hiện tại − Tổng giá vốn. Chưa tính thuế/phí bán trong tương lai.",
    "top30": "30 mã được xếp hạng từ danh sách quét theo điểm kỹ thuật và mức độ nổi bật. Nhấn vào một dòng để mở ngay phần phân tích và biểu đồ của mã đó.",
    "portfolio": "Mỗi lần mua chỉ cần nhập Mã, Giá mua và Số lượng. Hệ thống tự gộp các lần mua cùng mã thành một dòng và tính giá vốn bình quân.",
}

def embed_html(html_doc: str, height: int) -> None:
    """Nhúng HTML có JavaScript. st.components.v1.html đã bị Streamlit gỡ dần
    (hạn 2026-06-01), nên ưu tiên st.iframe và chỉ dùng API cũ khi bản Streamlit chưa có."""
    if hasattr(st, "iframe"):
        st.iframe(html_doc, height=height)
    else:
        components.html(html_doc, height=height, scrolling=False)


def apply_css():
    st.markdown(APP_CSS, unsafe_allow_html=True)
    # Hạn chế trình duyệt tự dịch các mã viết tắt như HOSE/UPCoM thành từ ngữ sai.
    script = """<script>
    try {
      const d = window.parent.document;
      d.documentElement.lang = 'vi';
      d.documentElement.setAttribute('translate','no');
      d.body && d.body.setAttribute('translate','no');
    } catch(e) {}
    </script>"""
    try:
        st.html(script, unsafe_allow_javascript=True)
    except TypeError:
        # Streamlit cũ chưa có unsafe_allow_javascript.
        try:
            components.html(script, height=0)
        except Exception:
            pass
    except Exception:
        pass


def section_title(text: str, tip: str | None = None):
    safe_text = html.escape(text)
    info = ""
    if tip:
        safe_tip = html.escape(tip, quote=True)
        info = f'<span class="info-dot" title="{safe_tip}">ⓘ</span>'
    st.markdown(f'<div class="section-vn">{safe_text}{info}</div>', unsafe_allow_html=True)


def fmt_price(v):
    if pd.isna(v):
        return "—"
    return f"{float(v):,.0f}"


def fmt_pct(v, digits=2):
    if pd.isna(v):
        return "—"
    return f"{float(v):+.{digits}f}%"


def safe_num(v, default=np.nan):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _clean_history(df: pd.DataFrame) -> pd.DataFrame:
    h = df.copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume", "ema20", "ma50", "ma200"]:
        if c in h.columns:
            h[c] = pd.to_numeric(h[c], errors="coerce")
    h = h.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    for c in ["open", "high", "low"]:
        if c not in h.columns:
            h[c] = h["close"]
        h[c] = h[c].fillna(h["close"])
    if "volume" not in h.columns:
        h["volume"] = 0
    h["volume"] = h["volume"].fillna(0)
    return h


def _chart_epoch(value) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    # Encode wall-clock time as UTC so Lightweight Charts shows Vietnam wall time
    # without an unwanted timezone shift in the embedded component.
    return int(ts.tz_localize("UTC").timestamp())


def _aggregate_chart_frame(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    h = _clean_history(df)
    if h.empty:
        return h
    interval = str(interval or "1D").upper()
    if interval in ("1W", "1M"):
        work = h.copy()
        key = work["date"].dt.to_period("W-FRI" if interval == "1W" else "M")
        groups = []
        for _, g in work.groupby(key, sort=True):
            g = g.sort_values("date")
            groups.append({
                "date": g["date"].iloc[-1],
                "open": float(g["open"].iloc[0]),
                "high": float(pd.to_numeric(g["high"], errors="coerce").max()),
                "low": float(pd.to_numeric(g["low"], errors="coerce").min()),
                "close": float(g["close"].iloc[-1]),
                "volume": float(pd.to_numeric(g["volume"], errors="coerce").fillna(0).sum()),
            })
        h = pd.DataFrame(groups)
    close = pd.to_numeric(h["close"], errors="coerce")
    # MA chỉ vẽ khi đủ số phiên; min_periods=1 trước đây vẽ "MA200" giả ngay từ cây nến đầu.
    h["ema20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    h["ma50"] = close.rolling(50, min_periods=50).mean()
    h["ma200"] = close.rolling(200, min_periods=200).mean()
    h["volume_sma20"] = pd.to_numeric(h.get("volume", 0), errors="coerce").fillna(0).rolling(20, min_periods=1).mean()
    return h.reset_index(drop=True)


def _round_list(values, digits: int = 2) -> list:
    return [None if not np.isfinite(v) else round(float(v), digits) for v in np.asarray(values, dtype=float)]


def _chart_interval_payload(df: pd.DataFrame, interval: str, overlays: list[str]) -> dict:
    """Dữ liệu một khung nến dạng CỘT (mảng số) — JS tự dựng lại đối tượng.

    Dạng cột + làm tròn giúp payload nhỏ hơn nhiều so với danh sách dict cho từng nến.
    """
    h = _aggregate_chart_frame(df, interval)
    if h is None or h.empty:
        return {"t": [], "first": None, "last": None}
    t = [_chart_epoch(d) for d in h["date"]]
    out = {
        "t": t,
        "o": _round_list(h["open"]), "h": _round_list(h["high"]),
        "l": _round_list(h["low"]), "c": _round_list(h["close"]),
        "v": _round_list(h["volume"], 0), "vs": _round_list(h["volume_sma20"], 0),
        "lines": {},
        "first": t[0], "last": t[-1],
    }
    for name in ("EMA20", "MA50", "MA200"):
        key = name.lower()
        if name in overlays and key in h.columns:
            out["lines"][name] = _round_list(h[key])
    return out


def trend_windows(df: pd.DataFrame) -> list[dict]:
    """Tính xu hướng 1/3/6 tháng bằng hồi quy log-price trên toàn cửa sổ."""
    h = _clean_history(df)
    specs = [("1T", 21), ("3T", 63), ("6T", 126)]
    out = []
    for label, sessions in specs:
        if len(h) < max(8, sessions // 2):
            continue
        w = h.tail(min(sessions, len(h))).copy()
        close = pd.to_numeric(w["close"], errors="coerce").dropna()
        if len(close) < 8:
            continue
        x = np.arange(len(close), dtype=float)
        y = np.log(np.maximum(close.to_numpy(dtype=float), 1e-9))
        slope, intercept = np.polyfit(x, y, 1)
        yhat = intercept + slope * x
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = max(0.0, min(1.0, 1.0 - ss_res / ss_tot)) if ss_tot > 1e-12 else 0.0
        ret = (float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100 if float(close.iloc[0]) else 0.0
        slope_21 = (np.exp(slope * 21) - 1) * 100
        direction = 1 if slope_21 > 0.6 else (-1 if slope_21 < -0.6 else 0)
        if r2 >= 0.55:
            strength = "RÕ"
        elif r2 >= 0.28:
            strength = "VỪA"
        else:
            strength = "NHIỄU"
        out.append({
            "label": label,
            "sessions": sessions,
            "return_pct": ret,
            "slope_pct_21": slope_21,
            "log_slope_daily": float(slope),
            "direction": direction,
            "r2": r2,
            "strength": strength,
        })
    return out


def combined_trend_projection(df: pd.DataFrame, horizon: int = 30) -> dict:
    """Gộp xu hướng 1T/3T/6T thành MỘT đường ngoại suy tương lai.

    Trọng số ưu tiên cửa sổ gần hơn nhưng giảm ảnh hưởng của cửa sổ nhiễu bằng R².
    Đây là ngoại suy kỹ thuật từ lịch sử, không phải dự báo xác suất hay cam kết giá.
    """
    h = _clean_history(df)
    trends = trend_windows(h)
    if h.empty or not trends:
        return {"points": [], "components": [], "direction": 0, "projected_return_pct": 0.0, "strength": "—", "confidence": 0.0}

    base_w = {"1T": 0.45, "3T": 0.35, "6T": 0.20}
    num = den = conf_num = 0.0
    for t in trends:
        # Cửa sổ nhiễu vẫn đóng góp nhưng nhẹ hơn, tránh để 1 tháng biến động mạnh lấn át hoàn toàn.
        reliability = 0.30 + 0.70 * float(t.get("r2", 0.0))
        w = base_w.get(t["label"], 0.1) * reliability
        num += float(t.get("log_slope_daily", 0.0)) * w
        conf_num += float(t.get("r2", 0.0)) * w
        den += w
    slope = num / den if den else 0.0
    confidence = conf_num / den if den else 0.0

    # Giới hạn riêng phần HIỂN THỊ để một cửa sổ cực đoan không làm méo toàn biểu đồ.
    raw_ret = float(np.exp(slope * horizon) - 1.0)
    shown_ret = float(np.clip(raw_ret, -0.30, 0.30))
    shown_slope = float(np.log1p(shown_ret) / horizon) if horizon else 0.0

    last_date = pd.Timestamp(h.iloc[-1]["date"]).normalize()
    last_price = float(h.iloc[-1]["close"])
    future_dates = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=horizon)
    points = [{"time": last_date.strftime("%Y-%m-%d"), "value": last_price}]
    for i, d in enumerate(future_dates, start=1):
        points.append({"time": pd.Timestamp(d).strftime("%Y-%m-%d"), "value": float(last_price * np.exp(shown_slope * i))})

    pct = shown_ret * 100.0
    direction = 1 if pct > 1.0 else (-1 if pct < -1.0 else 0)
    strength = "RÕ" if confidence >= 0.55 else ("VỪA" if confidence >= 0.28 else "NHIỄU")
    return {
        "points": points,
        "components": trends,
        "direction": direction,
        "projected_return_pct": pct,
        "strength": strength,
        "confidence": confidence,
        "horizon": horizon,
        "raw_return_pct": raw_ret * 100.0,
    }


def tradingview_lightweight_chart(
    df: pd.DataFrame,
    ticker: str,
    support: float | None = None,
    resistance: float | None = None,
    cost_basis: float | None = None,
    overlays: list[str] | None = None,
    chart_type: str = "Nến",
    height: int = 600,
    info: dict | None = None,
    hourly_df: pd.DataFrame | None = None,
):
    """TradingView Lightweight Charts với toàn bộ thông tin chính đặt ngay trong khung biểu đồ."""
    h = _clean_history(df)
    if h.empty:
        st.info("Chưa có lịch sử giá để vẽ biểu đồ.")
        return

    overlays = overlays or []
    info = info or {}

    daily_payload = _chart_interval_payload(h, "1D", overlays)
    weekly_payload = _chart_interval_payload(h, "1W", overlays)
    monthly_payload = _chart_interval_payload(h, "1M", overlays)
    hourly_payload = _chart_interval_payload(hourly_df, "1H", overlays) if isinstance(hourly_df, pd.DataFrame) and not hourly_df.empty else {
        "t": [], "first": None, "last": None
    }

    projection = combined_trend_projection(h, horizon=30)
    projection = dict(projection or {})
    ppoints = []
    for point in projection.get("points", []) or []:
        q = dict(point)
        try:
            q["time"] = _chart_epoch(pd.Timestamp(q.get("time")))
        except Exception:
            continue
        ppoints.append(q)
    projection["points"] = ppoints

    # 1D không còn bị nhúng 2 lần; 1W/1M gộp từ 1D; dữ liệu dạng cột đã làm tròn.
    payload = {
        "ticker": ticker,
        "intervals": {
            "1H": hourly_payload,
            "1D": daily_payload,
            "1W": weekly_payload,
            "1M": monthly_payload,
        },
        "support": None if support is None or pd.isna(support) else float(support),
        "resistance": None if resistance is None or pd.isna(resistance) else float(resistance),
        "costBasis": None if cost_basis is None or pd.isna(cost_basis) else float(cost_basis),
        "projection": projection,
        "chartType": chart_type,
        "info": info,
        "theme": THEME,
    }
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    t = THEME
    html_doc = f"""
    <link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
      html,body {{margin:0;background:{t['bg']};}}
      #tv-root {{--bg:{t['bg']};--ink:{t['ink']};--ink2:{t['ink2']};--ink3:{t['ink3']};--up:{t['up']};--down:{t['down']};--wait:{t['wait']};--accent:{t['accent']};
        --hl:inset 0 1px 0 #fff;--ring:0 0 0 1px rgba(15,23,42,.06);
        --sh:var(--hl),var(--ring),0 2px 4px rgba(15,23,42,.05),0 14px 24px -8px rgba(15,23,42,.18),0 28px 44px -24px rgba(30,41,90,.26);
        --sh-sm:var(--hl),var(--ring),0 1px 2px rgba(15,23,42,.06),0 8px 14px -6px rgba(15,23,42,.16);
        --well:inset 0 2px 4px rgba(15,23,42,.1),inset 0 0 0 1px rgba(15,23,42,.06);
        box-sizing:border-box;height:{height - 36}px;margin:16px 14px 20px;display:flex;flex-direction:column;gap:18px;
        font-family:'Inter','Segoe UI',Arial,sans-serif;color:var(--ink);}}
      #tv-root * {{box-sizing:border-box;}}
      .nm-card {{background:linear-gradient(180deg,#ffffff,#f8f9fd);border-radius:20px;box-shadow:var(--sh);}}
      .nm-well {{background:#f3f5fa;border-radius:12px;box-shadow:var(--well);}}
      #info-strip {{flex:0 0 auto;padding:16px 20px 14px;}}
      .head {{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap;}}
      #info-ticker {{font-size:25px;font-weight:800;letter-spacing:-.03em;}}
      #info-price {{font-size:22px;font-weight:800;letter-spacing:-.02em;}}
      #info-place {{font-size:12px;color:var(--ink2);font-weight:600;padding:3px 10px;}}
      #info-source {{margin-left:auto;font-size:11.5px;color:var(--ink3);}}
      #info-metrics {{display:flex;gap:9px;flex-wrap:wrap;margin-top:12px;font-size:12px;}}
      #info-metrics span {{padding:5px 11px;border-radius:999px;background:#fff;color:var(--ink2);box-shadow:var(--sh-sm);}}
      #info-metrics b {{color:var(--ink);}}
      #info-rec {{margin-top:12px;font-size:13.2px;line-height:1.5;color:var(--ink);}}
      .sec {{margin-top:12px;padding:12px 14px;font-size:12.8px;line-height:1.55;color:var(--ink2);}}
      .sec b.k {{color:var(--ink);font-weight:700;letter-spacing:.01em;}}
      .sec .row + .row {{margin-top:5px;}}
      #buy-action, #portfolio-action {{font-weight:800;}}
      .qa {{margin-top:12px;}}
      .qa-title {{font-size:11.5px;font-weight:700;letter-spacing:.08em;color:var(--ink3);margin:0 0 8px 2px;}}
      .qa-row {{display:flex;gap:10px;align-items:center;}}
      #qa-input {{flex:1 1 auto;height:40px;border:none;outline:none;padding:0 14px;font:inherit;font-size:12.8px;color:var(--ink);
        background:#f3f5fa;border-radius:12px;box-shadow:var(--well);}}
      #qa-ask {{height:40px;padding:0 20px;border:none;border-radius:12px;color:#fff;font:inherit;font-weight:700;font-size:12.5px;cursor:pointer;
        background:linear-gradient(180deg,#6366f1,#4f46e5);box-shadow:inset 0 1px 0 rgba(255,255,255,.35),0 1px 2px rgba(49,46,129,.35),0 10px 18px -8px rgba(79,70,229,.7);}}
      #qa-ask:active {{box-shadow:inset 0 2px 6px rgba(30,27,75,.5);}}
      #qa-output {{display:none;margin-top:12px;padding:12px 14px;font-size:12.8px;line-height:1.5;}}
      .toolbar {{flex:0 0 auto;display:flex;align-items:center;gap:8px;padding:10px 14px;min-width:0;}}
      .toolbar button {{white-space:nowrap;flex:0 0 auto;border:none;background:linear-gradient(180deg,#fff,#f6f7fb);color:var(--ink2);
        font:inherit;font-size:11.5px;font-weight:600;border-radius:9px;padding:6px 11px;cursor:pointer;
        box-shadow:var(--sh-sm);transition:transform .15s,box-shadow .15s,color .15s;}}
      .toolbar button:hover {{color:var(--accent);transform:translateY(-1px);}}
      .toolbar button.active {{color:#fff;background:linear-gradient(180deg,#6366f1,#4f46e5);box-shadow:inset 0 1px 0 rgba(255,255,255,.35),0 8px 14px -6px rgba(79,70,229,.7);}}
      .toolbar .lbl {{margin-left:6px;color:var(--ink3);font-size:11px;font-weight:600;}}
      #candle-interval {{height:30px;min-width:92px;border:none;outline:none;padding:0 10px;font:inherit;font-size:11.5px;font-weight:600;color:var(--ink);
        background:#f3f5fa;border-radius:9px;box-shadow:var(--well);}}
      #candle-interval:disabled {{opacity:.45;}}
      #trend-summary {{margin-left:auto;font-size:12px;white-space:nowrap;min-width:0;overflow:hidden;text-overflow:ellipsis;}}
      .toolbar .lbl, #candle-interval {{flex:0 0 auto;white-space:nowrap;}}
      .chart-card {{position:relative;flex:1 1 auto;min-height:320px;padding:14px;}}
      .chart-well {{position:relative;height:100%;border-radius:14px;overflow:hidden;background:#fff;box-shadow:var(--well);}}
      #tv-chart {{height:100%;width:100%;}}
      #ohlc-tip {{position:absolute;left:14px;top:10px;z-index:4;font-size:12px;color:var(--ink2);font-weight:600;pointer-events:none;}}
      .legend {{position:absolute;left:14px;bottom:30px;z-index:4;font-size:11px;color:var(--ink2);pointer-events:none;
        background:rgba(255,255,255,.9);padding:3px 9px;border-radius:999px;box-shadow:var(--sh-sm);}}
      #load-error {{display:none;padding:18px;color:var(--down);}}
      .foot {{flex:0 0 auto;display:flex;justify-content:space-between;gap:10px;font-size:11px;color:var(--ink3);padding:0 6px;}}
      .foot a {{color:var(--ink3);white-space:nowrap;}}
    </style>
    <div id="tv-root" translate="no" class="notranslate">
      <div id="info-strip" class="nm-card">
        <div class="head">
          <span id="info-ticker"></span>
          <span id="info-price"></span>
          <span id="info-place" class="nm-well"></span>
          <span id="info-source"></span>
        </div>
        <div id="info-metrics"></div>
        <div id="info-rec"></div>
        <div id="decision-detail" class="sec nm-well">
          <div id="buy-action-row" class="row" style="font-size:14px"><b class="k">KHUYẾN NGHỊ MUA HIỆN TẠI:</b> <span id="buy-action"></span></div>
          <div class="row"><b class="k">VÌ SAO:</b> <span id="why-text"></span></div>
          <div class="row"><b class="k" style="color:var(--accent)">DẤU HIỆU CHỜ ĐIỂM MUA:</b> <span id="entry-text"></span></div>
          <div id="invalid-row" class="row"><b class="k" style="color:var(--wait)">TÍN HIỆU HỦY KỊCH BẢN:</b> <span id="invalid-text"></span></div>
          <div id="portfolio-row" style="display:none;margin-top:8px;padding-top:8px;border-top:1px dashed #c3ccd8">
            <div id="portfolio-action-line" class="row" style="font-size:14px"><b class="k">HÀNH ĐỘNG HIỆN TẠI:</b> <span id="portfolio-action"></span></div>
            <div class="row"><b class="k">VỊ THẾ:</b> <span id="portfolio-status"></span></div>
            <div class="row"><b class="k" style="color:var(--up)">GIỮ KHI:</b> <span id="portfolio-hold"></span></div>
            <div class="row"><b class="k" style="color:var(--down)">CẮT LỖ/GIẢM TỶ TRỌNG KHI:</b> <span id="portfolio-cut"></span></div>
            <div class="row"><b class="k" style="color:var(--wait)">CHỐT LỜI KHI:</b> <span id="portfolio-take"></span></div>
          </div>
        </div>
        <div id="qa-box-1619i" class="qa" data-build="qa-1619i">
          <div class="qa-title">HỎI ĐÁP NHANH VỀ MÃ ĐANG CHỌN</div>
          <div class="qa-row">
            <input id="qa-input" type="text" placeholder="Nhập câu hỏi, ví dụ: nếu thủng hỗ trợ thì làm gì?" />
            <button id="qa-ask">Phân tích</button>
          </div>
          <div id="qa-output" class="nm-well">
            <div style="margin-bottom:4px"><b class="k" style="color:var(--accent)">TRẢ LỜI:</b> <span id="qa-reply"></span></div>
            <div style="margin-bottom:4px"><b class="k" style="color:var(--up)">KHUYẾN NGHỊ:</b> <span id="qa-reco" style="font-weight:800"></span></div>
            <div><b class="k" style="color:var(--wait)">PHÂN TÍCH:</b> <span id="qa-analysis"></span></div>
          </div>
        </div>
      </div>
      <div class="toolbar nm-card">
        <button data-range="1m">1T</button><button data-range="3m">3T</button><button data-range="6m">6T</button><button data-range="1y">1N</button><button data-range="3y">3N</button><button data-range="all">Tất cả</button>
        <span class="lbl">1 nến:</span>
        <select id="candle-interval" title="Chọn khung thời gian của một cây nến">
          <option value="1H">1 giờ</option><option value="1D" selected>1 ngày</option><option value="1W">1 tuần</option><option value="1M">1 tháng</option>
        </select>
        <span id="trend-summary"></span>
      </div>
      <div class="chart-card nm-card">
        <div class="chart-well">
          <div id="tv-chart"></div>
          <div id="ohlc-tip"></div>
          <div class="legend"><span style="color:{t['up']}">■ KL tăng</span> · <span style="color:{t['down']}">■ KL giảm</span> · <span style="color:{t['accent']}">━ SMA20 KL</span></div>
          <div id="load-error">Không tải được TradingView Lightweight Charts. Hãy kiểm tra kết nối Internet rồi tải lại trang.</div>
        </div>
      </div>
      <div class="foot">
        <span>HT/KC = hỗ trợ/kháng cự. Đường chấm tương lai = ngoại suy xu hướng tổng hợp 1T–3T–6T, không phải cam kết giá.</span>
        <a href="https://www.tradingview.com/" target="_blank" rel="noopener">TradingView Lightweight Charts™</a>
      </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@5.0.6/dist/lightweight-charts.standalone.production.js"></script>
    <script>
    (function() {{
      const D = {data_json};
      const T = D.theme || {{}};
      // Dựng lại đối tượng từ dữ liệu dạng cột (nhỏ hơn nhiều khi truyền).
      const UPV = 'rgba(5,150,105,.45)', DNV = 'rgba(220,38,38,.42)';
      function expand(set) {{
        if (!set || !Array.isArray(set.t) || !set.t.length) return {{candles:[],volumes:[],volumeSma20:[],lines:{{}},first:null,last:null}};
        const candles = [], volumes = [], volumeSma20 = [], lines = {{}};
        for (let i = 0; i < set.t.length; i++) {{
          const time = set.t[i], o = set.o[i], h = set.h[i], l = set.l[i], c = set.c[i];
          candles.push({{time, open:o, high:h, low:l, close:c}});
          volumes.push({{time, value:set.v[i] || 0, color: c >= o ? UPV : DNV}});
          volumeSma20.push({{time, value:set.vs[i] || 0}});
        }}
        Object.entries(set.lines || {{}}).forEach(([name, vals]) => {{
          lines[name] = [];
          vals.forEach((v, i) => {{ if (v !== null && v !== undefined) lines[name].push({{time:set.t[i], value:v}}); }});
        }});
        return {{candles, volumes, volumeSma20, lines, first:set.first, last:set.last}};
      }}
      const raw = D.intervals || {{}};
      D.intervals = {{}};
      ['1H','1D','1W','1M'].forEach(k => {{ D.intervals[k] = expand(raw[k]); }});
      D.candles = D.intervals['1D'].candles; D.volumes = D.intervals['1D'].volumes;
      D.volumeSma20 = D.intervals['1D'].volumeSma20; D.lines = D.intervals['1D'].lines;
      D.first = D.intervals['1D'].first; D.last = D.intervals['1D'].last;

      const L = window.LightweightCharts;
      const I = D.info || {{}};
      const UP = T.up || '#059669', DOWN = T.down || '#e11d48', WAIT = T.wait || '#d97706', INK2 = T.ink2 || '#5b6b80', ACC = T.accent || '#4f46e5';
      const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}}[c]));
      const nfmt = v => (v===null || v===undefined || Number.isNaN(Number(v))) ? '—' : Math.round(Number(v)).toLocaleString('en-US');
      const pfmt = v => (v===null || v===undefined || Number.isNaN(Number(v))) ? '—' : `${{Number(v)>=0?'+':''}}${{Number(v).toFixed(2)}}%`;
      const SV = (p) => `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px">${{p}}</svg>`;
      const ICO = {{up:SV('<path d="M6 15l6-6 6 6"/>'), down:SV('<path d="M6 9l6 6 6-6"/>'), flat:SV('<path d="M6 12h12"/>')}};
      const chg = Number(I.change_pct || 0);
      const up = chg > 0, down = chg < 0;
      const pcolor = up ? UP : (down ? DOWN : INK2);
      document.getElementById('info-ticker').textContent = D.ticker;
      const priceEl = document.getElementById('info-price');
      priceEl.style.color = pcolor;
      priceEl.innerHTML = `${{nfmt(I.price)}} ${{up?ICO.up:(down?ICO.down:ICO.flat)}} ${{pfmt(chg)}}`;
      const placeText = [I.sector,I.exchange].filter(Boolean).join(' · ');
      const placeEl = document.getElementById('info-place');
      placeEl.textContent = placeText; if (!placeText) placeEl.style.display = 'none';
      document.getElementById('info-source').textContent = [I.date_text,I.source].filter(Boolean).join(' · ');
      const metricBits = [
        `<span>Độ rõ <b>${{esc(I.confidence ?? '—')}}/100</b></span>`,
        `<span>RSI <b>${{esc(I.rsi ?? '—')}}</b></span>`,
        `<span>KL/TB20 <b>${{esc(I.volume_ratio ?? '—')}} lần</b></span>`,
      ];
      if (I.quantity !== null && I.quantity !== undefined && !Number.isNaN(Number(I.quantity))) {{
        metricBits.push(`<span>SL <b>${{nfmt(I.quantity)}}</b></span>`);
      }}
      if (I.avg_cost !== null && I.avg_cost !== undefined && !Number.isNaN(Number(I.avg_cost))) {{
        metricBits.push(`<span>Giá vốn <b>${{nfmt(I.avg_cost)}}</b></span>`);
      }}
      if (I.market_value !== null && I.market_value !== undefined && !Number.isNaN(Number(I.market_value))) {{
        metricBits.push(`<span>Giá trị <b>${{nfmt(I.market_value)}}</b></span>`);
      }}
      if (I.pnl_pct !== null && I.pnl_pct !== undefined && !Number.isNaN(Number(I.pnl_pct))) {{
        const pv = Number(I.pnl_pct); const pc = pv>=0 ? UP : DOWN;
        metricBits.push(`<span>Lãi/Lỗ <b style="color:${{pc}}">${{pv>=0?ICO.up:ICO.down}} ${{Math.abs(pv).toFixed(2)}}%</b></span>`);
      }}
      if (I.pnl_amount !== null && I.pnl_amount !== undefined && !Number.isNaN(Number(I.pnl_amount))) {{
        const pa = Number(I.pnl_amount); const pc = pa>=0 ? UP : DOWN;
        metricBits.push(`<span>P/L <b style="color:${{pc}}">${{pa>=0?'+':'-'}}${{nfmt(Math.abs(pa))}}</b></span>`);
      }}
      document.getElementById('info-metrics').innerHTML = metricBits.join('');
      const demo = I.demo_warning ? `<b style="color:${{DOWN}}">MÔ PHỎNG · </b>` : '';
      document.getElementById('info-rec').innerHTML = `${{demo}}<b>${{esc(I.recommendation || '')}}</b>${{I.explanation ? ' · '+esc(I.explanation) : ''}}`;
      const buyActionEl = document.getElementById('buy-action');
      const buyAction = I.buy_action || 'CHƯA XÁC ĐỊNH';
      buyActionEl.textContent = buyAction;
      buyActionEl.style.color = buyAction.includes('CÓ THỂ MUA') ? UP : (buyAction.includes('CHƯA NÊN') ? DOWN : WAIT);
      document.getElementById('why-text').textContent = I.why || 'Chưa đủ dữ liệu để giải thích sâu.';
      document.getElementById('entry-text').textContent = I.entry || 'Chưa đủ dữ liệu để xác định điều kiện chờ điểm mua.';
      document.getElementById('invalid-text').textContent = I.invalid || 'Chưa xác định.';
      if (I.portfolio_plan) {{
        document.getElementById('portfolio-row').style.display = 'block';
        const actionEl = document.getElementById('portfolio-action');
        const actionText = I.portfolio_plan.action || 'CHƯA XÁC ĐỊNH';
        actionEl.textContent = actionText;
        actionEl.style.color = actionText.includes('BÁN') ? DOWN : (actionText.includes('GIỮ') ? UP : WAIT);
        document.getElementById('portfolio-status').textContent = I.portfolio_plan.status || '';
        document.getElementById('portfolio-hold').textContent = I.portfolio_plan.hold || 'Chưa xác định.';
        document.getElementById('portfolio-cut').textContent = I.portfolio_plan.cut || 'Chưa xác định.';
        document.getElementById('portfolio-take').textContent = I.portfolio_plan.take || 'Chưa xác định.';
      }}
      const qaInput = document.getElementById('qa-input');
      const qaAsk = document.getElementById('qa-ask');
      const qaOut = document.getElementById('qa-output');
      const qaReply = document.getElementById('qa-reply');
      const qaReco = document.getElementById('qa-reco');
      const qaAnalysis = document.getElementById('qa-analysis');
      const numFmt = (v) => (v===null || v===undefined || Number.isNaN(Number(v))) ? '—' : Math.round(Number(v)).toLocaleString('en-US');
      const recoColor = (txt) => txt.includes('BÁN') || txt.includes('GIẢM') || txt.includes('TRÁNH') ? DOWN : ((txt.includes('CHỜ') || txt.includes('QUAN SÁT') || txt.includes('XÁC NHẬN')) ? WAIT : UP);
      function buildQAAnswer(q) {{
        const t = String(q || '').trim();
        const s = t.toLowerCase();
        const buyAction = String(I.buy_action || 'CHỜ XÁC NHẬN');
        const plan = I.portfolio_plan || null;
        const support = D.support;
        const resistance = D.resistance;
        const ema20v = D.lines && D.lines.EMA20 && D.lines.EMA20.length ? D.lines.EMA20[D.lines.EMA20.length - 1].value : null;
        const ma50v = D.lines && D.lines.MA50 && D.lines.MA50.length ? D.lines.MA50[D.lines.MA50.length - 1].value : null;
        let reply = `Với ${{D.ticker}}, hành động ưu tiên hiện tại là ${{buyAction.toLowerCase()}}.`;
        let reco = plan ? (plan.action || buyAction) : buyAction;
        let analysis = [I.why || '', I.entry || '', I.invalid || ''].filter(Boolean).join(' ');

        if (!s) return {{reply, reco, analysis}};
        if (s.includes('mua') || s.includes('vào') || s.includes('vao') || s.includes('mở vị thế') || s.includes('mo vi the')) {{
          reco = buyAction;
          reply = `Hiện tại với ${{D.ticker}}, hệ thống nghiêng về ${{buyAction.toLowerCase()}}.`;
          analysis = `${{I.why || ''}} Điểm vào theo dõi: ${{I.entry || ''}} Tránh mua khi: ${{I.invalid || ''}}`.trim();
        }} else if (s.includes('giữ') || s.includes('giu') || s.includes('nắm giữ') || s.includes('nam giu')) {{
          reco = plan ? (plan.action || 'GIỮ / THEO DÕI') : buyAction;
          reply = `Nếu tiếp tục giữ ${{D.ticker}}, nên bám theo các mốc kỹ thuật gần thay vì giữ vô điều kiện.`;
          analysis = plan ? `${{plan.hold || ''}} ${{plan.cut || ''}}`.trim() : `${{I.why || ''}} ${{I.invalid || ''}}`.trim();
        }} else if (s.includes('bán') || s.includes('ban') || s.includes('chốt') || s.includes('chot') || s.includes('giảm tỷ trọng') || s.includes('giam ty trong')) {{
          reco = plan ? (plan.action || 'BÁN / GIẢM TỶ TRỌNG') : 'CHỐT TỪNG PHẦN / QUAN SÁT';
          reply = `Khi hỏi về bán/chốt lời, ưu tiên bán theo điều kiện xác nhận chứ không mặc định bán ngay.`;
          analysis = plan ? `${{plan.take || ''}} ${{plan.cut || ''}}`.trim() : `${{I.invalid || ''}} ${{I.entry || ''}}`.trim();
        }} else if (s.includes('thủng hỗ trợ') || s.includes('thung ho tro') || s.includes('gãy hỗ trợ') || s.includes('gay ho tro') || s.includes('mất hỗ trợ') || s.includes('mat ho tro') || s.includes('ema20') || s.includes('ma50')) {{
          reco = plan ? (plan.action || 'BÁN / GIẢM TỶ TRỌNG') : 'TẠM DỪNG MUA / QUAN SÁT RỦI RO';
          const lv = [];
          if (support!==null && support!==undefined) lv.push(`hỗ trợ ${{numFmt(support)}}`);
          if (ema20v!==null && ema20v!==undefined && s.includes('ema20')) lv.push(`EMA20 ${{numFmt(ema20v)}}`);
          if (ma50v!==null && ma50v!==undefined && s.includes('ma50')) lv.push(`MA50 ${{numFmt(ma50v)}}`);
          reply = `Nếu ${{D.ticker}} thủng ${{lv.length ? lv.join(' / ') : 'mốc hỗ trợ gần'}} thì nên ưu tiên hạ rủi ro thay vì kỳ vọng hồi ngay.`;
          analysis = plan ? (plan.cut || '') : (I.invalid || '');
        }} else if (s.includes('vượt kháng cự') || s.includes('vuot khang cu') || s.includes('vượt cản') || s.includes('vuot can') || s.includes('breakout')) {{
          reco = plan ? 'GIỮ THÊM / CHỐT TỪNG PHẦN' : 'THEO DÕI MUA / GIỮ THÊM';
          reply = `Nếu ${{D.ticker}} vượt kháng cự ${{numFmt(resistance)}} với khối lượng xác nhận, nên chờ xác nhận giữ cản thay vì mua đuổi mù quáng.`;
          analysis = plan ? (plan.take || '') : (I.entry || '');
        }} else if (s.includes('lỗ') || s.includes('lo') || s.includes('âm') || s.includes('am') || s.includes('kẹp')) {{
          reco = plan ? (plan.action || 'GIỮ / GIẢM TỶ TRỌNG') : 'QUAN SÁT MỐC CẮT LỖ';
          reply = `Khi vị thế đang lỗ, điều quan trọng là quản trị rủi ro theo mốc kỹ thuật, không bình quân giá xuống vô điều kiện.`;
          analysis = plan ? `${{plan.status || ''}} ${{plan.cut || ''}}`.trim() : (I.invalid || '');
        }} else if (s.includes('lãi') || s.includes('lai') || s.includes('có lời') || s.includes('co loi') || s.includes('chốt lời') || s.includes('chot loi')) {{
          reco = plan ? (plan.action || 'GIỮ / CHỐT TỪNG PHẦN') : 'CHỐT TỪNG PHẦN KHI GẶP CẢN';
          reply = `Nếu vị thế đang có lãi, nên ưu tiên chốt theo kháng cự hoặc khi tín hiệu suy yếu thay vì cố giữ toàn bộ.`;
          analysis = plan ? (plan.take || '') : (`Kháng cự gần ${{numFmt(resistance)}}. ` + (I.invalid || '')).trim();
        }} else if (s.includes('hôm nay') || s.includes('hom nay') || s.includes('phiên này') || s.includes('phien nay') || s.includes('tăng hay giảm') || s.includes('tang hay giam')) {{
          reply = `Trong phiên cập nhật gần nhất, ${{D.ticker}} đang biến động ${{pfmt(I.change_pct)}} so với phiên trước.`;
          reco = String(I.recommendation || buyAction);
          analysis = `Giá hiện tại: ${{numFmt(I.price)}}. ${{I.why || ''}}`.trim();
        }}
        return {{reply, reco, analysis}};
      }}
      function runQA() {{
        const ans = buildQAAnswer(qaInput ? qaInput.value : '');
        qaReply.textContent = ans.reply || 'Chưa có trả lời.';
        qaReco.textContent = ans.reco || 'CHỜ THÊM DỮ LIỆU';
        qaReco.style.color = recoColor(String(ans.reco || ''));
        qaAnalysis.textContent = ans.analysis || 'Chưa có phân tích bổ sung.';
        qaOut.style.display = 'block';
      }}
      if (qaAsk) qaAsk.addEventListener('click', runQA);
      if (qaInput) qaInput.addEventListener('keydown', (e) => {{ if (e.key === 'Enter') {{ e.preventDefault(); runQA(); }} }});

      if (!L) {{ document.getElementById('load-error').style.display='block'; return; }}
      const el = document.getElementById('tv-chart');
      const chart = L.createChart(el, {{
        autoSize:true,
        layout:{{background:{{type:'solid',color:'#ffffff'}},textColor:INK2,fontSize:12}},
        localization:{{priceFormatter:(price)=>Math.round(Number(price)).toLocaleString('en-US')}},
        grid:{{vertLines:{{color:'#eef1f6'}},horzLines:{{color:'#eef1f6'}}}},
        crosshair:{{mode:L.CrosshairMode.Normal}},
        rightPriceScale:{{borderColor:'#e3e8f0',scaleMargins:{{top:0.08,bottom:0.24}}}},
        timeScale:{{visible:true,borderVisible:true,ticksVisible:true,borderColor:'#e3e8f0',timeVisible:false,secondsVisible:false,rightOffset:8,barSpacing:7,minBarSpacing:2}},
        handleScroll:{{mouseWheel:true,pressedMouseMove:true,horzTouchDrag:true,vertTouchDrag:true}},
        handleScale:{{axisPressedMouseMove:true,mouseWheel:true,pinch:true}},
      }});

      let main;
      if (D.chartType === 'Đường') {{
        main = chart.addSeries(L.LineSeries, {{color:ACC,lineWidth:2,priceLineVisible:false}});
        main.setData(D.candles.map(x => ({{time:x.time,value:x.close}})));
      }} else if (D.chartType === 'Vùng') {{
        main = chart.addSeries(L.AreaSeries, {{lineColor:ACC,topColor:'rgba(79,70,229,.28)',bottomColor:'rgba(79,70,229,.02)',lineWidth:2,priceLineVisible:false}});
        main.setData(D.candles.map(x => ({{time:x.time,value:x.close}})));
      }} else {{
        main = chart.addSeries(L.CandlestickSeries, {{upColor:UP,downColor:DOWN,borderVisible:false,wickUpColor:UP,wickDownColor:DOWN}});
        main.setData(D.candles);
      }}

      const vol = chart.addSeries(L.HistogramSeries, {{priceFormat:{{type:'volume'}},priceScaleId:'vol',lastValueVisible:false,priceLineVisible:false}});
      vol.setData(D.volumes);
      const volSma20 = chart.addSeries(L.LineSeries, {{color:ACC,lineWidth:2,priceScaleId:'vol',priceFormat:{{type:'volume'}},priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false,title:'SMA20 KL'}});
      volSma20.setData(D.volumeSma20 || []);
      chart.priceScale('vol').applyOptions({{scaleMargins:{{top:0.80,bottom:0.08}}}});

      const maColors = {{EMA20:'#f59e0b',MA50:'#7c3aed',MA200:'#0f172a'}};
      const maSeries = {{}};
      ['EMA20','MA50','MA200'].forEach(name => {{
        if (!(name in (D.lines || {{}}))) return;
        const s = chart.addSeries(L.LineSeries, {{color:maColors[name],lineWidth:1,priceLineVisible:false,lastValueVisible:false,title:name}});
        s.setData((D.lines || {{}})[name] || []);
        maSeries[name] = s;
      }});

      // Hỗ trợ/kháng cự: đường ngang nét đứt, giá hiển thị trực tiếp ở trục phải.
      if (D.support !== null) main.createPriceLine({{price:D.support,color:UP,lineWidth:1,lineStyle:L.LineStyle.Dashed,axisLabelVisible:true,title:'HT'}});
      if (D.resistance !== null) main.createPriceLine({{price:D.resistance,color:DOWN,lineWidth:1,lineStyle:L.LineStyle.Dashed,axisLabelVisible:true,title:'KC'}});
      if (D.costBasis !== null) main.createPriceLine({{price:D.costBasis,color:ACC,lineWidth:1,lineStyle:L.LineStyle.Dotted,axisLabelVisible:true,title:'GV'}});

      // Chỉ MỘT đường tương lai tổng hợp 1T–3T–6T.
      const P = D.projection || {{}};
      let projSeries = null;
      if (Array.isArray(P.points) && P.points.length > 1) {{
        const c = P.direction > 0 ? UP : (P.direction < 0 ? DOWN : WAIT);
        const proj = chart.addSeries(L.LineSeries, {{
          color:c,lineWidth:2,lineStyle:L.LineStyle.Dashed,priceLineVisible:false,lastValueVisible:false,
          crosshairMarkerVisible:true,title:'Xu hướng tổng hợp 1T–3T–6T'
        }});
        projSeries = proj;
        proj.setData(P.points);
        const lastP = P.points[P.points.length-1];
        const marker = {{
          time:lastP.time,
          position:P.direction >= 0 ? 'belowBar' : 'aboveBar',
          color:c,
          shape:P.direction>0 ? 'arrowUp' : (P.direction<0 ? 'arrowDown' : 'circle'),
          text:`Tổng hợp ${{P.direction>0?'▲':(P.direction<0?'▼':'→')}} ${{Number(P.projected_return_pct||0)>=0?'+':''}}${{Number(P.projected_return_pct||0).toFixed(1)}}% / ${{P.horizon||30}} phiên · ${{P.strength||''}}`
        }};
        if (L.createSeriesMarkers) L.createSeriesMarkers(proj, [marker]);
        else if (proj.setMarkers) proj.setMarkers([marker]);
      }}

      const comps = (P.components || []).map(t => `${{t.label}} ${{Number(t.return_pct||0)>=0?'+':''}}${{Number(t.return_pct||0).toFixed(1)}}%`).join(' · ');
      const overall = `Tổng hợp ${{P.direction>0?ICO.up:(P.direction<0?ICO.down:ICO.flat)}} ${{Number(P.projected_return_pct||0)>=0?'+':''}}${{Number(P.projected_return_pct||0).toFixed(1)}}%/${{P.horizon||30}}P · ${{P.strength||'—'}}`;
      const trendEl = document.getElementById('trend-summary');
      const dailyTrendHtml = `<span style="color:${{INK2}}">${{comps}}</span><span style="margin-left:8px;color:${{P.direction>0?UP:(P.direction<0?DOWN:WAIT)}};font-weight:700">${{overall}}</span>`;
      trendEl.innerHTML = dailyTrendHtml;

      let currentInterval = '1D';
      let currentSet = (D.intervals || {{}})['1D'] || {{candles:D.candles,volumes:D.volumes,volumeSma20:D.volumeSma20,lines:D.lines,first:D.first,last:D.last}};
      let currentFirst = Number(currentSet.first || D.first || 0);
      let currentLast = Number(currentSet.last || D.last || 0);
      const projectionLast = (P.points && P.points.length) ? Number(P.points[P.points.length-1].time) : currentLast;

      function setMainData(candles) {{
        if (D.chartType === 'Đường' || D.chartType === 'Vùng') main.setData((candles||[]).map(x => ({{time:x.time,value:x.close}})));
        else main.setData(candles || []);
      }}

      function applyInterval(kind) {{
        const set = (D.intervals || {{}})[kind];
        if (!set || !Array.isArray(set.candles) || !set.candles.length) return false;
        currentInterval = kind;
        currentSet = set;
        currentFirst = Number(set.first || set.candles[0].time);
        currentLast = Number(set.last || set.candles[set.candles.length-1].time);
        setMainData(set.candles);
        vol.setData(set.volumes || []);
        volSma20.setData(set.volumeSma20 || []);
        Object.entries(maSeries).forEach(([name,series]) => series.setData(((set.lines||{{}})[name]) || []));
        if (projSeries) projSeries.setData(kind === '1D' ? (P.points || []) : []);
        trendEl.innerHTML = kind === '1D' ? dailyTrendHtml : `<span style="color:${{INK2}}">Khung nến ${{kind}} · Xu hướng tổng hợp 1T–3T–6T vẫn tính trên dữ liệu ngày</span>`;
        chart.timeScale().applyOptions({{timeVisible: kind === '1H', secondsVisible:false}});
        chart.timeScale().fitContent();
        return true;
      }}

      const intervalSelect = document.getElementById('candle-interval');
      const hourly = (D.intervals || {{}})['1H'];
      if (!hourly || !Array.isArray(hourly.candles) || !hourly.candles.length) {{
        const hopt = intervalSelect.querySelector('option[value="1H"]');
        if (hopt) {{ hopt.disabled = true; hopt.textContent = '1 giờ (nguồn chưa có)'; }}
      }}
      intervalSelect.addEventListener('change', () => {{
        if (!applyInterval(intervalSelect.value)) {{ intervalSelect.value = currentInterval; }}
      }});

      function showRange(kind) {{
        if (kind === 'all') {{ chart.timeScale().fitContent(); return; }}
        const d = new Date(currentLast * 1000);
        if (kind === '1m') d.setUTCMonth(d.getUTCMonth()-1);
        if (kind === '3m') d.setUTCMonth(d.getUTCMonth()-3);
        if (kind === '6m') d.setUTCMonth(d.getUTCMonth()-6);
        if (kind === '1y') d.setUTCFullYear(d.getUTCFullYear()-1);
        if (kind === '3y') d.setUTCFullYear(d.getUTCFullYear()-3);
        const fromCandidate = Math.floor(d.getTime()/1000);
        const from = Math.max(currentFirst, fromCandidate);
        const to = currentInterval === '1D' ? Math.max(currentLast, projectionLast) : currentLast;
        chart.timeScale().setVisibleRange({{from:from,to:to}});
      }}
      document.querySelectorAll('#tv-root button[data-range]').forEach(b => b.addEventListener('click',()=>{{
        document.querySelectorAll('#tv-root button[data-range]').forEach(x=>x.classList.remove('active'));
        b.classList.add('active');
        showRange(b.dataset.range);
      }}));
      const defaultBtn = document.querySelector('#tv-root button[data-range="all"]');
      if (defaultBtn) defaultBtn.classList.add('active');
      applyInterval('1D');
      showRange('all');

      chart.subscribeCrosshairMove(param => {{
        const tip = document.getElementById('ohlc-tip');
        if (!param || !param.time || !param.seriesData) {{ tip.textContent=''; return; }}
        const d = param.seriesData.get(main);
        if (!d) {{ tip.textContent=''; return; }}
        if (d.open !== undefined) tip.textContent = `O ${{Math.round(d.open).toLocaleString('en-US')}}  H ${{Math.round(d.high).toLocaleString('en-US')}}  L ${{Math.round(d.low).toLocaleString('en-US')}}  C ${{Math.round(d.close).toLocaleString('en-US')}}`;
        else tip.textContent = `Giá ${{Math.round(d.value).toLocaleString('en-US')}}`;
      }});
    }})();
    </script>
    """
    embed_html(html_doc, height)

def trend_text(r: pd.Series) -> str:
    close, ema20, ma50, ma200 = (safe_num(r.get(x)) for x in ["close", "ema20", "ma50", "ma200"])
    if all(pd.notna(x) for x in [close, ema20, ma50, ma200]):
        if close > ema20 > ma50 > ma200:
            return "TĂNG RÕ"
        if close > ema20 and ema20 > ma50:
            return "TĂNG NGẮN HẠN"
        if close < ema20 < ma50:
            return "YẾU"
    return "CHƯA RÕ"


def buy_action_now(r: pd.Series, regime: dict) -> str:
    # Hành động mua ngắn gọn tại thời điểm cập nhật.
    if bool(r.get("price_only", False)):
        return "CHỜ PHÂN TÍCH KỸ THUẬT"
    signal = str(r.get("signal", "") or "").upper().strip()
    score = int(safe_num(r.get("score"), 0))
    confidence = int(safe_num(r.get("confidence"), 0))
    breakdown = bool(r.get("breakdown20", False))
    breakout = bool(r.get("breakout20", False))
    rsi = safe_num(r.get("rsi14"), 50)
    volr = safe_num(r.get("volume_ratio20"), 1)
    close = safe_num(r.get("close"))
    ema20 = safe_num(r.get("ema20"))
    regime_score = int(safe_num(regime.get("score"), 0))

    if breakdown or signal in {"BÁN", "GIẢM TỶ TRỌNG"} or score < 42:
        return "CHƯA NÊN MUA"
    price_ok = pd.notna(close) and pd.notna(ema20) and close >= ema20
    if (
        signal == "MUA MẠNH" and confidence >= 72 and regime_score >= 55
        and rsi < 74 and price_ok and volr >= 0.9
    ):
        return "CÓ THỂ MUA THĂM DÒ NGAY"
    if (
        signal in {"MUA", "MUA MẠNH"} and breakout and confidence >= 70
        and regime_score >= 50 and rsi < 76 and volr >= 1.2
    ):
        return "CÓ THỂ MUA THĂM DÒ NGAY"
    if signal in {"MUA", "MUA MẠNH"}:
        return "CHỜ ĐIỂM MUA TỐT"
    if signal == "THEO DÕI" or score >= 45:
        return "CHỜ XÁC NHẬN"
    return "CHƯA NÊN MUA"


def general_recommendation(r: pd.Series, regime: dict) -> str:
    score = int(safe_num(r.get("score"), 0))
    signal = str(r.get("signal", ""))
    breakdown = bool(r.get("breakdown20", False))
    rsi = safe_num(r.get("rsi14"), 50)
    if breakdown or signal == "BÁN":
        return "TRÁNH MUA / ƯU TIÊN QUẢN TRỊ RỦI RO"
    if signal == "GIẢM TỶ TRỌNG":
        return "CHƯA NÊN MUA, CHỜ ỔN ĐỊNH LẠI"
    if signal == "MUA MẠNH" and regime.get("score", 0) >= 55 and rsi < 75:
        return "CÓ THỂ CÂN NHẮC MUA THĂM DÒ"
    if signal == "MUA":
        return "CÓ THỂ THEO DÕI ĐIỂM MUA, KHÔNG MUA ĐUỔI"
    if signal == "THEO DÕI":
        return "THEO DÕI, CHỜ TÍN HIỆU XÁC NHẬN"
    if score >= 45:
        return "TRUNG LẬP, ƯU TIÊN QUAN SÁT"
    return "RỦI RO CAO, CHƯA PHÙ HỢP ĐỂ MUA"


def short_reason(r: pd.Series) -> str:
    close, ema20, ma50 = safe_num(r.get("close")), safe_num(r.get("ema20")), safe_num(r.get("ma50"))
    rsi, volr = safe_num(r.get("rsi14"), 50), safe_num(r.get("volume_ratio20"), 1)
    bits = []
    if pd.notna(close) and pd.notna(ema20):
        bits.append("giá trên EMA20" if close > ema20 else "giá dưới EMA20")
    if pd.notna(ema20) and pd.notna(ma50):
        bits.append("xu hướng ngắn hạn tốt" if ema20 > ma50 else "xu hướng ngắn hạn còn yếu")
    if rsi >= 75:
        bits.append(f"RSI {rsi:.0f} khá nóng")
    elif rsi >= 52:
        bits.append(f"RSI {rsi:.0f} tích cực")
    elif rsi < 40:
        bits.append(f"RSI {rsi:.0f} yếu")
    if volr >= 1.5:
        bits.append(f"khối lượng {volr:.1f} lần TB20")
    elif volr < 0.7:
        bits.append("khối lượng thấp")
    if bool(r.get("breakout20", False)):
        bits.append("vượt kháng cự 20 phiên")
    if bool(r.get("breakdown20", False)):
        bits.append("thủng hỗ trợ 20 phiên")
    return "; ".join(bits[:4]).capitalize() + "."



def decision_explanation(r: pd.Series, regime: dict, history: pd.DataFrame) -> dict:
    """Diễn giải có cấu trúc: vì sao xu hướng/khuyến nghị và điều kiện chờ điểm mua.

    Các câu này chỉ giải thích bộ quy tắc đang dùng, không phải cam kết dự báo.
    """
    close = safe_num(r.get("close"))
    ema20 = safe_num(r.get("ema20"))
    ma50 = safe_num(r.get("ma50"))
    ma200 = safe_num(r.get("ma200"))
    rsi = safe_num(r.get("rsi14"), 50)
    volr = safe_num(r.get("volume_ratio20"), 1)
    support = safe_num(r.get("support20"))
    resistance = safe_num(r.get("resistance20"))
    score = int(safe_num(r.get("score"), 0))
    confidence = int(safe_num(r.get("confidence"), 0))
    breakout = bool(r.get("breakout20", False))
    breakdown = bool(r.get("breakdown20", False))
    proj = combined_trend_projection(history, horizon=30)
    comps = {str(x.get("label")): x for x in proj.get("components", [])}

    why = []
    if all(pd.notna(x) for x in [close, ema20, ma50, ma200]):
        if close > ema20 > ma50 > ma200:
            why.append("Giá đang nằm trên EMA20, MA50 và MA200 theo đúng thứ tự tăng; cấu trúc xu hướng là tích cực.")
        elif close > ema20 and ema20 > ma50:
            why.append("Giá đang trên EMA20 và EMA20 trên MA50; xu hướng ngắn hạn tích cực nhưng chưa chắc đã đồng thuận với dài hạn.")
        elif close < ema20 < ma50:
            why.append("Giá nằm dưới EMA20 và EMA20 dưới MA50; lực giá ngắn hạn đang yếu.")
        else:
            why.append("Các đường EMA/MA đang đan xen; xu hướng chưa đồng thuận hoàn toàn.")

    if comps:
        bits = []
        for label in ("1T", "3T", "6T"):
            c = comps.get(label)
            if c:
                bits.append(f"{label} {safe_num(c.get('return_pct'), 0):+.1f}% ({c.get('strength','—').lower()})")
        if bits:
            why.append("Diễn biến lịch sử: " + ", ".join(bits) + ".")
    if proj.get("points"):
        direction = "tăng" if proj.get("direction", 0) > 0 else ("giảm" if proj.get("direction", 0) < 0 else "đi ngang")
        why.append(
            f"Đường xu hướng tổng hợp 1T–3T–6T đang nghiêng {direction}, ngoại suy khoảng {safe_num(proj.get('projected_return_pct'),0):+.1f}%/30 phiên; độ rõ {str(proj.get('strength','—')).lower()}."
        )

    if rsi >= 75:
        why.append(f"RSI {rsi:.0f} ở vùng cao: động lượng mạnh nhưng rủi ro mua đuổi tăng.")
    elif rsi >= 52:
        why.append(f"RSI {rsi:.0f} cho thấy động lượng bên mua đang khá tích cực.")
    elif rsi < 40:
        why.append(f"RSI {rsi:.0f} cho thấy động lượng yếu.")
    else:
        why.append(f"RSI {rsi:.0f} ở vùng trung tính, chưa tạo ưu thế rõ.")

    if volr >= 1.5:
        why.append(f"Khối lượng hiện tại khoảng {volr:.1f} lần TB20, cho thấy dòng tiền đang hoạt động mạnh hơn bình thường.")
    elif volr < 0.8:
        why.append(f"Khối lượng chỉ khoảng {volr:.1f} lần TB20, nên tín hiệu giá hiện tại có độ xác nhận thấp hơn.")
    else:
        why.append(f"Khối lượng khoảng {volr:.1f} lần TB20, chưa phải mức bùng nổ.")

    if breakout:
        why.append("Giá đã vượt vùng kháng cự 20 phiên; cần theo dõi khả năng giữ được vùng vừa vượt.")
    if breakdown:
        why.append("Giá đã thủng hỗ trợ 20 phiên; đây là tín hiệu làm giảm mạnh chất lượng điểm mua.")
    if pd.notna(support) and pd.notna(resistance) and pd.notna(close):
        ds = relative_pct(close, support)
        dr = relative_pct(resistance, close)
        why.append(f"Hỗ trợ gần {support:,.0f} (cách giá {ds:+.1f}%), kháng cự gần {resistance:,.0f} (còn khoảng {dr:+.1f}% phía trên).")
    why.append(f"Độ rõ tín hiệu {confidence}/100; điểm kỹ thuật {score}/100; bối cảnh thị trường {int(safe_num(regime.get('score'),0))}/100.")

    entry = []
    # Kịch bản mua theo breakout hoặc pullback, ưu tiên điều kiện xác nhận thay vì một giá duy nhất.
    if pd.notna(resistance):
        entry.append(
            f"Kịch bản vượt cản: chờ giá đóng cửa vượt {resistance:,.0f} và khối lượng tối thiểu khoảng 1,3 lần TB20; tốt hơn nếu RSI nằm 50–70."
        )
    if pd.notna(ema20):
        if pd.notna(close) and close >= ema20:
            entry.append(f"Kịch bản điều chỉnh: giá lùi về quanh EMA20 {ema20:,.0f}, không đóng cửa thủng vùng này và xuất hiện nến hồi phục/khối lượng cải thiện.")
        else:
            entry.append(f"Nếu đang dưới EMA20, ưu tiên chờ giá lấy lại EMA20 {ema20:,.0f} rồi giữ được ít nhất một phiên thay vì mua khi đang yếu.")
    if pd.notna(support):
        entry.append(f"Kịch bản bắt nhịp tại hỗ trợ: chỉ cân nhắc khi vùng {support:,.0f} được giữ, giá tạo đáy cao dần và RSI quay lên trên 45–50.")
    if breakdown:
        entry.insert(0, "Hiện chưa phải điểm mua ưu tiên vì hỗ trợ đã bị phá; cần chờ hình thành nền giá mới trước.")
    elif rsi > 75:
        entry.insert(0, "Không ưu tiên mua đuổi khi RSI đang quá cao; chờ nhịp cân bằng hoặc retest.")

    invalid = []
    if pd.notna(support):
        invalid.append(f"Tránh mở vị thế mới nếu giá đóng cửa thủng {support:,.0f}, đặc biệt khi khối lượng tăng.")
    if pd.notna(ma50):
        invalid.append(f"Nếu giá mất MA50 {ma50:,.0f} và EMA20 tiếp tục dốc xuống, tín hiệu xu hướng sẽ xấu đi.")

    return {
        "why": " ".join(why[:7]),
        "entry": " ".join(entry[:4]),
        "invalid": " ".join(invalid[:2]),
        "projection": proj,
    }


def portfolio_exit_plan(r: pd.Series, portfolio_row: pd.Series) -> dict:
    """Kế hoạch giữ/bán theo dữ liệu kỹ thuật hiện tại của vị thế đang nắm giữ."""
    close = safe_num(r.get("close"))
    avg = safe_num(portfolio_row.get("avg_cost"))
    pnl_pct = safe_num(portfolio_row.get("pnl_pct"))
    support = safe_num(r.get("support20"))
    resistance = safe_num(r.get("resistance20"))
    ema20 = safe_num(r.get("ema20"))
    ma50 = safe_num(r.get("ma50"))
    rsi = safe_num(r.get("rsi14"), 50)
    volr = safe_num(r.get("volume_ratio20"), 1)
    breakdown = bool(r.get("breakdown20", False))
    breakout = bool(r.get("breakout20", False))

    hold = []
    if pd.notna(close) and pd.notna(ema20):
        hold.append(f"Tiếp tục giữ khi giá còn trên EMA20 {ema20:,.0f}" if close >= ema20 else f"Muốn giữ tiếp cần quan sát khả năng lấy lại EMA20 {ema20:,.0f}")
    if pd.notna(support):
        hold.append(f"và chưa đóng cửa thủng hỗ trợ {support:,.0f}")
    if volr >= 1.2:
        hold.append(f"khối lượng đang {volr:.1f} lần TB20")

    cut = []
    if pd.notna(support):
        cut.append(f"Cắt/giảm tỷ trọng nếu giá đóng cửa dưới hỗ trợ {support:,.0f}, nhất là khi KL >1,3 lần TB20.")
    if pd.notna(ma50):
        cut.append(f"Tín hiệu bán mạnh hơn nếu đồng thời mất MA50 {ma50:,.0f} và EMA20 quay xuống.")
    if breakdown:
        cut.insert(0, "Hỗ trợ hiện đã bị phá, vì vậy ưu tiên bảo toàn vốn hơn là bình quân giá xuống.")

    take = []
    if pd.notna(resistance):
        take.append(f"Có thể chốt một phần khi giá tiến tới kháng cự {resistance:,.0f} nhưng không vượt được sau 1–3 phiên.")
        take.append(f"Nếu vượt {resistance:,.0f} với KL ≥1,3–1,5 lần TB20, có thể giữ phần còn lại và dời mốc bảo vệ theo EMA20 thay vì bán hết ngay.")
    if rsi >= 75:
        take.append(f"RSI {rsi:.0f} đã cao; nếu xuất hiện nến đảo chiều kèm KL lớn thì ưu tiên khóa lợi nhuận.")
    if breakout:
        take.append("Mã đang có tín hiệu vượt cản; ưu tiên quan sát khả năng giữ cản cũ trở thành hỗ trợ mới.")

    status = f"Vị thế đang {'lãi' if pnl_pct >= 0 else 'lỗ'} {abs(pnl_pct):.2f}%" if pd.notna(pnl_pct) else "Chưa tính được lãi/lỗ"
    if pd.notna(avg):
        status += f" so với giá vốn {avg:,.0f}."
    else:
        status += "."
    action = str(portfolio_row.get("action_now", "") or "").strip()
    if not action:
        action = "BÁN / GIẢM TỶ TRỌNG" if breakdown else "GIỮ TIẾP"
    return {
        "action": action,
        "status": status,
        "hold": " ".join(hold) + ("." if hold else ""),
        "cut": " ".join(cut[:3]),
        "take": " ".join(take[:3]),
    }


def relative_pct(price, level):
    p, l = safe_num(price), safe_num(level)
    if pd.isna(p) or pd.isna(l) or l == 0:
        return np.nan
    return (p / l - 1) * 100


def readable_analysis(r: pd.Series, regime: dict) -> list[str]:
    close = safe_num(r.get("close"))
    ema20, ma50, ma200 = (safe_num(r.get(x)) for x in ["ema20", "ma50", "ma200"])
    rsi, volr = safe_num(r.get("rsi14"), 50), safe_num(r.get("volume_ratio20"), 1)
    support, resistance = safe_num(r.get("support20")), safe_num(r.get("resistance20"))
    ret20, rs20 = safe_num(r.get("ret_20d"), 0), safe_num(r.get("relative_strength20"), 0)
    out = []
    if pd.notna(close) and pd.notna(ema20):
        d = relative_pct(close, ema20)
        out.append(f"Giá hiện tại {'cao hơn' if d >= 0 else 'thấp hơn'} EMA20 khoảng {abs(d):.1f}%. Đây là cách đơn giản để nhìn lực giá ngắn hạn.")
    if pd.notna(close) and pd.notna(ma50) and pd.notna(ma200):
        d50, d200 = relative_pct(close, ma50), relative_pct(close, ma200)
        out.append(f"So với MA50, giá chênh {d50:+.1f}%; so với MA200 chênh {d200:+.1f}%. Hai mốc này giúp nhận biết xu hướng trung hạn và dài hơn.")
    if rsi >= 75:
        out.append(f"RSI đang ở {rsi:.0f}: lực tăng mạnh nhưng đã khá nóng, vì vậy mua đuổi có rủi ro cao hơn.")
    elif rsi >= 52:
        out.append(f"RSI đang ở {rsi:.0f}: động lượng khá tích cực và chưa ở vùng quá nóng.")
    elif rsi < 40:
        out.append(f"RSI đang ở {rsi:.0f}: động lượng yếu, nên ưu tiên quan sát thay vì vội mua.")
    else:
        out.append(f"RSI đang ở {rsi:.0f}: động lượng trung tính, chưa có ưu thế rõ cho bên mua hoặc bán.")
    if volr >= 1.5:
        out.append(f"Khối lượng bằng {volr:.1f} lần trung bình 20 phiên: dòng tiền/giao dịch đang cao hơn bình thường.")
    elif volr >= 0.8:
        out.append(f"Khối lượng bằng {volr:.1f} lần trung bình 20 phiên: mức giao dịch chưa có đột biến lớn.")
    else:
        out.append(f"Khối lượng chỉ bằng {volr:.1f} lần trung bình 20 phiên: sự quan tâm hiện tại tương đối thấp.")
    if pd.notna(support) and pd.notna(resistance) and pd.notna(close):
        ds, dr = relative_pct(close, support), relative_pct(resistance, close)
        out.append(f"Hỗ trợ gần nhất khoảng {support:,.0f} (giá cao hơn hỗ trợ {ds:.1f}%); kháng cự khoảng {resistance:,.0f} (còn khoảng {dr:.1f}% nếu đi lên tới vùng này).")
    out.append(f"Trong 20 phiên, mã thay đổi {ret20:+.1f}% và mạnh hơn/yếu hơn VN-Index khoảng {rs20:+.1f} điểm %. Bối cảnh thị trường hiện được chấm {regime.get('score', 0)}/100 - {str(regime.get('label', '')).lower()}.")
    return out


def data_basis(r: pd.Series) -> pd.DataFrame:
    close = safe_num(r.get("close"))
    rows = [
        ("Giá mới nhất", fmt_price(close), "Giá dùng để tính tín hiệu ở lần cập nhật hiện tại"),
        ("Thay đổi phiên", fmt_pct(r.get("change_pct")), "So với giá đóng cửa phiên trước"),
        ("So với EMA20", fmt_pct(relative_pct(close, r.get("ema20"))), "Dương thường tốt hơn cho xu hướng ngắn hạn"),
        ("So với MA50", fmt_pct(relative_pct(close, r.get("ma50"))), "Dương nghĩa là giá đang cao hơn trung bình 50 phiên"),
        ("RSI(14)", f"{safe_num(r.get('rsi14'), 50):.1f}", "Đo động lượng; vùng quá cao có thể làm tăng rủi ro mua đuổi"),
        ("Khối lượng / TB20", f"{safe_num(r.get('volume_ratio20'), 0):.2f} lần", "Trên 1 nghĩa là giao dịch cao hơn trung bình 20 phiên"),
        ("Lợi suất 20 phiên", fmt_pct(r.get("ret_20d")), "Mức tăng/giảm của giá trong khoảng 20 phiên"),
        ("Mạnh hơn VN-Index", fmt_pct(r.get("relative_strength20")), "Dương nghĩa là mã làm tốt hơn VN-Index trong cùng giai đoạn"),
        ("Hỗ trợ", fmt_price(r.get("support20")), "Vùng cần quan sát khi giá giảm"),
        ("Kháng cự", fmt_price(r.get("resistance20")), "Vùng cần quan sát khi giá tăng"),
    ]
    return pd.DataFrame(rows, columns=["Chỉ số", "Giá trị", "Hiểu đơn giản"])


def stock_detail_panel(
    ticker: str,
    market: pd.DataFrame,
    histories: dict,
    regime: dict,
    key_prefix: str,
    demo_warning: bool = False,
    portfolio_row: pd.Series | None = None,
    compact: bool = False,
    hourly_df: pd.DataFrame | None = None,
):
    if ticker not in set(market["ticker"].astype(str)) or ticker not in histories:
        st.warning(f"Chưa có dữ liệu biểu đồ cho mã {ticker}. Hãy bấm CẬP NHẬT TOÀN BỘ.")
        return
    row = market.set_index("ticker").loc[ticker]

    decision = decision_explanation(row, regime, histories[ticker])
    portfolio_plan = None
    if portfolio_row is not None:
        rec = str(portfolio_row.get("recommendation", "") or "")
        explanation = str(portfolio_row.get("explanation", "") or "")
        pnl_pct = safe_num(portfolio_row.get("pnl_pct"))
        portfolio_plan = portfolio_exit_plan(row, portfolio_row)
    else:
        rec, explanation = general_recommendation(row, regime), short_reason(row)
        pnl_pct = np.nan

    current_buy_action = buy_action_now(row, regime)

    # Chỉ để các nút điều khiển bên ngoài; giá, điểm, RSI, KL/TB20, khuyến nghị,
    # nguồn dữ liệu, hỗ trợ/kháng cự và xu hướng đều nằm TRONG biểu đồ.
    ctrl1, ctrl2 = st.columns([0.68, 1.42])
    with ctrl1:
        chart_type = st.selectbox("Kiểu biểu đồ", ["Nến", "Đường", "Vùng"], key=f"{key_prefix}_type_v169")
    with ctrl2:
        overlays = st.multiselect(
            "Chỉ báo", ["EMA20", "MA50", "MA200"], default=["EMA20", "MA50", "MA200"],
            key=f"{key_prefix}_overlays_v169",
        )

    date_value = row.get("date")
    date_text = pd.to_datetime(date_value).strftime("%d/%m/%Y") if pd.notna(date_value) else "—"
    exchange = str(row.get("exchange", "") or "").upper().strip()
    exchange = {"HOSE": "HSX", "HSX": "HSX", "HNX": "HNX", "UPCOM": "UPCoM"}.get(exchange, exchange)
    chart_info = {
        "price": None if pd.isna(safe_num(row.get("close"))) else safe_num(row.get("close")),
        "change_pct": safe_num(row.get("change_pct"), 0),
        "sector": str(row.get("sector", "") or ""),
        "exchange": exchange,
        "score": int(safe_num(row.get("score"), 0)),
        "confidence": int(safe_num(row.get("confidence"), 0)),
        "buy_action": current_buy_action,
        "rsi": round(safe_num(row.get("rsi14"), 50), 1),
        "volume_ratio": round(safe_num(row.get("volume_ratio20"), 0), 2),
        "recommendation": rec,
        "explanation": explanation,
        "date_text": date_text,
        "source": str(row.get("price_time", "") or ""),
        "demo_warning": bool(demo_warning),
        "pnl_pct": None if pd.isna(pnl_pct) else float(pnl_pct),
        "pnl_amount": None if portfolio_row is None or pd.isna(safe_num(portfolio_row.get("pnl"))) else float(safe_num(portfolio_row.get("pnl"))),
        "avg_cost": None if portfolio_row is None or pd.isna(safe_num(portfolio_row.get("avg_cost"))) else float(safe_num(portfolio_row.get("avg_cost"))),
        "quantity": None if portfolio_row is None or pd.isna(safe_num(portfolio_row.get("quantity"))) else float(safe_num(portfolio_row.get("quantity"))),
        "market_value": None if portfolio_row is None or pd.isna(safe_num(portfolio_row.get("market_value"))) else float(safe_num(portfolio_row.get("market_value"))),
        "why": decision.get("why", ""),
        "entry": decision.get("entry", ""),
        "invalid": decision.get("invalid", ""),
        "portfolio_plan": portfolio_plan,
    }

    tradingview_lightweight_chart(
        histories[ticker], ticker,
        support=safe_num(row.get("support20")),
        resistance=safe_num(row.get("resistance20")),
        cost_basis=(safe_num(portfolio_row.get("avg_cost")) if portfolio_row is not None else None),
        overlays=overlays,
        chart_type=chart_type,
        height=(1080 if portfolio_row is not None else (880 if compact else 960)),
        info=chart_info,
        hourly_df=hourly_df,
    )

    if not compact:
        with st.expander("Xem diễn giải chi tiết bằng chữ", expanded=False):
            left, right = st.columns([1.2, 1])
            with left:
                for line in readable_analysis(row, regime):
                    st.write("•", line)
            with right:
                st.dataframe(data_basis(row), width="stretch", hide_index=True, height=388)

