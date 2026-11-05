"""Visual theme. System fonts only: no Google Fonts or CDN requests, so the UI makes no network calls."""
import streamlit as st

CSS = """
<style>
:root{
  --bg:#0F1220; --surface:#171B31; --raised:#1F2444; --line:#2A3052; --text:#E6E8F5; --muted:#9AA3C7;
  --violet:#8B7CFF; --blue:#4DA3FF; --good:#4ADE9A; --warn:#F5B84A; --bad:#FF7A90;
  --font:"Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,"Helvetica Neue",Arial,sans-serif;
}
html, body, [class*="css"], .stApp { font-family: var(--font); }
.stApp { background: radial-gradient(1200px 500px at 70% -10%, rgba(139,124,255,.14), transparent 60%), var(--bg); }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1100px; }
h1, h2, h3 { letter-spacing: -0.02em; font-weight: 650; }

/* sidebar */
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
.ps-brand { display:flex; align-items:center; gap:.65rem; margin-bottom:.2rem; }
.ps-brand b { font-size:1.12rem; letter-spacing:-.01em; }
.ps-tag { color: var(--muted); font-size:.82rem; margin-bottom:1rem; }
.ps-side-title { color: var(--muted); font-size:.82rem; margin:1.1rem 0 .35rem; }

/* hero */
.ps-hero { padding: 1.4rem 0 .6rem; }
.ps-hero h1 { font-size: 2.35rem; line-height:1.12; margin:0 0 .6rem; max-width: 760px; }
.ps-hero p { color: var(--muted); font-size:1.05rem; max-width: 640px; margin:0 0 .4rem; }
.ps-page-head h2 { font-size:1.7rem; margin:.2rem 0 .2rem; }
.ps-page-head p { color: var(--muted); margin:0 0 1rem; max-width: 680px; }

/* search box */
[data-testid="stForm"] { border:1px solid var(--line); background: var(--surface); border-radius:18px; padding:1rem 1.1rem .6rem; }
[data-testid="stTextInput"] input { background: var(--bg); border:1px solid var(--line); border-radius:12px;
  font-size:1.08rem; padding:.85rem 1rem; color: var(--text); }
[data-testid="stTextInput"] input:focus { border-color: var(--violet); box-shadow:0 0 0 3px rgba(139,124,255,.25); }
.stButton > button, [data-testid="stFormSubmitButton"] > button, [data-testid="stBaseButton-secondary"] {
  border-radius:10px; border:1px solid var(--line); transition: border-color .15s, background .15s; }
.stButton > button:hover { border-color: var(--violet); }
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
  background: linear-gradient(135deg, #8B7CFF, #4DA3FF); border:0; color:#0F1220; font-weight:650; }
:focus-visible { outline: 2px solid var(--blue) !important; outline-offset: 2px; }

/* cards */
.ps-card { background: var(--surface); border:1px solid var(--line); border-radius:14px; padding:1rem 1.15rem; margin:.7rem 0; }
.ps-card.accent { border-left:3px solid var(--violet); }
.ps-head { display:flex; align-items:center; gap:.5rem; flex-wrap:wrap; margin-bottom:.5rem; }
.ps-doc { font-weight:650; font-size:1rem; }
.ps-text { color: #D5D9EE; line-height:1.62; font-size:.96rem; max-width: 78ch; }
.ps-text mark { background: rgba(139,124,255,.30); color: #fff; padding:0 .12em; border-radius:4px; }
.ps-text mark.shared { background: rgba(77,163,255,.30); }
.ps-chip { display:inline-block; font-size:.76rem; padding:.12rem .55rem; border-radius:999px; border:1px solid var(--line);
  color: var(--muted); background: var(--raised); }
.ps-chip.violet { color:#C9C2FF; border-color:#4B4290; }
.ps-chip.good { color: var(--good); border-color:#25674B; }
.ps-chip.warn { color: var(--warn); border-color:#7A5A1E; }
.ps-chip.bad { color: var(--bad); border-color:#7A2F3E; }
.ps-bar { height:4px; border-radius:4px; background: var(--raised); margin:.6rem 0 .1rem; overflow:hidden; }
.ps-bar > span { display:block; height:100%; background: linear-gradient(90deg, #8B7CFF, #4DA3FF); }
.ps-meta { color: var(--muted); font-size:.8rem; margin-top:.5rem; }
.ps-empty { text-align:center; padding:2.2rem 1rem; border:1px dashed var(--line); border-radius:16px; background: rgba(23,27,49,.6); }
.ps-empty h3 { margin:.2rem 0 .3rem; }
.ps-empty p { color: var(--muted); max-width: 520px; margin:0 auto .8rem; }
.ps-stat { background: var(--surface); border:1px solid var(--line); border-radius:12px; padding:.7rem .9rem; }
.ps-stat b { font-size:1.35rem; display:block; }
.ps-stat span { color: var(--muted); font-size:.82rem; }
.ps-answer { background: linear-gradient(180deg, rgba(139,124,255,.10), rgba(77,163,255,.04)); border:1px solid #3A3F78;
  border-radius:16px; padding:1.1rem 1.3rem; margin:.8rem 0; font-size:1.04rem; line-height:1.65; }
.ps-pair { display:grid; grid-template-columns:1fr 1fr; gap:.8rem; }
@media (max-width: 760px) { .ps-pair { grid-template-columns:1fr; } .ps-hero h1 { font-size:1.8rem; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
"""

BRAND = """
<div class="ps-brand">
<svg width="30" height="30" viewBox="0 0 28 28" aria-hidden="true"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#8B7CFF"/><stop offset="1" stop-color="#4DA3FF"/></linearGradient></defs>
<circle cx="12" cy="12" r="8" fill="none" stroke="url(#g)" stroke-width="2.4"/>
<line x1="18" y1="18" x2="25" y2="25" stroke="url(#g)" stroke-width="2.8" stroke-linecap="round"/>
<circle cx="12" cy="10.5" r="2" fill="#E6E8F5"/><path d="M12 12 L10.6 16 H13.4 Z" fill="#E6E8F5"/></svg>
<b>PrivateSearch AI</b></div>
<div class="ps-tag">Your personal knowledge engine</div>
"""


def inject_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
