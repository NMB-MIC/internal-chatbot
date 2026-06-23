from __future__ import annotations

import html
import streamlit as st


MIC_9000_CSS = """
<style>
:root {
    --mic-ink: #202124;
    --mic-black: #15161b;
    --mic-black-2: #1b1d24;
    --mic-black-3: #222530;
    --mic-green: #003c33;
    --mic-green-2: #064e42;
    --mic-navy: #071829;
    --mic-canvas: #fbfaf7;
    --mic-panel: #ffffff;
    --mic-stone: #eeece7;
    --mic-soft: #f4f3ef;
    --mic-soft-2: #f8f7f4;
    --mic-blue: #4c6ee6;
    --mic-blue-bright: #91a7ff;
    --mic-coral: #ff7759;
    --mic-coral-soft: #fff2ed;
    --mic-hairline: #deddd7;
    --mic-hairline-dark: rgba(255,255,255,0.14);
    --mic-muted: #6d6d78;
    --mic-muted-light: #b9bcc8;
    --mic-success: #45bd7d;
    --mic-amber: #f2b75f;
    --mic-danger: #d95a5a;
    --mic-radius-xl: 24px;
    --mic-radius-lg: 18px;
    --mic-radius-md: 12px;
    --mic-shadow-soft: 0 18px 48px rgba(15, 20, 35, 0.06);
}

html, body, [class*="css"] {
    font-family: Inter, Arial, ui-sans-serif, system-ui, sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 70% 0%, rgba(76,110,230,0.08), transparent 28%),
        linear-gradient(180deg, #ffffff 0%, var(--mic-canvas) 42%, #f6f4ef 100%);
    color: var(--mic-ink);
}

[data-testid="stHeader"] {
    background: rgba(251,250,247,0.86);
    border-bottom: 1px solid rgba(32,33,36,0.08);
    backdrop-filter: blur(12px);
}

.block-container {
    max-width: 1480px;
    padding-top: 2.0rem;
    padding-bottom: 7rem;
}

/* -------------------------------------------------------------------------
   Sidebar: dark, readable, and aggressively protected from white-on-white.
   ------------------------------------------------------------------------- */
[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 20% 0%, rgba(76,110,230,0.18), transparent 28%),
        linear-gradient(180deg, #111216 0%, #17171c 42%, #111216 100%) !important;
    color: #f8f9ff !important;
    border-right: 1px solid rgba(255,255,255,0.10);
}

[data-testid="stSidebar"] > div {
    background: transparent !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #f7f8ff !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.14) !important;
    margin: 1rem 0;
}

/* Sidebar text inputs, selectboxes, textareas, number inputs */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="textarea"],
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="base-input"],
[data-testid="stSidebar"] [data-baseweb="datepicker"] input {
    background-color: #242732 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border-color: rgba(255,255,255,0.20) !important;
    caret-color: #ffffff !important;
}

[data-testid="stSidebar"] input::placeholder,
[data-testid="stSidebar"] textarea::placeholder {
    color: #aeb3c4 !important;
    -webkit-text-fill-color: #aeb3c4 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] svg,
[data-testid="stSidebar"] [data-baseweb="input"] svg {
    color: #f7f8ff !important;
    fill: #f7f8ff !important;
}

/* Streamlit popovers are portal-rendered outside sidebar, so style globally. */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
ul[role="listbox"],
[role="listbox"],
[data-baseweb="menu"] {
    background: #1b1d24 !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    box-shadow: 0 20px 54px rgba(0,0,0,0.34) !important;
}

[role="option"],
[role="option"] span,
[role="listbox"] li,
[role="listbox"] div {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

[role="option"]:hover,
[role="option"][aria-selected="true"],
[role="listbox"] li:hover {
    background: #2b3142 !important;
    color: #ffffff !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stFormSubmitButton > button,
[data-testid="stSidebar"] button[kind="secondary"] {
    background: rgba(255,255,255,0.055) !important;
    color: #f8f9ff !important;
    border: 1px solid rgba(255,255,255,0.22) !important;
    border-radius: 999px !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] .stFormSubmitButton > button:hover,
[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: rgba(76,110,230,0.22) !important;
    color: #ffffff !important;
    border-color: rgba(145,167,255,0.92) !important;
}

[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stFormSubmitButton > button p {
    color: inherit !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255,255,255,0.035) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 16px !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] details,
[data-testid="stSidebar"] details {
    border-color: rgba(255,255,255,0.14) !important;
    background: rgba(255,255,255,0.02) !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px dashed rgba(255,255,255,0.24) !important;
    border-radius: 16px !important;
}

[data-testid="stSidebar"] [aria-disabled="true"],
[data-testid="stSidebar"] [disabled] {
    opacity: 1 !important;
    color: #9aa1b8 !important;
    -webkit-text-fill-color: #9aa1b8 !important;
}

/* Global controls */
.stButton > button,
.stFormSubmitButton > button,
button[kind="secondary"] {
    border-radius: 999px !important;
    border: 1px solid var(--mic-hairline) !important;
    transition: 120ms ease;
    box-shadow: none !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover,
button[kind="secondary"]:hover {
    border-color: var(--mic-blue) !important;
    color: var(--mic-blue) !important;
}

button[kind="primary"],
.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
    background: var(--mic-black) !important;
    color: #ffffff !important;
    border-color: var(--mic-black) !important;
}

[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.78);
    border: 1px solid rgba(32,33,36,0.08);
    border-radius: 20px;
    padding: 0.7rem 0.95rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 12px 28px rgba(15,20,35,0.035);
}

[data-testid="stChatInput"] textarea {
    border-radius: 18px !important;
}

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid var(--mic-hairline);
    border-radius: 16px;
    padding: 12px 14px;
    box-shadow: 0 10px 28px rgba(15,20,35,0.035);
}

/* Commercial MIC components */
.mic-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 4px 0 18px 0;
}

.mic-brand-title {
    color: #ffffff;
    font-size: 21px;
    line-height: 1.1;
    letter-spacing: -0.04em;
    font-weight: 700;
}

.mic-brand-subtitle {
    margin-top: 4px;
    color: #b9becf;
    font-size: 11px;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

.mic-lens {
    position: relative;
    width: 42px;
    height: 42px;
    flex: 0 0 42px;
    border-radius: 999px;
    background:
        radial-gradient(circle at 50% 42%, #f5f7ff 0%, #91a7ff 14%, #4c6ee6 34%, #26366e 55%, #11131d 76%, #050507 100%);
    border: 1px solid rgba(145,167,255,0.78);
    box-shadow: 0 0 0 6px rgba(76,110,230,0.10), 0 0 24px rgba(76,110,230,0.42);
}

.mic-lens::after {
    content: "";
    position: absolute;
    inset: 11px;
    border-radius: 999px;
    background: rgba(255,255,255,0.36);
    filter: blur(4px);
}

.mic-hero {
    max-width: 780px;
    margin: 8vh auto 2rem auto;
    text-align: center;
}

.mic-hero .mic-lens {
    width: 78px;
    height: 78px;
    margin: 0 auto 24px auto;
    box-shadow: 0 0 0 10px rgba(76,110,230,0.08), 0 0 42px rgba(76,110,230,0.35);
}

.mic-hero-label,
.mic-eyebrow {
    color: var(--mic-blue);
    font-size: 12px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    font-weight: 700;
}

.mic-hero h1 {
    margin: 12px 0 6px 0;
    color: var(--mic-ink);
    font-size: clamp(54px, 8vw, 88px);
    line-height: 0.95;
    letter-spacing: -0.075em;
    font-weight: 600;
}

.mic-hero p {
    margin: 16px 0 0 0;
    color: #61616f;
    font-size: 18px;
    line-height: 1.5;
}

.mic-app-header {
    display: grid;
    grid-template-columns: 1.3fr 1fr;
    gap: 20px;
    align-items: stretch;
    padding: 22px 24px;
    margin-bottom: 20px;
    background: rgba(255,255,255,0.78);
    border: 1px solid rgba(32,33,36,0.08);
    border-radius: 28px;
    box-shadow: var(--mic-shadow-soft);
    backdrop-filter: blur(10px);
}

.mic-app-title {
    margin: 6px 0 6px 0;
    font-size: clamp(34px, 4vw, 54px);
    letter-spacing: -0.07em;
    line-height: 0.95;
    font-weight: 700;
}

.mic-app-copy {
    margin: 0;
    color: var(--mic-muted);
    line-height: 1.45;
    max-width: 760px;
}

.mic-status-strip,
.mic-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
}

.mic-chip {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 10px;
    border-radius: 999px;
    background: #ffffff;
    border: 1px solid var(--mic-hairline);
    color: #2d2d35;
    font-size: 12px;
    font-weight: 650;
}

.mic-chip-dark {
    background: #181a21;
    border-color: #181a21;
    color: #ffffff;
}

.mic-chip-green {
    background: #ecfff5;
    border-color: #b8ecd1;
    color: #17653d;
}

.mic-chip-amber {
    background: #fff7e8;
    border-color: #f4d9a4;
    color: #7b4c08;
}

.mic-chip-coral {
    background: var(--mic-coral-soft);
    border-color: #ffc8ba;
    color: #7e2b1b;
}

.mic-panel {
    padding: 17px 18px;
    margin-bottom: 14px;
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(32,33,36,0.08);
    border-radius: 22px;
    box-shadow: 0 12px 30px rgba(15,20,35,0.04);
}

.mic-panel-dark {
    background: linear-gradient(180deg, #071829 0%, #05231f 100%);
    border: 1px solid rgba(255,255,255,0.10);
    color: #ffffff;
}

.mic-panel-title {
    margin: 0 0 10px 0;
    color: inherit;
    font-size: 14px;
    font-weight: 800;
    letter-spacing: 0.02em;
}

.mic-kv {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 7px 0;
    border-bottom: 1px solid rgba(125,125,135,0.12);
    font-size: 13px;
}

.mic-kv:last-child { border-bottom: none; }
.mic-kv span { color: var(--mic-muted); }
.mic-kv strong { color: var(--mic-ink); text-align: right; }
.mic-panel-dark .mic-kv { border-bottom-color: rgba(255,255,255,0.12); }
.mic-panel-dark .mic-kv span { color: #bfc7d8; }
.mic-panel-dark .mic-kv strong { color: #ffffff; }

.mic-source-label {
    color: var(--mic-muted);
    font-size: 12px;
    line-height: 1.45;
}

.mic-source-card {
    padding: 12px 13px;
    margin: 8px 0;
    border: 1px solid var(--mic-hairline);
    border-radius: 14px;
    background: #ffffff;
}

.mic-trace {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 12px;
}

.mic-escalation {
    margin: 8px 0 12px 0;
    padding: 12px 14px;
    border-left: 3px solid var(--mic-coral);
    border-radius: 10px;
    background: var(--mic-coral-soft);
    color: #6b2f24;
    font-size: 13px;
}

.mic-small-muted {
    color: var(--mic-muted);
    font-size: 12px;
}

.mic-session-pill {
    display: inline-flex;
    max-width: 100%;
    padding: 4px 8px;
    border-radius: 999px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    color: #c7cada;
    font-size: 11px;
}

.mic-sidebar-section {
    margin: 12px 0 8px 0;
    color: #939aae !important;
    font-size: 11px;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    font-weight: 800;
}

.mic-session-row {
    padding: 9px 10px;
    margin: 6px 0;
    border-radius: 12px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
}

.mic-session-active {
    background: rgba(76,110,230,0.18);
    border-color: rgba(145,167,255,0.45);
}

.mic-session-title {
    color: #ffffff;
    font-size: 13px;
    font-weight: 700;
    line-height: 1.3;
}

.mic-session-meta {
    color: #aeb3c4;
    font-size: 11px;
    margin-top: 4px;
}

.mic-warning-box {
    padding: 12px 14px;
    border-radius: 14px;
    background: #fff5ee;
    border: 1px solid #ffd2c2;
    color: #7e2b1b;
    font-size: 13px;
}

@media (max-width: 1100px) {
    .mic-app-header { grid-template-columns: 1fr; }
    .block-container { padding-left: 1.1rem; padding-right: 1.1rem; }
}
</style>
"""


def inject_styles() -> None:
    st.markdown(MIC_9000_CSS, unsafe_allow_html=True)


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def render_brand_lockup() -> None:
    st.markdown(
        """
        <div class="mic-brand">
            <div class="mic-lens"></div>
            <div>
                <div class="mic-brand-title">MIC 9000</div>
                <div class="mic-brand-subtitle">Internal AI Support</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <section class="mic-hero">
            <div class="mic-lens"></div>
            <div class="mic-hero-label">Manufacturing Improvement Yokoten Center</div>
            <h1>MIC 9000</h1>
            <p>Ask a question, select a document, or open a previous session from the sidebar.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def chip(label: str, kind: str = "") -> str:
    cls = "mic-chip" + (f" mic-chip-{kind}" if kind else "")
    return f'<span class="{cls}">{esc(label)}</span>'


def sidebar_section(label: str) -> None:
    st.markdown(f'<div class="mic-sidebar-section">{esc(label)}</div>', unsafe_allow_html=True)
