from __future__ import annotations

import streamlit as st


MIC_9000_CSS = """
<style>
:root {
    --mic-ink: #212121;
    --mic-black: #17171c;
    --mic-green: #003c33;
    --mic-navy: #071829;
    --mic-canvas: #ffffff;
    --mic-stone: #eeece7;
    --mic-blue: #4c6ee6;
    --mic-blue-bright: #8da4ff;
    --mic-coral: #ff7759;
    --mic-hairline: #d9d9dd;
    --mic-muted: #75758a;
    --mic-soft: #f7f7f5;
}

html, body, [class*="css"] {
    font-family:
        Inter,
        Arial,
        ui-sans-serif,
        system-ui,
        sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: var(--mic-canvas);
    color: var(--mic-ink);
}

[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.92);
    border-bottom: 1px solid var(--mic-hairline);
}

[data-testid="stSidebar"] {
    background: var(--mic-black);
    color: #ffffff;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

[data-testid="stSidebar"] * {
    color: #ffffff;
}

[data-testid="stSidebar"] [data-baseweb="input"] input,
[data-testid="stSidebar"] textarea {
    color: #ffffff !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.16);
}

[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stFormSubmitButton > button {
    background: rgba(255, 255, 255, 0.045) !important;
    color: #f7f8ff !important;
    border: 1px solid rgba(255, 255, 255, 0.22) !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] .stFormSubmitButton > button:hover {
    background: rgba(76, 110, 230, 0.20) !important;
    color: #ffffff !important;
    border-color: rgba(141, 164, 255, 0.92) !important;
}

[data-testid="stSidebar"] .stButton > button:focus,
[data-testid="stSidebar"] .stFormSubmitButton > button:focus {
    color: #ffffff !important;
    border-color: var(--mic-blue-bright) !important;
}

[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stFormSubmitButton > button p {
    color: inherit !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.025);
    border-radius: 12px;
}
.block-container {
    max-width: 1080px;
    padding-top: 2.4rem;
    padding-bottom: 7rem;
}

div[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.75rem;
}

div[data-testid="stChatMessage"]:has(
    div[data-testid="stMarkdownContainer"]
) {
    border: 1px solid #f0f0f0;
}

.stButton > button,
.stFormSubmitButton > button {
    border-radius: 999px;
    border: 1px solid var(--mic-hairline);
    padding: 0.45rem 0.9rem;
    transition: 120ms ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    border-color: var(--mic-blue);
    color: var(--mic-blue);
}

[data-testid="stFileUploader"] {
    border-radius: 12px;
}

details {
    border-radius: 12px !important;
    border-color: #e5e7eb !important;
}

.mic-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 4px 0 18px 0;
}

.mic-brand-title {
    font-size: 20px;
    line-height: 1.1;
    letter-spacing: -0.35px;
    font-weight: 600;
}

.mic-brand-subtitle {
    margin-top: 4px;
    color: #b9b9c4;
    font-size: 12px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.mic-lens {
    position: relative;
    width: 42px;
    height: 42px;
    flex: 0 0 42px;
    border-radius: 999px;
    background:
        radial-gradient(
            circle at 50% 45%,
            #f3f6ff 0%,
            var(--mic-blue-bright) 13%,
            var(--mic-blue) 32%,
            #23346d 52%,
            #11131d 73%,
            #050507 100%
        );
    border: 1px solid rgba(141, 164, 255, 0.78);
    box-shadow:
        0 0 0 6px rgba(76, 110, 230, 0.10),
        0 0 24px rgba(76, 110, 230, 0.42);
}

.mic-lens::after {
    content: "";
    position: absolute;
    inset: 11px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.36);
    filter: blur(4px);
}

.mic-hero {
    max-width: 760px;
    margin: 12vh auto 2rem auto;
    text-align: center;
}

.mic-hero .mic-lens {
    width: 78px;
    height: 78px;
    margin: 0 auto 24px auto;
    box-shadow:
        0 0 0 10px rgba(76, 110, 230, 0.08),
        0 0 42px rgba(76, 110, 230, 0.35);
}

.mic-hero-label {
    color: var(--mic-blue);
    font-size: 12px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

.mic-hero h1 {
    margin: 12px 0 6px 0;
    color: var(--mic-ink);
    font-size: clamp(54px, 8vw, 86px);
    line-height: 0.95;
    letter-spacing: -0.07em;
    font-weight: 500;
}

.mic-hero p {
    margin: 16px 0 0 0;
    color: #616161;
    font-size: 18px;
    line-height: 1.45;
}

.mic-topbar {
    display: flex;
    justify-content: space-between;
    gap: 18px;
    align-items: flex-end;
    padding: 0 0 18px 0;
    margin: 0 0 16px 0;
    border-bottom: 1px solid var(--mic-hairline);
}

.mic-topbar-label {
    color: var(--mic-blue);
    font-size: 12px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.mic-topbar-title {
    margin-top: 5px;
    font-size: 30px;
    letter-spacing: -0.04em;
    line-height: 1.05;
}

.mic-topbar-meta {
    color: var(--mic-muted);
    font-size: 12px;
}

.mic-status-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    padding: 5px 0;
    font-size: 12px;
}

.mic-status-label {
    color: #dddddf;
}

.mic-status-value {
    color: #b8b8c4;
}

.mic-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    margin-right: 8px;
    border-radius: 999px;
    vertical-align: 1px;
}

.mic-dot-green {
    background: #55c78a;
    box-shadow: 0 0 10px rgba(85, 199, 138, 0.55);
}

.mic-dot-amber {
    background: #f2b75f;
    box-shadow: 0 0 10px rgba(242, 183, 95, 0.45);
}

.mic-dot-red {
    background: #e26060;
    box-shadow: 0 0 10px rgba(226, 96, 96, 0.45);
}

.mic-source-label {
    color: var(--mic-muted);
    font-size: 12px;
    line-height: 1.45;
}

.mic-trace {
    font-family:
        ui-monospace,
        SFMono-Regular,
        Menlo,
        Monaco,
        Consolas,
        monospace;
    font-size: 12px;
}

.mic-escalation {
    margin: 8px 0 12px 0;
    padding: 12px 14px;
    border-left: 3px solid var(--mic-coral);
    border-radius: 8px;
    background: #fff6f3;
    color: #6b2f24;
    font-size: 13px;
}

.mic-small-muted {
    color: var(--mic-muted);
    font-size: 12px;
}
</style>
"""


def inject_styles() -> None:
    st.markdown(
        MIC_9000_CSS,
        unsafe_allow_html=True,
    )


def render_brand_lockup() -> None:
    st.markdown(
        """
        <div class="mic-brand">
            <div class="mic-lens"></div>
            <div>
                <div class="mic-brand-title">MIC 9000</div>
                <div class="mic-brand-subtitle">
                    Internal AI Support
                </div>
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
            <div class="mic-hero-label">
                Manufacturing Improvement Yokoten Center
            </div>
            <h1>MIC 9000</h1>
            <p>
                How may I assist you?
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )