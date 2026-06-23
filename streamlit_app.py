from __future__ import annotations

import hmac
import os
from typing import Any

import pandas as pd
import streamlit as st

from app.config import settings
from app.memory.models import ChatMessage, RetrievalRunLog, RetrievalSourceLog
from app.services.runtime import BackendBundle, build_backend, check_runtime_health
from app.services.runtime_diagnostics import build_runtime_diagnostics_snapshot
from app.ui.styles import chip, esc, inject_styles, render_brand_lockup, render_empty_state, sidebar_section


st.set_page_config(
    page_title=settings.app_name,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()

LOCAL_UI_USER_ID = "local-user"
SCOPE_LABELS = {
    "auto": "Auto routing",
    "prefer_selected": "Prefer selected",
    "strict_selected": "Strict selected",
}
SCOPE_VALUES = {value: key for key, value in SCOPE_LABELS.items()}


@st.cache_resource(show_spinner=False)
def get_backend() -> BackendBundle:
    return build_backend(warm_embedding=True)


@st.cache_data(ttl=15, show_spinner=False)
def get_health_snapshot() -> dict[str, Any]:
    return check_runtime_health(get_backend()).to_dict()


@st.cache_data(ttl=10, show_spinner=False)
def get_runtime_diagnostics_snapshot(selected_document: str | None) -> dict[str, Any]:
    return build_runtime_diagnostics_snapshot(
        get_backend(),
        selected_document=selected_document,
    ).to_dict()


backend = get_backend()
memory = backend.memory


def bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def short_id(value: object, size: int = 10) -> str:
    text = str(value or "")
    return text if len(text) <= size else text[:size] + "…"


def derive_session_title(text: str) -> str:
    compact = " ".join(text.strip().split())
    max_chars = int(getattr(settings, "ui_session_title_max_chars", 48))
    if len(compact) <= max_chars:
        return compact or "New chat"
    return compact[: max_chars - 1].rstrip() + "…"


def list_ui_sessions(search: str = ""):
    limit = int(getattr(settings, "ui_session_list_limit", 30))
    candidate_limit = max(limit * 5, 100)
    search_norm = search.strip().lower()
    sessions = [
        session
        for session in memory.list_sessions(limit=candidate_limit)
        if session.user_id == LOCAL_UI_USER_ID
    ]
    if search_norm:
        sessions = [
            session
            for session in sessions
            if search_norm in (session.title or "").lower()
            or search_norm in session.session_id.lower()
        ]
    return sessions[:limit]


def ensure_active_session() -> str:
    active_session_id = st.session_state.get("active_session_id")
    if active_session_id:
        session = memory.get_session(str(active_session_id))
        if session is not None and session.user_id == LOCAL_UI_USER_ID:
            return str(active_session_id)

    sessions = list_ui_sessions()
    if sessions:
        active_session_id = sessions[0].session_id
    else:
        active_session_id = memory.create_session(
            user_id=LOCAL_UI_USER_ID,
            title="New chat",
        ).session_id

    st.session_state["active_session_id"] = active_session_id
    return str(active_session_id)


def create_new_session() -> None:
    session = memory.create_session(user_id=LOCAL_UI_USER_ID, title="New chat")
    st.session_state["active_session_id"] = session.session_id
    st.session_state["pending_prompt"] = None
    st.session_state["last_graph_result"] = None


def select_session(session_id: str) -> None:
    st.session_state["active_session_id"] = session_id
    st.session_state["pending_prompt"] = None
    st.session_state["last_graph_result"] = None


def init_state() -> None:
    st.session_state.setdefault("active_document_scope", None)
    st.session_state.setdefault(
        "document_behavior",
        getattr(settings, "document_scope_default_behavior", "prefer_selected"),
    )
    st.session_state.setdefault("ui_show_sources", True)
    st.session_state.setdefault("ui_show_trace", False)
    st.session_state.setdefault("ui_show_inspector", True)
    st.session_state.setdefault("access_level", "user")
    st.session_state.setdefault("session_search", "")
    st.session_state.setdefault("pending_prompt", None)


init_state()


def security_enabled() -> bool:
    return bool_env("MIC_SECURITY_ENABLED", default=False)


def admin_actions_enabled() -> bool:
    return bool_env("MIC_ADMIN_ACTIONS_ENABLED", default=False)


def display_unlock_hints() -> bool:
    return bool_env("MIC_DISPLAY_UNLOCK_HINTS", default=False)


def current_access_level() -> str:
    if not security_enabled():
        return st.session_state.get("access_level", "developer")
    return st.session_state.get("access_level", "user")


def is_developer_access() -> bool:
    return current_access_level() in {"developer", "admin"}


def is_admin_access() -> bool:
    return current_access_level() == "admin"


def unlock_access(role: str, token: str) -> bool:
    expected = os.getenv("MIC_ADMIN_TOKEN" if role == "admin" else "MIC_DEVELOPER_TOKEN", "")
    if expected and hmac.compare_digest(token.strip(), expected):
        st.session_state["access_level"] = role
        return True
    return False


def render_security_panel() -> None:
    access = current_access_level()
    st.markdown(
        f"""
        <div class="mic-panel mic-panel-dark">
            <div class="mic-panel-title">Security</div>
            <div class="mic-kv"><span>Access</span><strong>{esc(access)}</strong></div>
            <div class="mic-kv"><span>Security</span><strong>{esc(str(security_enabled()))}</strong></div>
            <div class="mic-kv"><span>Admin actions</span><strong>{esc(str(admin_actions_enabled()))}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not security_enabled():
        st.caption("Security is disabled. Developer panels are available in local mode.")
        return

    with st.form("security_unlock_form"):
        role = st.selectbox("Unlock role", options=["developer", "admin"])
        token = st.text_input("Access token", type="password")
        submitted = st.form_submit_button("Unlock", use_container_width=True)

    if submitted:
        if unlock_access(role, token):
            st.success(f"Unlocked {role} access.")
            st.rerun()
        else:
            st.error("Invalid token.")

    if st.button("Lock session", use_container_width=True):
        st.session_state["access_level"] = "user"
        st.session_state["ui_show_trace"] = False
        st.rerun()

    if display_unlock_hints():
        st.caption("Hint display is enabled. Disable MIC_DISPLAY_UNLOCK_HINTS in production.")


def render_status_row(*, label: str, ok: bool, value: str, amber: bool = False) -> None:
    dot_class = "mic-dot-amber" if amber else ("mic-dot-green" if ok else "mic-dot-red")
    st.markdown(
        f"""
        <div class="mic-kv">
            <span><span class="mic-dot {dot_class}"></span>{esc(label)}</span>
            <strong>{esc(value)}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_health_panel(health: dict[str, Any]) -> None:
    point_count = health.get("indexed_point_count")
    render_status_row(
        label="Ollama",
        ok=bool(health.get("ollama_ok")),
        value="online" if health.get("ollama_ok") else "offline",
    )
    render_status_row(
        label="Model",
        ok=bool(health.get("configured_model_available")),
        value=str(getattr(settings, "ollama_model", "unknown")),
    )
    render_status_row(
        label="Qdrant",
        ok=bool(health.get("qdrant_ok")),
        value="online" if health.get("qdrant_ok") else "offline",
    )
    render_status_row(
        label="Index",
        ok=bool(health.get("collection_exists")),
        amber=bool(health.get("qdrant_ok") and not health.get("collection_exists")),
        value=f"{point_count} points" if point_count is not None else "not indexed",
    )
    render_status_row(
        label="SQLite",
        ok=bool(health.get("sqlite_ok")),
        value="WAL ready" if health.get("sqlite_ok") else "unavailable",
    )
    warmup = health.get("embedding_warmup_seconds")
    render_status_row(
        label="BGE-M3",
        ok=bool(health.get("embedding_ok")),
        value=f"{warmup:.2f}s warm-up" if isinstance(warmup, (int, float)) else "ready",
    )


def active_diagnostics() -> dict[str, Any]:
    return get_runtime_diagnostics_snapshot(st.session_state.get("active_document_scope"))


def render_app_header(active_session: Any, diagnostics: dict[str, Any]) -> None:
    latest_manifest = diagnostics.get("latest_manifest") or {}
    qdrant = diagnostics.get("qdrant") or {}
    kb = diagnostics.get("knowledge_base") or {}
    consistency = diagnostics.get("consistency") or {}
    title = active_session.title if active_session and active_session.title else "New chat"
    selected_doc = st.session_state.get("active_document_scope") or "Auto routing"
    behavior = st.session_state.get("document_behavior", "auto")
    chips = "".join(
        [
            chip(str((latest_manifest.get("mode") or os.getenv("MIC_INDEX_MODE") or "local")), "dark"),
            chip(f"Qdrant {qdrant.get('point_count', '—')} pts", "green" if qdrant.get("ok") else "coral"),
            chip(f"Docs {kb.get('document_count', '—')}", ""),
            chip("Index matched" if consistency.get("active_index_matches_latest_manifest") else "Index mismatch", "green" if consistency.get("active_index_matches_latest_manifest") else "coral"),
            chip(f"Access {current_access_level()}", "dark"),
        ]
    )
    st.markdown(
        f"""
        <section class="mic-app-header">
            <div>
                <div class="mic-eyebrow">MIC 9000 · Internal AI Support</div>
                <div class="mic-app-title">{esc(title)}</div>
                <p class="mic-app-copy">
                    Grounded internal Q&A with SQLite memory, Qdrant retrieval, selected-document scope, source inspection, and production diagnostics.
                </p>
                <div class="mic-chip-row" style="margin-top: 14px;">{chips}</div>
            </div>
            <div class="mic-panel mic-panel-dark" style="margin:0;">
                <div class="mic-panel-title">Current workspace</div>
                <div class="mic-kv"><span>Session</span><strong>{esc(short_id(active_session.session_id if active_session else '', 18))}</strong></div>
                <div class="mic-kv"><span>Document</span><strong>{esc(selected_doc)}</strong></div>
                <div class="mic-kv"><span>Scope</span><strong>{esc(SCOPE_LABELS.get(behavior, behavior))}</strong></div>
                <div class="mic-kv"><span>Sources</span><strong>{esc(str(st.session_state.get('ui_show_sources', True)))}</strong></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_session_row(session: Any, active_session_id: str) -> None:
    selected = session.session_id == active_session_id
    cls = "mic-session-row mic-session-active" if selected else "mic-session-row"
    title = session.title or "Untitled chat"
    meta = f"{short_id(session.session_id, 8)} · {session.updated_at[:19] if session.updated_at else ''}"
    st.markdown(
        f"""
        <div class="{cls}">
            <div class="mic-session-title">{esc(title)}</div>
            <div class="mic-session-meta">{esc(meta)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button(
        "Open" if not selected else "Current",
        key=f"session_select_{session.session_id}",
        use_container_width=True,
        disabled=selected,
    ):
        select_session(session.session_id)
        st.rerun()


def render_sidebar(active_session_id: str) -> tuple[bool, dict[str, Any]]:
    with st.sidebar:
        render_brand_lockup()
        if st.button("＋ New chat", use_container_width=True, type="primary"):
            create_new_session()
            st.rerun()

        sidebar_section("Conversations")
        st.text_input("Search conversations", key="session_search", placeholder="Search title or session id")
        for session in list_ui_sessions(st.session_state.get("session_search", "")):
            render_session_row(session, active_session_id)

        with st.expander("Session controls", expanded=False):
            active_session = memory.get_session(active_session_id)
            current_title = active_session.title if active_session and active_session.title else "Untitled chat"
            with st.form("rename_session_form"):
                new_title = st.text_input("Rename conversation", value=current_title)
                rename_submitted = st.form_submit_button("Rename", use_container_width=True)
            if rename_submitted:
                memory.rename_session(session_id=active_session_id, title=new_title)
                st.rerun()
            st.markdown("<div class='mic-warning-box'>Delete only removes this local SQLite chat session.</div>", unsafe_allow_html=True)
            if st.button("Delete conversation", key="delete_active_session", use_container_width=True):
                memory.delete_session(active_session_id)
                st.session_state.pop("active_session_id", None)
                st.rerun()

        sidebar_section("Document control")
        available_documents = backend.knowledge_base.list_document_paths()
        document_options = ["Auto · no selected document"] + available_documents
        current_scope = st.session_state.get("active_document_scope")
        current_index = document_options.index(current_scope) if current_scope in available_documents else 0
        selected_scope = st.selectbox("Reference document", options=document_options, index=current_index)
        st.session_state["active_document_scope"] = None if selected_scope.startswith("Auto ·") else selected_scope

        behavior_options = [SCOPE_LABELS["auto"], SCOPE_LABELS["prefer_selected"], SCOPE_LABELS["strict_selected"]]
        current_behavior = st.session_state.get("document_behavior", "prefer_selected")
        if current_behavior not in SCOPE_LABELS:
            current_behavior = "prefer_selected"
        behavior_label = st.radio(
            "Scope behavior",
            options=behavior_options,
            index=behavior_options.index(SCOPE_LABELS[current_behavior]),
            disabled=st.session_state.get("active_document_scope") is None,
        )
        st.session_state["document_behavior"] = SCOPE_VALUES.get(behavior_label, "auto") if st.session_state.get("active_document_scope") else "auto"

        st.toggle("Show sources", key="ui_show_sources")

        sidebar_section("Security")
        render_security_panel()
        developer_access = is_developer_access()
        if developer_access:
            st.toggle("Show retrieval trace", key="ui_show_trace")
            st.toggle("Show right inspector", key="ui_show_inspector")
        else:
            st.session_state["ui_show_trace"] = False
            st.session_state["ui_show_inspector"] = True

        sidebar_section("System health")
        health = get_health_snapshot()
        render_health_panel(health)
        if st.button("Refresh health", use_container_width=True):
            get_health_snapshot.clear()
            get_runtime_diagnostics_snapshot.clear()
            st.rerun()

        if developer_access:
            sidebar_section("Developer tools")
            with st.expander("Diagnostics", expanded=False):
                diagnostics = active_diagnostics()
                render_runtime_summary(diagnostics)
                if st.button("Refresh diagnostics", key="refresh_diag_sidebar", use_container_width=True):
                    get_runtime_diagnostics_snapshot.clear()
                    st.rerun()

            with st.expander("Knowledge base operations", expanded=False):
                if not (is_admin_access() and admin_actions_enabled()):
                    st.info("Index-changing actions require admin access and MIC_ADMIN_ACTIONS_ENABLED=true.")
                render_kb_operations(enabled=is_admin_access() and admin_actions_enabled())

    return developer_access, health


def render_kb_operations(*, enabled: bool) -> None:
    category = st.selectbox(
        "Document category",
        options=["uncategorized", "company_info", "admin_support", "developer_support", "kafka_iot_support"],
        disabled=not enabled,
    )
    uploaded_files = st.file_uploader(
        "Stage documents",
        type=[ext.lstrip(".") for ext in backend.knowledge_base.supported_extensions],
        accept_multiple_files=True,
        max_upload_size=getattr(settings, "kb_upload_max_mb", 50),
        disabled=not enabled,
    )
    if st.button("Stage selected documents", use_container_width=True, disabled=(not enabled or not uploaded_files)):
        try:
            staged = backend.knowledge_base.stage_uploaded_files(uploaded_files, category=category)
            st.success(f"Staged {len(staged)} file(s).")
        except Exception as exc:
            st.error(str(exc))
    staged_files = backend.knowledge_base.list_staged_files()
    st.caption(f"Pending staged files: {len(staged_files)}")
    if staged_files:
        st.dataframe(pd.DataFrame([item.to_dict() for item in staged_files]), use_container_width=True, hide_index=True)
    if st.button("Clear staging area", use_container_width=True, disabled=(not enabled or not staged_files)):
        cleared = backend.knowledge_base.clear_staging()
        st.success(f"Cleared {cleared} file(s).")
        st.rerun()
    if st.button("Approve staged files and rebuild", type="primary", use_container_width=True, disabled=not enabled):
        try:
            with st.status("Rebuilding knowledge base...", expanded=True) as status:
                st.write("Promoting staged documents")
                st.write("Chunking documents")
                st.write("Embedding chunks")
                st.write("Rebuilding Qdrant collection")
                report = backend.knowledge_base.rebuild(promote_staged=True)
                status.update(label="Knowledge base rebuilt", state="complete")
            st.session_state["latest_rebuild_report"] = report.to_dict()
            get_health_snapshot.clear()
            get_runtime_diagnostics_snapshot.clear()
            st.success("Knowledge base rebuild completed.")
        except Exception as exc:
            st.error(repr(exc))
    if st.session_state.get("latest_rebuild_report"):
        with st.expander("Latest rebuild report", expanded=False):
            st.json(st.session_state["latest_rebuild_report"])


def retrieval_maps(session_id: str) -> tuple[dict[int, RetrievalRunLog], dict[int, list[RetrievalSourceLog]], list[RetrievalRunLog]]:
    runs = memory.list_retrieval_runs(session_id)
    run_by_assistant_id = {run.assistant_message_id: run for run in runs}
    sources_by_run_id = {run.retrieval_run_id: memory.list_retrieval_sources(run.retrieval_run_id) for run in runs}
    return run_by_assistant_id, sources_by_run_id, runs


def source_location(source: RetrievalSourceLog) -> str:
    parts: list[str] = []
    if source.page_number is not None:
        parts.append(f"page {source.page_number}")
    if source.sheet_name:
        parts.append(f"sheet {source.sheet_name}")
    if source.row_start is not None and source.row_end is not None:
        parts.append(f"rows {source.row_start}-{source.row_end}")
    return "; ".join(parts) or "document"


def render_sources(run: RetrievalRunLog, sources: list[RetrievalSourceLog], developer_mode: bool) -> None:
    if not st.session_state.get("ui_show_sources", True):
        return
    visible_sources = sources if developer_mode else [source for source in sources if source.cited]
    if not visible_sources:
        return
    with st.expander(f"Sources ({len(visible_sources)})", expanded=False):
        table_rows = [
            {
                "id": source.source_id or "retrieved",
                "document": source.source_path,
                "location": source_location(source),
                "score": round(float(source.score), 4),
                "accepted": source.accepted,
                "cited": source.cited,
            }
            for source in visible_sources
        ]
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        for source in visible_sources:
            st.markdown(
                f"""
                <div class="mic-source-card">
                    <strong>[{esc(source.source_id or 'retrieved')}] {esc(source.source_path)}</strong><br />
                    <span class="mic-source-label">{esc(source_location(source))} · score={source.score:.4f} · accepted={esc(source.accepted)} · cited={esc(source.cited)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.code(source.text_preview, language=None)


def render_developer_trace(message: ChatMessage, run: RetrievalRunLog | None) -> None:
    if not (is_developer_access() and st.session_state.get("ui_show_trace", False)):
        return
    with st.expander("Developer trace", expanded=False):
        st.json(
            {
                "message_id": message.message_id,
                "metadata": message.metadata,
                "retrieval_run": run.to_dict() if run else None,
            }
        )


def render_conversation(session_id: str, developer_mode: bool) -> None:
    messages = memory.load_all_messages(session_id)
    run_by_assistant_id, sources_by_run_id, _runs = retrieval_maps(session_id)
    for message in messages:
        role = "assistant" if message.role == "assistant" else "user"
        avatar = "🤖" if role == "assistant" else ":material/person:"
        with st.chat_message(role, avatar=avatar):
            st.markdown(message.content)
            if role == "assistant" and message.metadata.get("escalation_recommended"):
                st.markdown(
                    '<div class="mic-escalation">Human support escalation is recommended.</div>',
                    unsafe_allow_html=True,
                )
            run = run_by_assistant_id.get(message.message_id)
            if run is not None:
                render_sources(run, sources_by_run_id.get(run.retrieval_run_id, []), developer_mode)
            render_developer_trace(message, run)


def render_prompt_suggestions() -> None:
    suggestions = [
        "What can you do?",
        "What is the pipeline raw Kafka topic?",
        "How does the machine-status runbook work?",
        "What is Apiwit's current company?",
    ]
    columns = st.columns(2)
    for index, suggestion in enumerate(suggestions):
        with columns[index % 2]:
            if st.button(suggestion, key=f"suggestion_{index}", use_container_width=True):
                st.session_state["pending_prompt"] = suggestion
                st.rerun()


def latest_run_and_sources(session_id: str) -> tuple[RetrievalRunLog | None, list[RetrievalSourceLog]]:
    _run_by_message, sources_by_run_id, runs = retrieval_maps(session_id)
    if not runs:
        return None, []
    run = runs[-1]
    return run, sources_by_run_id.get(run.retrieval_run_id, [])


def render_source_inspector(session_id: str, developer_mode: bool) -> None:
    run, sources = latest_run_and_sources(session_id)
    if not run:
        st.info("No retrieval run yet for this session.")
        return
    visible_sources = sources if developer_mode else [source for source in sources if source.cited]
    st.markdown('<div class="mic-panel-title">Latest sources</div>', unsafe_allow_html=True)
    if not visible_sources:
        st.caption("No visible cited sources for the latest answer.")
        return
    rows = [
        {
            "rank": source.source_rank,
            "id": source.source_id or "retrieved",
            "document": source.source_path,
            "score": round(float(source.score), 4),
            "accepted": source.accepted,
            "cited": source.cited,
        }
        for source in visible_sources
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    selected_idx = st.selectbox(
        "Preview source",
        options=list(range(len(visible_sources))),
        format_func=lambda i: f"{visible_sources[i].source_id or 'retrieved'} · {visible_sources[i].source_path}",
        key="source_preview_select",
    )
    selected = visible_sources[selected_idx]
    st.code(selected.text_preview, language=None)


def render_trace_inspector(session_id: str) -> None:
    if not is_developer_access():
        st.info("Unlock developer access to view retrieval trace.")
        return
    run, _sources = latest_run_and_sources(session_id)
    if not run:
        st.info("No retrieval trace yet.")
        return
    st.markdown('<div class="mic-panel-title">Retrieval trace</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw hits", run.raw_hit_count)
    c2.metric("Accepted", run.accepted_hit_count)
    c3.metric("Threshold", run.similarity_threshold)
    st.json(run.to_dict())


def render_runtime_summary(diagnostics: dict[str, Any]) -> None:
    latest = diagnostics.get("latest_manifest") or {}
    qdrant = diagnostics.get("qdrant") or {}
    kb = diagnostics.get("knowledge_base") or {}
    consistency = diagnostics.get("consistency") or {}
    st.markdown(
        f"""
        <div class="mic-panel mic-panel-dark">
            <div class="mic-panel-title">Runtime Inspector</div>
            <div class="mic-kv"><span>Qdrant</span><strong>{esc('OK' if qdrant.get('ok') else 'Issue')}</strong></div>
            <div class="mic-kv"><span>Active points</span><strong>{esc(qdrant.get('point_count', 'unknown'))}</strong></div>
            <div class="mic-kv"><span>Manifest</span><strong>{esc(short_id(latest.get('rebuild_id'), 20))}</strong></div>
            <div class="mic-kv"><span>Mode</span><strong>{esc(latest.get('mode', 'unknown'))}</strong></div>
            <div class="mic-kv"><span>Index matched</span><strong>{esc(consistency.get('active_index_matches_latest_manifest'))}</strong></div>
            <div class="mic-kv"><span>Documents</span><strong>{esc(kb.get('document_count', 'unknown'))}</strong></div>
            <div class="mic-kv"><span>Safety</span><strong>{esc(latest.get('accepted_files', '—'))} accepted · {esc(latest.get('quarantined_files', '—'))} quarantined · {esc(latest.get('rejected_files', '—'))} rejected</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_runtime_diagnostics(diagnostics: dict[str, Any]) -> None:
    render_runtime_summary(diagnostics)
    if is_developer_access():
        with st.expander("Manifest inventory", expanded=False):
            inventory = diagnostics.get("manifest_inventory") or []
            if inventory:
                st.dataframe(pd.DataFrame(inventory), use_container_width=True, hide_index=True)
        with st.expander("Full diagnostics JSON", expanded=False):
            st.json(diagnostics)


def render_inspector(session_id: str, diagnostics: dict[str, Any], developer_mode: bool) -> None:
    st.markdown('<div class="mic-panel"><div class="mic-panel-title">Inspector</div>', unsafe_allow_html=True)
    tab_sources, tab_trace, tab_runtime = st.tabs(["Sources", "Trace", "Runtime"])
    with tab_sources:
        render_source_inspector(session_id, developer_mode)
    with tab_trace:
        render_trace_inspector(session_id)
    with tab_runtime:
        render_runtime_diagnostics(diagnostics)
    st.markdown('</div>', unsafe_allow_html=True)


def process_prompt(session_id: str, prompt: str) -> None:
    session = memory.get_session(session_id)
    if session is None:
        raise KeyError(f"Unknown session: {session_id}")

    document_scope = st.session_state.get("active_document_scope")
    document_behavior = st.session_state.get("document_behavior", "auto")
    if not document_scope:
        document_behavior = "auto"

    with st.spinner("MIC 9000 is processing your request..."):
        result = backend.graph.invoke(
            session_id=session_id,
            user_message=prompt,
            document_scope=document_scope,
            document_behavior=document_behavior,
        )

    if not session.title or session.title == "New chat":
        memory.rename_session(session_id=session_id, title=derive_session_title(prompt))

    st.session_state["last_graph_result"] = result


active_session_id = ensure_active_session()
developer_mode, _health = render_sidebar(active_session_id)
active_session = memory.get_session(active_session_id)
diagnostics = active_diagnostics()

render_app_header(active_session, diagnostics)

show_inspector = bool(st.session_state.get("ui_show_inspector", True))
if show_inspector:
    chat_col, inspector_col = st.columns([0.68, 0.32], gap="large")
else:
    chat_col = st.container()
    inspector_col = None

with chat_col:
    messages = memory.load_all_messages(active_session_id)
    if not messages:
        render_empty_state()
        render_prompt_suggestions()
    else:
        render_conversation(session_id=active_session_id, developer_mode=developer_mode)

if inspector_col is not None:
    with inspector_col:
        render_inspector(active_session_id, diagnostics, developer_mode)

prompt = st.chat_input("Ask MIC 9000...")
pending_prompt = st.session_state.pop("pending_prompt", None)
submitted_prompt = pending_prompt or prompt

if submitted_prompt:
    try:
        process_prompt(active_session_id, str(submitted_prompt))
        st.rerun()
    except Exception as exc:
        st.error("MIC 9000 could not process the request.")
        st.exception(exc)
