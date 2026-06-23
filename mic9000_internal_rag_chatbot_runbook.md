MIC 9000 is an internal RAG chatbot and operations console for the Manufacturing Improvement Yokoten Center. It provides grounded internal-document question answering, developer support, technical troubleshooting, chat history, source inspection, runtime diagnostics, evaluation gates, and production release controls through a Streamlit interface backed by Qdrant, SQLite, Ollama, and BGE-M3 embeddings.

# MIC 9000 Internal RAG Chatbot Runbook

A production-oriented internal support chatbot that reads approved internal documents, retrieves relevant chunks from Qdrant, generates grounded answers through a local Ollama LLM, stores conversations and retrieval traces in SQLite, and exposes the full workflow through a Streamlit operations UI.

---

## Overview

MIC 9000 is designed for internal technical support, document question answering, and developer troubleshooting within the MIC division.

The system receives user questions through Streamlit, routes the request through an internal support graph, optionally constrains retrieval to a selected document, retrieves grounded evidence from Qdrant, generates an answer with source references, stores the full conversation and retrieval metadata in SQLite, and surfaces operational diagnostics for developer/admin users.

The system is intended to support:

- Internal document Q&A
- Developer support
- Admin support
- Kafka / IoT / machine-log troubleshooting
- Grounded answers with visible sources
- Session history and conversation resume
- Runtime health monitoring
- Production index safety checks
- Evaluation-based release gating

The system is not intended to answer from unapproved documents, expose private secrets, run arbitrary code, or replace human escalation for unresolved operational incidents.

---

## Production Deployment

Prerequisites:

- Python virtual environment is active
- `.env.production` exists
- Qdrant is running on `localhost:6333`
- Ollama is running on `localhost:11434`
- `gemma4:26b` is available in Ollama
- Approved documents are placed under `data/documents/`
- Production index has been rebuilt and verified

### 1. Activate Environment

```bash
source .venv/bin/activate
```
### 1.5 Start Qdrant

```bash
docker run --rm \
  --name internal-rag-qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
  qdrant/qdrant
```
### 2. Load Production Environment

```bash
set -a
source .env.production
set +a
```

### 3. Run Release Checks

```bash
bash scripts/run_release_checks.sh .env.production
```

Expected:

```text
failure_count: 0
Release checks complete
```

### 4. Run Final Release Gate

```bash
bash scripts/final_release_check.sh .env.production
```

Expected:

```text
FINAL RELEASE CHECK: PASS
```

### 5. Start Streamlit

Manual command:

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8000
```

Preferred helper script:

```bash
bash scripts/run_streamlit_prod.sh .env.production
```

Access from another machine:

```text
http://<server-ip>:8000
```

---

## Normal Operations

### Start Services

1. Start Qdrant.
2. Start Ollama.
3. Confirm `gemma4:26b` exists.
4. Load `.env.production`.
5. Run Streamlit.

### Daily Health Check

```bash
bash scripts/run_release_checks.sh .env.production
```

For lighter checks, use the Streamlit Runtime panel or diagnostics script.

### Inspect Latest Release

```bash
ls -lh storage/releases/
python scripts/print_release_summary.py storage/releases/<release_manifest>.json
```

### Backup

```bash
bash scripts/backup_mic9000.sh .env.production
```

### Restore

```bash
bash scripts/restore_mic9000_backup.sh <backup_file>
```

### Collect Ops Bundle

```bash
bash scripts/collect_ops_bundle.sh .env.production
```

---

## Acceptance Gates

MIC 9000 is production-ready only when all of the following are true:

- Streamlit app launches successfully
- Qdrant is reachable
- Ollama is reachable
- SQLite parent path exists
- Latest manifest exists
- Latest manifest mode is `production`
- Latest manifest has indexed points
- Active Qdrant point count matches latest manifest
- Production rejected files count is `0`
- Security is enabled
- Public trace is disabled
- Public diagnostics are disabled
- Unlock hints are disabled
- Admin actions are disabled unless intentionally enabled
- Eval suites pass
- Final release check passes

Expected final release gate:

```text
FINAL RELEASE CHECK: PASS
```

---

## Troubleshooting

### Streamlit App Does Not Start

Check:

```bash
python -m py_compile streamlit_app.py app/ui/*.py
streamlit --version
```

Then run:

```bash
set -a
source .env.production
set +a
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8000
```

### Qdrant Unreachable

Check:

```bash
curl http://localhost:6333/collections
```

If unreachable:

- Start Qdrant
- Verify port `6333`
- Verify `QDRANT_URL` in `.env.production`

### Ollama Unreachable

Check:

```bash
curl http://localhost:11434/api/tags
ollama list
```

If model is missing:

```bash
ollama pull gemma4:26b
```

### No Sources Shown

Check:

- Show sources toggle is enabled
- Question is routed to RAG, not direct identity route
- Selected document exists
- Qdrant point count is greater than zero
- Similarity threshold is not too high
- Current document filter is not too restrictive

### Wrong Document Used

Use strict selected-document mode.

Verify selected document:

```text
selected_document_exists = true
```

Run diagnostics from the developer panel.

### Sidebar Text Is Hard to Read

This is a UI regression.

Check `app/ui/styles.py` and verify sidebar controls explicitly style:

- Selectbox surface
- Dropdown menu
- Input text
- Placeholder text
- Disabled text
- Button text
- Hover and focus states

Do not use broad white text rules without also forcing dark control backgrounds.

### HF Hub Warning Appears

Warning:

```text
You are sending unauthenticated requests to the HF Hub.
```

This is not fatal if the embedding model is already cached.

To reduce warnings and improve download reliability, set:

```text
HF_TOKEN=<token>
```

### Admin Rebuild Button Disabled

Production default:

```text
MIC_ADMIN_ACTIONS_ENABLED=false
```

This is intentional.

Enable admin actions only when performing controlled maintenance.

---

## Code Cleanup Notes

Do not delete without backup:

```text
data/sqlite/chat_history.db
storage/index_manifests/
storage/releases/
storage/backups/
storage/index_quarantine/
eval_reports/
.env.production
```

---

## Current Release Baseline

Latest confirmed release candidate:

```text
Version: 1.0.0-rc1
Release gate: pass
UI runtime: Streamlit
Index mode: production
Production rejected files: 0
Eval suites: pass
```

After removing the CV and rebuilding the index, regenerate the release manifest so the frozen release state matches the new approved document set.

---

## Operational Summary

```text
User
  -> Streamlit UI
  -> Internal Support Graph
  -> Route + Document Scope
  -> Hybrid Retrieval from Qdrant
  -> Grounded Answer from Ollama
  -> SQLite Conversation + Retrieval Logs
  -> Streamlit Sources / Trace / Diagnostics
```

MIC 9000 is ready for internal use when the production index contains only approved documents, runtime diagnostics pass, eval suites pass, and the final release gate reports:

```text
FINAL RELEASE CHECK: PASS
```
