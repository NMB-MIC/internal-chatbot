# MIC 9000 v1.0-RC Release Notes

## Release status

MIC 9000 v1.0-RC is the first production-freeze candidate for the internal RAG chatbot.

## Final accepted architecture

- UI runtime: Streamlit
- Retrieval: hybrid dense/lexical RAG with selected-document behavior
- Vector store: Qdrant
- Embeddings: BAAI/bge-m3
- LLM serving: Ollama / Gemma 4 26B configuration
- Memory: MIC SQLite memory as source of truth
- Index safety: production/development modes, manifesting, quarantine, snapshots
- Evaluation: repeatable eval suites with isolated sessions
- Deployment/ops: env template, preflight, production readiness, backup/restore, release checks

## Release gate status expected

```text
runbook_regression      PASS
document_mode_smoke     PASS
thai_runbook_smoke      PASS
production readiness    PASS
active index match      PASS
```

## Production index baseline

The current production-clean baseline is expected to contain:

```text
accepted_files: 2
indexed_points: 38
quarantined_files: 10
rejected_files: 0
```

The accepted production documents are expected to include:

```text
developer_support/machine_status_prediction_runbook.md
```

## Notes

Batch 16 is a freeze/release layer only. No behavior change should be introduced here.
