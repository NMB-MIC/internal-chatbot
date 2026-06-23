# MIC 9000 Eval Report — document_mode_smoke

- Suite version: `1.1`
- Started: `2026-06-22T07:59:20+00:00`
- Completed: `2026-06-22T07:59:39+00:00`
- Duration: `18.932s`
- Session ID: `c29eb7f5-fadd-4647-9df4-6c940874a997`

## Summary

| Metric | Value |
|---|---:|
| Total | 3 |
| Passed | 2 |
| Failed | 1 |
| Skipped | 0 |
| Pass rate | 66.67% |

## Cases

| Status | Case | Latency | Failures |
|---|---|---:|---|
| PASS | `strict_runbook_health` | 11.222s |  |
| FAIL | `strict_runbook_mqtt_topic` | 4.470s | forbidden fragment present: cannot confirm |
| PASS | `strict_cv_current_company` | 3.221s |  |

## Failed case details

### strict_runbook_mqtt_topic

**Question**

What is the raw Kafka topic?

**Failures**

- forbidden fragment present: cannot confirm

**Answer**

```text
I cannot confirm the raw Kafka topic for the health endpoint from the available internal knowledge base. The provided documents specify that the health endpoint is accessed via a URL (e.g., `http://172.18.0.2:30080/health` [S1]) and that the raw Kafka topic used in the pipeline is `iot.machine.status.raw` [S3, S7], but there is no mention of a Kafka topic specifically associated with the health endpoint itself.
```
