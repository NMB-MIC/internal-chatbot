# MIC 9000 Eval Report — document_mode_smoke

- Suite version: `1.0`
- Started: `2026-06-22T04:24:17+00:00`
- Completed: `2026-06-22T04:24:36+00:00`
- Duration: `18.787s`
- Session ID: `0bc3718f-44e9-4d51-9d9c-ad811545a5f8`

## Summary

| Metric | Value |
|---|---:|
| Total | 3 |
| Passed | 1 |
| Failed | 2 |
| Skipped | 0 |
| Pass rate | 33.33% |

## Cases

| Status | Case | Latency | Failures |
|---|---|---:|---|
| PASS | `strict_runbook_health` | 12.351s |  |
| FAIL | `prefer_runbook_mqtt_topic` | 3.195s | missing expected fragment: iot.machine.status.raw<br>answer does not contain a valid [S#] citation<br>expected raw_hit_count=24, got 8<br>expected accepted_hit_count=8, got 4<br>expected similarity_threshold=0.28, got 0.35<br>expected selected source path in retrieval summary: developer_support/machine_status_prediction_runbook.md |
| FAIL | `strict_cv_current_company` | 3.215s | expected raw_hit_count=24, got 4<br>expected accepted_hit_count=8, got 4 |

## Failed case details

### prefer_runbook_mqtt_topic

**Question**

What is the raw Kafka topic?

**Failures**

- missing expected fragment: iot.machine.status.raw
- answer does not contain a valid [S#] citation
- expected raw_hit_count=24, got 8
- expected accepted_hit_count=8, got 4
- expected similarity_threshold=0.28, got 0.35
- expected selected source path in retrieval summary: developer_support/machine_status_prediction_runbook.md

**Answer**

```text
I cannot confirm the raw Kafka topic for the health endpoint from the available internal knowledge base, as the provided documents do not contain information regarding a health endpoint's Kafka topic.
```

### strict_cv_current_company

**Question**

What is Apiwit's current company?

**Failures**

- expected raw_hit_count=24, got 4
- expected accepted_hit_count=8, got 4

**Answer**

```text
Apiwit Nathong's current company is NMB-Minebea Thai Ltd., where he has been working as an AI Engineer since September 2025 [S2].
```
