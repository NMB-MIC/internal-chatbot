# Machine Log Streaming Runbook — Batch 3 Synthetic Sample



## Local Setup



The local machine-log streaming service reads synthetic machine events, validates the payload structure, serialises each event, and sends the result to the configured Kafka broker. During local development, verify the environment file, broker reachability, topic mapping, and producer logs before changing application code.

The local machine-log streaming service reads synthetic machine events, validates the payload structure, serialises each event, and sends the result to the configured Kafka broker. During local development, verify the environment file, broker reachability, topic mapping, and producer logs before changing application code.

The local machine-log streaming service reads synthetic machine events, validates the payload structure, serialises each event, and sends the result to the configured Kafka broker. During local development, verify the environment file, broker reachability, topic mapping, and producer logs before changing application code.

The local machine-log streaming service reads synthetic machine events, validates the payload structure, serialises each event, and sends the result to the configured Kafka broker. During local development, verify the environment file, broker reachability, topic mapping, and producer logs before changing application code.

The local machine-log streaming service reads synthetic machine events, validates the payload structure, serialises each event, and sends the result to the configured Kafka broker. During local development, verify the environment file, broker reachability, topic mapping, and producer logs before changing application code.

The local machine-log streaming service reads synthetic machine events, validates the payload structure, serialises each event, and sends the result to the configured Kafka broker. During local development, verify the environment file, broker reachability, topic mapping, and producer logs before changing application code.

The local machine-log streaming service reads synthetic machine events, validates the payload structure, serialises each event, and sends the result to the configured Kafka broker. During local development, verify the environment file, broker reachability, topic mapping, and producer logs before changing application code.



## Example Local Command



```bash

export KAFKA_BROKER_URL='use-approved-internal-value'

export KAFKA_TOPIC='use-plant-specific-mapping'

python -m app.synthetic_producer --dry-run

```



## Troubleshooting Order



When messages do not appear downstream, investigate the data path in a fixed order. Confirm that source events exist, then inspect the producer service, then verify network reachability, authentication, broker health, topic configuration, and consumer lag. Do not assume that different plants share identical configuration values.

When messages do not appear downstream, investigate the data path in a fixed order. Confirm that source events exist, then inspect the producer service, then verify network reachability, authentication, broker health, topic configuration, and consumer lag. Do not assume that different plants share identical configuration values.

When messages do not appear downstream, investigate the data path in a fixed order. Confirm that source events exist, then inspect the producer service, then verify network reachability, authentication, broker health, topic configuration, and consumer lag. Do not assume that different plants share identical configuration values.

When messages do not appear downstream, investigate the data path in a fixed order. Confirm that source events exist, then inspect the producer service, then verify network reachability, authentication, broker health, topic configuration, and consumer lag. Do not assume that different plants share identical configuration values.

When messages do not appear downstream, investigate the data path in a fixed order. Confirm that source events exist, then inspect the producer service, then verify network reachability, authentication, broker health, topic configuration, and consumer lag. Do not assume that different plants share identical configuration values.

When messages do not appear downstream, investigate the data path in a fixed order. Confirm that source events exist, then inspect the producer service, then verify network reachability, authentication, broker health, topic configuration, and consumer lag. Do not assume that different plants share identical configuration values.

When messages do not appear downstream, investigate the data path in a fixed order. Confirm that source events exist, then inspect the producer service, then verify network reachability, authentication, broker health, topic configuration, and consumer lag. Do not assume that different plants share identical configuration values.

When messages do not appear downstream, investigate the data path in a fixed order. Confirm that source events exist, then inspect the producer service, then verify network reachability, authentication, broker health, topic configuration, and consumer lag. Do not assume that different plants share identical configuration values.