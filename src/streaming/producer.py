"""
In this file I am the PRODUCER: I replay my saved NAB temperature readings into Kafka,
one at a time, to fake a live sensor stream.

I read the raw CSV, then send each reading as a small JSON message to the
"temperature-readings" topic. I pause briefly between sends so it behaves like a real
stream arriving over time instead of dumping everything at once. Later the consumer
reads these messages and runs the model on them.
"""
import json
import time
import pandas as pd
from kafka import KafkaProducer

TOPIC = "temperature-readings"
import os
BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")
CSV_PATH = "../../data/raw/machine_temp.csv"
DELAY_SECONDS = 0.05   # small gap between messages so it looks like a live stream

# value_serializer turns my Python dict into JSON bytes automatically before sending.
producer = KafkaProducer(
    bootstrap_servers=BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

df = pd.read_csv(CSV_PATH)
print(f"streaming {len(df)} readings into topic '{TOPIC}' ...")

sent = 0
for row in df.itertuples(index=False):
    message = {"timestamp": str(row.timestamp), "value": float(row.value)}
    producer.send(TOPIC, message)
    sent += 1
    if sent % 500 == 0:
        print(f"  sent {sent} readings")
    time.sleep(DELAY_SECONDS)

# flush makes sure every buffered message is actually delivered before I exit.
producer.flush()
print(f"done. sent {sent} readings total.")