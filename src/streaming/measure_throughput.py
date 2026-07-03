"""
In this file I measure the real THROUGHPUT of my fast path: how many readings per
second my consumer can read from Kafka and score with the model.

This is the honest number behind a "events per second" claim. It is NOT the agent
path (that runs only on rare flagged anomalies and is slow by design). It is the fast
path doing real work: read, window, run the Transformer.

I assume the topic is already full of readings (run the producer first). I read them
all as fast as possible, score each, and divide count by elapsed time.
"""
import json
import sys
import time
from collections import deque

import numpy as np
import torch
from kafka import KafkaConsumer

sys.path.append("../model")
from transformer import AnomalyTransformer

TOPIC = "temperature-readings"
BROKER = "localhost:9092"
CONFIG_PATH = "../../models/detector_config.json"
WEIGHTS_PATH = "../../models/transformer.pt"

device = "mps" if torch.backends.mps.is_available() else "cpu"
cfg = json.load(open(CONFIG_PATH))
WINDOW = cfg["window"]
TMIN, TMAX = cfg["min"], cfg["max"]

model = AnomalyTransformer(window=WINDOW).to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

def scale(v):
    return (v - TMIN) / (TMAX - TMIN)

consumer = KafkaConsumer(
    TOPIC, bootstrap_servers=BROKER,
    auto_offset_reset="earliest",
    value_deserializer=lambda b: json.loads(b.decode("utf-8")),
    consumer_timeout_ms=5000,   # stop after 5s of no new messages
)

buffer = deque(maxlen=WINDOW)
scored = 0

# Warm up the model once so the first-call cost does not skew the rate.
_ = model(torch.zeros(1, WINDOW).to(device))

print("measuring throughput (reading + scoring as fast as possible) ...")
start = time.perf_counter()
for msg in consumer:
    reading = msg.value
    buffer.append(scale(reading["value"]))
    if len(buffer) < WINDOW:
        continue
    window = torch.tensor(np.array(buffer, dtype="float32")).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(window).item()
    scored += 1
elapsed = time.perf_counter() - start

print("\n=== THROUGHPUT RESULT ===")
print(f"readings scored: {scored}")
print(f"elapsed: {elapsed:.2f} s")
if elapsed > 0:
    print(f"throughput: {scored / elapsed:.0f} readings/second (scored by the fast path)")
