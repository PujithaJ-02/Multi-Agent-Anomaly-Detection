"""
In this file I measure throughput with BATCHED scoring.

Before, I scored one window per model call, which wastes time on per-call overhead.
Here I collect windows into batches and score a whole batch in ONE model call. The
fixed overhead is paid once per batch instead of once per window, so throughput rises.

Honest tradeoff: batching means waiting to fill a batch before scoring, which adds a
little latency per reading. In a real system I would tune batch size to balance
throughput against latency. This measures the fast path's capacity.
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
BATCH_SIZE = 256   # score this many windows in one model call

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
    consumer_timeout_ms=5000,
)

buffer = deque(maxlen=WINDOW)
batch = []
scored = 0

# Warm up so the first call does not skew timing.
_ = model(torch.zeros(BATCH_SIZE, WINDOW).to(device))

def score_batch(windows):
    x = torch.tensor(np.array(windows, dtype="float32")).to(device)
    with torch.no_grad():
        _ = model(x)

print(f"measuring BATCHED throughput (batch size {BATCH_SIZE}) ...")
start = time.perf_counter()
for msg in consumer:
    buffer.append(scale(msg.value["value"]))
    if len(buffer) < WINDOW:
        continue
    batch.append(list(buffer))
    if len(batch) >= BATCH_SIZE:
        score_batch(batch)
        scored += len(batch)
        batch = []
# score any leftover windows
if batch:
    score_batch(batch)
    scored += len(batch)
elapsed = time.perf_counter() - start

print("\n=== BATCHED THROUGHPUT RESULT ===")
print(f"readings scored: {scored}")
print(f"elapsed: {elapsed:.2f} s")
if elapsed > 0:
    print(f"throughput: {scored / elapsed:.0f} readings/second (batched fast path)")
print(f"(compare: single-item was 787/sec)")
