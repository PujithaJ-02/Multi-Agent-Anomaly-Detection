"""
In this file I measure the real latency of my streaming system.

I measure TWO things separately, because my design has two speeds:
  1. fast-path latency: time to SCORE and FLAG one reading (model only, milliseconds)
  2. slow-path latency: time for the agent graph to produce an ALERT (LLM, seconds)

I report both honestly. Conflating them would be misleading. The "before" baseline (a
batch job that runs every few minutes, leaving a multi-minute detection gap) is framed
honestly in my notes, not measured here; here I measure the real streaming "after".
"""
import json
import sys
import time
from collections import deque

import numpy as np
import torch
from kafka import KafkaConsumer

sys.path.append("../model")
sys.path.append("../agents")
from transformer import AnomalyTransformer
from graph import build_graph

TOPIC = "temperature-readings"
BROKER = "localhost:9092"
CONFIG_PATH = "../../models/detector_config.json"
WEIGHTS_PATH = "../../models/transformer.pt"

device = "mps" if torch.backends.mps.is_available() else "cpu"
cfg = json.load(open(CONFIG_PATH))
WINDOW = cfg["window"]
THRESHOLD = cfg["threshold"]
TMIN, TMAX = cfg["min"], cfg["max"]

model = AnomalyTransformer(window=WINDOW).to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

def scale(v):
    return (v - TMIN) / (TMAX - TMIN)

graph = build_graph()

consumer = KafkaConsumer(
    TOPIC, bootstrap_servers=BROKER,
    auto_offset_reset="earliest",
    value_deserializer=lambda b: json.loads(b.decode("utf-8")),
)

buffer = deque(maxlen=WINDOW)
recent = deque(maxlen=10)

fast_times = []     # milliseconds to score+flag one reading
slow_times = []     # seconds for the agent graph to produce an alert
MAX_ANOMALIES = 5   # only fully process a few, since the LLM is slow

print("measuring latency on the live stream ...")
seen = 0
handled = 0
for msg in consumer:
    reading = msg.value
    recent.append(round(reading["value"], 1))
    buffer.append(scale(reading["value"]))
    seen += 1
    if len(buffer) < WINDOW:
        continue

    # --- FAST PATH: time the scoring + flag decision ---
    t0 = time.perf_counter()
    window = torch.tensor(np.array(buffer, dtype="float32")).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(window).item()
    score = abs(pred - scale(reading["value"]))
    flagged = score > THRESHOLD
    fast_ms = (time.perf_counter() - t0) * 1000
    fast_times.append(fast_ms)

    # --- SLOW PATH: time the full agent graph to an alert ---
    if flagged and handled < MAX_ANOMALIES:
        t1 = time.perf_counter()
        graph.invoke({
            "timestamp": reading["timestamp"], "value": reading["value"],
            "recent": list(recent),
            "anomaly_type": None, "severity": None,
            "alert_sent": None, "alert_message": None,
        })
        slow_s = time.perf_counter() - t1
        slow_times.append(slow_s)
        handled += 1
        print(f"  [{handled}] fast-path {fast_ms:.2f} ms, slow-path {slow_s:.2f} s")

    if handled >= MAX_ANOMALIES:
        break

# --- Honest summary ---
print("\n=== LATENCY RESULTS ===")
print(f"readings scored: {len(fast_times)}")
print(f"fast path (score + flag): avg {np.mean(fast_times):.2f} ms, "
      f"max {np.max(fast_times):.2f} ms")
print(f"slow path (full agent alert): avg {np.mean(slow_times):.2f} s, "
      f"max {np.max(slow_times):.2f} s")
print("\nbaseline for comparison: a batch job running every few minutes would leave")
print("up to a multi-minute detection gap. streaming replaces that with the above.")
