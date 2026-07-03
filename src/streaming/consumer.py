"""
In this file I am the CONSUMER: the fast path, now connected to the slow path.

I read readings from Kafka, keep a rolling 64-window, run the model, and flag anomalies
(same as before). The NEW part: when I flag an anomaly, I do NOT run the agents here,
because the local LLM takes seconds and that would stall the stream. Instead I drop the
flagged anomaly into a background QUEUE and immediately go read the next reading.

A separate WORKER thread pulls flagged anomalies off the queue and runs the LangGraph
agent graph on them. This keeps the fast path fast and the slow path independent, which
is the whole point of the two-speed design.

Honest scope: this is one process on one machine. In production the handoff would be a
second Kafka topic so the agent service could scale on its own. The queue here shows the
same decoupling, simply.
"""
import json
import sys
import time
import threading
import queue
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
print(f"loaded model config: window={WINDOW} threshold={THRESHOLD:.5f}")

model = AnomalyTransformer(window=WINDOW).to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

def scale(v):
    return (v - TMIN) / (TMAX - TMIN)

# The background queue: the fast path drops flagged anomalies here and moves on.
anomaly_queue = queue.Queue()

# The slow-path worker: pulls flagged anomalies and runs the agent graph.
def slow_path_worker():
    graph = build_graph()
    recent_buffer = deque(maxlen=10)  # a little context for the agents
    while True:
        item = anomaly_queue.get()      # waits here until something is flagged
        if item is None:
            break
        reading, recent = item
        print(f"\n--- slow path handling anomaly at {reading['timestamp']} ---")
        graph.invoke({
            "timestamp": reading["timestamp"],
            "value": reading["value"],
            "recent": recent,
            "anomaly_type": None, "severity": None,
            "alert_sent": None, "alert_message": None,
        })
        anomaly_queue.task_done()

worker = threading.Thread(target=slow_path_worker, daemon=True)
worker.start()

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=BROKER,
    auto_offset_reset="earliest",
    value_deserializer=lambda b: json.loads(b.decode("utf-8")),
)

print("consumer running (fast path + slow-path worker). reading stream ...")
buffer = deque(maxlen=WINDOW)
recent_values = deque(maxlen=10)   # raw recent values to give the agents context
seen = 0
flagged = 0
for msg in consumer:
    reading = msg.value
    recent_values.append(round(reading["value"], 1))
    buffer.append(scale(reading["value"]))
    seen += 1

    if len(buffer) < WINDOW:
        continue

    window = torch.tensor(np.array(buffer, dtype="float32")).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(window).item()
    score = abs(pred - scale(reading["value"]))

    if score > THRESHOLD:
        flagged += 1
        # Hand off to the slow path WITHOUT waiting. Fast path keeps moving.
        anomaly_queue.put((reading, list(recent_values)))

    if seen % 1000 == 0:
        print(f"  fast path: processed {seen}, flagged {flagged}, "
              f"queue depth {anomaly_queue.qsize()}")
