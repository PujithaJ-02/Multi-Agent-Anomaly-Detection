"""
In this file I am the CONSUMER: the fast path. I read temperature readings from Kafka
one at a time, keep a rolling window of the last 64, run my trained model on that
window, and flag the reading if its anomaly score crosses the threshold.

Key idea (stateful streaming): the model needs 64 readings to make one prediction, but
Kafka delivers them one by one. So I hold the last 64 in a rolling buffer. Each new
reading pushes out the oldest. I can only start scoring once I have collected 64.
"""
import json
import sys
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
THRESHOLD = cfg["threshold"]
TMIN, TMAX = cfg["min"], cfg["max"]
print(f"loaded model config: window={WINDOW} threshold={THRESHOLD:.5f}")

model = AnomalyTransformer(window=WINDOW).to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

def scale(v):
    return (v - TMIN) / (TMAX - TMIN)

buffer = deque(maxlen=WINDOW)

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=BROKER,
    auto_offset_reset="earliest",
    value_deserializer=lambda b: json.loads(b.decode("utf-8")),
)

print("consumer running. waiting for readings ...")
seen = 0
flagged = 0
for msg in consumer:
    reading = msg.value
    buffer.append(scale(reading["value"]))
    seen += 1

    if len(buffer) < WINDOW:
        continue

    window = torch.tensor(np.array(buffer, dtype="float32")).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(window).item()

    actual = scale(reading["value"])
    score = abs(pred - actual)

    if score > THRESHOLD:
        flagged += 1
        print(f"ANOMALY  ts={reading['timestamp']}  value={reading['value']:.2f}  score={score:.4f}")

    if seen % 1000 == 0:
        print(f"  processed {seen} readings, flagged {flagged} so far")
