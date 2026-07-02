"""
In this file I am testing my first lever for improving recall: the threshold.

Recall was low (0.29) because my threshold (99th percentile of training error) was
strict, so the model stayed silent on subtle anomalies. Here I sweep several
thresholds at once and print precision/recall/F1 for each, so I can SEE the tradeoff
instead of guessing one number. I change only this one lever; everything else is the
same as evaluate.py.
"""
import json
import numpy as np
import pandas as pd
import torch
from transformer import AnomalyTransformer

RAW_CSV = "../../data/raw/machine_temp.csv"
LABELS = "../../data/raw/labels.json"
LABEL_KEY = "realKnownCause/machine_temperature_system_failure.csv"
PROC = "../../data/processed"

device = "mps" if torch.backends.mps.is_available() else "cpu"
scaler = json.load(open(f"{PROC}/scaler.json"))
WINDOW = scaler["window"]

X_train = torch.tensor(np.load(f"{PROC}/X_train.npy"))
X_test = torch.tensor(np.load(f"{PROC}/X_test.npy"))
y_train = torch.tensor(np.load(f"{PROC}/y_train.npy"))
y_test = torch.tensor(np.load(f"{PROC}/y_test.npy"))

model = AnomalyTransformer().to(device)
model.load_state_dict(torch.load("../../models/transformer.pt", map_location=device))
model.eval()

def scores(X, y):
    errs = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = X[i:i + 512].to(device)
            errs.append((model(xb).cpu() - y[i:i + 512]).abs())
    return torch.cat(errs).numpy()

train_err = scores(X_train, y_train)
test_err = scores(X_test, y_test)

# Same label alignment as before.
df = pd.read_csv(RAW_CSV, parse_dates=["timestamp"])
split_idx = int(len(df) * 0.6)
test_times = df["timestamp"].to_numpy()[np.arange(len(X_test)) + split_idx + WINDOW]
windows = json.load(open(LABELS))[LABEL_KEY]
true = np.zeros(len(X_test), dtype=bool)
for start, end in windows:
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    true |= (test_times >= np.datetime64(s)) & (test_times <= np.datetime64(e))

print(f"{'percentile':>10} {'threshold':>10} {'precision':>10} {'recall':>8} {'F1':>7}")
for pct in [90, 92, 94, 95, 96, 97, 98, 99]:
    thr = float(np.percentile(train_err, pct))
    flag = test_err > thr
    tp = int((flag & true).sum())
    fp = int((flag & ~true).sum())
    fn = int((~flag & true).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    print(f"{pct:>10} {thr:>10.5f} {p:>10.3f} {r:>8.3f} {f1:>7.3f}")
