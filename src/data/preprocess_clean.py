"""
In this file I am building a CLEANER training set to test lever 2 for improving recall.

My hypothesis: two anomaly windows fell inside my training data, so the model learned
to treat some anomalies as normal, which blunted its ability to be surprised by real
anomalies later. Here I remove any training reading that falls inside a labeled
anomaly window, so the model learns a purer picture of "normal".

Honest caveat: removing those readings leaves small time-gaps in training and makes it
a bit smaller. That is a minor imperfection I accept in order to test the hypothesis.
The test set is built exactly as before so the comparison is fair.
"""
import os
import json
import numpy as np
import pandas as pd

RAW_CSV = "../../data/raw/machine_temp.csv"
LABELS = "../../data/raw/labels.json"
LABEL_KEY = "realKnownCause/machine_temperature_system_failure.csv"
OUT_DIR = "../../data/processed_clean"
WINDOW = 64
TRAIN_FRAC = 0.6

df = pd.read_csv(RAW_CSV, parse_dates=["timestamp"])
values = df["value"].to_numpy(dtype="float32")
times = df["timestamp"].to_numpy()

# Mark which readings fall inside a labeled anomaly window.
windows = json.load(open(LABELS))[LABEL_KEY]
is_anom = np.zeros(len(df), dtype=bool)
for start, end in windows:
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    is_anom |= (times >= np.datetime64(s)) & (times <= np.datetime64(e))

split_idx = int(len(values) * TRAIN_FRAC)

# Training: keep only the NON-anomaly readings from the training portion.
train_mask = np.zeros(len(values), dtype=bool)
train_mask[:split_idx] = True
train_clean_mask = train_mask & ~is_anom
train_raw = values[train_clean_mask]

# Test: exactly as before (the whole later portion, untouched) for a fair comparison.
test_raw = values[split_idx:]

print("original train readings:", split_idx)
print("clean train readings:", len(train_raw), "(removed", split_idx - len(train_raw), "anomaly readings)")

# Scale using clean-train stats only.
tmin, tmax = train_raw.min(), train_raw.max()
scale = lambda a: (a - tmin) / (tmax - tmin)
train_s, test_s = scale(train_raw), scale(test_raw)

def make_windows(series, w):
    X, y = [], []
    for i in range(len(series) - w):
        X.append(series[i:i + w]); y.append(series[i + w])
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")

X_train, y_train = make_windows(train_s, WINDOW)
X_test, y_test = make_windows(test_s, WINDOW)
print("clean train windows:", X_train.shape, " test windows:", X_test.shape)

os.makedirs(OUT_DIR, exist_ok=True)
np.save(f"{OUT_DIR}/X_train.npy", X_train)
np.save(f"{OUT_DIR}/y_train.npy", y_train)
np.save(f"{OUT_DIR}/X_test.npy", X_test)
np.save(f"{OUT_DIR}/y_test.npy", y_test)
json.dump({"min": float(tmin), "max": float(tmax), "window": WINDOW},
          open(f"{OUT_DIR}/scaler.json", "w"))
print("saved to", OUT_DIR)
