"""
In this file I am testing lever 2: does training on cleaned data improve recall?

I train a fresh model on the cleaned training set (anomaly readings removed), then
immediately evaluate it on the SAME untouched test set and threshold rule as before,
so the comparison against my baseline (precision 0.86, recall 0.29) is fair. I change
only the training data. Everything else is held constant.
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformer import AnomalyTransformer

PROC = "../../data/processed_clean"
RAW_CSV = "../../data/raw/machine_temp.csv"
LABELS = "../../data/raw/labels.json"
LABEL_KEY = "realKnownCause/machine_temperature_system_failure.csv"
device = "mps" if torch.backends.mps.is_available() else "cpu"

X_train = torch.tensor(np.load(f"{PROC}/X_train.npy"))
y_train = torch.tensor(np.load(f"{PROC}/y_train.npy"))
X_test = torch.tensor(np.load(f"{PROC}/X_test.npy"))
y_test = torch.tensor(np.load(f"{PROC}/y_test.npy"))
WINDOW = json.load(open(f"{PROC}/scaler.json"))["window"]

# Train fresh, same settings as the original run.
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
model = AnomalyTransformer().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
for epoch in range(15):
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        total += loss.item()
    print(f"epoch {epoch+1:2d}/15  loss {total/len(loader):.6f}")

def scores(X, y):
    errs = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            errs.append((model(X[i:i+512].to(device)).cpu() - y[i:i+512]).abs())
    return torch.cat(errs).numpy()

train_err = scores(X_train, y_train)
test_err = scores(X_test, y_test)
threshold = float(np.percentile(train_err, 99))

# Same label alignment as the baseline eval.
df = pd.read_csv(RAW_CSV, parse_dates=["timestamp"])
split_idx = int(len(df) * 0.6)
test_times = df["timestamp"].to_numpy()[np.arange(len(X_test)) + split_idx + WINDOW]
windows = json.load(open(LABELS))[LABEL_KEY]
true = np.zeros(len(X_test), dtype=bool)
for start, end in windows:
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    true |= (test_times >= np.datetime64(s)) & (test_times <= np.datetime64(e))

flag = test_err > threshold
tp = int((flag & true).sum()); fp = int((flag & ~true).sum()); fn = int((~flag & true).sum())
p = tp/(tp+fp) if (tp+fp) else 0.0
r = tp/(tp+fn) if (tp+fn) else 0.0
f1 = 2*p*r/(p+r) if (p+r) else 0.0
print("\n--- CLEAN-DATA RESULT (99th percentile threshold) ---")
print(f"TP {tp}  FP {fp}  FN {fn}")
print(f"precision {p:.3f}  recall {r:.3f}  F1 {f1:.3f}")
print("baseline was: precision 0.860  recall 0.292  F1 0.436")
