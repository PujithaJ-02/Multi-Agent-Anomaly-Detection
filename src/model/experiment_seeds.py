"""
In this file I am measuring my model honestly across several random seeds.

A single training run is one draw from a random process, so its score is partly luck.
Here I run the exact same setup (clean data, window 64) with 5 different seeds and
report the average and the spread. This does NOT make the model better; it gives me a
truthful picture of how good it already is, and how stable that result is. Reporting a
mean and spread is more honest than cherry-picking one lucky run.
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformer import AnomalyTransformer
from seed import set_seed

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

# Precompute the test labels once (same for every seed).
df = pd.read_csv(RAW_CSV, parse_dates=["timestamp"])
split_idx = int(len(df) * 0.6)
tt = df["timestamp"].to_numpy()[np.arange(len(X_test)) + split_idx + WINDOW]
windows = json.load(open(LABELS))[LABEL_KEY]
true = np.zeros(len(X_test), dtype=bool)
for start, end in windows:
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    true |= (tt >= np.datetime64(s)) & (tt <= np.datetime64(e))

def train_and_eval(seed):
    set_seed(seed)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
    model = AnomalyTransformer(window=WINDOW).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(15):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()

    def scores(X, y):
        e = []
        with torch.no_grad():
            for i in range(0, len(X), 512):
                e.append((model(X[i:i+512].to(device)).cpu() - y[i:i+512]).abs())
        return torch.cat(e).numpy()

    thr = float(np.percentile(scores(X_train, y_train), 99))
    flag = scores(X_test, y_test) > thr
    tp = int((flag & true).sum()); fp = int((flag & ~true).sum()); fn = int((~flag & true).sum())
    p = tp/(tp+fp) if (tp+fp) else 0.0
    r = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*p*r/(p+r) if (p+r) else 0.0
    return p, r, f1

seeds = [1, 2, 3, 4, 5]
results = []
print(f"{'seed':>5} {'precision':>10} {'recall':>8} {'F1':>7}")
for s in seeds:
    p, r, f1 = train_and_eval(s)
    results.append((p, r, f1))
    print(f"{s:>5} {p:>10.3f} {r:>8.3f} {f1:>7.3f}")

arr = np.array(results)
mean = arr.mean(axis=0)
std = arr.std(axis=0)
print("\n--- HONEST SUMMARY across 5 seeds ---")
print(f"precision: {mean[0]:.3f} +/- {std[0]:.3f}")
print(f"recall:    {mean[1]:.3f} +/- {std[1]:.3f}")
print(f"F1:        {mean[2]:.3f} +/- {std[2]:.3f}")
