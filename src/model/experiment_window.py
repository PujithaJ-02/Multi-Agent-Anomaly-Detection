"""
In this file I am testing lever 3: does a different window size improve results?

I keep the cleaned-training approach (which already helped) and change ONLY the window
size. For each size I rebuild the clean windows, train a fresh model, and evaluate on
the same time-based test split with the same 99th-percentile threshold rule. Then I
compare F1 across sizes. I change one thing at a time so I know what caused any change.

Current best (window 64, clean data): precision 0.703, recall 0.571, F1 0.630.
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformer import AnomalyTransformer

RAW_CSV = "../../data/raw/machine_temp.csv"
LABELS = "../../data/raw/labels.json"
LABEL_KEY = "realKnownCause/machine_temperature_system_failure.csv"
TRAIN_FRAC = 0.6
device = "mps" if torch.backends.mps.is_available() else "cpu"

df = pd.read_csv(RAW_CSV, parse_dates=["timestamp"])
values = df["value"].to_numpy(dtype="float32")
times = df["timestamp"].to_numpy()

windows_lbl = json.load(open(LABELS))[LABEL_KEY]
is_anom = np.zeros(len(df), dtype=bool)
for start, end in windows_lbl:
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    is_anom |= (times >= np.datetime64(s)) & (times <= np.datetime64(e))

split_idx = int(len(values) * TRAIN_FRAC)

def build(win):
    # clean training values (non-anomaly, training portion only)
    tmask = np.zeros(len(values), dtype=bool); tmask[:split_idx] = True
    train_raw = values[tmask & ~is_anom]
    test_raw = values[split_idx:]
    tmin, tmax = train_raw.min(), train_raw.max()
    sc = lambda a: (a - tmin) / (tmax - tmin)
    tr, te = sc(train_raw), sc(test_raw)
    def mk(series):
        X, y = [], []
        for i in range(len(series) - win):
            X.append(series[i:i+win]); y.append(series[i+win])
        return torch.tensor(np.array(X, dtype="float32")), torch.tensor(np.array(y, dtype="float32"))
    return mk(tr), mk(te)

def run(win):
    (Xtr, ytr), (Xte, yte) = build(win)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=128, shuffle=True)
    model = AnomalyTransformer(window=win).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(15):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    def sc(X, y):
        e = []
        with torch.no_grad():
            for i in range(0, len(X), 512):
                e.append((model(X[i:i+512].to(device)).cpu() - y[i:i+512]).abs())
        return torch.cat(e).numpy()
    thr = float(np.percentile(sc(Xtr, ytr), 99))
    test_err = sc(Xte, yte)
    tt = df["timestamp"].to_numpy()[np.arange(len(Xte)) + split_idx + win]
    true = np.zeros(len(Xte), dtype=bool)
    for start, end in windows_lbl:
        s, e = pd.to_datetime(start), pd.to_datetime(end)
        true |= (tt >= np.datetime64(s)) & (tt <= np.datetime64(e))
    flag = test_err > thr
    tp = int((flag & true).sum()); fp = int((flag & ~true).sum()); fn = int((~flag & true).sum())
    p = tp/(tp+fp) if (tp+fp) else 0.0
    r = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*p*r/(p+r) if (p+r) else 0.0
    return p, r, f1

print(f"{'window':>7} {'~hours':>7} {'precision':>10} {'recall':>8} {'F1':>7}")
for win in [16, 32, 64, 128, 192]:
    p, r, f1 = run(win)
    print(f"{win:>7} {win*5/60:>7.1f} {p:>10.3f} {r:>8.3f} {f1:>7.3f}")
