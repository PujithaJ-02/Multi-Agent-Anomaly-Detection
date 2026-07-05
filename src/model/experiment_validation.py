"""
In this file I properly tune the threshold with a VALIDATION set, then measure ONCE on
the untouched test set. This is the rigorous 3-way split (train / validation / test) I
skipped earlier.

Why: picking the threshold on training data and reporting on test is defensible but not
ideal. The honest method tunes on validation and touches test only for the final number.

I pick the threshold that maximizes VALIDATION F1 (balanced), not precision alone,
because maximizing precision alone just makes the model silent and tanks recall. I
report BOTH precision and recall on test, averaged over seeds, whatever they come out to.
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

X_train_full = torch.tensor(np.load(f"{PROC}/X_train.npy"))
y_train_full = torch.tensor(np.load(f"{PROC}/y_train.npy"))
X_test = torch.tensor(np.load(f"{PROC}/X_test.npy"))
y_test = torch.tensor(np.load(f"{PROC}/y_test.npy"))
WINDOW = json.load(open(f"{PROC}/scaler.json"))["window"]

# Carve the last 20% of the training windows as VALIDATION (time-ordered, no shuffle).
n = len(X_train_full)
cut = int(n * 0.8)
X_tr, y_tr = X_train_full[:cut], y_train_full[:cut]
X_val, y_val = X_train_full[cut:], y_train_full[cut:]

# Build the true-label mask for the TEST set (same alignment as before).
df = pd.read_csv(RAW_CSV, parse_dates=["timestamp"])
split_idx = int(len(df) * 0.6)
test_times = df["timestamp"].to_numpy()[np.arange(len(X_test)) + split_idx + WINDOW]
windows = json.load(open(LABELS))[LABEL_KEY]
true_test = np.zeros(len(X_test), dtype=bool)
for s, e in windows:
    s, e = pd.to_datetime(s), pd.to_datetime(e)
    true_test |= (test_times >= np.datetime64(s)) & (test_times <= np.datetime64(e))

# The validation windows sit right before the test set in time. Their true labels:
# validation covers training indices [cut : n], which map to raw positions after WINDOW.
val_pred_idx = np.arange(cut, n) + WINDOW
val_times = df["timestamp"].to_numpy()[val_pred_idx]
true_val = np.zeros(len(X_val), dtype=bool)
for s, e in windows:
    s, e = pd.to_datetime(s), pd.to_datetime(e)
    true_val |= (val_times >= np.datetime64(s)) & (val_times <= np.datetime64(e))

def prf(flag, true):
    tp = int((flag & true).sum()); fp = int((flag & ~true).sum()); fn = int((~flag & true).sum())
    p = tp/(tp+fp) if (tp+fp) else 0.0
    r = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*p*r/(p+r) if (p+r) else 0.0
    return p, r, f1

def run(seed):
    set_seed(seed)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=128, shuffle=True)
    model = AnomalyTransformer(window=WINDOW).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lf = nn.MSELoss()
    for _ in range(15):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); lf(model(xb), yb).backward(); opt.step()

    def err(X, y):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 512):
                out.append((model(X[i:i+512].to(device)).cpu() - y[i:i+512]).abs())
        return torch.cat(out).numpy()

    val_err = err(X_val, y_val)
    test_err = err(X_test, y_test)

    # Find the threshold that maximizes VALIDATION F1 (balanced).
    best_thr, best_f1 = None, -1
    for pct in np.arange(90, 99.6, 0.5):
        thr = np.percentile(val_err, pct)
        _, _, f1 = prf(val_err > thr, true_val)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    # Also find the threshold that maximizes validation PRECISION (to show the tradeoff).
    best_thr_p, best_p = float(np.percentile(val_err, 99)), -1
    for pct in np.arange(90, 99.9, 0.2):
        thr = np.percentile(val_err, pct)
        p, r, _ = prf(val_err > thr, true_val)
        if p > best_p and r > 0.05:   # require SOME recall, not zero
            best_p, best_thr_p = p, thr

    p_bal, r_bal, f1_bal = prf(test_err > best_thr, true_test)
    p_hi, r_hi, f1_hi = prf(test_err > best_thr_p, true_test)
    return (p_bal, r_bal, f1_bal), (p_hi, r_hi, f1_hi)

seeds = [1, 2, 3, 4, 5]
bal, hi = [], []
for s in seeds:
    b, h = run(s)
    bal.append(b); hi.append(h)
    print(f"seed {s}: balanced P{b[0]:.3f} R{b[1]:.3f} F1{b[2]:.3f} | "
          f"precision-focused P{h[0]:.3f} R{h[1]:.3f} F1{h[2]:.3f}")

bal, hi = np.array(bal), np.array(hi)
print("\n=== TUNED ON VALIDATION, MEASURED ON TEST (5 seeds) ===")
print(f"Balanced (max val F1):     P {bal[:,0].mean():.3f}  R {bal[:,1].mean():.3f}  F1 {bal[:,2].mean():.3f}")
print(f"Precision-focused:         P {hi[:,0].mean():.3f}  R {hi[:,1].mean():.3f}  F1 {hi[:,2].mean():.3f}")
print("\nHonest: the precision-focused row shows what chasing precision does to recall.")
