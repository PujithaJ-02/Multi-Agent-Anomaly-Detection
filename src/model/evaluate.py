"""
In this file I am measuring whether the model actually catches anomalies.

Low training loss only told me the model predicts normal readings well. Here I run
it on the TEST data it never saw, turn each prediction error into an anomaly score,
pick a threshold, and check my flags against the real labeled anomalies to get
precision, recall and F1. These are the numbers that actually matter.

For the threshold I use the 99th percentile of the training errors, meaning I flag
anything more surprising than 99% of what the model saw as normal. I am NOT tuning
the threshold on the test set, because that would fake better numbers.
"""
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
model.eval()   # turn off training behavior, I am only predicting now

def scores(X, y):
    # I run the model with no training and take the absolute gap between the
    # prediction and the real next reading. That gap is my anomaly score.
    errs = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = X[i:i + 512].to(device)
            pred = model(xb).cpu()
            errs.append((pred - y[i:i + 512]).abs())
    return torch.cat(errs).numpy()

train_err = scores(X_train, y_train)
test_err = scores(X_test, y_test)

threshold = float(np.percentile(train_err, 99))
print("threshold:", round(threshold, 6))

# Now I figure out which real timestamp each test window is predicting, so I can
# check it against the labeled anomaly windows.
df = pd.read_csv(RAW_CSV, parse_dates=["timestamp"])
n_total = len(df)
split_idx = int(n_total * 0.6)   # same split the preprocess step used
test_pred_idx = np.arange(len(X_test)) + split_idx + WINDOW
test_times = df["timestamp"].to_numpy()[test_pred_idx]

# For each test point, is its timestamp inside any labeled anomaly window?
windows = json.load(open(LABELS))[LABEL_KEY]
true = np.zeros(len(X_test), dtype=bool)
for start, end in windows:
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    true |= (test_times >= np.datetime64(s)) & (test_times <= np.datetime64(e))

pred_flag = test_err > threshold

tp = int((pred_flag & true).sum())
fp = int((pred_flag & ~true).sum())
fn = int((~pred_flag & true).sum())
precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

print(f"true anomaly points in test: {int(true.sum())}")
print(f"flagged points: {int(pred_flag.sum())}")
print(f"TP {tp}  FP {fp}  FN {fn}")
print(f"precision {precision:.3f}  recall {recall:.3f}  F1 {f1:.3f}")

# Save a plot so I can SEE the errors, the threshold line, and the true windows.
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(test_times, test_err, linewidth=0.6, color="#1769AA", label="anomaly score")
ax.axhline(threshold, color="green", linestyle="--", label="threshold")
for start, end in windows:
    ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), color="red", alpha=0.25)
ax.set_title("Test anomaly scores (red = true anomaly, green = threshold)")
ax.legend()
fig.tight_layout()
fig.savefig("../../notebooks/02_eval.png", dpi=110)
print("saved plot to notebooks/02_eval.png")
