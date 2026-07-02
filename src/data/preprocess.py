"""
In this file I am taking the raw temperature data and getting it ready for the model.

I am doing three things here:
1. I am splitting the data by time, not randomly. If I shuffle a time series, the
   model gets to peek at the future, which fakes good results. So I keep the early
   readings for training and the later ones for testing.
2. I am scaling the numbers down to roughly 0 to 1 so the model trains better, but
   I only use the training data to figure out the scaling. If I used everything, the
   test data would leak into training.
3. I am cutting the long line of numbers into windows of 64 readings each.
"""
import os
import json
import numpy as np
import pandas as pd

RAW_CSV = "data/raw/machine_temp.csv"
OUT_DIR = "data/processed"
WINDOW = 64          # 64 readings is about 5 hours since each reading is 5 min apart
TRAIN_FRAC = 0.6     # I am keeping the first 60% of time for training

df = pd.read_csv(RAW_CSV, parse_dates=["timestamp"])
values = df["value"].to_numpy(dtype="float32")

# Here I am splitting by time. Earlier readings train, later readings test.
split_idx = int(len(values) * TRAIN_FRAC)
train_raw = values[:split_idx]
test_raw = values[split_idx:]

# Now I am finding the min and max from the TRAINING data only, and I will use these
# same two numbers to scale the test data too. The test values might go a little
# outside 0 to 1 and that is fine, that is expected.
tmin, tmax = train_raw.min(), train_raw.max()

def scale(a):
    return (a - tmin) / (tmax - tmin)

train_s = scale(train_raw)
test_s = scale(test_raw)

# Here I am making the windows. For each example I take 64 readings as the input (X)
# and the very next reading as the answer (y). The model learns to guess the next
# value from the previous 64.
def make_windows(series, w):
    X, y = [], []
    for i in range(len(series) - w):
        X.append(series[i:i + w])
        y.append(series[i + w])
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")

X_train, y_train = make_windows(train_s, WINDOW)
X_test, y_test = make_windows(test_s, WINDOW)

print("train windows:", X_train.shape, " test windows:", X_test.shape)

# Now I am saving everything so the next file can just load it instead of redoing all this.
os.makedirs(OUT_DIR, exist_ok=True)
np.save(f"{OUT_DIR}/X_train.npy", X_train)
np.save(f"{OUT_DIR}/y_train.npy", y_train)
np.save(f"{OUT_DIR}/X_test.npy", X_test)
np.save(f"{OUT_DIR}/y_test.npy", y_test)
json.dump({"min": float(tmin), "max": float(tmax), "window": WINDOW},
          open(f"{OUT_DIR}/scaler.json", "w"))
print("saved to", OUT_DIR)