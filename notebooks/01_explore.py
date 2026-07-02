"""
In this file I am exploring the NAB machine-temperature data before I build anything.

I want to actually see the data first: how big it is, what columns it has, the date
range, and what the anomalies look like on a plot.
"""
import matplotlib
matplotlib.use("Agg")

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CSV_PATH = "data/raw/machine_temp.csv"
LABELS_PATH = "data/raw/labels.json"
LABEL_KEY = "realKnownCause/machine_temperature_system_failure.csv"

df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])

print("=" * 45)
print("DATASET SUMMARY")
print("=" * 45)
print("Total readings (rows):", len(df))
print("Columns / features:", list(df.columns))
print("Number of columns:", df.shape[1])
print("Date range:", df["timestamp"].min(), "to", df["timestamp"].max())
print("Time span:", df["timestamp"].max() - df["timestamp"].min())
print("Reading interval:", df["timestamp"].diff().median())
print("Any missing values?", df.isnull().sum().to_dict())
print("-" * 45)
print("Value statistics:")
print(df["value"].describe())
print("=" * 45)

labels = json.load(open(LABELS_PATH))
windows = labels[LABEL_KEY]
print("Anomaly windows:", len(windows))
for start, end in windows:
    print("  ", start, "->", end)

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(df["timestamp"], df["value"], linewidth=0.6, color="#1769AA", label="temperature")
for start, end in windows:
    ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), color="red", alpha=0.25)

ax.set_title("NAB machine temperature (red = labeled anomaly windows)")
ax.set_xlabel("time")
ax.set_ylabel("temperature")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax.legend()
fig.tight_layout()

out = "notebooks/01_explore.png"
fig.savefig(out, dpi=110)
print("Saved plot to", out)
