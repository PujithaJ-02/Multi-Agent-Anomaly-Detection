# Multi-Agent Anomaly Detection System

Real-time anomaly detection on streaming data using a PyTorch Transformer and a multi-agent pipeline.

## The idea

The system is designed to run at two speeds:

- **Fast path.** A Transformer model scores every incoming reading and cheaply filters out the normal ones. This is built to keep up with a heavy stream.
- **Slow path.** Only the small number of suspicious readings get escalated to a slower, multi-agent reasoning layer that decides what the anomaly is, how serious it is, and whether to raise an alert.

This split is what lets the system stay fast on a busy stream while still applying real judgment to the readings that actually matter.

## Current status

This project is being built step by step. This section describes honestly what works today versus what is still planned.

**Working now:**

- Data exploration and preparation on a real anomaly-detection dataset (the NAB machine-temperature stream).
- A PyTorch Transformer (encoder-based, forecasting approach) that learns normal behaviour and scores anomalies by how wrong its next-value prediction is.
- An honest evaluation pipeline: time-based train/test split (no future leakage), a threshold chosen from training errors (not tuned on the test set), and results measured across 5 random seeds so the numbers are reproducible and not a lucky single run.

**Planned (not built yet):**

- Kafka streaming to feed the model a live event stream.
- The multi-agent reasoning layer (LangChain and LangGraph) for the slow path.
- Dagster to orchestrate data prep, training, and scheduled retraining.
- Docker Compose to run the whole system with one command.

The repository name mentions "multi-agent" because that is the design goal. The agent layer is on the roadmap above and is not implemented yet.

## Results so far

Measured on held-out test data, averaged across 5 random seeds:

| Metric | Value |
|--------|-------|
| Precision | 0.76 (plus or minus 0.05) |
| Recall | 0.43 (plus or minus 0.04) |
| F1 | 0.55 (plus or minus 0.03) |

These are numbers I measured myself and can reproduce. Notes on how I got here:

- Training on data that accidentally included anomalies gave high precision (0.86) but poor recall (0.29). Removing the anomalies from the training data nearly doubled recall, at some cost to precision. That trade-off is intentional and documented.
- A single training run is noisy, so I report the mean and spread across seeds rather than one cherry-picked result.

Full reasoning and every decision are recorded in docs/PROJECT_LOG.md.

## Dataset

Numenta Anomaly Benchmark (NAB), the machine_temperature_system_failure stream: one temperature reading every 5 minutes from an industrial machine that eventually failed, with labeled anomaly periods. The data is small and slow, and is used here to learn the full pipeline. A larger, higher-throughput dataset is planned before the streaming claims are finalised.

The raw data is not committed. To fetch it:

    curl -sL "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv" -o data/raw/machine_temp.csv
    curl -sL "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json" -o data/raw/labels.json

## Tech stack

Python, PyTorch. Planned: Kafka, LangChain, LangGraph, Dagster, Docker Compose.

## Setup

This project uses uv (https://github.com/astral-sh/uv) for environment management.

    uv sync

## How to run (current pipeline)

    # 1. Explore the data
    uv run python notebooks/01_explore.py

    # 2. Prepare windowed data (cleaned training set)
    cd src/data && uv run python preprocess_clean.py && cd ../..

    # 3. Train the final model (reproducible, seeded)
    cd src/model && uv run python train_final.py && cd ../..

    # 4. Measure honestly across seeds
    cd src/model && uv run python experiment_seeds.py && cd ../..

## Project structure

    src/
      data/       data loading, windowing, cleaning
      model/      the Transformer, training, evaluation, experiments
    notebooks/    data exploration
    data/         raw and processed data (not committed)
    models/       saved weights (not committed)
    docs/         project log with every decision and result

## License

See LICENSE.
