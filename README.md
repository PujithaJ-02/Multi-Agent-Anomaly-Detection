# Multi-Agent Anomaly Detection System

Real-time anomaly detection on a streaming data pipeline. A PyTorch Transformer scores every reading on a fast path, and flagged anomalies are escalated to a LangGraph multi-agent layer that reasons about them before raising an alert.

## The idea: two speeds

The system runs at two speeds, and this split is the core design.

- Fast path: a Transformer scores every incoming reading and cheaply flags the suspicious ones. Built for throughput. Measured at about 1.3 ms per reading, and about 2,700 readings per second with batched scoring.
- Slow path: only flagged readings are handed off (through a background queue) to a team of agents that classify the anomaly, judge its severity, and decide whether to alert. This uses a language model, so it is slower, about 0.65 s per anomaly when warm.

Running the language model only on the small number of flagged readings, instead of every reading, is what keeps the system fast while still applying real reasoning where it matters.

## What it does

1. A producer streams temperature readings into Kafka, one at a time.
2. A consumer (the fast path) keeps a rolling window of the last 64 readings, runs the Transformer, and flags readings whose prediction error crosses a threshold.
3. Flagged readings go into a background queue. A worker runs a LangGraph agent graph on each: context, then classify (spike, drop, drift), then severity (low, medium, high), then a branch: high or medium severity raises an alert, low is logged only.
4. Dagster orchestrates the offline lifecycle: preparing data, training, and evaluating the model on a schedule. It does not run the live stream.

## Results (measured, reproducible)

Model performance on held-out test data, averaged across 5 random seeds:

| Metric | Value |
|--------|-------|
| Precision | 0.76 (plus or minus 0.05) |
| Recall | 0.43 (plus or minus 0.04) |
| F1 | 0.55 (plus or minus 0.03) |

Performance, measured end to end:

| Stage | Result |
|-------|--------|
| Fast path throughput | about 2,700 readings/second (batched); 787/second single-item |
| Fast path latency (score and flag) | about 1.3 ms per reading (warm) |
| Slow path latency (full agent alert) | about 0.65 s per anomaly (warm), one-time cold start of a few seconds |

Notes on how these were obtained, in the spirit of not overclaiming:

- Training on data that accidentally included anomalies gave high precision (0.86) but poor recall (0.29). Removing the anomalies from the training data raised recall to about 0.57, at some cost to precision. That trade-off is intentional.
- A single training run is noisy, so I report the mean and spread across 5 seeds rather than one lucky run.
- Throughput was raised from 787 to about 2,700 readings/second by batching model inference (scoring many windows in one call instead of one at a time), which cuts per-call overhead. Batching trades a little latency for throughput; the batch size is tunable.
- The reported "before" baseline for latency (a batch job leaving a multi-minute detection gap) is a framing of a typical batch interval, not a measured number. The streaming latencies above are measured.

## Architecture and honest boundaries

- Kafka and the producer run in Docker (see below).
- The consumer runs on the host machine, not in a container, for two honest reasons: its agents call a local LLM (Ollama) running on the host, and the model uses Apple MPS (the Mac GPU), which a Linux container cannot access. Containers reach Kafka at kafka:29092; host tools reach it at localhost:9092.

## Tech stack

Python, PyTorch, Apache Kafka, LangChain, LangGraph, Dagster, Docker Compose, Ollama (local LLM: llama3.2).

## Requirements

- Docker Desktop (running)
- uv (https://github.com/astral-sh/uv) for the Python environment
- Ollama (https://ollama.com) with the llama3.2 model pulled, for the agent layer:

      ollama pull llama3.2

## Setup

      uv sync

Fetch the dataset (not committed):

      curl -sL "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv" -o data/raw/machine_temp.csv
      curl -sL "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json" -o data/raw/labels.json

## How to run

Start Kafka and the producer (streams the data automatically):

      docker compose up --build

In another terminal, run the consumer (fast path plus agent slow path) on the host:

      cd src/streaming && uv run python consumer.py

Measure fast-path throughput (single-item vs batched):

      cd src/streaming && uv run python measure_throughput.py && cd ../..
      cd src/streaming && uv run python measure_throughput_batched.py && cd ../..

To retrain or evaluate through the orchestrator, launch the Dagster UI:

      cd src/orchestration && uv run dagster dev -f definitions.py

Then open http://127.0.0.1:3000 and materialize the assets.

## Training and evaluation (offline)

      cd src/data && uv run python preprocess_clean.py && cd ../..
      cd src/model && uv run python train_final.py && cd ../..
      cd src/model && uv run python experiment_seeds.py && cd ../..

## Dataset

Numenta Anomaly Benchmark (NAB), the machine_temperature_system_failure stream: one temperature reading every 5 minutes from an industrial machine that eventually failed, with labeled anomaly periods. Small and slow, used here to build and learn the full pipeline. A larger, higher-throughput dataset would be the next step to stress the throughput claims.

## Project structure

      src/
        data/          data loading, windowing, cleaning
        model/         the Transformer, training, evaluation
        streaming/     Kafka producer, consumer (fast path + queue), throughput benchmarks
        agents/        the LangGraph agent graph (slow path)
        orchestration/ Dagster definitions
      docker/          Dockerfile for the producer
      docker-compose.yml
      notebooks/       data exploration

## License

See LICENSE.