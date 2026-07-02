# Multi-Agent Anomaly Detection System

Real-time anomaly detection on a high-throughput event stream. A PyTorch
Transformer scores every event on a fast path. Flagged candidates are escalated
to a LangGraph multi-agent layer that reasons about and classifies each alert.

## Architecture
(two-speed diagram goes in docs/architecture.png)

## Metrics
TODO: fill in with numbers I measured myself and can reproduce.
- Precision / recall / F1: pending Step 2
- Throughput (events per second): pending Step 3
- Alert latency (baseline vs optimized): pending Step 6

## How to run
TODO: docker compose up (added in Step 8)

## Stack
Python, PyTorch, Kafka, LangChain, LangGraph, Dagster, Docker Compose
