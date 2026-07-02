# Design Note

## Problem
What anomaly am I detecting, and on what data?
(Leave a placeholder. We fill this in Step 1 when we pick the dataset.)

## The two-speed architecture
- Fast path: what runs on EVERY event, and why it must stay cheap.
- Slow path: what runs ONLY on flagged candidates, and why it is too slow
  to run on everything.
- One sentence: how do I get high throughput AND low latency at the same time?

## What each agent decides
List the agent nodes and the single job each one has.

## What Dagster does, and what it does NOT
Be precise. It runs the offline data and training lifecycle, not the live stream.

## Success metrics and how I measure them
Precision, recall, throughput, latency. One line each on how it is computed.

## Out of scope
What am I deliberately NOT building, so the project stays finishable?
