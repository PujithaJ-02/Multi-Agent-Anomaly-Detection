"""
In this file I orchestrate the OFFLINE machine-learning lifecycle with Dagster.

Important honesty: Dagster does NOT run my live Kafka stream. My live detection (fast
path + agents) runs continuously as services. Dagster manages the offline lifecycle:
preparing data, training the model, evaluating it, and (later) retraining on a schedule.
Think of it as the maintenance crew that keeps the model fresh in the background.

I model three assets that depend on each other:
  processed_data -> trained_model -> evaluation_report
Dagster understands the chain and can rebuild them in order, on demand or on a schedule.
"""
import json
import subprocess
from pathlib import Path

from dagster import asset, Definitions, AssetExecutionContext

# Paths are relative to the project root when Dagster runs from there.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
PROC_DIR = PROJECT_ROOT / "data" / "processed_clean"


@asset
def processed_data(context: AssetExecutionContext) -> dict:
    # I rebuild the cleaned, windowed data by running my existing preprocess script.
    context.log.info("building cleaned + windowed data")
    subprocess.run(
        ["uv", "run", "python", "preprocess_clean.py"],
        cwd=str(PROJECT_ROOT / "src" / "data"),
        check=True,
    )
    # Report a small fact so Dagster shows something meaningful.
    import numpy as np
    x = np.load(PROC_DIR / "X_train.npy")
    context.add_output_metadata({"train_windows": int(x.shape[0])})
    return {"train_windows": int(x.shape[0])}


@asset(deps=[processed_data])
def trained_model(context: AssetExecutionContext) -> dict:
    # I train the final seeded model by running my existing training script.
    context.log.info("training the model")
    subprocess.run(
        ["uv", "run", "python", "train_final.py"],
        cwd=str(PROJECT_ROOT / "src" / "model"),
        check=True,
    )
    cfg = json.load(open(MODEL_DIR / "detector_config.json"))
    context.add_output_metadata({"threshold": cfg["threshold"], "window": cfg["window"]})
    return cfg


@asset(deps=[trained_model])
def evaluation_report(context: AssetExecutionContext) -> dict:
    # I re-run the honest multi-seed evaluation to produce current metrics.
    context.log.info("evaluating across seeds")
    result = subprocess.run(
        ["uv", "run", "python", "experiment_seeds.py"],
        cwd=str(PROJECT_ROOT / "src" / "model"),
        check=True, capture_output=True, text=True,
    )
    # Save the raw output so the report is inspectable.
    report_path = MODEL_DIR / "eval_report.txt"
    report_path.write_text(result.stdout)
    context.log.info(result.stdout)
    context.add_output_metadata({"report_saved_to": str(report_path)})
    return {"saved": str(report_path)}


defs = Definitions(assets=[processed_data, trained_model, evaluation_report])
