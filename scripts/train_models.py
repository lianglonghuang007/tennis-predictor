"""Train final production models on the full dataset and persist them to disk.

Run once (or whenever the underlying data/features change) — NOT per API
request. The Optuna search that picks LightGBM's hyperparameters takes
several minutes; an API server should load an already-trained model, not
retrain on every prediction request.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

# This script lives in scripts/, not the project root, so the root (which
# holds the `src` package) isn't on sys.path by default — add it explicitly
# before importing anything from `src`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.build_features import build_feature_table
from src.models.baseline import train_baseline
from src.models.lightgbm_model import train_lightgbm

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Reused from the tuning run already validated in Week 1 (5 time-based CV
# folds on the training period, 20 Optuna trials). Re-running the full
# ~8-minute search here would very likely land on a near-identical result,
# since it's the exact same training data — so it's reused, not repeated.
LIGHTGBM_PARAMS = {
    "num_leaves": 66,
    "learning_rate": 0.010537269367072208,
    "min_child_samples": 49,
    "feature_fraction": 0.6650934114406604,
    "bagging_fraction": 0.8659540676940825,
    "bagging_freq": 1,
    "lambda_l1": 4.517400500695315,
    "lambda_l2": 1.633865844586511,
    "n_estimators": 281,
}


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    processed_path = project_root / "data" / "processed" / "atp_matches_combined.csv"
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    logger.info(f"Loading {processed_path}")
    df = pd.read_csv(processed_path, low_memory=False, parse_dates=["tourney_date"])

    logger.info("Building feature table")
    table = build_feature_table(df)

    # Train on ALL available data for the deployed model — the train/val/test
    # split exists to evaluate how good a model is, not to handicap the model
    # that actually gets shipped. Val/test already served their evaluation
    # purpose (see README.md for those results).
    logger.info(f"Training Bradley-Terry baseline on {len(table):,} matches")
    baseline_model = train_baseline(table)
    joblib.dump(baseline_model, models_dir / "baseline_pipeline.joblib")

    logger.info(f"Training LightGBM on {len(table):,} matches")
    lgbm_model = train_lightgbm(table, LIGHTGBM_PARAMS)
    joblib.dump(lgbm_model, models_dir / "lightgbm_model.joblib")

    logger.info(f"Saved models to {models_dir}/")


if __name__ == "__main__":
    main()
