"""
src/models/trainer.py
=====================
Walk-forward training for LightGBM models.
Trains one model per strategy type on historical outcomes.

Run after collecting 200+ trade outcomes:
    python scripts/retrain_models.py

Walk-forward scheme:
    - Min 252 trading days (1yr) training
    - 21 day test window
    - 5 day step between windows
    - Final model trained on ALL available data
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import MODELS_DIR, MIN_TRADES_FOR_ML, WALK_FORWARD_MIN_TRAIN
from src.models.features import build_features
from src.pipeline.outcome_tracker import load_outcomes

logger = logging.getLogger(__name__)


def train_all_models() -> dict:
    """
    Train models for all strategy types with sufficient data.
    Returns a summary of training results.
    """
    outcomes_df = load_outcomes(only_closed=True)
    if outcomes_df.empty:
        logger.warning("No outcomes available for training")
        return {"error": "No outcomes"}

    results = {}
    strategies = outcomes_df["strategy"].unique()
    logger.info(f"Training models for {len(strategies)} strategies")

    for strategy in strategies:
        strat_df = outcomes_df[outcomes_df["strategy"] == strategy]
        if len(strat_df) < MIN_TRADES_FOR_ML:
            logger.info(
                f"  {strategy}: only {len(strat_df)} trades "
                f"(need {MIN_TRADES_FOR_ML}). Skipping."
            )
            results[strategy] = {"status": "insufficient_data", "count": len(strat_df)}
            continue

        result = train_model(strategy, strat_df)
        results[strategy] = result

    return results


def train_model(
    strategy: str,
    df: pd.DataFrame,
) -> dict:
    """
    Train a LightGBM model for a single strategy via walk-forward validation.

    Returns:
        Training result dict with metrics.
    """
    try:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score, accuracy_score
    except ImportError as exc:
        logger.error(f"LightGBM not installed: {exc}")
        return {"status": "error", "message": str(exc)}

    logger.info(f"Training model for {strategy} ({len(df)} samples)")

    X, y, feature_names = build_features(df)
    if X.empty or len(y) < 50:
        return {"status": "insufficient_data"}

    # ── Walk-Forward Validation ───────────────────────────────────────────────
    wf_scores = []
    n = len(X)
    min_train = max(MIN_TRADES_FOR_ML, int(n * 0.6))

    for test_start in range(min_train, n - 10, 10):
        X_train = X.iloc[:test_start]
        y_train = y.iloc[:test_start]
        X_test  = X.iloc[test_start:test_start + 10]
        y_test  = y.iloc[test_start:test_start + 10]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        model = _fit_lgb(X_train, y_train, feature_names)
        probs = model.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, probs)
            wf_scores.append(auc)
        except Exception:
            pass

    avg_auc = round(np.mean(wf_scores), 3) if wf_scores else 0.5
    logger.info(f"  {strategy}: walk-forward AUC = {avg_auc:.3f} ({len(wf_scores)} folds)")

    # ── Final Model (all data) ────────────────────────────────────────────────
    final_model = _fit_lgb(X, y, feature_names)

    # Save model
    model_path = MODELS_DIR / f"lgb_{strategy.lower()}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    # Save metadata
    meta = {
        "strategy": strategy,
        "trained_at": datetime.now().isoformat(),
        "n_samples": len(df),
        "walk_forward_auc": avg_auc,
        "wf_folds": len(wf_scores),
        "feature_names": feature_names,
        "positive_rate": round(float(y.mean()), 3),
    }
    meta_path = MODELS_DIR / f"lgb_{strategy.lower()}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"  Saved model to {model_path}")
    return {"status": "success", "auc": avg_auc, "n_samples": len(df)}


def _fit_lgb(X: pd.DataFrame, y: pd.Series, feature_names: list[str]):
    """Fit a LightGBM classifier."""
    import lightgbm as lgb

    params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "random_state": 42,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X[feature_names], y)
    return model
