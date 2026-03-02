"""
src/models/predictor.py
=======================
ML-based probability of profit predictor.
Replaces the rules-based confidence score once 200+ outcomes are available.

Uses LightGBM with walk-forward cross-validation.
Separate model per strategy type.

Usage:
    from src.models.predictor import load_predictor, predict_win_prob

    predictor = load_predictor("BULL_PUT_SPREAD")
    prob = predict_win_prob(predictor, features_row)
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import MODELS_DIR, MIN_TRADES_FOR_ML

logger = logging.getLogger(__name__)


def load_predictor(strategy: str) -> Optional[object]:
    """
    Load a trained LightGBM model for a strategy.
    Returns None if no model is available (use rules-based fallback).
    """
    path = MODELS_DIR / f"lgb_{strategy.lower()}.pkl"
    if not path.exists():
        logger.debug(f"No ML model for {strategy}")
        return None
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded ML model: {strategy}")
        return model
    except Exception as exc:
        logger.warning(f"Failed to load model for {strategy}: {exc}")
        return None


def predict_win_prob(
    model: object,
    features: pd.Series,
    feature_names: list[str],
) -> float:
    """
    Predict probability of win using trained model.

    Returns:
        Float 0.0–1.0 probability of profitable trade.
    """
    if model is None:
        return 0.5

    try:
        X = features[feature_names].fillna(0).values.reshape(1, -1)
        prob = model.predict_proba(X)[0][1]
        return round(float(prob), 3)
    except Exception as exc:
        logger.debug(f"Prediction failed: {exc}")
        return 0.5


def is_ml_ready(strategy: str = None) -> bool:
    """Check if ML model is available for a strategy."""
    if strategy:
        return (MODELS_DIR / f"lgb_{strategy.lower()}.pkl").exists()
    # Check if any model exists
    return any(MODELS_DIR.glob("lgb_*.pkl"))


def get_model_info(strategy: str) -> dict:
    """Return metadata about a trained model."""
    meta_path = MODELS_DIR / f"lgb_{strategy.lower()}_meta.json"
    if not meta_path.exists():
        return {"available": False, "strategy": strategy}
    import json
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        meta["available"] = True
        return meta
    except Exception:
        return {"available": False, "strategy": strategy}
