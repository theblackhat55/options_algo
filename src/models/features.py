"""
src/models/features.py
======================
Feature engineering for the ML predictor.
Converts trade outcomes DataFrame into a features matrix for training.

Features used:
    - IV rank, IV percentile, IV/HV ratio
    - ADX, RSI, trend strength, direction score
    - RS rank, sector (encoded)
    - DTE at entry, spread width, short delta
    - Market context (VIX proxy, breadth)
    - Binary flags: bb_squeeze, ema_alignment

Usage:
    from src.models.features import build_features

    X, y, feature_names = build_features(outcomes_df)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# Feature column definitions
NUMERIC_FEATURES = [
    "iv_rank", "iv_hv_ratio", "adx", "rsi", "trend_strength",
    "direction_score", "rs_rank", "dte_at_entry", "spread_width",
    "short_delta", "prob_profit", "confidence",
]

CATEGORICAL_FEATURES = [
    "regime", "iv_regime", "direction", "sector",
]

TARGET_COL = "won"


def build_features(
    df: pd.DataFrame,
    encode_categoricals: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Build feature matrix X and target y from outcomes DataFrame.

    Args:
        df: Trade outcomes DataFrame (from outcome_tracker.load_outcomes)
        encode_categoricals: If True, label-encode categorical features

    Returns:
        (X, y, feature_names) tuple
    """
    df = df.copy()

    # Drop rows with missing target
    df = df[df[TARGET_COL].notna()].copy()
    if df.empty:
        return pd.DataFrame(), pd.Series(), []

    y = df[TARGET_COL].astype(int)

    # Numeric features
    X_parts = []
    feature_names = []

    num_df = df[NUMERIC_FEATURES].copy()
    num_df = num_df.fillna(num_df.median())
    X_parts.append(num_df)
    feature_names.extend(NUMERIC_FEATURES)

    # Categorical features (label encoded)
    if encode_categoricals:
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                le = LabelEncoder()
                encoded = le.fit_transform(df[col].fillna("UNKNOWN"))
                X_parts.append(pd.Series(encoded, name=col, index=df.index))
                feature_names.append(col)

    # Derived features
    if "iv_rank" in df.columns and "dte_at_entry" in df.columns:
        iv_dte = df["iv_rank"] * df["dte_at_entry"] / 252
        iv_dte.name = "iv_rank_x_dte"
        X_parts.append(iv_dte.fillna(0))
        feature_names.append("iv_rank_x_dte")

    if "adx" in df.columns and "trend_strength" in df.columns:
        trend_comp = df["adx"] * df["trend_strength"]
        trend_comp.name = "adx_x_trend"
        X_parts.append(trend_comp.fillna(0))
        feature_names.append("adx_x_trend")

    X = pd.concat(X_parts, axis=1)
    X.columns = feature_names

    logger.info(f"Feature matrix: {X.shape} | Positive rate: {y.mean():.1%}")
    return X, y, feature_names
