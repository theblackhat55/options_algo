#!/usr/bin/env python3
"""
scripts/retrain_models.py
=========================
Weekly ML model retrain.
Trains one LightGBM model per strategy with sufficient outcome data.

Usage:
    python scripts/retrain_models.py
    python scripts/retrain_models.py --strategy BULL_PUT_SPREAD
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import MIN_TRADES_FOR_ML
from src.models.trainer import train_all_models, train_model
from src.pipeline.outcome_tracker import load_outcomes, get_win_rate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # Print current stats
    df = load_outcomes(only_closed=True)
    logger.info(f"Total closed trades: {len(df)}")

    if df.empty:
        logger.info(f"No outcomes yet. Need {MIN_TRADES_FOR_ML} to activate ML.")
        return

    # Win rates by strategy
    strategies = df["strategy"].unique() if not df.empty else []
    for strat in strategies:
        stats = get_win_rate(strat)
        logger.info(
            f"  {strat}: {stats['count']} trades | "
            f"Win rate: {stats.get('win_rate', 'N/A')}% | "
            f"Avg P&L: ${stats.get('avg_pnl', 0):.2f}"
        )

    # Single strategy mode
    if "--strategy" in sys.argv:
        idx = sys.argv.index("--strategy")
        if idx + 1 < len(sys.argv):
            strat = sys.argv[idx + 1]
            strat_df = df[df["strategy"] == strat] if not df.empty else df
            result = train_model(strat, strat_df)
            logger.info(f"Training result: {result}")
            return

    # Train all
    results = train_all_models()
    logger.info("\nTraining summary:")
    for strategy, result in results.items():
        logger.info(f"  {strategy}: {result}")


if __name__ == "__main__":
    main()
