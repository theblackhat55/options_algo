"""
config/settings.py
==================
Central configuration for options_algo.
API keys loaded from environment or .env file.
All paths are resolved relative to project root.
"""
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CALENDAR_DIR = DATA_DIR / "calendar"
OUTPUT_DIR = PROJECT_ROOT / "output"
SIGNALS_DIR = OUTPUT_DIR / "signals"
MODELS_DIR = OUTPUT_DIR / "models"
TRADES_DIR = OUTPUT_DIR / "trades"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Create all directories on import
for _d in [RAW_DIR, PROCESSED_DIR, CALENDAR_DIR, SIGNALS_DIR,
           MODELS_DIR, TRADES_DIR, REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── API Keys ─────────────────────────────────────────────────────────────────
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")           # Free tier: earnings calendar
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")        # Free tier: backup quotes
TRADIER_API_KEY = os.getenv("TRADIER_API_KEY", "")          # Backup options data
TRADIER_ACCOUNT_ID = os.getenv("TRADIER_ACCOUNT_ID", "")

# ─── Universe ─────────────────────────────────────────────────────────────────
UNIVERSE_SIZE = os.getenv("UNIVERSE_SIZE", "SP100")   # "SP100" or "SP500"

# ─── Liquidity Filters ────────────────────────────────────────────────────────
MIN_OPTION_VOLUME = int(os.getenv("MIN_OPTION_VOLUME", "500"))
MIN_OPEN_INTEREST = int(os.getenv("MIN_OPEN_INTEREST", "1000"))
MAX_BID_ASK_SPREAD_PCT = float(os.getenv("MAX_BID_ASK_SPREAD_PCT", "5.0"))
MIN_STOCK_PRICE = float(os.getenv("MIN_STOCK_PRICE", "20.0"))
MIN_AVG_VOLUME = int(os.getenv("MIN_AVG_VOLUME", "1000000"))

# ─── IV Parameters ────────────────────────────────────────────────────────────
IV_LOOKBACK_DAYS = int(os.getenv("IV_LOOKBACK_DAYS", "252"))   # 1 year
IV_HIGH_THRESHOLD = float(os.getenv("IV_HIGH_THRESHOLD", "70"))
IV_LOW_THRESHOLD = float(os.getenv("IV_LOW_THRESHOLD", "30"))
HV_WINDOW = int(os.getenv("HV_WINDOW", "20"))

# ─── Technical Parameters ─────────────────────────────────────────────────────
EMA_FAST = int(os.getenv("EMA_FAST", "10"))
EMA_MEDIUM = int(os.getenv("EMA_MEDIUM", "20"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "50"))
ADX_PERIOD = int(os.getenv("ADX_PERIOD", "14"))
ADX_TRENDING_THRESHOLD = float(os.getenv("ADX_TRENDING_THRESHOLD", "25"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "30"))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "70"))
BB_PERIOD = int(os.getenv("BB_PERIOD", "20"))
BB_STD = float(os.getenv("BB_STD", "2.0"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))

# ─── Strategy Parameters ──────────────────────────────────────────────────────
DEFAULT_DTE_PREMIUM_SELL = int(os.getenv("DEFAULT_DTE_PREMIUM_SELL", "45"))
DEFAULT_DTE_DIRECTIONAL = int(os.getenv("DEFAULT_DTE_DIRECTIONAL", "21"))
DEFAULT_DTE_BUTTERFLY = int(os.getenv("DEFAULT_DTE_BUTTERFLY", "30"))
DEFAULT_DTE_IC = int(os.getenv("DEFAULT_DTE_IC", "45"))

MAX_RISK_PER_TRADE_PCT = float(os.getenv("MAX_RISK_PER_TRADE_PCT", "2.0"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "5"))
PROFIT_TARGET_PCT = float(os.getenv("PROFIT_TARGET_PCT", "50"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "100"))

# Default spread parameters
DEFAULT_SPREAD_WIDTH = float(os.getenv("DEFAULT_SPREAD_WIDTH", "5.0"))
DEFAULT_SHORT_DELTA = float(os.getenv("DEFAULT_SHORT_DELTA", "0.25"))
IC_WING_DELTA = float(os.getenv("IC_WING_DELTA", "0.16"))

# ─── ML Parameters ────────────────────────────────────────────────────────────
WALK_FORWARD_MIN_TRAIN = int(os.getenv("WALK_FORWARD_MIN_TRAIN", "252"))
WALK_FORWARD_TEST = int(os.getenv("WALK_FORWARD_TEST", "21"))
WALK_FORWARD_STEP = int(os.getenv("WALK_FORWARD_STEP", "5"))
MIN_TRADES_FOR_ML = int(os.getenv("MIN_TRADES_FOR_ML", "200"))

# ─── Delivery ─────────────────────────────────────────────────────────────────
WHATSAPP_NUMBER = os.getenv("WHATSAPP_NUMBER", "")            # Set in .env

# ─── Directional Balance & Snap-back Guards (V2) ────────────────────────────
# Maximum fraction of MAX_POSITIONS allowed in one direction (BULLISH or BEARISH)
MAX_SAME_DIRECTION_PCT = float(os.getenv("MAX_SAME_DIRECTION_PCT", "60"))

# Minimum IV-RV spread (vol points) required to flag premium_rich = True.
# Below this threshold, "high IV" is just elevated realized vol — not true edge.
MIN_IV_RV_SPREAD_CREDIT = float(os.getenv("MIN_IV_RV_SPREAD_CREDIT", "5.0"))

# ATR-units threshold for snap-back regime detection (OVERSOLD/OVERBOUGHT regimes).
# If a stock moves more than this many ATRs in 5 days it qualifies for snap-back check.
SNAPBACK_ATR_THRESHOLD = float(os.getenv("SNAPBACK_ATR_THRESHOLD", "2.0"))

# 3-day ROC (%) needed to confirm the snap-back is actually underway.
# Must be positive (for OVERSOLD_BOUNCE) or negative (for OVERBOUGHT_DROP).
SNAPBACK_ROC_THRESHOLD = float(os.getenv("SNAPBACK_ROC_THRESHOLD", "1.0"))

# SPY 5-day return gate for directional trades (%).
# Bearish signals are skipped when SPY is above this; bullish when SPY is below -this.
SPY_DIRECTIONAL_GATE_PCT = float(os.getenv("SPY_DIRECTIONAL_GATE_PCT", "1.0"))

# ─── VIX Circuit Breaker Levels (V2) ─────────────────────────────────────────
# VIX threshold above which we move to CAUTION (credit/neutral only).
VIX_CAUTION_LEVEL = float(os.getenv("VIX_CAUTION_LEVEL", "28.0"))

# VIX threshold above which we move to DEFENSIVE (neutral-only, no new directional).
VIX_DEFENSIVE_LEVEL = float(os.getenv("VIX_DEFENSIVE_LEVEL", "35.0"))

# VIX threshold above which we halt all new trades (LIQUIDATION mode).
VIX_LIQUIDATION_LEVEL = float(os.getenv("VIX_LIQUIDATION_LEVEL", "45.0"))

# Rolling window (days) used to compute the VIX 5-day average for spike detection.
VIX_SPIKE_WINDOW = int(os.getenv("VIX_SPIKE_WINDOW", "5"))

# If VIX rises more than this % above its recent average, flag as a spike.
VIX_SPIKE_THRESHOLD_PCT = float(os.getenv("VIX_SPIKE_THRESHOLD_PCT", "25.0"))

# ─── Sector Concentration (V2) ───────────────────────────────────────────────
# Maximum number of picks allowed from the same GICS sector.
MAX_PER_SECTOR = int(os.getenv("MAX_PER_SECTOR", "2"))

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── IBKR Settings ─────────────────────────────────────────────────────────────
IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", "4002"))
IBKR_CLIENT_ID_OPTIONS = int(os.getenv("IBKR_CLIENT_ID_OPTIONS", "11"))
IBKR_ENABLED = os.getenv("IBKR_ENABLED", "true").lower() == "true"
IBKR_TIMEOUT = int(os.getenv("IBKR_TIMEOUT", "10"))
IBKR_MAX_CONTRACTS_PER_TICKER = int(os.getenv("IBKR_MAX_CONTRACTS_PER_TICKER", "50"))

# ─── Long Option Parameters (V3) ─────────────────────────────────────────────
# DTE for long calls/puts — enough time for the move, limits theta burn
DEFAULT_DTE_LONG_OPTION = int(os.getenv("DEFAULT_DTE_LONG_OPTION", "35"))

# Delta target for long call/put selection (0.60-0.70 = slightly ITM for higher prob)
LONG_OPTION_DELTA = float(os.getenv("LONG_OPTION_DELTA", "0.65"))

# Minimum confidence required to recommend a naked long option (higher bar than spreads)
LONG_OPTION_MIN_CONFIDENCE = float(os.getenv("LONG_OPTION_MIN_CONFIDENCE", "0.65"))

# Profit target for long options: close at 100% gain (option doubled)
LONG_OPTION_PROFIT_TARGET_PCT = float(os.getenv("LONG_OPTION_PROFIT_TARGET_PCT", "100"))

# Stop loss for long options: close at 50% loss of premium paid
LONG_OPTION_STOP_LOSS_PCT = float(os.getenv("LONG_OPTION_STOP_LOSS_PCT", "50"))

# Time stop: close position if DTE falls below this regardless of P/L
LONG_OPTION_TIME_STOP_DTE = int(os.getenv("LONG_OPTION_TIME_STOP_DTE", "10"))

# Maximum theta decay rate: daily_theta / premium must be < this value
LONG_OPTION_MAX_THETA_RATE = float(os.getenv("LONG_OPTION_MAX_THETA_RATE", "0.03"))

# Maximum % of total capital allocated to long options across all positions
LONG_OPTION_MAX_ALLOCATION_PCT = float(os.getenv("LONG_OPTION_MAX_ALLOCATION_PCT", "30"))

# IV Rank ceiling — never buy premium when IV rank is above this
LONG_OPTION_IV_RANK_CEILING = float(os.getenv("LONG_OPTION_IV_RANK_CEILING", "40"))

# ─── Technical Analysis Parameters (V3) ──────────────────────────────────────
# Support/Resistance: lookback period for volume-profile pivot detection
SR_LOOKBACK_DAYS = int(os.getenv("SR_LOOKBACK_DAYS", "60"))

# Number of price bins for volume profile clustering
SR_VOLUME_BINS = int(os.getenv("SR_VOLUME_BINS", "50"))

# Proximity threshold: price within this % of S/R is "near" a level
SR_PROXIMITY_PCT = float(os.getenv("SR_PROXIMITY_PCT", "1.5"))

# Volume confirmation: breakout requires volume > this multiple of 20d avg
BREAKOUT_VOLUME_MULTIPLIER = float(os.getenv("BREAKOUT_VOLUME_MULTIPLIER", "1.5"))

# RSI divergence: minimum number of bars to look back for swing detection
DIVERGENCE_LOOKBACK = int(os.getenv("DIVERGENCE_LOOKBACK", "14"))

# Volume climax: volume > this multiple of 20d avg = climax event
VOLUME_CLIMAX_MULTIPLIER = float(os.getenv("VOLUME_CLIMAX_MULTIPLIER", "3.0"))

# ─── Polygon IV Snapshot Store (V3) ──────────────────────────────────────────
IV_SNAPSHOT_DIR = PROCESSED_DIR / "iv_snapshots"
IV_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# Minimum days of IV snapshot history before using real data over proxy
IV_SNAPSHOT_MIN_HISTORY = int(os.getenv("IV_SNAPSHOT_MIN_HISTORY", "20"))
