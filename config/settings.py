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
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "your_finnhub_api_key_here")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "your_alpha_vantage_key_here")
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
WHATSAPP_NUMBER = os.getenv("WHATSAPP_NUMBER", "your_whatsapp_number_here")

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
