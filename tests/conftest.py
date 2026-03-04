"""
tests/conftest.py
=================
Shared pytest fixtures and path setup.
Ensures the project root is on sys.path for all tests.
"""
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to sys.path so imports work without installation
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Set minimal env vars so settings.py doesn't fail
os.environ.setdefault("POLYGON_API_KEY", "")
os.environ.setdefault("FINNHUB_API_KEY", "test_key")
os.environ.setdefault("ALPHA_VANTAGE_KEY", "test_key")
os.environ.setdefault("TRADIER_API_KEY", "")
os.environ.setdefault("LOG_LEVEL", "WARNING")

# ── Stub optional heavy/unavailable packages so modules import cleanly ────────
# These are only needed at runtime (network calls) — tests patch them out.
# yfinance gets a real ModuleSpec so pandas_ta_classic._meta's find_spec() works.
import importlib.machinery
from unittest.mock import MagicMock
_yf_mock = MagicMock()
_yf_mock.__spec__ = importlib.machinery.ModuleSpec("yfinance", loader=None)
sys.modules.setdefault("yfinance", _yf_mock)
for _pkg in ("xgboost", "lightgbm", "catboost", "polygon"):
    sys.modules.setdefault(_pkg, MagicMock())
