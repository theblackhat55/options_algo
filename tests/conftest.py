"""
tests/conftest.py
=================
Shared pytest fixtures and path setup.
Ensures the project root is on sys.path for all tests.
"""
import sys
import os
from pathlib import Path

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
