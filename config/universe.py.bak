"""
config/universe.py
==================
Stock universe definitions.
S&P 100 components with sector classification.
Includes ETF universe for sector analysis and premium selling.
"""
from __future__ import annotations

# ─── S&P 100 Universe with Sectors ───────────────────────────────────────────
SP100_TICKERS: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "AMZN": "Technology", "META": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "ADBE": "Technology", "CRM": "Technology",
    "CSCO": "Technology", "INTC": "Technology", "ORCL": "Technology",
    "QCOM": "Technology", "TXN": "Technology", "AMD": "Technology",
    "IBM": "Technology", "NOW": "Technology", "INTU": "Technology",

    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials",  "MS": "Financials",  "BLK": "Financials",
    "C": "Financials",   "SCHW": "Financials", "AXP": "Financials",
    "USB": "Financials", "BK": "Financials",   "COF": "Financials",

    # Healthcare
    "UNH": "Healthcare", "JNJ": "Healthcare",  "PFE": "Healthcare",
    "ABBV": "Healthcare","MRK": "Healthcare",  "LLY": "Healthcare",
    "TMO": "Healthcare", "ABT": "Healthcare",  "BMY": "Healthcare",
    "AMGN": "Healthcare","GILD": "Healthcare", "MDT": "Healthcare",
    "ISRG": "Healthcare","REGN": "Healthcare", "VRTX": "Healthcare",

    # Consumer Discretionary
    "TSLA": "Consumer Disc", "HD": "Consumer Disc",   "NKE": "Consumer Disc",
    "MCD": "Consumer Disc",  "SBUX": "Consumer Disc", "LOW": "Consumer Disc",
    "TJX": "Consumer Disc",  "BKNG": "Consumer Disc", "CMG": "Consumer Disc",

    # Consumer Staples
    "PG": "Consumer Staples",   "KO": "Consumer Staples",   "PEP": "Consumer Staples",
    "COST": "Consumer Staples", "WMT": "Consumer Staples",  "CL": "Consumer Staples",
    "MDLZ": "Consumer Staples", "KMB": "Consumer Staples",

    # Industrials
    "CAT": "Industrials", "BA": "Industrials",  "HON": "Industrials",
    "UPS": "Industrials", "RTX": "Industrials", "GE": "Industrials",
    "DE": "Industrials",  "LMT": "Industrials", "MMM": "Industrials",
    "UNP": "Industrials", "FDX": "Industrials",

    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "SLB": "Energy", "EOG": "Energy", "MPC": "Energy",

    # Communication Services
    "DIS": "Communication",  "CMCSA": "Communication", "NFLX": "Communication",
    "T": "Communication",    "VZ": "Communication",    "TMUS": "Communication",

    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "AEP": "Utilities",

    # Real Estate
    "AMT": "Real Estate", "PLD": "Real Estate", "EQIX": "Real Estate",

    # Materials
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
    "FCX": "Materials",

    # Index ETFs (for market context)
    "SPY": "Index", "QQQ": "Index", "IWM": "Index", "DIA": "Index",

    # Sector ETFs (for sector rotation analysis)
    "XLF": "Sector", "XLE": "Sector", "XLK": "Sector",
    "XLV": "Sector", "XLI": "Sector", "XLP": "Sector",
    "XLY": "Sector", "XLU": "Sector", "XLRE": "Sector",
    "XLC": "Sector", "XLB": "Sector",

    # Volatility proxy
    "VXX": "Volatility",
}

# ─── Premium Selling Universe (highest liquidity, tightest spreads) ───────────
PREMIUM_SELL_UNIVERSE: list[str] = [
    "SPY", "QQQ", "IWM",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AMD", "JPM", "BAC", "XOM", "GS", "NFLX", "AVGO", "CRM",
    "XLF", "XLE", "XLK", "XLV",
]

# ─── Directional Trade Universe (individual stocks) ───────────────────────────
DIRECTIONAL_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AMD", "AVGO", "CRM", "ADBE", "NOW", "INTU", "QCOM",
    "JPM", "GS", "MS", "BAC",
    "UNH", "LLY", "ABBV", "MRK",
    "XOM", "CVX", "COP",
    "NFLX", "DIS",
    "CAT", "DE", "BA",
    "COST", "HD", "LOW",
]

# ─── Sector ETF Map ───────────────────────────────────────────────────────────
SECTOR_ETF_MAP: dict[str, str] = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Consumer Disc": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Communication": "XLC",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}


def get_universe(size: str = "SP100") -> list[str]:
    """Return ticker list based on universe size."""
    if size == "SP100":
        return list(SP100_TICKERS.keys())
    # Future: load SP500 from CSV
    return list(SP100_TICKERS.keys())


def get_tradeable_universe(size: str = "SP100") -> list[str]:
    """Return only stocks (exclude ETFs) for individual stock analysis."""
    return [
        t for t, sector in SP100_TICKERS.items()
        if sector not in ("Index", "Sector", "Volatility")
    ]


def get_sector(ticker: str) -> str:
    """Return sector for a ticker."""
    return SP100_TICKERS.get(ticker, "Unknown")


def get_sector_etf(ticker: str) -> str | None:
    """Return the sector ETF for a given stock's sector."""
    sector = get_sector(ticker)
    return SECTOR_ETF_MAP.get(sector)


def get_tickers_by_sector(sector: str) -> list[str]:
    """Return all tickers in a given sector."""
    return [t for t, s in SP100_TICKERS.items() if s == sector]
