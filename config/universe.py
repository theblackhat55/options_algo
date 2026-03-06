"""
config/universe.py
==================
Stock universe definitions.
Liquidity-filtered options universe based on actual Polygon/Massive flat-file data.
Minimum avg daily option volume ~10,000+ for stocks, ~5,000+ for ETFs.
Updated 2026-03-06 using 3-day average from options flat files.
"""
from __future__ import annotations

# ─── Liquidity-Filtered Options Universe with Sectors ─────────────────────────
SP100_TICKERS: dict[str, str] = {
    # Technology — all high liquidity
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "AMZN": "Technology", "META": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "ADBE": "Technology", "CRM": "Technology",
    "CSCO": "Technology", "INTC": "Technology", "ORCL": "Technology",
    "QCOM": "Technology", "AMD": "Technology", "IBM": "Technology",
    "INTU": "Technology", "NOW": "Technology",
    # New tech additions (high liquidity)
    "PLTR": "Technology", "MU": "Technology", "TSM": "Technology",
    "CRWD": "Technology", "PANW": "Technology", "DELL": "Technology",
    "SHOP": "Technology", "SNOW": "Technology",

    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "C": "Financials",
    "SCHW": "Financials", "AXP": "Financials", "COF": "Financials",
    # New financial additions
    "COIN": "Financials", "HOOD": "Financials", "SOFI": "Financials",
    "PYPL": "Financials", "BRKB": "Financials",

    # Healthcare
    "UNH": "Healthcare", "JNJ": "Healthcare", "PFE": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "LLY": "Healthcare",
    "ABT": "Healthcare", "BMY": "Healthcare", "AMGN": "Healthcare",
    "GILD": "Healthcare",
    # New healthcare
    "MRNA": "Healthcare",

    # Consumer Discretionary
    "TSLA": "Consumer Disc", "HD": "Consumer Disc", "NKE": "Consumer Disc",
    "MCD": "Consumer Disc", "SBUX": "Consumer Disc", "LOW": "Consumer Disc",
    "BKNG": "Consumer Disc", "CMG": "Consumer Disc",
    # New consumer disc
    "UBER": "Consumer Disc", "RIVN": "Consumer Disc",

    # Consumer Staples
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "COST": "Consumer Staples", "WMT": "Consumer Staples",

    # Industrials
    "CAT": "Industrials", "BA": "Industrials", "DE": "Industrials",
    "GE": "Industrials", "RTX": "Industrials", "FDX": "Industrials",
    "UPS": "Industrials", "LMT": "Industrials", "HON": "Industrials",
    # New industrials
    "AAL": "Industrials", "DAL": "Industrials", "UAL": "Industrials",

    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "SLB": "Energy", "EOG": "Energy", "MPC": "Energy",
    # New energy
    "OXY": "Energy", "HAL": "Energy",

    # Communication Services
    "DIS": "Communication", "CMCSA": "Communication", "NFLX": "Communication",
    "T": "Communication", "VZ": "Communication",
    # New communication
    "BABA": "Communication", "SNAP": "Communication",

    # Crypto/Digital Assets (new sector)
    "MSTR": "Crypto", "IBIT": "Crypto",

    # Materials
    "FCX": "Materials",

    # Index ETFs
    "SPY": "Index", "QQQ": "Index", "IWM": "Index", "DIA": "Index",

    # Sector ETFs (keep liquid ones only)
    "XLF": "Sector", "XLE": "Sector", "XLK": "Sector",
    "XLP": "Sector", "XLB": "Sector",

    # Macro ETFs (new — uncorrelated exposure)
    "GLD": "Commodity", "SLV": "Commodity", "TLT": "Bond",

    # Volatility proxy
    "VXX": "Volatility",
}

# ─── Removed due to low options liquidity (<5000 avg daily volume) ────────────
# TMO, TMUS, ISRG, CL, MDLZ, TJX, VRTX, DUK, SO, LIN, UNP, BLK,
# PLD, AMT, AEP, REGN, SHW, BK, XLRE, EQIX, APD, USB, KMB, MMM,
# TXN, MDT, NEE, XLV, XLI, XLC, XLY, NKE kept (borderline but iconic)

# ─── Premium Selling Universe (highest liquidity, tightest spreads) ───────────
PREMIUM_SELL_UNIVERSE: list[str] = [
    "SPY", "QQQ", "IWM",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AMD", "JPM", "BAC", "XOM", "GS", "NFLX", "AVGO", "CRM",
    "PLTR", "SOFI", "MU", "COIN", "MSTR", "HOOD",
    "XLF", "XLE", "XLK",
    "GLD", "SLV", "TLT",
]

# ─── Directional Trade Universe (individual stocks) ───────────────────────────
DIRECTIONAL_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AMD", "AVGO", "CRM", "ADBE", "NOW", "INTU", "QCOM",
    "PLTR", "MU", "TSM", "CRWD", "PANW", "DELL", "SHOP",
    "JPM", "GS", "MS", "BAC", "COIN", "HOOD", "SOFI", "PYPL",
    "UNH", "LLY", "ABBV", "MRK", "MRNA",
    "XOM", "CVX", "COP", "OXY",
    "NFLX", "DIS", "BABA",
    "CAT", "DE", "BA",
    "COST", "HD", "LOW",
    "TSLA", "UBER", "MSTR",
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
    "Crypto": "IBIT",
    "Commodity": "GLD",
    "Bond": "TLT",
}


def get_universe(size: str = "SP100") -> list[str]:
    """Return ticker list based on universe size."""
    if size == "SP100":
        return list(SP100_TICKERS.keys())
    return list(SP100_TICKERS.keys())


def get_tradeable_universe(size: str = "SP100") -> list[str]:
    """Return only stocks (exclude ETFs) for individual stock analysis."""
    return [
        t for t, sector in SP100_TICKERS.items()
        if sector not in ("Index", "Sector", "Volatility", "Commodity", "Bond")
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
