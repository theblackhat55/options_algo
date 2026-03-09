from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OptionsSurfaceSnapshot:
    ticker: str
    spot_price: float
    total_contracts: int

    atm_iv: float
    near_atm_iv: float
    mid_atm_iv: float
    term_slope: float
    term_ratio: float

    call_iv_mean: float
    put_iv_mean: float
    skew_proxy: float

    put_call_oi_ratio: float
    put_call_volume_ratio: float

    avg_spread_pct: float
    median_spread_pct: float
    liquid_contract_ratio: float

    top_oi_strike: float
    top_oi_strike_distance_pct: float
    top_volume_strike: float
    top_volume_strike_distance_pct: float

    front_expiry_concentration: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_mean(series: pd.Series) -> float:
    s = _to_numeric(series).dropna()
    return float(s.mean()) if not s.empty else 0.0


def _safe_median(series: pd.Series) -> float:
    s = _to_numeric(series).dropna()
    return float(s.median()) if not s.empty else 0.0


def _safe_sum(series: pd.Series) -> float:
    s = _to_numeric(series).dropna()
    return float(s.sum()) if not s.empty else 0.0


def _nearest_atm_row(df: pd.DataFrame, spot_price: float) -> Optional[pd.Series]:
    if df.empty or "strike" not in df.columns:
        return None
    temp = df.copy()
    temp["strike_num"] = _to_numeric(temp["strike"])
    temp = temp.dropna(subset=["strike_num"])
    if temp.empty:
        return None
    temp["_atm_diff"] = (temp["strike_num"] - float(spot_price)).abs()
    idx = temp["_atm_diff"].idxmin()
    return temp.loc[idx]


def _expiry_groups(chain: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    if chain.empty or "expiration" not in chain.columns:
        return []
    groups = []
    for exp, g in chain.groupby("expiration"):
        if g is None or g.empty:
            continue
        groups.append((str(exp), g.copy()))
    groups.sort(key=lambda x: x[0])
    return groups


def _atm_iv_for_expiry(exp_df: pd.DataFrame, spot_price: float) -> float:
    row = _nearest_atm_row(exp_df, spot_price)
    if row is None:
        return 0.0
    iv = row.get("implied_volatility", 0.0)
    try:
        iv = float(iv)
    except Exception:
        return 0.0
    return iv * 100.0 if 0 < iv < 1 else iv


def _strike_distance_pct(strike: float, spot: float) -> float:
    if spot <= 0:
        return 0.0
    return ((strike - spot) / spot) * 100.0


def analyze_options_surface(
    ticker: str,
    spot_price: float,
    chain: pd.DataFrame,
    max_spread_pct_for_liquid: float = 5.0,
) -> OptionsSurfaceSnapshot:
    """
    Compute reusable options-surface summary metrics from the current chain.
    Safe for partial/incomplete chains.
    """
    if chain is None or chain.empty:
        return OptionsSurfaceSnapshot(
            ticker=ticker,
            spot_price=float(spot_price),
            total_contracts=0,
            atm_iv=0.0,
            near_atm_iv=0.0,
            mid_atm_iv=0.0,
            term_slope=0.0,
            term_ratio=0.0,
            call_iv_mean=0.0,
            put_iv_mean=0.0,
            skew_proxy=0.0,
            put_call_oi_ratio=0.0,
            put_call_volume_ratio=0.0,
            avg_spread_pct=0.0,
            median_spread_pct=0.0,
            liquid_contract_ratio=0.0,
            top_oi_strike=0.0,
            top_oi_strike_distance_pct=0.0,
            top_volume_strike=0.0,
            top_volume_strike_distance_pct=0.0,
            front_expiry_concentration=0.0,
        )

    df = chain.copy()

    for col in ["strike", "implied_volatility", "open_interest", "volume", "bid_ask_spread_pct"]:
        if col in df.columns:
            df[col] = _to_numeric(df[col])

    total_contracts = int(len(df))

    expiry_groups = _expiry_groups(df)
    near_atm_iv = 0.0
    mid_atm_iv = 0.0

    if len(expiry_groups) >= 1:
        near_atm_iv = _atm_iv_for_expiry(expiry_groups[0][1], spot_price)
    if len(expiry_groups) >= 2:
        mid_atm_iv = _atm_iv_for_expiry(expiry_groups[min(1, len(expiry_groups) - 1)][1], spot_price)
    elif len(expiry_groups) == 1:
        mid_atm_iv = near_atm_iv

    atm_iv = near_atm_iv if near_atm_iv > 0 else _safe_mean(df.get("implied_volatility", pd.Series(dtype=float)))
    if 0 < atm_iv < 1:
        atm_iv *= 100.0

    term_slope = mid_atm_iv - near_atm_iv if near_atm_iv > 0 and mid_atm_iv > 0 else 0.0
    term_ratio = (mid_atm_iv / near_atm_iv) if near_atm_iv > 0 else 0.0

    calls = df[df.get("type", pd.Series(dtype=object)).astype(str).str.lower() == "call"].copy()
    puts = df[df.get("type", pd.Series(dtype=object)).astype(str).str.lower() == "put"].copy()

    call_iv_mean = _safe_mean(calls.get("implied_volatility", pd.Series(dtype=float)))
    put_iv_mean = _safe_mean(puts.get("implied_volatility", pd.Series(dtype=float)))
    if 0 < call_iv_mean < 1:
        call_iv_mean *= 100.0
    if 0 < put_iv_mean < 1:
        put_iv_mean *= 100.0

    skew_proxy = put_iv_mean - call_iv_mean if put_iv_mean and call_iv_mean else 0.0

    call_oi = _safe_sum(calls.get("open_interest", pd.Series(dtype=float)))
    put_oi = _safe_sum(puts.get("open_interest", pd.Series(dtype=float)))
    call_vol = _safe_sum(calls.get("volume", pd.Series(dtype=float)))
    put_vol = _safe_sum(puts.get("volume", pd.Series(dtype=float)))

    put_call_oi_ratio = (put_oi / call_oi) if call_oi > 0 else 0.0
    put_call_volume_ratio = (put_vol / call_vol) if call_vol > 0 else 0.0

    avg_spread_pct = _safe_mean(df.get("bid_ask_spread_pct", pd.Series(dtype=float)))
    median_spread_pct = _safe_median(df.get("bid_ask_spread_pct", pd.Series(dtype=float)))

    liquid_contracts = df[df.get("bid_ask_spread_pct", pd.Series(dtype=float)).fillna(np.inf) <= max_spread_pct_for_liquid]
    liquid_contract_ratio = (len(liquid_contracts) / len(df)) if len(df) > 0 else 0.0

    top_oi_strike = 0.0
    top_oi_strike_distance_pct = 0.0
    if "open_interest" in df.columns and "strike" in df.columns and not df["open_interest"].dropna().empty:
        oi_idx = df["open_interest"].fillna(-1).idxmax()
        top_oi_strike = float(df.loc[oi_idx, "strike"]) if pd.notna(df.loc[oi_idx, "strike"]) else 0.0
        top_oi_strike_distance_pct = _strike_distance_pct(top_oi_strike, float(spot_price))

    top_volume_strike = 0.0
    top_volume_strike_distance_pct = 0.0
    if "volume" in df.columns and "strike" in df.columns and not df["volume"].dropna().empty:
        vol_idx = df["volume"].fillna(-1).idxmax()
        top_volume_strike = float(df.loc[vol_idx, "strike"]) if pd.notna(df.loc[vol_idx, "strike"]) else 0.0
        top_volume_strike_distance_pct = _strike_distance_pct(top_volume_strike, float(spot_price))

    front_expiry_concentration = 0.0
    if expiry_groups:
        front_expiry_contracts = len(expiry_groups[0][1])
        front_expiry_concentration = front_expiry_contracts / len(df) if len(df) > 0 else 0.0

    return OptionsSurfaceSnapshot(
        ticker=ticker,
        spot_price=float(spot_price),
        total_contracts=total_contracts,
        atm_iv=round(float(atm_iv), 4),
        near_atm_iv=round(float(near_atm_iv), 4),
        mid_atm_iv=round(float(mid_atm_iv), 4),
        term_slope=round(float(term_slope), 4),
        term_ratio=round(float(term_ratio), 4),
        call_iv_mean=round(float(call_iv_mean), 4),
        put_iv_mean=round(float(put_iv_mean), 4),
        skew_proxy=round(float(skew_proxy), 4),
        put_call_oi_ratio=round(float(put_call_oi_ratio), 4),
        put_call_volume_ratio=round(float(put_call_volume_ratio), 4),
        avg_spread_pct=round(float(avg_spread_pct), 4),
        median_spread_pct=round(float(median_spread_pct), 4),
        liquid_contract_ratio=round(float(liquid_contract_ratio), 4),
        top_oi_strike=round(float(top_oi_strike), 4),
        top_oi_strike_distance_pct=round(float(top_oi_strike_distance_pct), 4),
        top_volume_strike=round(float(top_volume_strike), 4),
        top_volume_strike_distance_pct=round(float(top_volume_strike_distance_pct), 4),
        front_expiry_concentration=round(float(front_expiry_concentration), 4),
    )
