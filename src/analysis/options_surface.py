from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class OptionsSurfaceSnapshot:
    ticker: str
    spot_price: float

    atm_iv: float = 0.0
    near_atm_iv: float = 0.0
    mid_atm_iv: float = 0.0

    term_slope: float = 0.0
    term_ratio: float = 0.0

    call_iv_mean: float = 0.0
    put_iv_mean: float = 0.0
    skew_proxy: float = 0.0

    put_call_oi_ratio: float = 0.0
    put_call_volume_ratio: float = 0.0

    avg_spread_pct: float = 0.0
    median_spread_pct: float = 0.0
    liquid_contract_ratio: float = 0.0

    valid_quote_ratio: float = 0.0
    valid_spread_count: int = 0
    spread_sample_size: int = 0

    top_oi_strike: float = 0.0
    top_oi_strike_distance_pct: float = 0.0
    top_volume_strike: float = 0.0
    top_volume_strike_distance_pct: float = 0.0

    front_expiry_concentration: float = 0.0
    total_contracts: int = 0
    quote_source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_numeric(series: pd.Series | Any) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(pd.Series(series), errors="coerce")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if not np.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_mean(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return 0.0
    series = _to_numeric(series).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) == 0:
        return 0.0
    return _safe_float(series.mean(), 0.0)


def _compute_mid(df: pd.DataFrame) -> pd.Series:
    if "mid" in df.columns:
        mid = _to_numeric(df["mid"])
        if len(mid) > 0:
            return mid
    bid = _to_numeric(df["bid"]) if "bid" in df.columns else pd.Series(0.0, index=df.index)
    ask = _to_numeric(df["ask"]) if "ask" in df.columns else pd.Series(0.0, index=df.index)
    return (bid + ask) / 2.0


def _valid_quote_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty or "bid" not in df.columns or "ask" not in df.columns:
        return pd.Series(False, index=df.index)

    bid = _to_numeric(df["bid"])
    ask = _to_numeric(df["ask"])
    mid = _compute_mid(df)

    mask = (
        bid.notna()
        & ask.notna()
        & mid.notna()
        & (bid > 0)
        & (ask > 0)
        & (ask >= bid)
        & (mid > 0)
    )
    return mask.fillna(False)


def _safe_expiration_to_naive(series: pd.Series) -> pd.Series:
    exp = pd.to_datetime(series, errors="coerce")

    try:
        if getattr(exp.dt, "tz", None) is not None:
            exp = exp.dt.tz_convert(None)
    except Exception:
        try:
            exp = exp.dt.tz_localize(None)
        except Exception:
            pass

    return exp


def _normalize_chain(chain: pd.DataFrame) -> pd.DataFrame:
    if chain is None or chain.empty:
        return pd.DataFrame()

    df = chain.copy()

    rename_map: dict[str, str] = {}
    if "contract_type" not in df.columns and "option_type" in df.columns:
        rename_map["option_type"] = "contract_type"
    if "contract_type" not in df.columns and "type" in df.columns:
        rename_map["type"] = "contract_type"

    if "strike" not in df.columns and "strike_price" in df.columns:
        rename_map["strike_price"] = "strike"
    if "expiration" not in df.columns and "expiration_date" in df.columns:
        rename_map["expiration_date"] = "expiration"

    if "implied_volatility" not in df.columns and "iv" in df.columns:
        rename_map["iv"] = "implied_volatility"

    if rename_map:
        df = df.rename(columns=rename_map)

    for col in [
        "strike",
        "bid",
        "ask",
        "last",
        "mid",
        "implied_volatility",
        "open_interest",
        "volume",
        "underlying_price",
        "dte",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "contract_type" in df.columns:
        df["contract_type"] = df["contract_type"].astype(str).str.upper().str.strip()

    if "expiration" in df.columns:
        df["expiration"] = _safe_expiration_to_naive(df["expiration"])

    if "expiration" in df.columns and "dte" not in df.columns:
        today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        df["dte"] = (df["expiration"] - today).dt.days.astype(float)

    return df


def _atm_subset(df: pd.DataFrame, spot_price: float, width: float = 0.05) -> pd.DataFrame:
    if df.empty or "strike" not in df.columns or spot_price <= 0:
        return pd.DataFrame()
    strike = _to_numeric(df["strike"])
    dist = (strike - float(spot_price)).abs() / max(float(spot_price), 1e-9)
    return df.loc[dist <= width].copy()


def _term_metrics(df: pd.DataFrame) -> tuple[float, float]:
    if df.empty or "dte" not in df.columns or "implied_volatility" not in df.columns:
        return 0.0, 0.0

    sub = df.copy()
    sub["dte"] = _to_numeric(sub["dte"])
    sub["implied_volatility"] = _to_numeric(sub["implied_volatility"])
    sub = sub.dropna(subset=["dte", "implied_volatility"])

    if sub.empty:
        return 0.0, 0.0

    front = sub[(sub["dte"] >= 7) & (sub["dte"] <= 45)]
    back = sub[(sub["dte"] > 45) & (sub["dte"] <= 120)]

    front_iv = _safe_mean(front["implied_volatility"]) if not front.empty else 0.0
    back_iv = _safe_mean(back["implied_volatility"]) if not back.empty else 0.0

    term_slope = back_iv - front_iv if (front_iv or back_iv) else 0.0
    term_ratio = (back_iv / front_iv) if front_iv > 0 else 0.0
    return _safe_float(term_slope), _safe_float(term_ratio)


def _skew_metrics(df: pd.DataFrame) -> tuple[float, float, float]:
    if df.empty or "contract_type" not in df.columns or "implied_volatility" not in df.columns:
        return 0.0, 0.0, 0.0

    iv = _to_numeric(df["implied_volatility"])
    calls = iv[df["contract_type"] == "CALL"]
    puts = iv[df["contract_type"] == "PUT"]

    call_iv_mean = _safe_mean(calls)
    put_iv_mean = _safe_mean(puts)
    skew_proxy = put_iv_mean - call_iv_mean
    return call_iv_mean, put_iv_mean, skew_proxy


def _put_call_ratios(df: pd.DataFrame) -> tuple[float, float]:
    if df.empty or "contract_type" not in df.columns:
        return 0.0, 0.0

    oi = _to_numeric(df["open_interest"]) if "open_interest" in df.columns else pd.Series(0.0, index=df.index)
    volume = _to_numeric(df["volume"]) if "volume" in df.columns else pd.Series(0.0, index=df.index)

    put_oi = _safe_float(oi[df["contract_type"] == "PUT"].sum(), 0.0)
    call_oi = _safe_float(oi[df["contract_type"] == "CALL"].sum(), 0.0)
    put_vol = _safe_float(volume[df["contract_type"] == "PUT"].sum(), 0.0)
    call_vol = _safe_float(volume[df["contract_type"] == "CALL"].sum(), 0.0)

    oi_ratio = (put_oi / call_oi) if call_oi > 0 else 0.0
    vol_ratio = (put_vol / call_vol) if call_vol > 0 else 0.0
    return _safe_float(oi_ratio), _safe_float(vol_ratio)


def _spread_stats(df: pd.DataFrame) -> tuple[float, float, float, int, int]:
    if df.empty or "bid" not in df.columns or "ask" not in df.columns:
        return 0.0, 0.0, 0.0, 0, len(df)

    valid_mask = _valid_quote_mask(df)
    spread_sample_size = int(len(df))
    valid_spread_count = int(valid_mask.sum())

    if spread_sample_size == 0:
        return 0.0, 0.0, 0.0, 0, 0

    valid_quote_ratio = float(valid_spread_count / spread_sample_size)

    if valid_spread_count == 0:
        return 0.0, 0.0, valid_quote_ratio, 0, spread_sample_size

    bid = _to_numeric(df.loc[valid_mask, "bid"])
    ask = _to_numeric(df.loc[valid_mask, "ask"])
    mid = _compute_mid(df.loc[valid_mask])

    spread_pct = ((ask - bid) / mid).replace([np.inf, -np.inf], np.nan).dropna()
    spread_pct = spread_pct[(spread_pct >= 0) & (spread_pct <= 0.25)]

    if spread_pct.empty:
        return 0.0, 0.0, valid_quote_ratio, valid_spread_count, spread_sample_size

    return (
        _safe_float(spread_pct.mean(), 0.0),
        _safe_float(spread_pct.median(), 0.0),
        _safe_float(valid_quote_ratio, 0.0),
        _safe_int(valid_spread_count, 0),
        _safe_int(spread_sample_size, 0),
    )


def _liquid_ratio(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0

    valid_mask = _valid_quote_mask(df)
    if int(valid_mask.sum()) == 0:
        return 0.0

    sub = df.loc[valid_mask].copy()
    bid = _to_numeric(sub["bid"]) if "bid" in sub.columns else pd.Series(0.0, index=sub.index)
    ask = _to_numeric(sub["ask"]) if "ask" in sub.columns else pd.Series(0.0, index=sub.index)
    mid = _compute_mid(sub)
    volume = _to_numeric(sub["volume"]) if "volume" in sub.columns else pd.Series(0.0, index=sub.index)
    oi = _to_numeric(sub["open_interest"]) if "open_interest" in sub.columns else pd.Series(0.0, index=sub.index)

    spread_pct = ((ask - bid) / mid).replace([np.inf, -np.inf], np.nan)
    liquid_mask = (
        spread_pct.notna()
        & (spread_pct >= 0)
        & (spread_pct <= 0.15)
        & ((volume > 0) | (oi > 0))
    )

    ratio = float(liquid_mask.mean()) if len(sub) > 0 else 0.0
    return _safe_float(min(max(ratio, 0.0), 1.0), 0.0)


def _dominant_strike(
    df: pd.DataFrame,
    field: str,
    spot_price: float,
) -> tuple[float, float]:
    if df.empty or field not in df.columns or "strike" not in df.columns:
        return 0.0, 0.0

    sub = df.copy()
    sub[field] = _to_numeric(sub[field])
    sub["strike"] = _to_numeric(sub["strike"])
    sub = sub.dropna(subset=[field, "strike"])

    if sub.empty:
        return 0.0, 0.0

    grouped = sub.groupby("strike", dropna=True)[field].sum()
    if grouped.empty:
        return 0.0, 0.0

    top_strike = _safe_float(grouped.idxmax(), 0.0)
    if top_strike <= 0 or spot_price <= 0:
        return top_strike, 0.0

    dist_pct = abs(top_strike - float(spot_price)) / max(float(spot_price), 1e-9)
    return top_strike, _safe_float(dist_pct, 0.0)




def _quote_source_summary(df: pd.DataFrame) -> str:
    try:
        if df is None or df.empty or "source" not in df.columns:
            return "NONE"
        src = df["source"].astype(str).str.lower()
        if src.str.contains("ibkr").any():
            return "IBKR"
        if src.str.contains("massive").any() or src.str.contains("polygon").any():
            valid = (
                pd.to_numeric(df.get("bid", 0), errors="coerce").fillna(0) > 0
            ) & (
                pd.to_numeric(df.get("ask", 0), errors="coerce").fillna(0) > 0
            )
            if valid.any():
                return "MASSIVE_DELAYED"
        return "NONE"
    except Exception:
        return "NONE"

def _front_expiry_concentration(df: pd.DataFrame) -> float:
    if df.empty or "expiration" not in df.columns:
        return 0.0

    exp = pd.to_datetime(df["expiration"], errors="coerce").dropna()
    if exp.empty:
        return 0.0

    counts = exp.value_counts()
    total = int(counts.sum())
    if total <= 0:
        return 0.0

    return _safe_float(counts.iloc[0] / total, 0.0)


def analyze_options_surface(
    ticker: str,
    spot_price: float,
    chain: pd.DataFrame,
) -> OptionsSurfaceSnapshot:
    df = _normalize_chain(chain)

    if df.empty:
        return OptionsSurfaceSnapshot(
            ticker=ticker,
            spot_price=_safe_float(spot_price, 0.0),
            total_contracts=0,
        )

    total_contracts = int(len(df))

    atm = _atm_subset(df, spot_price, width=0.03)
    near_atm = _atm_subset(df, spot_price, width=0.05)
    mid_atm = _atm_subset(df, spot_price, width=0.10)

    atm_iv = _safe_mean(atm["implied_volatility"]) if "implied_volatility" in atm.columns else 0.0
    near_atm_iv = _safe_mean(near_atm["implied_volatility"]) if "implied_volatility" in near_atm.columns else 0.0
    mid_atm_iv = _safe_mean(mid_atm["implied_volatility"]) if "implied_volatility" in mid_atm.columns else 0.0

    term_slope, term_ratio = _term_metrics(df)
    call_iv_mean, put_iv_mean, skew_proxy = _skew_metrics(df)
    put_call_oi_ratio, put_call_volume_ratio = _put_call_ratios(df)

    (
        avg_spread_pct,
        median_spread_pct,
        valid_quote_ratio,
        valid_spread_count,
        spread_sample_size,
    ) = _spread_stats(df)

    liquid_contract_ratio = _liquid_ratio(df)

    top_oi_strike, top_oi_strike_distance_pct = _dominant_strike(df, "open_interest", spot_price)
    top_volume_strike, top_volume_strike_distance_pct = _dominant_strike(df, "volume", spot_price)
    front_expiry_concentration = _front_expiry_concentration(df)

    return OptionsSurfaceSnapshot(
        ticker=ticker,
        spot_price=_safe_float(spot_price, 0.0),
        atm_iv=_safe_float(atm_iv, 0.0),
        near_atm_iv=_safe_float(near_atm_iv, 0.0),
        mid_atm_iv=_safe_float(mid_atm_iv, 0.0),
        term_slope=_safe_float(term_slope, 0.0),
        term_ratio=_safe_float(term_ratio, 0.0),
        call_iv_mean=_safe_float(call_iv_mean, 0.0),
        put_iv_mean=_safe_float(put_iv_mean, 0.0),
        skew_proxy=_safe_float(skew_proxy, 0.0),
        put_call_oi_ratio=_safe_float(put_call_oi_ratio, 0.0),
        put_call_volume_ratio=_safe_float(put_call_volume_ratio, 0.0),
        avg_spread_pct=_safe_float(avg_spread_pct, 0.0),
        median_spread_pct=_safe_float(median_spread_pct, 0.0),
        liquid_contract_ratio=_safe_float(liquid_contract_ratio, 0.0),
        valid_quote_ratio=_safe_float(valid_quote_ratio, 0.0),
        valid_spread_count=_safe_int(valid_spread_count, 0),
        spread_sample_size=_safe_int(spread_sample_size, 0),
        top_oi_strike=_safe_float(top_oi_strike, 0.0),
        top_oi_strike_distance_pct=_safe_float(top_oi_strike_distance_pct, 0.0),
        top_volume_strike=_safe_float(top_volume_strike, 0.0),
        top_volume_strike_distance_pct=_safe_float(top_volume_strike_distance_pct, 0.0),
        front_expiry_concentration=_safe_float(front_expiry_concentration, 0.0),
        total_contracts=_safe_int(total_contracts, 0),
        quote_source=_quote_source_summary(df),
    )
