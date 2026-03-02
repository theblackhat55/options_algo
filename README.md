# Options Algo

Automated options strategy scanner and trade generator for S&P 100 stocks.
Runs nightly after market close, delivers top picks via WhatsApp through OpenClaw.

## Strategy Coverage

| Strategy | Regime | IV | Type |
|---|---|---|---|
| Bull Call Spread | Uptrend | Low / Normal | Debit |
| Bear Put Spread | Downtrend | Low / Normal | Debit |
| Bull Put Spread | Uptrend | High | Credit |
| Bear Call Spread | Downtrend | High | Credit |
| Iron Condor | Range-bound | High / Normal | Credit |
| Long Butterfly | Range-bound / Squeeze | Low | Debit |

## Architecture

```
options_algo/
├── config/
│   ├── settings.py          # API keys, thresholds, paths
│   ├── universe.py          # S&P 100 tickers + sector map
│   └── strategies.py        # Strategy parameters
├── src/
│   ├── data/
│   │   ├── stock_fetcher.py       # OHLCV download + cache (yfinance)
│   │   ├── options_fetcher.py     # Options chains (Polygon → Tradier → yfinance)
│   │   ├── earnings_calendar.py   # Earnings dates (Finnhub)
│   │   └── market_context.py      # VIX proxy, breadth, sector ETFs
│   ├── analysis/
│   │   ├── technical.py           # Regime classifier (8 regimes)
│   │   ├── volatility.py          # IV rank, IV percentile, IV/HV ratio
│   │   ├── options_analytics.py   # Black-Scholes Greeks, prob calcs
│   │   └── relative_strength.py   # Mansfield RS vs SPY + sector
│   ├── strategy/
│   │   ├── selector.py            # Regime × IV → strategy matrix
│   │   ├── credit_spread.py       # Bull put + bear call spreads
│   │   ├── bull_call_spread.py
│   │   ├── bear_put_spread.py
│   │   ├── iron_condor.py
│   │   └── butterfly.py
│   ├── risk/
│   │   ├── event_filter.py        # Earnings / event exclusion
│   │   ├── position_sizer.py      # Kelly criterion sizing
│   │   └── portfolio.py           # Portfolio Greeks + position limits
│   ├── screener/
│   │   └── composite_screener.py  # Unified screener interface
│   ├── models/
│   │   ├── features.py            # ML feature engineering
│   │   ├── trainer.py             # Walk-forward LightGBM training
│   │   └── predictor.py           # Model loading + inference
│   └── pipeline/
│       ├── nightly_scan.py        # Main orchestrator
│       ├── morning_brief.py       # WhatsApp formatter
│       ├── position_monitor.py    # Intraday exit alerts
│       └── outcome_tracker.py     # Paper trade log + ML dataset
├── dashboard/
│   └── app.py               # Streamlit dashboard (6 pages)
├── scripts/
│   ├── setup_data.py        # Initial 2yr data backfill
│   ├── daily_scan.sh        # Cron shell wrapper
│   ├── morning_report.sh    # Morning brief shell wrapper
│   └── retrain_models.py    # Weekly ML retrain
└── tests/
    ├── test_volatility.py
    ├── test_strategies.py
    ├── test_screener.py
    └── test_pipeline.py
```

## Budget

| Item | Cost/Month |
|---|---|
| Polygon Options Starter | $29 |
| Polygon Stocks Starter | $29 |
| Tradier (free w/ account) | $0 |
| Finnhub (existing) | $0 |
| Hetzner server (shared) | $0 |
| **Total** | **$58** |

## Server Setup (Hetzner)

```bash
cd /root
git clone https://github.com/theblackhat55/options_algo.git
cd options_algo

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy and fill in API keys
cp .env.example .env
nano .env

# Backfill 2 years of price data
python scripts/setup_data.py

# Test the pipeline (no live options data)
python -m src.pipeline.nightly_scan --dry-run

# Launch dashboard
streamlit run dashboard/app.py --server.port 8501
```

## OpenClaw Cron Jobs

```bash
# Nightly scan — 9:30 PM ET (02:30 UTC) Tue–Sat
openclaw cron add \
  --name "Options nightly scan" \
  --cron "30 2 * * 2-6" \
  --tz UTC --exact \
  --session isolated \
  --message "Run the options algo nightly scan. Execute: sudo /root/options_algo/.venv/bin/python3 -m src.pipeline.nightly_scan. Show the top 5 picks with strategy, strikes, credit/debit, max risk, and probability of profit." \
  --announce --channel whatsapp --to "your_whatsapp_number_here" \
  --timeout-seconds 600

# Morning brief — 9:00 AM ET (14:00 UTC) Mon–Fri
openclaw cron add \
  --name "Options morning brief" \
  --cron "0 14 * * 1-5" \
  --tz UTC --exact \
  --session isolated \
  --message "Run: sudo /root/options_algo/.venv/bin/python3 -m src.pipeline.morning_brief. Send the formatted options picks for today via WhatsApp. Include all trade details." \
  --announce --channel whatsapp --to "your_whatsapp_number_here" \
  --timeout-seconds 120
```

## Pipeline Flow

```
After close (9:30 PM ET)
        │
        ▼
1. Update OHLCV (yfinance incremental)
        │
        ▼
2. Pre-filter: price ≥ $20, avg vol ≥ 1M
        │
        ▼
3. Market context: VIX proxy, breadth, sector ETFs
        │
        ▼
4. Classify regimes (8 regimes per stock)
        │
        ▼
5. Relative strength vs SPY + sector
        │
        ▼
6. IV analysis: rank, percentile, IV/HV ratio
        │
        ▼
7. Strategy selection (regime × IV matrix)
        │
        ▼
8. Event filter: skip stocks with earnings in window
        │
        ▼
9. Construct trades: fetch live chain → find optimal strikes
        │
        ▼
10. Rank by composite score → top 5 picks
        │
        ▼
11. Save JSON signal → format WhatsApp message
```

## Regime → Strategy Matrix

```
                    IV HIGH (≥70)     IV NORMAL       IV LOW (≤30)
STRONG UPTREND      Bull Put Spread   Bull Call Spread Bull Call Spread
UPTREND             Bull Put Spread   Bull Call Spread Bull Call Spread
RANGE BOUND         Iron Condor       Iron Condor      Long Butterfly
DOWNTREND           Bear Call Spread  Bear Put Spread  Bear Put Spread
STRONG DOWNTREND    Bear Call Spread  Bear Put Spread  Bear Put Spread
SQUEEZE             Iron Condor       Long Butterfly   Long Butterfly
REVERSAL UP         Bull Put Spread   Bull Call Spread Bull Call Spread
REVERSAL DOWN       Bear Call Spread  Bear Put Spread  Bear Put Spread
```

## ML Layer (activates after 200+ outcomes)

Features used per trade:
- IV rank, IV percentile, IV/HV ratio, IV trend
- ADX, RSI, trend strength, direction score, EMA alignment
- Relative strength rank (vs SPY), RS trend
- DTE at entry, spread width, short delta
- Sector (encoded), BB squeeze flag

Model: LightGBM binary classifier (win/loss)
Validation: Walk-forward, minimum 1yr training window
Replaces rules-based confidence score when ready.

## Exit Rules

| Strategy | Profit Target | Stop Loss |
|---|---|---|
| Credit spreads | 50% of max credit | 100% of credit (spread at 2× credit) |
| Debit spreads | 50% of max profit | 50% of debit paid |
| Iron Condor | 50% of total credit | 200% of credit |
| Butterfly | 40% of max profit | 50% of debit |

## Development

```bash
# Run tests
pytest tests/ -v --tb=short

# Run tests with coverage
pytest tests/ --cov=src --cov=config --cov-report=term-missing

# Dry-run scan (no network calls for options)
python -m src.pipeline.nightly_scan --dry-run

# Format morning brief from last signal
python -m src.pipeline.morning_brief

# Retrain ML models
python scripts/retrain_models.py
```

## Phases

- **Phase 1 (Week 1):** Data pipeline, regime classifier, IV analysis, credit spreads, nightly scan ✅
- **Phase 2 (Week 2):** All strategy constructors, Iron Condor, Butterfly ✅
- **Phase 3 (Week 3):** Streamlit dashboard, position monitor, outcome tracker ✅
- **Phase 4 (Month 2+):** ML layer activates after 200+ paper trade outcomes

## License

Private — theblackhat55
