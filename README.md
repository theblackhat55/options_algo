# Options Algo

Systematic options trading system for the S&P 100 universe.
Runs nightly after market close, delivers top picks via WhatsApp through OpenClaw.

## Strategies

| Strategy | When | IV Regime |
|---|---|---|
| Bull Put Spread | Uptrend | HIGH |
| Bull Call Spread | Uptrend | LOW / NORMAL |
| Bear Call Spread | Downtrend | HIGH |
| Bear Put Spread | Downtrend | LOW / NORMAL |
| Iron Condor | Range-bound | HIGH / NORMAL |
| Long Butterfly | Range-bound / Squeeze | LOW |

---

## Architecture

```
options_algo/
├── config/              # Settings, universe, strategy params
├── src/
│   ├── data/            # OHLCV + options chain fetchers
│   ├── analysis/        # Technical regimes, IV, Greeks, RS
│   ├── strategy/        # Six spread constructors + selector
│   ├── screener/        # Composite screener
│   ├── risk/            # Position sizing, portfolio, event filter
│   ├── models/          # LightGBM ML layer (activates after 200 trades)
│   └── pipeline/        # Nightly scan, morning brief, monitor, tracker
├── dashboard/           # Streamlit dashboard
├── scripts/             # Setup, retrain, cron shell scripts
└── tests/               # pytest test suite
```

---

## Quick Start

### 1. Clone and install

```bash
cd /root
git clone https://github.com/theblackhat55/options_algo.git
cd options_algo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
nano .env   # add POLYGON_API_KEY at minimum
```

### 3. Download historical data

```bash
python scripts/setup_data.py --small   # 10 tickers, test run
python scripts/setup_data.py           # Full SP100 universe
```

### 4. Run nightly scan

```bash
# Dry run (no live options data, fast)
python -m src.pipeline.nightly_scan --dry-run

# Full run (requires Polygon API key)
python -m src.pipeline.nightly_scan
```

### 5. View morning brief

```bash
python -m src.pipeline.morning_brief
```

### 6. Launch dashboard

```bash
streamlit run dashboard/app.py --server.port 8501
```

---

## Pipeline Flow

```
After market close (9:30 PM ET)
        │
        ▼
1. Update OHLCV data (yfinance, incremental)
        │
        ▼
2. Pre-filter: price ≥ $20, avg vol ≥ 1M
        │
        ▼
3. Market context: VIX proxy, SPY trend, breadth
        │
        ▼
4. Classify regimes (ADX, RSI, EMAs, Bollinger)
        │
        ▼
5. Relative strength vs SPY + sector ETF
        │
        ▼
6. IV analysis: IV rank, IV percentile, IV/HV ratio
        │
        ▼
7. Strategy selection (regime × IV matrix)
        │
        ▼
8. Event filter: skip stocks with earnings in DTE window
        │
        ▼
9. Construct trades: optimal strikes via delta/price
        │
        ▼
10. Rank: confidence × PoP × (1/RR) × EV bonus
        │
        ▼
11. Save signal JSON → WhatsApp morning brief
```

---

## Configuration

All parameters are in `config/settings.py` and overridable via `.env`:

| Parameter | Default | Description |
|---|---|---|
| `MAX_RISK_PER_TRADE_PCT` | 2.0 | Max % of account risked per trade |
| `MAX_POSITIONS` | 5 | Maximum concurrent open positions |
| `PROFIT_TARGET_PCT` | 50 | Close credit spreads at 50% of max profit |
| `STOP_LOSS_PCT` | 100 | Close at 100% of credit received (2× debit) |
| `IV_HIGH_THRESHOLD` | 70 | IV rank above = sell premium |
| `IV_LOW_THRESHOLD` | 30 | IV rank below = buy premium |
| `DEFAULT_DTE_PREMIUM_SELL` | 45 | DTE for credit spreads |
| `DEFAULT_DTE_DIRECTIONAL` | 21 | DTE for debit spreads |
| `DEFAULT_SHORT_DELTA` | 0.25 | Short strike delta target |
| `IC_WING_DELTA` | 0.16 | Iron condor wing delta |

---

## Data Sources

| Source | Plan | Purpose |
|---|---|---|
| Polygon.io Options | Starter ($29/mo) | Options chains, IV, Greeks |
| Polygon.io Stocks | Starter ($29/mo) | Real-time OHLCV |
| yfinance | Free | OHLCV fallback, options fallback |
| Finnhub | Free | Earnings calendar |
| Tradier | Free (brokerage) | Backup options data |
| Alpha Vantage | Free | Backup quotes |

---

## OpenClaw Cron Jobs

Add from the OpenClaw user on the Hetzner server:

```bash
# Nightly scan — 9:30 PM ET (02:30 UTC) Tue-Sat
openclaw cron add \
  --name "Options nightly scan" \
  --cron "30 2 * * 2-6" \
  --tz UTC --exact \
  --session isolated \
  --message "Run the options algo nightly scan. Execute: sudo /root/options_algo/.venv/bin/python3 -m src.pipeline.nightly_scan. Show the top 5 picks with strategy, strikes, credit/debit, max risk, and probability of profit." \
  --announce --channel whatsapp --to "your_whatsapp_number_here" \
  --timeout-seconds 600

# Morning brief — 9:00 AM ET (14:00 UTC) Mon-Fri
openclaw cron add \
  --name "Options morning brief" \
  --cron "0 14 * * 1-5" \
  --tz UTC --exact \
  --session isolated \
  --message "Run: sudo /root/options_algo/.venv/bin/python3 -m src.pipeline.morning_brief. Send the formatted options picks for today via WhatsApp. Include all trade details." \
  --announce --channel whatsapp --to "your_whatsapp_number_here" \
  --timeout-seconds 120
```

---

## ML Layer (Phase 3 — activates after 200 trades)

Once 200+ paper trade outcomes are recorded in `output/trades/trade_outcomes.jsonl`,
train LightGBM models per strategy:

```bash
python scripts/retrain_models.py
```

Features: IV rank, IV/HV ratio, ADX, RSI, trend strength, direction score,
RS rank, sector, DTE, spread width, short delta, confidence.

Target: binary win/loss. Walk-forward validation with quarter-Kelly sizing.

---

## Testing

```bash
# Full test suite
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov=config --cov-report=term-missing

# Single module
pytest tests/test_strategies.py -v
pytest tests/test_volatility.py -v
pytest tests/test_screener.py -v
pytest tests/test_pipeline.py -v
```

---

## Budget

| Item | Cost/Month |
|---|---|
| Polygon.io Options Starter | $29 |
| Polygon.io Stocks Starter | $29 |
| Hetzner server (shared with spx_algo) | $0 |
| All other data sources | $0 |
| **Total** | **$58** |

Reserve: $42/month for future upgrades.

---

## Roadmap

- [x] Phase 1 — Data pipeline, regime classifier, IV analysis
- [x] Phase 1 — Six strategy constructors (all spread types)
- [x] Phase 1 — Nightly scan pipeline + WhatsApp morning brief
- [x] Phase 1 — Event filter, position sizing, portfolio tracker
- [x] Phase 1 — Outcome tracker (paper trade log)
- [x] Phase 1 — Streamlit dashboard
- [ ] Phase 2 — Live Polygon options chains in production
- [ ] Phase 2 — Position monitor intraday alerts via OpenClaw
- [ ] Phase 3 — ML layer after 200+ outcomes (LightGBM per strategy)
- [ ] Phase 4 — Strike optimizer (ML-driven strike selection)
- [ ] Phase 4 — Sector rotation overlay

---

## Server Setup (Hetzner — shared with spx_algo)

```bash
# Same server, separate virtualenv
cd /root
git clone https://github.com/theblackhat55/options_algo.git
cd options_algo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && nano .env
python scripts/setup_data.py
```

Shared resources: 8GB RAM, sufficient for both spx_algo and options_algo
running sequentially (scans staggered by 30 min).
