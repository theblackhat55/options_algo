# Options Algo

> **Systematic options trading system for the S&P 100 universe.**
> Scans nightly, monitors positions intraday, and delivers picks + alerts via WhatsApp through OpenClaw.

---

## Table of Contents

1. [Strategies](#strategies)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Configuration Reference](#configuration-reference)
5. [Daily Workflow](#daily-workflow)
6. [Pipeline Flow](#pipeline-flow)
7. [Streamlit Dashboard](#streamlit-dashboard)
8. [Cron Jobs](#cron-jobs)
9. [Position Monitor](#position-monitor)
10. [IV Snapshot Collector](#iv-snapshot-collector)
11. [Backtesting Long Options](#backtesting-long-options)
12. [ML Layer](#ml-layer)
13. [Testing](#testing)
14. [Data Sources & Cost](#data-sources--cost)
15. [Server Setup (Hetzner)](#server-setup-hetzner)
16. [Roadmap](#roadmap)

---

## Strategies

Eight strategies are fully implemented. Strategy selection is automatic based on the regime × IV matrix:

| Strategy | Direction | IV Regime | Type |
|---|---|---|---|
| Bull Put Spread | BULLISH | HIGH | Credit |
| Bear Call Spread | BEARISH | HIGH | Credit |
| Bull Call Spread | BULLISH | LOW / NORMAL | Debit |
| Bear Put Spread | BEARISH | LOW / NORMAL | Debit |
| Iron Condor | NEUTRAL | HIGH / NORMAL | Credit |
| Long Butterfly | NEUTRAL | LOW (squeeze) | Debit |
| **Long Call** | BULLISH | LOW (< 40 IVR) | Single-leg debit |
| **Long Put** | BEARISH | LOW (< 40 IVR) | Single-leg debit |

> Long Call / Long Put are auto-upgraded from Bull/Bear spreads when IV rank is low, confidence ≥ 65%, and the ML blend confirms edge.

---

## Architecture

```
options_algo/
├── config/
│   ├── settings.py          # All parameters — overridable via .env
│   ├── strategies.py        # StrategyConfig per strategy (DTE, delta, targets)
│   └── universe.py          # SP100 ticker universe + sector map
│
├── src/
│   ├── data/
│   │   ├── stock_fetcher.py         # OHLCV download (Polygon / yfinance)
│   │   ├── options_fetcher.py       # Options chains (Polygon / Tradier / yfinance)
│   │   ├── market_context.py        # VIX tier, SPY trend, market breadth
│   │   ├── ibkr_live.py             # IBKR real-time price + flow enrichment
│   │   └── earnings_calendar.py     # Finnhub earnings cache (auto-pruned daily)
│   │
│   ├── analysis/
│   │   ├── technical.py             # Regime classifier (ADX, RSI, EMAs, Bollinger)
│   │   ├── volatility.py            # IV rank, IV/HV ratio, IV-RV spread, skew
│   │   ├── options_analytics.py     # Black-Scholes Greeks (r from RISK_FREE_RATE env)
│   │   ├── relative_strength.py     # RS vs SPY + sector ETF
│   │   ├── levels.py                # Support/Resistance via volume profile
│   │   └── patterns.py              # Bollinger squeeze, divergence, breakout
│   │
│   ├── strategy/
│   │   ├── selector.py              # Regime × IV matrix + long-option upgrade logic
│   │   ├── credit_spread.py         # Bull Put / Bear Call constructors
│   │   ├── bull_call_spread.py      # Bull Call Spread constructor
│   │   ├── bear_put_spread.py       # Bear Put Spread constructor
│   │   ├── iron_condor.py           # Iron Condor constructor
│   │   ├── butterfly.py             # Long Butterfly constructor
│   │   ├── long_call.py             # Long Call constructor
│   │   └── long_put.py              # Long Put constructor
│   │
│   ├── risk/
│   │   ├── portfolio.py             # Position tracker, close_position(), fcntl locking
│   │   ├── sizing.py                # Kelly fraction, max-risk per trade
│   │   └── event_filter.py          # Earnings window filter (skip risky dates)
│   │
│   ├── models/
│   │   ├── trainer.py               # LightGBM training (per strategy, walk-forward)
│   │   └── predictor.py             # Win-probability prediction (23-feature vector)
│   │
│   └── pipeline/
│       ├── nightly_scan.py          # Main scan: Steps 1-10 → top_picks JSON
│       ├── morning_brief.py         # WhatsApp-formatted morning summary
│       ├── position_monitor.py      # Intraday lifecycle: P&L checks, auto-close
│       ├── intraday_alerts.py       # VIX spike detector, breakout scanner
│       └── outcome_tracker.py       # Paper trade JSONL log (fcntl-locked)
│
├── dashboard/
│   ├── app.py                       # ✅ Canonical Streamlit dashboard (9 pages)
│   ├── ticker_analysis.py           # Per-ticker deep-dive component
│   └── unified_app.py               # Compatibility shim → imports from app.py
│
├── scripts/
│   ├── daily_scan.sh                # Cron wrapper for nightly_scan
│   ├── morning_report.sh            # Cron wrapper for morning_brief
│   ├── monitor_positions.sh         # Cron wrapper for position_monitor
│   ├── capture_iv_snapshot.py       # Nightly IV history collector
│   ├── backtest_long_options.py     # Walk-forward backtest for LONG_CALL/LONG_PUT
│   ├── retrain_models.py            # Weekly ML retrain
│   ├── check_outcomes.py            # Morning outcome updater
│   └── setup_data.py                # First-run OHLCV download
│
└── tests/                           # pytest suite — 356 tests
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/theblackhat55/options_algo.git
cd options_algo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
nano .env
```

**Minimum required:**

```ini
POLYGON_API_KEY=your_polygon_key_here   # required for options chains
FINNHUB_API_KEY=your_finnhub_key_here   # free — earnings calendar
WHATSAPP_NUMBER=+1xxxxxxxxxx            # OpenClaw delivery target
```

### 3. Download historical data

```bash
# Test run — 10 tickers only
python scripts/setup_data.py --small

# Full SP100 universe (~5 min)
python scripts/setup_data.py
```

### 4. Dry-run the nightly scan

```bash
# Fast dry run — uses HV×1.15 proxy IV (no options chain fetch)
python -m src.pipeline.nightly_scan --dry-run
```

Expected output:
```
TOP 5 PICKS
===========
#1 AAPL   | BULL_PUT_SPREAD      | Conf=72% | BULLISH | IV rank=68% | Price=$195.40
   Credit: $0.85 | Max Risk: $4.15 | PoP: 76%
   Strong uptrend (ADX=32), high IV rank (68th%), RS rank 82nd pctile
...
```

### 5. View the morning brief

```bash
python -m src.pipeline.morning_brief
```

### 6. Launch the dashboard

```bash
streamlit run dashboard/app.py --server.port 8501
```

Then open **http://localhost:8501** in your browser.

---

## Configuration Reference

All settings live in `config/settings.py` and can be overridden in `.env`.

### Core Risk

| Variable | Default | Description |
|---|---|---|
| `MAX_POSITIONS` | `5` | Maximum concurrent open positions |
| `MAX_RISK_PER_TRADE_PCT` | `2.0` | Max % of account per trade |
| `PROFIT_TARGET_PCT` | `50` | Close credit spreads at 50% of max profit |
| `STOP_LOSS_PCT` | `100` | Close at 100% of credit received |
| `MAX_SAME_DIRECTION_PCT` | `60` | Max % of positions in one direction (e.g. 60% × 5 = 3 bullish max) |
| `RISK_FREE_RATE` | `0.053` | Black-Scholes risk-free rate (~5.3% T-bill) — update via `.env` |

### IV Thresholds

| Variable | Default | Description |
|---|---|---|
| `IV_HIGH_THRESHOLD` | `70` | IV rank ≥ 70 → sell premium |
| `IV_LOW_THRESHOLD` | `30` | IV rank ≤ 30 → buy premium |
| `MIN_IV_RV_SPREAD_CREDIT` | `5.0` | Min IV−HV spread (vol pts) to flag premium-rich |

### VIX Circuit Breaker

| Variable | Default | Behaviour |
|---|---|---|
| `VIX_CAUTION_LEVEL` | `28` | Allow credit strategies + NEUTRAL only |
| `VIX_DEFENSIVE_LEVEL` | `35` | NEUTRAL direction only (Iron Condor / Butterfly) |
| `VIX_LIQUIDATION_LEVEL` | `45` | Halt all new trade generation |
| `VIX_SPIKE_WINDOW` | `5` | Rolling window for spike detection |
| `VIX_SPIKE_THRESHOLD_PCT` | `25` | Alert if VIX rises >25% above rolling average |

### Long Option Parameters

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_DTE_LONG_OPTION` | `35` | Target DTE for long calls/puts |
| `LONG_OPTION_DELTA` | `0.65` | Target delta (ITM for edge) |
| `LONG_OPTION_MIN_CONFIDENCE` | `0.65` | Minimum confidence to keep long-option upgrade |
| `LONG_OPTION_PROFIT_TARGET_PCT` | `100` | Exit when option doubles |
| `LONG_OPTION_STOP_LOSS_PCT` | `50` | Exit if option loses 50% of premium paid |
| `LONG_OPTION_TIME_STOP_DTE` | `10` | Exit at ≤10 DTE regardless of P&L |
| `LONG_OPTION_IV_RANK_CEILING` | `40` | Never buy premium when IV rank > 40% |
| `LONG_OPTION_MAX_ALLOCATION_PCT` | `30` | Max % of portfolio in long options |

### Strategy DTE & Strikes

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_DTE_PREMIUM_SELL` | `45` | DTE for credit spreads |
| `DEFAULT_DTE_DIRECTIONAL` | `21` | DTE for debit spreads |
| `DEFAULT_DTE_IC` | `45` | DTE for Iron Condors |
| `DEFAULT_SPREAD_WIDTH` | `5` | Spread width in strike points |
| `DEFAULT_SHORT_DELTA` | `0.25` | Short strike delta target |
| `IC_WING_DELTA` | `0.16` | Iron Condor wing delta |

### SPY Directional Gate

| Variable | Default | Description |
|---|---|---|
| `SPY_DIRECTIONAL_GATE_PCT` | `1.0` | Skip bearish picks when SPY 5d return > +1%; skip bullish when < −1% |

### IBKR Live Enrichment

| Variable | Default | Description |
|---|---|---|
| `IBKR_ENABLED` | `true` | Enable real-time price + flow enrichment via IB Gateway |
| `IBKR_HOST` | `127.0.0.1` | IB Gateway host |
| `IBKR_PORT` | `4002` | IB Gateway port (4002 = paper, 4001 = live) |

---

## Daily Workflow

```
  ┌─────────────────────────────────────────────────────────┐
  │                   DAILY SCHEDULE (ET)                   │
  ├─────────────────────────────────────────────────────────┤
  │  4:15 PM  │  capture_iv_snapshot.py  →  IV history      │
  │  9:30 PM  │  nightly_scan.py         →  top picks JSON  │
  │  9:00 AM  │  morning_brief.py        →  WhatsApp brief  │
  │  Every 30m│  position_monitor.py     →  lifecycle check │
  │  Weekly   │  retrain_models.py       →  ML retrain      │
  └─────────────────────────────────────────────────────────┘
```

### 1 — After close: capture IV snapshot (4:15 PM ET)

```bash
python scripts/capture_iv_snapshot.py
# or for specific tickers:
python scripts/capture_iv_snapshot.py --tickers AAPL MSFT NVDA TSLA
```

Writes `data/processed/iv_snapshots/{TICKER}_iv_history.parquet`.
After 20+ trading days, replaces the HV×1.15 proxy with real IV history.

---

### 2 — Nightly scan (9:30 PM ET)

```bash
# Production run
python -m src.pipeline.nightly_scan

# Dry run (no options chain fetch, fast)
python -m src.pipeline.nightly_scan --dry-run
```

**10-step pipeline:**

1. Update OHLCV data (yfinance incremental)
2. Pre-filter universe: price ≥ $20, avg volume ≥ 1M
3. Get market context: VIX tier, SPY 5-day return, breadth score
4. Classify regimes: ADX, RSI, EMA crossovers, Bollinger squeeze
5. Relative strength: rank vs SPY + sector ETF
6. IV analysis: IV rank, IV/HV ratio, IV-RV spread
7. Strategy selection: regime × IV matrix + ML confidence blend
8. Event filter: skip tickers with earnings in DTE window
9. Construct trades: delta-targeted strikes, Greeks, PoP
10. Rank & finalize: top 5 by confidence × PoP × EV

Output saved to `output/signals/signal_YYYY-MM-DD.json`.

---

### 3 — Morning brief (9:00 AM ET)

```bash
python -m src.pipeline.morning_brief
```

Reads the latest signal JSON and formats it into a WhatsApp-ready message with:
- Priority rank, ticker, sector, strategy
- Direction, confidence %, IV rank, IV/HV ratio
- Trade specifics: credit/debit, max risk, PoP, strikes
- TA signals: breakouts, divergences, squeeze direction
- Long-option targets (if upgraded)
- RS rank and trend

---

### 4 — Position monitor (every 30 min, 9:30 AM–4:00 PM ET)

```bash
python -m src.pipeline.position_monitor
# or via the cron shell script:
bash scripts/monitor_positions.sh
```

For each open position:
- Fetches live price (IBKR if enabled, yfinance fallback)
- Checks profit target, stop loss, expiry, VIX spike
- **Auto-closes** hitting positions: calls `record_exit()` (P&L logged) + `close_position()` (portfolio updated)
- Generates WhatsApp alert for any action required

---

## Pipeline Flow

```
Market Close (4:00 PM ET)
        │
        ▼
[4:15 PM] capture_iv_snapshot.py
  └─ Polygon ATM IV + OI + volume → parquet history
        │
        ▼
[9:30 PM] nightly_scan.py
  ├─ Step 1: OHLCV update
  ├─ Step 2: Universe pre-filter (price, volume)
  ├─ Step 3: Market context (VIX circuit breaker, SPY gate)
  ├─ Step 4: Regime classification (ADX, RSI, EMAs, Bollinger)
  ├─ Step 5: Relative strength ranking
  ├─ Step 6: IV analysis (rank, percentile, IV/HV, skew)
  ├─ Step 7: Strategy selection + ML confidence blend (60/40)
  │           └─ Long option upgrade when IV rank < 40%
  ├─ Step 8: Earnings event filter
  ├─ Step 9: Trade construction (strikes, Greeks, PoP)
  └─ Step 10: Rank → top 5 picks → signal JSON
        │
        ▼
[9:00 AM] morning_brief.py
  └─ WhatsApp brief with all trade details
        │
        ▼
[Every 30 min] position_monitor.py
  ├─ Long options: check profit target (+100%), stop (−50%), time stop (≤10 DTE)
  ├─ Credit spreads: check decay target (50% of credit) and 2× stop
  ├─ Debit spreads: check gain/loss vs entry debit
  └─ Auto-close: record_exit() + close_position() → no stale positions
        │
        ▼
[Weekly] retrain_models.py
  └─ LightGBM retrain per strategy (activates after 200 closed trades)
```

---

## Streamlit Dashboard

Launch the dashboard:

```bash
# Local development
streamlit run dashboard/app.py

# Production (background, with port)
nohup streamlit run dashboard/app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  >> /var/log/options_dashboard.log 2>&1 &
```

Then open **http://your-server-ip:8501**.

### Dashboard Pages

| Page | Description |
|---|---|
| **Ticker Analysis** | Per-ticker deep dive: regime, IV rank, RS rank, S/R levels, patterns, options flow |
| **Today's Picks** | Latest signal JSON rendered as cards with trade details, TA signals, rationale |
| **Portfolio Overview** | Open positions table, direction balance pie chart, risk budget used, warnings |
| **IV Snapshots** | ATM IV history chart per ticker, IV rank, open interest, volume trend |
| **IV Heatmap** | Universe-wide IV rank heatmap — quickly spot outliers |
| **Regime Map** | SP100 regime classification grid (UPTREND, DOWNTREND, RANGE, SQUEEZE…) |
| **Trade Log** | Full paper trade history with P&L, outcome, close reason, days held |
| **Strategy Stats** | Win rate, avg P&L, avg P&L % per strategy (feeds ML training readiness) |
| **Run Scan** | Trigger a dry-run nightly scan from the browser and see live output |

### Keeping the dashboard alive with systemd

Create `/etc/systemd/system/options-dashboard.service`:

```ini
[Unit]
Description=Options Algo Streamlit Dashboard
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/options_algo
ExecStart=/root/options_algo/.venv/bin/streamlit run dashboard/app.py \
          --server.port 8501 \
          --server.address 0.0.0.0 \
          --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable options-dashboard
systemctl start options-dashboard
systemctl status options-dashboard
```

---

## Cron Jobs

All cron jobs assume the repo lives at `/root/options_algo`.
Replace `$WHATSAPP_NUMBER` with your number (e.g. `+1234567890`).

### Option A — Native crontab (recommended for Hetzner VPS)

Edit with `crontab -e`:

```cron
# ── Options Algo Cron Schedule ─────────────────────────────────────────────

# 1. IV snapshot: Mon-Fri at 4:15 PM ET (21:15 UTC)
15 21 * * 1-5  cd /root/options_algo && .venv/bin/python3 scripts/capture_iv_snapshot.py \
               >> /root/options_algo/logs/iv_snapshot_$(date +\%Y\%m\%d).log 2>&1

# 2. Nightly scan: Tue-Sat at 9:30 PM ET (02:30 UTC next day)
30 2 * * 2-6   /root/options_algo/scripts/daily_scan.sh \
               >> /root/options_algo/logs/nightly_scan_$(date +\%Y\%m\%d).log 2>&1

# 3. Morning brief: Mon-Fri at 9:00 AM ET (14:00 UTC)
0 14 * * 1-5   /root/options_algo/scripts/morning_report.sh \
               >> /root/options_algo/logs/morning_brief_$(date +\%Y\%m\%d).log 2>&1

# 4. Position monitor: every 30 min during market hours Mon-Fri (13:30-21:00 UTC)
*/30 13-21 * * 1-5  /root/options_algo/scripts/monitor_positions.sh \
                    >> /var/log/monitor_positions.log 2>&1

# 5. ML retrain: every Sunday at 6:00 AM UTC
0 6 * * 0      cd /root/options_algo && .venv/bin/python3 scripts/retrain_models.py \
               >> /root/options_algo/logs/retrain_$(date +\%Y\%m\%d).log 2>&1
```

Make scripts executable:

```bash
chmod +x /root/options_algo/scripts/*.sh
mkdir -p /root/options_algo/logs
```

---

### Option B — OpenClaw cron (WhatsApp delivery)

Use OpenClaw if you want scan results delivered directly to WhatsApp without a separate delivery step:

```bash
# 1. Nightly scan → WhatsApp results (9:30 PM ET, Tue-Sat)
openclaw cron add \
  --name "Options nightly scan" \
  --cron "30 2 * * 2-6" \
  --tz UTC --exact \
  --session isolated \
  --message "Run the options algo nightly scan in /root/options_algo. \
Execute: cd /root/options_algo && .venv/bin/python3 -m src.pipeline.nightly_scan. \
Show the top 5 picks with: strategy, strikes, credit/debit, max risk, probability of profit, and rationale." \
  --announce --channel whatsapp --to "$WHATSAPP_NUMBER" \
  --timeout-seconds 600

# 2. Morning brief → WhatsApp (9:00 AM ET, Mon-Fri)
openclaw cron add \
  --name "Options morning brief" \
  --cron "0 14 * * 1-5" \
  --tz UTC --exact \
  --session isolated \
  --message "Run: cd /root/options_algo && .venv/bin/python3 -m src.pipeline.morning_brief. \
Send the formatted options picks for today via WhatsApp. Include all trade details, TA signals, and long-option targets." \
  --announce --channel whatsapp --to "$WHATSAPP_NUMBER" \
  --timeout-seconds 120

# 3. Position monitor alert → WhatsApp (every 30 min, market hours)
openclaw cron add \
  --name "Options position monitor" \
  --cron "*/30 13-21 * * 1-5" \
  --tz UTC --exact \
  --session isolated \
  --message "Run: cd /root/options_algo && .venv/bin/python3 -m src.pipeline.position_monitor. \
If any position has hit profit target, stop loss, or VIX spike alert, format a clear action message and send via WhatsApp." \
  --announce --channel whatsapp --to "$WHATSAPP_NUMBER" \
  --timeout-seconds 120

# 4. IV snapshot (silent — no WhatsApp needed)
openclaw cron add \
  --name "IV snapshot collector" \
  --cron "15 21 * * 1-5" \
  --tz UTC --exact \
  --session isolated \
  --message "Run: cd /root/options_algo && .venv/bin/python3 scripts/capture_iv_snapshot.py. Log results only." \
  --timeout-seconds 300
```

View all scheduled jobs: `openclaw cron list`

---

### Log rotation

```bash
# /etc/logrotate.d/options_algo
/root/options_algo/logs/*.log {
    daily
    rotate 14
    compress
    missingok
    notifempty
    create 0640 root root
}
```

---

## Position Monitor

The position monitor (`src/pipeline/position_monitor.py`) runs every 30 minutes during market hours and checks every open position against exit rules:

### Exit rules by strategy type

| Strategy | Profit Target | Stop Loss | Time Stop |
|---|---|---|---|
| Credit Spreads (BPS, BCS) | 50% of credit received | 100% of credit (2× loss) | ≤5 DTE warning |
| Debit Spreads (BcS, BePS) | 50% of debit paid | 50% of debit paid | ≤5 DTE warning |
| Long Call / Long Put | +100% (option doubles) | −50% of premium paid | ≤10 DTE hard exit |
| Iron Condor | 50% of credit | 200% of credit | ≤5 DTE warning |

### Auto-close behaviour

When a CLOSE alert is triggered, the monitor automatically:

1. Calls `record_exit(trade_id, exit_price, close_reason)` → logs P&L to `output/trades/trade_outcomes.jsonl`
2. Calls `close_position(ticker, strategy, trade_id)` → marks position as CLOSED in `output/trades/positions.json`
3. Sends WhatsApp alert with ticker, reason, and P&L

No manual intervention required — the system handles the full lifecycle.

### VIX spike emergency

If VIX spikes > 25% above its 5-day rolling average, a `VIX_SPIKE` alert is broadcast for all open positions. If VIX reaches the DEFENSIVE level (≥35), a `VIX_DEFENSIVE` alert recommends closing all credit spreads.

---

## IV Snapshot Collector

```bash
# All tickers in universe
python scripts/capture_iv_snapshot.py

# Specific tickers only
python scripts/capture_iv_snapshot.py --tickers AAPL MSFT NVDA TSLA AMD

# Dry run (print output, don't write files)
python scripts/capture_iv_snapshot.py --dry-run
```

**Output:** `data/processed/iv_snapshots/{TICKER}_iv_history.parquet`

Each file has columns: `date`, `atm_iv`, `call_iv`, `put_iv`, `skew`, `open_interest`, `volume`, `source`.

After **20+ trading days** of snapshots, `analyze_iv()` uses real IV history instead of the HV×1.15 proxy, making IV rank and IV percentile accurate.

---

## Backtesting Long Options

Walk-forward backtest for LONG_CALL and LONG_PUT strategies:

```bash
# Backtest long calls — 2022 to 2024
python scripts/backtest_long_options.py \
  --type CALL \
  --start 2022-01-01 \
  --end 2024-12-31

# Backtest long puts
python scripts/backtest_long_options.py \
  --type PUT \
  --start 2022-01-01 \
  --end 2024-12-31

# Single ticker
python scripts/backtest_long_options.py \
  --type CALL \
  --tickers AAPL MSFT NVDA \
  --start 2023-01-01 \
  --end 2024-12-31
```

**Methodology:**
- Rolls options every `entry_frequency_days` (default: 21 days)
- IV recomputed daily from 20-day rolling historical volatility
- Exit hierarchy: profit target → stop loss → time stop → expiry
- T = DTE/365 (calendar-day Black-Scholes convention)
- Strike step = ~1% of stock price, rounded to $2.50

---

## ML Layer

The ML layer activates automatically once **200+ closed paper trades** are recorded.

### How it works

- **Model:** LightGBM binary classifier (win/loss) per strategy
- **Feature vector (23 features):** IV rank, IV/HV ratio, IV-RV spread, ADX, RSI, trend score, direction score, RS rank, sector encoded, DTE, spread width, short delta, confidence, PoP, VIX tier, SPY 5d return, breadth, snap-back flag, is_long_option, options flow score
- **Blend:** 60% heuristic confidence + 40% ML win probability
- **Guard:** after blending, LONG_CALL/LONG_PUT are downgraded back to spreads if blended confidence < 65%

### Check readiness

```bash
python scripts/check_outcomes.py
```

### Manual retrain

```bash
# Retrain all strategies with enough data
python scripts/retrain_models.py

# Retrain a specific strategy only
python scripts/retrain_models.py --strategy BULL_PUT_SPREAD
```

Models saved to `models/` directory. Walk-forward validation with quarter-Kelly sizing weights the most recent quarter at 2×.

---

## Testing

```bash
# Full test suite (356 tests, ~3s)
pytest tests/ -q

# With verbose output
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov=config --cov-report=term-missing

# Single module
pytest tests/test_strategies.py -v
pytest tests/test_volatility.py -v
pytest tests/test_pipeline.py -v
pytest tests/test_position_monitor.py -v
pytest tests/test_backtest.py -v
pytest tests/test_dashboard.py -v
```

Current status: **356/356 passing** (commit `ea7abf4`).

---

## Data Sources & Cost

| Source | Plan | Cost | Purpose |
|---|---|---|---|
| Polygon.io Options | Starter | $29/mo | Options chains, IV, Greeks, snapshots |
| Polygon.io Stocks | Starter | $29/mo | Real-time OHLCV |
| yfinance | Free | $0 | OHLCV fallback, options fallback |
| Finnhub | Free tier | $0 | Earnings calendar |
| Tradier | Free (brokerage) | $0 | Backup options data |
| Alpha Vantage | Free tier | $0 | Backup quotes |
| IBKR IB Gateway | Free (brokerage) | $0 | Live real-time enrichment |
| **Total** | | **$58/mo** | |

> Hetzner VPS is shared with `spx_algo` — scans staggered by 30 min.

---

## Server Setup (Hetzner)

```bash
# 1. Clone
cd /root
git clone https://github.com/theblackhat55/options_algo.git
cd options_algo

# 2. Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
nano .env   # set POLYGON_API_KEY, FINNHUB_API_KEY, WHATSAPP_NUMBER

# 4. Download data
python scripts/setup_data.py

# 5. First dry run
python -m src.pipeline.nightly_scan --dry-run

# 6. Install cron jobs (see Cron Jobs section above)
crontab -e

# 7. Start dashboard (systemd — see Streamlit Dashboard section)
systemctl enable options-dashboard
systemctl start options-dashboard
```

### Required directories (auto-created on first run)

```
output/
├── signals/          # Nightly scan JSON outputs
└── trades/
    ├── positions.json          # Open position state
    └── trade_outcomes.jsonl    # Paper trade log (ML training data)

data/processed/
├── iv_snapshots/     # Per-ticker IV history parquet files
└── ...

models/               # Trained LightGBM models (after 200+ trades)
logs/                 # Cron job log files
```

---

## Roadmap

### Completed ✅

- [x] V1 — Data pipeline, regime classifier, IV analysis
- [x] V1 — Six spread constructors + selector (regime × IV matrix)
- [x] V1 — Nightly scan + WhatsApp morning brief
- [x] V1 — Event filter, position sizing, portfolio tracker, outcome JSONL
- [x] V1 — Streamlit dashboard (9 pages)
- [x] V2 — VIX circuit breaker (CAUTION / DEFENSIVE / LIQUIDATION)
- [x] V2 — Directional balance cap (`MAX_SAME_DIRECTION_PCT`)
- [x] V2 — IV-RV spread premium filter
- [x] V2 — SPY directional gate
- [x] V2 — Sector limits per scan
- [x] V3 — Long Call / Long Put strategies (upgrade from spreads at low IV)
- [x] V3 — TA signals: S/R levels, Bollinger squeeze, divergence, breakout scanner
- [x] V3 — IBKR live data enrichment (real-time price + options flow)
- [x] V3 — IV snapshot collector → real IV rank (replaces HV proxy)
- [x] Phase 4A — Position lifecycle monitor (intraday P&L checks, auto-close)
- [x] Phase 4B — Morning brief with TA signals + long-option details
- [x] Phase 4C — Dashboard: Portfolio Overview, IV Snapshots, Ticker Analysis
- [x] Phase 4D — Walk-forward backtest for long options
- [x] Phase 4E — ML confidence blend (23-feature LightGBM, 60/40 blend)
- [x] Phase 4F — Intraday alerts: VIX spike detector + breakout scanner

### Pending 🔲

- [ ] Live Polygon real-time options chain in production scan (currently dry-run default)
- [ ] Sector rotation overlay (weight picks by sector momentum)
- [ ] ML strike optimizer (predict optimal delta given regime + IV rank)
- [ ] Automated paper→live trade execution via IBKR API
- [ ] Mobile-responsive dashboard layout
