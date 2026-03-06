# Rudra Trading Engine — Getting Started Guide

Complete guide for setting up, running, testing, and deploying the Rudra Trading Engine.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [First-Time Setup](#first-time-setup)
4. [Running the Apps](#running-the-apps)
5. [Configuration](#configuration)
6. [Testing](#testing)
7. [Backtesting](#backtesting)
8. [LLM Supervisor (Rudra)](#llm-supervisor-rudra)
9. [Multi-Broker Trading](#multi-broker-trading)
10. [Secret Management](#secret-management)
11. [Building & Deploying (Docker)](#building--deploying-docker)
12. [Monitoring & Operations](#monitoring--operations)
13. [Troubleshooting](#troubleshooting)
14. [Project Structure](#project-structure)

---

## Overview

Three independent FastAPI apps, each a standalone monolith with an embedded HTML dashboard:

| App | File | Default Port | Broker API | Purpose |
|-----|------|-------------|------------|---------|
| **Rudra Trading Engine** | `gap_fade_app.py` | 8002 (`GAP_FADE_PORT`) | Alpaca, OANDA | Gap fade + intraday strategies, LLM supervisor |
| **Main Trading Bot** | `trading_bot_commentary_updated.py` | 8000 | Schwab | Commentary-driven trading bot |
| **Backtest Dashboard** | `claude_backtest_app.py` | 8001 | Schwab (optional) | Rules-engine backtesting UI |

The **Rudra Trading Engine** (`gap_fade_app.py`) is the primary, actively developed app. This guide focuses on it.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Tested on 3.11, 3.12 |
| PostgreSQL | 14+ | Database for price data, trades, state |
| Git | 2.30+ | With git-crypt for secret management |
| git-crypt | 0.7+ | `sudo apt install git-crypt` |
| GPG | 2.2+ | For git-crypt key management |

**Broker accounts** (at least one):
- [Alpaca](https://alpaca.markets) — Free paper trading account (US equities)
- [OANDA](https://www.oanda.com) — Free practice account (forex)
- [Schwab](https://developer.schwab.com) — For main bot only

---

## First-Time Setup

### 1. Clone and Decrypt Secrets

```bash
git clone git@github.com:harichada/trading_bot_commentary.git
cd trading_bot_commentary

# Unlock encrypted .env files (requires authorized GPG key)
./scripts/setup-encryption.sh

# If you don't have a GPG key yet, the script will generate one.
# Ask the repo owner to add your key: ./scripts/onboard-team-member.sh your@email.com
```

See [docs/SECRETS.md](docs/SECRETS.md) for full details on secret management.

### 2. Install Dependencies

```bash
# Rudra Trading Engine only (recommended for most users)
pip install -r requirements-gap-fade.txt

# Full stack (all three apps, ML, NLP, etc.)
pip install -r requirements.txt

# PostgreSQL driver (included in requirements but listed for clarity)
pip install psycopg2-binary
```

### 3. Set Up PostgreSQL

```bash
# Create database and user
sudo -u postgres psql <<SQL
CREATE USER rudra WITH PASSWORD 'rudra_dev_2024';
CREATE DATABASE rudra_dev OWNER rudra;
SQL

# Or use Docker (from infra):
# make deploy-monitoring  → starts PostgreSQL at localhost:5432
```

The app auto-creates all 14 tables on first startup. No manual schema migration needed.

### 4. Configure Environment

```bash
# Copy the template
cp .env.example .env_dev

# Edit with your API keys
nano .env_dev
```

**Minimum required** for paper trading:

```bash
# Alpaca (get from https://app.alpaca.markets → Paper Trading → API Keys)
ALPACA_API_KEY=PK...
ALPACA_SECRET_KEY=...

# PostgreSQL
DATABASE_URL=postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev

# System
GAP_FADE_PORT=8003
```

### 5. Load Historical Price Data

The app needs historical daily bars for gap scanning and backtesting:

```bash
# Start the app — it will scan and cache data from Alpaca on first run
python gap_fade_app.py

# Or use the migration script for bulk loading from an existing SQLite DB:
python migrate_sqlite_to_pg.py
```

The database currently holds ~25M rows of daily bars (2006–present, ~2000+ symbols).

---

## Running the Apps

### Rudra Trading Engine (primary)

```bash
# Load env vars and start
export $(grep -v '^#' .env_dev | xargs)
python gap_fade_app.py

# Dashboard: http://localhost:8003
```

The dashboard shows:
- Live equity curve and P&L
- Active positions with real-time prices
- Gap scan candidates
- Rudra LLM chat interface
- Backtest controls
- Configuration panel
- Activity feed and trading journal

### Main Trading Bot

```bash
export $(grep -v '^#' .env_dev | xargs)
python trading_bot_commentary_updated.py
# Dashboard: http://localhost:8000
```

### Backtest Dashboard

```bash
export $(grep -v '^#' .env_dev | xargs)
python claude_backtest_app.py
# Dashboard: http://localhost:8001
```

### Professional Bot (safe mode)

Wraps the main bot with `safe_imports.py` to block xgboost/lightgbm (prevents segfaults):

```bash
python run_professional_bot.py
```

---

## Configuration

### Runtime Config (via Dashboard)

All trading parameters are tunable at runtime through the dashboard's **Config** panel. Changes take effect immediately and persist across restarts (stored in PostgreSQL).

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gap_threshold` | 0.02 | Minimum gap % to qualify (2%) |
| `min_avg_volume` | 50000 | Minimum 20-day average volume |
| `max_positions` | 3 | Maximum concurrent positions |
| `stop_loss_pct` | 0.03 | Stop loss distance (3%) |
| `position_size_pct` | 0.10 | Position size as % of equity |
| `adaptive_stops` | true | ATR-based dynamic stops |
| `regime_filter` | true | Skip trades in unfavorable regimes |
| `reentry_enabled` | true | Re-enter after stop-out if reversal |
| `gap_down_enabled` | true | Long entries on gap-downs |
| `llm_enabled` | false | Enable Rudra LLM supervisor |

### Config Profiles

Save and load named config snapshots via the dashboard:

- **Save**: Config panel → "Save Profile" → name it
- **Load**: Config panel → "Load Profile" → select one
- **Export/Import**: Download as JSON, upload to another instance

### Drawdown Circuit Breakers

Graduated risk reduction during drawdowns:

| Tier | Drawdown | Action |
|------|----------|--------|
| Tier 1 | 5% | Reduce position sizes by 50% |
| Tier 2 | 8% | Reduce by 75%, skip new entries |
| Hard Stop | 10% | Halt all trading |

---

## Testing

Tests use `pytest` + `pytest-asyncio`. Each test file is self-contained (no shared conftest).

```bash
# Run the most comprehensive test suite
python -m pytest test_institutional_core.py -v

# Run all test files
python -m pytest test_*.py -v

# Specific test suites
python -m pytest test_coordinator.py -v          # Multi-broker coordinator
python -m pytest test_oanda_adapter.py -v         # OANDA broker adapter
python -m pytest test_broker_adapter.py -v        # Alpaca broker adapter
python -m pytest test_indicator_engine.py -v      # Technical indicators
python -m pytest test_strategy_selector.py -v     # Strategy routing
python -m pytest test_intraday_orb.py -v          # ORB breakout strategy
python -m pytest test_momentum_strategy.py -v     # Momentum surge strategy
python -m pytest test_pullback_strategy.py -v     # Pullback entry strategy
python -m pytest test_range_strategy.py -v        # Range trade strategy
python -m pytest test_classic_profit_protection.py -v  # Profit protection
python -m pytest test_gap_fade_backtester.py -v   # Backtester
python -m pytest test_professional_bot.py -v      # Professional bot
python -m pytest test_components_isolated.py -v   # Isolated components
python -m pytest test_safe_imports.py -v          # xgboost/lightgbm blocking

# Run with coverage
python -m pytest test_institutional_core.py -v --cov=. --cov-report=term-missing
```

### Syntax Check (quick validation)

```bash
python -c "import ast; ast.parse(open('gap_fade_app.py').read()); print('OK')"
```

### Smoke Test (imports and startup)

```bash
python -c "
from gap_fade_app import APP_VERSION, RELEASE_NOTES_HTML, get_price_db
print(f'Version: {APP_VERSION}')
print(f'Release notes: {len(RELEASE_NOTES_HTML)} chars')
db = get_price_db()
print(f'DB connected, trades: {db.count_trades()}')
"
```

---

## Backtesting

### Via Dashboard

1. Open the dashboard → **Backtest** tab
2. Set date range, symbols, and strategy parameters
3. Toggle feature flags (adaptive stops, regime filter, etc.)
4. Click **Run Backtest**
5. Results show equity curve, trade list, and performance metrics

### Via CLI

```bash
# Gap fade backtester (PostgreSQL data source)
python gap_fade_backtester.py

# Vectorbt-powered backtester
python bt_backtest.py
```

### Key Metrics

| Metric | Calculation |
|--------|-------------|
| Sharpe Ratio | Daily equity returns (not per-trade) |
| Max Drawdown | Equity curve peak-to-trough |
| Win Rate | Winning trades / total trades |
| Profit Factor | Gross profit / gross loss |
| Average R | Average P&L / average risk |

---

## LLM Supervisor (Rudra)

Rudra is an autonomous LLM that monitors positions, makes trading decisions, and manages risk.

### Setup

```bash
# Install Ollama (https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# Build the Rudra model
ollama create rudra -f Modelfile.gapfade

# Enable in config (dashboard or env)
# Set llm_enabled=true in the Config panel
```

### Supported LLM Providers

| Provider | Config | Model |
|----------|--------|-------|
| Ollama (local) | `llm_provider=ollama` | `rudra` or `gpt-oss:20b` |
| OpenAI-compatible | `llm_provider=openai` | Any GPT model |
| Groq | `llm_provider=groq` | Llama, Mixtral |
| Together | `llm_provider=together` | Any hosted model |

### What Rudra Does

- **Scans** for gap candidates and evaluates entry quality
- **Enters** positions with LLM-reasoned sizing and stop placement
- **Monitors** open positions and adjusts stops
- **Exits** based on target, pullback, or reversal signals
- **Reflects** on trades and updates its decision journal
- **Learns** from past trades and adapts strategy

### Chat Interface

The dashboard has a built-in chat. Example commands:

```
"What positions are open?"
"Close AAPL position"
"Switch to classic gap fade strategy"
"Review current risk exposure"
"Pause trading for 30 minutes"
```

---

## Multi-Broker Trading

The engine supports simultaneous trading across multiple brokers via the `MultiBrokerCoordinator`.

### Supported Brokers

| Broker | Asset Class | Adapter | Env Vars |
|--------|-------------|---------|----------|
| Alpaca | US Equities | `AlpacaBrokerAdapter` | `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` |
| OANDA | Forex | `OandaBrokerAdapter` | `OANDA_ACCOUNT_ID`, `OANDA_TOKEN` |

### How It Works

- Alpaca is always created as the primary broker
- OANDA is automatically added if its env vars are set
- Each broker gets its own `GapFadeLiveTrader` with separate state
- `CombinedRiskManager` checks cross-broker daily loss, drawdown, and position count

### API Endpoints

```
GET  /api/brokers                    # List all brokers and status
GET  /api/state/combined             # Combined state across all brokers
GET  /api/state/{broker_id}          # State for a specific broker
POST /api/brokers/{id}/enable        # Enable a broker
POST /api/brokers/{id}/disable       # Disable a broker
POST /api/brokers/{id}/start         # Start trading on a broker
POST /api/brokers/{id}/stop          # Stop trading on a broker
```

---

## Secret Management

All `.env` files are encrypted in the repo using **git-crypt** (AES-256). See [docs/SECRETS.md](docs/SECRETS.md) for full documentation.

### Quick Reference

```bash
# First-time setup (after clone)
./scripts/setup-encryption.sh

# Add a team member
./scripts/onboard-team-member.sh their@email.com their-key.gpg

# Daily use — edit normally, encryption is transparent
nano .env_dev
git add .env_dev && git commit -m "Update keys"  # encrypted in git
```

### Backup Your GPG Key

```bash
# Export (store securely — password manager, encrypted drive)
gpg --armor --export-secret-keys harikishorereddy@gmail.com > gpg-private-key.gpg

# Also export the git-crypt symmetric key
git-crypt export-key ~/git-crypt-key.bin
```

---

## Building & Deploying (Docker)

### Build Pipeline

```bash
cd ~/claude/infra

# Build image from current branch
make build

# Test the image
make test-image

# Deploy to environments
make deploy-dev                    # Dev (port 8004)
make deploy-sit TAG=<sha>          # SIT (port 8005)
make deploy-prod TAG=<sha>         # Prod (port 8003, requires confirmation)
```

### How Builds Work

1. **Infra bootstrap** (`infra/scripts/build.sh`) clones/fetches the trading bot repo
2. **Repo build script** (`scripts/build.sh`) runs:
   - Resolves version from git tags → `build_info.json`
   - Auto-generates `CHANGELOG.md` entry from git commit messages
   - Copies infra modules into build context
   - Runs `docker build` with version args
3. Docker image tagged with git SHA, version, and `latest`

### Version & Release Notes

- **Version**: Auto-detected from git tags (e.g., `v12.2` → shown in dashboard footer)
- **Release notes**: Parsed from `CHANGELOG.md` at app startup → displayed in dashboard modal
- **CHANGELOG.md**: Auto-updated by `scripts/build.sh` from commit messages on new tags

```bash
# Typical release workflow
git commit -m "Fix: description of fix"
git tag -a v13.0 -m "v13.0: summary"
git push origin feature/backtesting --tags
cd ~/claude/infra && make build    # CHANGELOG auto-generated, version baked in
```

### Environments

| Environment | Port | Env File | Database |
|-------------|------|----------|----------|
| Dev | 8004 | `.env_dev` | `rudra_dev` |
| SIT | 8005 | `.env_sit` | `rudra_sit` |
| Prod | 8003 | `.env_prod` | `rudra_prod` |

### Systemd (bare-metal alternative)

```bash
# Install as a systemd service
sudo ./install-service.sh

# Manage
sudo systemctl start gap-fade
sudo systemctl status gap-fade
journalctl -u gap-fade -f
```

---

## Monitoring & Operations

### Health Check

```bash
curl http://localhost:8003/api/health
```

### Key API Endpoints

```
GET  /api/health                     # Health check
GET  /api/state                      # Current trading state
GET  /api/positions                  # Active positions
GET  /api/trades/history             # Trade history
GET  /api/config                     # Current config
POST /api/config                     # Update config
POST /api/start                      # Start trading
POST /api/stop                       # Stop trading
GET  /api/backtest/status            # Backtest progress
GET  /api/journal                    # Trading journal entries
WS   /ws                            # WebSocket for live updates
```

### Database

```bash
# Connect to dev database
psql -U rudra -d rudra_dev

# Key tables
\dt                                  # List all tables (14 total)
SELECT COUNT(*) FROM daily_bars;     # ~25M rows of price data
SELECT COUNT(*) FROM trades;         # Completed trades
SELECT * FROM trader_state;          # App state (replaces JSON files)
SELECT * FROM config_history ORDER BY id DESC LIMIT 5;  # Config audit trail
```

### Logs

```bash
# App logs (stdout)
python gap_fade_app.py 2>&1 | tee app.log

# Docker logs
make logs-dev    # or logs-sit, logs-prod

# Systemd logs
journalctl -u gap-fade -f --since "1 hour ago"
```

---

## Troubleshooting

### App Won't Start

| Symptom | Cause | Fix |
|---------|-------|-----|
| `psycopg2.OperationalError: connection refused` | PostgreSQL not running | `sudo systemctl start postgresql` |
| `ModuleNotFoundError: No module named 'psycopg2'` | Missing driver | `pip install psycopg2-binary` |
| `ALPACA_API_KEY not set` | Missing env vars | `export $(grep -v '^#' .env_dev \| xargs)` |
| Port already in use | Another instance running | `lsof -i :8003` → kill it, or change `GAP_FADE_PORT` |

### Trading Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| No gap candidates | Market closed or filters too strict | Lower `gap_threshold`, check market hours |
| Positions not opening | `max_positions` reached or halted | Check daily stats, circuit breaker status |
| LLM not responding | Ollama not running or wrong model | `ollama list`, check `llm_model` config |
| State not persisting | Database connection lost | Check `DATABASE_URL`, verify PostgreSQL |

### Git-Crypt Issues

| Symptom | Fix |
|---------|-----|
| `.env_dev` shows binary | Run `git-crypt unlock` |
| "no GPG secret key" | Import your key: `gpg --import private-key.gpg` |
| New clone can't decrypt | Ask repo owner to run `onboard-team-member.sh` |

See [docs/SECRETS.md](docs/SECRETS.md) for more troubleshooting.

### Reset State

```bash
# Clear trader state in database (keeps price data and trades)
psql -U rudra -d rudra_dev -c "DELETE FROM trader_state;"

# Full reset (careful — deletes trade history too)
psql -U rudra -d rudra_dev -c "DELETE FROM trades; DELETE FROM trader_state; DELETE FROM journal_entries;"
```

---

## Project Structure

```
trading_bot_commentary/
├── gap_fade_app.py                  # Rudra Trading Engine (main app, ~20K lines)
├── trading_bot_commentary_updated.py # Main trading bot (~16K lines)
├── claude_backtest_app.py           # Backtest dashboard (~10K lines)
├── run_professional_bot.py          # Safe-mode wrapper for main bot
│
├── .env_dev                         # Dev secrets (encrypted via git-crypt)
├── .env.example                     # Template showing required env vars
├── .gitattributes                   # git-crypt encryption rules
│
├── CHANGELOG.md                     # Release notes (auto-updated by build)
├── CLAUDE.md                        # AI assistant instructions
│
├── brokers/                         # Multi-broker adapters
│   ├── base.py                      #   AbstractBroker interface
│   ├── alpaca_adapter.py            #   Alpaca (US equities)
│   ├── oanda_adapter.py             #   OANDA (forex)
│   ├── coordinator.py               #   MultiBrokerCoordinator
│   └── market_hours.py              #   Market session detection
│
├── gap_fade_strategies/             # Pluggable trading strategies
│   ├── classic_gap_fade.py          #   Core gap fade strategy
│   ├── orb_breakout.py              #   Opening range breakout
│   ├── momentum_surge.py            #   Momentum surge
│   ├── pullback_entry.py            #   Pullback entry
│   ├── range_trade.py               #   Range trading
│   ├── vwap_gap_fade.py             #   VWAP-based gap fade
│   ├── confluence_gap.py            #   Multi-signal confluence
│   ├── minervini_trend.py           #   Minervini trend template
│   ├── strategy_selector.py         #   Condition-based routing
│   └── indicators.py                #   Technical indicator engine
│
├── scripts/
│   ├── build.sh                     # Docker build (version, changelog, image)
│   ├── setup-encryption.sh          # One-time git-crypt setup
│   └── onboard-team-member.sh       # Add GPG user to git-crypt
│
├── docs/
│   └── SECRETS.md                   # Secret management documentation
│
├── auth.py                          # OAuth2 (Google/GitHub/Discord)
├── safe_imports.py                  # Block xgboost/lightgbm (segfault fix)
├── Modelfile.gapfade                # Ollama model definition for Rudra
├── migrate_sqlite_to_pg.py          # One-time SQLite → PostgreSQL migration
│
├── requirements-gap-fade.txt        # Minimal deps (Rudra engine only)
├── requirements.txt                 # Full deps (all apps)
│
├── gap-fade.service                 # systemd unit file
├── install-service.sh               # systemd installer
│
└── test_*.py                        # Test suites (self-contained, pytest)
```

### Database Schema (14 tables)

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `daily_bars` | Historical price data (~25M rows) | `symbol, date, OHLCV` |
| `trades` | Completed trade records | `symbol, entry/exit price, pnl` |
| `trader_state` | App state persistence | `key, state_json` |
| `journal_entries` | Decision audit trail | `timestamp, type, content` |
| `config_profiles` | Named config snapshots | `name, config_json` |
| `config_history` | Config change audit log | `timestamp, changes_json` |
| `llm_calls` | LLM call log | `timestamp, prompt, response` |
| `market_events` | Detected market events | `timestamp, type, data` |
| `api_calls` | API call audit log | `endpoint, status, response_time` |
| `performance_snapshots` | Periodic equity snapshots | `equity, trades, drawdown` |
| `signals_intraday` | Intraday strategy signals | `symbol, strategy, direction` |
| `candidates_rejected` | Filtered-out candidates | `symbol, rejection_reason` |

---

## Getting Help

- **CLAUDE.md**: AI assistant instructions and architecture reference
- **CHANGELOG.md**: Version history and recent changes
- **docs/SECRETS.md**: Secret management with git-crypt
- **Dashboard**: Click the version number in the footer for release notes
