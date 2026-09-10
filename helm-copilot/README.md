# Helm — Personal Market Co-Pilot

A dark, kinetic, trust-forward trader co-pilot UI. Institutional quality, retail-usable.

**Philosophy**: Facts before story. Wait/Do-nothing as first-class verdicts. As-of timestamps and Live/Stale trust signals everywhere. No personalized financial advice — informational/educational only.

## Quick Start

```bash
# From monorepo root
cd helm-copilot
npm install
npm run dev
```

Opens at [http://localhost:3001](http://localhost:3001).

**With a running bot** (typically `:9000`):
```bash
# Default wires to http://127.0.0.1:9000
npm run dev
```

**Standalone demo** (no bot needed):
```bash
# Falls back to demo fixtures automatically
npm run dev
# → UI shows DEMO badge, simulated data
```

## Configuration

Create a `.env` file (or copy from `.env.example`):

```bash
cp .env.example .env
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_BASE` | `http://127.0.0.1:9000` | Trading bot API base URL |
| `VITE_API_KEY` | _(empty)_ | Optional Bearer token for API auth |

## API Contract

Helm wires defensively to these endpoints (gracefully falls back to demo fixtures when unavailable):

### Required Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Bot running state, mode, positions count |
| `/api/positions/db` | GET | Current open positions with P&L |
| `/api/system-stats` | GET | NewsBus stats, supervisor status, profile info |
| `/api/news-bus/{symbol}` | GET | News items for a symbol (query: `max_age_sec`) |

### Optional Endpoints (probed on startup)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/market-indices` | GET | SPY, DIA, QQQ, IWM, VIX prices |
| `/api/account-stats` | GET | Balance, buying power, day P&L |
| `/api/decisions` | GET | Engine decisions from Postgres |
| `/api/decision-snapshots` | GET | ML decision snapshots with full context |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `/ws` | Real-time dashboard updates (positions, screener, account) |

Query param: `?token=<auth_token>` if authentication is required.

**Message Types:**
- `dashboard_update` — Account, positions, screener, live quotes
- `sentiment_update` — Market sentiment data
- `commentary` — Bot commentary messages

## Screens

### Live Pulse (`/`)
Ranked decision stream — 8-second glance design:
- **ACT** (cyan) — signal_buy / signal_sell
- **WAIT** (amber) — skip, conditions unclear
- **PASS** (gray) — veto, blocked by risk guard

Smooth stream-insert animation, as-of chips on every card.

### Decision Card (`/decision/:id`)
Full detail view — verdict + key numbers above the fold:
- Hero banner: symbol, verdict badge, price, as-of timestamp
- Metrics strip: Confidence / RSI / Volume / Sentiment
- Thesis, risks, news sources with tier badges
- Technical context (regime, VIX, strategy, gate)
- UI-only actions: Paper Trade, Dismiss, Mute Symbol

### Portfolio Radar (`/radar`)
Positions grid with Live vs Simulated split:
- Separate cards for LIVE (broker) and SIM (paper) positions
- Readable P&L: price, dollar gain/loss, percent
- Managed vs hands-off indicator (cyan border = bot-managed)
- Stop/target levels, entry time

### Inbox (`/inbox`) — Stub
Placeholder for future alerts and notifications.

### Settings (`/settings`) — Stub
Connection info and placeholder settings panels.

## Trust Signals

Helm shows data freshness everywhere:

| Badge | Meaning |
|-------|---------|
| 🟢 **LIVE** | Connected, data < 5s old |
| 🟡 **STALE** | Data > 30s old |
| 🟣 **DEMO** | Offline, showing fixtures |
| 🔴 **OFFLINE** | Cannot reach API |

The persistent **Trust Strip** at the bottom shows:
- Connection state
- As-of timestamp
- Bot active/stopped
- Trading mode (LIVE/SIM/OFF)
- Compliance disclaimer

## Demo Mode

When the API is unreachable, Helm automatically switches to demo mode:
- 3 sample decision cards (ACT-BUY / WAIT / PASS-VETO)
- 2 sample positions with P&L
- Sample market indices (SPY, DIA, QQQ, IWM, VIX)
- **All data clearly labeled DEMO** — never shown as LIVE

The UI degrades gracefully: connection banner appears, trust badges flip to DEMO/OFFLINE, and demo fixtures populate the screens.

## Project Structure

```
helm-copilot/
├── src/
│   ├── api/           # API client and TypeScript types
│   │   ├── client.ts  # Fetch wrapper, WebSocket factory
│   │   └── types.ts   # Response/message type definitions
│   ├── components/    # React components
│   │   ├── ui/        # Base UI components (Badge, Card, Button)
│   │   ├── Shell.tsx  # Main layout with nav
│   │   ├── LivePulse.tsx
│   │   ├── DecisionCard.tsx
│   │   ├── PortfolioRadar.tsx
│   │   └── ...
│   ├── hooks/         # React hooks
│   │   ├── useApi.ts  # Data fetching with fallback
│   │   ├── useWebSocket.ts
│   │   └── useConnectionStatus.ts
│   └── fixtures/      # Demo data
│       └── demo.ts    # Sample decisions, positions, indices
├── .env.example
├── package.json
├── tailwind.config.js
├── vite.config.ts
└── README.md
```

## Build

```bash
npm run build
```

Output in `dist/` — static files ready for any hosting.

## Development

```bash
npm run dev      # Start dev server on :3001
npm run build    # Production build
npm run preview  # Preview production build
npm run lint     # ESLint check
```

## Stack

- **Vite** — Fast dev/build
- **React 18** — UI framework
- **TypeScript** — Type safety
- **Tailwind CSS** — Utility-first styling
- **React Router** — Client-side routing
- **Lucide React** — Icons

## Compliance Notice

Helm is an **informational/educational tool only**. It does not provide personalized financial advice. All trading decisions are made at your own risk. Past performance is not indicative of future results.

---

Built for the trading bot monorepo. Independent of `dashboardUI/` (Next.js).
