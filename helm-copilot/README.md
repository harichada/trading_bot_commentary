# Helm — Personal Market Co-Pilot

A dark, kinetic, trust-forward trader co-pilot UI. Institutional quality, retail-usable.

**Philosophy**: Facts before story. Wait/Do-nothing as first-class verdicts. As-of timestamps and Live/Stale trust signals everywhere. No personalized financial advice — informational/educational only.

## Quick Start

```bash
cd helm-copilot
npm install
npm run dev
```

Open [http://localhost:3001](http://localhost:3001) in your browser.

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
Ranked decision/event stream showing the bot's analysis in real-time:
- Signal BUY/SELL → **ACT** verdicts (cyan)
- Skip → **WAIT** verdicts (amber)
- Veto → **DO NOTHING** verdicts (gray)

### Decision Card (`/decision/:id`)
Full detail view for a decision:
- Verdict with confidence score
- Thesis and entry details
- Risk factors
- News sources with tier ratings
- Technical context (regime, RSI, volume)
- UI-only actions: Paper Trade, Dismiss, Mute Symbol

### Portfolio Radar (`/radar`)
Live positions grid:
- Real-time P&L (from WebSocket or REST fallback)
- Stop loss / take profit levels
- Position mode (LIVE vs SIM)
- Stale data indicators

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

When the API is unreachable, Helm automatically switches to demo mode with:
- 3 sample decision cards (BUY/WAIT/VETO examples)
- 2 sample positions
- Sample market indices
- All data clearly labeled as DEMO

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
