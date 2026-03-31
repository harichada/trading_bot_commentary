import { useState } from 'react'
import s from './Guide.module.css'

const SECTIONS = [
  { id: 'overview', title: 'Overview', content: `
Rudra Trading Engine is an automated gap-fade trading system for US equities. It scans for overnight gap-up stocks, waits for exhaustion signals after the opening range builds (9:30-9:45), and enters short positions when the gap shows signs of fading.

The engine runs on Alpaca (paper or live), with real-time data from Alpaca SIP feed and historical data from PostgreSQL.
  `},
  { id: 'strategies', title: 'Strategies', content: `
**Exhaustion Gap Fade** (PRIMARY) — Waits until 9:45 for Opening Range, enters when price < OR high AND price < gap open. PF 6.94, 83% WR on live trades.

**Classic Gap Fade** — Blind 9:31 market entry on top gap-up candidates. Legacy strategy, underperforms exhaustion entry.

**Gap Continuation** — Shorts big gap-downs (>4%), wide 2.5% stops, 3pm time exit. PF 1.14 backtested.

**Gap Bounce** — Longs oversold gap-downs (gap>4% + RSI<40), no stop/target, time exit at 3pm.
  `},
  { id: 'risk', title: 'Risk Management', content: `
- **Max Daily Loss**: 2% of equity — halts all trading for the day
- **Max Drawdown**: 5% — emergency stop, requires manual restart
- **Position Sizing**: Based on stop distance, max 20% of equity per position
- **GTC Stop Orders**: Broker-side stops survive overnight, synced when trailing tightens
- **EOD Watchdog**: Independent task that force-closes all positions by 4:00 PM ET
- **Circuit Breaker**: 3 consecutive losses triggers cooldown period
  `},
  { id: 'architecture', title: 'Architecture', content: `
- **gap_fade_app.py** — Main FastAPI application (port 8003)
- **PostgreSQL** — Price database (25M+ daily bars)
- **Alpaca API** — SIP data feed + order execution
- **Ollama/Claude** — LLM supervisor for trade decisions (optional)
- **Telegram/Email** — Trade alerts via alerter.py
- **React Frontend** — This dashboard (frontend-v2/)
  `},
  { id: 'api', title: 'API Endpoints', content: `
- \`GET /api/health\` — System health check
- \`GET /api/state\` — Full trader state (equity, positions, candidates)
- \`GET /api/trades\` — Historical trade log
- \`GET /api/strategies\` — Available strategies
- \`POST /api/start\` — Start trading loop
- \`POST /api/stop\` — Stop trading loop
- \`POST /api/scan\` — Run pre-market scan
- \`GET /api/db/stats\` — Database statistics
- \`POST /api/llm/chat\` — Chat with Rudra AI
- \`POST /api/backtest\` — Run backtest
  `},
  { id: 'config', title: 'Configuration', content: `
All config is managed via \`GapFadeConfig\` dataclass. Key parameters:

- \`gap_threshold\`: Minimum gap % to consider (default 3%)
- \`max_positions\`: Maximum concurrent positions (default 5)
- \`stop_loss_pct\`: Initial stop loss percentage
- \`entry_cutoff_hour/min\`: Last allowed entry time
- \`pyramid_enabled\`: Add to winners at +3% intervals
- \`llm_enabled\`: Enable Rudra LLM supervisor
- \`catalyst_enabled\`: Filter by earnings/M&A catalysts
  `},
]

export default function Guide() {
  const [active, setActive] = useState('overview')
  const section = SECTIONS.find(s => s.id === active) ?? SECTIONS[0]

  return (
    <div className={s.layout}>
      <nav className={s.nav}>
        <div className={s.navTitle}>Documentation</div>
        {SECTIONS.map(sec => (
          <button key={sec.id}
            className={`${s.navItem} ${active === sec.id ? s.navActive : ''}`}
            onClick={() => setActive(sec.id)}
          >{sec.title}</button>
        ))}
      </nav>
      <div className={s.content}>
        <h2 className={s.heading}>{section.title}</h2>
        <div className={s.body}>
          {section.content.split('\n').map((line, i) => {
            const trimmed = line.trim()
            if (!trimmed) return <br key={i} />
            if (trimmed.startsWith('- ')) return <div key={i} className={s.listItem}>{trimmed.slice(2).replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/`(.*?)`/g, '<code>$1</code>')}</div>
            return <p key={i} dangerouslySetInnerHTML={{ __html: trimmed.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/`(.*?)`/g, '<code>$1</code>') }} />
          })}
        </div>
      </div>
    </div>
  )
}
