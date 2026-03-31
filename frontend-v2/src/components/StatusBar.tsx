import { useState } from 'react'
import { useStore } from '../hooks/useStore'
import s from './StatusBar.module.css'

export default function StatusBar() {
  const store = useStore()
  const [showNotes, setShowNotes] = useState(false)

  return (
    <>
      <div className={s.bar}>
        <span className={s.item}>API: {store.wsLatencyMs > 0 ? `${store.wsLatencyMs}ms` : '—'}</span>
        <span className={s.sep}>|</span>
        <span className={s.item}>Last trade: {store.lastTradeTime || '—'}</span>
        <span className={s.sep}>|</span>
        <span className={`${s.item} ${store.connected ? s.connUp : s.connIdle}`}>
          {store.connected ? '● Connected' : '○ Connecting...'}
        </span>
        <span className={s.version} onClick={() => setShowNotes(true)}>v12.0</span>
      </div>

      {/* Release Notes Modal */}
      {showNotes && (
        <div className={s.modalOverlay} onClick={() => setShowNotes(false)}>
          <div className={s.modal} onClick={e => e.stopPropagation()}>
            <div className={s.modalHeader}>
              <span>Release Notes</span>
              <button className={s.modalClose} onClick={() => setShowNotes(false)}>✕</button>
            </div>
            <div className={s.modalBody}>
              <h3>v12.0 — 2026-03-05</h3>
              <ul>
                <li>Fix: evaluate_exit() bug — duplicate price arg</li>
                <li>Fix: Raise min_avg_volume 5K to 50K</li>
                <li>Migration: Full SQLite to PostgreSQL (25M+ rows)</li>
              </ul>
              <h3>v11.0 — Exhaustion Entry Engine</h3>
              <ul>
                <li>Exhaustion entry: wait for OR build, enter on weakness</li>
                <li>Sweep detector: liquidity sweep above OR high</li>
                <li>Pyramiding: add to winners at +3% intervals</li>
                <li>Smart tiered trailing stops (Tier 0-4)</li>
              </ul>
              <h3>v10.0 — Institution-Grade Backtesting</h3>
              <ul>
                <li>Walk-forward optimization with parameter grid</li>
                <li>Monte Carlo block bootstrap (10K simulations)</li>
                <li>Deflated Sharpe ratio for multiple testing</li>
              </ul>
              <h3>v9.0 — Multi-Broker Architecture</h3>
              <ul>
                <li>Alpaca + OANDA broker coordination</li>
                <li>Cross-broker risk management</li>
                <li>Market hours per-asset-class</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
