import { useState, useEffect, useRef, useCallback } from 'react'
import { createChart, type IChartApi, type ISeriesApi, ColorType, LineStyle, CandlestickSeries, LineSeries } from 'lightweight-charts'
import s from './Replay.module.css'

const LAB_WS = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/lab/replay`

interface Trade { symbol: string; action: string; price: number; direction: string; pnl?: number; reason?: string; ts: string; shares?: number; stop?: number; target?: number; equity?: number }

interface ChartRef {
  chart: IChartApi
  candle: ISeriesApi<'Candlestick'>
  vwap: ISeriesApi<'Line'>
  barTimes: Set<number>  // track all candle timestamps for marker matching
}

function SymbolChart({ symbol, chartRef }: { symbol: string; chartRef: React.MutableRefObject<Record<string, ChartRef>> }) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return

    // Clear previous chart if exists
    if (chartRef.current[symbol]) {
      try { chartRef.current[symbol].chart.remove() } catch { /* */ }
    }

    const chart = createChart(containerRef.current, {
      layout: { background: { type: ColorType.Solid, color: '#0d0d0d' }, textColor: '#888', fontFamily: 'JetBrains Mono', fontSize: 10 },
      grid: { vertLines: { color: 'rgba(255,255,255,0.02)' }, horzLines: { color: 'rgba(255,255,255,0.02)' } },
      crosshair: { vertLine: { color: '#00FFBB', width: 1, style: LineStyle.Dashed }, horzLine: { color: '#00FFBB', width: 1, style: LineStyle.Dashed } },
      rightPriceScale: { borderColor: 'rgba(255,255,255,0.06)' },
      timeScale: { borderColor: 'rgba(255,255,255,0.06)', timeVisible: true, secondsVisible: false },
      height: 280, width: containerRef.current.offsetWidth,
    })

    let candle: ISeriesApi<'Candlestick'>
    let vwap: ISeriesApi<'Line'>
    try {
      candle = chart.addSeries(CandlestickSeries, { upColor: '#00FFBB', downColor: '#FF4D6A', borderUpColor: '#00FFBB', borderDownColor: '#FF4D6A', wickUpColor: '#00FFBB', wickDownColor: '#FF4D6A' })
      vwap = chart.addSeries(LineSeries, { color: '#3B82F6', lineWidth: 1, lineStyle: LineStyle.Dashed, priceLineVisible: false })
    } catch {
      candle = (chart as any).addCandlestickSeries({ upColor: '#00FFBB', downColor: '#FF4D6A', borderUpColor: '#00FFBB', borderDownColor: '#FF4D6A', wickUpColor: '#00FFBB', wickDownColor: '#FF4D6A' })
      vwap = (chart as any).addLineSeries({ color: '#3B82F6', lineWidth: 1, lineStyle: LineStyle.Dashed, priceLineVisible: false })
    }

    chartRef.current[symbol] = { chart, candle, vwap, barTimes: new Set() }

    const ro = new ResizeObserver(() => {
      if (containerRef.current) chart.applyOptions({ width: containerRef.current.offsetWidth })
    })
    ro.observe(containerRef.current)

    return () => { ro.disconnect(); chart.remove(); delete chartRef.current[symbol] }
  }, [symbol])

  return <div ref={containerRef} className={s.chartBox} />
}

export default function Replay() {
  const [symbols, setSymbols] = useState('TSLA,NVDA,AAPL,AMD,PLTR,MSFT')
  const [strategy, setStrategy] = useState('orb_breakout')
  const [date, setDate] = useState('2026-03-10')
  const [endDate, setEndDate] = useState('')
  const [currentDay, setCurrentDay] = useState('')
  const [dayCount, setDayCount] = useState(0)
  const [speed, setSpeed] = useState(10)
  const [strategies, setStrategies] = useState<string[]>([])
  const [running, setRunning] = useState(false)
  const [activeSymbols, setActiveSymbols] = useState<string[]>([])
  const [trades, setTrades] = useState<Trade[]>([])
  const [metrics, setMetrics] = useState<Record<string, unknown> | null>(null)
  const [barCount, setBarCount] = useState(0)
  const [totalBars, setTotalBars] = useState(0)
  const [warning, setWarning] = useState('')

  const chartsRef = useRef<Record<string, ChartRef>>({})
  const markersRef = useRef<Record<string, Array<{ time: number; position: string; color: string; shape: string; text: string; size: number }>>>({})
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    fetch('/api/lab/strategies').then(r => r.json()).then(d => {
      setStrategies((d.strategies ?? []).map((s: Record<string, unknown>) => String(s.id)))
    }).catch(() => {})
  }, [])

  const start = () => {
    const syms = symbols.split(',').map(s => s.trim().toUpperCase()).filter(Boolean)
    if (!syms.length) return

    // Close old WS
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null }

    // Reset state
    setTrades([])
    setMetrics(null)
    setBarCount(0)
    setWarning('')
    markersRef.current = {}

    // Set symbols first so charts mount
    setActiveSymbols(syms)
    setRunning(true)

    // Wait for charts to mount, then connect
    setTimeout(() => {
      // Clear existing chart data
      for (const sym of syms) {
        const ref = chartsRef.current[sym]
        if (ref) {
          ref.candle.setData([])
          ref.vwap.setData([])
          ref.barTimes.clear()
          try { ref.candle.setMarkers([]) } catch { /* */ }
        }
        markersRef.current[sym] = []
      }

      const ws = new WebSocket(LAB_WS)
      wsRef.current = ws

      ws.onopen = () => {
        ws.send(JSON.stringify({ action: 'start', strategy_id: strategy, date, end_date: endDate || '', symbols: syms, speed, max_positions: syms.length }))
      }

      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data)
          handleMessage(msg)
        } catch { /* */ }
      }

      ws.onclose = () => setRunning(false)
      ws.onerror = () => setRunning(false)
    }, 800)
  }

  const handleMessage = useCallback((msg: Record<string, unknown>) => {
    const type = msg.type as string

    if (type === 'init') {
      setTotalBars(msg.total_bars as number ?? 0)
      setDayCount(msg.days as number ?? 1)
    }

    if (type === 'day_change') {
      setCurrentDay(msg.date as string ?? '')
    }

    if (type === 'warning') {
      setWarning(msg.message as string ?? '')
    }

    if (type === 'bar') {
      const sym = msg.symbol as string
      const ref = chartsRef.current[sym]
      if (ref) {
        const time = Math.floor(new Date(msg.ts as string).getTime() / 1000) as any
        ref.candle.update({ time, open: msg.o as number, high: msg.h as number, low: msg.l as number, close: msg.c as number })
        ref.barTimes.add(time)
        // Auto-scroll to latest
        ref.chart.timeScale().scrollToRealTime()
      }
      setBarCount(msg.i as number ?? 0)
    }

    if (type === 'indicator') {
      const sym = msg.symbol as string
      const ref = chartsRef.current[sym]
      if (ref && (msg.vwap as number) > 0) {
        // Use the latest bar time for VWAP point
        const times = [...ref.barTimes]
        if (times.length > 0) {
          const time = times[times.length - 1] as any
          ref.vwap.update({ time, value: msg.vwap as number })
        }
      }
    }

    if (type === 'trade') {
      const t: Trade = msg as any
      setTrades(prev => [t, ...prev])

      // Add marker to chart
      const sym = msg.symbol as string
      const markers = markersRef.current[sym]
      const ref = chartsRef.current[sym]
      if (markers && ref) {
        const rawTime = Math.floor(new Date(msg.ts as string).getTime() / 1000)
        // Find closest bar time (markers must match candle times exactly)
        const times = [...ref.barTimes]
        let time = rawTime
        if (times.length > 0) {
          time = times.reduce((closest, t) => Math.abs(t - rawTime) < Math.abs(closest - rawTime) ? t : closest)
        }

        const isEntry = msg.action === 'entry'
        const isLong = msg.direction === 'long'
        const pnl = msg.pnl as number ?? 0

        markers.push({
          time: time as any,
          position: isEntry ? (isLong ? 'belowBar' : 'aboveBar') : (isLong ? 'aboveBar' : 'belowBar'),
          color: isEntry ? '#00FFBB' : (pnl >= 0 ? '#4ade80' : '#FF4D6A'),
          shape: isEntry ? (isLong ? 'arrowUp' : 'arrowDown') : 'circle',
          text: isEntry
            ? `${(msg.direction as string)?.toUpperCase()} $${(msg.price as number)?.toFixed(2)}`
            : `${msg.reason} $${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}`,
          size: 2,
        })

        // Sort and set markers (must be time-ordered)
        const sorted = [...markers].sort((a, b) => (a.time as number) - (b.time as number))
        try { ref.candle.setMarkers(sorted as any) } catch { /* */ }
      }
    }

    if (type === 'done') {
      setMetrics(msg.metrics as Record<string, unknown>)
      setRunning(false)
    }

    if (type === 'error') {
      setWarning(msg.message as string ?? 'Error')
      setRunning(false)
    }
  }, [])

  const stop = () => {
    wsRef.current?.send(JSON.stringify({ action: 'stop' }))
    wsRef.current?.close()
    setRunning(false)
  }

  return (
    <div className={s.page}>
      {/* Controls */}
      <div className={s.controls}>
        <div className={s.field}>
          <label>Strategy</label>
          <select value={strategy} onChange={e => setStrategy(e.target.value)}>
            {strategies.map(sid => <option key={sid} value={sid}>{sid.replace(/_/g, ' ')}</option>)}
          </select>
        </div>
        <div className={s.field}>
          <label>Start Date</label>
          <input type="date" value={date} onChange={e => setDate(e.target.value)} />
        </div>
        <div className={s.field}>
          <label>End Date</label>
          <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} placeholder="Single day" />
        </div>
        <div className={s.field}>
          <label>Symbols</label>
          <input value={symbols} onChange={e => setSymbols(e.target.value)} placeholder="TSLA,NVDA,AAPL" style={{ width: 280 }} />
        </div>
        <div className={s.field}>
          <label>Speed</label>
          <select value={speed} onChange={e => setSpeed(+e.target.value)}>
            <option value={1}>1x (real-time)</option>
            <option value={4}>4x</option>
            <option value={10}>10x</option>
            <option value={25}>25x</option>
            <option value={50}>50x</option>
            <option value={100}>100x</option>
            <option value={0}>Instant</option>
          </select>
        </div>
        {!running ? (
          <button className={s.playBtn} onClick={start}>▶ Play</button>
        ) : (
          <button className={s.stopBtn} onClick={stop}>■ Stop</button>
        )}
        {running && <span className={s.progress}>{barCount}/{totalBars} bars{currentDay ? ` · ${currentDay}` : ''}{dayCount > 1 ? ` · ${dayCount} days` : ''}</span>}
      </div>

      {/* Warning */}
      {warning && <div className={s.warning}>{warning}</div>}

      {/* Chart Grid */}
      <div className={s.chartGrid} style={{ gridTemplateColumns: `repeat(${Math.min(activeSymbols.length, 3)}, 1fr)` }}>
        {activeSymbols.map(sym => (
          <div key={sym} className={s.chartPanel}>
            <div className={s.chartHeader}>
              <span className={s.chartSymbol}>{sym}</span>
              {trades.filter(t => t.symbol === sym).length > 0 && (
                <span className={s.chartTradeCount}>{trades.filter(t => t.symbol === sym).length} trades</span>
              )}
            </div>
            <SymbolChart symbol={sym} chartRef={chartsRef} />
          </div>
        ))}
      </div>

      {/* Trade Feed + Metrics */}
      <div className={s.bottomPanel}>
        <div className={s.tradeFeed}>
          <div className={s.feedTitle}>Trade Feed ({trades.length})</div>
          {trades.map((t, i) => (
            <div key={i} className={`${s.feedItem} ${t.action === 'entry' ? s.feedEntry : (t.pnl && t.pnl >= 0 ? s.feedWin : s.feedLoss)}`}>
              <span className={s.feedTime}>{t.ts?.split('T')[1]?.slice(0, 8)}</span>
              <span className={s.feedAction}>{t.action === 'entry' ? '→ BUY' : '← SELL'}</span>
              <span className={s.feedSym}>{t.symbol}</span>
              <span className={t.direction === 'long' ? s.up : s.down}>{t.direction?.toUpperCase()}</span>
              <span className={s.mono}>${t.price?.toFixed(2)}</span>
              {t.pnl != null && <span className={`${s.mono} ${t.pnl >= 0 ? s.up : s.down}`}>{t.pnl >= 0 ? '+' : ''}${t.pnl.toFixed(2)}</span>}
              {t.reason && <span className={s.feedReason}>[{t.reason}]</span>}
              {t.equity != null && <span className={s.feedEquity}>Eq: ${t.equity.toLocaleString()}</span>}
            </div>
          ))}
          {trades.length === 0 && <div className={s.feedEmpty}>Waiting for trades...</div>}
        </div>

        {metrics ? (
          <div className={s.metricsPanel}>
            <div className={s.feedTitle}>Results</div>
            <div className={s.metricRow}><span>Trades</span><span className={s.mono}>{String(metrics.total_trades)}</span></div>
            <div className={s.metricRow}><span>Wins</span><span className={`${s.mono} ${s.up}`}>{String(metrics.wins)}</span></div>
            <div className={s.metricRow}><span>Losses</span><span className={`${s.mono} ${s.down}`}>{String(metrics.losses)}</span></div>
            <div className={s.metricRow}><span>Win Rate</span><span className={s.mono}>{(Number(metrics.win_rate) * 100).toFixed(1)}%</span></div>
            <div className={s.metricRow}><span>Net P&L</span><span className={`${s.mono} ${Number(metrics.net_pnl) >= 0 ? s.up : s.down}`}>${Number(metrics.net_pnl).toFixed(2)}</span></div>
            <div className={s.metricRow}><span>PF</span><span className={s.mono}>{Number(metrics.profit_factor).toFixed(2)}</span></div>
            <div className={s.metricRow}><span>Equity</span><span className={s.mono}>${Number(metrics.final_equity).toLocaleString()}</span></div>
          </div>
        ) : (
          <div className={s.metricsPanel}>
            <div className={s.feedTitle}>Metrics</div>
            <div className={s.feedEmpty}>Run replay to see results</div>
          </div>
        )}
      </div>
    </div>
  )
}
