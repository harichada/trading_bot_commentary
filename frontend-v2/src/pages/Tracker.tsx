import { useState, useEffect, useRef, useCallback } from 'react'
import { createChart, type IChartApi, type ISeriesApi, ColorType, LineStyle, CandlestickSeries } from 'lightweight-charts'
import { useToast } from '../components/Toast'
import s from './Tracker.module.css'

function ChartPanel({ symbol, onRemove }: { symbol: string; onRemove: () => void }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const [tf, setTf] = useState('5Min')
  const [price, setPrice] = useState<number | null>(null)
  const { showToast } = useToast()

  const loadBars = useCallback(async (timeframe: string) => {
    try {
      const data = await fetch(`/api/tracker/bars?symbol=${symbol}&timeframe=${timeframe}&limit=200`).then(r => r.json())
      const bars = (data.bars ?? data ?? []).map((b: Record<string, unknown>) => ({
        time: Number(b.time ?? b.t ?? 0),
        open: Number(b.open ?? b.o ?? 0), high: Number(b.high ?? b.h ?? 0),
        low: Number(b.low ?? b.l ?? 0), close: Number(b.close ?? b.c ?? 0),
      })).filter((b: { time: number }) => b.time > 0)
      if (seriesRef.current && bars.length) {
        seriesRef.current.setData(bars)
        setPrice(bars[bars.length - 1].close)
      }
    } catch { /* silent */ }
  }, [symbol])

  useEffect(() => {
    if (!containerRef.current) return
    let chart: IChartApi
    try {
    chart = createChart(containerRef.current, {
      layout: { background: { type: ColorType.Solid, color: '#0d0d0d' }, textColor: '#888', fontFamily: 'JetBrains Mono', fontSize: 10 },
      grid: { vertLines: { color: 'rgba(255,255,255,0.03)' }, horzLines: { color: 'rgba(255,255,255,0.03)' } },
      crosshair: { vertLine: { color: '#00FFBB', width: 1, style: LineStyle.Dashed, labelBackgroundColor: '#00FFBB' }, horzLine: { color: '#00FFBB', width: 1, style: LineStyle.Dashed, labelBackgroundColor: '#00FFBB' } },
      rightPriceScale: { borderColor: 'rgba(255,255,255,0.06)' },
      timeScale: { borderColor: 'rgba(255,255,255,0.06)', timeVisible: true, secondsVisible: false },
      height: 200, width: containerRef.current.offsetWidth,
    })
    let series: ISeriesApi<'Candlestick'>
    try {
      // v5 API
      series = chart.addSeries(CandlestickSeries, { upColor: '#00FFBB', downColor: '#FF4D6A', borderUpColor: '#00FFBB', borderDownColor: '#FF4D6A', wickUpColor: '#00FFBB', wickDownColor: '#FF4D6A' })
    } catch {
      // v4 fallback
      series = (chart as unknown as Record<string, unknown>).addCandlestickSeries({ upColor: '#00FFBB', downColor: '#FF4D6A', borderUpColor: '#00FFBB', borderDownColor: '#FF4D6A', wickUpColor: '#00FFBB', wickDownColor: '#FF4D6A' }) as ISeriesApi<'Candlestick'>
    }
    chartRef.current = chart; seriesRef.current = series
    const ro = new ResizeObserver(() => {
      if (containerRef.current) chart.applyOptions({ width: containerRef.current.offsetWidth })
    })
    ro.observe(containerRef.current)
    loadBars(tf)
    return () => { ro.disconnect(); chart.remove() }
    } catch (err) { console.warn('Chart init failed:', err) }
  }, [symbol]) // eslint-disable-line

  const changeTf = (newTf: string) => { setTf(newTf); loadBars(newTf) }

  const order = async (side: 'buy' | 'sell') => {
    const qty = prompt(`${side.toUpperCase()} ${symbol} — quantity:`)
    if (!qty || isNaN(Number(qty))) return
    try {
      const r = await fetch('/api/tracker/order', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ symbol, side, qty: Number(qty) }) }).then(r => r.json())
      showToast(r.error ? `Error: ${r.error}` : `${side.toUpperCase()} ${qty} ${symbol} filled`, r.error ? 'error' : 'success')
    } catch { showToast('Order failed', 'error') }
  }

  return (
    <div className={s.chartPanel}>
      <div className={s.chartHeader}>
        <span className={s.chartSymbol}>{symbol}</span>
        {price && <span className={s.chartPrice}>${price.toFixed(2)}</span>}
        <div className={s.chartActions}>
          <button className={s.buyBtn} onClick={() => order('buy')}>Buy</button>
          <button className={s.shortBtn} onClick={() => order('sell')}>Short</button>
          <button className={s.removeBtn} onClick={onRemove}>×</button>
        </div>
      </div>
      <div className={s.tfBar}>
        {['1Min', '5Min', '15Min', '1Hour', '1Day'].map(t => (
          <button key={t} className={`${s.tfBtn} ${tf === t ? s.tfActive : ''}`} onClick={() => changeTf(t)}>{t}</button>
        ))}
      </div>
      <div ref={containerRef} className={s.chartContainer} />
    </div>
  )
}

export default function Tracker() {
  const [watchlist, setWatchlist] = useState<string[]>(() => {
    try { return JSON.parse(localStorage.getItem('tracker_watchlist') ?? '[]') } catch { return [] }
  })
  const [positions, setPositions] = useState<Record<string, unknown>[]>([])
  const [input, setInput] = useState('')

  useEffect(() => {
    localStorage.setItem('tracker_watchlist', JSON.stringify(watchlist))
  }, [watchlist])

  useEffect(() => {
    const load = () => {
      fetch('/api/tracker/positions').then(r => r.json()).then(d => {
        const p = d?.positions ?? d
        setPositions(Array.isArray(p) ? p : [])
      }).catch(() => {})
    }
    load()
    const timer = setInterval(load, 10000)
    return () => clearInterval(timer)
  }, [])

  const addSymbol = () => {
    const sym = input.trim().toUpperCase()
    if (!sym || watchlist.includes(sym) || watchlist.length >= 6) return
    setWatchlist(prev => [...prev, sym])
    setInput('')
    fetch('/api/tracker/watchlist', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ symbol: sym }) })
  }

  const removeSymbol = (sym: string) => setWatchlist(prev => prev.filter(s => s !== sym))

  return (
    <div className={s.page}>
      {/* Controls */}
      <div className={s.controls}>
        <input className={s.input} placeholder="Add symbol..." value={input}
          onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && addSymbol()} />
        <button className={s.addBtn} onClick={addSymbol}>+ Add</button>
        <span className={s.watchCount}>{watchlist.length} / 6 symbols</span>
      </div>

      {/* Broker Positions */}
      {positions.length > 0 && (
        <div className={s.posSection}>
          <div className={s.sectionTitle}>Broker Positions</div>
          <table className={s.table}>
            <thead><tr><th>Symbol</th><th>Qty</th><th>Mkt Value</th><th>Avg Price</th><th>Last</th><th>P&L</th></tr></thead>
            <tbody>{positions.map((p, i) => (
              <tr key={i} onClick={() => { const sym = String(p.symbol ?? ''); if (sym && !watchlist.includes(sym)) setWatchlist(prev => [...prev.slice(0, 5), sym]) }}>
                <td className={s.ticker}>{String(p.symbol ?? '')}</td>
                <td className={s.mono}>{String(p.qty ?? p.shares ?? '')}</td>
                <td className={s.mono}>${Number(p.market_value ?? 0).toFixed(0)}</td>
                <td className={s.mono}>${Number(p.avg_entry_price ?? p.avg_price ?? 0).toFixed(2)}</td>
                <td className={s.mono}>${Number(p.current_price ?? p.last ?? 0).toFixed(2)}</td>
                <td className={`${s.mono} ${Number(p.unrealized_pl ?? p.pnl ?? 0) >= 0 ? s.up : s.down}`}>
                  ${Number(p.unrealized_pl ?? p.pnl ?? 0).toFixed(2)}
                </td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      )}

      {/* Chart Grid */}
      <div className={s.chartGrid}>
        {watchlist.map(sym => (
          <ChartPanel key={sym} symbol={sym} onRemove={() => removeSymbol(sym)} />
        ))}
        {watchlist.length === 0 && <div className={s.empty}>Add symbols to see live charts</div>}
      </div>
    </div>
  )
}
