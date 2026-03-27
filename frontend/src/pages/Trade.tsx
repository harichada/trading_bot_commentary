import { useState, useCallback } from 'react'
import Chart from '../components/Chart'
import OrderForm from '../components/OrderForm'
import StatusBanner from '../components/StatusBanner'
import { usePolling } from '../hooks/usePolling'
import { useFormatters } from '../hooks/useFormatters'
import { idApi, type IdHealth } from '../api/client'

export default function Trade() {
  const { formatCurrency } = useFormatters()
  const [symbol, setSymbol] = useState('SPY')
  const [orderStatus, setOrderStatus] = useState<string | null>(null)

  const { data: health, error } = usePolling<IdHealth>(
    useCallback(() => idApi.getHealth(), []),
    10000,
  )

  const handleSubmit = async (order: { symbol: string; side: 'buy' | 'sell'; type: string; qty: number; price?: number }) => {
    setOrderStatus('Submitting...')
    try {
      await idApi.submitOrder({
        symbol: order.symbol,
        side: order.side,
        qty: order.qty,
        type: order.type,
        price: order.price,
      })
      setOrderStatus(`Order submitted: ${order.side.toUpperCase()} ${order.qty} ${order.symbol}`)
      setTimeout(() => setOrderStatus(null), 5000)
    } catch (err) {
      setOrderStatus(`Order failed: ${String(err)}`)
      setTimeout(() => setOrderStatus(null), 5000)
    }
  }

  const tradingActive = health?.trading_active ?? false

  return (
    <div className="space-y-4">
      <StatusBanner label="Intraday Engine" error={error} wsConnected={true} />

      {/* Symbol Header */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-bg-card flex items-center justify-center text-sm font-bold text-cyan border border-border">
            {symbol.charAt(0)}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                className="text-lg font-bold text-text-primary bg-transparent border-b border-border focus:outline-none focus:border-cyan w-24"
              />
              <span className="text-[10px] px-2 py-0.5 rounded bg-green/10 text-green font-semibold">EQUITY</span>
            </div>
            <p className="text-xs text-text-muted">
              Intraday Engine: {tradingActive ? 'Active' : 'Inactive'} |
              Strategies: {health?.strategies_enabled ?? '\u2014'} |
              Positions: {health?.positions ?? '\u2014'}
            </p>
          </div>
        </div>

        {orderStatus && (
          <div className={`ml-auto px-3 py-1.5 rounded-lg text-xs font-semibold ${
            orderStatus.includes('failed') ? 'bg-red/10 text-red' : 'bg-green/10 text-green'
          }`}>
            {orderStatus}
          </div>
        )}
      </div>

      {/* Chart + Order Form */}
      <div className="grid grid-cols-3 gap-4">
        {/* Chart */}
        <div className="col-span-2 bg-bg-card rounded-lg border border-border p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-text-primary uppercase tracking-wider">
              {symbol} - Intraday
            </h3>
            <span className="text-[10px] text-text-muted">
              Chart shows generated data. Real price bars require a market data feed.
            </span>
          </div>
          <Chart type="candlestick" height={400} />
        </div>

        {/* Order Form */}
        <div className="col-span-1">
          <OrderForm
            symbol={symbol}
            currentPrice={0}
            onSubmit={handleSubmit}
          />
        </div>
      </div>
    </div>
  )
}
