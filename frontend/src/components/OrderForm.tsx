import { useState } from 'react'

interface OrderFormProps {
  symbol?: string
  currentPrice?: number
  onSubmit?: (order: { symbol: string; side: 'buy' | 'sell'; type: string; qty: number; price?: number; takeProfit?: number; stopLoss?: number }) => void
}

export default function OrderForm({ symbol = 'SPY', currentPrice = 0, onSubmit }: OrderFormProps) {
  const [orderType, setOrderType] = useState<'MARKET' | 'LIMIT' | 'STOP'>('MARKET')
  const [side, setSide] = useState<'buy' | 'sell'>('buy')
  const [price, setPrice] = useState(currentPrice || 0)
  const [qty, setQty] = useState(100)
  const [takeProfit, setTakeProfit] = useState<string>('')
  const [stopLoss, setStopLoss] = useState<string>('')

  const estimatedValue = qty * price
  const _unusedSide = side

  const handleSubmit = (submitSide: 'buy' | 'sell') => {
    setSide(submitSide)
    onSubmit?.({
      symbol,
      side: submitSide,
      type: orderType.toLowerCase(),
      qty,
      price: orderType !== 'MARKET' ? price : undefined,
      takeProfit: takeProfit ? parseFloat(takeProfit) : undefined,
      stopLoss: stopLoss ? parseFloat(stopLoss) : undefined,
    })
  }

  return (
    <div className="bg-bg-card rounded-lg border border-border p-5">
      {/* Order Type Tabs */}
      <div className="flex gap-1 mb-5 bg-bg-primary rounded-lg p-1">
        {(['MARKET', 'LIMIT', 'STOP'] as const).map((type) => (
          <button
            key={type}
            onClick={() => setOrderType(type)}
            className={`flex-1 py-2 text-xs font-semibold rounded-md transition-colors ${
              orderType === type
                ? 'bg-bg-card-hover text-text-primary'
                : 'text-text-muted hover:text-text-secondary'
            }`}
          >
            {type}
          </button>
        ))}
      </div>

      {/* Price Input */}
      <div className="mb-4">
        <label className="text-xs text-text-secondary uppercase tracking-wider mb-1.5 block">
          {orderType === 'MARKET' ? 'Market Price (USD)' : `${orderType} Price (USD)`}
        </label>
        <input
          type="number"
          value={price}
          onChange={(e) => setPrice(parseFloat(e.target.value) || 0)}
          disabled={orderType === 'MARKET'}
          className="w-full bg-bg-primary border border-border rounded-lg px-4 py-2.5 font-mono text-sm text-text-primary focus:outline-none focus:border-cyan disabled:opacity-50"
        />
      </div>

      {/* Quantity */}
      <div className="mb-4">
        <label className="text-xs text-text-secondary uppercase tracking-wider mb-1.5 block">
          Quantity (Shares)
        </label>
        <input
          type="number"
          value={qty}
          onChange={(e) => setQty(parseInt(e.target.value) || 0)}
          className="w-full bg-bg-primary border border-border rounded-lg px-4 py-2.5 font-mono text-sm text-text-primary focus:outline-none focus:border-cyan"
        />
        <div className="flex gap-2 mt-2">
          {[25, 50, 75, 100].map((pct) => (
            <button
              key={pct}
              onClick={() => setQty(Math.floor((pct / 100) * 1000))}
              className="flex-1 py-1 text-[10px] font-semibold rounded bg-bg-primary border border-border text-text-muted hover:text-text-secondary hover:border-border-light transition-colors"
            >
              {pct === 100 ? 'Max' : `${pct}%`}
            </button>
          ))}
        </div>
      </div>

      {/* Risk Management */}
      <div className="mb-4 border-t border-border pt-4">
        <p className="text-xs text-text-secondary uppercase tracking-wider mb-3">Risk Management</p>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-[10px] text-text-muted uppercase mb-1 block">Take Profit (USD)</label>
            <input
              type="number"
              value={takeProfit}
              onChange={(e) => setTakeProfit(e.target.value)}
              placeholder="--"
              className="w-full bg-bg-primary border border-border rounded px-3 py-2 font-mono text-xs text-text-primary focus:outline-none focus:border-green placeholder:text-text-muted"
            />
          </div>
          <div>
            <label className="text-[10px] text-text-muted uppercase mb-1 block">Stop Loss (USD)</label>
            <input
              type="number"
              value={stopLoss}
              onChange={(e) => setStopLoss(e.target.value)}
              placeholder="--"
              className="w-full bg-bg-primary border border-border rounded px-3 py-2 font-mono text-xs text-text-primary focus:outline-none focus:border-red placeholder:text-text-muted"
            />
          </div>
        </div>
      </div>

      {/* Estimated Value */}
      <div className="mb-5 space-y-1.5">
        <div className="flex justify-between text-xs">
          <span className="text-text-muted">Estimated Order Value</span>
          <span className="font-mono text-text-primary">
            ${estimatedValue.toLocaleString('en-US', { minimumFractionDigits: 2 })} USD
          </span>
        </div>
        <div className="flex justify-between text-xs">
          <span className="text-text-muted">Estimated Commission</span>
          <span className="font-mono text-text-secondary">0.00 USD (Commission-Free)</span>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="grid grid-cols-2 gap-3">
        <button
          onClick={() => handleSubmit('buy')}
          className="py-3 rounded-lg bg-green text-black font-semibold text-sm hover:bg-green/90 transition-colors"
        >
          BUY SHARES
        </button>
        <button
          onClick={() => handleSubmit('sell')}
          className="py-3 rounded-lg bg-red text-white font-semibold text-sm hover:bg-red/90 transition-colors"
        >
          SELL SHARES
        </button>
      </div>
    </div>
  )
}
