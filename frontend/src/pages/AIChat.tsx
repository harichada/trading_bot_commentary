import { useState, useRef, useEffect, useCallback } from 'react'
import { usePolling } from '../hooks/usePolling'
import { useWebSocket } from '../hooks/useWebSocket'
import { gfApi, type GfHealth, type GfState } from '../api/client'
import { useFormatters } from '../hooks/useFormatters'

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  actions?: string[]
}

export default function AIChat() {
  const { formatCurrency } = useFormatters()
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content: 'Welcome to Rudra Terminal. I can provide information about the current trading state. Try asking about positions, equity, or trading status.',
    },
  ])
  const [input, setInput] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const { data: health } = usePolling<GfHealth>(
    useCallback(() => gfApi.getHealth(), []),
    15000,
  )
  const { data: state } = usePolling<GfState>(
    useCallback(() => gfApi.getState(), []),
    15000,
  )
  const { connected: wsConnected } = useWebSocket('/gf-ws')

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const generateResponse = (query: string): string => {
    const q = query.toLowerCase()

    if (q.includes('position') || q.includes('holding')) {
      const positions = state?.positions
      if (!positions || !Array.isArray(positions) || positions.length === 0) {
        return 'No open positions currently.'
      }
      const lines = positions.map(p =>
        `${p.symbol}: ${p.direction.toUpperCase()} ${p.qty} @ ${formatCurrency(p.entry_price)}` +
        (p.current_price != null ? ` (now ${formatCurrency(p.current_price)})` : '')
      )
      return `Current open positions (${positions.length}):\n\n${lines.join('\n')}`
    }

    if (q.includes('equity') || q.includes('balance') || q.includes('account')) {
      const equity = health?.equity ?? state?.equity
      const pnl = health?.daily_pnl ?? state?.daily_pnl
      if (equity == null) return 'Equity data not available. Is the engine running?'
      let resp = `Current equity: ${formatCurrency(equity as number)}`
      if (pnl != null) resp += `\nDaily P&L: ${formatCurrency(pnl as number)}`
      if (state?.starting_equity != null) resp += `\nStarting equity: ${formatCurrency(state.starting_equity as number)}`
      return resp
    }

    if (q.includes('status') || q.includes('health')) {
      if (!health) return 'Engine health data not available.'
      return [
        `Trader status: ${health.trader_status ?? 'unknown'}`,
        `Trading halted: ${health.trading_halted ? 'YES' : 'NO'}`,
        `Positions: ${health.positions ?? 0}`,
        `Uptime: ${health.uptime_seconds != null ? `${Math.floor(health.uptime_seconds as number / 60)}m` : 'unknown'}`,
        `WebSocket: ${wsConnected ? 'Connected' : 'Disconnected'}`,
      ].join('\n')
    }

    if (q.includes('trade') || q.includes('history')) {
      const trades = state?.trades
      if (!trades || !Array.isArray(trades) || trades.length === 0) {
        return 'No trade history available for today.'
      }
      const wins = state?.win_count ?? 0
      const losses = state?.loss_count ?? 0
      const total = wins + losses
      const wr = total > 0 ? ((wins / total) * 100).toFixed(1) : '0'
      return `Trade history: ${total} trades (${wins}W / ${losses}L, ${wr}% win rate)\n\nRecent trades:\n` +
        trades.slice(0, 5).map(t =>
          `${t.symbol} ${t.direction} | Entry: ${formatCurrency(t.entry_price)} Exit: ${formatCurrency(t.exit_price)} | P&L: ${formatCurrency(t.pnl)}`
        ).join('\n')
    }

    return 'I can answer questions about: positions, equity/balance, status/health, trade history. The AI chat is a local assistant that queries the live trading API -- it does not use a language model.'
  }

  const handleSend = () => {
    if (!input.trim()) return
    const userMsg: ChatMessage = { role: 'user', content: input }
    const response = generateResponse(input)
    const assistantMsg: ChatMessage = { role: 'assistant', content: response }
    setMessages(prev => [...prev, userMsg, assistantMsg])
    setInput('')
  }

  return (
    <div className="h-full flex gap-4">
      {/* Left Sidebar */}
      <div className="w-48 shrink-0 space-y-4">
        <div className="bg-bg-card rounded-lg border border-border p-3">
          <h3 className="text-[10px] text-text-muted uppercase tracking-wider mb-2">Rudra Terminal</h3>
          <p className="text-[10px] text-text-muted">
            Engine: {health ? 'Online' : 'Offline'}
          </p>
          <p className="text-[10px] text-text-muted">
            WS: {wsConnected ? 'Connected' : 'Disconnected'}
          </p>
        </div>

        <div className="bg-bg-card rounded-lg border border-border p-3">
          <h3 className="text-[10px] text-text-muted uppercase tracking-wider mb-2">Quick Queries</h3>
          <div className="space-y-1">
            {['Show positions', 'Account equity', 'Engine status', 'Trade history'].map((item) => (
              <button
                key={item}
                onClick={() => { setInput(item); }}
                className="w-full text-left text-xs text-cyan hover:text-cyan/80 py-1 transition-colors"
              >
                {item}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col bg-bg-card rounded-lg border border-border">
        {/* Chat Header */}
        <div className="px-4 py-3 border-b border-border flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-green/10 flex items-center justify-center">
            <span className="text-green text-sm font-bold">R</span>
          </div>
          <div>
            <h2 className="text-sm font-semibold text-text-primary">Rudra Assistant</h2>
            <p className="text-[10px] text-text-muted">
              {health ? 'Connected to live engine' : 'Engine offline - limited data'}
            </p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green' : 'bg-red'}`} />
            <span className="text-[10px] text-text-muted">{wsConnected ? 'LIVE' : 'OFFLINE'}</span>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((msg, i) => (
            <div key={i} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}>
              {msg.role === 'assistant' && (
                <div className="w-7 h-7 rounded-full bg-green/10 flex items-center justify-center shrink-0">
                  <span className="text-green text-xs font-bold">R</span>
                </div>
              )}
              <div className={`max-w-2xl ${msg.role === 'user' ? 'bg-bg-card-hover' : 'bg-bg-primary'} rounded-lg p-3 border border-border`}>
                <p className="text-sm text-text-primary whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                {msg.actions && (
                  <div className="flex gap-2 mt-3">
                    {msg.actions.map((action) => (
                      <button key={action} className="px-3 py-1.5 text-[10px] font-bold rounded bg-green text-black hover:bg-green/90 transition-colors">
                        {action}
                      </button>
                    ))}
                  </div>
                )}
              </div>
              {msg.role === 'user' && (
                <div className="w-7 h-7 rounded-full bg-purple/10 flex items-center justify-center shrink-0">
                  <span className="text-purple text-xs font-bold">U</span>
                </div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="px-4 py-3 border-t border-border">
          <div className="flex items-center gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Ask about positions, equity, status, trades..."
              className="flex-1 bg-bg-primary border border-border rounded-lg px-4 py-2.5 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-green"
            />
            <button
              onClick={handleSend}
              className="w-10 h-10 rounded-full bg-green flex items-center justify-center hover:bg-green/90 transition-colors"
            >
              <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 text-black">
                <path d="M10 18l-6-6h4V2h4v10h4l-6 6z" transform="rotate(-90 10 10)" />
              </svg>
            </button>
          </div>
          <p className="text-[10px] text-text-muted mt-1">Press ENTER to send. Queries live trading data.</p>
        </div>
      </div>

      {/* Right Sidebar - Live Stats */}
      <div className="w-56 shrink-0">
        <div className="bg-bg-card rounded-lg border border-border p-3">
          <h3 className="text-[10px] text-text-muted uppercase tracking-wider mb-3">Live Engine Stats</h3>
          <div className="space-y-3">
            <div>
              <p className="text-[10px] text-text-muted">Equity</p>
              <p className="font-mono text-sm font-semibold text-text-primary">
                {health?.equity != null ? formatCurrency(health.equity as number) : '\u2014'}
              </p>
            </div>
            <div>
              <p className="text-[10px] text-text-muted">Daily P&L</p>
              <p className={`font-mono text-sm font-semibold ${
                (health?.daily_pnl as number ?? 0) >= 0 ? 'text-green' : 'text-red'
              }`}>
                {health?.daily_pnl != null ? formatCurrency(health.daily_pnl as number) : '\u2014'}
              </p>
            </div>
            <div>
              <p className="text-[10px] text-text-muted">Positions</p>
              <p className="font-mono text-sm text-text-primary">{health?.positions ?? 0}</p>
            </div>
            <div>
              <p className="text-[10px] text-text-muted">Status</p>
              <p className={`font-mono text-sm ${health?.trading_halted ? 'text-red' : 'text-green'}`}>
                {health?.trader_status ?? 'Unknown'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
