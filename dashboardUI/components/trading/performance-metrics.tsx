"use client"

import { useState, useEffect } from "react"
import { 
  AreaChart, 
  Area, 
  BarChart,
  Bar,
  XAxis, 
  YAxis, 
  ResponsiveContainer,
  Cell,
  Tooltip
} from "recharts"
import { 
  Target, 
  TrendingUp, 
  Clock, 
  Zap,
  Award,
  BarChart3,
  ArrowUpRight,
  ArrowDownRight
} from "lucide-react"

// Deterministic so SSR and client render identically (no hydration mismatch):
// fixed date anchor + seeded PRNG instead of new Date()/Math.random() at module load.
function mulberry32(seed: number) {
  return function () {
    seed |= 0
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const PNL_ANCHOR = new Date(Date.UTC(2026, 5, 22)) // fixed constructor → deterministic
const _pnlRand = mulberry32(0x5eed1234)
const pnlHistory = Array.from({ length: 30 }, (_, i) => {
  const date = new Date(PNL_ANCHOR)
  date.setUTCDate(date.getUTCDate() - (29 - i))
  const pnl = (_pnlRand() - 0.4) * 2000
  return {
    date: date.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" }),
    pnl,
    cumulative: 0,
  }
})

// Calculate cumulative P&L
let cumulative = 0
pnlHistory.forEach(day => {
  cumulative += day.pnl
  day.cumulative = cumulative
})

// Daily P&L breakdown for waterfall
const dailyBreakdown = [
  { name: "PLTR", pnl: 1779.20 },
  { name: "SNAP", pnl: -33.04 },
  { name: "COIN", pnl: -63.00 },
  { name: "PINS", pnl: -187.07 },
  { name: "HOGE", pnl: 17.55 },
  { name: "Fees", pnl: -12.50 },
]

interface MetricCardMiniProps {
  icon: React.ReactNode
  label: string
  value: string
  subValue?: string
  trend?: "up" | "down"
  colorClass?: string
}

function MetricCardMini({ icon, label, value, subValue, trend, colorClass = "text-muted-foreground" }: MetricCardMiniProps) {
  return (
    <div className="lift flex items-start gap-3 p-4 rounded-xl glass-subtle">
      <div className={`p-2 rounded-lg bg-secondary/40 ${colorClass}`}>
        {icon}
      </div>
      <div>
        <p className="eyebrow">{label}</p>
        <div className="flex items-baseline gap-1.5">
          <span className="text-lg font-semibold font-mono tabular-nums text-foreground">{value}</span>
          {trend && (
            trend === "up"
              ? <ArrowUpRight className="h-3.5 w-3.5 text-success" />
              : <ArrowDownRight className="h-3.5 w-3.5 text-destructive" />
          )}
        </div>
        {subValue && <p className="text-[11px] text-muted-foreground">{subValue}</p>}
      </div>
    </div>
  )
}

export function PerformanceMetrics() {
  const [selectedView, setSelectedView] = useState<"cumulative" | "daily">("cumulative")

  // Calculate stats
  const winningDays = pnlHistory.filter(d => d.pnl > 0).length
  const totalDays = pnlHistory.length
  const winRate = ((winningDays / totalDays) * 100).toFixed(1)
  const avgWin = pnlHistory.filter(d => d.pnl > 0).reduce((sum, d) => sum + d.pnl, 0) / winningDays
  const avgLoss = Math.abs(pnlHistory.filter(d => d.pnl < 0).reduce((sum, d) => sum + d.pnl, 0) / (totalDays - winningDays))
  const profitFactor = (avgWin * winningDays) / (avgLoss * (totalDays - winningDays))
  const maxDrawdown = Math.min(...pnlHistory.map(d => d.cumulative))

  return (
    <div className="rounded-xl glass overflow-hidden reveal">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-border/20">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold tracking-tight text-foreground">Performance Analytics</h3>
        </div>
        <div className="flex items-center gap-1 p-1 rounded-lg bg-secondary/30">
          <button
            onClick={() => setSelectedView("cumulative")}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
              selectedView === "cumulative"
                ? 'bg-accent text-accent-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            Cumulative
          </button>
          <button
            onClick={() => setSelectedView("daily")}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
              selectedView === "daily"
                ? 'bg-accent text-accent-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            Daily
          </button>
        </div>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 p-4 border-b border-border/20">
        <MetricCardMini
          icon={<Target className="h-4 w-4" />}
          label="Win Rate"
          value={`${winRate}%`}
          subValue={`${winningDays}/${totalDays} days`}
          trend="up"
        />
        <MetricCardMini
          icon={<Award className="h-4 w-4" />}
          label="Profit Factor"
          value={profitFactor.toFixed(2)}
          subValue="Risk/Reward"
          trend={profitFactor > 1 ? "up" : "down"}
        />
        <MetricCardMini
          icon={<TrendingUp className="h-4 w-4" />}
          label="Avg Win"
          value={`$${avgWin.toFixed(0)}`}
          subValue="Per winning day"
        />
        <MetricCardMini
          icon={<BarChart3 className="h-4 w-4" />}
          label="Max Drawdown"
          value={`$${Math.abs(maxDrawdown).toFixed(0)}`}
          subValue="Peak to trough"
        />
      </div>
      
      {/* Chart */}
      <div className="h-[200px] p-4">
        {selectedView === "cumulative" ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={pnlHistory} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="cumulativeGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="oklch(0.74 0.105 205)" stopOpacity={0.12} />
                  <stop offset="100%" stopColor="oklch(0.74 0.105 205)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis 
                dataKey="date" 
                axisLine={false}
                tickLine={false}
                tick={{ fill: 'oklch(0.5 0 0)', fontSize: 10 }}
                interval="preserveStartEnd"
              />
              <YAxis 
                axisLine={false}
                tickLine={false}
                tick={{ fill: 'oklch(0.5 0 0)', fontSize: 10 }}
                tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                width={45}
              />
              <Tooltip 
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload
                    return (
                      <div className="glass rounded-lg px-3 py-2 border border-border/30">
                        <p className="text-xs text-muted-foreground">{data.date}</p>
                        <p className={`text-sm font-semibold ${data.cumulative >= 0 ? 'text-success' : 'text-destructive'}`}>
                          ${data.cumulative.toFixed(2)}
                        </p>
                      </div>
                    )
                  }
                  return null
                }}
              />
              <Area
                type="monotone"
                dataKey="cumulative"
                stroke="oklch(0.74 0.105 205)"
                strokeWidth={1.5}
                fill="url(#cumulativeGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={pnlHistory.slice(-14)} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <XAxis 
                dataKey="date" 
                axisLine={false}
                tickLine={false}
                tick={{ fill: 'oklch(0.5 0 0)', fontSize: 10 }}
              />
              <YAxis 
                axisLine={false}
                tickLine={false}
                tick={{ fill: 'oklch(0.5 0 0)', fontSize: 10 }}
                tickFormatter={(value) => `$${value.toFixed(0)}`}
                width={50}
              />
              <Tooltip 
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload
                    return (
                      <div className="glass rounded-lg px-3 py-2 border border-border/30">
                        <p className="text-xs text-muted-foreground">{data.date}</p>
                        <p className={`text-sm font-semibold ${data.pnl >= 0 ? 'text-success' : 'text-destructive'}`}>
                          {data.pnl >= 0 ? '+' : ''}${data.pnl.toFixed(2)}
                        </p>
                      </div>
                    )
                  }
                  return null
                }}
              />
              <Bar dataKey="pnl" radius={[3, 3, 0, 0]}>
                {pnlHistory.slice(-14).map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.pnl >= 0 ? "oklch(0.745 0.135 158)" : "oklch(0.64 0.185 22)"}
                    fillOpacity={0.85}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
      
      {/* Today's Breakdown */}
      <div className="px-5 py-4 border-t border-border/20">
        <p className="eyebrow mb-3">Today&apos;s P&amp;L Breakdown</p>
        <div className="flex items-center gap-2 overflow-x-auto pb-2">
          {dailyBreakdown.map((item) => (
            <div
              key={item.name}
              className={`lift flex-shrink-0 px-3 py-2 rounded-lg border ${
                item.pnl >= 0
                  ? 'bg-success/[0.10] border-success/20'
                  : 'bg-destructive/[0.10] border-destructive/20'
              }`}
            >
              <p className="text-[11px] font-medium text-muted-foreground">{item.name}</p>
              <p className={`text-sm font-semibold font-mono tabular-nums ${
                item.pnl >= 0 ? 'text-success' : 'text-destructive'
              }`}>
                {item.pnl >= 0 ? '+' : ''}${item.pnl.toFixed(2)}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
