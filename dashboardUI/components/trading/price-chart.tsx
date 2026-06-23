"use client"

import { useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, ReferenceLine, Tooltip, CartesianGrid } from "recharts"
import { TrendingUp, TrendingDown, Maximize2, ChevronDown, Layers, Clock } from "lucide-react"

// Deterministic PRNG so SSR and client render identical data (no hydration mismatch)
function mulberry32(seed: number) {
  return function () {
    seed |= 0
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function hashSymbol(s: string) {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

function formatBarTime(i: number) {
  const total = 570 + i * 5 // minutes from midnight, market opens 9:30
  let h = Math.floor(total / 60)
  const m = total % 60
  const ampm = h >= 12 ? "PM" : "AM"
  h = h % 12 || 12
  return `${h}:${m.toString().padStart(2, "0")} ${ampm}`
}

function generatePriceData(symbol: string, basePrice: number, volatility = 0.02) {
  const rand = mulberry32(hashSymbol(symbol))
  const data: { time: string; price: number; volume: number }[] = []
  let price = basePrice * (1 - volatility * 2)
  for (let i = 0; i < 78; i++) {
    const change = (rand() - 0.48) * volatility * basePrice
    price = Math.max(price + change, basePrice * 0.9)
    data.push({
      time: formatBarTime(i),
      price,
      volume: Math.floor(rand() * 500000) + 100000,
    })
  }
  return data
}

const symbols = [
  { symbol: "PLTR", name: "Palantir Technologies", price: 142.1, change: 4.72, basePrice: 135 },
  { symbol: "TSLA", name: "Tesla Inc", price: 391.77, change: 7.57, basePrice: 365 },
  { symbol: "NVDA", name: "NVIDIA Corp", price: 285.44, change: -2.31, basePrice: 292 },
  { symbol: "SNAP", name: "Snap Inc", price: 6.03, change: -0.33, basePrice: 6.05 },
  { symbol: "COIN", name: "Coinbase Global", price: 195.61, change: -0.21, basePrice: 196 },
]

const timeframes = ["1D", "5D", "1M", "3M", "1Y", "ALL"]

export function PriceChart() {
  const [selectedSymbol, setSelectedSymbol] = useState(symbols[0])
  const [selectedTimeframe, setSelectedTimeframe] = useState("1D")
  const priceData = useMemo(
    () => generatePriceData(selectedSymbol.symbol, selectedSymbol.basePrice),
    [selectedSymbol],
  )

  const isPositive = selectedSymbol.change >= 0
  // Literal oklch — CSS var() does NOT resolve inside SVG presentation attributes (recharts)
  const lineColor = isPositive ? "oklch(0.745 0.135 158)" : "oklch(0.64 0.185 22)"
  const openPrice = priceData[0]?.price || selectedSymbol.basePrice
  const highPrice = Math.max(...priceData.map((d) => d.price))
  const lowPrice = Math.min(...priceData.map((d) => d.price))

  return (
    <div className="overflow-hidden rounded-lg glass">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-5 py-3.5">
        <div className="flex items-center gap-4">
          {/* Symbol selector */}
          <div className="group relative">
            <button className="flex items-center gap-2.5 rounded-md px-2 py-1.5 transition-colors hover:bg-secondary/60">
              <div className="flex h-9 w-9 items-center justify-center rounded-md bg-secondary font-mono text-xs font-bold text-foreground">
                {selectedSymbol.symbol.slice(0, 2)}
              </div>
              <div className="text-left">
                <div className="flex items-center gap-1.5">
                  <span className="font-semibold tracking-tight text-foreground">{selectedSymbol.symbol}</span>
                  <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
                </div>
                <span className="text-xs text-muted-foreground">{selectedSymbol.name}</span>
              </div>
            </button>

            <div className="invisible absolute left-0 top-full z-50 mt-1.5 w-64 rounded-lg glass-blur p-1.5 opacity-0 shadow-xl transition-all group-hover:visible group-hover:opacity-100">
              {symbols.map((s) => (
                <button
                  key={s.symbol}
                  onClick={() => setSelectedSymbol(s)}
                  className={`flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 transition-colors lift ${
                    s.symbol === selectedSymbol.symbol ? "bg-secondary" : ""
                  }`}
                >
                  <div className="flex h-7 w-7 items-center justify-center rounded-md bg-secondary font-mono text-[11px] font-bold text-foreground">
                    {s.symbol.slice(0, 2)}
                  </div>
                  <div className="flex-1 text-left">
                    <span className="text-sm font-medium text-foreground">{s.symbol}</span>
                    <span className="block text-xs text-muted-foreground">{s.name}</span>
                  </div>
                  <span
                    className={`font-mono text-xs tabular-nums ${s.change >= 0 ? "text-success" : "text-destructive"}`}
                  >
                    {s.change >= 0 ? "+" : "−"}
                    {Math.abs(s.change).toFixed(2)}%
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Price */}
          <div className="border-l border-border pl-4">
            <div className="flex items-baseline gap-2.5">
              <span className="font-mono text-2xl font-semibold tabular-nums tracking-tight text-foreground">
                ${selectedSymbol.price.toFixed(2)}
              </span>
              <div className={`flex items-center gap-1 text-sm font-medium ${isPositive ? "text-success" : "text-destructive"}`}>
                {isPositive ? <TrendingUp className="h-3.5 w-3.5" /> : <TrendingDown className="h-3.5 w-3.5" />}
                <span className="font-mono tabular-nums">
                  {isPositive ? "+" : "−"}
                  {Math.abs(selectedSymbol.change).toFixed(2)}%
                </span>
              </div>
            </div>
            <div className="mt-1 flex items-center gap-3 font-mono text-[11px] tabular-nums text-muted-foreground">
              <span><span className="eyebrow !text-muted-foreground/70">O</span> {openPrice.toFixed(2)}</span>
              <span><span className="eyebrow !text-muted-foreground/70">H</span> {highPrice.toFixed(2)}</span>
              <span><span className="eyebrow !text-muted-foreground/70">L</span> {lowPrice.toFixed(2)}</span>
              <span><span className="eyebrow !text-muted-foreground/70">Vol</span> 12.4M</span>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-0.5 rounded-md border border-border bg-secondary/40 p-0.5">
            {timeframes.map((tf) => (
              <button
                key={tf}
                onClick={() => setSelectedTimeframe(tf)}
                className={`rounded px-2.5 py-1 font-mono text-[11px] font-medium transition-colors ${
                  tf === selectedTimeframe ? "bg-secondary text-foreground" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {tf}
              </button>
            ))}
          </div>
          <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-foreground">
            <Layers className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-foreground">
            <Maximize2 className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Chart */}
      <div className="h-[320px] px-2 py-3">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={priceData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={lineColor} stopOpacity={0.14} />
                <stop offset="100%" stopColor={lineColor} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="oklch(1 0 0 / 0.05)" vertical={false} />
            <XAxis
              dataKey="time"
              axisLine={false}
              tickLine={false}
              tick={{ fill: "oklch(0.62 0.006 264)", fontSize: 10 }}
              tickMargin={8}
              interval="preserveStartEnd"
              minTickGap={48}
            />
            <YAxis
              domain={["dataMin - 2", "dataMax + 2"]}
              axisLine={false}
              tickLine={false}
              tick={{ fill: "oklch(0.62 0.006 264)", fontSize: 10 }}
              tickMargin={8}
              tickFormatter={(value) => `$${value.toFixed(0)}`}
              width={48}
              orientation="right"
            />
            <Tooltip
              cursor={{ stroke: "oklch(1 0 0 / 0.12)", strokeWidth: 1 }}
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload
                  return (
                    <div className="rounded-md glass-blur px-3 py-2">
                      <p className="eyebrow !text-muted-foreground/80">{data.time}</p>
                      <p className="font-mono text-sm font-semibold tabular-nums text-foreground">
                        ${data.price.toFixed(2)}
                      </p>
                      <p className="font-mono text-[11px] tabular-nums text-muted-foreground">
                        Vol {(data.volume / 1000).toFixed(0)}K
                      </p>
                    </div>
                  )
                }
                return null
              }}
            />
            <ReferenceLine y={openPrice} stroke="oklch(1 0 0 / 0.18)" strokeDasharray="2 4" strokeWidth={1} />
            <Area
              type="monotone"
              dataKey="price"
              stroke={lineColor}
              strokeWidth={1.5}
              fill="url(#priceGradient)"
              animationDuration={700}
              activeDot={{ r: 3, strokeWidth: 0, fill: lineColor }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Footer stats */}
      <div className="flex items-center justify-between border-t border-border px-5 py-2.5">
        <div className="flex items-center gap-6">
          <Stat label="Avg Vol" value="15.2M" />
          <Stat label="52W High" value="$185.50" />
          <Stat label="52W Low" value="$68.20" />
          <Stat label="Mkt Cap" value="$302.1B" />
        </div>
        <div className="flex items-center gap-1.5 text-muted-foreground">
          <Clock className="h-3 w-3" />
          <span className="font-mono text-[11px] tabular-nums">4:00 PM EDT</span>
        </div>
      </div>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="eyebrow">{label}</span>
      <p className="font-mono text-xs font-medium tabular-nums text-foreground">{value}</p>
    </div>
  )
}
