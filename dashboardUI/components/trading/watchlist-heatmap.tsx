"use client"

import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import { 
  Grip,
  Plus,
  Settings,
  TrendingUp,
  TrendingDown,
  Star
} from "lucide-react"

interface WatchlistItem {
  symbol: string
  name: string
  price: number
  change: number
  volume: string
  marketCap: string
  isWatched: boolean
}

const watchlistData: WatchlistItem[] = [
  { symbol: "TSLA", name: "Tesla Inc", price: 391.77, change: 7.57, volume: "45.2M", marketCap: "1.24T", isWatched: true },
  { symbol: "NVDA", name: "NVIDIA Corp", price: 285.44, change: -2.31, volume: "32.1M", marketCap: "7.02T", isWatched: true },
  { symbol: "AAPL", name: "Apple Inc", price: 198.52, change: 1.23, volume: "28.4M", marketCap: "3.05T", isWatched: true },
  { symbol: "AMD", name: "AMD Inc", price: 124.88, change: 2.45, volume: "18.7M", marketCap: "202B", isWatched: true },
  { symbol: "PLTR", name: "Palantir", price: 142.10, change: 4.72, volume: "12.4M", marketCap: "302B", isWatched: true },
  { symbol: "SNAP", name: "Snap Inc", price: 6.03, change: -0.33, volume: "8.9M", marketCap: "9.8B", isWatched: true },
  { symbol: "SPY", name: "S&P 500 ETF", price: 585.23, change: 0.89, volume: "52.1M", marketCap: "-", isWatched: true },
  { symbol: "QQQ", name: "Nasdaq ETF", price: 498.76, change: 1.12, volume: "38.2M", marketCap: "-", isWatched: true },
  { symbol: "COIN", name: "Coinbase", price: 195.61, change: -0.21, volume: "6.2M", marketCap: "49.2B", isWatched: false },
  { symbol: "MARA", name: "Marathon", price: 18.45, change: 3.28, volume: "14.1M", marketCap: "5.8B", isWatched: false },
  { symbol: "RIVN", name: "Rivian", price: 16.89, change: -1.45, volume: "9.3M", marketCap: "17.2B", isWatched: false },
  { symbol: "META", name: "Meta", price: 523.45, change: 0.67, volume: "11.2M", marketCap: "1.33T", isWatched: false },
]

// Map change magnitude to a low-alpha tint bucket (intensity by alpha, not saturation)
const maxChange = Math.max(...watchlistData.map(s => Math.abs(s.change)))

// Full static class strings so Tailwind's JIT scanner can detect them.
const SUCCESS_TINTS = [
  "bg-success/[0.08]",
  "bg-success/[0.11]",
  "bg-success/[0.14]",
  "bg-success/[0.18]",
]
const DESTRUCTIVE_TINTS = [
  "bg-destructive/[0.08]",
  "bg-destructive/[0.11]",
  "bg-destructive/[0.14]",
  "bg-destructive/[0.18]",
]

function tintClass(change: number): string {
  const intensity = maxChange > 0 ? Math.abs(change) / maxChange : 0
  let idx = 0
  if (intensity >= 0.75) idx = 3
  else if (intensity >= 0.5) idx = 2
  else if (intensity >= 0.25) idx = 1
  return change >= 0 ? SUCCESS_TINTS[idx] : DESTRUCTIVE_TINTS[idx]
}

export function WatchlistHeatmap() {
  const [viewMode, setViewMode] = useState<"heatmap" | "list">("heatmap")
  const [hoveredSymbol, setHoveredSymbol] = useState<string | null>(null)

  return (
    <div className="rounded-xl glass overflow-hidden reveal">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-border/60">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold tracking-tight text-foreground">Watchlist</h3>
          <Badge variant="outline" className="text-[10px] eyebrow">
            {watchlistData.length} symbols
          </Badge>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 p-1 rounded-lg bg-secondary/30">
            <button
              onClick={() => setViewMode("heatmap")}
              className={`p-1.5 rounded-md transition-all ${
                viewMode === "heatmap"
                  ? 'bg-accent text-accent-foreground'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Grip className="h-4 w-4" />
            </button>
            <button
              onClick={() => setViewMode("list")}
              className={`p-1.5 rounded-md transition-all ${
                viewMode === "list"
                  ? 'bg-accent text-accent-foreground'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Settings className="h-4 w-4" />
            </button>
          </div>
          <button className="p-2 rounded-lg bg-secondary/30 text-muted-foreground hover:text-foreground transition-colors">
            <Plus className="h-4 w-4" />
          </button>
        </div>
      </div>
      
      {/* Heatmap View */}
      {viewMode === "heatmap" ? (
        <div className="p-4">
          <div className="grid grid-cols-4 gap-2">
            {watchlistData.map((item) => {
              const isPositive = item.change >= 0
              const isHovered = hoveredSymbol === item.symbol

              return (
                <button
                  key={item.symbol}
                  onMouseEnter={() => setHoveredSymbol(item.symbol)}
                  onMouseLeave={() => setHoveredSymbol(null)}
                  className={`lift relative p-3 rounded-lg border border-border/60 ${tintClass(item.change)} ${
                    isHovered ? 'z-10' : ''
                  }`}
                >
                  {item.isWatched && (
                    <Star className="absolute top-2 right-2 h-3 w-3 text-chart-4 fill-chart-4" />
                  )}
                  <div className="text-left">
                    <p className="font-semibold text-foreground text-sm">{item.symbol}</p>
                    <p className="font-mono tabular-nums text-foreground/90 text-xs mt-1">
                      ${item.price.toFixed(2)}
                    </p>
                    <div className={`flex items-center gap-1 mt-1 text-xs font-mono tabular-nums font-medium ${
                      isPositive ? 'text-success' : 'text-destructive'
                    }`}>
                      {isPositive ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                      {isPositive ? '+' : ''}{item.change.toFixed(2)}%
                    </div>
                  </div>

                  {/* Hover tooltip */}
                  {isHovered && (
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 rounded-lg glass-blur border border-border/60 whitespace-nowrap z-20">
                      <p className="text-xs font-medium text-foreground">{item.name}</p>
                      <p className="text-[10px] text-muted-foreground font-mono tabular-nums">Vol: {item.volume} | MCap: {item.marketCap}</p>
                    </div>
                  )}
                </button>
              )
            })}
          </div>
          
          {/* Legend */}
          <div className="flex items-center justify-center gap-6 mt-4 pt-4 border-t border-border/60">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded border border-border/60 bg-destructive/[0.18]" />
              <span className="eyebrow">Bearish</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded border border-border/60 bg-muted/20" />
              <span className="eyebrow">Neutral</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded border border-border/60 bg-success/[0.18]" />
              <span className="eyebrow">Bullish</span>
            </div>
          </div>
        </div>
      ) : (
        /* List View */
        <div className="divide-y divide-border/40">
          {watchlistData.map((item) => (
            <div
              key={item.symbol}
              className="lift flex items-center justify-between px-5 py-3"
            >
              <div className="flex items-center gap-3">
                <div className="flex items-center justify-center w-9 h-9 rounded-lg border border-border/60 bg-card text-xs font-semibold text-foreground">
                  {item.symbol.slice(0, 2)}
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-foreground text-sm">{item.symbol}</span>
                    {item.isWatched && <Star className="h-3 w-3 text-chart-4 fill-chart-4" />}
                  </div>
                  <span className="text-xs text-muted-foreground">{item.name}</span>
                </div>
              </div>
              <div className="text-right">
                <p className="font-mono tabular-nums text-foreground text-sm">${item.price.toFixed(2)}</p>
                <p className={`text-xs font-mono tabular-nums font-medium ${
                  item.change >= 0 ? 'text-success' : 'text-destructive'
                }`}>
                  {item.change >= 0 ? '+' : ''}{item.change.toFixed(2)}%
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
