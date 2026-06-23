"use client"

import { useEffect, useState, useRef } from "react"
import { Pause, Play } from "lucide-react"

interface TickerItem {
  symbol: string
  price: number
  change: number
  tick?: "up" | "down" | null
}

const initialTickers: TickerItem[] = [
  { symbol: "PINS", price: 28.36, change: 8.88 },
  { symbol: "SNAP", price: 6.81, change: 6.52 },
  { symbol: "TSLA", price: 391.77, change: 7.57 },
  { symbol: "LCID", price: 8.23, change: -6.48 },
  { symbol: "COIN", price: 335.89, change: 5.98 },
  { symbol: "PLTR", price: 142.11, change: 4.72 },
  { symbol: "RIOT", price: 17.41, change: -3.52 },
  { symbol: "FUBO", price: 13.16, change: 6.95 },
  { symbol: "MARA", price: 18.47, change: -0.14 },
  { symbol: "RIVN", price: 18.45, change: 2.85 },
  { symbol: "NVDA", price: 124.82, change: 3.21 },
  { symbol: "AMD", price: 156.34, change: 2.45 },
]

export function TickerBar() {
  const [tickers, setTickers] = useState(initialTickers)
  const [isPaused, setIsPaused] = useState(false)
  const [mounted, setMounted] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  // Gate randomized updates behind mount → first paint stays deterministic (no hydration mismatch)
  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (!mounted) return
    const interval = setInterval(() => {
      setTickers((prev) => {
        const randomIndex = Math.floor(Math.random() * prev.length)
        return prev.map((ticker, idx) => {
          if (idx === randomIndex) {
            const delta = (Math.random() - 0.5) * 0.8
            return {
              ...ticker,
              price: ticker.price + delta,
              change: ticker.change + (Math.random() - 0.5) * 0.15,
              tick: delta >= 0 ? "up" : "down",
            }
          }
          return { ...ticker, tick: null }
        })
      })
    }, 1600)
    return () => clearInterval(interval)
  }, [mounted])

  return (
    <div className="relative overflow-hidden border-b border-border bg-background/60">
      <div className="pointer-events-none absolute left-0 top-0 bottom-0 z-10 w-16 bg-gradient-to-r from-background to-transparent" />
      <div className="pointer-events-none absolute right-0 top-0 bottom-0 z-10 w-24 bg-gradient-to-l from-background to-transparent" />

      <button
        onClick={() => setIsPaused(!isPaused)}
        className="absolute right-3 top-1/2 z-20 -translate-y-1/2 rounded-md border border-border bg-secondary/60 p-1 text-muted-foreground transition-colors hover:text-foreground"
        aria-label={isPaused ? "Resume ticker" : "Pause ticker"}
      >
        {isPaused ? <Play className="h-3 w-3" /> : <Pause className="h-3 w-3" />}
      </button>

      <div
        ref={containerRef}
        className={`flex ${isPaused ? "" : "animate-ticker"}`}
        style={{ width: "max-content" }}
      >
        {[...tickers, ...tickers, ...tickers].map((ticker, idx) => {
          const up = ticker.change >= 0
          return (
            <div
              key={`${ticker.symbol}-${idx}`}
              className={`flex shrink-0 items-center gap-2.5 border-r border-border/40 px-4 py-2 ${
                ticker.tick === "up" ? "flash-up" : ticker.tick === "down" ? "flash-down" : ""
              }`}
            >
              <span className="text-xs font-semibold tracking-tight text-foreground">{ticker.symbol}</span>
              <span className="font-mono text-xs tabular-nums text-muted-foreground">
                {ticker.price.toFixed(2)}
              </span>
              <span
                className={`font-mono text-[11px] tabular-nums ${up ? "text-success" : "text-destructive"}`}
              >
                {up ? "+" : "−"}
                {Math.abs(ticker.change).toFixed(2)}%
              </span>
            </div>
          )
        })}
      </div>

      <style jsx>{`
        @keyframes ticker {
          0% { transform: translateX(0); }
          100% { transform: translateX(-33.333%); }
        }
        .animate-ticker { animation: ticker 48s linear infinite; }
        .animate-ticker:hover { animation-play-state: paused; }
      `}</style>
    </div>
  )
}
