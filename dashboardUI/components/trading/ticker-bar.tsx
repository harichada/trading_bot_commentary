"use client"

import { useEffect, useState, useRef } from "react"
import { Pause, Play } from "lucide-react"
import { useWsDashboard } from "@/hooks/use-bot-data"

interface TickerItem {
  symbol: string
  price: number
  change: number
  tick?: "up" | "down" | null
}

const fallbackTickers: TickerItem[] = [
  { symbol: "NVDA", price: 0, change: 0 },
  { symbol: "TSLA", price: 0, change: 0 },
  { symbol: "PLTR", price: 0, change: 0 },
]

export function TickerBar() {
  const { screener, liveQuotes, connected } = useWsDashboard()
  const [tickers, setTickers] = useState<TickerItem[]>(fallbackTickers)
  const [isPaused, setIsPaused] = useState(false)
  const prevPricesRef = useRef<Record<string, number>>({})
  const containerRef = useRef<HTMLDivElement>(null)

  // Update tickers when screener data arrives from WebSocket
  useEffect(() => {
    if (screener.length === 0) return

    const prevPrices = prevPricesRef.current
    const newTickers = screener.map((item) => {
      const prevPrice = prevPrices[item.symbol] ?? item.last
      const currentPrice = liveQuotes[item.symbol]?.price ?? item.last
      const tick: "up" | "down" | null =
        currentPrice > prevPrice ? "up" : currentPrice < prevPrice ? "down" : null

      prevPrices[item.symbol] = currentPrice

      return {
        symbol: item.symbol,
        price: currentPrice,
        change: item.change,
        tick,
      }
    })

    setTickers(newTickers)
  }, [screener, liveQuotes])

  const hasData = tickers.length > 0 && tickers.some((t) => t.price > 0)

  return (
    <div className="relative overflow-hidden border-b border-border bg-background/60">
      <div className="pointer-events-none absolute left-0 top-0 bottom-0 z-10 w-16 bg-gradient-to-r from-background to-transparent" />
      <div className="pointer-events-none absolute right-0 top-0 bottom-0 z-10 w-24 bg-gradient-to-l from-background to-transparent" />

      <div className="absolute right-3 top-1/2 z-20 -translate-y-1/2 flex items-center gap-2">
        {!connected && (
          <span className="text-[10px] text-muted-foreground">Connecting…</span>
        )}
        <button
          onClick={() => setIsPaused(!isPaused)}
          className="rounded-md border border-border bg-secondary/60 p-1 text-muted-foreground transition-colors hover:text-foreground"
          aria-label={isPaused ? "Resume ticker" : "Pause ticker"}
        >
          {isPaused ? <Play className="h-3 w-3" /> : <Pause className="h-3 w-3" />}
        </button>
      </div>

      <div
        ref={containerRef}
        className={`flex ${isPaused ? "" : hasData ? "animate-ticker" : ""}`}
        style={{ width: "max-content" }}
      >
        {(hasData ? [...tickers, ...tickers, ...tickers] : tickers).map((ticker, idx) => {
          const up = ticker.change >= 0
          const showPrice = ticker.price > 0
          return (
            <div
              key={`${ticker.symbol}-${idx}`}
              className={`flex shrink-0 items-center gap-2.5 border-r border-border/40 px-4 py-2 ${
                ticker.tick === "up" ? "flash-up" : ticker.tick === "down" ? "flash-down" : ""
              }`}
            >
              <span className="text-xs font-semibold tracking-tight text-foreground">{ticker.symbol}</span>
              <span className="font-mono text-xs tabular-nums text-muted-foreground">
                {showPrice ? ticker.price.toFixed(2) : "—"}
              </span>
              {showPrice && (
                <span
                  className={`font-mono text-[11px] tabular-nums ${up ? "text-success" : "text-destructive"}`}
                >
                  {up ? "+" : "−"}
                  {Math.abs(ticker.change).toFixed(2)}%
                </span>
              )}
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
