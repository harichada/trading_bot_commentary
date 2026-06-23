"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import {
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  ChevronDown,
  ArrowRight
} from "lucide-react"

const recentSymbols = ["PLTR", "TSLA", "NVDA", "AAPL", "SNAP", "COIN"]

export function QuickTrade() {
  const [orderType, setOrderType] = useState<"buy" | "sell">("buy")
  const [selectedSymbol, setSelectedSymbol] = useState("PLTR")
  const [quantity, setQuantity] = useState("100")
  const [orderStyle, setOrderStyle] = useState<"market" | "limit" | "stop">("market")
  const [limitPrice, setLimitPrice] = useState("")

  const currentPrice = 142.10
  const estimatedCost = parseInt(quantity || "0") * currentPrice
  const isBuy = orderType === "buy"

  return (
    <div className="rounded-xl glass overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-border/40">
        <h3 className="text-sm font-semibold tracking-tight text-foreground">Quick Trade</h3>
        <span className="eyebrow">Order entry</span>
      </div>

      <div className="p-4 space-y-4">
        {/* Buy/Sell Toggle */}
        <div className="grid grid-cols-2 gap-1 p-0.5 rounded-lg bg-secondary/30 border border-border/40">
          <button
            onClick={() => setOrderType("buy")}
            className={`py-2.5 rounded-md font-semibold text-sm transition-colors flex items-center justify-center gap-2 ${
              isBuy
                ? 'bg-success text-success-foreground'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <TrendingUp className="h-4 w-4" />
            BUY
          </button>
          <button
            onClick={() => setOrderType("sell")}
            className={`py-2.5 rounded-md font-semibold text-sm transition-colors flex items-center justify-center gap-2 ${
              !isBuy
                ? 'bg-destructive text-destructive-foreground'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <TrendingDown className="h-4 w-4" />
            SELL
          </button>
        </div>

        {/* Symbol Selector */}
        <div className="space-y-1.5">
          <label className="eyebrow">Symbol</label>
          <div className="relative">
            <select
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              className="w-full h-11 px-3.5 rounded-lg bg-secondary/30 border border-border/40 text-foreground font-mono tabular-nums font-semibold appearance-none cursor-pointer focus:outline-none focus:ring-1 focus:ring-accent/50 focus:border-accent/50"
            >
              {recentSymbols.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
            <ChevronDown className="absolute right-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
          </div>
          <div className="flex items-center justify-between text-xs pt-0.5">
            <span className="text-muted-foreground">Current price</span>
            <span className="font-mono tabular-nums font-medium text-foreground">${currentPrice.toFixed(2)}</span>
          </div>
        </div>

        {/* Order Type */}
        <div className="space-y-1.5">
          <label className="eyebrow">Order type</label>
          <div className="grid grid-cols-3 gap-1.5">
            {(["market", "limit", "stop"] as const).map((type) => (
              <button
                key={type}
                onClick={() => setOrderStyle(type)}
                className={`py-2 rounded-md text-xs font-medium transition-colors capitalize border ${
                  orderStyle === type
                    ? 'bg-accent/12 text-accent border-accent/30'
                    : 'bg-secondary/30 text-muted-foreground border-border/40 hover:text-foreground'
                }`}
              >
                {type}
              </button>
            ))}
          </div>
        </div>

        {/* Quantity Input */}
        <div className="space-y-1.5">
          <label className="eyebrow">Quantity</label>
          <div className="relative">
            <input
              type="number"
              value={quantity}
              onChange={(e) => setQuantity(e.target.value)}
              className="w-full h-11 px-3.5 rounded-lg bg-secondary/30 border border-border/40 text-foreground font-mono tabular-nums font-semibold text-base focus:outline-none focus:ring-1 focus:ring-accent/50 focus:border-accent/50"
              placeholder="0"
            />
            <div className="absolute right-2.5 top-1/2 -translate-y-1/2 flex items-center gap-1">
              <button
                onClick={() => setQuantity(String(Math.max(0, parseInt(quantity || "0") - 10)))}
                className="px-2 py-1 rounded bg-secondary/50 text-xs font-mono tabular-nums font-medium text-muted-foreground hover:text-foreground transition-colors"
              >
                -10
              </button>
              <button
                onClick={() => setQuantity(String(parseInt(quantity || "0") + 10))}
                className="px-2 py-1 rounded bg-secondary/50 text-xs font-mono tabular-nums font-medium text-muted-foreground hover:text-foreground transition-colors"
              >
                +10
              </button>
            </div>
          </div>
          <div className="flex items-center gap-1.5">
            {[25, 50, 75, 100].map((pct) => (
              <button
                key={pct}
                onClick={() => setQuantity(String(Math.floor((30960.83 / currentPrice) * (pct / 100))))}
                className="flex-1 py-1.5 rounded-md bg-secondary/20 text-xs font-mono tabular-nums font-medium text-muted-foreground hover:text-foreground hover:bg-secondary/40 transition-colors"
              >
                {pct}%
              </button>
            ))}
          </div>
        </div>

        {/* Limit Price (if applicable) */}
        {orderStyle !== "market" && (
          <div className="space-y-1.5">
            <label className="eyebrow">
              {orderStyle === "limit" ? "Limit price" : "Stop price"}
            </label>
            <div className="relative">
              <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-sm text-muted-foreground font-mono">$</span>
              <input
                type="number"
                value={limitPrice}
                onChange={(e) => setLimitPrice(e.target.value)}
                className="w-full h-11 pl-7 pr-3.5 rounded-lg bg-secondary/30 border border-border/40 text-foreground font-mono tabular-nums font-semibold focus:outline-none focus:ring-1 focus:ring-accent/50 focus:border-accent/50"
                placeholder={currentPrice.toFixed(2)}
                step="0.01"
              />
            </div>
          </div>
        )}

        {/* Order Summary */}
        <div className="p-4 rounded-lg bg-secondary/20 border border-border/40 space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Estimated cost</span>
            <span className="font-mono tabular-nums font-semibold text-foreground text-base">
              ${estimatedCost.toLocaleString('en-US', { minimumFractionDigits: 2 })}
            </span>
          </div>
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Commission</span>
            <span className="font-mono tabular-nums text-foreground">$0.00</span>
          </div>
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Available buying power</span>
            <span className="font-mono tabular-nums text-muted-foreground">$30,960.83</span>
          </div>

          {estimatedCost > 30960.83 && (
            <div className="flex items-center gap-2 p-2 rounded-md bg-destructive/[0.08] border border-destructive/25 text-destructive text-xs">
              <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
              <span>Insufficient buying power</span>
            </div>
          )}
        </div>

        {/* Submit Button */}
        <Button
          className={`w-full h-12 text-sm font-semibold rounded-lg transition-colors ${
            isBuy
              ? 'bg-success hover:bg-success/90 text-success-foreground'
              : 'bg-destructive hover:bg-destructive/90 text-destructive-foreground'
          }`}
          disabled={!quantity || parseInt(quantity) === 0 || estimatedCost > 30960.83}
        >
          {isBuy ? "Buy" : "Sell"} <span className="font-mono tabular-nums mx-1">{quantity || 0}</span> {selectedSymbol}
          <ArrowRight className="h-4 w-4 ml-2" />
        </Button>

        {/* Disclaimer */}
        <p className="text-[10px] leading-relaxed text-muted-foreground text-center">
          Orders are executed at best available price. Market orders execute immediately during market hours.
        </p>
      </div>
    </div>
  )
}
