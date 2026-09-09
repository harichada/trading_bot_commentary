"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import {
  X,
  ExternalLink,
  ChevronRight,
  Loader2,
  Brain
} from "lucide-react"
import { usePositions, fmtUsd } from "@/hooks/use-bot-data"
import { useDecisionCard } from "@/components/trading/decision-card"
import type { Position as ApiPosition } from "@/lib/api"

interface Position {
  symbol: string
  shares: number
  avgCost: number
  current: number
  marketValue: number
  dayPL: number
  totalPL: number
  change: number
  isLongTerm?: boolean
  stopLoss?: number
  target?: number
  managedByBot?: boolean
  side?: string
  strategy?: string | null
  updatedAt?: string | null
}

interface PositionsTableProps {
  title: string
  badge?: string
  badgeVariant?: "default" | "outline" | "secondary"
  positions: Position[]
  showStopLoss?: boolean
  showTarget?: boolean
  loading?: boolean
  error?: string | null
}

function apiToDisplayPosition(pos: ApiPosition): Position {
  const change = pos.entry_price > 0
    ? ((pos.current_price - pos.entry_price) / pos.entry_price) * 100
    : 0

  return {
    symbol: pos.symbol,
    shares: pos.quantity,
    avgCost: pos.entry_price,
    current: pos.current_price,
    marketValue: pos.quantity * pos.current_price,
    dayPL: pos.unrealized_pnl,
    totalPL: pos.unrealized_pnl,
    change,
    stopLoss: pos.stop_loss > 0 ? pos.stop_loss : undefined,
    target: pos.take_profit > 0 ? pos.take_profit : undefined,
    managedByBot: pos.managed_by_bot,
    side: pos.side,
    strategy: pos.strategy,
    updatedAt: pos.updated_at,
  }
}

export function PositionsTable({
  title,
  badge,
  badgeVariant = "default",
  positions,
  showStopLoss,
  showTarget,
  loading,
  error,
}: PositionsTableProps) {
  const [hoveredRow, setHoveredRow] = useState<string | null>(null)

  const isLive = badgeVariant !== "outline"

  return (
    <div className="glass rounded-lg overflow-hidden reveal">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2.5">
          <h3 className="text-sm font-semibold tracking-tight text-foreground">{title}</h3>
          {badge && (
            <span
              className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-semibold tracking-wide ${
                isLive
                  ? "border-success/30 bg-success/[0.10] text-success"
                  : "border-border/70 bg-transparent text-muted-foreground"
              }`}
            >
              {badge}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 text-muted-foreground">
          {loading ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <span className="text-xs tabular-nums">
              {positions.length} position{positions.length !== 1 ? 's' : ''}
            </span>
          )}
          <ChevronRight className="h-4 w-4" />
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        {error ? (
          <div className="px-4 py-8 text-center text-sm text-muted-foreground">
            {error}
          </div>
        ) : positions.length === 0 && !loading ? (
          <div className="px-4 py-8 text-center text-sm text-muted-foreground">
            No open positions
          </div>
        ) : (
          <table className="w-full">
            <thead>
              <tr className="border-b border-border/50">
                <th className="eyebrow text-left px-4 py-2.5">Symbol</th>
                <th className="eyebrow text-right px-4 py-2.5">Shares</th>
                <th className="eyebrow text-right px-4 py-2.5">Avg Cost</th>
                <th className="eyebrow text-right px-4 py-2.5">Current</th>
                {!showStopLoss && (
                  <th className="eyebrow text-right px-4 py-2.5">Mkt Value</th>
                )}
                {showStopLoss && (
                  <th className="eyebrow text-right px-4 py-2.5">Stop Loss</th>
                )}
                {showTarget && (
                  <th className="eyebrow text-right px-4 py-2.5">Target</th>
                )}
                <th className="eyebrow text-right px-4 py-2.5">P&L</th>
                <th className="eyebrow text-right px-4 py-2.5">% Change</th>
                <th className="eyebrow text-right px-4 py-2.5">Actions</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((position) => (
                <tr
                  key={position.symbol}
                  className={`lift border-b border-border/50 ${
                    hoveredRow === position.symbol ? 'bg-accent/[0.04]' : ''
                  }`}
                  onMouseEnter={() => setHoveredRow(position.symbol)}
                  onMouseLeave={() => setHoveredRow(null)}
                >
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold tracking-tight text-foreground text-sm">{position.symbol}</span>
                      {position.managedByBot && (
                        <span className="inline-flex items-center rounded border border-accent/30 px-1 py-0 text-[9px] font-medium tracking-wide text-accent">
                          BOT
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="text-right px-4 py-3 font-mono tabular-nums text-sm text-muted-foreground">
                    {position.shares.toLocaleString()}
                  </td>
                  <td className="text-right px-4 py-3 font-mono tabular-nums text-sm text-muted-foreground">
                    {fmtUsd(position.avgCost)}
                  </td>
                  <td className="text-right px-4 py-3 font-mono tabular-nums text-sm text-foreground">
                    {fmtUsd(position.current)}
                  </td>
                  {!showStopLoss && (
                    <td className="text-right px-4 py-3 font-mono tabular-nums text-sm text-muted-foreground">
                      {fmtUsd(position.marketValue)}
                    </td>
                  )}
                  {showStopLoss && (
                    <td className="text-right px-4 py-3 font-mono tabular-nums text-sm text-muted-foreground">
                      {position.stopLoss ? fmtUsd(position.stopLoss) : '—'}
                    </td>
                  )}
                  {showTarget && (
                    <td className="text-right px-4 py-3 font-mono tabular-nums text-sm text-muted-foreground">
                      {position.target ? fmtUsd(position.target) : '—'}
                    </td>
                  )}
                  <td className="text-right px-4 py-3">
                    <PnLCell value={position.totalPL} />
                  </td>
                  <td className="text-right px-4 py-3">
                    <ChangeCell value={position.change} />
                  </td>
                  <td className="text-right px-4 py-3">
                    <PositionActions position={position} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

function PositionActions({ position }: { position: Position }) {
  const { openDecision } = useDecisionCard()

  const handleViewDecision = () => {
    const stance = position.side === "short" ? "short" : "long"
    const keyFacts: Array<{ label: string; value: string }> = [
      { label: "Entry Price", value: `$${position.avgCost.toFixed(2)}` },
      { label: "Current Price", value: `$${position.current.toFixed(2)}` },
      { label: "Quantity", value: position.shares.toLocaleString() },
    ]
    if (position.stopLoss) {
      keyFacts.push({ label: "Stop Loss", value: `$${position.stopLoss.toFixed(2)}` })
    }
    if (position.target) {
      keyFacts.push({ label: "Take Profit", value: `$${position.target.toFixed(2)}` })
    }
    openDecision({
      symbol: position.symbol,
      timestamp: position.updatedAt ?? new Date().toISOString(),
      stance: stance as "long" | "short",
      keyFacts,
      source: position.strategy ?? undefined,
    })
  }

  return (
    <div className="flex items-center justify-end gap-1.5">
      {position.managedByBot && (
        <Button
          variant="ghost"
          size="sm"
          className="h-7 w-7 p-0 text-accent hover:text-accent hover:bg-accent/10"
          onClick={handleViewDecision}
          title="View decision"
        >
          <Brain className="h-3.5 w-3.5" />
        </Button>
      )}
      <Button
        variant="ghost"
        size="sm"
        className="h-7 w-7 p-0 text-muted-foreground hover:text-foreground"
      >
        <ExternalLink className="h-3.5 w-3.5" />
      </Button>
      <Button
        variant="ghost"
        size="sm"
        className="h-7 px-2.5 text-xs gap-1.5 text-muted-foreground hover:text-destructive"
      >
        <X className="h-3 w-3" />
        Close
      </Button>
    </div>
  )
}

function PnLCell({ value }: { value: number }) {
  const isZero = value === 0
  const isPositive = value > 0
  return (
    <span className={`font-mono tabular-nums text-sm ${
      isZero
        ? 'text-muted-foreground'
        : isPositive
          ? 'text-success'
          : 'text-destructive'
    }`}>
      {fmtUsd(value, true)}
    </span>
  )
}

function ChangeCell({ value }: { value: number }) {
  const isZero = Math.abs(value) < 0.01
  const isPositive = value > 0
  return (
    <span className={`font-mono tabular-nums text-sm ${
      isZero
        ? 'text-muted-foreground'
        : isPositive
          ? 'text-success'
          : 'text-destructive'
    }`}>
      {isPositive ? '+' : ''}{value.toFixed(2)}%
    </span>
  )
}

export function LivePositionsSection() {
  const { data, loading, error, live } = usePositions()

  const positions = data?.positions ?? []

  const livePositions = positions
    .filter((p) => p.mode === "live")
    .map(apiToDisplayPosition)

  const simPositions = positions
    .filter((p) => p.mode === "simulation")
    .map(apiToDisplayPosition)

  return (
    <section className="space-y-6">
      <PositionsTable
        title="Live Account Positions"
        badge={live ? "LIVE" : "OFFLINE"}
        badgeVariant={live ? "default" : "outline"}
        positions={livePositions}
        loading={loading}
        error={error}
      />

      <PositionsTable
        title="Simulated Positions"
        badge="SIM"
        badgeVariant="outline"
        positions={simPositions}
        showStopLoss
        showTarget
        loading={loading}
        error={error}
      />
    </section>
  )
}
