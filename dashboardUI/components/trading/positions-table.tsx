"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import {
  X,
  ExternalLink,
  ChevronRight
} from "lucide-react"

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
}

interface PositionsTableProps {
  title: string
  badge?: string
  badgeVariant?: "default" | "outline" | "secondary"
  positions: Position[]
  showStopLoss?: boolean
  showTarget?: boolean
}

export const schwabPositions: Position[] = [
  { symbol: "HOGE", shares: 5100, avgCost: 0.02, current: 0.00, marketValue: 2.04, dayPL: 17.55, totalPL: 0.00, change: 0.00 },
  { symbol: "PLTR", shares: 278, avgCost: 173.22, current: 142.10, marketValue: 39503.80, dayPL: 1779.20, totalPL: 0.00, change: 4.72, isLongTerm: true },
]

export const simulatedPositions: Position[] = [
  { symbol: "SNAP", shares: 1652, avgCost: 6.05, current: 6.03, marketValue: 9961.56, dayPL: -33.04, totalPL: -33.04, change: -0.33, stopLoss: 5.94, target: 6.26 },
  { symbol: "COIN", shares: 150, avgCost: 196.03, current: 195.61, marketValue: 29341.50, dayPL: -63.00, totalPL: -63.00, change: -0.21, stopLoss: 194.70, target: 198.69 },
  { symbol: "PINS", shares: 1439, avgCost: 20.43, current: 20.30, marketValue: 29211.70, dayPL: -187.07, totalPL: -187.07, change: -0.64, stopLoss: 20.10, target: 20.70 },
]

export function PositionsTable({
  title,
  badge,
  badgeVariant = "default",
  positions,
  showStopLoss,
  showTarget
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
          <span className="text-xs tabular-nums">
            {positions.length} position{positions.length !== 1 ? 's' : ''}
          </span>
          <ChevronRight className="h-4 w-4" />
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
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
              <th className="eyebrow text-right px-4 py-2.5">Day P&L</th>
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
                    {position.isLongTerm && (
                      <span className="inline-flex items-center rounded border border-border/70 px-1 py-0 text-[9px] font-medium tracking-wide text-muted-foreground">
                        LT
                      </span>
                    )}
                  </div>
                </td>
                <td className="text-right px-4 py-3 font-mono tabular-nums text-sm text-muted-foreground">
                  {position.shares.toLocaleString()}
                </td>
                <td className="text-right px-4 py-3 font-mono tabular-nums text-sm text-muted-foreground">
                  ${position.avgCost.toFixed(2)}
                </td>
                <td className="text-right px-4 py-3 font-mono tabular-nums text-sm text-foreground">
                  ${position.current.toFixed(2)}
                </td>
                {!showStopLoss && (
                  <td className="text-right px-4 py-3 font-mono tabular-nums text-sm text-muted-foreground">
                    ${position.marketValue.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                  </td>
                )}
                {showStopLoss && (
                  <td className="text-right px-4 py-3 font-mono tabular-nums text-sm text-muted-foreground">
                    ${position.stopLoss?.toFixed(2) ?? '-'}
                  </td>
                )}
                {showTarget && (
                  <td className="text-right px-4 py-3 font-mono tabular-nums text-sm text-muted-foreground">
                    ${position.target?.toFixed(2) ?? '-'}
                  </td>
                )}
                <td className="text-right px-4 py-3">
                  <PnLCell value={position.dayPL} />
                </td>
                <td className="text-right px-4 py-3">
                  <ChangeCell value={position.change} />
                </td>
                <td className="text-right px-4 py-3">
                  <div className="flex items-center justify-end gap-1.5">
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
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
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
      {isZero ? '' : isPositive ? '+' : '-'}${Math.abs(value).toLocaleString('en-US', { minimumFractionDigits: 2 })}
    </span>
  )
}

function ChangeCell({ value }: { value: number }) {
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
      {isPositive ? '+' : ''}{value.toFixed(2)}%
    </span>
  )
}
