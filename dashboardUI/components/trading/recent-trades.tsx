"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import {
  RefreshCw,
  Clock,
  Filter,
  Download,
  ChevronRight
} from "lucide-react"

interface Trade {
  symbol: string
  entryTime: string
  exitTime: string
  pnl: number
  reason: string
  type: "buy" | "sell"
}

const recentTrades: Trade[] = [
  { symbol: "RIOT", entryTime: "2:11:18 PM", exitTime: "3:51:49 PM", pnl: 348.20, reason: "manual_override", type: "buy" },
  { symbol: "LCID", entryTime: "1:15:08 PM", exitTime: "3:15:58 PM", pnl: 498.96, reason: "manual_override", type: "buy" },
  { symbol: "NVDA", entryTime: "12:58:50 PM", exitTime: "3:52:02 PM", pnl: 270.66, reason: "manual_override", type: "buy" },
  { symbol: "TSLA", entryTime: "12:36:28 PM", exitTime: "3:52:05 PM", pnl: 66.96, reason: "manual_override", type: "buy" },
  { symbol: "PLTR", entryTime: "12:28:32 PM", exitTime: "3:52:13 PM", pnl: 638.82, reason: "manual_override", type: "buy" },
  { symbol: "PINS", entryTime: "12:28:19 PM", exitTime: "3:52:08 PM", pnl: 753.15, reason: "manual_override", type: "buy" },
  { symbol: "TSLA", entryTime: "9:40:11 AM", exitTime: "10:28:47 AM", pnl: -78.02, reason: "stop_loss", type: "buy" },
]

function getReasonBadge(reason: string) {
  switch (reason) {
    case "manual_override":
      return (
        <span className="inline-flex items-center rounded border border-border/70 bg-transparent px-1.5 py-0.5 text-[10px] font-medium tracking-wide text-muted-foreground">
          Manual
        </span>
      )
    case "stop_loss":
      return (
        <span className="inline-flex items-center rounded border border-destructive/30 bg-destructive/[0.08] px-1.5 py-0.5 text-[10px] font-medium tracking-wide text-destructive">
          Stop Loss
        </span>
      )
    case "target_hit":
      return (
        <span className="inline-flex items-center rounded border border-success/30 bg-success/[0.08] px-1.5 py-0.5 text-[10px] font-medium tracking-wide text-success">
          Target Hit
        </span>
      )
    default:
      return (
        <span className="inline-flex items-center rounded border border-border/70 bg-transparent px-1.5 py-0.5 text-[10px] font-medium tracking-wide text-muted-foreground">
          {reason.replace('_', ' ')}
        </span>
      )
  }
}

export function RecentTrades() {
  const [isRefreshing, setIsRefreshing] = useState(false)

  const handleRefresh = () => {
    setIsRefreshing(true)
    setTimeout(() => setIsRefreshing(false), 1000)
  }

  const totalPnL = recentTrades.reduce((sum, trade) => sum + trade.pnl, 0)
  const winningTrades = recentTrades.filter(t => t.pnl > 0).length
  const winRate = (winningTrades / recentTrades.length) * 100

  return (
    <div className="glass rounded-lg overflow-hidden reveal">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2.5">
          <h3 className="text-sm font-semibold tracking-tight text-foreground">Recent Bot Trades</h3>
          <span className="inline-flex items-center rounded border border-border/70 bg-transparent px-1.5 py-0.5 text-[10px] font-medium tracking-wide text-muted-foreground">
            Today
          </span>
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            className="h-7 gap-1.5 text-xs text-muted-foreground hover:text-foreground"
          >
            <Filter className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Filter</span>
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-7 gap-1.5 text-xs text-muted-foreground hover:text-foreground"
          >
            <Download className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Export</span>
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-7 w-7 p-0 text-muted-foreground hover:text-foreground"
            onClick={handleRefresh}
          >
            <RefreshCw className={`h-3.5 w-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
          </Button>
        </div>
      </div>

      {/* Stats Bar */}
      <div className="flex items-center gap-6 px-4 py-2.5 border-b border-border/50">
        <div className="flex items-center gap-2">
          <span className="eyebrow">Total P&L</span>
          <span className={`text-sm font-mono tabular-nums ${totalPnL >= 0 ? 'text-success' : 'text-destructive'}`}>
            {totalPnL >= 0 ? '+' : '-'}${Math.abs(totalPnL).toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </span>
        </div>
        <div className="w-px h-4 bg-border/50" />
        <div className="flex items-center gap-2">
          <span className="eyebrow">Trades</span>
          <span className="text-sm font-mono text-foreground tabular-nums">{recentTrades.length}</span>
        </div>
        <div className="w-px h-4 bg-border/50" />
        <div className="flex items-center gap-2">
          <span className="eyebrow">Win Rate</span>
          <span className={`text-sm font-mono tabular-nums ${winRate >= 50 ? 'text-success' : 'text-destructive'}`}>
            {winRate.toFixed(0)}%
          </span>
        </div>
        <div className="ml-auto flex items-center gap-1.5 text-[10px] text-muted-foreground">
          <Clock className="h-3 w-3" />
          Updated just now
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border/50">
              <th className="eyebrow text-left px-4 py-2.5">Symbol</th>
              <th className="eyebrow text-left px-4 py-2.5">Entry Time</th>
              <th className="eyebrow text-left px-4 py-2.5">Exit Time</th>
              <th className="eyebrow text-right px-4 py-2.5">P&L</th>
              <th className="eyebrow text-left px-4 py-2.5">Reason</th>
            </tr>
          </thead>
          <tbody>
            {recentTrades.map((trade, idx) => (
              <tr
                key={`${trade.symbol}-${idx}`}
                className="lift border-b border-border/50"
              >
                <td className="px-4 py-3">
                  <span className="font-semibold tracking-tight text-foreground text-sm">{trade.symbol}</span>
                </td>
                <td className="px-4 py-3">
                  <span className="font-mono tabular-nums text-sm text-muted-foreground">{trade.entryTime}</span>
                </td>
                <td className="px-4 py-3">
                  <span className="font-mono tabular-nums text-sm text-muted-foreground">{trade.exitTime}</span>
                </td>
                <td className="px-4 py-3 text-right">
                  <span className={`font-mono tabular-nums text-sm ${
                    trade.pnl >= 0 ? 'text-success' : 'text-destructive'
                  }`}>
                    {trade.pnl >= 0 ? '+' : '-'}${Math.abs(trade.pnl).toLocaleString('en-US', { minimumFractionDigits: 2 })}
                  </span>
                </td>
                <td className="px-4 py-3">
                  {getReasonBadge(trade.reason)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-center py-2.5 border-t border-border/50">
        <Button variant="ghost" size="sm" className="h-7 gap-1.5 text-xs text-muted-foreground hover:text-foreground">
          View all trades
          <ChevronRight className="h-3.5 w-3.5" />
        </Button>
      </div>
    </div>
  )
}
