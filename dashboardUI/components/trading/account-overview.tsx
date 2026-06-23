"use client"

import { Wallet, Zap, Activity, TrendingUp, ArrowUpRight, ArrowDownRight } from "lucide-react"
import { useAccountStats, fmtUsd } from "@/hooks/use-bot-data"
import type { AccountSource } from "@/lib/api"

interface MetricCardProps {
  title: string
  value: string
  subValue?: string
  trend?: "up" | "down" | "neutral"
  trendValue?: string
  icon: React.ReactNode
  valueColor?: "default" | "success" | "destructive"
  index?: number
}

function MetricCard({ title, value, subValue, trend, trendValue, icon, valueColor = "default", index = 0 }: MetricCardProps) {
  const valueClass =
    valueColor === "success" ? "text-success" : valueColor === "destructive" ? "text-destructive" : "text-foreground"

  return (
    <div className="reveal group relative overflow-hidden rounded-lg glass p-5 lift" style={{ animationDelay: `${index * 60}ms` }}>
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-secondary text-muted-foreground transition-colors group-hover:text-foreground">
            {icon}
          </div>
          <span className="eyebrow">{title}</span>
        </div>
        {trend && trend !== "neutral" && trendValue && (
          <div className={`flex items-center gap-0.5 text-xs font-medium tnum ${trend === "up" ? "text-success" : "text-destructive"}`}>
            {trend === "up" ? <ArrowUpRight className="h-3.5 w-3.5" /> : <ArrowDownRight className="h-3.5 w-3.5" />}
            {trendValue}
          </div>
        )}
      </div>
      <div className="mt-4">
        <p className={`font-mono text-[1.75rem] font-semibold leading-none tracking-tight tnum ${valueClass}`}>{value}</p>
        {subValue && <p className="mt-2 text-xs text-muted-foreground">{subValue}</p>}
      </div>
    </div>
  )
}

function SourceBadge({ source, live }: { source: AccountSource | undefined; live: boolean }) {
  if (!live) {
    return (
      <span className="flex items-center gap-1.5 rounded border border-border px-1.5 py-0.5">
        <span className="inline-flex h-1.5 w-1.5 rounded-full bg-warning" />
        <span className="eyebrow !text-muted-foreground">Connecting…</span>
      </span>
    )
  }
  const isSchwab = source === "schwab"
  return (
    <span className={`flex items-center gap-1.5 rounded border px-1.5 py-0.5 ${isSchwab ? "border-success/30" : "border-border"}`}>
      <span className={`live-dot inline-flex h-1.5 w-1.5 rounded-full ${isSchwab ? "bg-success text-success" : "bg-muted-foreground text-muted-foreground"}`} />
      <span className="eyebrow !text-muted-foreground">{isSchwab ? "Live · Schwab" : source === "simulation" ? "Simulated" : "Internal"}</span>
    </span>
  )
}

export function AccountOverview() {
  const { data, live } = useAccountStats()
  const s = data?.stats

  const dayPnl = s?.daily_pnl ?? null
  const totalPnl = s?.total_pnl ?? null
  const posCount = s?.position_count

  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold tracking-tight text-foreground">Account Overview</h2>
        <SourceBadge source={data?.source} live={live} />
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard index={0} title="Total Value" value={fmtUsd(s?.balance)} subValue="Account balance" icon={<Wallet className="h-4 w-4" />} />
        <MetricCard
          index={1}
          title="Buying Power"
          value={fmtUsd(s?.buying_power)}
          subValue="Available for trading"
          icon={<Zap className="h-4 w-4" />}
          valueColor={s != null && s.buying_power < 0 ? "destructive" : "default"}
        />
        <MetricCard
          index={2}
          title="Day P&L"
          value={fmtUsd(dayPnl, true)}
          subValue={posCount != null ? `${posCount} open position${posCount === 1 ? "" : "s"}` : undefined}
          trend={dayPnl == null ? undefined : dayPnl >= 0 ? "up" : "down"}
          icon={<Activity className="h-4 w-4" />}
          valueColor={dayPnl == null ? "default" : dayPnl >= 0 ? "success" : "destructive"}
        />
        <MetricCard
          index={3}
          title="Total P&L"
          value={fmtUsd(totalPnl, true)}
          subValue="Open positions"
          trend={totalPnl == null ? undefined : totalPnl >= 0 ? "up" : "down"}
          icon={<TrendingUp className="h-4 w-4" />}
          valueColor={totalPnl == null ? "default" : totalPnl >= 0 ? "success" : "destructive"}
        />
      </div>
    </section>
  )
}
