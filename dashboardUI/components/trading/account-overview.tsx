"use client"

import { useState, useEffect } from "react"
import { Wallet, Zap, Activity, DollarSign, ArrowUpRight, ArrowDownRight } from "lucide-react"

interface MetricCardProps {
  title: string
  value: string
  subValue?: string
  trend?: "up" | "down" | "neutral"
  trendValue?: string
  icon: React.ReactNode
  accentColor?: "default" | "success" | "warning"
  valueColor?: "default" | "success" | "destructive"
  index?: number
}

function MetricCard({
  title,
  value,
  subValue,
  trend,
  trendValue,
  icon,
  valueColor = "default",
  index = 0,
}: MetricCardProps) {
  const valueClass =
    valueColor === "success"
      ? "text-success"
      : valueColor === "destructive"
        ? "text-destructive"
        : "text-foreground"

  return (
    <div
      className="reveal group relative overflow-hidden rounded-lg glass p-5 lift"
      style={{ animationDelay: `${index * 60}ms` }}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-secondary text-muted-foreground transition-colors group-hover:text-foreground">
            {icon}
          </div>
          <span className="eyebrow">{title}</span>
        </div>
        {trend && trend !== "neutral" && (
          <div
            className={`flex items-center gap-0.5 text-xs font-medium tnum ${
              trend === "up" ? "text-success" : "text-destructive"
            }`}
          >
            {trend === "up" ? (
              <ArrowUpRight className="h-3.5 w-3.5" />
            ) : (
              <ArrowDownRight className="h-3.5 w-3.5" />
            )}
            {trendValue}
          </div>
        )}
      </div>

      <div className="mt-4">
        <p className={`font-mono text-[1.75rem] font-semibold leading-none tracking-tight tnum ${valueClass}`}>
          {value}
        </p>
        {subValue && <p className="mt-2 text-xs text-muted-foreground">{subValue}</p>}
      </div>
    </div>
  )
}

export function AccountOverview() {
  const [dayPnL, setDayPnL] = useState(1780.73)

  useEffect(() => {
    const interval = setInterval(() => {
      setDayPnL((prev) => prev + (Math.random() - 0.5) * 50)
    }, 5000)
    return () => clearInterval(interval)
  }, [])

  const pnlPositive = dayPnL >= 0

  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold tracking-tight text-foreground">Account Overview</h2>
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <span className="live-dot inline-flex h-1.5 w-1.5 rounded-full bg-success text-success" />
          <span className="eyebrow !text-muted-foreground">Live · updated just now</span>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          index={0}
          title="Total Value"
          value="$21,237.67"
          trend="up"
          trendValue="+5.8%"
          subValue="vs. yesterday"
          icon={<Wallet className="h-4 w-4" />}
        />
        <MetricCard
          index={1}
          title="Buying Power"
          value="$30,960.83"
          subValue="Available for trading"
          icon={<Zap className="h-4 w-4" />}
        />
        <MetricCard
          index={2}
          title="Day P&L"
          value={`${pnlPositive ? "+" : "−"}$${Math.abs(dayPnL).toLocaleString("en-US", {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
          })}`}
          trend={pnlPositive ? "up" : "down"}
          trendValue="+8.4%"
          subValue="12 trades executed"
          icon={<Activity className="h-4 w-4" />}
          valueColor={pnlPositive ? "success" : "destructive"}
        />
        <MetricCard
          index={3}
          title="Cash Balance"
          value="$8,450.00"
          subValue="Settled funds"
          icon={<DollarSign className="h-4 w-4" />}
        />
      </div>
    </section>
  )
}
