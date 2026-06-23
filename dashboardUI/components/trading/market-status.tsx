"use client"

import { useState, useEffect } from "react"
import {
  Sun,
  Moon,
  Sunrise,
  Clock,
  Globe,
  TrendingUp,
  TrendingDown,
  Activity
} from "lucide-react"

interface MarketIndex {
  name: string
  value: number
  change: number
  changePercent: number
}

const marketIndices: MarketIndex[] = [
  { name: "S&P 500", value: 5847.23, change: 52.18, changePercent: 0.90 },
  { name: "NASDAQ", value: 18562.45, change: 178.32, changePercent: 0.97 },
  { name: "DOW", value: 43567.12, change: -45.23, changePercent: -0.10 },
  { name: "VIX", value: 14.23, change: -0.82, changePercent: -5.45 },
]

function getMarketStatus(now: Date | null): { status: "pre" | "open" | "after" | "closed"; label: string; color: string } {
  if (!now) return { status: "closed", label: "Loading…", color: "text-muted-foreground" }
  const hours = now.getHours()
  const minutes = now.getMinutes()
  const time = hours * 60 + minutes
  const day = now.getDay()

  // Weekend check
  if (day === 0 || day === 6) {
    return { status: "closed", label: "Weekend - Market Closed", color: "text-muted-foreground" }
  }

  // Pre-market: 4:00 AM - 9:30 AM
  if (time >= 240 && time < 570) {
    return { status: "pre", label: "Pre-Market", color: "text-foreground" }
  }

  // Market hours: 9:30 AM - 4:00 PM
  if (time >= 570 && time < 960) {
    return { status: "open", label: "Market Open", color: "text-success" }
  }

  // After hours: 4:00 PM - 8:00 PM
  if (time >= 960 && time < 1200) {
    return { status: "after", label: "After Hours", color: "text-foreground" }
  }

  return { status: "closed", label: "Market Closed", color: "text-muted-foreground" }
}

function getTimeUntilNextSession(now: Date | null): string {
  if (!now) return "—"
  const hours = now.getHours()
  const minutes = now.getMinutes()
  const time = hours * 60 + minutes

  // Calculate time until market open (9:30 AM = 570 minutes)
  if (time < 570) {
    const diff = 570 - time
    const h = Math.floor(diff / 60)
    const m = diff % 60
    return `${h}h ${m}m until open`
  }

  // Calculate time until market close (4:00 PM = 960 minutes)
  if (time >= 570 && time < 960) {
    const diff = 960 - time
    const h = Math.floor(diff / 60)
    const m = diff % 60
    return `${h}h ${m}m until close`
  }

  return "Opens at 9:30 AM ET"
}

export function MarketStatus() {
  // null until mounted → SSR and first client paint match (no hydration mismatch)
  const [currentTime, setCurrentTime] = useState<Date | null>(null)
  const marketStatus = getMarketStatus(currentTime)
  const timeUntil = getTimeUntilNextSession(currentTime)

  useEffect(() => {
    setCurrentTime(new Date())
    const interval = setInterval(() => {
      setCurrentTime(new Date())
    }, 1000)
    return () => clearInterval(interval)
  }, [])

  const StatusIcon = marketStatus.status === "open"
    ? Sun
    : marketStatus.status === "pre"
      ? Sunrise
      : marketStatus.status === "after"
        ? Moon
        : Clock

  const isOpen = marketStatus.status === "open"

  return (
    <div className="flex items-center justify-between px-5 py-2.5 border-b border-border/50">
      {/* Left: Market Status */}
      <div className="flex items-center gap-5">
        <div className="flex items-center gap-2.5">
          <StatusIcon className={`h-4 w-4 ${marketStatus.color}`} />
          <div className="flex flex-col leading-tight">
            <div className="flex items-center gap-1.5">
              {isOpen && (
                <span className="live-dot inline-flex h-1.5 w-1.5 rounded-full bg-success text-success" />
              )}
              <span className={`text-sm font-semibold tracking-tight ${marketStatus.color}`}>
                {marketStatus.label}
              </span>
            </div>
            <span className="text-[11px] text-muted-foreground tabular-nums">{timeUntil}</span>
          </div>
        </div>

        <div className="h-7 w-px bg-border/50" />

        {/* Current Time */}
        <div className="flex items-center gap-2">
          <Clock className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-sm font-mono text-foreground tabular-nums">
            {currentTime
              ? currentTime.toLocaleTimeString('en-US', {
                  hour: '2-digit',
                  minute: '2-digit',
                  second: '2-digit',
                  hour12: true,
                })
              : '--:--:-- --'}
          </span>
          <span className="eyebrow">ET</span>
        </div>
      </div>

      {/* Center: Major Indices */}
      <div className="hidden lg:flex items-center gap-5">
        {marketIndices.map((index, i) => (
          <div key={index.name} className="flex items-center gap-5">
            {i > 0 && <div className="h-5 w-px bg-border/50" />}
            <div className="flex items-center gap-2">
              <span className="eyebrow">{index.name}</span>
              <span className="text-sm font-mono text-foreground tabular-nums">
                {index.value.toLocaleString('en-US', { minimumFractionDigits: 2 })}
              </span>
              <div className={`flex items-center gap-0.5 text-xs font-mono tabular-nums ${
                index.changePercent >= 0 ? 'text-success' : 'text-destructive'
              }`}>
                {index.changePercent >= 0
                  ? <TrendingUp className="h-3 w-3" />
                  : <TrendingDown className="h-3 w-3" />
                }
                {index.changePercent >= 0 ? '+' : ''}{index.changePercent.toFixed(2)}%
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Right: Connection Status */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-1.5 rounded border border-border/70 px-2 py-1">
          <Activity className="h-3.5 w-3.5 text-success" />
          <span className="eyebrow">Live Feed</span>
        </div>
        <div className="hidden md:flex items-center gap-1.5">
          <Globe className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-[11px] text-muted-foreground">NYSE • NASDAQ</span>
        </div>
      </div>
    </div>
  )
}
