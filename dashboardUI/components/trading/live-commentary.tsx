"use client"

import { useState, useMemo } from "react"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  AlertTriangle,
  Lightbulb,
  BarChart3,
  Activity,
  CheckCircle2,
  XCircle,
  Minus,
  Brain,
  Wifi,
  WifiOff,
  ChevronRight
} from "lucide-react"
import { useWsCommentary } from "@/hooks/use-bot-data"
import { useDecisionCard, isDecisionCommentary } from "@/components/trading/decision-card"
import type { Commentary } from "@/lib/api"

type CommentaryFilter = "all" | "decisions" | "opportunities" | "warnings" | "technical"

const typeConfig: Record<string, { icon: typeof Brain; label: string; color: string; accent: string }> = {
  DECISION:        { icon: Brain,          label: "SIGNAL",  color: "text-accent",            accent: "bg-accent/50" },
  SIGNAL:          { icon: Brain,          label: "SIGNAL",  color: "text-accent",            accent: "bg-accent/50" },
  DATA:            { icon: Lightbulb,      label: "DATA",    color: "text-muted-foreground",  accent: "bg-border" },
  OPPORTUNITY:     { icon: Lightbulb,      label: "DATA",    color: "text-muted-foreground",  accent: "bg-border" },
  WARNING:         { icon: AlertTriangle,  label: "ALERT",   color: "text-chart-4",           accent: "bg-chart-4/60" },
  RISK_ALERT:      { icon: AlertTriangle,  label: "ALERT",   color: "text-chart-4",           accent: "bg-chart-4/60" },
  TECHNICAL:       { icon: BarChart3,      label: "TECH",    color: "text-muted-foreground",  accent: "bg-border" },
  MARKET_ANALYSIS: { icon: BarChart3,      label: "TECH",    color: "text-muted-foreground",  accent: "bg-border" },
  INFO:            { icon: Activity,       label: "INFO",    color: "text-muted-foreground",  accent: "bg-border" },
}

const defaultConfig = { icon: Activity, label: "INFO", color: "text-muted-foreground", accent: "bg-border" }

function getConfig(type: string) {
  return typeConfig[type.toUpperCase()] ?? defaultConfig
}

function matchesFilter(type: string, filter: CommentaryFilter): boolean {
  if (filter === "all") return true
  const upper = type.toUpperCase()
  switch (filter) {
    case "decisions":
      return upper === "DECISION" || upper === "SIGNAL"
    case "opportunities":
      return upper === "DATA" || upper === "OPPORTUNITY"
    case "warnings":
      return upper === "WARNING" || upper === "RISK_ALERT"
    case "technical":
      return upper === "TECHNICAL" || upper === "MARKET_ANALYSIS"
    default:
      return true
  }
}

function formatTime(isoString: string): string {
  try {
    const date = new Date(isoString)
    return date.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      second: "2-digit",
      hour12: true,
    })
  } catch {
    return "—"
  }
}

function getStatusIcon(status: "bullish" | "bearish" | "neutral") {
  switch (status) {
    case "bullish":
      return <CheckCircle2 className="h-3.5 w-3.5 text-success" />
    case "bearish":
      return <XCircle className="h-3.5 w-3.5 text-destructive" />
    default:
      return <Minus className="h-3.5 w-3.5 text-muted-foreground" />
  }
}

interface IndicatorDisplay {
  label: string
  value: string
  status: "bullish" | "bearish" | "neutral"
}

function parseIndicators(data: Record<string, unknown> | null | undefined): IndicatorDisplay[] {
  if (!data) return []
  const indicators = data.indicators as IndicatorDisplay[] | undefined
  if (Array.isArray(indicators)) {
    return indicators.filter(
      (i) => i && typeof i.label === "string" && typeof i.value === "string"
    )
  }
  return []
}

export function LiveCommentary() {
  const [activeTab, setActiveTab] = useState<CommentaryFilter>("all")
  const { items, connected } = useWsCommentary()

  const filteredItems = useMemo(
    () => items.filter((item) => matchesFilter(item.type, activeTab)),
    [items, activeTab]
  )

  const tabs: { id: CommentaryFilter; label: string }[] = [
    { id: "all", label: "All" },
    { id: "decisions", label: "Signals" },
    { id: "opportunities", label: "Data" },
    { id: "warnings", label: "Alerts" },
    { id: "technical", label: "Tech" },
  ]

  return (
    <div className="flex flex-col h-full bg-sidebar">
      {/* Header */}
      <div className="shrink-0 px-4 py-4 border-b border-border/40">
        <div className="flex items-center justify-between gap-3 mb-3.5">
          <div className="min-w-0">
            <h2 className="text-sm font-semibold tracking-tight text-foreground">Live Commentary</h2>
            <p className="eyebrow mt-0.5">Bot decision feed</p>
          </div>
          <div className="flex items-center gap-1.5 shrink-0">
            {connected ? (
              <>
                <span className="live-dot relative flex h-1.5 w-1.5">
                  <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-success" />
                </span>
                <span className="eyebrow">Live</span>
              </>
            ) : (
              <>
                <WifiOff className="h-3.5 w-3.5 text-muted-foreground" />
                <span className="eyebrow text-muted-foreground">Connecting…</span>
              </>
            )}
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 p-0.5 rounded-lg bg-secondary/30 border border-border/40">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 px-2 py-1.5 rounded-md text-[11px] font-medium transition-colors ${
                activeTab === tab.id
                  ? "bg-accent/12 text-accent"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <ScrollArea className="flex-1">
        <div className="px-4">
          {filteredItems.length === 0 ? (
            <div className="py-8 text-center text-sm text-muted-foreground">
              {connected ? "No commentary yet" : "Waiting for connection…"}
            </div>
          ) : (
            filteredItems.map((item, idx) => (
              <CommentaryItem key={`${item.timestamp}-${idx}`} item={item} isFirst={idx === 0} />
            ))
          )}
        </div>
      </ScrollArea>

      {/* Footer Stats */}
      <div className="shrink-0 px-4 py-3 border-t border-border/40 bg-secondary/20">
        <div className="flex items-center justify-between">
          <span className="eyebrow">Messages</span>
          <span className="text-xs font-semibold font-mono tabular-nums text-foreground">
            {items.length}
          </span>
        </div>
        <div className="flex items-center justify-between mt-1.5">
          <span className="eyebrow">Status</span>
          <span className={`text-xs font-semibold font-mono tabular-nums ${connected ? "text-success" : "text-muted-foreground"}`}>
            {connected ? "Connected" : "Disconnected"}
          </span>
        </div>
      </div>
    </div>
  )
}

function CommentaryItem({ item, isFirst }: { item: Commentary; isFirst: boolean }) {
  const config = getConfig(item.type)
  const Icon = config.icon
  const indicators = parseIndicators(item.data)
  const { openFromCommentary } = useDecisionCard()
  const isClickable = isDecisionCommentary(item.type)

  const handleClick = () => {
    if (isClickable) {
      openFromCommentary(item)
    }
  }

  return (
    <article
      onClick={handleClick}
      className={`relative flex gap-3 py-3.5 border-b border-border/40 ${isFirst ? "reveal" : ""} ${
        isClickable ? "cursor-pointer hover:bg-accent/[0.04] transition-colors group" : ""
      }`}
    >
      {/* Severity left bar */}
      <span className={`absolute left-0 top-3.5 bottom-3.5 w-0.5 rounded-full ${config.accent}`} />

      <Icon className={`h-3.5 w-3.5 mt-0.5 shrink-0 ${config.color}`} />

      <div className="min-w-0 flex-1 space-y-1.5">
        {/* Header row */}
        <div className="flex items-baseline justify-between gap-2">
          <span className="text-[13px] font-medium text-foreground leading-snug">{item.title}</span>
          <div className="flex items-center gap-1.5 shrink-0">
            <time className="eyebrow font-mono tabular-nums">{formatTime(item.timestamp)}</time>
            {isClickable && (
              <ChevronRight className="h-3 w-3 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
            )}
          </div>
        </div>

        {/* Tags */}
        <div className="flex items-center gap-1.5">
          <span className="eyebrow px-1.5 py-px rounded border border-border/60 text-muted-foreground">
            {config.label}
          </span>
          {item.symbol && (
            <span className="eyebrow px-1.5 py-px rounded border border-accent/25 bg-accent/[0.06] text-accent font-mono tabular-nums">
              {item.symbol}
            </span>
          )}
          {isClickable && (
            <span className="eyebrow px-1.5 py-px rounded border border-accent/40 bg-accent/10 text-accent">
              TAP FOR DETAIL
            </span>
          )}
        </div>

        {/* Content */}
        {item.message && (
          <pre className="text-[11px] text-muted-foreground font-mono tabular-nums whitespace-pre-wrap leading-relaxed">
            {item.message}
          </pre>
        )}

        {/* Confidence Bar */}
        {item.confidence != null && item.confidence > 0 && (
          <div className="flex items-center gap-2.5 pt-0.5">
            <div className="flex-1 h-1 bg-secondary/60 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-700 ease-out ${
                  item.confidence >= 60
                    ? 'bg-success'
                    : item.confidence >= 40
                      ? 'bg-chart-4'
                      : 'bg-destructive'
                }`}
                style={{ width: `${item.confidence}%` }}
              />
            </div>
            <span className={`text-[11px] font-semibold font-mono tabular-nums ${
              item.confidence >= 60 ? 'text-success' : item.confidence >= 40 ? 'text-chart-4' : 'text-destructive'
            }`}>
              {item.confidence.toFixed(1)}%
            </span>
          </div>
        )}

        {/* Technical Indicators */}
        {indicators.length > 0 && (
          <div className="space-y-1 pt-0.5">
            {indicators.map((indicator, i) => (
              <div key={i} className="flex items-center gap-2 text-[11px]">
                {getStatusIcon(indicator.status)}
                <span className="text-muted-foreground truncate">{indicator.label}</span>
                <span className="text-foreground font-medium ml-auto shrink-0">{indicator.value}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </article>
  )
}
