"use client"

import { useState, useEffect } from "react"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  AlertTriangle,
  Lightbulb,
  BarChart3,
  Activity,
  CheckCircle2,
  XCircle,
  Minus,
  Brain
} from "lucide-react"

type CommentaryType = "all" | "decisions" | "opportunities" | "warnings" | "technical"

interface CommentaryItem {
  id: string
  type: CommentaryType
  title: string
  content: string
  time: string
  symbol?: string
  confidence?: number
  indicators?: {
    label: string
    value: string
    status: "bullish" | "bearish" | "neutral"
  }[]
}

const commentary: CommentaryItem[] = [
  {
    id: "1",
    type: "warnings",
    title: "Market Afterhours — Bot Paused",
    content: "Markets are currently afterhours. Pausing analysis until the next regular session opens at 2026-04-16 09:30:00.",
    time: "4:00:42 PM",
  },
  {
    id: "2",
    type: "decisions",
    title: "ML Signal: BUY PLTR",
    content: "Confidence: 65.0%\nKey factors: Multiple technical factors",
    time: "4:00:28 PM",
    symbol: "PLTR",
    confidence: 65,
  },
  {
    id: "3",
    type: "technical",
    title: "Technical Analysis: PLTR",
    content: "",
    time: "4:00:08 PM",
    symbol: "PLTR",
    indicators: [
      { label: "Price above 50 SMA", value: "Uptrend", status: "bullish" },
      { label: "RSI at 63.2", value: "Neutral", status: "neutral" },
      { label: "MACD below signal", value: "Bearish momentum", status: "bearish" },
      { label: "ADX at 41.2", value: "Strong trend", status: "bullish" },
    ],
  },
  {
    id: "4",
    type: "opportunities",
    title: "Data Retrieved: PLTR",
    content: "Got 39957 price candles from Schwab\ncandle_count: 39957.00\ntimeframe: 1d,30m",
    time: "4:00:07 PM",
    symbol: "PLTR",
  },
  {
    id: "5",
    type: "all",
    title: "Analyzing PLTR",
    content: "Fetching price data and calculating technical indicators...",
    time: "4:00:05 PM",
    symbol: "PLTR",
  },
  {
    id: "6",
    type: "decisions",
    title: "ML Signal: BUY TSLA",
    content: "Confidence: 50.9%\nKey factors: Multiple technical factors",
    time: "4:00:03 PM",
    symbol: "TSLA",
    confidence: 50.9,
  },
  {
    id: "7",
    type: "technical",
    title: "Technical Analysis: TSLA",
    content: "",
    time: "3:59:58 PM",
    symbol: "TSLA",
    indicators: [
      { label: "Price above 50 SMA", value: "Uptrend", status: "bullish" },
      { label: "RSI at 53.6", value: "Neutral", status: "neutral" },
      { label: "MACD below signal", value: "Bearish momentum", status: "bearish" },
      { label: "ADX at 28.9", value: "Strong trend", status: "bullish" },
    ],
  },
]

// Chrome-neutral by default. Color is reserved for semantic meaning
// (signal outcome shown via confidence / indicator status, not the tag itself).
const typeConfig = {
  decisions:     { icon: Brain,          label: "SIGNAL",  color: "text-accent",            accent: "bg-accent/50" },
  opportunities: { icon: Lightbulb,      label: "DATA",    color: "text-muted-foreground",  accent: "bg-border" },
  warnings:      { icon: AlertTriangle,  label: "ALERT",   color: "text-chart-4",           accent: "bg-chart-4/60" },
  technical:     { icon: BarChart3,      label: "TECH",    color: "text-muted-foreground",  accent: "bg-border" },
  all:           { icon: Activity,       label: "INFO",    color: "text-muted-foreground",  accent: "bg-border" },
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

export function LiveCommentary() {
  const [activeTab, setActiveTab] = useState<CommentaryType>("all")
  const [isLive, setIsLive] = useState(true)

  const filteredCommentary = activeTab === "all"
    ? commentary
    : commentary.filter(item => item.type === activeTab)

  // Simulate live updates
  useEffect(() => {
    const interval = setInterval(() => {
      setIsLive(prev => !prev)
    }, 2000)
    return () => clearInterval(interval)
  }, [])

  const tabs: { id: CommentaryType; label: string }[] = [
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
            <span className={`relative flex h-1.5 w-1.5 transition-opacity ${isLive ? 'opacity-100' : 'opacity-40'}`}>
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-success" />
            </span>
            <span className="eyebrow">Live</span>
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
          {filteredCommentary.map((item, idx) => {
            const config = typeConfig[item.type] || typeConfig.all
            const Icon = config.icon

            return (
              <article
                key={item.id}
                className={`relative flex gap-3 py-3.5 border-b border-border/40 ${idx === 0 ? "reveal" : ""}`}
              >
                {/* Severity left bar */}
                <span className={`absolute left-0 top-3.5 bottom-3.5 w-0.5 rounded-full ${config.accent}`} />

                <Icon className={`h-3.5 w-3.5 mt-0.5 shrink-0 ${config.color}`} />

                <div className="min-w-0 flex-1 space-y-1.5">
                  {/* Header row */}
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="text-[13px] font-medium text-foreground leading-snug">{item.title}</span>
                    <time className="eyebrow font-mono tabular-nums shrink-0">{item.time}</time>
                  </div>

                  {/* Tags */}
                  {(item.symbol || true) && (
                    <div className="flex items-center gap-1.5">
                      <span className="eyebrow px-1.5 py-px rounded border border-border/60 text-muted-foreground">
                        {config.label}
                      </span>
                      {item.symbol && (
                        <span className="eyebrow px-1.5 py-px rounded border border-accent/25 bg-accent/[0.06] text-accent font-mono tabular-nums">
                          {item.symbol}
                        </span>
                      )}
                    </div>
                  )}

                  {/* Content */}
                  {item.content && (
                    <pre className="text-[11px] text-muted-foreground font-mono tabular-nums whitespace-pre-wrap leading-relaxed">
                      {item.content}
                    </pre>
                  )}

                  {/* Confidence Bar */}
                  {item.confidence && (
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
                  {item.indicators && (
                    <div className="space-y-1 pt-0.5">
                      {item.indicators.map((indicator, i) => (
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
          })}
        </div>
      </ScrollArea>

      {/* Footer Stats */}
      <div className="shrink-0 px-4 py-3 border-t border-border/40 bg-secondary/20">
        <div className="flex items-center justify-between">
          <span className="eyebrow">Today&apos;s signals</span>
          <span className="text-xs font-semibold font-mono tabular-nums text-foreground">24</span>
        </div>
        <div className="flex items-center justify-between mt-1.5">
          <span className="eyebrow">Win rate</span>
          <span className="text-xs font-semibold font-mono tabular-nums text-success">67.3%</span>
        </div>
      </div>
    </div>
  )
}
