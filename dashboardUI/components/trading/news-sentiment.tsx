"use client"

import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { RefreshCw, TrendingUp, ExternalLink, Zap, ArrowUpRight, Sparkles } from "lucide-react"
import { Area, AreaChart, ResponsiveContainer, XAxis, YAxis, Tooltip } from "recharts"

interface SentimentData {
  symbol: string
  score: number
}

const watchlistSentiment: SentimentData[] = [
  { symbol: "TSLA", score: 24 },
  { symbol: "NVDA", score: 22 },
  { symbol: "AMD", score: 28 },
  { symbol: "AAPL", score: 22 },
  { symbol: "SPY", score: 16 },
  { symbol: "MARA", score: 23 },
]

const velocityData = [
  { time: "9:30", value: 12, articles: 8 },
  { time: "10:00", value: 28, articles: 15 },
  { time: "10:30", value: 45, articles: 22 },
  { time: "11:00", value: 38, articles: 18 },
  { time: "11:30", value: 52, articles: 28 },
  { time: "12:00", value: 48, articles: 24 },
  { time: "12:30", value: 35, articles: 16 },
  { time: "13:00", value: 62, articles: 32 },
  { time: "13:30", value: 58, articles: 30 },
  { time: "14:00", value: 50, articles: 26 },
]

const newsHeadlines = [
  {
    title: "Uber Push May Mark Turning Point in Strategy",
    source: "Yahoo Finance",
    time: "Just now",
    sentiment: "neutral",
  },
  {
    title: "Tesla Regains 50-Day Line: Here's When The Stock Could Become Actionable",
    source: "Investor's Business Daily",
    time: "5 min ago",
    sentiment: "bullish",
  },
  {
    title: "Stocks making big moves: Globalstar, Tesla, HighPeak Energy, Shopify",
    source: "CNBC",
    time: "12 min ago",
    sentiment: "bullish",
  },
]

export function NewsSentiment() {
  const [isRefreshing, setIsRefreshing] = useState(false)
  
  const handleRefresh = () => {
    setIsRefreshing(true)
    setTimeout(() => setIsRefreshing(false), 1000)
  }

  const overallSentiment = 22
  const sentimentLabel = overallSentiment > 15 ? "BULLISH" : overallSentiment < -15 ? "BEARISH" : "NEUTRAL"

  const isPositive = overallSentiment >= 0

  return (
    <div className="rounded-xl glass overflow-hidden reveal">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-border/20">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold tracking-tight text-foreground">News Sentiment</h3>
          <Badge variant="outline" className="eyebrow border-border/40 bg-secondary/30 text-muted-foreground gap-1">
            <Sparkles className="h-3 w-3" />
            AI Analyzed
          </Badge>
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="h-8 gap-2 text-xs hover:bg-secondary/50"
          onClick={handleRefresh}
        >
          <RefreshCw className={`h-3.5 w-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>
      
      <div className="p-5 space-y-6">
        {/* Main Sentiment + Chart Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Sentiment Gauge - 2 cols */}
          <div className="lg:col-span-2 space-y-4">
            <div className="p-5 rounded-xl glass-subtle">
              <div className="text-center space-y-3">
                <p className="eyebrow">Market Sentiment</p>
                <Badge
                  className={`gap-1.5 text-[10px] font-semibold px-2.5 py-0.5 ${
                    isPositive
                      ? "bg-success/[0.10] text-success border-success/20"
                      : "bg-destructive/[0.10] text-destructive border-destructive/20"
                  }`}
                >
                  <TrendingUp className="h-3 w-3" />
                  {sentimentLabel}
                </Badge>
                <div className="pt-1">
                  <span className={`text-4xl font-semibold font-mono tabular-nums ${isPositive ? "text-success" : "text-destructive"}`}>
                    {isPositive ? "+" : ""}{overallSentiment}
                  </span>
                </div>

                {/* Sentiment Bar */}
                <div className="pt-3 space-y-2">
                  <div className="flex justify-between eyebrow">
                    <span>Bearish</span>
                    <span>Bullish</span>
                  </div>
                  <div className="h-1.5 bg-secondary/40 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ease-out ${isPositive ? "bg-success/70" : "bg-destructive/70"}`}
                      style={{ width: `${50 + overallSentiment}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Watchlist Sentiment Pills */}
            <div className="space-y-3">
              <p className="eyebrow">Watchlist Sentiment</p>
              <div className="flex flex-wrap gap-2">
                {watchlistSentiment.map((item) => {
                  const up = item.score >= 0
                  return (
                    <div
                      key={item.symbol}
                      className="lift group flex items-center gap-2 px-3 py-1.5 rounded-lg glass-subtle cursor-pointer"
                    >
                      <span className="text-xs font-semibold text-foreground">{item.symbol}</span>
                      <span className={`flex items-center gap-0.5 text-xs font-mono tabular-nums ${up ? "text-success" : "text-destructive"}`}>
                        {up ? <ArrowUpRight className="h-3 w-3" /> : null}
                        {up ? "+" : ""}{item.score}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>

          {/* News Velocity Chart - 3 cols */}
          <div className="lg:col-span-3 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Zap className="h-3.5 w-3.5 text-muted-foreground" />
                <span className="eyebrow">News Velocity</span>
              </div>
              <div className="flex items-center gap-5">
                <div className="text-right">
                  <p className="eyebrow">Peak Today</p>
                  <p className="text-base font-semibold font-mono tabular-nums text-foreground">62</p>
                </div>
                <div className="text-right">
                  <p className="eyebrow">Articles</p>
                  <p className="text-base font-semibold font-mono tabular-nums text-foreground">219</p>
                </div>
              </div>
            </div>

            <div className="h-[180px] w-full rounded-xl glass-subtle p-3">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={velocityData}>
                  <defs>
                    <linearGradient id="velocityGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="oklch(0.74 0.105 205)" stopOpacity={0.14} />
                      <stop offset="95%" stopColor="oklch(0.74 0.105 205)" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis
                    dataKey="time"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: "oklch(0.5 0 0)", fontSize: 10 }}
                    dy={10}
                  />
                  <YAxis
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: "oklch(0.5 0 0)", fontSize: 10 }}
                    dx={-10}
                    width={30}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "oklch(0.12 0.015 270)",
                      border: "1px solid oklch(1 0 0 / 0.08)",
                      borderRadius: "10px",
                      fontSize: "12px",
                      boxShadow: "0 10px 40px -10px oklch(0 0 0 / 0.5)",
                    }}
                    labelStyle={{ color: "oklch(0.98 0 0)", fontWeight: 600 }}
                    itemStyle={{ color: "oklch(0.74 0.105 205)" }}
                  />
                  <Area
                    type="monotone"
                    dataKey="value"
                    stroke="oklch(0.74 0.105 205)"
                    strokeWidth={1.5}
                    fill="url(#velocityGradient)"
                    dot={false}
                    activeDot={{
                      r: 4,
                      fill: "oklch(0.74 0.105 205)",
                      stroke: "oklch(0.12 0.015 270)",
                      strokeWidth: 2
                    }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* News Headlines */}
        <div className="space-y-3">
          <p className="eyebrow">Recent Headlines</p>
          <div className="grid gap-2">
            {newsHeadlines.map((news, idx) => (
              <div
                key={idx}
                className="lift group flex items-start justify-between gap-4 p-3.5 rounded-xl glass-subtle cursor-pointer"
              >
                <div className="flex items-start gap-3 flex-1 min-w-0">
                  <div className={`mt-1 w-2 h-2 rounded-full shrink-0 ${
                    news.sentiment === "bullish" 
                      ? "bg-success" 
                      : news.sentiment === "bearish" 
                        ? "bg-destructive" 
                        : "bg-muted-foreground"
                  }`} />
                  <div className="space-y-1 min-w-0">
                    <p className="text-sm text-foreground font-medium leading-snug line-clamp-1 group-hover:text-accent transition-colors">
                      {news.title}
                    </p>
                    <p className="text-[11px] text-muted-foreground">
                      {news.source} <span className="opacity-50">•</span> {news.time}
                    </p>
                  </div>
                </div>
                <ExternalLink className="h-4 w-4 text-muted-foreground shrink-0 opacity-0 group-hover:opacity-100 transition-all mt-1" />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
