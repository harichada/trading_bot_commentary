"use client"

import { useState } from "react"
import { TickerBar } from "@/components/trading/ticker-bar"
import { DashboardHeader } from "@/components/trading/dashboard-header"
import { MarketStatus } from "@/components/trading/market-status"
import { AccountOverview } from "@/components/trading/account-overview"
import { PriceChart } from "@/components/trading/price-chart"
import { PortfolioAllocation } from "@/components/trading/portfolio-allocation"
import { PerformanceMetrics } from "@/components/trading/performance-metrics"
import { LivePositionsSection } from "@/components/trading/positions-table"
import { NewsSentiment } from "@/components/trading/news-sentiment"
import { WatchlistHeatmap } from "@/components/trading/watchlist-heatmap"
import { QuickTrade } from "@/components/trading/quick-trade"
import { RecentTrades } from "@/components/trading/recent-trades"
import { LiveCommentary } from "@/components/trading/live-commentary"
import { CommandPalette } from "@/components/trading/command-palette"
import { DecisionCardProvider } from "@/components/trading/decision-card"

export default function TradingDashboard() {
  const [activeTab, setActiveTab] = useState("dashboard")

  return (
    <DecisionCardProvider>
    <div className="min-h-screen bg-background flex flex-col">
      <CommandPalette onNavigate={setActiveTab} />
      {/* Atmosphere — one faint accent wash + a quiet terminal grid */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-40 left-1/3 w-[720px] h-[480px] bg-accent/[0.03] rounded-full blur-[140px]" />
        <div
          className="absolute inset-0 opacity-[0.45]"
          style={{
            backgroundImage:
              "linear-gradient(to right, oklch(1 0 0 / 0.015) 1px, transparent 1px), linear-gradient(to bottom, oklch(1 0 0 / 0.015) 1px, transparent 1px)",
            backgroundSize: "48px 48px",
            maskImage: "radial-gradient(ellipse 80% 55% at 50% 0%, black, transparent 78%)",
            WebkitMaskImage: "radial-gradient(ellipse 80% 55% at 50% 0%, black, transparent 78%)",
          }}
        />
      </div>
      
      {/* Ticker Bar */}
      <TickerBar />
      
      {/* Market Status Bar */}
      <MarketStatus />
      
      {/* Dashboard Header */}
      <DashboardHeader activeTab={activeTab} onTabChange={setActiveTab} />
      
      {/* Main Content */}
      <div className="flex-1 flex relative">
        {/* Left Sidebar - Live Commentary */}
        <aside className="w-80 xl:w-[360px] border-r border-border/30 hidden lg:flex flex-col shrink-0 bg-sidebar/50">
          <LiveCommentary />
        </aside>
        
        {/* Main Dashboard Area */}
        <main className="flex-1 overflow-auto">
          <div className="p-6 xl:p-8 space-y-6 xl:space-y-8 max-w-[1800px]">
            
            {/* Top Row: Account Overview */}
            <AccountOverview />
            
            {/* Second Row: Chart + Portfolio + Quick Trade */}
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
              {/* Price Chart - Takes up most space */}
              <div className="xl:col-span-7">
                <PriceChart />
              </div>
              
              {/* Portfolio Allocation */}
              <div className="xl:col-span-3">
                <PortfolioAllocation />
              </div>
              
              {/* Quick Trade Panel */}
              <div className="xl:col-span-2">
                <QuickTrade />
              </div>
            </div>
            
            {/* Third Row: Performance + Watchlist */}
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
              <div className="xl:col-span-8">
                <PerformanceMetrics />
              </div>
              <div className="xl:col-span-4">
                <WatchlistHeatmap />
              </div>
            </div>
            
            {/* Fourth Row: Positions Tables */}
            <LivePositionsSection />
            
            {/* Fifth Row: News Sentiment */}
            <NewsSentiment />
            
            {/* Sixth Row: Recent Trades */}
            <RecentTrades />
          </div>
        </main>
        
        {/* Right Sidebar - Quick Trade (alternative placement for smaller screens) */}
      </div>
    </div>
    </DecisionCardProvider>
  )
}
