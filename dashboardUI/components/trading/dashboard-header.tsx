"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import {
  RefreshCw,
  Play,
  Square,
  Settings,
  Activity,
  BarChart3,
  Newspaper,
  Cpu,
  Wifi,
  Shield,
  Search,
} from "lucide-react"
import { PulseChip } from "@/components/trading/decision-card"

interface DashboardHeaderProps {
  activeTab: string
  onTabChange: (tab: string) => void
}

function openCommandPalette() {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent("open-command-palette"))
  }
}

export function DashboardHeader({ activeTab, onTabChange }: DashboardHeaderProps) {
  const [isRefreshing, setIsRefreshing] = useState(false)

  const handleRefresh = () => {
    setIsRefreshing(true)
    setTimeout(() => setIsRefreshing(false), 1000)
  }

  const tabs = [
    { id: "dashboard", label: "Dashboard", icon: BarChart3 },
    { id: "news", label: "News & Sentiment", icon: Newspaper },
    { id: "backtesting", label: "Backtesting", icon: Activity },
  ]

  return (
    <header className="sticky top-0 z-40 glass-blur border-b border-border">
      <div className="flex h-14 items-center gap-4 px-5">
        {/* Wordmark */}
        <div className="flex items-center gap-2.5 pr-1">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-accent/15 ring-1 ring-inset ring-accent/25">
            <span className="font-mono text-[13px] font-bold leading-none text-accent">A</span>
          </div>
          <div className="leading-none">
            <div className="text-[13px] font-semibold tracking-tight text-foreground">
              Atlas<span className="text-muted-foreground"> Terminal</span>
            </div>
          </div>
        </div>

        <div className="mx-1 h-5 w-px bg-border" />

        {/* Nav */}
        <nav className="hidden items-center gap-0.5 md:flex">
          {tabs.map((tab) => {
            const Icon = tab.icon
            const isActive = activeTab === tab.id
            return (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                className={`flex items-center gap-2 rounded-md px-3 py-1.5 text-[13px] font-medium transition-colors ${
                  isActive
                    ? "bg-secondary text-foreground"
                    : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                <span className="hidden lg:inline">{tab.label}</span>
              </button>
            )
          })}
        </nav>

        {/* Command palette trigger */}
        <button
          onClick={openCommandPalette}
          className="ml-auto hidden items-center gap-2 rounded-md border border-border bg-secondary/40 px-2.5 py-1.5 text-[13px] text-muted-foreground transition-colors hover:bg-secondary/70 hover:text-foreground sm:flex"
        >
          <Search className="h-3.5 w-3.5" />
          <span className="hidden lg:inline">Search & commands</span>
          <kbd className="ml-1 hidden rounded border border-border bg-background px-1.5 font-mono text-[10px] leading-4 text-muted-foreground lg:inline">
            ⌘K
          </kbd>
        </button>

        {/* Toggles */}
        <div className="hidden items-center gap-3 xl:flex">
          <ToggleControl id="simulation" label="Sim" defaultChecked />
          <ToggleControl id="confirmations" label="Confirms" icon={Shield} defaultChecked />
          <ToggleControl id="ml-prediction" label="ML" icon={Cpu} defaultChecked />
        </div>

        <div className="mx-1 hidden h-5 w-px bg-border xl:block" />

        {/* Actions */}
        <div className="flex items-center gap-1.5">
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 text-muted-foreground hover:text-foreground"
            onClick={handleRefresh}
          >
            <RefreshCw className={`h-4 w-4 ${isRefreshing ? "animate-spin" : ""}`} />
          </Button>
          <Button
            size="sm"
            className="h-8 gap-1.5 bg-success text-success-foreground hover:bg-success/90"
          >
            <Play className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Start</span>
          </Button>
          <Button variant="outline" size="sm" className="h-8 gap-1.5 border-border text-muted-foreground hover:text-foreground">
            <Square className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Stop</span>
          </Button>
          <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-foreground">
            <Settings className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Status strip */}
      <div className="flex items-center gap-4 border-t border-border/60 px-5 py-1.5">
        <StatusItem label="Bot active" live />
        <StatusBadge icon={Shield} label="Schwab" tone="success" />
        <StatusBadge icon={Wifi} label="WebSocket" tone="success" />
        <StatusBadge icon={Cpu} label="ML model" tone="neutral" />
        <div className="mx-1 h-4 w-px bg-border/60" />
        <PulseChip />
        <div className="ml-auto flex items-center gap-1.5">
          <span className="eyebrow">Session</span>
          <span className="font-mono text-xs tabular-nums text-foreground">4h 23m</span>
        </div>
      </div>
    </header>
  )
}

function ToggleControl({
  id,
  label,
  icon: Icon,
  defaultChecked,
}: {
  id: string
  label: string
  icon?: React.ComponentType<{ className?: string }>
  defaultChecked?: boolean
}) {
  return (
    <div className="flex items-center gap-1.5">
      {Icon && <Icon className="h-3.5 w-3.5 text-muted-foreground" />}
      <Label htmlFor={id} className="cursor-pointer text-xs font-medium text-muted-foreground">
        {label}
      </Label>
      <Switch id={id} defaultChecked={defaultChecked} className="scale-[0.7]" />
    </div>
  )
}

function StatusItem({ label, live }: { label: string; live?: boolean }) {
  return (
    <div className="flex items-center gap-1.5">
      <span
        className={`live-dot inline-flex h-1.5 w-1.5 rounded-full ${
          live ? "bg-success text-success" : "bg-muted-foreground text-muted-foreground"
        }`}
      />
      <span className="eyebrow !text-muted-foreground">{label}</span>
    </div>
  )
}

function StatusBadge({
  icon: Icon,
  label,
  tone,
}: {
  icon: React.ComponentType<{ className?: string }>
  label: string
  tone: "success" | "neutral" | "error"
}) {
  const toneClass =
    tone === "success"
      ? "text-success"
      : tone === "error"
        ? "text-destructive"
        : "text-muted-foreground"
  return (
    <div className="flex items-center gap-1.5">
      <Icon className={`h-3 w-3 ${toneClass}`} />
      <span className="eyebrow !text-muted-foreground">{label}</span>
    </div>
  )
}
