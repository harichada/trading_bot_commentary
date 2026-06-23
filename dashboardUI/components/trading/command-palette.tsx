"use client"

import { useEffect, useState } from "react"
import { Command } from "cmdk"
import {
  LayoutDashboard,
  Newspaper,
  Activity,
  Play,
  Square,
  RefreshCw,
  Settings,
  Search,
  TrendingUp,
  Wallet,
  Bot,
  CircleSlash,
} from "lucide-react"

interface CommandPaletteProps {
  onNavigate?: (tab: string) => void
}

type Item = {
  group: string
  icon: React.ComponentType<{ className?: string }>
  label: string
  hint?: string
  keywords?: string
  run: () => void
  tone?: "default" | "success" | "destructive"
}

export function CommandPalette({ onNavigate }: CommandPaletteProps) {
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault()
        setOpen((o) => !o)
      }
      if (e.key === "Escape") setOpen(false)
    }
    const onOpen = () => setOpen(true)
    window.addEventListener("keydown", onKey)
    window.addEventListener("open-command-palette", onOpen as EventListener)
    return () => {
      window.removeEventListener("keydown", onKey)
      window.removeEventListener("open-command-palette", onOpen as EventListener)
    }
  }, [])

  const close = () => setOpen(false)
  const go = (tab: string) => {
    onNavigate?.(tab)
    close()
  }

  const items: Item[] = [
    { group: "Navigate", icon: LayoutDashboard, label: "Dashboard", hint: "G D", run: () => go("dashboard") },
    { group: "Navigate", icon: Newspaper, label: "News & Sentiment", hint: "G N", run: () => go("news") },
    { group: "Navigate", icon: Activity, label: "Backtesting", hint: "G B", run: () => go("backtesting") },
    { group: "Bot control", icon: Play, label: "Start bot", keywords: "run resume engine", tone: "success", run: close },
    { group: "Bot control", icon: Square, label: "Stop bot", keywords: "halt kill pause", tone: "destructive", run: close },
    { group: "Bot control", icon: RefreshCw, label: "Refresh positions", keywords: "reload sync", run: close },
    { group: "Bot control", icon: CircleSlash, label: "Flatten all positions", keywords: "close exit emergency", tone: "destructive", run: close },
    { group: "View", icon: Wallet, label: "Jump to Positions", keywords: "holdings", run: close },
    { group: "View", icon: TrendingUp, label: "Jump to Performance", keywords: "pnl equity", run: close },
    { group: "View", icon: Bot, label: "Jump to Live Commentary", keywords: "log feed decisions", run: close },
    { group: "View", icon: Settings, label: "Open Settings", keywords: "config preferences", run: close },
  ]

  const groups = Array.from(new Set(items.map((i) => i.group)))

  if (!open) return null

  return (
    <div className="fixed inset-0 z-[100] flex items-start justify-center px-4 pt-[12vh]">
      {/* Scrim */}
      <button
        aria-label="Close command palette"
        onClick={close}
        className="absolute inset-0 bg-background/70 backdrop-blur-sm"
      />

      <Command
        loop
        className="reveal relative w-full max-w-xl overflow-hidden rounded-xl border border-border bg-popover shadow-2xl"
        style={{ animationDuration: "180ms" }}
      >
        <div className="flex items-center gap-2.5 border-b border-border px-4">
          <Search className="h-4 w-4 shrink-0 text-muted-foreground" />
          <Command.Input
            autoFocus
            placeholder="Search symbols, run commands…"
            className="h-12 w-full bg-transparent text-sm text-foreground outline-none placeholder:text-muted-foreground"
          />
          <kbd className="hidden rounded border border-border bg-secondary px-1.5 font-mono text-[10px] leading-4 text-muted-foreground sm:inline">
            ESC
          </kbd>
        </div>

        <Command.List className="max-h-[52vh] overflow-y-auto p-2">
          <Command.Empty className="py-8 text-center text-sm text-muted-foreground">
            No results found.
          </Command.Empty>

          {groups.map((group) => (
            <Command.Group
              key={group}
              heading={<span className="eyebrow px-2">{group}</span>}
              className="mb-1 [&_[cmdk-group-heading]]:px-1 [&_[cmdk-group-heading]]:py-1.5"
            >
              {items
                .filter((i) => i.group === group)
                .map((item) => {
                  const Icon = item.icon
                  const toneClass =
                    item.tone === "success"
                      ? "text-success"
                      : item.tone === "destructive"
                        ? "text-destructive"
                        : "text-muted-foreground"
                  return (
                    <Command.Item
                      key={item.label}
                      value={`${item.label} ${item.keywords ?? ""}`}
                      onSelect={item.run}
                      className="flex cursor-pointer items-center gap-2.5 rounded-md px-2.5 py-2 text-sm text-foreground data-[selected=true]:bg-secondary"
                    >
                      <Icon className={`h-4 w-4 ${toneClass}`} />
                      <span className="flex-1">{item.label}</span>
                      {item.hint && (
                        <kbd className="font-mono text-[10px] tabular-nums text-muted-foreground">{item.hint}</kbd>
                      )}
                    </Command.Item>
                  )
                })}
            </Command.Group>
          ))}
        </Command.List>

        <div className="flex items-center justify-between border-t border-border px-3 py-2">
          <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
            <span className="flex items-center gap-1">
              <kbd className="rounded border border-border bg-secondary px-1 font-mono text-[10px]">↑↓</kbd> navigate
            </span>
            <span className="flex items-center gap-1">
              <kbd className="rounded border border-border bg-secondary px-1 font-mono text-[10px]">↵</kbd> select
            </span>
          </div>
          <span className="eyebrow">Atlas Terminal</span>
        </div>
      </Command>
    </div>
  )
}
