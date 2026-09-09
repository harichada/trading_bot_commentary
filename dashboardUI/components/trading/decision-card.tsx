"use client"

import { createContext, useContext, useState, useCallback, useMemo } from "react"
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import {
  Drawer,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
} from "@/components/ui/drawer"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Clock,
  AlertTriangle,
  ShieldAlert,
  Target,
  Wifi,
  WifiOff,
  Activity,
  Brain,
  Info,
  XCircle,
} from "lucide-react"
import { useIsMobile } from "@/hooks/use-mobile"
import { useWsCommentary, useBotStatus, usePositions } from "@/hooks/use-bot-data"
import type { Commentary, Position } from "@/lib/api"

export type Stance = "long" | "short" | "do_nothing"

export interface DecisionData {
  symbol: string
  timestamp: string
  stance: Stance
  confidence?: number | null
  keyFacts?: Array<{ label: string; value: string; asOf?: string }>
  invalidation?: string
  risks?: string[]
  source?: string
  wakeReason?: string
}

interface DecisionCardContextValue {
  isOpen: boolean
  decision: DecisionData | null
  openDecision: (decision: DecisionData) => void
  openFromCommentary: (commentary: Commentary) => void
  openFromPosition: (position: Position) => void
  close: () => void
}

const DecisionCardContext = createContext<DecisionCardContextValue | null>(null)

export function useDecisionCard() {
  const ctx = useContext(DecisionCardContext)
  if (!ctx) {
    throw new Error("useDecisionCard must be used within DecisionCardProvider")
  }
  return ctx
}

function normalizeStance(raw: unknown): Stance {
  if (typeof raw !== "string") return "do_nothing"
  const lower = raw.toLowerCase().trim()
  if (
    lower === "long" ||
    lower === "buy" ||
    lower === "bullish" ||
    lower === "enter_long"
  ) {
    return "long"
  }
  if (
    lower === "short" ||
    lower === "sell" ||
    lower === "bearish" ||
    lower === "enter_short"
  ) {
    return "short"
  }
  if (
    lower === "hold" ||
    lower === "do_nothing" ||
    lower === "stand_down" ||
    lower === "standdown" ||
    lower === "neutral" ||
    lower === "wait" ||
    lower === "no_action" ||
    lower.includes("stand down") ||
    lower.includes("do nothing") ||
    lower.includes("no action")
  ) {
    return "do_nothing"
  }
  return "do_nothing"
}

function parseKeyFacts(data: Record<string, unknown> | null | undefined): Array<{ label: string; value: string; asOf?: string }> | undefined {
  if (!data) return undefined
  const facts = data.key_facts ?? data.keyFacts ?? data.facts
  if (!Array.isArray(facts)) return undefined
  const parsed = facts.filter(
    (f): f is { label: string; value: string; asOf?: string } =>
      f && typeof f === "object" && typeof f.label === "string" && typeof f.value === "string"
  )
  return parsed.length > 0 ? parsed : undefined
}

function parseInvalidation(data: Record<string, unknown> | null | undefined): string | undefined {
  if (!data) return undefined
  const inv = data.invalidation ?? data.invalidates_if ?? data.stop_condition
  return typeof inv === "string" && inv.trim() ? inv.trim() : undefined
}

function parseRisks(data: Record<string, unknown> | null | undefined): string[] | undefined {
  if (!data) return undefined
  const risks = data.risks ?? data.risk_factors ?? data.warnings
  if (!Array.isArray(risks)) return undefined
  const parsed = risks.filter((r): r is string => typeof r === "string" && r.trim().length > 0)
  return parsed.length > 0 ? parsed : undefined
}

function parseSource(data: Record<string, unknown> | null | undefined): string | undefined {
  if (!data) return undefined
  const src = data.source ?? data.model ?? data.strategy
  return typeof src === "string" && src.trim() ? src.trim() : undefined
}

function parseWakeReason(data: Record<string, unknown> | null | undefined): string | undefined {
  if (!data) return undefined
  const wake = data.wake_reason ?? data.wakeReason ?? data.trigger ?? data.catalyst
  return typeof wake === "string" && wake.trim() ? wake.trim() : undefined
}

function commentaryToDecision(commentary: Commentary): DecisionData {
  const data = commentary.data ?? {}
  const stanceRaw = data.stance ?? data.direction ?? data.action ?? data.signal ?? commentary.type
  return {
    symbol: commentary.symbol ?? "—",
    timestamp: commentary.timestamp,
    stance: normalizeStance(stanceRaw),
    confidence: commentary.confidence,
    keyFacts: parseKeyFacts(data),
    invalidation: parseInvalidation(data),
    risks: parseRisks(data),
    source: parseSource(data),
    wakeReason: parseWakeReason(data),
  }
}

function positionToDecision(position: Position): DecisionData {
  const stance: Stance = position.side === "short" ? "short" : "long"
  const keyFacts: Array<{ label: string; value: string }> = [
    { label: "Entry Price", value: `$${position.entry_price.toFixed(2)}` },
    { label: "Current Price", value: `$${position.current_price.toFixed(2)}` },
    { label: "Quantity", value: position.quantity.toLocaleString() },
  ]
  if (position.stop_loss > 0) {
    keyFacts.push({ label: "Stop Loss", value: `$${position.stop_loss.toFixed(2)}` })
  }
  if (position.take_profit > 0) {
    keyFacts.push({ label: "Take Profit", value: `$${position.take_profit.toFixed(2)}` })
  }
  return {
    symbol: position.symbol,
    timestamp: position.updated_at ?? new Date().toISOString(),
    stance,
    keyFacts,
    source: position.strategy ?? undefined,
  }
}

export function DecisionCardProvider({ children }: { children: React.ReactNode }) {
  const [isOpen, setIsOpen] = useState(false)
  const [decision, setDecision] = useState<DecisionData | null>(null)

  const openDecision = useCallback((d: DecisionData) => {
    setDecision(d)
    setIsOpen(true)
  }, [])

  const openFromCommentary = useCallback((commentary: Commentary) => {
    openDecision(commentaryToDecision(commentary))
  }, [openDecision])

  const openFromPosition = useCallback((position: Position) => {
    openDecision(positionToDecision(position))
  }, [openDecision])

  const close = useCallback(() => {
    setIsOpen(false)
  }, [])

  const value = useMemo(
    () => ({ isOpen, decision, openDecision, openFromCommentary, openFromPosition, close }),
    [isOpen, decision, openDecision, openFromCommentary, openFromPosition, close]
  )

  return (
    <DecisionCardContext.Provider value={value}>
      {children}
      <DecisionCardPanel />
    </DecisionCardContext.Provider>
  )
}

function formatAsOf(isoString: string): string {
  try {
    const date = new Date(isoString)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMin = Math.floor(diffMs / 60000)
    if (diffMin < 1) return "just now"
    if (diffMin < 60) return `${diffMin}m ago`
    const diffHr = Math.floor(diffMin / 60)
    if (diffHr < 24) return `${diffHr}h ago`
    return date.toLocaleDateString("en-US", { month: "short", day: "numeric" })
  } catch {
    return "—"
  }
}

type ConnectionState = "live" | "stale" | "offline"

function useConnectionState(): ConnectionState {
  const { connected } = useWsCommentary()
  const { data: botStatus, live: botLive } = useBotStatus()

  if (!connected) return "offline"
  if (!botLive || !botStatus?.bot_running) return "stale"
  return "live"
}

function ConnectionBadge({ state }: { state: ConnectionState }) {
  switch (state) {
    case "live":
      return (
        <Badge variant="outline" className="gap-1.5 border-success/40 bg-success/[0.08] text-success">
          <span className="relative flex h-1.5 w-1.5">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-success opacity-75" />
            <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-success" />
          </span>
          Live
        </Badge>
      )
    case "stale":
      return (
        <Badge variant="outline" className="gap-1.5 border-chart-4/40 bg-chart-4/[0.08] text-chart-4">
          <Clock className="h-3 w-3" />
          Stale
        </Badge>
      )
    case "offline":
      return (
        <Badge variant="outline" className="gap-1.5 border-muted-foreground/40 text-muted-foreground">
          <WifiOff className="h-3 w-3" />
          Offline
        </Badge>
      )
  }
}

function StanceDisplay({ stance }: { stance: Stance }) {
  switch (stance) {
    case "long":
      return (
        <div className="flex items-center gap-2.5">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-success/15 ring-1 ring-success/30">
            <TrendingUp className="h-5 w-5 text-success" />
          </div>
          <div>
            <span className="text-lg font-semibold text-success">Long</span>
            <p className="text-xs text-muted-foreground">Bullish position</p>
          </div>
        </div>
      )
    case "short":
      return (
        <div className="flex items-center gap-2.5">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-destructive/15 ring-1 ring-destructive/30">
            <TrendingDown className="h-5 w-5 text-destructive" />
          </div>
          <div>
            <span className="text-lg font-semibold text-destructive">Short</span>
            <p className="text-xs text-muted-foreground">Bearish position</p>
          </div>
        </div>
      )
    case "do_nothing":
      return (
        <div className="flex items-center gap-2.5">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted ring-1 ring-border">
            <Minus className="h-5 w-5 text-muted-foreground" />
          </div>
          <div>
            <span className="text-lg font-semibold text-foreground">Hold</span>
            <p className="text-xs text-muted-foreground">No action recommended</p>
          </div>
        </div>
      )
  }
}

function ConfidenceBar({ value }: { value: number | null | undefined }) {
  if (value == null) {
    return (
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-muted-foreground">Confidence</span>
          <span className="text-sm font-mono tabular-nums text-muted-foreground">—</span>
        </div>
        <div className="h-1.5 w-full rounded-full bg-secondary/60" />
      </div>
    )
  }

  const color =
    value >= 70
      ? "bg-success"
      : value >= 50
        ? "bg-chart-4"
        : "bg-destructive"

  const textColor =
    value >= 70
      ? "text-success"
      : value >= 50
        ? "text-chart-4"
        : "text-destructive"

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground">Confidence</span>
        <span className={`text-sm font-semibold font-mono tabular-nums ${textColor}`}>
          {value.toFixed(1)}%
        </span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-secondary/60 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
        />
      </div>
    </div>
  )
}

function DecisionCardContent({ decision, connectionState }: { decision: DecisionData | null; connectionState: ConnectionState }) {
  if (!decision) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 p-8 text-center">
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-muted">
          <Brain className="h-6 w-6 text-muted-foreground" />
        </div>
        <div>
          <p className="text-sm font-medium text-foreground">No decision yet</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Waiting for the bot to generate a trading signal
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-5 p-4">
      {/* Header: Symbol + As-of + Connection State */}
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-2xl font-bold tracking-tight text-foreground font-mono">
            {decision.symbol}
          </h3>
          <p className="mt-0.5 text-xs text-muted-foreground flex items-center gap-1.5">
            <Clock className="h-3 w-3" />
            {formatAsOf(decision.timestamp)}
          </p>
        </div>
        <ConnectionBadge state={connectionState} />
      </div>

      {/* Stance */}
      <div className="rounded-lg border border-border/60 bg-card/50 p-4">
        <StanceDisplay stance={decision.stance} />
      </div>

      {/* Confidence */}
      <ConfidenceBar value={decision.confidence} />

      {/* Key Facts */}
      {decision.keyFacts && decision.keyFacts.length > 0 && (
        <div className="space-y-2.5">
          <h4 className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            <Info className="h-3.5 w-3.5" />
            Key Facts
          </h4>
          <div className="space-y-1.5">
            {decision.keyFacts.map((fact, i) => (
              <div
                key={i}
                className="flex items-center justify-between gap-3 rounded-md border border-border/40 bg-secondary/20 px-3 py-2"
              >
                <span className="text-xs text-muted-foreground">{fact.label}</span>
                <div className="flex items-center gap-2 text-right">
                  <span className="text-sm font-medium text-foreground">{fact.value}</span>
                  {fact.asOf && (
                    <span className="text-[10px] text-muted-foreground">({fact.asOf})</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Invalidation */}
      {decision.invalidation && (
        <div className="space-y-2">
          <h4 className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            <XCircle className="h-3.5 w-3.5" />
            Invalidation
          </h4>
          <div className="rounded-md border border-chart-4/30 bg-chart-4/[0.06] px-3 py-2.5">
            <p className="text-sm text-chart-4">{decision.invalidation}</p>
          </div>
        </div>
      )}

      {/* Risks */}
      {decision.risks && decision.risks.length > 0 && (
        <div className="space-y-2.5">
          <h4 className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            <ShieldAlert className="h-3.5 w-3.5" />
            Risks
          </h4>
          <ul className="space-y-1.5">
            {decision.risks.map((risk, i) => (
              <li
                key={i}
                className="flex items-start gap-2 rounded-md border border-destructive/20 bg-destructive/[0.04] px-3 py-2"
              >
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-destructive/70" />
                <span className="text-sm text-foreground/90">{risk}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Footer: Wake Reason / Source */}
      {(decision.wakeReason || decision.source) && (
        <div className="mt-auto border-t border-border/40 pt-4">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-muted-foreground">
            {decision.wakeReason && (
              <div className="flex items-center gap-1.5">
                <Activity className="h-3 w-3" />
                <span>Wake: {decision.wakeReason}</span>
              </div>
            )}
            {decision.source && (
              <div className="flex items-center gap-1.5">
                <Target className="h-3 w-3" />
                <span>Source: {decision.source}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function DecisionCardPanel() {
  const { isOpen, decision, close } = useDecisionCard()
  const isMobile = useIsMobile()
  const connectionState = useConnectionState()

  if (isMobile) {
    return (
      <Drawer open={isOpen} onOpenChange={(open) => !open && close()}>
        <DrawerContent className="max-h-[85vh]">
          <DrawerHeader className="border-b border-border/40 pb-3">
            <DrawerTitle className="text-base">Decision Card</DrawerTitle>
          </DrawerHeader>
          <ScrollArea className="flex-1 overflow-auto">
            <DecisionCardContent decision={decision} connectionState={connectionState} />
          </ScrollArea>
        </DrawerContent>
      </Drawer>
    )
  }

  return (
    <Sheet open={isOpen} onOpenChange={(open) => !open && close()}>
      <SheetContent
        side="right"
        className="w-[400px] max-w-[100vw] p-0 flex flex-col"
      >
        <SheetHeader className="shrink-0 border-b border-border/40 p-4">
          <SheetTitle className="text-base">Decision Card</SheetTitle>
        </SheetHeader>
        <ScrollArea className="flex-1">
          <DecisionCardContent decision={decision} connectionState={connectionState} />
        </ScrollArea>
      </SheetContent>
    </Sheet>
  )
}

export function isDecisionCommentary(type: string): boolean {
  const upper = type.toUpperCase()
  return (
    upper === "DECISION" ||
    upper === "SIGNAL" ||
    upper === "SIGNAL_GENERATION" ||
    upper === "RISK_ASSESSMENT" ||
    upper.includes("DECISION") ||
    upper.includes("SIGNAL")
  )
}

export function PulseChip() {
  const { openDecision } = useDecisionCard()
  const { items, connected } = useWsCommentary()
  const { data: botStatus, live: botLive } = useBotStatus()

  const latestDecision = useMemo(() => {
    const decisionItem = items.find((item) => isDecisionCommentary(item.type))
    return decisionItem ? commentaryToDecision(decisionItem) : null
  }, [items])

  const connectionState: ConnectionState = !connected
    ? "offline"
    : !botLive || !botStatus?.bot_running
      ? "stale"
      : "live"

  const handleClick = () => {
    if (latestDecision) {
      openDecision(latestDecision)
    } else {
      openDecision({
        symbol: "—",
        timestamp: new Date().toISOString(),
        stance: "do_nothing",
      })
    }
  }

  const stanceColor =
    latestDecision?.stance === "long"
      ? "text-success border-success/30 bg-success/[0.08]"
      : latestDecision?.stance === "short"
        ? "text-destructive border-destructive/30 bg-destructive/[0.08]"
        : "text-muted-foreground border-border bg-secondary/30"

  const stanceLabel =
    latestDecision?.stance === "long"
      ? "Long"
      : latestDecision?.stance === "short"
        ? "Short"
        : "Hold"

  return (
    <button
      onClick={handleClick}
      className={`flex items-center gap-2 rounded-md border px-2.5 py-1.5 text-[13px] font-medium transition-all hover:ring-2 hover:ring-accent/20 ${stanceColor}`}
    >
      {connectionState === "live" && (
        <span className="relative flex h-1.5 w-1.5">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-current opacity-75" />
          <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-current" />
        </span>
      )}
      {connectionState === "stale" && <Clock className="h-3 w-3" />}
      {connectionState === "offline" && <WifiOff className="h-3 w-3" />}
      <span className="hidden sm:inline">{latestDecision?.symbol ?? "Decision"}</span>
      <span className="font-mono text-xs">{stanceLabel}</span>
    </button>
  )
}
