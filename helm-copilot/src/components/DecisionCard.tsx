import { useLocation, useNavigate } from 'react-router-dom'
import { clsx } from 'clsx'
import {
  ArrowLeft,
  TrendingUp,
  TrendingDown,
  Pause,
  XCircle,
  AlertTriangle,
  FileText,
  BarChart3,
  Newspaper,
  Target,
  Shield,
  Eye,
  EyeOff,
  BookmarkPlus,
} from 'lucide-react'
import { Card, CardContent, CardHeader } from './ui/Card'
import { Badge, DemoBadge, AsOfChip } from './ui/Badge'
import { Button } from './ui/Button'
import type { DecisionSnapshot } from '../api/types'
import { DEMO_DECISION_CARDS } from '../fixtures/demo'

function getVerdictInfo(action: string): { 
  label: string
  subLabel: string
  variant: 'act' | 'wait' | 'nothing'
  Icon: typeof TrendingUp
  description: string
  color: string
} {
  switch (action) {
    case 'signal_buy':
      return { 
        label: 'ACT', 
        subLabel: 'BUY',
        variant: 'act', 
        Icon: TrendingUp,
        description: 'Signal favors entry',
        color: 'cyan',
      }
    case 'signal_sell':
      return { 
        label: 'ACT', 
        subLabel: 'SELL',
        variant: 'act', 
        Icon: TrendingDown,
        description: 'Signal favors exit',
        color: 'cyan',
      }
    case 'skip':
      return { 
        label: 'WAIT', 
        subLabel: 'HOLD',
        variant: 'wait', 
        Icon: Pause,
        description: 'Conditions unclear — wait for better setup',
        color: 'amber',
      }
    case 'veto':
      return { 
        label: 'PASS', 
        subLabel: 'NO ACTION',
        variant: 'nothing', 
        Icon: XCircle,
        description: 'Blocked by risk guard',
        color: 'zinc',
      }
    default:
      return { 
        label: action.toUpperCase(), 
        subLabel: '',
        variant: 'nothing', 
        Icon: Pause,
        description: 'Unknown decision type',
        color: 'zinc',
      }
  }
}

function safeNumber(val: unknown, fallback = 0): number {
  if (typeof val === 'number' && isFinite(val)) return val
  return fallback
}

interface StatRowProps {
  label: string
  value: string | number
  subValue?: string
  highlight?: boolean
  muted?: boolean
}

function StatRow({ label, value, subValue, highlight, muted }: StatRowProps) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-helm-border/50 last:border-0">
      <span className="text-xs text-zinc-500">{label}</span>
      <div className="text-right">
        <span className={clsx(
          'mono-nums text-sm font-medium',
          highlight && 'text-cyan-400',
          muted && 'text-zinc-500',
          !highlight && !muted && 'text-white'
        )}>
          {value}
        </span>
        {subValue && (
          <span className="block text-[10px] text-zinc-600">{subValue}</span>
        )}
      </div>
    </div>
  )
}

export function DecisionCard() {
  const navigate = useNavigate()
  const location = useLocation()
  
  // Get decision from location state or fallback to demo
  const decision: DecisionSnapshot = location.state?.decision ?? DEMO_DECISION_CARDS[0]
  // NEVER label demo as live — explicit check
  const isDemo = decision.mode === 'demo' || !location.state?.decision
  
  const { label, subLabel, variant, Icon, description } = getVerdictInfo(decision.action)
  const extra = decision.extra as {
    thesis?: string
    risks?: string[]
    sources?: Array<{ name: string; tier: number; headline: string }>
  }

  // Safe number extraction
  const price = safeNumber(decision.price_vol?.price)
  const confidence = safeNumber(decision.confidence)
  const rsi = safeNumber(decision.price_vol?.rsi) * 100
  const volRatio = safeNumber(decision.price_vol?.volume_ratio, 1)
  const sentiment = safeNumber(decision.news?.avg_sentiment)

  return (
    <div className="p-4 max-w-3xl mx-auto pb-24">
      {/* Back Button */}
      <button
        onClick={() => navigate(-1)}
        className="flex items-center gap-1.5 text-zinc-500 hover:text-white transition-colors mb-3"
      >
        <ArrowLeft className="w-3.5 h-3.5" />
        <span className="text-xs">Pulse</span>
      </button>

      {/* Hero: Verdict + Key Numbers (8-second glance) */}
      <Card className="mb-3 overflow-hidden">
        {/* Verdict Banner */}
        <div className={clsx(
          'px-4 py-3 flex items-center justify-between',
          variant === 'act' && 'bg-gradient-to-r from-cyan-500/15 to-cyan-500/5',
          variant === 'wait' && 'bg-gradient-to-r from-amber-500/10 to-amber-500/5',
          variant === 'nothing' && 'bg-gradient-to-r from-zinc-600/15 to-zinc-600/5',
        )}>
          <div className="flex items-center gap-3">
            <div className={clsx(
              'p-2.5 rounded-lg',
              variant === 'act' && 'bg-cyan-500/20',
              variant === 'wait' && 'bg-amber-500/15',
              variant === 'nothing' && 'bg-zinc-600/20',
            )}>
              <Icon className={clsx(
                'w-5 h-5',
                variant === 'act' && 'text-cyan-400',
                variant === 'wait' && 'text-amber-400',
                variant === 'nothing' && 'text-zinc-400',
              )} />
            </div>
            
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xl font-bold text-white tracking-tight">
                  {decision.symbol || '—'}
                </span>
                <Badge variant={variant} size="lg">
                  {label}{subLabel ? ` · ${subLabel}` : ''}
                </Badge>
                {isDemo && <DemoBadge />}
              </div>
              <p className="text-xs text-zinc-400 mt-0.5">{description}</p>
            </div>
          </div>
          
          <div className="text-right">
            <span className="mono-nums text-xl font-bold text-white leading-data block">
              ${price.toFixed(2)}
            </span>
            <AsOfChip time={decision.ts} className="mt-1" />
          </div>
        </div>
        
        {/* Key Metrics Strip — above fold */}
        <div className="grid grid-cols-4 divide-x divide-helm-border/50 bg-helm-surface/50">
          <div className="px-3 py-2.5 text-center">
            <span className={clsx(
              'mono-nums text-lg font-bold leading-data block',
              confidence >= 0.7 && 'text-emerald-400',
              confidence >= 0.5 && confidence < 0.7 && 'text-amber-400',
              confidence < 0.5 && 'text-zinc-500',
            )}>
              {(confidence * 100).toFixed(0)}%
            </span>
            <p className="text-[10px] text-zinc-500 uppercase tracking-wide">Conf</p>
          </div>
          
          <div className="px-3 py-2.5 text-center">
            <span className={clsx(
              'mono-nums text-lg font-bold text-white leading-data block',
              rsi > 70 && 'text-red-400',
              rsi < 30 && 'text-emerald-400',
            )}>
              {rsi.toFixed(0)}
            </span>
            <p className="text-[10px] text-zinc-500 uppercase tracking-wide">RSI</p>
          </div>
          
          <div className="px-3 py-2.5 text-center">
            <span className={clsx(
              'mono-nums text-lg font-bold leading-data block',
              volRatio >= 1.5 && 'text-cyan-400',
              volRatio < 1.5 && 'text-white',
            )}>
              {volRatio.toFixed(1)}×
            </span>
            <p className="text-[10px] text-zinc-500 uppercase tracking-wide">Vol</p>
          </div>
          
          <div className="px-3 py-2.5 text-center">
            <span className={clsx(
              'mono-nums text-lg font-bold leading-data block',
              sentiment > 0.3 && 'text-emerald-400',
              sentiment < -0.3 && 'text-red-400',
              Math.abs(sentiment) <= 0.3 && 'text-zinc-400',
            )}>
              {sentiment >= 0 ? '+' : ''}{sentiment.toFixed(2)}
            </span>
            <p className="text-[10px] text-zinc-500 uppercase tracking-wide">Sent</p>
          </div>
        </div>
      </Card>

      {/* Thesis */}
      {extra?.thesis && (
        <Card className="mb-4">
          <CardHeader>
            <div className="flex items-center gap-2">
              <FileText className="w-4 h-4 text-cyan-400" />
              <h3 className="font-medium text-white">Thesis</h3>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-zinc-300 leading-relaxed">
              {extra.thesis}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Entry Details (if ACT) */}
      {decision.would_entry_price && (
        <Card className="mb-4">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Target className="w-4 h-4 text-cyan-400" />
              <h3 className="font-medium text-white">Entry Details</h3>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <div className="px-4">
              <StatRow 
                label="Entry Price" 
                value={`$${decision.would_entry_price.toFixed(2)}`}
              />
              {decision.would_stop_loss && (
                <StatRow 
                  label="Stop Loss" 
                  value={`$${decision.would_stop_loss.toFixed(2)}`}
                  subValue={`${(((decision.would_entry_price - decision.would_stop_loss) / decision.would_entry_price) * 100).toFixed(1)}% risk`}
                />
              )}
              {decision.would_take_profit && (
                <StatRow 
                  label="Take Profit" 
                  value={`$${decision.would_take_profit.toFixed(2)}`}
                  subValue={`${(((decision.would_take_profit - decision.would_entry_price) / decision.would_entry_price) * 100).toFixed(1)}% reward`}
                />
              )}
              {decision.would_size_shares && (
                <StatRow 
                  label="Position Size" 
                  value={`${decision.would_size_shares} shares`}
                  subValue={decision.would_size_mult ? `${(decision.would_size_mult * 100).toFixed(0)}% sizing` : undefined}
                />
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Risks */}
      {extra?.risks && extra.risks.length > 0 && (
        <Card className="mb-4">
          <CardHeader>
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-400" />
              <h3 className="font-medium text-white">Risks</h3>
            </div>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {extra.risks.map((risk, i) => (
                <li key={i} className="flex items-start gap-2 text-sm">
                  <Shield className="w-4 h-4 text-amber-400 mt-0.5 flex-shrink-0" />
                  <span className={clsx(
                    'text-zinc-300',
                    risk.startsWith('VETOED') && 'text-red-400 font-medium'
                  )}>
                    {risk}
                  </span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {/* News Sources */}
      {extra?.sources && extra.sources.length > 0 && (
        <Card className="mb-4">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Newspaper className="w-4 h-4 text-cyan-400" />
              <h3 className="font-medium text-white">Sources</h3>
              <Badge variant="default" size="sm">
                {decision.news.article_count} articles
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <ul className="space-y-3">
              {extra.sources.map((source, i) => (
                <li key={i} className="flex items-start gap-3">
                  <Badge 
                    variant={source.tier === 1 ? 'success' : 'default'} 
                    size="sm"
                    className="mt-0.5"
                  >
                    T{source.tier}
                  </Badge>
                  <div>
                    <span className="text-xs text-zinc-500">{source.name}</span>
                    <p className="text-sm text-zinc-300">{source.headline}</p>
                  </div>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {/* Technical Context */}
      <Card className="mb-4">
        <CardHeader>
          <div className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-cyan-400" />
            <h3 className="font-medium text-white">Technical Context</h3>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <div className="px-4">
            <StatRow 
              label="Regime" 
              value={decision.regime.regime.replace(/_/g, ' ')}
            />
            <StatRow 
              label="SPY Slope" 
              value={`${decision.regime.spy_slope_pct >= 0 ? '+' : ''}${(decision.regime.spy_slope_pct * 100).toFixed(2)}%`}
            />
            {decision.regime.vix && (
              <StatRow 
                label="VIX" 
                value={decision.regime.vix.toFixed(1)}
              />
            )}
            <StatRow 
              label="Strategy" 
              value={decision.strategy_id.replace(/_/g, ' ')}
            />
            {decision.gate_name && (
              <StatRow 
                label="Gate" 
                value={decision.gate_name.replace(/_/g, ' ')}
                highlight
              />
            )}
          </div>
        </CardContent>
      </Card>

      {/* Action Buttons (UI-only for v1) */}
      <Card>
        <CardContent className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm">
              <BookmarkPlus className="w-4 h-4" />
              Paper Trade
            </Button>
            <Button variant="ghost" size="sm">
              <EyeOff className="w-4 h-4" />
              Dismiss
            </Button>
          </div>
          <Button variant="ghost" size="sm">
            <Eye className="w-4 h-4" />
            Mute Symbol
          </Button>
        </CardContent>
      </Card>

      {/* Demo Notice */}
      {isDemo && (
        <div className="mt-6 p-4 rounded-lg bg-purple-500/10 border border-purple-500/20">
          <p className="text-sm text-purple-400">
            <strong>Demo Mode:</strong> This is a simulated decision card. 
            Action buttons are UI-only in v1 — no actual orders are placed.
          </p>
        </div>
      )}
    </div>
  )
}
