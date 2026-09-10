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
import { Badge, DemoBadge } from './ui/Badge'
import { Button } from './ui/Button'
import type { DecisionSnapshot } from '../api/types'
import { DEMO_DECISION_CARDS } from '../fixtures/demo'

function getVerdictInfo(action: string): { 
  label: string
  variant: 'act' | 'wait' | 'nothing'
  Icon: typeof TrendingUp
  description: string
} {
  switch (action) {
    case 'signal_buy':
      return { 
        label: 'ACT — BUY', 
        variant: 'act', 
        Icon: TrendingUp,
        description: 'Signal indicates a buying opportunity'
      }
    case 'signal_sell':
      return { 
        label: 'ACT — SELL', 
        variant: 'act', 
        Icon: TrendingDown,
        description: 'Signal indicates a selling opportunity'
      }
    case 'skip':
      return { 
        label: 'WAIT', 
        variant: 'wait', 
        Icon: Pause,
        description: 'Conditions not favorable — waiting for better setup'
      }
    case 'veto':
      return { 
        label: 'DO NOTHING', 
        variant: 'nothing', 
        Icon: XCircle,
        description: 'Signal blocked by risk guard'
      }
    default:
      return { 
        label: action.toUpperCase(), 
        variant: 'nothing', 
        Icon: Pause,
        description: 'Unknown decision type'
      }
  }
}

function formatTimestamp(ts: string): string {
  const date = new Date(ts)
  return date.toLocaleString('en-US', {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  })
}

interface StatRowProps {
  label: string
  value: string | number
  subValue?: string
  highlight?: boolean
}

function StatRow({ label, value, subValue, highlight }: StatRowProps) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-helm-border last:border-0">
      <span className="text-sm text-zinc-400">{label}</span>
      <div className="text-right">
        <span className={clsx(
          'mono-nums text-sm font-medium',
          highlight ? 'text-cyan-400' : 'text-white'
        )}>
          {value}
        </span>
        {subValue && (
          <span className="block text-xs text-zinc-500">{subValue}</span>
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
  const isDemo = decision.mode === 'demo' || !location.state?.decision
  
  const { label, variant, Icon, description } = getVerdictInfo(decision.action)
  const extra = decision.extra as {
    thesis?: string
    risks?: string[]
    sources?: Array<{ name: string; tier: number; headline: string }>
  }

  return (
    <div className="p-4 max-w-3xl mx-auto pb-24">
      {/* Back Button */}
      <button
        onClick={() => navigate(-1)}
        className="flex items-center gap-2 text-zinc-400 hover:text-white transition-colors mb-4"
      >
        <ArrowLeft className="w-4 h-4" />
        <span className="text-sm">Back to Pulse</span>
      </button>

      {/* Verdict Card */}
      <Card className="mb-4">
        <CardHeader className="pb-2">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div className={clsx(
                'p-3 rounded-lg',
                variant === 'act' && 'bg-cyan-500/20',
                variant === 'wait' && 'bg-amber-500/20',
                variant === 'nothing' && 'bg-zinc-500/20',
              )}>
                <Icon className={clsx(
                  'w-6 h-6',
                  variant === 'act' && 'text-cyan-400',
                  variant === 'wait' && 'text-amber-400',
                  variant === 'nothing' && 'text-zinc-400',
                )} />
              </div>
              
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-2xl font-bold text-white">
                    {decision.symbol}
                  </span>
                  <Badge variant={variant} className="text-sm px-3 py-1">
                    {label}
                  </Badge>
                  {isDemo && <DemoBadge />}
                </div>
                <p className="text-sm text-zinc-400 mt-1">{description}</p>
              </div>
            </div>
            
            <div className="text-right">
              <span className="mono-nums text-2xl font-bold text-white">
                ${decision.price_vol.price.toFixed(2)}
              </span>
              <p className="text-xs text-zinc-500 mt-1">
                {formatTimestamp(decision.ts)}
              </p>
            </div>
          </div>
        </CardHeader>
        
        <CardContent>
          {/* Key Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 py-4 border-b border-helm-border">
            <div className="text-center">
              <span className={clsx(
                'mono-nums text-xl font-bold',
                decision.confidence >= 0.7 && 'text-emerald-400',
                decision.confidence >= 0.5 && decision.confidence < 0.7 && 'text-amber-400',
                decision.confidence < 0.5 && 'text-zinc-400',
              )}>
                {(decision.confidence * 100).toFixed(0)}%
              </span>
              <p className="text-xs text-zinc-500 mt-1">Confidence</p>
            </div>
            
            <div className="text-center">
              <span className="mono-nums text-xl font-bold text-white">
                {(decision.price_vol.rsi * 100).toFixed(0)}
              </span>
              <p className="text-xs text-zinc-500 mt-1">RSI</p>
            </div>
            
            <div className="text-center">
              <span className="mono-nums text-xl font-bold text-white">
                {decision.price_vol.volume_ratio.toFixed(1)}x
              </span>
              <p className="text-xs text-zinc-500 mt-1">Vol Ratio</p>
            </div>
            
            <div className="text-center">
              <span className={clsx(
                'mono-nums text-xl font-bold',
                decision.news.avg_sentiment > 0.3 && 'text-emerald-400',
                decision.news.avg_sentiment < -0.3 && 'text-red-400',
                Math.abs(decision.news.avg_sentiment) <= 0.3 && 'text-zinc-400',
              )}>
                {decision.news.avg_sentiment >= 0 ? '+' : ''}{decision.news.avg_sentiment.toFixed(2)}
              </span>
              <p className="text-xs text-zinc-500 mt-1">Sentiment</p>
            </div>
          </div>
        </CardContent>
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
