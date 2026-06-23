"use client"

import { useState } from "react"
import { PieChart, Pie, Cell, ResponsiveContainer, Sector } from "recharts"
import { TrendingUp, TrendingDown, DollarSign, Percent } from "lucide-react"

const portfolioData = [
  { name: "PLTR", value: 39503.80, shares: 278, color: "oklch(0.74 0.105 205)", change: 4.72 },
  { name: "COIN", value: 29341.50, shares: 150, color: "oklch(0.74 0.13 158)", change: -0.21 },
  { name: "PINS", value: 29211.70, shares: 1439, color: "oklch(0.80 0.115 88)", change: -0.64 },
  { name: "SNAP", value: 9961.56, shares: 1652, color: "oklch(0.66 0.16 22)", change: -0.33 },
  { name: "Cash", value: 8450.00, shares: 0, color: "oklch(0.70 0.115 295)", change: 0 },
]

const totalValue = portfolioData.reduce((sum, item) => sum + item.value, 0)

interface ActiveShapeProps {
  cx: number
  cy: number
  midAngle: number
  innerRadius: number
  outerRadius: number
  startAngle: number
  endAngle: number
  fill: string
  payload: typeof portfolioData[0]
  percent: number
  value: number
}

const renderActiveShape = (props: ActiveShapeProps) => {
  const { cx, cy, innerRadius, outerRadius, startAngle, endAngle, fill } = props

  return (
    <g>
      {/* Calm emphasis: a slightly wider, full-opacity sector — no glow/shadow */}
      <Sector
        cx={cx}
        cy={cy}
        innerRadius={innerRadius}
        outerRadius={outerRadius + 4}
        startAngle={startAngle}
        endAngle={endAngle}
        fill={fill}
      />
      {/* Thin inner ring to seat the active slice */}
      <Sector
        cx={cx}
        cy={cy}
        innerRadius={innerRadius - 2}
        outerRadius={innerRadius}
        startAngle={startAngle}
        endAngle={endAngle}
        fill={fill}
        opacity={0.5}
      />
    </g>
  )
}

export function PortfolioAllocation() {
  const [activeIndex, setActiveIndex] = useState(0)
  const activeItem = portfolioData[activeIndex]

  return (
    <div className="rounded-xl glass overflow-hidden h-full reveal">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-border/60">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold tracking-tight text-foreground">Portfolio Allocation</h3>
        </div>
        <div className="flex items-center gap-2">
          <DollarSign className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-semibold font-mono tabular-nums text-foreground">
            ${totalValue.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </span>
        </div>
      </div>
      
      {/* Chart with Center Content */}
      <div className="relative h-[280px] p-4">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              activeIndex={activeIndex}
              activeShape={renderActiveShape as any}
              data={portfolioData}
              cx="50%"
              cy="50%"
              innerRadius={70}
              outerRadius={100}
              paddingAngle={2}
              dataKey="value"
              onMouseEnter={(_, index) => setActiveIndex(index)}
              animationBegin={0}
              animationDuration={800}
            >
              {portfolioData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.color}
                  stroke="oklch(0.178 0.004 264)"
                  strokeWidth={1}
                  style={{ cursor: 'pointer' }}
                />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
        
        {/* Center Content */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center">
            <p className="text-2xl font-semibold tracking-tight text-foreground">{activeItem.name}</p>
            <p className="text-lg font-semibold font-mono tabular-nums text-muted-foreground">
              {((activeItem.value / totalValue) * 100).toFixed(1)}%
            </p>
            <div className={`flex items-center justify-center gap-1 mt-1 text-sm font-mono tabular-nums font-medium ${
              activeItem.change >= 0 ? 'text-success' : 'text-destructive'
            }`}>
              {activeItem.change !== 0 && (
                activeItem.change >= 0 
                  ? <TrendingUp className="h-3.5 w-3.5" />
                  : <TrendingDown className="h-3.5 w-3.5" />
              )}
              {activeItem.change !== 0 ? `${activeItem.change >= 0 ? '+' : ''}${activeItem.change}%` : '-'}
            </div>
          </div>
        </div>
      </div>
      
      {/* Legend */}
      <div className="px-5 py-4 border-t border-border/60 space-y-1">
        {portfolioData.map((item, index) => (
          <button
            key={item.name}
            onClick={() => setActiveIndex(index)}
            className={`lift w-full flex items-center justify-between px-3 py-2 rounded-lg border ${
              index === activeIndex ? 'border-border/60 bg-secondary/40' : 'border-transparent'
            }`}
          >
            <div className="flex items-center gap-3">
              <div
                className="w-2.5 h-2.5 rounded-sm"
                style={{ backgroundColor: item.color }}
              />
              <span className={`eyebrow ${
                index === activeIndex ? 'text-foreground' : ''
              }`}>
                {item.name}
              </span>
            </div>
            <div className="flex items-center gap-4">
              <span className="text-sm font-mono tabular-nums text-muted-foreground">
                ${item.value.toLocaleString('en-US', { minimumFractionDigits: 2 })}
              </span>
              <span className={`text-xs font-mono tabular-nums font-medium w-16 text-right ${
                item.change >= 0 ? 'text-success' : 'text-destructive'
              }`}>
                {item.change !== 0 ? `${item.change >= 0 ? '+' : ''}${item.change}%` : '-'}
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
