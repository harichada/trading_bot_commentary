import { useRef, useEffect } from 'react'
import {
  Chart, LineController, LineElement, PointElement, Filler,
  LinearScale, CategoryScale, Tooltip,
} from 'chart.js'

Chart.register(LineController, LineElement, PointElement, Filler, LinearScale, CategoryScale, Tooltip)

interface Props {
  dates: string[]
  equity: number[]
  spy?: number[]
  showSpy?: boolean
  height?: number
}

export default function EquityChart({ dates, equity, spy = [], showSpy = false, height = 240 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const chartRef = useRef<Chart | null>(null)

  useEffect(() => {
    if (!canvasRef.current || !equity.length) return

    if (chartRef.current) chartRef.current.destroy()

    chartRef.current = new Chart(canvasRef.current, {
      type: 'line',
      data: {
        labels: dates,
        datasets: [
          {
            label: 'Portfolio Equity',
            data: equity,
            borderColor: '#00FFBB',
            borderWidth: 2,
            tension: 0.4,
            fill: true,
            backgroundColor: (ctx) => {
              const { chart } = ctx
              const { ctx: c, chartArea } = chart
              if (!chartArea) return 'transparent'
              const g = c.createLinearGradient(0, chartArea.top, 0, chartArea.bottom)
              g.addColorStop(0, 'rgba(0, 255, 187, 0.25)')
              g.addColorStop(0.5, 'rgba(0, 255, 187, 0.08)')
              g.addColorStop(1, 'rgba(0, 255, 187, 0)')
              return g
            },
            pointRadius: 0,
            pointHoverRadius: 5,
            pointHoverBackgroundColor: '#00FFBB',
            pointHoverBorderColor: '#000',
            pointHoverBorderWidth: 2,
          },
          {
            label: 'SPY Benchmark',
            data: spy,
            borderColor: '#3B82F6',
            borderWidth: 1,
            borderDash: [4, 4],
            tension: 0.4,
            fill: false,
            pointRadius: 0,
            hidden: !showSpy,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(17, 17, 17, 0.95)',
            borderColor: 'rgba(0, 255, 187, 0.3)',
            borderWidth: 1,
            padding: 12,
            titleColor: '#F0F0F0',
            bodyColor: '#A0A0A0',
            titleFont: { family: 'JetBrains Mono', size: 12, weight: 600 },
            bodyFont: { family: 'JetBrains Mono', size: 11 },
            callbacks: {
              label: (ctx) => {
                const prefix = ctx.datasetIndex === 0 ? 'Equity' : 'SPY'
                return `  ${prefix}: $${ctx.parsed.y.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
              },
            },
          },
        },
        scales: {
          x: {
            grid: { color: 'rgba(255,255,255,0.04)' },
            ticks: { color: '#606060', font: { family: 'JetBrains Mono', size: 10 }, maxTicksLimit: 8, maxRotation: 0 },
            border: { color: 'rgba(255,255,255,0.07)' },
          },
          y: {
            grid: { color: 'rgba(255,255,255,0.04)' },
            ticks: {
              color: '#606060',
              font: { family: 'JetBrains Mono', size: 10 },
              callback: (v) => {
                const n = Number(v)
                if (n >= 1_000_000) return '$' + (n / 1_000_000).toFixed(1) + 'M'
                if (n >= 1000) return '$' + (n / 1000).toFixed(0) + 'K'
                return '$' + n
              },
            },
            border: { color: 'rgba(255,255,255,0.07)' },
            position: 'right',
          },
        },
      },
    })

    return () => { chartRef.current?.destroy(); chartRef.current = null }
  }, [dates, equity, spy, showSpy])

  return (
    <div style={{ height, position: 'relative' }}>
      <canvas ref={canvasRef} />
      {!equity.length && (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
          No equity data
        </div>
      )}
    </div>
  )
}
