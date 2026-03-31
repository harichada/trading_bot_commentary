import { useRef, useEffect } from 'react'
import {
  Chart, BarController, BarElement, LinearScale, CategoryScale, Tooltip,
} from 'chart.js'

Chart.register(BarController, BarElement, LinearScale, CategoryScale, Tooltip)

interface Props {
  pnlValues: number[]
  height?: number
}

export default function PnlHistogram({ pnlValues, height = 160 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const chartRef = useRef<Chart | null>(null)

  useEffect(() => {
    if (!canvasRef.current || !pnlValues.length) return
    if (chartRef.current) chartRef.current.destroy()

    const min = Math.min(...pnlValues)
    const max = Math.max(...pnlValues)
    const binCount = 20
    const binWidth = (max - min) / binCount || 1
    const bins = Array.from({ length: binCount }, (_, i) => min + i * binWidth)
    const counts = new Array(binCount).fill(0)
    pnlValues.forEach(v => {
      const idx = Math.min(Math.floor((v - min) / binWidth), binCount - 1)
      counts[idx]++
    })

    chartRef.current = new Chart(canvasRef.current, {
      type: 'bar',
      data: {
        labels: bins.map(b => {
          const abs = Math.abs(b)
          const sign = b < 0 ? '-' : '+'
          return abs >= 1000 ? `${sign}$${(abs / 1000).toFixed(1)}K` : `${sign}$${abs.toFixed(0)}`
        }),
        datasets: [{
          data: counts,
          backgroundColor: bins.map(b => b >= 0 ? 'rgba(0, 200, 83, 0.5)' : 'rgba(255, 59, 92, 0.5)'),
          borderColor: bins.map(b => b >= 0 ? '#00C853' : '#FF3B5C'),
          borderWidth: 1,
          borderRadius: 3,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(17,17,17,0.95)',
            borderColor: 'rgba(255,255,255,0.1)',
            borderWidth: 1,
            padding: 8,
            callbacks: { label: (ctx) => ` ${ctx.parsed.y} trades` },
          },
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#606060', font: { family: 'JetBrains Mono', size: 9 }, maxTicksLimit: 8, maxRotation: 0 },
          },
          y: {
            grid: { color: 'rgba(255,255,255,0.04)' },
            ticks: { color: '#606060', font: { family: 'JetBrains Mono', size: 9 } },
          },
        },
      },
    })

    return () => { chartRef.current?.destroy(); chartRef.current = null }
  }, [pnlValues])

  return (
    <div style={{ height }}>
      <canvas ref={canvasRef} />
      {!pnlValues.length && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)', fontSize: 12 }}>
          No trade data
        </div>
      )}
    </div>
  )
}
