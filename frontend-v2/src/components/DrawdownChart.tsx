import { useRef, useEffect } from 'react'
import {
  Chart, LineController, LineElement, PointElement, Filler,
  LinearScale, CategoryScale, Tooltip,
} from 'chart.js'

Chart.register(LineController, LineElement, PointElement, Filler, LinearScale, CategoryScale, Tooltip)

interface Props {
  dates: string[]
  values: number[]  // always <= 0
  height?: number
}

export default function DrawdownChart({ dates, values, height = 80 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const chartRef = useRef<Chart | null>(null)

  useEffect(() => {
    if (!canvasRef.current || !values.length) return
    if (chartRef.current) chartRef.current.destroy()

    chartRef.current = new Chart(canvasRef.current, {
      type: 'line',
      data: {
        labels: dates,
        datasets: [{
          label: 'Drawdown',
          data: values,
          borderColor: '#FF3B5C',
          borderWidth: 1.5,
          tension: 0.3,
          fill: true,
          backgroundColor: (ctx) => {
            const { ctx: c, chartArea } = ctx.chart
            if (!chartArea) return 'transparent'
            const g = c.createLinearGradient(0, chartArea.top, 0, chartArea.bottom)
            g.addColorStop(0, 'rgba(255, 59, 92, 0)')
            g.addColorStop(1, 'rgba(255, 59, 92, 0.25)')
            return g
          },
          pointRadius: 0,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(17,17,17,0.95)',
            borderColor: 'rgba(255,59,92,0.3)',
            borderWidth: 1,
            padding: 8,
            callbacks: { label: (ctx) => `  DD: ${ctx.parsed.y.toFixed(2)}%` },
          },
        },
        scales: {
          x: { display: false },
          y: {
            max: 0,
            grid: { color: 'rgba(255,255,255,0.03)' },
            ticks: { callback: (v) => Number(v).toFixed(1) + '%', color: '#606060', font: { family: 'JetBrains Mono', size: 9 }, maxTicksLimit: 3 },
            border: { color: 'rgba(255,255,255,0.07)' },
            position: 'right',
          },
        },
      },
    })

    return () => { chartRef.current?.destroy(); chartRef.current = null }
  }, [dates, values])

  return <div style={{ height, borderTop: '1px solid var(--border)', padding: '0 var(--space-4) var(--space-4)' }}><canvas ref={canvasRef} /></div>
}
