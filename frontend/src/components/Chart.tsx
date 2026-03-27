import { useEffect, useRef, useState } from 'react'
import { createChart, type IChartApi, type ISeriesApi, type CandlestickData, type Time, ColorType } from 'lightweight-charts'

interface ChartProps {
  data?: CandlestickData<Time>[]
  volumeData?: { time: Time; value: number; color: string }[]
  height?: number
  type?: 'candlestick' | 'area'
  showVolume?: boolean
  emaLines?: { period: number; color: string; data: { time: Time; value: number }[] }[]
}

function generateMockCandles(): { candles: CandlestickData<Time>[]; volumes: { time: Time; value: number; color: string }[] } {
  const candles: CandlestickData<Time>[] = []
  const volumes: { time: Time; value: number; color: string }[] = []
  let price = 880
  const now = Math.floor(Date.now() / 1000)
  const daySeconds = 86400

  for (let i = 200; i >= 0; i--) {
    const time = (now - i * daySeconds) as Time
    const open = price + (Math.random() - 0.48) * 10
    const close = open + (Math.random() - 0.45) * 15
    const high = Math.max(open, close) + Math.random() * 8
    const low = Math.min(open, close) - Math.random() * 8
    candles.push({ time, open, high, low, close })
    volumes.push({
      time,
      value: Math.floor(Math.random() * 50000000) + 10000000,
      color: close >= open ? 'rgba(34, 197, 94, 0.3)' : 'rgba(239, 68, 68, 0.3)',
    })
    price = close
  }
  return { candles, volumes }
}

function computeEMA(data: CandlestickData<Time>[], period: number): { time: Time; value: number }[] {
  const result: { time: Time; value: number }[] = []
  const k = 2 / (period + 1)
  let ema = 0

  for (let i = 0; i < data.length; i++) {
    if (i === 0) {
      ema = data[i].close
    } else {
      ema = data[i].close * k + ema * (1 - k)
    }
    if (i >= period - 1) {
      result.push({ time: data[i].time, value: ema })
    }
  }
  return result
}

export default function Chart({ data, volumeData, height = 400, type = 'candlestick', showVolume = true, emaLines }: ChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | ISeriesApi<'Area'> | null>(null)
  const [timeRange, setTimeRange] = useState<string>('1D')

  const _unusedTimeRange = timeRange

  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#000000' },
        textColor: '#94a3b8',
        fontFamily: 'Inter, sans-serif',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: '#1a1a1a' },
        horzLines: { color: '#1a1a1a' },
      },
      crosshair: {
        mode: 0,
        vertLine: { color: '#94a3b8', width: 1, style: 2 },
        horzLine: { color: '#94a3b8', width: 1, style: 2 },
      },
      rightPriceScale: {
        borderColor: '#1a1a1a',
        scaleMargins: { top: 0.1, bottom: showVolume ? 0.25 : 0.1 },
      },
      timeScale: {
        borderColor: '#1a1a1a',
        timeVisible: true,
      },
      width: containerRef.current.clientWidth,
      height,
    })

    chartRef.current = chart

    const mock = generateMockCandles()
    const candleData = data ?? mock.candles
    const volData = volumeData ?? mock.volumes

    if (type === 'candlestick') {
      const series = chart.addCandlestickSeries({
        upColor: '#22c55e',
        downColor: '#ef4444',
        borderUpColor: '#22c55e',
        borderDownColor: '#ef4444',
        wickUpColor: '#22c55e',
        wickDownColor: '#ef4444',
      })
      series.setData(candleData)
      seriesRef.current = series

      // EMA overlays
      const emaConfigs = emaLines ?? [
        { period: 9, color: '#06b6d4', data: computeEMA(candleData, 9) },
        { period: 21, color: '#a855f7', data: computeEMA(candleData, 21) },
        { period: 50, color: '#eab308', data: computeEMA(candleData, 50) },
      ]
      for (const ema of emaConfigs) {
        const line = chart.addLineSeries({
          color: ema.color,
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        })
        line.setData(ema.data)
      }
    } else {
      const series = chart.addAreaSeries({
        topColor: 'rgba(34, 197, 94, 0.4)',
        bottomColor: 'rgba(34, 197, 94, 0.0)',
        lineColor: '#22c55e',
        lineWidth: 2,
      })
      const areaData = candleData.map((c) => ({ time: c.time, value: c.close }))
      series.setData(areaData)
      seriesRef.current = series as unknown as ISeriesApi<'Candlestick'>
    }

    if (showVolume) {
      const volumeSeries = chart.addHistogramSeries({
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
      })
      chart.priceScale('volume').applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      })
      volumeSeries.setData(volData)
    }

    chart.timeScale().fitContent()

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth })
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
      chartRef.current = null
      seriesRef.current = null
    }
  }, [data, volumeData, height, type, showVolume, emaLines])

  const timeRanges = type === 'candlestick'
    ? ['1m', '5m', '15m', '1D']
    : ['1D', '5D', '1M', 'YTD', '1Y', 'ALL']

  return (
    <div className="relative">
      <div className="absolute top-2 right-2 z-10 flex gap-1">
        {timeRanges.map((tr) => (
          <button
            key={tr}
            onClick={() => setTimeRange(tr)}
            className={`px-2 py-1 text-[10px] font-semibold rounded transition-colors ${
              timeRange === tr
                ? 'bg-green/20 text-green'
                : 'bg-bg-card text-text-muted hover:text-text-secondary'
            }`}
          >
            {tr}
          </button>
        ))}
      </div>
      <div ref={containerRef} className="w-full" />
    </div>
  )
}
