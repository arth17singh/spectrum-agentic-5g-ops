import { scaleLinear } from 'd3-scale'
import { useEffect, useRef, useState } from 'react'
import { cssVar } from '@/core/palette'
import { AXIS_TEXT, Tooltip, useSize, useThemeKey } from './common'

/** Time × latency-bucket heatmap (canvas). Cell = request count; a p95 line is overlaid. Buckets are
 *  upper bounds in ms and drawn on a log axis (equal height per bucket, which is log-spaced). */
export function LatencyHeatmap({ columns, buckets, height = 220, xFormat = (t) => `${Math.round(t)}s` }: {
  columns: { t: number; counts: number[]; p95: number }[]; buckets: number[]; height?: number; xFormat?: (t: number) => string
}) {
  const { ref, width } = useSize()
  const canvas = useRef<HTMLCanvasElement>(null)
  const theme = useThemeKey()
  const [hover, setHover] = useState<{ c: number; b: number; x: number; y: number } | null>(null)
  const m = { l: 56, r: 10, t: 6, b: 22 }
  const w = Math.max(10, width - m.l - m.r)
  const h = Math.max(10, height - m.t - m.b)
  const nb = buckets.length
  const cw = columns.length ? w / columns.length : w
  const bh = h / nb
  // y position of a latency value (ms): log interpolation between bucket bounds
  const yOf = (v: number) => {
    const lo = [0.5 * buckets[0], ...buckets]
    for (let i = 0; i < nb; i++) {
      if (v <= lo[i + 1]) {
        const k = (Math.log(Math.max(v, lo[i])) - Math.log(lo[i])) / (Math.log(lo[i + 1]) - Math.log(lo[i]))
        return h - (i + Math.min(1, Math.max(0, k))) * bh
      }
    }
    return 0
  }
  useEffect(() => {
    const cv = canvas.current
    if (!cv || !width) return
    const dpr = window.devicePixelRatio || 1
    cv.width = w * dpr
    cv.height = h * dpr
    const ctx = cv.getContext('2d')!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)
    const max = Math.max(1, ...columns.flatMap((c) => c.counts))
    const color = scaleLinear<string>().domain([0, max]).range([cssVar('--seq-lo'), cssVar('--seq-hi')])
    const empty = cssVar('--surface-2')
    columns.forEach((col, ci) => {
      col.counts.forEach((n, bi) => {
        ctx.fillStyle = n ? color(n) : empty
        ctx.fillRect(ci * cw, h - (bi + 1) * bh + 0.5, Math.max(0.6, cw - 0.6), bh - 1)
      })
    })
    // p95 overlay
    ctx.strokeStyle = cssVar('--warn')
    ctx.lineWidth = 2
    ctx.lineJoin = 'round'
    ctx.beginPath()
    columns.forEach((col, ci) => {
      const px = ci * cw + cw / 2
      const py = yOf(col.p95)
      if (ci === 0) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    })
    ctx.stroke()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [columns, width, height, theme])
  const ticks = columns.length ? [0, Math.floor(columns.length / 2), columns.length - 1] : []
  const last = columns[columns.length - 1]
  return (
    <div ref={ref} className="relative" style={{ height }}>
      {width > 0 && (
        <>
          <svg width={width} height={height} className="pointer-events-none absolute inset-0" aria-hidden>
            {buckets.map((b, i) => (
              <text key={b} x={m.l - 6} y={m.t + h - (i + 0.5) * bh} dy="0.32em" textAnchor="end" style={AXIS_TEXT} className="tnum">
                {b >= 1000 ? `≤${b / 1000}s` : `≤${b}ms`}
              </text>
            ))}
            {ticks.map((ci) => (
              <text key={ci} x={m.l + ci * cw + cw / 2} y={height - 6} textAnchor={ci === 0 ? 'start' : ci === columns.length - 1 ? 'end' : 'middle'} style={AXIS_TEXT} className="tnum">
                {xFormat(columns[ci].t - (last?.t ?? 0))}
              </text>
            ))}
          </svg>
          <canvas ref={canvas} role="img" aria-label="Latency heatmap with p95 line" className="absolute"
            style={{ left: m.l, top: m.t, width: w, height: h }}
            onPointerMove={(e) => {
              const r = e.currentTarget.getBoundingClientRect()
              const px = e.clientX - r.left
              const py = e.clientY - r.top
              const c = Math.floor(px / cw)
              const b = nb - 1 - Math.floor(py / bh)
              if (c >= 0 && c < columns.length && b >= 0 && b < nb) setHover({ c, b, x: px + m.l, y: py + m.t })
              else setHover(null)
            }}
            onPointerLeave={() => setHover(null)} />
        </>
      )}
      {hover && columns[hover.c] && (
        <Tooltip x={hover.x} y={hover.y} containerWidth={width} title={`${xFormat(columns[hover.c].t - (last?.t ?? 0))}, bucket ${hover.b === 0 ? '0' : buckets[hover.b - 1]}–${buckets[hover.b]} ms`}
          rows={[{ label: 'requests', value: columns[hover.c].counts[hover.b] }, { label: 'p95 this second', value: `${columns[hover.c].p95.toFixed(0)} ms`, color: 'var(--warn)' }]} />
      )}
    </div>
  )
}
