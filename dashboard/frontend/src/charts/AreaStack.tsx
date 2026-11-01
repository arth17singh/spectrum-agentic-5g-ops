import { bisector } from 'd3-array'
import { scaleLinear } from 'd3-scale'
import { area, curveMonotoneX, stack } from 'd3-shape'
import { useMemo, useState } from 'react'
import { AXIS_TEXT, Legend, Tooltip, useSize } from './common'

// Stacked area over a shared x. Fixed series order (color follows the entity); crosshair tooltip
// lists every series at the hovered x.

export interface StackSeries { id: string; label: string; color: string }

export function AreaStack({ xs, rows, series, height = 160, xFormat = (x) => String(x), yFormat = (y) => String(Math.round(y)), legend = true }: {
  xs: number[]
  /** one row per x: { [series id]: value } */
  rows: Record<string, number>[]
  series: StackSeries[]
  height?: number
  xFormat?: (x: number) => string
  yFormat?: (y: number) => string
  legend?: boolean
}) {
  const { ref, width } = useSize()
  const [hover, setHover] = useState<{ i: number; px: number; py: number } | null>(null)
  const m = { l: 36, r: 10, t: 6, b: 20 }
  const w = Math.max(10, width - m.l - m.r)
  const h = Math.max(10, height - m.t - m.b)
  const layers = useMemo(() => stack<Record<string, number>>().keys(series.map((s) => s.id)).value((d, k) => d[k] ?? 0)(rows), [rows, series])
  const ymax = Math.max(1, ...layers.flatMap((l) => l.map((p) => p[1])))
  const x = scaleLinear().domain([xs[0] ?? 0, xs[xs.length - 1] ?? 1]).range([0, w])
  const y = scaleLinear().domain([0, ymax]).nice(4).range([h, 0])
  const mk = area<[number, number]>().x((_, i) => x(xs[i])).y0((p) => y(p[0])).y1((p) => y(p[1])).curve(curveMonotoneX)
  const bis = bisector<number, number>((v) => v).center
  return (
    <div>
      {legend && series.length > 1 && <Legend className="mb-2" items={series.map((s) => ({ label: s.label, color: s.color, kind: 'rect' }))} />}
      <div ref={ref} className="relative" style={{ height }}>
        {width > 0 && xs.length > 1 && (
          <svg width={width} height={height} role="img" aria-label={series.map((s) => s.label).join(', ')}>
            <g transform={`translate(${m.l},${m.t})`}>
              {y.ticks(Math.min(4, Math.max(1, Math.ceil(ymax)))).filter((t) => Number.isInteger(t)).map((t) => (
                <g key={t} transform={`translate(0,${y(t)})`}>
                  <line x1={0} x2={w} style={{ stroke: 'var(--hairline)' }} />
                  <text x={-6} dy="0.32em" textAnchor="end" style={AXIS_TEXT} className="tnum">{yFormat(t)}</text>
                </g>
              ))}
              {layers.map((l, k) => (
                <path key={series[k].id} d={mk(l as unknown as [number, number][]) ?? ''} style={{ fill: series[k].color, stroke: 'var(--surface-1)' }} fillOpacity={0.85} strokeWidth={1} />
              ))}
              <line x1={0} x2={w} y1={h} y2={h} style={{ stroke: 'var(--axis)' }} />
              {x.ticks(5).map((t) => (
                <text key={t} x={x(t)} y={h + 14} textAnchor="middle" style={AXIS_TEXT} className="tnum">{xFormat(t)}</text>
              ))}
              {hover && <line x1={x(xs[hover.i])} x2={x(xs[hover.i])} y1={0} y2={h} style={{ stroke: 'var(--ink-2)' }} strokeOpacity={0.6} />}
              <rect width={w} height={h} fill="transparent"
                onPointerMove={(e) => {
                  const r = e.currentTarget.getBoundingClientRect()
                  const i = bis(xs, x.invert(e.clientX - r.left))
                  setHover({ i, px: e.clientX - r.left + m.l, py: e.clientY - r.top + m.t })
                }}
                onPointerLeave={() => setHover(null)} />
            </g>
          </svg>
        )}
        {hover && rows[hover.i] && (
          <Tooltip x={hover.px} y={hover.py} containerWidth={width} title={xFormat(xs[hover.i])}
            rows={[...series].reverse().map((s) => ({ color: s.color, label: s.label, value: yFormat(rows[hover.i][s.id] ?? 0) }))} />
        )}
      </div>
    </div>
  )
}
