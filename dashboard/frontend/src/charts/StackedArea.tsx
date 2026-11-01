import { bisector } from 'd3-array'
import { scaleLinear } from 'd3-scale'
import { area, curveMonotoneX, stack, stackOrderNone, stackOffsetNone } from 'd3-shape'
import { useMemo, useState } from 'react'
import { AXIS_TEXT, Legend, Tooltip, useSize } from './common'

/** Stacked area over time. Keys keep a fixed order (color follows the entity, never its rank). */
export function StackedArea({ data, keys, colors, labels, height = 200, xFormat = (x) => String(x), yFormat = (y) => String(y) }: {
  data: ({ x: number } & Record<string, number>)[]; keys: string[]; colors: Record<string, string>; labels?: Record<string, string>
  height?: number; xFormat?: (x: number) => string; yFormat?: (y: number) => string
}) {
  const { ref, width } = useSize()
  const [hx, setHx] = useState<number | null>(null)
  const m = { l: 48, r: 12, t: 8, b: 22 }
  const w = Math.max(10, width - m.l - m.r)
  const h = Math.max(10, height - m.t - m.b)
  const series = useMemo(() => stack<any>().keys(keys).order(stackOrderNone).offset(stackOffsetNone)(data), [data, keys])
  const xs = data.map((d) => d.x)
  const x = scaleLinear().domain(xs.length ? [Math.min(...xs), Math.max(...xs)] : [0, 1]).range([0, w])
  const ymax = Math.max(1, ...series.flatMap((s) => s.map((p) => p[1])))
  const y = scaleLinear().domain([0, ymax]).nice(4).range([h, 0])
  const mk = area<any>().x((p) => x(p.data.x)).y0((p) => y(p[0])).y1((p) => y(p[1])).curve(curveMonotoneX)
  const i = hx == null ? -1 : bisector<any, number>((d) => d.x).center(data, hx)
  const row = i >= 0 ? data[i] : null
  return (
    <div>
      <Legend className="mb-2" items={keys.map((k) => ({ label: labels?.[k] ?? k, color: colors[k], kind: 'rect' as const }))} />
      <div ref={ref} className="relative" style={{ height }}>
        {width > 0 && (
          <svg width={width} height={height} role="img" aria-label="Stacked area">
            <g transform={`translate(${m.l},${m.t})`}>
              {y.ticks(4).map((t) => (
                <g key={t} transform={`translate(0,${y(t)})`}>
                  <line x1={0} x2={w} style={{ stroke: 'var(--hairline)' }} />
                  <text x={-8} dy="0.32em" textAnchor="end" style={AXIS_TEXT} className="tnum">{yFormat(t)}</text>
                </g>
              ))}
              <line x1={0} x2={w} y1={h} y2={h} style={{ stroke: 'var(--axis)' }} />
              {x.ticks(5).map((t) => <text key={t} x={x(t)} y={h + 15} textAnchor="middle" style={AXIS_TEXT} className="tnum">{xFormat(t)}</text>)}
              {series.map((s) => (
                <path key={s.key} d={mk(s) ?? ''} style={{ fill: colors[s.key], stroke: 'var(--surface-1)' }} strokeWidth={1} fillOpacity={0.85} />
              ))}
              {row && <line x1={x(row.x)} x2={x(row.x)} y1={0} y2={h} style={{ stroke: 'var(--ink-2)' }} strokeOpacity={0.7} />}
              <rect width={w} height={h} fill="transparent"
                onPointerMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); setHx(x.invert(e.clientX - r.left)) }}
                onPointerLeave={() => setHx(null)} />
            </g>
          </svg>
        )}
        {row && (
          <Tooltip x={x(row.x) + m.l} y={20} containerWidth={width} title={xFormat(row.x)}
            rows={[...keys].reverse().map((k) => ({ color: colors[k], label: labels?.[k] ?? k, value: yFormat(row[k] ?? 0) }))} />
        )}
      </div>
    </div>
  )
}
