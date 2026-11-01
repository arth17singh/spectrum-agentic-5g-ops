import { bisector, extent } from 'd3-array'
import { scaleLinear, scaleLog } from 'd3-scale'
import { area, curveMonotoneX, curveStepAfter, line } from 'd3-shape'
import { useMemo, useState, type ReactNode } from 'react'
import { AXIS_TEXT, Legend, Tooltip, useSize, type TipRow } from './common'

export interface Pt { x: number; y: number }
export interface Series {
  id: string
  label: string
  color: string
  points: Pt[]
  width?: number
  opacity?: number
  dashed?: boolean
  /** draw a 10% wash under this series */
  area?: boolean
  /** direct label at the line end */
  endLabel?: boolean
}

export interface LineChartProps {
  series: Series[]
  height?: number
  xDomain?: [number, number]
  yDomain?: [number, number]
  yLog?: boolean
  step?: boolean
  xFormat?: (x: number) => string
  yFormat?: (y: number) => string
  tipTitle?: (x: number) => ReactNode
  annotations?: { x: number; label: string; kind?: string }[]
  thresholds?: { y: number; label: string; color?: string }[]
  bands?: { y0: number; y1: number; label?: string }[]
  legend?: boolean
  xTicks?: number
  yTicks?: number
  now?: number
  margin?: { l?: number; r?: number; t?: number; b?: number }
}

export function LineChart({
  series, height = 200, xDomain, yDomain, yLog, step, xFormat = (x) => String(Math.round(x)), yFormat = (y) => y.toFixed(2),
  tipTitle, annotations = [], thresholds = [], bands = [], legend, xTicks = 6, yTicks = 4, now, margin,
}: LineChartProps) {
  const { ref, width } = useSize()
  const [hover, setHover] = useState<{ x: number; px: number; py: number } | null>(null)
  const m = { l: margin?.l ?? 40, r: margin?.r ?? (series.some((s) => s.endLabel) ? 64 : 12), t: margin?.t ?? 8, b: margin?.b ?? 22 }
  const w = Math.max(10, width - m.l - m.r)
  const h = Math.max(10, height - m.t - m.b)

  const all = useMemo(() => series.flatMap((s) => s.points), [series])
  const xd = xDomain ?? ((extent(all, (p) => p.x) as [number, number])[0] != null ? (extent(all, (p) => p.x) as [number, number]) : [0, 1])
  let yd = yDomain ?? (extent(all, (p) => p.y) as [number, number])
  if (yd[0] == null) yd = [0, 1]
  if (yd[0] === yd[1]) yd = [yd[0] - 1, yd[1] + 1]
  for (const t of thresholds) if (!yDomain) yd = [Math.min(yd[0], t.y), Math.max(yd[1], t.y * 1.05)]

  const x = scaleLinear().domain(xd).range([0, w])
  const y = (yLog ? scaleLog().domain([Math.max(1e-6, yd[0]), yd[1]]) : scaleLinear().domain(yd).nice(yTicks)).range([h, 0])
  const curve = step ? curveStepAfter : curveMonotoneX
  const mkLine = line<Pt>().x((p) => x(p.x)).y((p) => y(p.y)).curve(curve)
  const mkArea = area<Pt>().x((p) => x(p.x)).y0(h).y1((p) => y(p.y)).curve(curve)
  const showLegend = legend ?? series.length > 1

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const r = (e.currentTarget as SVGRectElement).getBoundingClientRect()
    const px = e.clientX - r.left
    setHover({ x: x.invert(px), px: px + m.l, py: e.clientY - r.top + m.t })
  }

  const bis = bisector<Pt, number>((p) => p.x).center
  const tipRows: TipRow[] = []
  let snapX: number | null = null
  if (hover) {
    for (const s of series) {
      if (!s.points.length) continue
      const i = bis(s.points, hover.x)
      const p = s.points[Math.max(0, Math.min(s.points.length - 1, i))]
      if (snapX == null || Math.abs(p.x - hover.x) < Math.abs(snapX - hover.x)) snapX = p.x
      tipRows.push({ color: s.color, label: s.label, value: yFormat(p.y), dashed: s.dashed })
    }
  }

  return (
    <div>
      {showLegend && <Legend className="mb-2" items={series.map((s) => ({ label: s.label, color: s.color, dashed: s.dashed }))} />}
      <div ref={ref} className="relative" style={{ height }}>
        {width > 0 && (
          <svg width={width} height={height} role="img" aria-label={series.map((s) => s.label).join(', ')}>
            <g transform={`translate(${m.l},${m.t})`}>
              {bands.map((b, i) => (
                <rect key={i} x={0} width={w} y={y(b.y1)} height={Math.max(0, y(b.y0) - y(b.y1))} style={{ fill: 'var(--surface-2)' }} opacity={i % 2 ? 0.5 : 0.9} />
              ))}
              {(y as any).ticks(yTicks).map((t: number) => (
                <g key={t} transform={`translate(0,${y(t)})`}>
                  <line x1={0} x2={w} style={{ stroke: 'var(--hairline)' }} />
                  <text x={-8} dy="0.32em" textAnchor="end" style={AXIS_TEXT} className="tnum">{yFormat(t)}</text>
                </g>
              ))}
              <line x1={0} x2={w} y1={h} y2={h} style={{ stroke: 'var(--axis)' }} />
              {x.ticks(xTicks).map((t) => (
                <text key={t} x={x(t)} y={h + 15} textAnchor="middle" style={AXIS_TEXT} className="tnum">{xFormat(t)}</text>
              ))}
              {thresholds.map((t) => (
                <g key={t.label}>
                  <line x1={0} x2={w} y1={y(t.y)} y2={y(t.y)} style={{ stroke: t.color ?? 'var(--ink-3)' }} strokeWidth={1} />
                  <text x={w} y={y(t.y) - 4} textAnchor="end" style={{ ...AXIS_TEXT, fill: 'var(--ink-2)' }}>{t.label}</text>
                </g>
              ))}
              {annotations.filter((a) => a.x >= xd[0] && a.x <= xd[1]).map((a, i) => (
                <g key={i} transform={`translate(${x(a.x)},0)`}>
                  <line y1={0} y2={h} style={{ stroke: 'var(--ink-3)' }} strokeOpacity={0.5} />
                  <rect x={-3.5} y={h - 3.5} width={7} height={7} transform={`rotate(45 0 ${h})`} style={{ fill: a.kind === 'checkpoint' ? 'var(--learn)' : 'var(--ink-2)' }} />
                  <title>{a.label}</title>
                </g>
              ))}
              {series.map((s) => s.area && s.points.length > 1 && (
                <path key={`a-${s.id}`} d={mkArea(s.points) ?? ''} style={{ fill: s.color }} opacity={0.1} />
              ))}
              {series.map((s) => s.points.length > 1 && (
                <path key={s.id} d={mkLine(s.points) ?? ''} fill="none" style={{ stroke: s.color }} strokeWidth={s.width ?? 2}
                  strokeOpacity={s.opacity ?? 1} strokeLinejoin="round" strokeLinecap="round" strokeDasharray={s.dashed ? '4 3' : undefined} />
              ))}
              {series.filter((s) => s.endLabel && s.points.length).map((s) => {
                const p = s.points[s.points.length - 1]
                return (
                  <g key={`e-${s.id}`}>
                    <circle cx={x(p.x)} cy={y(p.y)} r={4} style={{ fill: s.color, stroke: 'var(--surface-1)' }} strokeWidth={2} />
                    <text x={x(p.x) + 8} y={y(p.y)} dy="0.32em" style={{ fill: 'var(--ink-1)', fontSize: 11, fontWeight: 500 }}>{s.label}</text>
                  </g>
                )
              })}
              {now != null && now >= xd[0] && now <= xd[1] && (
                <line x1={x(now)} x2={x(now)} y1={0} y2={h} style={{ stroke: 'var(--release)' }} strokeOpacity={0.6} />
              )}
              {hover && snapX != null && (
                <line x1={x(snapX)} x2={x(snapX)} y1={0} y2={h} style={{ stroke: 'var(--ink-2)' }} strokeOpacity={0.6} />
              )}
              <rect width={w} height={h} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
            </g>
          </svg>
        )}
        {hover && tipRows.length > 0 && (
          <Tooltip x={hover.px} y={hover.py} containerWidth={width} title={snapX != null ? (tipTitle ? tipTitle(snapX) : xFormat(snapX)) : undefined} rows={tipRows} />
        )}
      </div>
    </div>
  )
}
