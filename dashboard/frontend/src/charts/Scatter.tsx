import { extent } from 'd3-array'
import { scaleLinear, scaleLog } from 'd3-scale'
import { line } from 'd3-shape'
import { useState, type ReactNode } from 'react'
import { AXIS_TEXT, Tooltip, useSize, type TipRow } from './common'

export interface Dot {
  id: string
  x: number
  y: number
  color: string
  label?: string
  showLabel?: boolean
  ring?: boolean
  shape?: 'circle' | 'diamond' | 'cross'
  tip?: TipRow[]
  title?: ReactNode
}

/** Scatter with nearest-point hover (hit target = closest dot, not the painted pixels). */
export function Scatter({ dots, height = 220, xLog, xDomain, yDomain, xFormat = (v) => String(v), yFormat = (v) => String(v), xLabel, yLabel, hlines = [], frontier, onClick }: {
  dots: Dot[]; height?: number; xLog?: boolean; xDomain?: [number, number]; yDomain?: [number, number]
  xFormat?: (v: number) => string; yFormat?: (v: number) => string; xLabel?: string; yLabel?: string
  hlines?: { y: number; label: string; color?: string }[]; frontier?: string[]; onClick?: (d: Dot) => void
}) {
  const { ref, width } = useSize()
  const [hover, setHover] = useState<Dot | null>(null)
  const m = { l: 44, r: 16, t: 10, b: xLabel ? 34 : 22 }
  const w = Math.max(10, width - m.l - m.r)
  const h = Math.max(10, height - m.t - m.b)
  const xe = xDomain ?? (extent(dots, (d) => d.x) as [number, number])
  const ye = yDomain ?? (extent(dots, (d) => d.y) as [number, number])
  const pad = (a: [number, number]) => (a[0] === a[1] ? [a[0] - 1, a[1] + 1] : [a[0] - (a[1] - a[0]) * 0.08, a[1] + (a[1] - a[0]) * 0.08]) as [number, number]
  const x = xLog ? scaleLog().domain([xe[0] * 0.7, xe[1] * 1.4]).range([0, w]) : scaleLinear().domain(xDomain ?? pad(xe)).range([0, w]).nice()
  const y = scaleLinear().domain(yDomain ?? pad(ye)).range([h, 0]).nice()
  const fr = frontier ? dots.filter((d) => frontier.includes(d.id)).sort((a, b) => a.x - b.x) : []

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - r.left
    const py = e.clientY - r.top
    let best: Dot | null = null
    let bd = 40 * 40
    for (const d of dots) {
      const dx = x(d.x) - px
      const dy = y(d.y) - py
      const dd = dx * dx + dy * dy
      if (dd < bd) { bd = dd; best = d }
    }
    setHover(best)
  }

  // log axes: keep only 1-2-5 ticks so labels never collide
  const xticks = xLog
    ? (x as any).ticks(5).filter((t: number) => [1, 2, 5].includes(Math.round(t / 10 ** Math.floor(Math.log10(t) + 1e-9))))
    : (x as any).ticks(6)
  return (
    <div ref={ref} className="relative" style={{ height }}>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label={`${yLabel ?? 'y'} by ${xLabel ?? 'x'}`}>
          <g transform={`translate(${m.l},${m.t})`}>
            {y.ticks(5).map((t) => (
              <g key={t} transform={`translate(0,${y(t)})`}>
                <line x1={0} x2={w} style={{ stroke: 'var(--hairline)' }} />
                <text x={-8} dy="0.32em" textAnchor="end" style={AXIS_TEXT} className="tnum">{yFormat(t)}</text>
              </g>
            ))}
            <line x1={0} x2={w} y1={h} y2={h} style={{ stroke: 'var(--axis)' }} />
            {xticks.map((t: number) => (
              <text key={t} x={x(t)} y={h + 15} textAnchor="middle" style={AXIS_TEXT} className="tnum">{xFormat(t)}</text>
            ))}
            {xLabel && <text x={w} y={h + 30} textAnchor="end" style={AXIS_TEXT}>{xLabel}</text>}
            {yLabel && <text x={0} y={-2} style={AXIS_TEXT}>{yLabel}</text>}
            {hlines.map((l) => (
              <g key={l.label}>
                <line x1={0} x2={w} y1={y(l.y)} y2={y(l.y)} style={{ stroke: l.color ?? 'var(--ink-3)' }} />
                <text x={w} y={y(l.y) - 4} textAnchor="end" style={{ ...AXIS_TEXT, fill: 'var(--ink-2)' }}>{l.label}</text>
              </g>
            ))}
            {fr.length > 1 && (
              <path d={line<Dot>().x((d) => x(d.x)).y((d) => y(d.y))(fr) ?? ''} fill="none" style={{ stroke: 'var(--ink-3)' }} strokeWidth={1.5} />
            )}
            {dots.map((d) => {
              const cx = x(d.x)
              const cy = y(d.y)
              const active = hover?.id === d.id
              return (
                <g key={d.id} transform={`translate(${cx},${cy})`} style={{ cursor: onClick ? 'pointer' : undefined }}>
                  {d.ring && <circle r={9} fill="none" style={{ stroke: d.color }} strokeWidth={2} />}
                  {d.shape === 'diamond' ? (
                    <rect x={-4.5} y={-4.5} width={9} height={9} transform="rotate(45)" style={{ fill: d.color, stroke: 'var(--surface-1)' }} strokeWidth={2} />
                  ) : d.shape === 'cross' ? (
                    <path d="M-5,-5L5,5M5,-5L-5,5" style={{ stroke: d.color }} strokeWidth={2.5} strokeLinecap="round" />
                  ) : (
                    <circle r={active ? 5.5 : 4.5} style={{ fill: d.color, stroke: 'var(--surface-1)' }} strokeWidth={2} />
                  )}
                  {(d.showLabel || active) && d.label && (
                    <text x={cx > w - 110 ? -10 : 10} textAnchor={cx > w - 110 ? 'end' : 'start'} dy="0.32em" style={{ fill: 'var(--ink-1)', fontSize: 11, fontWeight: 500 }}>{d.label}</text>
                  )}
                </g>
              )
            })}
            <rect width={w} height={h} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setHover(null)}
              onClick={() => hover && onClick?.(hover)} />
          </g>
        </svg>
      )}
      {hover && (
        <Tooltip x={x(hover.x) + m.l} y={y(hover.y) + m.t} containerWidth={width} title={hover.title ?? hover.label}
          rows={hover.tip ?? [{ label: xLabel ?? 'x', value: xFormat(hover.x) }, { label: yLabel ?? 'y', value: yFormat(hover.y) }]} />
      )}
    </div>
  )
}
