import { scaleLinear } from 'd3-scale'
import { useState } from 'react'
import { AXIS_TEXT, Tooltip, useSize } from './common'

/** Histogram (single series) with vertical percentile markers. Bars <= 24px, 4px rounded tops, 2px gaps. */
export function Histogram({ bins, color = 'var(--s1)', height = 180, xFormat = (v) => String(Math.round(v)), markers = [], yLabel = 'share' }: {
  bins: { x0: number; x1: number; v: number }[]; color?: string; height?: number; xFormat?: (v: number) => string
  markers?: { x: number; label: string }[]; yLabel?: string
}) {
  const { ref, width } = useSize()
  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(null)
  const m = { l: 36, r: 12, t: 18, b: 22 }
  const w = Math.max(10, width - m.l - m.r)
  const h = Math.max(10, height - m.t - m.b)
  const x0 = bins[0]?.x0 ?? 0
  const x1 = bins[bins.length - 1]?.x1 ?? 1
  const x = scaleLinear().domain([x0, x1]).range([0, w])
  const vmax = Math.max(1e-9, ...bins.map((b) => b.v))
  const y = scaleLinear().domain([0, vmax]).range([h, 0]).nice(3)
  return (
    <div ref={ref} className="relative" style={{ height }}>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label="Latency distribution">
          <g transform={`translate(${m.l},${m.t})`}>
            {y.ticks(3).map((t) => (
              <g key={t} transform={`translate(0,${y(t)})`}>
                <line x1={0} x2={w} style={{ stroke: 'var(--hairline)' }} />
                <text x={-6} dy="0.32em" textAnchor="end" style={AXIS_TEXT} className="tnum">{`${Math.round(t * 100)}%`}</text>
              </g>
            ))}
            {bins.map((b, i) => {
              const bx = x(b.x0) + 1
              const bw = Math.max(1, Math.min(24, x(b.x1) - x(b.x0) - 2))
              const by = y(b.v)
              const bh = h - by
              const r = Math.min(4, bw / 2, bh)
              return (
                <path key={i}
                  d={`M${bx},${h}V${by + r}Q${bx},${by} ${bx + r},${by}H${bx + bw - r}Q${bx + bw},${by} ${bx + bw},${by + r}V${h}Z`}
                  style={{ fill: color }} opacity={hover && hover.i !== i ? 0.55 : 1}
                  onPointerMove={(e) => {
                    const rc = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
                    setHover({ i, x: e.clientX - rc.left, y: e.clientY - rc.top })
                  }}
                  onPointerLeave={() => setHover(null)} />
              )
            })}
            <line x1={0} x2={w} y1={h} y2={h} style={{ stroke: 'var(--axis)' }} />
            {x.ticks(6).map((t) => (
              <text key={t} x={x(t)} y={h + 15} textAnchor="middle" style={AXIS_TEXT} className="tnum">{xFormat(t)}</text>
            ))}
            {markers.filter((mk) => mk.x >= x0 && mk.x <= x1).map((mk) => (
              <g key={mk.label} transform={`translate(${x(mk.x)},0)`}>
                <line y1={-6} y2={h} style={{ stroke: 'var(--ink-1)' }} strokeWidth={1} />
                <text x={3} y={-8} style={{ fill: 'var(--ink-1)', fontSize: 11, fontWeight: 500 }}>{mk.label}</text>
              </g>
            ))}
          </g>
        </svg>
      )}
      {hover && (
        <Tooltip x={hover.x} y={hover.y} containerWidth={width} title={`${xFormat(bins[hover.i].x0)}–${xFormat(bins[hover.i].x1)}`}
          rows={[{ label: yLabel, value: `${(bins[hover.i].v * 100).toFixed(1)}%` }]} />
      )}
    </div>
  )
}
