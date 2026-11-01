import { scaleLinear } from 'd3-scale'
import { area, curveBasis } from 'd3-shape'
import { AXIS_TEXT, useSize } from './common'

/** Stacked distributions (one per row) with a vertical marker; mass beyond the marker is shaded. */
export function Ridgeline({ rows, range, marker, markerLabel, rowHeight = 30, overlap = 1.6, labelWidth = 56 }: {
  rows: { id: string; label: string; color: string; counts: number[]; note?: string }[]
  range: [number, number]; marker?: number; markerLabel?: string; rowHeight?: number; overlap?: number; labelWidth?: number
}) {
  const { ref, width } = useSize()
  const top = 12
  const H = rows.length * rowHeight + top + 22
  const w = Math.max(10, width - labelWidth - 84)
  const x = scaleLinear().domain(range).range([0, w])
  const max = Math.max(1, ...rows.flatMap((r) => r.counts))
  return (
    <div ref={ref} style={{ height: H }}>
      {width > 0 && (
        <svg width={width} height={H} role="img" aria-label="Per-sample gradient norm distributions">
          <defs>
            <pattern id="ridge-hatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1="0" y1="0" x2="0" y2="4" stroke="currentColor" strokeWidth="1.5" />
            </pattern>
          </defs>
          <g transform={`translate(${labelWidth},${top})`}>
            {x.ticks(6).map((t) => (
              <g key={t} transform={`translate(${x(t)},0)`}>
                <line y1={0} y2={rows.length * rowHeight} style={{ stroke: 'var(--hairline)' }} />
                <text y={rows.length * rowHeight + 15} textAnchor="middle" style={AXIS_TEXT} className="tnum">{t.toFixed(1)}</text>
              </g>
            ))}
            {rows.map((r, i) => {
              const n = r.counts.length
              const bw = (range[1] - range[0]) / n
              const base = (i + 1) * rowHeight
              const y = scaleLinear().domain([0, max]).range([0, rowHeight * overlap])
              const pts = r.counts.map((c, k) => ({ x: range[0] + (k + 0.5) * bw, c }))
              const mk = area<{ x: number; c: number }>().x((p) => x(p.x)).y0(base).y1((p) => base - y(p.c)).curve(curveBasis)
              const clipped = pts.filter((p) => marker == null || p.x >= marker)
              return (
                <g key={r.id}>
                  <path d={mk(pts) ?? ''} style={{ fill: r.color, stroke: r.color }} fillOpacity={0.16} strokeWidth={1.5} />
                  {marker != null && clipped.length > 1 && (
                    <path d={mk(clipped) ?? ''} fill="url(#ridge-hatch)" style={{ color: r.color }} opacity={0.75} />
                  )}
                  <line x1={0} x2={w} y1={base} y2={base} style={{ stroke: 'var(--axis)' }} />
                  <text x={-8} y={base - 4} textAnchor="end" style={{ ...AXIS_TEXT, fill: 'var(--ink-2)' }}>{r.label}</text>
                  {r.note && <text x={w + 8} y={base - 4} style={{ ...AXIS_TEXT, fill: 'var(--ink-1)' }} className="tnum">{r.note}</text>}
                </g>
              )
            })}
            {marker != null && (
              <g transform={`translate(${x(marker)},0)`}>
                <line y1={-6} y2={rows.length * rowHeight} style={{ stroke: 'var(--ink-1)' }} strokeWidth={1.5} />
                <text x={4} y={-1} style={{ fill: 'var(--ink-1)', fontSize: 11, fontWeight: 500 }}>{markerLabel}</text>
              </g>
            )}
          </g>
        </svg>
      )}
    </div>
  )
}
