import { scaleLinear } from 'd3-scale'
import { useState, type ReactNode } from 'react'
import { AXIS_TEXT, Tooltip, useSize, type TipRow } from './common'

export interface GanttSeg { start: number; end?: number; label: string; color: string; hatched?: boolean; opacity?: number; tip?: TipRow[] }
export interface GanttRow { id: string; label: ReactNode; color?: string; segs: GanttSeg[]; note?: ReactNode }

/** Timeline of phases per row; a live cursor marks "now". Open segments run to now. */
export function Gantt({ rows, domain, now, rowHeight = 22, xFormat = (v) => `${Math.round(v)}s`, labelWidth = 74, dividers = [] }: {
  rows: GanttRow[]; domain: [number, number]; now?: number; rowHeight?: number; xFormat?: (v: number) => string; labelWidth?: number
  dividers?: { x: number; label: string }[]
}) {
  const { ref, width } = useSize()
  const [hover, setHover] = useState<{ seg: GanttSeg; row: GanttRow; x: number; y: number } | null>(null)
  const top = 16
  const H = rows.length * rowHeight + top + 18
  const w = Math.max(10, width - labelWidth - 8)
  const x = scaleLinear().domain(domain).range([0, w]).clamp(true)
  return (
    <div ref={ref} className="relative" style={{ height: H }}>
      {width > 0 && (
        <svg width={width} height={H} role="img" aria-label="Round timeline">
          <defs>
            <pattern id="gantt-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="5" height="5" fill="transparent" />
              <line x1="0" y1="0" x2="0" y2="5" stroke="rgba(255,255,255,0.45)" strokeWidth="2" />
            </pattern>
          </defs>
          <g transform={`translate(${labelWidth},${top})`}>
            {x.ticks(6).map((t) => (
              <g key={t} transform={`translate(${x(t)},0)`}>
                <line y1={-4} y2={rows.length * rowHeight} style={{ stroke: 'var(--hairline)' }} />
                <text y={rows.length * rowHeight + 14} textAnchor="middle" style={AXIS_TEXT} className="tnum">{xFormat(t)}</text>
              </g>
            ))}
            {dividers.map((d) => (
              <g key={d.label} transform={`translate(${x(d.x)},0)`}>
                <line y1={-14} y2={rows.length * rowHeight} style={{ stroke: 'var(--axis)' }} />
                <text x={4} y={-6} style={{ ...AXIS_TEXT, fill: 'var(--ink-2)' }}>{d.label}</text>
              </g>
            ))}
            {rows.map((row, i) => (
              <g key={row.id} transform={`translate(0,${i * rowHeight})`}>
                {row.segs.map((s, j) => {
                  const end = s.end ?? now ?? domain[1]
                  const x0 = x(s.start)
                  const x1 = x(end)
                  if (x1 - x0 < 0.5) return null
                  return (
                    <g key={j}
                      onPointerMove={(e) => {
                        const r = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
                        setHover({ seg: s, row, x: e.clientX - r.left, y: e.clientY - r.top })
                      }}
                      onPointerLeave={() => setHover(null)}>
                      <rect x={x0} y={4} width={Math.max(1, x1 - x0 - 1)} height={rowHeight - 8} rx={2}
                        style={{ fill: s.color }} opacity={s.opacity ?? 0.9} />
                      {s.hatched && <rect x={x0} y={4} width={Math.max(1, x1 - x0 - 1)} height={rowHeight - 8} rx={2} fill="url(#gantt-hatch)" />}
                    </g>
                  )
                })}
              </g>
            ))}
            {now != null && now >= domain[0] && now <= domain[1] && (
              <g transform={`translate(${x(now)},0)`}>
                <line y1={-6} y2={rows.length * rowHeight} style={{ stroke: 'var(--ink-1)' }} strokeWidth={1} />
                <circle cy={-6} r={3} style={{ fill: 'var(--ink-1)' }} />
              </g>
            )}
          </g>
          {rows.map((row, i) => (
            <foreignObject key={row.id} x={0} y={top + i * rowHeight} width={labelWidth - 6} height={rowHeight}>
              <div className="flex h-full items-center gap-1.5 truncate text-cap text-ink-2">
                {row.color && <span className="size-2 shrink-0 rounded-full" style={{ background: row.color }} />}
                <span className="truncate">{row.label}</span>
              </div>
            </foreignObject>
          ))}
        </svg>
      )}
      {hover && (
        <Tooltip x={hover.x} y={hover.y} containerWidth={width} title={<>{hover.row.label} · {hover.seg.label}</>}
          rows={hover.seg.tip ?? [{ label: 'duration', value: `${((hover.seg.end ?? now ?? 0) - hover.seg.start).toFixed(1)} s` }]} />
      )}
    </div>
  )
}
