import { scaleLinear } from 'd3-scale'
import { useMemo, useState } from 'react'
import { planeColor } from '@/core/palette'
import { AXIS_TEXT, Tooltip, useSize } from './common'

export interface Span {
  trace_id: string
  span_id: string
  parent: string | null
  name: string
  plane: string
  start: number // seconds
  end: number
  attrs: Record<string, unknown>
  status: 'ok' | 'error' | string
}

const fmtDur = (s: number) => (s < 0.001 ? `${Math.round(s * 1e6)} µs` : s < 1 ? `${(s * 1000).toFixed(s < 0.01 ? 1 : 0)} ms` : s < 60 ? `${s.toFixed(2)} s` : `${(s / 60).toFixed(1)} min`)

/** OTel trace waterfall: spans ordered depth-first under their parent, bars coloured by plane. */
export function Waterfall({ spans, selected, onSelect, rowHeight = 22, labelWidth = 260 }: {
  spans: Span[]; selected?: string | null; onSelect?: (s: Span) => void; rowHeight?: number; labelWidth?: number
}) {
  const { ref, width } = useSize()
  const [hover, setHover] = useState<{ s: Span; x: number; y: number } | null>(null)
  const rows = useMemo(() => {
    const kids = new Map<string | null, Span[]>()
    const ids = new Set(spans.map((s) => s.span_id))
    for (const s of spans) {
      const p = s.parent && ids.has(s.parent) ? s.parent : null
      kids.set(p, [...(kids.get(p) ?? []), s])
    }
    const out: { s: Span; depth: number }[] = []
    const walk = (p: string | null, depth: number) => {
      for (const s of (kids.get(p) ?? []).sort((a, b) => a.start - b.start)) {
        out.push({ s, depth })
        walk(s.span_id, depth + 1)
      }
    }
    walk(null, 0)
    return out
  }, [spans])
  if (!rows.length) return null
  const t0 = Math.min(...spans.map((s) => s.start))
  const t1 = Math.max(...spans.map((s) => s.end))
  const top = 18
  const H = rows.length * rowHeight + top + 4
  const w = Math.max(10, width - labelWidth - 70)
  const x = scaleLinear().domain([0, Math.max(1e-3, t1 - t0)]).range([0, w])
  return (
    <div ref={ref} className="relative" style={{ height: H }}>
      {width > 0 && (
        <svg width={width} height={H} role="img" aria-label="Trace waterfall">
          <g transform={`translate(${labelWidth},${top})`}>
            {x.ticks(6).map((t) => (
              <g key={t} transform={`translate(${x(t)},0)`}>
                <line y1={-4} y2={rows.length * rowHeight} style={{ stroke: 'var(--hairline)' }} />
                <text y={-7} textAnchor="middle" style={AXIS_TEXT} className="tnum">{fmtDur(t)}</text>
              </g>
            ))}
          </g>
          {rows.map(({ s, depth }, i) => {
            const y = top + i * rowHeight
            const x0 = labelWidth + x(s.start - t0)
            const bw = Math.max(2, x(s.end - s.start))
            const sel = selected === s.span_id
            const err = s.status === 'error'
            return (
              <g key={s.span_id} style={{ cursor: 'pointer' }} onClick={() => onSelect?.(s)}
                onPointerMove={(e) => {
                  const r = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
                  setHover({ s, x: e.clientX - r.left, y: e.clientY - r.top })
                }}
                onPointerLeave={() => setHover(null)}>
                <rect x={0} y={y} width={width} height={rowHeight} style={{ fill: sel ? 'var(--surface-3)' : 'transparent' }} />
                <foreignObject x={4} y={y} width={labelWidth - 10} height={rowHeight}>
                  <div className="flex h-full items-center gap-1.5 truncate text-cap" style={{ paddingLeft: depth * 12 }}>
                    <span className="size-1.5 shrink-0 rounded-full" style={{ background: planeColor(s.plane) }} />
                    <span className={`mono truncate ${err ? 'text-critical-ink' : 'text-ink-1'}`}>{s.name}</span>
                  </div>
                </foreignObject>
                <rect x={x0} y={y + 5} width={bw} height={rowHeight - 10} rx={2}
                  style={{ fill: planeColor(s.plane), stroke: err ? 'var(--critical)' : sel ? 'var(--ink-1)' : 'none' }}
                  strokeWidth={err || sel ? 1.5 : 0} opacity={0.9} />
                <text x={Math.min(x0 + bw + 6, labelWidth + w + 4)} y={y + rowHeight / 2} dy="0.32em" style={AXIS_TEXT} className="tnum">{fmtDur(s.end - s.start)}</text>
              </g>
            )
          })}
        </svg>
      )}
      {hover && (
        <Tooltip x={hover.x} y={hover.y} containerWidth={width} title={hover.s.name}
          rows={[{ label: 'duration', value: fmtDur(hover.s.end - hover.s.start) }, { label: 'offset', value: fmtDur(hover.s.start - t0) },
            { label: 'plane', value: hover.s.plane }, { label: 'status', value: hover.s.status }]} />
      )}
    </div>
  )
}

export { fmtDur }
