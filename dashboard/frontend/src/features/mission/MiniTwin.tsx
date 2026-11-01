import { scaleLinear } from 'd3-scale'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Tooltip, useSize, useThemeKey } from '@/charts/common'
import { cssVar } from '@/core/palette'
import { useLive, EMPTY } from '@/core/store'
import type { Rec } from '@/core/types'

// Read-only miniature of the network twin for Mission Control. Hex grid of the 40 sites, fill =
// PRB load (single-hue ramp), ring only when a site is not OK. The Swarm Tower owns the full map.

const SQ3 = Math.sqrt(3)

export function MiniTwin({ height = 250 }: { height?: number }) {
  const { ref, width } = useSize()
  const navigate = useNavigate()
  const sites: Rec[] = useLive((s) => s.twin.sites ?? EMPTY)
  const state: any[] = useLive((s) => s.twin.state ?? EMPTY)
  const layout = useLive((s) => s.twin.layout) ?? { cols: 8, rows: 5 }
  const [hover, setHover] = useState<{ id: string; x: number; y: number } | null>(null)
  const byId = useMemo(() => new Map(state.map((r) => [r[0], r])), [state])

  // flat-top hex, odd columns shifted down
  const cols = layout.cols
  const rows = layout.rows
  const r = Math.min((width - 24) / (1.5 * cols + 0.5), (height - 30) / (SQ3 * (rows + 0.5)))
  const gw = r * (1.5 * cols + 0.5)
  const gh = r * SQ3 * (rows + 0.5)
  const ox = (width - gw) / 2 + r
  const oy = (height - 24 - gh) / 2 + (r * SQ3) / 2
  const theme = useThemeKey()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const color = useMemo(() => scaleLinear<string>().domain([0, 100]).range([cssVar('--seq-lo'), cssVar('--seq-hi')]).clamp(true), [theme])
  const hexPath = (rr: number) =>
    Array.from({ length: 6 }, (_, k) => {
      const a = (Math.PI / 3) * k
      return `${k ? 'L' : 'M'}${(rr * Math.cos(a)).toFixed(2)},${(rr * Math.sin(a)).toFixed(2)}`
    }).join('') + 'Z'
  const center = (s: Rec) => ({ x: ox + s.col * 1.5 * r, y: oy + s.row * r * SQ3 + (s.col % 2 ? (r * SQ3) / 2 : 0) })
  const stadium = sites.filter((s) => s.tags?.includes('stadium'))

  const hovered = hover ? byId.get(hover.id) : null
  const hoveredSite = hover ? sites.find((s) => s.id === hover.id) : null

  return (
    <div ref={ref} className="relative" style={{ height }}>
      {width > 0 && r > 0 && (
        <svg width={width} height={height} role="img" aria-label="Network twin: load per cell site" className="cursor-pointer" onClick={() => navigate('/swarm')}>
          {sites.map((s) => {
            const st = byId.get(s.id)
            const prb = st?.[1] ?? 0
            const bh = st?.[6] ?? 'fibre'
            const sleeping = st?.[7] ?? 0
            const { x, y } = center(s)
            const bad = bh === 'down' || prb >= 85
            const ring = bh === 'down' ? 'var(--critical)' : bh === 'microwave' ? 'var(--warn)' : prb >= 85 ? 'var(--serious)' : null
            return (
              <g key={s.id} transform={`translate(${x},${y})`}
                onPointerEnter={(e) => {
                  const rc = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
                  setHover({ id: s.id, x: e.clientX - rc.left, y: e.clientY - rc.top })
                }}
                onPointerLeave={() => setHover(null)}>
                <path d={hexPath(r - 1.5)} style={{ fill: color(prb), transition: 'fill 600ms ease' }} />
                {ring && <path d={hexPath(r - 2.5)} fill="none" style={{ stroke: ring }} strokeWidth={2} className={bad ? 'pulse-twice' : ''} />}
                {sleeping > 0 && <circle r={2.4} cx={r * 0.42} cy={-r * 0.42} style={{ fill: 'var(--ink-3)' }} />}
                {s.tags?.some((t: string) => t === 'hospital' || t === 'psap') && (
                  <text textAnchor="middle" dy="0.35em" style={{ fill: 'var(--ink-1)', fontSize: Math.max(8, r * 0.45), fontWeight: 600 }}>
                    {s.tags.includes('psap') ? '911' : 'H'}
                  </text>
                )}
              </g>
            )
          })}
          {stadium.length > 0 && (() => {
            const pts = stadium.map(center)
            const x0 = Math.min(...pts.map((p) => p.x)) - r
            const x1 = Math.max(...pts.map((p) => p.x)) + r
            const y0 = Math.min(...pts.map((p) => p.y)) - r
            const y1 = Math.max(...pts.map((p) => p.y)) + r
            return (
              <g>
                <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} rx={8} fill="none" style={{ stroke: 'var(--ink-2)' }} strokeWidth={1} strokeDasharray="3 3" />
                <text x={x0 + 4} y={y0 - 4} style={{ fill: 'var(--ink-2)', fontSize: 10.5 }}>stadium district</text>
              </g>
            )
          })()}
          <g transform={`translate(12,${height - 14})`}>
            <text style={{ fill: 'var(--ink-3)', fontSize: 10.5 }} dy="0.32em">PRB load</text>
            {[0, 25, 50, 75, 100].map((v, i) => (
              <rect key={v} x={54 + i * 18} y={-5} width={16} height={10} rx={2} style={{ fill: color(v) }} />
            ))}
            <text x={54} y={-8} style={{ fill: 'var(--ink-3)', fontSize: 9.5 }}>0%</text>
            <text x={54 + 4 * 18 + 16} y={-8} textAnchor="end" style={{ fill: 'var(--ink-3)', fontSize: 9.5 }}>100%</text>
            <g transform="translate(160,0)">
              <rect x={0} y={-5} width={10} height={10} rx={2} fill="none" style={{ stroke: 'var(--critical)' }} strokeWidth={2} />
              <text x={14} dy="0.32em" style={{ fill: 'var(--ink-3)', fontSize: 10.5 }}>backhaul down</text>
              <rect x={100} y={-5} width={10} height={10} rx={2} fill="none" style={{ stroke: 'var(--warn)' }} strokeWidth={2} />
              <text x={114} dy="0.32em" style={{ fill: 'var(--ink-3)', fontSize: 10.5 }}>on microwave</text>
            </g>
          </g>
        </svg>
      )}
      {hover && hovered && (
        <Tooltip x={hover.x} y={hover.y} containerWidth={width} title={`${hover.id}, hub ${hoveredSite?.hub}${hoveredSite?.tags?.length ? `, ${hoveredSite.tags.join(', ')}` : ''}`}
          rows={[
            { label: 'PRB load', value: `${hovered[1].toFixed(0)}%` },
            { label: 'users', value: hovered[2].toLocaleString() },
            { label: 'throughput', value: `${hovered[3]} Mbps` },
            { label: 'backhaul', value: hovered[6] },
          ]} />
      )}
    </div>
  )
}
