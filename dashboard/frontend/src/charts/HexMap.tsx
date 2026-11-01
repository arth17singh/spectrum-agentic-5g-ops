import { scaleLinear } from 'd3-scale'
import { Moon } from 'lucide-react'
import { useMemo, useState, type ReactNode } from 'react'
import { cssVar } from '@/core/palette'
import { Tooltip, useSize, useThemeKey, type TipRow } from './common'

// Network twin as a flat-topped hex grid (odd columns shifted down, matching twin.neighbours()).
// Each site is split into three sector wedges (its cells); a 1.5px surface gap separates wedges.

export interface HexSite { id: string; col: number; row: number; hub: string; tags: string[]; vendor: string; cells: { id: string; sleeping: boolean }[] }
/** compact twin state row: [id, prb, users, thr, drop, kw, backhaul, sleeping_mask] */
export type SiteState = [string, number, number, number, number, number, string, number]
export type Layer = 'load' | 'throughput' | 'energy' | 'agents'

const SQ3 = Math.sqrt(3)

export function hexCenter(col: number, row: number, R: number) {
  return { x: R + col * 1.5 * R, y: (SQ3 / 2) * R + row * SQ3 * R + (col % 2 ? (SQ3 / 2) * R : 0) }
}

export function hexGridSize(cols: number, rows: number, R: number) {
  return { w: R * (1.5 * (cols - 1) + 2), h: SQ3 * R * rows + (SQ3 / 2) * R }
}

const vtx = (cx: number, cy: number, R: number, deg: number) => {
  const a = (deg * Math.PI) / 180
  return [cx + R * Math.cos(a), cy + R * Math.sin(a)] as const
}
const hexPath = (cx: number, cy: number, R: number) =>
  [0, 60, 120, 180, 240, 300].map((d, i) => `${i ? 'L' : 'M'}${vtx(cx, cy, R, d).join(',')}`).join('') + 'Z'
const wedgePath = (cx: number, cy: number, R: number, k: number) => {
  const a = 120 * k - 90 // edge midpoints sit at -90, 30, 150 for a flat-topped hex
  const m = (deg: number) => vtx(cx, cy, (R * SQ3) / 2, deg).join(',')
  return `M${cx},${cy}L${m(a)}L${vtx(cx, cy, R, a + 30).join(',')}L${vtx(cx, cy, R, a + 90).join(',')}L${m(a + 120)}Z`
}

export interface HexMapProps {
  sites: HexSite[]
  state: SiteState[]
  cols: number
  rows: number
  layer: Layer
  agentState?: Record<string, { state: string; status: string }>
  flashing?: Set<string>
  onSite?: (site: HexSite, st?: SiteState) => void
  height?: number
}

const LAYER_DOMAIN: Record<Exclude<Layer, 'agents'>, [number, number]> = { load: [0, 100], throughput: [0, 1100], energy: [0, 12] }
const LAYER_IDX: Record<Exclude<Layer, 'agents'>, number> = { load: 1, throughput: 3, energy: 5 }
export const LAYER_UNIT: Record<Exclude<Layer, 'agents'>, string> = { load: '% PRB', throughput: 'Mbps', energy: 'kW' }

export function HexMap({ sites, state, cols, rows, layer, agentState = {}, flashing = new Set(), onSite, height: fixedHeight }: HexMapProps) {
  const { ref, width } = useSize()
  const theme = useThemeKey()
  const [hover, setHover] = useState<{ id: string; x: number; y: number } | null>(null)
  const pad = { l: 14, r: 64, t: 26, b: 14 }
  const byWidth = (width - pad.l - pad.r) / (1.5 * (cols - 1) + 2)
  const R = Math.max(8, fixedHeight ? Math.min(byWidth, (fixedHeight - pad.t - pad.b) / (SQ3 * rows + SQ3 / 2)) : byWidth)
  const height = fixedHeight ?? Math.round(pad.t + pad.b + hexGridSize(cols, rows, R).h)
  const grid = hexGridSize(cols, rows, R)
  const ox = pad.l + Math.max(0, (width - pad.l - pad.r - grid.w) / 2)
  const oy = pad.t
  const byId = useMemo(() => new Map(state.map((s) => [s[0], s])), [state])

  const colors = useMemo(() => ({
    lo: cssVar('--seq-lo'), hi: cssVar('--seq-hi'), surface: cssVar('--surface-1'), surface2: cssVar('--surface-2'),
    ink1: cssVar('--ink-1'), ink2: cssVar('--ink-2'), ink3: cssVar('--ink-3'), crit: cssVar('--critical'), warn: cssVar('--warn'),
    operate: cssVar('--operate'), release: cssVar('--release'), axis: cssVar('--axis'),
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [theme])

  const fill = (st?: SiteState, sid?: string) => {
    if (layer === 'agents') {
      const a = agentState[`site-${sid?.slice(2)}`]
      if (!a) return colors.surface2
      if (a.status === 'quarantined') return colors.ink3
      if (a.state === 'thinking') return `color-mix(in oklab, ${colors.release} 55%, ${colors.surface2})`
      if (a.state === 'acting') return colors.operate
      return colors.surface2
    }
    if (!st) return colors.surface2
    const scale = scaleLinear<string>().domain(LAYER_DOMAIN[layer]).range([colors.lo, colors.hi]).clamp(true)
    return scale(st[LAYER_IDX[layer]] as number)
  }

  const center = (s: HexSite) => {
    const c = hexCenter(s.col, s.row, R)
    return { x: ox + c.x, y: oy + c.y }
  }

  // hubs at the centroid of their sites; core below the grid
  const hubs = useMemo(() => {
    const m = new Map<string, { x: number; y: number; n: number; sites: string[] }>()
    for (const s of sites) {
      const c = center(s)
      const h = m.get(s.hub) ?? { x: 0, y: 0, n: 0, sites: [] }
      h.x += c.x; h.y += c.y; h.n += 1; h.sites.push(s.id)
      m.set(s.hub, h)
    }
    return [...m.entries()].map(([id, h]) => ({ id, x: h.x / h.n, y: h.y / h.n, sites: h.sites })).sort((a, b) => a.id.localeCompare(b.id))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sites, R, ox, oy])
  const core = { x: ox + grid.w + 34, y: oy + grid.h / 2 }
  const hubStatus = (h: { sites: string[] }) => {
    const bh = h.sites.map((id) => byId.get(id)?.[6])
    return { cut: bh.some((b) => b === 'down' || b === 'microwave'), mw: bh.some((b) => b === 'microwave'), down: bh.some((b) => b === 'down') }
  }
  const mwTo: Record<string, string> = { H1: 'H2', H2: 'H1', H3: 'H2', H4: 'H3', H5: 'H4' }

  const stadium = sites.filter((s) => s.tags.includes('stadium')).map(center)
  const zone = stadium.length ? {
    x0: Math.min(...stadium.map((p) => p.x)) - R * 1.05, x1: Math.max(...stadium.map((p) => p.x)) + R * 1.05,
    y0: Math.min(...stadium.map((p) => p.y)) - R * 0.98, y1: Math.max(...stadium.map((p) => p.y)) + R * 0.98,
  } : null

  const hovered = hover ? sites.find((s) => s.id === hover.id) : null
  const hst = hovered ? byId.get(hovered.id) : undefined
  const tip: TipRow[] = hst ? [
    { label: 'PRB load', value: `${hst[1].toFixed(0)}%` },
    { label: 'users', value: hst[2].toLocaleString() },
    { label: 'throughput', value: `${hst[3].toFixed(0)} Mbps` },
    { label: 'drop rate', value: `${hst[4].toFixed(2)}%` },
    { label: 'power', value: `${hst[5].toFixed(1)} kW` },
    { label: 'backhaul', value: hst[6] },
    { label: 'cells asleep', value: String([0, 1, 2].filter((k) => hst[7] & (1 << k)).length) },
    { label: 'site agent', value: agentState[`site-${hovered!.id.slice(2)}`]?.state ?? '—' },
  ] : []

  return (
    <div ref={ref} className="relative" style={{ height }}>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label="Network twin: 40 cell sites, transport hubs and links">
          {zone && (
            <g>
              <rect x={zone.x0} y={zone.y0} width={zone.x1 - zone.x0} height={zone.y1 - zone.y0} rx={10} fill="none" style={{ stroke: colors.ink3 }} strokeDasharray="4 4" />
              <text x={zone.x0 + 6} y={zone.y0 - 6} style={{ fill: colors.ink2, fontSize: 11, fontWeight: 500 }}>Stadium district</text>
            </g>
          )}
          {sites.map((s) => {
            const st = byId.get(s.id)
            const { x, y } = center(s)
            const mask = st?.[7] ?? 0
            const bad = st ? (st[6] === 'down' ? colors.crit : st[1] >= 85 || st[4] >= 1 ? colors.warn : null) : null
            return (
              <g key={s.id} style={{ cursor: onSite ? 'pointer' : undefined }}
                onPointerMove={(e) => {
                  const r = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
                  setHover({ id: s.id, x: e.clientX - r.left, y: e.clientY - r.top })
                }}
                onPointerLeave={() => setHover(null)}
                onClick={() => onSite?.(s, st)}>
                {[0, 1, 2].map((k) => {
                  const asleep = !!(mask & (1 << k))
                  return (
                    <path key={k} d={wedgePath(x, y, R * 0.94, k)} style={{ fill: asleep ? colors.surface2 : fill(st, s.id), stroke: colors.surface }} strokeWidth={1.25} />
                  )
                })}
                {[0, 1, 2].filter((k) => mask & (1 << k)).map((k) => {
                  const a = ((120 * k - 30) * Math.PI) / 180
                  return <Moon key={`m${k}`} x={x + Math.cos(a) * R * 0.5 - 5} y={y + Math.sin(a) * R * 0.5 - 5} width={10} height={10} style={{ color: colors.ink2 }} aria-label="sleeping cell" />
                })}
                {bad && <path d={hexPath(x, y, R * 0.98)} fill="none" style={{ stroke: bad }} strokeWidth={2.25} />}
                {flashing.has(s.id) && <path d={hexPath(x, y, R * 1.02)} fill="none" style={{ stroke: colors.ink1 }} strokeWidth={2} className="pulse-twice" />}
                {hover?.id === s.id && <path d={hexPath(x, y, R * 0.98)} fill="none" style={{ stroke: colors.ink1 }} strokeWidth={1.5} />}
                {R > 22 && <text x={x} y={y + R * 0.62} textAnchor="middle" style={{ fill: colors.ink1, fontSize: 9.5, opacity: 0.85, pointerEvents: 'none' }}>{s.id.slice(2)}</text>}
                {s.tags.filter((t) => t === 'hospital' || t === 'psap').map((t) => (
                  <g key={t} transform={`translate(${x + R * 0.38},${y - R * 0.82})`} style={{ pointerEvents: 'none' }}>
                    <rect width={t === 'psap' ? 30 : 14} height={13} rx={3} style={{ fill: colors.ink1 }} />
                    <text x={t === 'psap' ? 15 : 7} y={9.6} textAnchor="middle" style={{ fill: colors.surface, fontSize: 9, fontWeight: 600 }}>{t === 'psap' ? 'PSAP' : 'H'}</text>
                  </g>
                ))}
              </g>
            )
          })}
          {/* transport overlay */}
          {hubs.map((h) => {
            const st = hubStatus(h)
            const mid = { x: (h.x + core.x) / 2, y: (h.y + core.y) / 2 }
            const to = hubs.find((x) => x.id === mwTo[h.id])
            return (
              <g key={h.id} style={{ pointerEvents: 'none' }}>
                <line x1={h.x} y1={h.y} x2={core.x} y2={core.y} style={{ stroke: st.cut ? colors.crit : colors.ink2 }} strokeWidth={st.cut ? 1.5 : 1.25} strokeOpacity={st.cut ? 0.9 : 0.45} strokeDasharray={st.cut ? '6 4' : undefined} />
                {st.cut && (
                  <g transform={`translate(${mid.x},${mid.y})`}>
                    <circle r={9} style={{ fill: colors.surface, stroke: colors.crit }} strokeWidth={1.5} />
                    <path d="M-4,-4L4,4M4,-4L-4,4" style={{ stroke: colors.crit }} strokeWidth={2} strokeLinecap="round" />
                    <text x={12} y={4} style={{ fill: colors.crit, fontSize: 11, fontWeight: 600, paintOrder: 'stroke', stroke: colors.surface, strokeWidth: 3, strokeLinejoin: 'round' as const }}>T{h.id.slice(1)} cut</text>
                  </g>
                )}
                {to && st.mw && (
                  <g>
                    <path d={`M${h.x},${h.y} Q${(h.x + to.x) / 2},${Math.min(h.y, to.y) - R * 1.6} ${to.x},${to.y}`} fill="none" style={{ stroke: colors.operate }} strokeWidth={2.5} strokeDasharray="2 4" strokeLinecap="round" />
                    <text x={(h.x + to.x) / 2} y={Math.min(h.y, to.y) - R * 0.95} textAnchor="middle" style={{ fill: colors.ink1, fontSize: 11, fontWeight: 500, paintOrder: 'stroke', stroke: colors.surface, strokeWidth: 3, strokeLinejoin: 'round' as const }}>MW{h.id.slice(1)} backup, 12 Gbps</text>
                  </g>
                )}
                <rect x={h.x - 6} y={h.y - 6} width={12} height={12} rx={2} style={{ fill: st.down ? colors.crit : colors.ink1, stroke: colors.surface }} strokeWidth={2} />
                <text x={h.x} y={h.y - 10} textAnchor="middle" style={{ fill: colors.ink1, fontSize: 10, fontWeight: 600, paintOrder: 'stroke', stroke: colors.surface, strokeWidth: 3, strokeLinejoin: 'round' as const }}>{h.id}</text>
              </g>
            )
          })}
          <g transform={`translate(${core.x},${core.y})`} style={{ pointerEvents: 'none' }}>
            <rect x={-18} y={-12} width={36} height={24} rx={4} style={{ fill: colors.surface2, stroke: colors.ink2 }} />
            <text textAnchor="middle" y={4} style={{ fill: colors.ink1, fontSize: 10, fontWeight: 600 }}>CORE</text>
          </g>
        </svg>
      )}
      {hover && hovered && <Tooltip x={hover.x} y={hover.y} containerWidth={width} title={<>{hovered.id}, {hovered.hub}, vendor {hovered.vendor}{hovered.tags.length ? `, ${hovered.tags.join(', ')}` : ''}</>} rows={tip} />}
    </div>
  )
}

export function HexLegend({ layer, extra }: { layer: Layer; extra?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-micro text-ink-3">
      {layer !== 'agents' ? (
        <span className="flex items-center gap-1.5">
          <span className="tnum">{LAYER_DOMAIN[layer][0]}</span>
          <span className="h-2 w-20 rounded-sm" style={{ background: 'linear-gradient(90deg, var(--seq-lo), var(--seq-hi))' }} />
          <span className="tnum">{LAYER_DOMAIN[layer][1]} {LAYER_UNIT[layer]}</span>
        </span>
      ) : (
        <>
          <span className="flex items-center gap-1"><span className="size-2.5 rounded-sm bg-s2 ring-1 ring-line" />idle</span>
          <span className="flex items-center gap-1"><span className="size-2.5 rounded-sm" style={{ background: 'color-mix(in oklab, var(--release) 55%, var(--surface-2))' }} />thinking</span>
          <span className="flex items-center gap-1"><span className="size-2.5 rounded-sm" style={{ background: 'var(--operate)' }} />acting</span>
          <span className="flex items-center gap-1"><span className="size-2.5 rounded-sm" style={{ background: 'var(--ink-3)' }} />quarantined</span>
        </>
      )}
      <span className="flex items-center gap-1.5"><svg width="16" height="6"><line x1="0" x2="16" y1="3" y2="3" stroke="var(--ink-2)" strokeOpacity="0.6" strokeWidth="1.5" /></svg>fibre to core</span>
      <span className="flex items-center gap-1.5"><svg width="16" height="6"><line x1="0" x2="16" y1="3" y2="3" stroke="var(--operate)" strokeWidth="2.5" strokeDasharray="2 4" strokeLinecap="round" /></svg>microwave backup</span>
      <span className="flex items-center gap-1"><Moon size={11} />sleeping cell</span>
      <span className="flex items-center gap-1"><span className="rounded-[3px] bg-ink-1 px-1 text-[9px] font-semibold text-page">H</span>hospital</span>
      <span className="flex items-center gap-1"><span className="size-2.5 rounded-sm ring-2 ring-warn" />over 85% load</span>
      <span className="flex items-center gap-1"><span className="size-2.5 rounded-sm ring-2 ring-critical" />backhaul down</span>
      {extra}
    </div>
  )
}
