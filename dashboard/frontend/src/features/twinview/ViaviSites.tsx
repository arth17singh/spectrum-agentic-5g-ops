import { useEffect, useMemo, useState } from 'react'
import { ExternalLink, Info } from 'lucide-react'
import { get } from '@/core/api'
import type { Rec } from '@/core/types'
import { AXIS_TEXT, useSize } from '@/charts/common'
import { Panel } from '@/ui'

// Viavi RSG scenarios and their sites, from the newest read-only extract mirrored from S3
// (outputs/viavi-sites/<UTC>/viavi_sites.json, GET /api/twinview/viavi-sites). Organiser data, shown for
// context only, never committed. The map is hand-built SVG: an equirectangular projection fitted to the
// sites' bounding box, no tiles and no external requests. Geo links open OpenStreetMap in a new tab.

interface Site { name: string; cells: number | null; lat: number | null; lon: number | null }
interface Scenario { name: string; cells: number | null; ues: number | null; note: string | null; sites: Site[] }

const osm = (lat: number, lon: number) => `https://www.openstreetmap.org/?mlat=${lat.toFixed(6)}&mlon=${lon.toFixed(6)}#map=15/${lat.toFixed(6)}/${lon.toFixed(6)}`
const located = (s: Site): s is Site & { lat: number; lon: number } => s.lat != null && s.lon != null

function GeoLink({ lat, lon, label = 'map' }: { lat: number; lon: number; label?: string }) {
  return (
    <a href={osm(lat, lon)} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-micro text-ink-3 underline decoration-dotted underline-offset-2 hover:text-ink-1">
      {label}<ExternalLink size={10} aria-hidden />
    </a>
  )
}

function centre(sites: (Site & { lat: number; lon: number })[]): { lat: number; lon: number } | null {
  if (!sites.length) return null
  return { lat: sites.reduce((a, s) => a + s.lat, 0) / sites.length, lon: sites.reduce((a, s) => a + s.lon, 0) / sites.length }
}

function SiteMap({ sites }: { sites: (Site & { lat: number; lon: number })[] }) {
  const { ref, width } = useSize()
  const [hover, setHover] = useState<number | null>(null)
  const height = Math.min(420, Math.max(240, width * 0.6))
  const pad = 28
  const lats = sites.map((s) => s.lat), lons = sites.map((s) => s.lon)
  let [la0, la1, lo0, lo1] = [Math.min(...lats), Math.max(...lats), Math.min(...lons), Math.max(...lons)]
  const span = Math.max(la1 - la0, (lo1 - lo0) * Math.cos(((la0 + la1) / 2) * Math.PI / 180), 0.005)
  // equirectangular, x scaled by cos(mean latitude); keep the aspect, centre the box
  const k = Math.cos(((la0 + la1) / 2) * Math.PI / 180)
  const cx = (lo0 + lo1) / 2, cy = (la0 + la1) / 2
  la0 = cy - span / 2; la1 = cy + span / 2
  const w = Math.max(10, width - 2 * pad), h = height - 2 * pad - 16  // room for the scale note
  const scale = Math.min(w, h) / span
  const px = (lon: number) => pad + w / 2 + (lon - cx) * k * scale
  const py = (lat: number) => pad + h / 2 - (lat - cy) * scale
  const maxCells = Math.max(1, ...sites.map((s) => s.cells ?? 1))
  const r = (c: number | null) => 4 + 10 * Math.sqrt((c ?? 1) / maxCells)
  const km = span * 111 // degrees of latitude -> km, for the scale note
  const hs = hover != null ? sites[hover] : null
  return (
    <div ref={ref} className="relative w-full" style={{ height }}>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label={`${sites.length} Viavi sites`}>
          <rect x={0.5} y={0.5} width={width - 1} height={height - 1} rx={8} style={{ fill: 'var(--surface-2)', stroke: 'var(--hairline)' }} />
          {[0.25, 0.5, 0.75].map((f) => (
            <g key={f}>
              <line x1={pad} x2={width - pad} y1={pad + h * f} y2={pad + h * f} style={{ stroke: 'var(--hairline)' }} strokeDasharray="2 4" />
              <line y1={pad} y2={height - pad} x1={pad + w * f} x2={pad + w * f} style={{ stroke: 'var(--hairline)' }} strokeDasharray="2 4" />
            </g>
          ))}
          {sites.map((s, i) => (
            <a key={`${s.name}-${i}`} href={osm(s.lat, s.lon)} target="_blank" rel="noopener noreferrer"
              onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} onFocus={() => setHover(i)} onBlur={() => setHover(null)}>
              <circle cx={px(s.lon)} cy={py(s.lat)} r={r(s.cells)} style={{ fill: 'var(--operate)', stroke: 'var(--surface-1)' }} strokeWidth={1.5} opacity={hover === i ? 1 : 0.8} />
              <title>{`${s.name}: ${s.cells ?? '?'} cells, ${s.lat.toFixed(5)}, ${s.lon.toFixed(5)} (opens OpenStreetMap)`}</title>
            </a>
          ))}
          <text x={pad} y={height - 8} style={AXIS_TEXT}>{`≈ ${km < 2 ? `${Math.round(km * 1000)} m` : `${km.toFixed(1)} km`} tall · north up`}</text>
        </svg>
      )}
      {hs && (
        <div className="pointer-events-none absolute rounded-md border border-line bg-s1 px-2 py-1 text-micro text-ink-1 shadow"
          style={{ left: Math.min(Math.max(0, px(hs.lon) + 10), Math.max(0, width - 180)), top: Math.max(0, py(hs.lat) - 40) }}>
          <div className="font-medium">{hs.name}</div>
          <div className="tnum text-ink-3">{hs.cells ?? '?'} cells · {hs.lat.toFixed(4)}, {hs.lon.toFixed(4)}</div>
        </div>
      )}
    </div>
  )
}

export function ViaviSites() {
  const [d, setD] = useState<Rec | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [pick, setPick] = useState<string | null>(null)
  useEffect(() => { get<Rec>('/api/twinview/viavi-sites').then(setD).catch((e) => setErr(String(e))) }, [])
  const scenarios: Scenario[] = useMemo(() => (d?.loaded ? d.scenarios ?? [] : []), [d])
  const pr = scenarios.filter((s) => /puerto rico/i.test(s.name))
  const current = scenarios.find((s) => s.name === pick) ?? pr[0] ?? scenarios[0] ?? null
  const sites = (current?.sites ?? []).filter(located)
  const c = centre(sites)
  return (
    <div className="space-y-4" data-testid="viavi-sites">
      <div className="flex items-start gap-2 rounded-md border px-4 py-2.5 text-cap text-ink-1" role="note" data-testid="viavi-banner"
        style={{ borderColor: 'color-mix(in oklab, var(--warn) 55%, transparent)', background: 'color-mix(in oklab, var(--warn) 10%, transparent)' }}>
        <Info size={14} className="mt-0.5 shrink-0" style={{ color: 'var(--warn-ink)' }} aria-hidden />
        <span>
          <span className="font-medium">Viavi RSG scenarios, read from Viavi, shown for context.</span> Our training, evaluations and federated learning ran on the
          Maveric 4-cell twin. Operating our RL policy on Viavi is a single measure-only run (see the Viavi rung on Trust Gate). Federated learning on Viavi
          regions is a planned next step.
        </span>
      </div>
      {err && <div className="text-cap text-critical-ink">{err}</div>}
      {!err && !d && <div className="text-cap text-ink-3">Loading…</div>}
      {d && !d.loaded && (
        <Panel title="Viavi sites" plane="operate">
          <div className="flex min-h-24 items-center justify-center rounded-md border border-dashed border-line text-cap text-ink-3" data-testid="viavi-empty">
            Viavi sites not loaded yet{d.reason ? ` (${d.reason})` : ''}.
          </div>
        </Panel>
      )}
      {d?.loaded && current && (
        <Panel title="Viavi sites" plane="operate"
          subtitle={`${scenarios.length} scenarios · ${d.source ?? 'Viavi RSG'} · read ${String(d.read_at ?? '').slice(0, 16).replace('T', ' ')}Z`}>
          <div className="mb-3 flex flex-wrap items-center gap-x-3 gap-y-2">
            <label className="flex w-full min-w-0 items-center gap-2 text-cap text-ink-2 md:w-auto md:flex-1">
              Scenario
              <select value={current.name} onChange={(e) => setPick(e.target.value)} data-testid="viavi-scenario"
                className="min-w-0 flex-1 rounded-md border border-line bg-s2 px-2 py-1 text-cap text-ink-1">
                {pr.length > 0 && <optgroup label="MWC Puerto Rico">{pr.map((s) => <option key={s.name} value={s.name}>{s.name}</option>)}</optgroup>}
                <optgroup label="Other scenarios">{scenarios.filter((s) => !pr.includes(s)).map((s) => <option key={s.name} value={s.name}>{s.name}</option>)}</optgroup>
              </select>
            </label>
            <span className="tnum text-micro text-ink-3">{current.cells ?? '?'} cells · {current.ues ?? '?'} UEs · {current.sites.length} sites{current.sites.length > sites.length ? ` (${current.sites.length - sites.length} without coordinates)` : ''}</span>
            {c && <GeoLink lat={c.lat} lon={c.lon} label="scenario centre on OpenStreetMap" />}
          </div>
          {current.note && <div className="mb-2 text-micro text-ink-3">{current.note}</div>}
          {sites.length ? <SiteMap sites={sites} /> : <div className="flex min-h-24 items-center justify-center rounded-md border border-dashed border-line text-cap text-ink-3">This scenario's extract has no site coordinates.</div>}
          <div className="mt-1 text-micro text-ink-3">Dot area grows with the site's cell count. Map drawn locally from the extract's coordinates; no map tiles are loaded.</div>
          {pr.length >= 3 && (
            <div className="mt-2 text-micro text-ink-2" data-testid="viavi-proposed">
              <span className="rounded border border-line px-1 font-semibold tracking-[0.04em] text-warn-ink">PROPOSED, NOT RUN</span>{' '}
              Future mapping: silo A, B, C ↔ {pr.slice(0, 3).map((s) => s.name.replace(/^.*?(?:-|–|:)\s*/, '')).join(', ')}.
            </div>
          )}
          <div className="mt-3 overflow-x-auto">
            <table className="w-full min-w-[420px] text-cap" data-testid="viavi-table">
              <thead><tr className="border-b border-line text-left text-micro text-ink-3">{['Site', 'Cells', 'Lat, lon', 'Map'].map((h) => <th key={h} className="px-3 py-1.5 font-medium">{h}</th>)}</tr></thead>
              <tbody>
                {current.sites.map((s, i) => (
                  <tr key={`${s.name}-${i}`} className="border-b border-line/60">
                    <td className="px-3 py-1.5 text-ink-1">{s.name}</td>
                    <td className="tnum px-3 text-ink-2">{s.cells ?? '—'}</td>
                    <td className="tnum mono px-3 text-micro text-ink-2">{located(s) ? `${s.lat.toFixed(5)}, ${s.lon.toFixed(5)}` : 'not in the extract'}</td>
                    <td className="px-3">{located(s) ? <GeoLink lat={s.lat} lon={s.lon} label="OpenStreetMap" /> : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
    </div>
  )
}
