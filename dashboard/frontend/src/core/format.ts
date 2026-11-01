// Formatting rules from DASHBOARD_SPEC.md §1.6. One place, so every screen speaks the same way.

export const shortHash = (h?: string | null, head = 6, tail = 4) =>
  !h ? '—' : h.length <= head + tail + 1 ? h : `${h.slice(0, head)}…${h.slice(-tail)}`

export function usd(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return '—'
  const a = Math.abs(v)
  if (a < 0.01 && a > 0) return `$${v.toFixed(4)}`
  if (a < 10) return `$${v.toFixed(2)}`
  if (a < 100) return `$${v.toFixed(1)}`
  if (a < 1e6) return `$${(v / 1000).toFixed(1)}K`
  return `$${(v / 1e6).toFixed(1)}M`
}

export function pct(v: number | null | undefined, fromFraction = false): string {
  if (v == null || Number.isNaN(v)) return '—'
  const x = fromFraction ? v * 100 : v
  return Math.abs(x) < 10 ? `${x.toFixed(1)}%` : `${Math.round(x)}%`
}

export function num(v: number | null | undefined, digits = 0): string {
  if (v == null || Number.isNaN(v)) return '—'
  return v.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

export function compact(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return '—'
  const a = Math.abs(v)
  if (a < 1000) return `${Math.round(v)}`
  if (a < 1e6) return `${(v / 1000).toFixed(a < 1e4 ? 1 : 0)}K`
  if (a < 1e9) return `${(v / 1e6).toFixed(1)}M`
  return `${(v / 1e9).toFixed(1)}B`
}

export function bytes(b: number | null | undefined): string {
  if (b == null) return '—'
  if (b === 0) return '0 B'
  const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  const i = Math.min(u.length - 1, Math.floor(Math.log(b) / Math.log(1024)))
  return `${(b / 1024 ** i).toFixed(i === 0 ? 0 : 1)} ${u[i]}`
}

export const eps = (e: number | null | undefined, delta = '1e-5') =>
  e == null ? '—' : `ε = ${e.toFixed(2)} (δ = ${delta})`

export function ms(v: number | null | undefined): string {
  if (v == null) return '—'
  if (v < 1) return `${Math.round(v * 1000)} µs`
  if (v < 1000) return `${Math.round(v)} ms`
  return `${(v / 1000).toFixed(v < 10000 ? 2 : 1)} s`
}

export function duration(s: number | null | undefined): string {
  if (s == null) return '—'
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return m ? `${m}m${String(sec).padStart(2, '0')}s` : `${sec}s`
}

/** "14:03:22.418" from an ISO timestamp, in the timestamp's own offset (sim time is PDT). */
export function clockTime(iso?: string, millis = true): string {
  if (!iso) return '—'
  const m = iso.match(/T(\d{2}:\d{2}:\d{2})(\.\d{3})?/)
  if (!m) return iso
  return millis ? `${m[1]}${m[2] ?? '.000'}` : m[1]
}

export function relative(simNow: number, simThen: number): string {
  const d = Math.max(0, simNow - simThen)
  if (d < 1) return 'just now'
  if (d < 60) return `${Math.floor(d)} s ago`
  if (d < 3600) return `${Math.floor(d / 60)} min ago`
  return `${Math.floor(d / 3600)} h ago`
}

/** sim seconds from an ISO timestamp produced by the backend sim clock (epoch 2026-10-31T21:30:00-07:00). */
const SIM_EPOCH_MS = Date.parse('2026-10-31T21:30:00-07:00')
export const simOf = (iso?: string) => (iso ? (Date.parse(iso) - SIM_EPOCH_MS) / 1000 : 0)

export const titleCase = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)
