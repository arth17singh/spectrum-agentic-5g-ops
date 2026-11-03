// Plain helpers and types for the Twin view (kept apart from components for fast refresh).

export type Prov = { state: 'empty' | 'real' | 'synthetic' | 'mixed' | 'unverified'; label: string; note: string; n_records: number; n_synthetic: number; n_unverified: number }

export type Load<T> = { state: 'loading' } | { state: 'error'; message: string } | { state: 'ok'; data: T }

export const amber = {
  color: 'var(--warn-ink)',
  borderColor: 'color-mix(in oklab, var(--warn) 55%, transparent)',
  background: 'color-mix(in oklab, var(--warn) 10%, transparent)',
}

/** Real wall-clock UTC from the record. */
export const utc = (iso?: string | null, withDate = false) =>
  !iso ? '—' : withDate ? `${iso.slice(0, 10)} ${iso.slice(11, 19)}Z` : `${iso.slice(11, 19)}Z`

export const signed = (v: unknown, digits = 3) => {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '—'
  const t = Number(v.toPrecision(digits))
  return `${t > 0 ? '+' : ''}${t}`
}

export const fixed = (v: unknown, d = 1) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : '—')
export const pct = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? `${Math.round(v * 100)}%` : '—')
