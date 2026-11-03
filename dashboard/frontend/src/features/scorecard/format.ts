// Pure formatting for the Scorecard. Targets are shown as they were written in the declaration.

export type UnitState = 'pass' | 'fail' | 'not checked' | 'inconsistent'

/** data-state values: stable strings for tests and styling. */
export const stateAttr = (s: string): string =>
  s === 'pass' || s === 'fail' || s === 'inconsistent' || s === 'unverified' || s === 'incomplete' || s === 'superseded' ? s
    : s === 'not checked' ? 'not-checked' : s === 'check failed' ? 'check-failed' : 'inconsistent'

const OP: Record<string, string> = { '>=': '≥', '<=': '≤', '==': '=', '>': '>', '<': '<' }

export const num = (v: unknown): string => {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '—'
  return String(Number(v.toPrecision(6)))
}

/** Declared unit as a suffix: "x" becomes "×", "%" stays tight, plain fractions and ratios carry none. */
const suffix = (unit?: string | null): string => {
  if (!unit || unit === 'fraction' || unit === 'ratio' || unit === 'count') return ''
  if (unit === 'x' || unit === '×') return '×'
  if (unit === '%') return '%'
  return ` ${unit}`
}

export const value = (v: unknown, unit?: string | null): string => (typeof v === 'number' && Number.isFinite(v) ? `${num(v)}${suffix(unit)}` : '—')

/** "≥ 1.05×" */
export const targetText = (t?: { op?: string; value?: unknown; unit?: string | null } | null): string =>
  !t ? 'no valid target' : `${OP[t.op ?? ''] ?? t.op ?? '?'} ${num(t.value)}${suffix(t.unit)}`

/** "1.3 (95% CI 1.1 to 1.5)" */
export const gainText = (g?: { value?: unknown; ci_low?: unknown; ci_high?: unknown } | null): string =>
  !g ? '' : `${num(g.value)}  [${num(g.ci_low)}, ${num(g.ci_high)}]`

/** Real wall-clock UTC. */
export const utc = (iso?: string | null): string => (!iso ? '—' : `${iso.slice(0, 10)} ${iso.slice(11, 19)}Z`)
