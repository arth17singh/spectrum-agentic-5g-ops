import clsx from 'clsx'
import type { CellId, Health } from './types'

/** One cell's KPIs on the preflight copy at the grade step (t3/evals/livefix_worker.side_effects). */
export interface CellSnapshot {
  power_dbm?: number | null
  tilt_deg?: number | null
  coverage_share?: number | null
  sla_share?: number | null
  sinr_p10_db?: number | null
  ue_count?: number | null
}

export interface SideEffectRow {
  cell: CellId
  without: CellSnapshot
  with: CellSnapshot
  healthWithout: Health
  healthWith: Health
  /** the change itself makes this cell worse than doing nothing (percell-v1 against "without") */
  worse: boolean
  reasons: string[]
}

export interface SideEffectsTableProps {
  rows: SideEffectRow[]
  /** what the prediction is, e.g. "preflight copy of the twin, users placed differently" */
  note?: string
  className?: string
}

const DOT: Record<Health, { bg: string; word: string }> = {
  good: { bg: 'var(--good)', word: 'green' },
  degraded: { bg: 'var(--warn)', word: 'amber' },
  critical: { bg: 'var(--critical)', word: 'red' },
  unknown: { bg: 'var(--surface-3)', word: 'unknown' },
}

function Dot({ h }: { h: Health }) {
  return (
    <span className="inline-flex items-center gap-1">
      <span className="inline-block size-2.5 rounded-full" style={{ background: DOT[h].bg }} aria-hidden />
      <span className="text-micro text-ink-2">{DOT[h].word}</span>
    </span>
  )
}

const n = (v: number | null | undefined, d: number) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : '—')

function Pair({ a, b, d, bad }: { a: number | null | undefined; b: number | null | undefined; d: number; bad?: boolean }) {
  const moved = typeof a === 'number' && typeof b === 'number' && Math.abs(a - b) >= 10 ** -d / 2
  return (
    <span className="tnum whitespace-nowrap">
      <span className="text-ink-3">{n(a, d)}</span>
      <span className="text-ink-3"> → </span>
      <span className={clsx(moved ? 'font-medium' : 'text-ink-2')} style={bad && moved ? { color: 'var(--critical-ink)' } : undefined}>{n(b, d)}</span>
    </span>
  )
}

/** Per cell: the network without the change → with it, predicted on the preflight copy of the twin, with each cell's
 *  health against its own healthy level. A row the change makes worse is flagged with the reasons. Prop-driven. */
export function SideEffectsTable({ rows, note, className }: SideEffectsTableProps) {
  const flagged = rows.filter((r) => r.worse)
  return (
    <div className={className} data-testid="side-effects">
      <table className="w-full text-cap">
        <thead>
          <tr className="text-left text-micro text-ink-3">
            <th className="font-normal">cell</th>
            <th className="font-normal">power dBm</th>
            <th className="font-normal">tilt °</th>
            <th className="font-normal">coverage</th>
            <th className="font-normal">SLA</th>
            <th className="font-normal">p10 SINR dB</th>
            <th className="font-normal">health</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.cell} className="border-t border-line align-top" data-worse={r.worse ? 'true' : undefined}
              style={r.worse ? { boxShadow: `inset 3px 0 0 ${r.healthWith === 'critical' ? 'var(--critical)' : 'var(--warn)'}` } : undefined}>
              <td className="py-1 pl-2 text-ink-2">cell {r.cell}</td>
              <td><Pair a={r.without.power_dbm} b={r.with.power_dbm} d={1} /></td>
              <td><Pair a={r.without.tilt_deg} b={r.with.tilt_deg} d={1} /></td>
              <td><Pair a={r.without.coverage_share} b={r.with.coverage_share} d={2} bad={r.worse} /></td>
              <td><Pair a={r.without.sla_share} b={r.with.sla_share} d={2} bad={r.worse} /></td>
              <td><Pair a={r.without.sinr_p10_db} b={r.with.sinr_p10_db} d={1} bad={r.worse} /></td>
              <td className="whitespace-nowrap"><Dot h={r.healthWithout} /> <span className="text-ink-3">→</span> <Dot h={r.healthWith} /></td>
            </tr>
          ))}
        </tbody>
      </table>
      {flagged.length > 0 ? (
        <ul className="mt-1 space-y-0.5 text-micro" style={{ color: 'var(--critical-ink)' }}>
          {flagged.map((r) => (
            <li key={r.cell}>Cell {r.cell} would be {DOT[r.healthWith].word} and worse than doing nothing: {r.reasons.slice(0, 3).join('; ') || 'below its healthy level'}</li>
          ))}
        </ul>
      ) : rows.length > 0 && <p className="mt-1 text-micro text-ink-3">No cell is predicted to be made worse by this change.</p>}
      {note && <p className="mt-1 text-micro text-ink-3">{note}</p>}
    </div>
  )
}
