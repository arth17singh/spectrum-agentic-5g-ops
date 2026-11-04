import type { Rec } from '@/core/types'
import type { CellId, CellState, ConfidenceBreakdown, ConfidenceTier, Health, SideEffectRow, Stage, StageState } from '@/components/livefix'

// Client for /api/livefix (laptop only) and the one place the screen maps backend shapes to the Live Fix components.

export async function lf<T = Rec>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`/api/livefix${path}`, body === undefined ? { cache: 'no-store' } : {
    method: 'POST', cache: 'no-store',
    headers: { 'Content-Type': 'application/json', 'X-Spectrum-Client': 'livefix' }, body: JSON.stringify(body),
  })
  const data = await res.json().catch(() => null)
  if (!res.ok) throw new Error(data?.detail?.message ?? `The backend answered ${res.status}.`)
  return data as T
}

const HEALTH: Record<string, Health> = { green: 'good', amber: 'degraded', red: 'critical' }

/** Backend cell rows (twin per-cell KPIs + status green/amber/red) -> CellHealthMap cells. */
export function toCells(rows: Rec[] | undefined, before?: Rec[] | null): CellState[] {
  const was = Object.fromEntries((before ?? []).map((c) => [c.cell, c]))
  return (rows ?? []).filter((c) => typeof c.cell === 'number').map((c) => {
    const b = was[c.cell]
    return {
      id: c.cell as CellId,
      power: typeof c.power_dbm === 'number' ? c.power_dbm : 0,
      tilt: typeof c.tilt_deg === 'number' ? c.tilt_deg : 0,
      health: c.health ?? HEALTH[c.status] ?? 'unknown',
      kpis: { coverage: c.coverage_share ?? null, sla: c.sla_share ?? null },
      before: b ? { power: b.power_dbm, tilt: b.tilt_deg, kpis: { coverage: b.coverage_share ?? null, sla: b.sla_share ?? null } } : undefined,
    }
  })
}

const END_FAIL = new Set(['rejected', 'no_proposal', 'failed', 'timed_out', 'not_recovered'])

/** The run's stage list for LoopTimeline. */
export function toStages(run: Rec | null): Stage[] {
  const st: string = run?.state ?? 'idle'
  const reached = new Set<string>((run?.stages ?? []).map((s: Rec) => String(s.stage)))
  const secs: Record<string, number> = Object.fromEntries((run?.stages ?? []).map((s: Rec) => [s.stage, s.seconds]))
  const order: [Stage['key'], string][] = [['green', 'healthy'], ['fault', 'faulted'], ['diagnosing', 'diagnosing'],
    ['awaiting_approval', 'awaiting_approval'], ['applied', 'applying'], ['recovered', 'recovered'], ['evidence', 'evidence']]
  const failedAt = END_FAIL.has(st) ? st : null
  const lastReached = [...order].reverse().find(([, s]) => reached.has(s))?.[1]
  // Network KPIs back but a cell still degraded (percell-v1): the Recovered stage is amber, not green.
  const cellsBad = run?.percell?.recovered === false
  const netOk = run?.grade?.success === true
  return order.map(([key, s]) => {
    let state: StageState = 'pending'
    let detail: string | undefined
    if (key === 'evidence') state = run?.evidence?.result ? 'done' : run?.end && run?.mode === 'live' ? 'failed' : run?.end ? 'skipped' : 'pending'
    else if (key === 'recovered') {
      state = st === 'recovered' ? 'done' : st === 'review' || (cellsBad && netOk) ? 'warn' : failedAt ? 'failed' : 'pending'
      if (cellsBad) detail = `per cell: ${percellParts(run?.percell).join(', ')}`
    } else if (reached.has(s)) state = s === st && !run?.end ? 'active' : failedAt && s === lastReached && failedAt !== 'not_recovered' ? 'failed' : 'done'
    if (key === 'evidence' && run?.evidence?.result) detail = 'result written'
    return { key, state, elapsedS: secs[s] ?? null, detail }
  })
}

/** "cell 3" / "cells 0, 3" */
export function cellList(cells: unknown): string {
  const c = Array.isArray(cells) ? cells : []
  return c.length === 1 ? `cell ${c[0]}` : `cells ${c.join(', ')}`
}

/** percell-v1's two kinds of degraded cell: still degraded by the fault ("not restored") and made worse by the change
 *  ("newly degraded": collateral, never a faulted cell). */
export function percellParts(pc: Rec | null | undefined): string[] {
  const coll: number[] = pc?.collateral ?? []
  const notRestored = (pc?.degraded ?? []).filter((c: number) => !coll.includes(c))
  return [
    ...(notRestored.length ? [`${cellList(notRestored)} not restored`] : []),
    ...(coll.length ? [`${cellList(coll)} newly degraded by the change`] : []),
  ]
}

/** The result line: the network grade next to the per-cell verdict (percell-v1). */
export function gradeLine(run: Rec): { text: string; ok: boolean; cellsOk: boolean | null } {
  const g = run.grade ?? {}
  const net = g.success ? `Network KPIs: recovered${typeof g.recovery === 'number' ? ` ${g.recovery.toFixed(2)}` : ''}` : 'Network KPIs: NOT recovered'
  const pc = run.percell
  if (!pc) {
    const why = run.percell_declared ? 'per-cell verdict not available (graded on the network rule)' : 'per-cell grader not running'
    return { text: `${net} · ${why}`, ok: !!g.success, cellsOk: null }
  }
  if (pc.recovered) return { text: `${net} · Per cell: recovered (every cell back to its healthy level)`, ok: !!g.success, cellsOk: true }
  const over = pc.overshoot?.power_above_healthy_db
  return {
    text: `${net} · Per cell: NOT recovered: ${percellParts(pc).join(' and ')}`
      + (typeof over === 'number' && over > 0.05 ? ` · ${over.toFixed(1)} dB of power above healthy` : ''),
    ok: false, cellsOk: false,
  }
}

/** One cell's percell-v1 status in words. */
export function cellTag(pc: Rec | null | undefined, c: Rec): string {
  if ((pc?.collateral ?? []).includes(c.cell)) return 'newly degraded by the change'
  return c.worse_than_before ? 'made worse by the change' : 'not restored'
}

const HEALTH_OF: Record<string, Health> = { green: 'good', amber: 'degraded', red: 'critical' }

/** The worker's side_effects cells -> SideEffectsTable rows. */
export function toSideEffects(se: Rec | null | undefined): SideEffectRow[] {
  return (se?.cells ?? []).filter((c: Rec) => typeof c.cell === 'number').map((c: Rec) => ({
    cell: c.cell as CellId, without: c.without ?? {}, with: c.with ?? {},
    healthWithout: HEALTH_OF[c.health_without] ?? 'unknown', healthWith: HEALTH_OF[c.health_with] ?? 'unknown',
    worse: !!c.worse, reasons: Array.isArray(c.reasons) ? c.reasons.map(String) : [],
  }))
}

const TIERS: Record<string, ConfidenceTier> = { one_person: 'one_person', two_person: 'two_person', hold: 'hold' }

export function toConfidence(c: Rec, lessons: Rec | null): { score: number; tier: ConfidenceTier; breakdown: ConfidenceBreakdown } {
  const b = c.breakdown ?? {}
  const hit = (lessons?.hits ?? []).find((h: Rec) => h.similarity >= 0.3)
  return {
    score: typeof c.score === 'number' ? c.score : 0,
    tier: TIERS[c.tier] ?? 'hold',
    breakdown: {
      preflight: b.preflight ? { passed: !!b.preflight.ok, predicted: b.preflight.deltas?.coverage_ratio ?? null, metric: 'Δcov' } : null,
      federated: typeof b.federated?.signal === 'number' ? b.federated.signal : null,
      lesson: hit ? hit.lesson_id : null,
    },
  }
}
