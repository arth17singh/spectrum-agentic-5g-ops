import type { Rec } from '@/core/types'
import type { CellId, CellState, ConfidenceBreakdown, ConfidenceTier, Health, Stage, StageState } from '@/components/livefix'

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
  return order.map(([key, s]) => {
    let state: StageState = 'pending'
    if (key === 'evidence') state = run?.evidence?.result ? 'done' : run?.end && run?.mode === 'live' ? 'failed' : run?.end ? 'skipped' : 'pending'
    else if (key === 'recovered') state = st === 'recovered' ? 'done' : failedAt ? 'failed' : 'pending'
    else if (reached.has(s)) state = s === st && !run?.end ? 'active' : failedAt && s === lastReached && failedAt !== 'not_recovered' ? 'failed' : 'done'
    return { key, state, elapsedS: secs[s] ?? null, detail: key === 'evidence' && run?.evidence?.result ? 'result written' : undefined }
  })
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
