import type { Rec } from '@/core/types'
import { Panel } from '@/ui'
import { RunLink, Verdict } from '../scorecard/RealRuns'
import { useOverview } from '../scorecard/overview'

// The Viavi rung: one measure-only run of our PPO policy on the team's Viavi RSG twin (unit t1.rl.viavi_operate),
// read from the evidence mirror. It has no AWS job or AWS-written log, so check_unit part c is expected to say
// CANNOT CHECK; that is shown as it is, never as a pass. Per-step items are shown as recorded.

const PART_COLOR: Record<string, string> = { PASS: 'var(--good-ink)', FAIL: 'var(--critical-ink)', 'CANNOT CHECK': 'var(--warn-ink)', SKIP: 'var(--ink-3)' }

const pick = (it: Rec, keys: string[]) => {
  for (const k of keys) if (it[k] != null) return it[k]
  const ex = it.extra ?? {}
  for (const k of keys) if (ex[k] != null) return ex[k]
  return null
}
const show = (v: unknown) => (v == null ? '—' : typeof v === 'number' ? String(Number(v.toPrecision(5))) : typeof v === 'string' ? v : JSON.stringify(v))

export function ViaviRung() {
  const { ov } = useOverview()
  const v: Rec | null | undefined = ov ? (ov as Rec).viavi_operate : undefined
  return (
    <Panel id="viavi-rung" plane="operate" title="Viavi rung: measure-only operate run" subtitle="our PPO policy played on the team's Viavi RSG twin; no training on Viavi">
      {v === undefined && <div className="text-cap text-ink-3">Loading…</div>}
      {v === null && <div className="rounded-md border border-dashed border-line p-4 text-center text-cap text-ink-3">Viavi operate run: not run yet.</div>}
      {v && (
        <div className="space-y-2" data-testid="viavi-rung">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-cap">
            <RunLink id={v.run_id} />
            <span className="text-micro text-ink-3">model {v.model_id} · declared {String(v.declared_at ?? '').slice(0, 19).replace('T', ' ')}Z · target {v.target?.metric} {v.target?.op} {v.target?.value}</span>
          </div>
          {v.result
            ? <div className="flex flex-wrap items-baseline gap-2"><Verdict s={v.result.state} why={v.result.why} /><span className="tnum text-cap text-ink-1">{v.result.value} {v.result.metric}</span></div>
            : <div className="text-cap text-warn-ink">Running: declared, no result yet ({v.items?.length ?? 0} of {v.n_items_declared ?? '?'} step records so far).</div>}
          <div className="flex flex-wrap gap-x-3 gap-y-1 text-micro">
            {v.check ? Object.entries(v.check.parts as Record<string, string>).map(([k, s]) => (
              <span key={k} style={{ color: PART_COLOR[s] ?? 'var(--ink-3)' }}>[{k}] {s}{k === 'c' && s === 'CANNOT CHECK' ? ' (expected: no AWS job or AWS-written log for this run)' : ''}</span>
            )) : <span className="text-ink-3">check_unit not cached yet. Part c is expected to say CANNOT CHECK: this run has no AWS window.</span>}
          </div>
          {(v.items ?? []).length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[620px] text-cap" data-testid="viavi-rung-steps">
                <thead><tr className="border-b border-line text-left text-micro text-ink-3">{['Step', 'Requested action', 'Filtered action', 'Rules fired', 'Objective', 'KPIs'].map((h) => <th key={h} className="px-2 py-1.5 font-medium">{h}</th>)}</tr></thead>
                <tbody>
                  {(v.items as Rec[]).map((it, i) => (
                    <tr key={i} className="border-b border-line/60 align-top">
                      <td className="mono px-2 py-1 text-micro text-ink-2">{show(it.item_id ?? i)}</td>
                      <td className="mono px-2 text-micro text-ink-1 break-all">{show(pick(it, ['requested_action', 'action_requested', 'requested']))}</td>
                      <td className="mono px-2 text-micro text-ink-1 break-all">{show(pick(it, ['filtered_action', 'action_filtered', 'action_applied', 'filtered']))}</td>
                      <td className="mono px-2 text-micro text-ink-2 break-all">{show(pick(it, ['rules_fired', 'rules']))}</td>
                      <td className="tnum px-2 text-micro text-ink-1">{show(pick(it, ['objective', 'score']))}</td>
                      <td className="mono px-2 text-micro text-ink-3 break-all">{show(pick(it, ['kpis', 'kpi']))}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </Panel>
  )
}
