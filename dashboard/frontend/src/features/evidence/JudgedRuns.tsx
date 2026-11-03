import clsx from 'clsx'
import { CircleDashed } from 'lucide-react'
import type { Rec } from '@/core/types'
import { Panel } from '@/ui'
import { checkWords } from './checkWords'
import { VerifyLive } from './VerifyLive'
import { RunLink, Superseded, Verdict } from '../scorecard/RealRuns'
import { num, targetText } from '../scorecard/format'
import { useOverview } from '../scorecard/overview'

// The curated judged layout from judging.json (10 slots + edge cases), with each slot's live status read from
// the mirrored result records: filled (PASS / MISS recomputed against the declared target) or pending.

const RunState = ({ r, verify = true }: { r: Rec; verify?: boolean }) => (
  <div className="flex flex-wrap items-baseline gap-x-2" data-run={r.run_id} data-state={r.state}>
    {r.state === 'missing'
      ? <span className="text-micro text-warn-ink">listed, but not in the evidence mirror</span>
      : r.state === 'declared' ? <span className="text-micro text-ink-3">declared, no result yet</span>
      : <><Verdict s={r.state} why={r.why} /><span className="tnum text-micro text-ink-1">{num(r.value)} {r.metric}</span><span className="text-micro text-ink-3">target {targetText(r.target)}</span></>}
    <RunLink id={r.run_id} />
    {r.check != null && <span className="text-micro text-ink-3">checked by check_unit: {checkWords(r.check, r.check_parts)}</span>}
    {r.superseded && <Superseded reason={r.superseded.reason} by={r.superseded.by} />}
    {verify && r.state !== 'missing' && <div className="basis-full"><VerifyLive runId={r.run_id} compact /></div>}
  </div>
)

function Slot({ s }: { s: Rec }) {
  const filled = (s.runs ?? []).length > 0
  return (
    <li className={clsx('min-w-0 rounded-md border px-3 py-2', filled ? 'border-line' : 'border-dashed border-line')} data-slot={s.slot} data-filled={filled ? 'true' : 'false'}>
      <div className="flex items-baseline gap-2">
        <span className="tnum w-6 shrink-0 text-cap font-semibold text-ink-3">{s.slot}</span>
        <span className="text-cap font-medium text-ink-1">{s.name}</span>
      </div>
      <div className="mt-1 min-w-0 space-y-1 pl-8">
        {filled ? (s.runs as Rec[]).map((r) => <RunState key={r.run_id} r={r} />)
          : <div className="flex items-start gap-1.5 text-micro text-ink-3"><CircleDashed size={12} className="mt-0.5 shrink-0" aria-hidden />pending{s.waiting_for ? `: ${s.waiting_for}` : ''}</div>}
        {s.note && <div className="text-micro text-ink-3">{s.note}</div>}
        {s.bar && <div className="text-micro text-ink-3">{s.bar}</div>}
        {(s.related_status ?? []).map((r: Rec) => <div key={r.run_id} className="text-micro text-ink-3">compared against: <RunState r={r} verify={false} /></div>)}
      </div>
    </li>
  )
}

export default function JudgedRuns() {
  const { ov, err } = useOverview()
  const j: Rec | null | undefined = ov?.judged
  return (
    <Panel id="judged-runs" plane="prove" className="mt-4" title="Judged runs"
      subtitle={j ? `${j.filled} of ${j.total} slots filled · primary ${j.primary_track ?? '—'} · judging.json` : 'judging.json'}>
      {err && !ov && <div className="text-cap text-critical-ink">Unavailable: <span className="mono">{err}</span></div>}
      {ov && !j && <div className="text-cap text-ink-2">No judging.json found. Copy it from the hand-in branch with <span className="mono">make evidence-sync HANDIN=&lt;branch&gt;</span>.</div>}
      {j && (
        <div className="grid min-w-0 gap-4 [overflow-wrap:anywhere] xl:grid-cols-[2fr_1fr]" data-testid="judged-runs">
          <ol className="min-w-0 space-y-2">{(j.slots as Rec[]).map((s) => <Slot key={String(s.slot)} s={s} />)}</ol>
          <div className="min-w-0">
            <div className="mb-2 text-cap font-medium text-ink-2">Edge cases</div>
            <ol className="space-y-2">{(j.edge_cases as Rec[]).map((s) => <Slot key={String(s.slot)} s={s} />)}</ol>
          </div>
        </div>
      )}
    </Panel>
  )
}
