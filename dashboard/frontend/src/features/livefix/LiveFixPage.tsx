import clsx from 'clsx'
import { ExternalLink } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  AttackResult, CellHealthMap, FaultPanel, FederatedForecastLine, LessonsCard, LoopTimeline,
  type AttackKey, type CellId, type FaultKey, type Lesson,
} from '@/components/livefix'
import type { Rec } from '@/core/types'
import { Button, PageHeader, Panel, Segmented } from '@/ui'
import { ApprovalCard } from './ApprovalCard'
import { cellTag, gradeLine, lf, percellParts, toCells, toStages } from './api'
import manifest from './manifest'

// Live Fix (/api/livefix): one fault, end to end. LIVE = the real kit (Gateway, Cedar, agents, approvals table) on
// the team's in-process Maveric twin; RECORDED = the judged swarm run's records, replayed. The page never approves
// on its own: a person decides in the approval card (PIN), on the Approvals page, on a paired phone or the kit CLI.

type Mode = 'live' | 'recorded'
const POLL_MS = 1500
const GREEN_HOLD_MS = 2500 // how long the healthy network stays on screen before the fault lands
const num = (v: unknown, d = 3) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : '—')

function Kpis({ h }: { h: Rec }) {
  const rows: [string, string, number][] = [['coverage_ratio', 'Coverage', 3], ['sla_ratio', 'SLA', 3], ['sinr_p10_db', 'p10 SINR (dB)', 2]]
  const now = h?.final ?? h?.faulted
  const ref = h?.final ? h?.healthy_final : h?.healthy
  return (
    <table className="w-full text-cap">
      <thead><tr className="text-left text-micro text-ink-3"><th className="font-normal">KPI</th><th className="font-normal">healthy</th><th className="font-normal">{h?.final ? 'after the fix' : 'now'}</th></tr></thead>
      <tbody>
        {rows.map(([k, label, d]) => {
          const bad = now && ref && typeof now[k] === 'number' && typeof ref[k] === 'number' && now[k] < ref[k] - (k === 'sinr_p10_db' ? 0.5 : 0.02)
          return (
            <tr key={k} className="border-t border-line">
              <td className="py-1 text-ink-2">{label}</td>
              <td className="tnum text-ink-1">{num(ref?.[k], d)}</td>
              <td className="tnum font-medium" style={{ color: now ? (bad ? 'var(--critical-ink)' : 'var(--good-ink, var(--ink-1))') : 'var(--ink-3)' }}>{num(now?.[k], d)}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

/** After a change whose per-cell check failed (state "review"): ask the agents to correct the degraded cells (the
 *  correction stops at Cedar's hold and a person's approval again) or finish the run as graded. */
function ReviewPanel({ run, onError }: { run: Rec; onError: (m: string) => void }) {
  const [busy, setBusy] = useState<string | null>(null)
  const pc = run.percell ?? {}
  const choose = (choice: 'correct' | 'finish') => {
    setBusy(choice)
    lf(`/runs/${encodeURIComponent(run.run_id)}/review`, { choice }).catch((e) => onError(String((e as Error).message))).finally(() => setBusy(null))
  }
  return (
    <div role="status" className="mb-4 rounded-[10px] border-2 p-3 text-cap" data-testid="livefix-review"
      style={{ borderColor: 'color-mix(in oklab, var(--warn) 65%, transparent)' }}>
      <p className="font-medium text-ink-1">
        {run.grade?.success ? 'The network KPIs recovered, but per cell it is not done: ' : 'The network KPIs did not recover, and per cell: '}
        {percellParts(pc).join(' and ')}.
      </p>
      <p className="mt-0.5 text-micro text-ink-3">“Ask the agents to correct” sends noc_lead the degraded cells’ symptoms on the same case; whatever it proposes
        stops at Cedar’s hold and needs a person’s approval again. “Finish” ends the run as graded (not recovered per cell).</p>
      <div className="mt-2 flex flex-wrap gap-2">
        <Button variant="primary" size="sm" onClick={() => choose('correct')} disabled={!!busy || !run.can_follow_up} data-testid="ask-correct">Ask the agents to correct</Button>
        <Button size="sm" onClick={() => choose('finish')} disabled={!!busy}>Finish</Button>
        {!run.can_follow_up && <span className="text-micro text-ink-3">No follow-ups or changes left in this run.</span>}
      </div>
    </div>
  )
}

function lessonsOf(run: Rec | null): Lesson[] {
  const out: Lesson[] = (run?.lessons?.hits ?? []).map((h: Rec) => ({
    id: h.lesson_id, symptoms: `similarity ${num(h.similarity, 2)}`, change: h.pattern_text, source: h.source ?? 'practice', verified: true,
  }))
  const l = run?.lesson_learned
  if (l?.status) out.push({ id: l.lesson_id ?? 'this run', symptoms: 'this incident', change: l.pattern_text ?? '—', source: 'this run (twin-verified)', verified: l.status === 'verified', reason: l.reason ?? undefined })
  return out
}

export default function LiveFixPage() {
  const [mode, setMode] = useState<Mode>('live')
  const [inc, setInc] = useState<Rec | null>(null)
  const [cell, setCell] = useState<CellId>(2)
  const [run, setRun] = useState<Rec | null>(null)
  const [act, setAct] = useState<Rec | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState<FaultKey | AttackKey | null>(null)
  const autoInject = useRef<string | null>(null)

  useEffect(() => { lf('/incidents').then(setInc).catch((e) => setErr(String(e.message ?? e))) }, [])
  useEffect(() => {
    lf<Rec>('/runs').then((r) => { const last = (r.runs ?? []).at(-1); if (last && !last.end) setRun(last) }).catch(() => {})
  }, [])

  const poll = useCallback(async (id: string) => {
    try { setRun(await lf(`/runs/${encodeURIComponent(id)}`)) } catch (e) { setErr(String((e as Error).message)) }
  }, [])
  const attacksPending = (run?.attacks?.length ?? 0) < (run?.attacks_requested ?? 0)
  useEffect(() => {
    if (!run?.run_id || (run.end && !attacksPending)) return
    const t = setInterval(() => { if (!document.hidden) poll(run.run_id) }, POLL_MS)
    return () => clearInterval(t)
  }, [run?.run_id, run?.end, attacksPending, poll])
  // the fault lands a moment after the healthy network is on screen
  useEffect(() => {
    if (!run || run.state !== 'healthy' || autoInject.current !== run.run_id) return
    const t = setTimeout(() => {
      autoInject.current = null
      lf(`/runs/${encodeURIComponent(run.run_id)}/inject`, {}).then(setRun).catch((e) => setErr(String(e.message)))
    }, GREEN_HOLD_MS)
    return () => clearTimeout(t)
  }, [run])
  useEffect(() => {
    if (!run?.run_id || run.mode !== 'live' || !run.cases?.faulted) return
    const get = () => lf(`/runs/${encodeURIComponent(run.run_id)}/activity`).then(setAct).catch(() => {})
    get()
    if (run.end) return
    const t = setInterval(() => { if (!document.hidden) get() }, 6000)
    return () => clearInterval(t)
  }, [run?.run_id, run?.mode, run?.cases?.faulted, run?.end])

  const running = !!run && !run.end
  const onInject = async (fault: FaultKey, c: CellId) => {
    setBusy(fault); setErr(null); setAct(null)
    try {
      const r = await lf('/runs', { fault_key: fault, cell: c, mode })
      autoInject.current = r.run_id
      setRun(r)
    } catch (e) { setErr(String((e as Error).message)) } finally { setBusy(null) }
  }
  const onAttack = async (key: AttackKey) => {
    if (!run) return
    setBusy(key); setErr(null)
    try { setRun({ ...(await lf(`/runs/${encodeURIComponent(run.run_id)}/attack`, { key })), attacks_requested: (run.attacks?.length ?? 0) + 1 }) }
    catch (e) { setErr(String((e as Error).message)) } finally { setBusy(null) }
  }
  const cont = () => run && lf(`/runs/${encodeURIComponent(run.run_id)}/continue`, {}).then(setRun).catch((e) => setErr(String(e.message)))
  const label = run?.label ?? (mode === 'live' ? 'LIVE' : 'RECORDED')
  const faultCells: CellId[] = run && !['healthy', 'preparing'].includes(run.state) ? (run.fault?.fault_cells ?? run.incident?.fault_cells ?? []) : []
  const fedEvents = (run?.events ?? []).filter((e: Rec) => e.kind === 'federated')
  const liveBlocked = mode === 'live' && inc && (!inc.live_ready || !inc.faults_ready)

  return (
    <div className="min-w-0" data-slice="livefix">
      <PageHeader title={manifest.title} description={manifest.description} tracks={manifest.tracks}
        actions={<span data-testid="livefix-mode" className="rounded border px-2 py-0.5 text-cap font-semibold"
          style={{ color: label === 'LIVE' ? 'var(--good-ink, var(--ink-1))' : 'var(--warn-ink)', borderColor: 'currentColor' }}
          title={label === 'LIVE' ? 'Real calls to the kit and the twin' : 'A replay of recorded runs: no AWS call'}>{label}</span>} />
      {err && <div role="alert" className="mb-3 rounded-md border border-critical/60 bg-critical/10 p-3 text-cap text-ink-1">{err}</div>}

      <div className="mb-3 flex flex-wrap items-center gap-3">
        <Segmented<Mode> ariaLabel="Mode" value={mode} onChange={(m) => { if (!running) setMode(m) }}
          options={[{ value: 'live', label: 'Live', title: 'Real calls' }, { value: 'recorded', label: 'Recorded', title: 'Fallback: replay the judged run' }]} />
        {liveBlocked && <span className="text-micro" style={{ color: 'var(--warn-ink)' }}>
          Live needs {!inc?.live_ready ? 'the kit folder and an AWS session' : 'the twin-side helper (LIVEFIX_T3_PYTHON)'} on this laptop; Recorded still works.</span>}
        {mode === 'recorded' && <span className="text-micro text-ink-3">Recorded: the judged run's incident with the same fault (the cell pick is ignored).</span>}
      </div>
      <FaultPanel className="mb-3" onInject={onInject} onAttack={onAttack} cell={cell} onCellChange={setCell}
        busy={busy} disabled={running || !!liveBlocked} />
      {run && mode === 'live' && run.mode === 'live' && run.cases?.healthy && (
        <p className="-mt-1 mb-3 text-micro text-ink-3">Attacks run on this run's case through the Gateway and are expected to be refused.
          {run.end ? '' : ' (They can be sent while the run is in progress.)'}</p>
      )}

      {run && <div className="mb-3 rounded-[10px] border border-line bg-s1 px-3 py-3"><LoopTimeline stages={toStages(run)} /></div>}
      {run?.state === 'review' && <ReviewPanel run={run} onError={setErr} />}
      {run?.approval && <div className="mb-4"><ApprovalCard run={run} act={act} /></div>}
      {run?.mode === 'recorded' && run.state === 'awaiting_approval' && (
        <button type="button" onClick={cont} className="mb-4 rounded-md border border-line px-3 py-1.5 text-cap text-ink-1">Continue (recorded)</button>
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <Panel title="The network" subtitle={run?.twin ?? 'Pick a fault and a cell to start'} className="lg:col-span-5">
          <CellHealthMap cells={toCells(run?.cells, run?.state === 'healthy' ? null : run?.cells_at_proposal)} faultCells={faultCells} />
          {run?.fault?.fault_label && <p className="mt-1 text-micro text-ink-3">For the audience: {run.fault.fault_label} on cell {run.fault.chosen_cell} ({run.fault.fault_about}). The agents get symptoms only, under an opaque label.</p>}
          {run?.headline && <div className="mt-3"><Kpis h={run.headline} /></div>}
          {run?.grade && (() => {
            const g = gradeLine(run)
            return (
              <p className="mt-2 text-cap font-medium" data-testid="livefix-grade" data-cells-ok={g.cellsOk === null ? 'na' : String(g.cellsOk)}
                style={{ color: g.ok && g.cellsOk !== false ? 'var(--good-ink, var(--ink-1))' : g.cellsOk === false && run.grade.success ? 'var(--warn-ink)' : 'var(--critical-ink)' }}>
                {g.text}
              </p>
            )
          })()}
          {run?.percell && !run.percell.recovered && (
            <ul className="mt-1 space-y-0.5 text-micro text-ink-3">
              {(run.percell.cells ?? []).filter((c: Rec) => c.health !== 'green').map((c: Rec) => (
                <li key={c.cell}>cell {c.cell} {c.health}, {cellTag(run.percell, c)}: {(c.reasons ?? []).filter((r: string) => !r.includes('not counted')).slice(0, 3).join('; ')}</li>
              ))}
            </ul>
          )}
        </Panel>

        <div className="space-y-4 lg:col-span-7">
          {(run?.attacks ?? []).map((a: Rec, i: number) => (
            <div key={i}>
              <AttackResult attack={a.attack} decision={a.decision === 'DENY' ? 'DENY' : a.decision === 'ALLOW' ? 'ALLOW' : 'ERROR'} policy={a.policy}
                agent={a.caller} tool={a.tool} awsTime={a.aws_time} awsTimeSource={a.aws_time ? 'gateway:http-date' : null} excerpt={a.text_excerpt} caseId={a.case_id} />
              <p className="mt-0.5 text-micro text-ink-3">{a.judged ? `Same call as judged case ${a.suite_id}.` : 'Not a judged case: this refusal is this call’s own outcome.'}</p>
            </div>
          ))}
          <Panel title="What happened" subtitle={run ? `run ${run.run_id}` : undefined}
            actions={run && <Link to={run.evidence_link} className="inline-flex items-center gap-1 text-cap text-ink-2 underline decoration-dotted"><ExternalLink size={13} />Open evidence</Link>}>
            {!run && <p className="text-cap text-ink-3">Pick a fault and a cell. The healthy network shows first, then the fault lands.</p>}
            <ol className="space-y-1 text-cap">
              {(run?.events ?? []).filter((e: Rec) => e.kind !== 'federated').map((e: Rec, i: number) => (
                <li key={i} className={clsx('flex gap-2', e.stage === 'failed' && 'text-critical-ink')}>
                  <span className="tnum shrink-0 text-micro text-ink-3">{String(e.t).slice(11, 19)}</span><span className="text-ink-1">{e.text}</span>
                </li>
              ))}
            </ol>
            {run?.problem && <p role="alert" className="mt-2 text-cap" style={{ color: 'var(--critical-ink)' }}>{run.problem}</p>}
            {run?.reply && <details className="mt-2"><summary className="cursor-pointer text-cap text-ink-2">noc_lead's last reply</summary><pre className="mt-1 text-micro whitespace-pre-wrap text-ink-2">{run.reply}</pre></details>}
            {run?.evidence && Object.keys(run.evidence).some((k) => k.endsWith('_error')) && (
              <p className="mt-2 text-micro" style={{ color: 'var(--warn-ink)' }}>Evidence: {Object.entries(run.evidence).filter(([k]) => k.endsWith('_error')).map(([k, v]) => `${k} ${v}`).join('; ')}</p>
            )}
          </Panel>
          {run?.mode === 'live' && (
            <Panel title="The agents' governed calls" subtitle={act?.source ?? 'the kit audit log in CloudWatch'}>
              {fedEvents.map((e: Rec, i: number) => (
                <FederatedForecastLine key={i} className="mb-1" predicted={e.federated?.signal ?? 0} change={run.approval?.change_text} actor="confidence check" operators={3} />
              ))}
              {act?.lag_note && <p className="mb-1 text-micro text-ink-3">{act.lag_note}</p>}
              {!act?.events?.length && <p className="text-cap text-ink-3">{act?.message ?? 'No calls logged yet.'}</p>}
              <ol className="max-h-[300px] space-y-0.5 overflow-y-auto text-cap">
                {(act?.events ?? []).map((e: Rec, i: number) => (
                  <li key={i} className="flex flex-wrap gap-x-2">
                    <span className="tnum text-micro text-ink-3">{String(e.time).slice(11, 19)}</span>
                    <span className="text-ink-2">{e.agent ?? e.by ?? e.kind}</span>
                    <span className="font-mono text-micro text-ink-1">{e.tool ?? e.kind}{e.to ? ` → ${e.to}` : ''}</span>
                    {e.decision && <span className="text-micro font-semibold" style={{ color: e.decision === 'DENY' ? 'var(--critical-ink)' : 'var(--ink-2)' }}>{e.decision}{e.rule ? ` (${e.rule})` : ''}{e.approval ? ` · ${e.approval}` : ''}</span>}
                  </li>
                ))}
              </ol>
            </Panel>
          )}
          {run?.mode === 'live' && (lessonsOf(run).length > 0 || run.lessons) && (
            <Panel title="Lessons" subtitle="verified fixes from practice incidents, replayed on the twin; offline / synthetic evaluation">
              <LessonsCard lessons={lessonsOf(run)} synthetic />
              {run.lessons && <p className="mt-1 text-micro text-ink-3">{run.lessons.n_lessons} lesson(s) in the store; the self-improvement unit (offline, synthetic stand-in) FAILED narrowly (CI low −0.002).</p>}
            </Panel>
          )}
          <p className="text-micro text-ink-3">Agents: {run?.models ?? '—'}. The page never decides: a person approves with the PIN, here, on the Approvals page, on a paired phone or with the kit CLI.</p>
        </div>
      </div>
    </div>
  )
}
