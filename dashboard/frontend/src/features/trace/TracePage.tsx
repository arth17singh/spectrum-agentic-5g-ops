import clsx from 'clsx'
import {
  Ban, CircleCheck, CircleX, Clock3, FlaskConical, PauseCircle, RefreshCw, Rocket, TriangleAlert, UserCheck,
} from 'lucide-react'
import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { get } from '@/core/api'
import type { Rec } from '@/core/types'
import { Button, CodeBlock, CopyText, EmptyState, PageHeader, Panel, RealChip, StatusPill } from '@/ui'
import manifest from './manifest'

// Everything here is read from GET /api/trace/cases and /api/trace/case/{id}, which rebuild the cases from the
// persisted decision, control and change records. When there are no records the page says so; it never fills in.

const POLL_MS = 10_000

/** Real wall-clock UTC from the record, never the demo clock. */
const utc = (iso?: string | null, withDate = false) =>
  !iso ? '—' : withDate ? `${iso.slice(0, 10)} ${iso.slice(11, 19)}Z` : `${iso.slice(11, 19)}Z`

const delta = (v: unknown) => {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '—'
  const t = Number(v.toPrecision(4))
  return `${t > 0 ? '+' : ''}${t}`
}

const caseKey = (c: Rec) => `${c.run_id ?? ''}\u0000${c.case_id}`

type Prov = { state: 'empty' | 'real' | 'synthetic' | 'mixed' | 'unverified'; label: string; note: string; claim: string; n_records: number; n_synthetic: number; n_unverified: number }

/** REAL (green) only when every displayed record has an AWS time and none is synthetic; anything else is amber. */
function ProvenanceChip({ p }: { p: Prov }) {
  if (p.state === 'empty') return null
  if (p.state === 'real') return <RealChip title={`${p.note} Read from the persisted evidence records.`} />
  return (
    <span data-provenance={p.state} title={p.note}
      className="flex items-center gap-1 rounded border px-1.5 py-0.5 text-micro font-semibold tracking-[0.04em]"
      style={{ color: 'var(--warn-ink)', borderColor: 'color-mix(in oklab, var(--warn) 55%, transparent)', background: 'color-mix(in oklab, var(--warn) 10%, transparent)' }}>
      <FlaskConical size={11} aria-hidden className="shrink-0" />
      {p.label}{p.state === 'mixed' || p.state === 'unverified' ? ` · ${p.note.replace(/\.$/, '')}` : ''}
    </span>
  )
}

type Load<T> = { state: 'loading' } | { state: 'error'; message: string } | { state: 'ok'; data: T }

export default function TracePage() {
  const [cases, setCases] = useState<Load<Rec>>({ state: 'loading' })
  const [sel, setSel] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const fetchCases = useCallback(() =>
    get<Rec>('/api/trace/cases')
      .then((data) => setCases({ state: 'ok', data }))
      .catch((e) => setCases({ state: 'error', message: String(e?.message ?? e) })), [])
  const refresh = () => { setBusy(true); fetchCases().finally(() => setBusy(false)) }

  useEffect(() => {
    fetchCases()
    const t = setInterval(fetchCases, POLL_MS)
    return () => clearInterval(t)
  }, [fetchCases])

  const rows: Rec[] = cases.state === 'ok' ? cases.data.cases ?? [] : []
  const picked = rows.find((c) => caseKey(c) === sel) ?? rows[0]
  const prov: Prov | null = cases.state === 'ok' && rows.length > 0 ? cases.data.provenance ?? null : null

  return (
    <div>
      <PageHeader title={manifest.title} tracks={manifest.tracks}
        description={prov?.claim ? `${manifest.description} ${prov.claim}` : manifest.description}
        actions={<>
          {prov && <ProvenanceChip p={prov} />}
          <Button icon={<RefreshCw size={14} className={clsx(busy && 'animate-spin')} />} disabled={busy} onClick={refresh}>Refresh</Button>
        </>} />
      {prov && prov.state !== 'real' && prov.state !== 'empty' && (
        <div className="mb-4 flex items-start gap-2 rounded-md border px-4 py-2.5 text-cap text-ink-1" role="status" data-provenance={prov.state}
          style={{ borderColor: 'color-mix(in oklab, var(--warn) 55%, transparent)', background: 'color-mix(in oklab, var(--warn) 10%, transparent)' }}>
          <FlaskConical size={14} className="mt-0.5 shrink-0" style={{ color: 'var(--warn-ink)' }} aria-hidden />
          <span><span className="font-medium">{prov.note}</span> Do not read these numbers as a measured result.</span>
        </div>
      )}

      {cases.state === 'loading' && <div className="text-cap text-ink-3">Loading the Track 3 records…</div>}
      {cases.state === 'error' && (
        <div className="rounded-md border border-critical/60 bg-critical/10 p-4 text-cap text-ink-1" role="alert">
          <span className="font-medium text-critical-ink">The Track 3 records could not be read.</span> {cases.message}
          <div className="mt-1 text-ink-3">This is an error, not an empty result: nothing has been assumed.</div>
        </div>
      )}
      {cases.state === 'ok' && rows.length === 0 && (
        <Panel id="trace-empty" plane="operate" title="Cases" subtitle="none yet">
          <EmptyState title="No Track 3 records yet"
            body="No case has produced a decision, a model pause or an applied change in the evidence records. When a run writes them, its cases appear here." />
        </Panel>
      )}
      {cases.state === 'ok' && rows.length > 0 && (
        <div className="grid grid-cols-12 items-start gap-4">
          <div className="col-span-12 xl:col-span-5">
            <CaseList rows={rows} selected={picked ? caseKey(picked) : null} onSelect={setSel} total={cases.data.provenance?.n_records ?? cases.data.n_records} />
          </div>
          <div className="col-span-12 min-w-0 xl:col-span-7">
            {picked && <CaseDetail summary={picked} />}
          </div>
        </div>
      )}
    </div>
  )
}

// ------------------------------------------------------------------ case list

function CaseList({ rows, selected, onSelect, total }: { rows: Rec[]; selected: string | null; onSelect: (k: string) => void; total?: number }) {
  return (
    <Panel id="trace-cases" plane="operate" title="Cases" subtitle={`${rows.length} case${rows.length === 1 ? '' : 's'} from ${total ?? '?'} records`} flush>
      <div className="overflow-x-auto">
        <table className="w-full text-cap">
          <thead>
            <tr className="border-b border-line text-left text-micro text-ink-3">
              {['Case', 'First (UTC)', 'Events', 'Deny', 'Allow', 'Pause', 'Applied', 'Sequence'].map((h) => <th key={h} className="px-3 py-2 font-medium whitespace-nowrap">{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((c) => {
              const k = caseKey(c)
              return (
                <tr key={k} onClick={() => onSelect(k)} aria-selected={selected === k}
                  className={clsx('cursor-pointer border-b border-line/60', selected === k ? 'bg-s2' : 'hover:bg-s2/60')}>
                  <td className="px-3 py-2">
                    <div className="mono whitespace-nowrap text-ink-1">{c.case_id}</div>
                    <div className="mono text-micro text-ink-3">{c.run_id ?? 'no run id'}{c.provenance?.state && c.provenance.state !== 'real' ? ` · ${String(c.provenance.label).toLowerCase()}` : ''}</div>
                  </td>
                  <td className="tnum mono px-3 whitespace-nowrap text-ink-3" title={c.first_time ?? ''}>{utc(c.first_time, true)}</td>
                  <td className="tnum px-3 text-right text-ink-1">{c.n_events}</td>
                  <td className="tnum px-3 text-right" style={{ color: c.denies ? 'var(--critical-ink)' : 'var(--ink-3)' }}>{c.denies}</td>
                  <td className="tnum px-3 text-right" style={{ color: c.allows ? 'var(--good-ink)' : 'var(--ink-3)' }}>{c.allows}</td>
                  <td className="tnum px-3 text-right" style={{ color: c.pauses ? 'var(--warn-ink)' : 'var(--ink-3)' }}>{c.pauses}</td>
                  <td className="tnum px-3 text-right text-ink-1">{c.applied}</td>
                  <td className="px-3">{c.check_passed ? (c.provenance?.state === 'real' ? <StatusPill status="passed" size="sm">complete</StatusPill> : <StatusPill status="warning" size="sm" title="The sequence is complete, but these records are not verified as a real run">complete, not real</StatusPill>) : <StatusPill status="warning" size="sm">incomplete</StatusPill>}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ one case

function CaseDetail({ summary }: { summary: Rec }) {
  const [d, setD] = useState<Load<Rec>>({ state: 'loading' })
  const { case_id, run_id } = summary
  const stamp = `${summary.n_events}|${summary.last_time}|${summary.check_passed}`

  useEffect(() => {
    let alive = true
    const q = run_id ? `?run_id=${encodeURIComponent(run_id)}` : ''
    get<Rec>(`/api/trace/case/${encodeURIComponent(case_id)}${q}`)
      .then((data) => alive && setD({ state: 'ok', data }))
      .catch((e) => alive && setD({ state: 'error', message: String(e?.message ?? e) }))
    return () => { alive = false }
  }, [case_id, run_id, stamp])

  if (d.state === 'loading') return <div className="text-cap text-ink-3">Loading {case_id}…</div>
  if (d.state === 'error') return <div className="rounded-md border border-critical/60 bg-critical/10 p-3 text-cap text-critical-ink" role="alert">Could not load {case_id}: {d.message}</div>
  const c = d.data
  return (
    <div className="flex flex-col gap-4">
      <SequenceCheck c={c} />
      <Panel id="trace-timeline" plane="operate" title={<span className="mono">{c.case_id}</span>}
        subtitle={`${c.timeline.length} events, ${utc(c.first_time, true)} to ${utc(c.last_time)}`}
        actions={c.provenance ? <ProvenanceChip p={c.provenance} /> : undefined}>
        <ol className="relative" aria-label="Timeline">
          {c.timeline.map((e: Rec, i: number) => <TimelineRow key={i} e={e} last={i === c.timeline.length - 1} />)}
        </ol>
      </Panel>
    </div>
  )
}

function SequenceCheck({ c }: { c: Rec }) {
  const k: Rec = c.check
  const cmd: string = c.show_record
  return (
    <Panel id="trace-check" plane="operate" title="Expected sequence" subtitle="case opened, agents asked, tools called, approval, change applied, observe"
      actions={k.passed ? (c.provenance?.state === 'real' ? <StatusPill status="passed">Sequence complete</StatusPill> : <StatusPill status="warning" title="The sequence is complete, but these records are not verified as a real run">Complete, not a real run</StatusPill>) : <StatusPill status="blocked">{k.order_ok ? 'Sequence incomplete' : 'Out of order'}</StatusPill>}>
      <div data-state={k.passed ? 'pass' : 'fail'} className="flex flex-wrap items-center gap-1.5">
        {(k.steps as Rec[]).map((s, i) => {
          const missing = s.required && !s.present
          return (
            <span key={s.step} className="flex items-center gap-1.5">
              {i > 0 && <span className="text-ink-3" aria-hidden>→</span>}
              <span className={clsx('inline-flex items-center gap-1 rounded-[4px] border px-1.5 py-0.5 text-cap', missing ? 'border-critical/60' : 'border-line')}
                style={{ color: missing ? 'var(--critical-ink)' : s.present ? 'var(--good-ink)' : 'var(--ink-3)' }}
                title={missing ? 'required step not found' : s.present ? 'found' : 'optional, not found'}>
                {missing ? <CircleX size={12} aria-hidden /> : s.present ? <CircleCheck size={12} aria-hidden /> : <Clock3 size={12} aria-hidden />}
                {String(s.step).replace('_', ' ')}{s.step === 'ask_agent' ? '*' : ''}
              </span>
            </span>
          )
        })}
      </div>
      {k.missing.length > 0 && <p className="mt-2 text-cap text-critical-ink">Missing: {(k.missing as string[]).map((m) => m.replace('_', ' ')).join(', ')}.</p>}
      {!k.order_ok && <p className="mt-2 text-cap text-critical-ink">The steps that are present did not happen in the declared order.</p>}
      {(k.notes as string[]).map((n) => <p key={n} className="mt-1 flex items-start gap-1.5 text-cap text-ink-2"><TriangleAlert size={13} className="mt-0.5 shrink-0" style={{ color: 'var(--warn-ink)' }} aria-hidden />{n}</p>)}
      <p className="mt-2 text-micro text-ink-3">* ask agent may be absent; every other step is required. A refused call does not count as having happened.</p>
      {c.other_runs?.length > 0 && <p className="mt-1 text-micro text-ink-3">The same case id also appears in: {c.other_runs.join(', ')}. Showing run {c.run_id}.</p>}
      <div className="mt-3 flex min-w-0 items-center gap-2">
        <div className="min-w-0 flex-1"><CodeBlock code={cmd} lang="text" className="!py-1.5 whitespace-pre-wrap break-all" /></div>
        <CopyText value={cmd} />
      </div>
      <div className="mt-1 text-micro text-ink-3">Run this from the repository root to print the same records outside the dashboard.</div>
    </Panel>
  )
}

// ------------------------------------------------------------------ timeline

const CAT: Record<string, { color: string; ink: string; Icon: typeof Ban; word: string }> = {
  allow: { color: 'var(--good)', ink: 'var(--good-ink)', Icon: CircleCheck, word: 'ALLOW' },
  deny: { color: 'var(--critical)', ink: 'var(--critical-ink)', Icon: Ban, word: 'DENY' },
  pause: { color: 'var(--warn)', ink: 'var(--warn-ink)', Icon: PauseCircle, word: 'DENY (paused)' },
  approval: { color: 'var(--warn)', ink: 'var(--warn-ink)', Icon: UserCheck, word: 'DENY (needs approval)' },
  applied: { color: 'var(--release)', ink: 'var(--ink-1)', Icon: Rocket, word: 'CHANGE APPLIED' },
  control: { color: 'var(--warn)', ink: 'var(--warn-ink)', Icon: PauseCircle, word: '' },
}

function TimelineRow({ e, last }: { e: Rec; last: boolean }) {
  const cat = CAT[e.category] ?? CAT.allow
  const word = e.category === 'control' ? String(e.decision ?? 'CONTROL') : cat.word
  const approved = e.step === 'approval' && e.category === 'allow'
  const gs: Rec | null = e.guardrail_scores
  return (
    <li className="relative flex gap-3 pb-4" data-category={e.category}>
      {!last && <span className="absolute top-6 bottom-0 left-[10px] w-px bg-line" aria-hidden />}
      <span className="relative z-[1] mt-0.5 grid size-5 shrink-0 place-items-center rounded-full" style={{ background: `color-mix(in oklab, ${cat.color} 18%, transparent)`, color: cat.ink }}>
        <cat.Icon size={13} aria-hidden />
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
          <span className="tnum mono text-cap text-ink-3" title={e.time ?? ''}>{utc(e.time)}</span>
          <span className="mono text-body font-medium text-ink-1">{e.tool ?? '—'}</span>
          <span className="rounded-[4px] px-1 text-micro font-semibold" style={{ color: cat.ink, background: `color-mix(in oklab, ${cat.color} 14%, transparent)` }}>{word}</span>
          {approved && <span className="inline-flex items-center gap-1 rounded-[4px] px-1 text-micro font-semibold" style={{ color: 'var(--good-ink)', background: 'color-mix(in oklab, var(--good) 14%, transparent)' }}><UserCheck size={11} aria-hidden />approved by a person</span>}
          {e.kind === 'log' && <span className="text-micro text-ink-3">from the tool log</span>}
        </div>
        <div className="mt-0.5 flex flex-wrap gap-x-3 gap-y-0.5 text-cap text-ink-2">
          {e.agent && <span>agent <span className="mono text-ink-1">{e.agent}</span></span>}
          {e.deciding_rule && <span>rule <span className="mono text-ink-1">{e.deciding_rule}</span></span>}
          {e.approval_id && <span>approval <span className="mono text-ink-1">{e.approval_id}</span></span>}
          {e.source && <span className="text-ink-3">{e.source}</span>}
        </div>
        {e.reason && <div className="mt-0.5 text-cap text-ink-2">{e.reason}</div>}
        {gs && Object.keys(gs).length > 0 && (
          <div className="mt-1 flex flex-wrap gap-1.5">
            {Object.entries(gs).map(([k, v]) => <span key={k} className="tnum mono rounded border border-line px-1 text-micro text-ink-2">{k} {typeof v === 'number' ? v : JSON.stringify(v)}</span>)}
          </div>
        )}
        {e.kind === 'change' && <ChangeDetail d={e.detail} />}
        {e.kind === 'log' && e.detail?.change_text && <div className="mono mt-1 text-cap text-ink-1">{e.detail.change_text}</div>}
      </div>
    </li>
  )
}

function ChangeDetail({ d }: { d: Rec }): ReactNode {
  const pred: Rec | undefined = d.predicted_delta
  const meas: Rec = d.measured_delta ?? {}
  const err: Rec = d.error ?? {}
  const keys = Array.from(new Set([...Object.keys(pred ?? {}), ...Object.keys(meas)]))
  return (
    <div className="mt-2 rounded-md border border-line bg-s2/50 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="mono text-cap text-ink-1" title="canonical change string, exactly as recorded">{d.change_text ?? '—'}</span>
        {d.item_id && <span className="mono text-micro text-ink-3">{d.item_id}</span>}
        {d.probe && <StatusPill status="neutral" size="sm" title="A fixed change with no model: it measures the twin, it does not test a prediction">probe</StatusPill>}
        {d.scored === false && <StatusPill status="warning" size="sm" title="This change is not scored against a prediction">unscored{d.unscored_reason ? `: ${d.unscored_reason}` : ''}</StatusPill>}
        {d.log_line_seen && <span className="text-micro text-ink-3">also in the tool log</span>}
      </div>
      <table className="mt-2 w-full text-cap">
        <thead>
          <tr className="text-left text-micro text-ink-3">
            <th className="py-1 pr-3 font-medium">Change in KPI</th>
            <th className="px-3 py-1 text-right font-medium">Predicted</th>
            <th className="px-3 py-1 text-right font-medium">Measured</th>
            <th className="py-1 pl-3 text-right font-medium">Abs. error</th>
          </tr>
        </thead>
        <tbody>
          {keys.map((k) => (
            <tr key={k} className="border-t border-line/60">
              <td className="mono py-1 pr-3 text-ink-2">{k}</td>
              <td className="tnum mono px-3 text-right text-ink-1">{pred ? delta(pred[k]) : '—'}</td>
              <td className="tnum mono px-3 text-right text-ink-1">{delta(meas[k])}</td>
              <td className="tnum mono py-1 pl-3 text-right text-ink-2">{typeof err[k] === 'number' ? Number(err[k].toPrecision(3)) : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-0.5 text-micro text-ink-3">
        {!pred && <span>no prediction recorded for this change{d.probe ? ' (probe)' : ''}</span>}
        {pred && <span>direction {d.direction_agree == null ? '—' : d.direction_agree ? 'agrees' : 'disagrees'}</span>}
        {typeof err.summary === 'number' && <span>mean abs. error <span className="tnum mono">{Number(err.summary.toPrecision(3))}</span></span>}
        {d.predictor && <span>predictor <span className="mono">{d.predictor.model_id} {d.predictor.version}</span></span>}
        {d.policy && <span>policy <span className="mono">{d.policy.model_id} {d.policy.version}</span></span>}
        {(d.trust_before != null || d.trust_after != null) && <span>trust <span className="tnum mono">{d.trust_before ?? '—'} → {d.trust_after ?? '—'}</span></span>}
        {d.twin && <span>twin <span className="mono">{d.twin.kind} step {d.twin.step}</span></span>}
      </div>
    </div>
  )
}
