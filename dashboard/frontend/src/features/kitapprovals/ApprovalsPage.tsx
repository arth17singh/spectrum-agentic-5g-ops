import clsx from 'clsx'
import { RefreshCw } from 'lucide-react'
import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { Button, PageHeader, Panel, StatusPill, type Status as Pill } from '@/ui'
import { age, clock, kit, PIN_COMMAND, type DecideOk, type Detail, type Pending, type Row, type Status } from './api'
import { ConfirmSheet } from './ConfirmSheet'
import manifest from './manifest'

const POLL_MS = 5000
const STALE_S = 30 * 60
const amber = { borderColor: 'color-mix(in oklab, var(--warn) 55%, transparent)', background: 'color-mix(in oklab, var(--warn) 10%, transparent)' }

const PILL: Record<string, [Pill, string]> = {
  PENDING: ['pending', 'Pending'], APPROVED: ['warning', 'Approved, not applied'], APPLIED: ['approved', 'Applied'], REJECTED: ['rejected', 'Rejected'],
}

function Note({ tone, children }: { tone: 'red' | 'amber' | 'info'; children: ReactNode }) {
  const cls = 'mb-3 rounded-md border px-4 py-2.5 text-cap text-ink-1'
  if (tone === 'red') return <div role="alert" className={clsx(cls, 'border-critical/60 bg-critical/10')}>{children}</div>
  if (tone === 'amber') return <div role="status" className={cls} style={amber}>{children}</div>
  return <div className={clsx(cls, 'border-line bg-s1 text-ink-2')}>{children}</div>
}

function LabelChip({ label }: { label: string }) {
  const live = label === 'LIVE'
  return (
    <span data-label={label} className="rounded border px-1.5 py-0.5 text-micro font-semibold tracking-[0.06em]"
      style={live ? { color: 'var(--good-ink)', borderColor: 'color-mix(in oklab, var(--good) 55%, transparent)' } : { color: 'var(--warn-ink)', ...amber }}>
      {label}
    </span>
  )
}

function RowItem({ row, selected, onSelect }: { row: Row; selected: boolean; onSelect: () => void }) {
  const [pill, word] = PILL[row.status] ?? ['neutral', row.status]
  return (
    <button type="button" onClick={onSelect} data-row={row.approval_id}
      className={clsx('w-full rounded-md border px-3 py-2 text-left transition-colors', selected ? 'border-ink-3 bg-s3' : 'border-line bg-s2 hover:bg-s3')}>
      <div className="flex items-center gap-2">
        <span className="font-mono text-cap text-ink-1">{row.approval_id}</span>
        <StatusPill status={pill} size="sm">{word}</StatusPill>
        <span className="ml-auto text-micro text-ink-3">{row.status === 'PENDING' ? `waiting ${age(row.age_s)}` : clock(row.applied_at ?? row.decided_at)}</span>
      </div>
      <div className="mt-1 truncate font-mono text-cap text-ink-2">{row.change_text ?? '—'}</div>
      <div className="mt-0.5 flex flex-wrap gap-x-3 text-micro text-ink-3">
        <span>{row.agent ?? '—'}</span><span>{row.case_id ?? '—'}</span>
        {!row.canonical && <span style={{ color: 'var(--warn-ink)' }}>{row.canonical_text ? 'not canonical' : 'kit example tool (not the twin)'}</span>}
        {row.proposal_only && <span>evaluation proposal: not for approval</span>}
        {row.status === 'PENDING' && !row.proposal_only && (row.age_s ?? 0) > STALE_S && <span style={{ color: 'var(--warn-ink)' }}>stale: waiting over 30 min</span>}
        {row.decided_by && <span>by {row.decided_by}</span>}
      </div>
    </button>
  )
}

function DetailView({ d, canDecide, onReview }: { d: Detail; canDecide: boolean; onReview: () => void }) {
  const [pill, word] = PILL[d.status] ?? ['neutral', d.status]
  const kv: [string, ReactNode][] = [
    ['Asked by', <>{d.agent ?? '—'} <span className="text-ink-3">(Gateway-verified token, interceptor)</span></>],
    ['Case', <>{d.case_id ?? '—'}{d.case ? <span className="text-ink-3"> · twin step {d.case.step_now}{d.case.twin_kind ? ` · ${d.case.twin_kind}` : ''}</span> : null}</>],
    ['Item', d.item_id ?? '—'],
    ['Waiting', d.status === 'PENDING' ? age(d.age_s) : `created ${clock(d.created_at)}`],
  ]
  if (d.decided_by) kv.push(['Decided', <>{d.status === 'APPLIED' ? 'APPROVED' : d.status} by {d.decided_by} at {clock(d.decided_at)}{d.decision_reason ? `: “${d.decision_reason}”` : ''}</>])
  if (d.applied_at) kv.push(['Applied', clock(d.applied_at)])
  return (
    <div className="space-y-4" data-detail={d.approval_id}>
      <div className="flex items-center gap-2">
        <span className="font-mono text-body text-ink-1">{d.approval_id}</span>
        <StatusPill status={pill}>{word}</StatusPill>
      </div>
      <pre className="overflow-x-auto rounded-md border border-line bg-s2 px-3 py-2 font-mono text-[15px] leading-6 text-ink-1">{d.change_text ?? '(no change text)'}</pre>
      {d.cells && (
        <table className="w-full text-cap">
          <thead className="text-ink-3"><tr><th className="text-left font-medium">Cell</th><th className="text-right font-medium">Δ power (dB)</th><th className="text-right font-medium">Δ tilt (°)</th></tr></thead>
          <tbody className="font-mono text-ink-1">{d.cells.map((c) => <tr key={c.cell}><td>{c.cell}</td><td className="text-right">{c.d_power_db.toFixed(1)}</td><td className="text-right">{c.d_tilt_deg.toFixed(1)}</td></tr>)}</tbody>
        </table>
      )}
      <dl className="grid grid-cols-[110px_1fr] gap-x-3 gap-y-1 text-cap">
        {kv.map(([k, v]) => <div key={k} className="contents"><dt className="text-ink-3">{k}</dt><dd className="min-w-0 text-ink-1">{v}</dd></div>)}
      </dl>
      <div className="text-cap">
        <div className="text-micro font-semibold tracking-[0.06em] text-ink-3 uppercase">Agent's stated reason (unverified)</div>
        <div className="mt-0.5 text-ink-2">{d.agent_reason || '—'}</div>
      </div>
      <div className="text-cap">
        <div className="text-micro font-semibold tracking-[0.06em] text-ink-3 uppercase">Prediction and trust</div>
        {d.prediction ? (
          <div className="mt-0.5 text-ink-1">
            {Object.entries(d.prediction.predicted_delta ?? {}).map(([k, v]) => <span key={k} className="mr-3 font-mono">{k} {v >= 0 ? '+' : ''}{v}</span>)}
            <div className="text-micro text-ink-3">{d.prediction.predictor.model_id}@{d.prediction.predictor.version}, step {d.prediction.step} · {d.prediction.source}</div>
          </div>
        ) : <div className="mt-0.5" style={{ color: 'var(--warn-ink)' }}>{d.prediction_note ?? d.case_note ?? 'No prediction.'}</div>}
        {d.trust && <div className="mt-1 text-ink-2">Trust {d.trust.trust ?? '—'} · gate {d.trust.gate_status} · n {d.trust.n ?? '—'} <span className="text-micro text-ink-3">({d.trust.source})</span></div>}
        {d.case_note && d.prediction && <div className="text-micro text-ink-3">{d.case_note}</div>}
      </div>
      {d.warnings.length > 0 && <Note tone="amber">{d.warnings.map((w) => <div key={w}>{w}</div>)}</Note>}
      {d.proposal_only && <Note tone="info"><span className="font-medium">Evaluation proposal (proposal-only, not for approval).</span> This case belongs to an evaluation run of the swarm: the change was recorded and is scored offline. Approve and reject are disabled.</Note>}
      {d.status === 'PENDING' && (
        <Button variant="primary" onClick={onReview} disabled={!canDecide || d.proposal_only} data-action="review">Review and decide…</Button>
      )}
      {d.status === 'APPROVED' && (
        <Note tone="amber"><span className="font-medium">Approved, not yet applied.</span> noc_lead applies it on its next apply_change call: say “Approved.” in <Link className="underline" to={d.links.ask_noc}>Ask the NOC</Link> for case {d.case_id} (or use the kit's <code>ask --case</code>).</Note>
      )}
      {d.status === 'APPLIED' && <Note tone="info">Applied at {clock(d.applied_at)}. See the measured result in <Link className="underline" to={d.links.twin_view}>Twin View</Link> and the trail in <Link className="underline" to={d.links.trace}>Trace</Link>.</Note>}
    </div>
  )
}

export default function ApprovalsPage() {
  const [status, setStatus] = useState<Status | null>(null)
  const [pending, setPending] = useState<Pending | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [detail, setDetail] = useState<Detail | null>(null)
  const [detailErr, setDetailErr] = useState<string | null>(null)
  const [sheet, setSheet] = useState(false)
  const [busy, setBusy] = useState(false)
  const [last, setLast] = useState<DecideOk | null>(null)

  const load = useCallback(async () => {
    const [s, p] = await Promise.all([kit.status(), kit.pending()])
    if (s.ok) setStatus(s.data)
    if (p.ok) setPending(p.data)
  }, [])
  const loadDetail = useCallback(async (id: string) => {
    const r = await kit.detail(id)
    if (r.ok) { setDetail(r.data); setDetailErr(null) } else { setDetail(null); setDetailErr(r.fail.message) }
  }, [])

  useEffect(() => {
    const first = window.setTimeout(load, 0)
    const t = setInterval(() => { if (!document.hidden) load() }, POLL_MS)
    return () => { clearTimeout(first); clearInterval(t) }
  }, [load])

  // Link contract for other pages (Ask the NOC): /approvals?case=<case_id> shows only that case's rows.
  const [params, setParams] = useSearchParams()
  const caseFilter = params.get('case')
  const allRows = pending?.rows ?? []
  const rows = caseFilter ? allRows.filter((r) => r.case_id === caseFilter) : allRows
  // with ?case=, open that case's waiting change without a click
  const current = selected ?? (caseFilter ? (rows.find((r) => r.status === 'PENDING')?.approval_id ?? null) : null)
  const sel = rows.find((r) => r.approval_id === current)
  const selStatus = sel?.status
  useEffect(() => {
    if (!current) return
    const t = window.setTimeout(() => loadDetail(current), 0)
    return () => clearTimeout(t)
  }, [current, selStatus, loadDetail])

  const refresh = () => { setBusy(true); load().finally(() => setBusy(false)); if (current) loadDetail(current) }
  const waiting = rows.filter((r) => r.status === 'PENDING' && !r.proposal_only)
  const proposals = rows.filter((r) => r.status === 'PENDING' && r.proposal_only)
  const decided = rows.filter((r) => r.status !== 'PENDING')
  const mode = status?.mode ?? pending?.mode
  const pinOk = !!status?.pin.set && !status.pin.locked_for_s
  const canDecide = mode === 'ok' && pinOk && !status?.audit_outbox && !!status?.approver
  const label = pending?.label ?? status?.label ?? '…'

  return (
    <div className="min-w-0" data-slice="kitapprovals">
      <PageHeader title={manifest.title} tracks={manifest.tracks} description={manifest.description}
        actions={<>
          <LabelChip label={label} />
          {status?.approver && <span className="text-cap text-ink-2" title={status.approver.aws_session}>approver {status.approver.pseudonym}</span>}
          <Button icon={<RefreshCw size={14} className={clsx(busy && 'animate-spin')} />} disabled={busy} onClick={refresh}>Refresh</Button>
        </>} />

      {mode && mode !== 'ok' && <Note tone={mode === 'offline' ? 'info' : 'red'}>{status?.message ?? pending?.message}</Note>}
      {status && status.audit_outbox > 0 && (
        <Note tone="red"><span className="font-medium">Decided but not audited.</span> {status.audit_outbox} decision{status.audit_outbox > 1 ? 's are' : ' is'} in the table but not yet in the CloudWatch control line. It is kept on this laptop with its original time and retried; no new decision until it is written. Press Refresh to retry.</Note>
      )}
      {status && !status.pin.set && mode === 'ok' && (
        <Note tone="amber">No approver PIN is set on this laptop, so decisions are disabled. In a terminal: <code className="font-mono">{PIN_COMMAND}</code></Note>
      )}
      {status && status.pin.locked_for_s > 0 && <Note tone="red">Too many wrong PINs: decisions are locked for {Math.ceil(status.pin.locked_for_s / 60)} more minutes.</Note>}
      {last && (
        <Note tone={last.audit === 'written' ? (last.status === 'APPROVED' ? 'amber' : 'info') : 'red'}>
          <span className="font-medium">{last.approval_id} {last.status} by {last.by} at {clock(last.decided_at)}.</span>{' '}
          {last.audit === 'written' ? (label === 'LIVE' ? 'Audit line written to CloudWatch.' : `Audit line written to the ${label} log, not CloudWatch.`) : last.audit_message}{' '}
          {last.status === 'APPROVED' && 'Not applied yet: say “Approved.” in Ask the NOC so noc_lead calls apply_change.'}
        </Note>
      )}
      <Note tone="info">Real decisions on the kit's table (team13-approvals), separate from the simulated Oversight desk. Approving does not apply a change: noc_lead applies it on its next apply_change call, after you say “Approved.” in Ask the NOC (or the kit's <code>ask --case</code>).</Note>

      {caseFilter && (
        <Note tone="info"><span data-filter="case">Showing case <span className="font-mono">{caseFilter}</span> only ({rows.length} of {allRows.length} rows).</span>{' '}
          <button type="button" className="underline" onClick={() => setParams({}, { replace: true })}>Show all</button></Note>
      )}
      <div className="grid gap-4 lg:grid-cols-[minmax(320px,2fr)_3fr]">
        <Panel title="Waiting for a decision" subtitle={pending ? `last read ${pending.read_at} · ${label}` : undefined}>
          {waiting.length === 0 ? (
            <div className="py-8 text-center" data-empty="pending">
              <div className="text-body font-medium text-ink-2">No change is waiting.</div>
              <div className="mt-1 text-cap text-ink-3">Last read {pending?.read_at ?? '—'}, {label}. A row appears when noc_lead asks to apply a change.</div>
            </div>
          ) : (
            <div className="space-y-2">{waiting.map((r) => <RowItem key={r.approval_id} row={r} selected={r.approval_id === current} onSelect={() => setSelected(r.approval_id)} />)}</div>
          )}
          {proposals.length > 0 && (
            <details className="mt-5" data-section="eval-proposals">
              <summary className="cursor-pointer text-micro font-semibold tracking-[0.06em] text-ink-3 uppercase">
                Evaluation proposals (proposal-only, not for approval) · {proposals.length}
              </summary>
              <div className="mt-2 space-y-2">{proposals.map((r) => <RowItem key={r.approval_id} row={r} selected={r.approval_id === current} onSelect={() => setSelected(r.approval_id)} />)}</div>
            </details>
          )}
          {decided.length > 0 && (
            <>
              <div className="mt-5 mb-2 text-micro font-semibold tracking-[0.06em] text-ink-3 uppercase">Decided</div>
              <div className="space-y-2">{decided.map((r) => <RowItem key={r.approval_id} row={r} selected={r.approval_id === current} onSelect={() => setSelected(r.approval_id)} />)}</div>
            </>
          )}
        </Panel>
        <Panel title="Change">
          {!current && <div className="py-8 text-center text-cap text-ink-3">Select a change on the left to review it.</div>}
          {current && detailErr && <Note tone="red">{detailErr}</Note>}
          {current && detail && detail.approval_id === current && <DetailView d={detail} canDecide={canDecide} onReview={() => setSheet(true)} />}
        </Panel>
      </div>

      {sheet && detail && status?.approver && (
        <ConfirmSheet row={detail} pseudonym={status.approver.pseudonym} onClose={() => setSheet(false)}
          onDone={(r) => { setSheet(false); setLast(r); load(); loadDetail(r.approval_id) }} />
      )}
    </div>
  )
}
