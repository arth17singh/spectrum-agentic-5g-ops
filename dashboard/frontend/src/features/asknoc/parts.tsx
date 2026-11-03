import clsx from 'clsx'
import { CircleStop, Send } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import { Link } from 'react-router-dom'
import type { Rec } from '@/core/types'
import { Button, CopyText, StatusPill } from '@/ui'
import { AskError } from './api'
import { copyFor, KIND_LABEL, TONE_CLASS, type Ctx } from './states'

// Every text from the kit or the operator is rendered as React text (never HTML). Only ids that match the
// strict case-/chg- patterns become links or copy buttons.

const MAX = 2000
const LABEL_TITLE: Record<string, string> = {
  LIVE: 'From the real checkpoint kit, followed by this laptop now.',
  RECORDED: 'Captured live earlier on this laptop; not re-checked.',
  TEST: 'From a local stand-in Gateway, not the real kit.',
}
const amber = {
  color: 'var(--warn-ink)',
  borderColor: 'color-mix(in oklab, var(--warn) 55%, transparent)',
  background: 'color-mix(in oklab, var(--warn) 10%, transparent)',
}
const time = (iso?: string | null) => (iso ? new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '')

export function LabelChip({ label, title }: { label: string; title?: string }) {
  const live = label === 'LIVE'
  return (
    <span title={title} data-label={label}
      className={clsx('rounded border px-1.5 py-0.5 text-micro font-semibold tracking-[0.04em]', live && 'border-good/50 bg-good/10 text-good-ink')}
      style={live ? undefined : amber}>{label}</span>
  )
}

export function HealthBanner({ health }: { health: Rec | null }) {
  if (!health || health.mode === 'ok') return null
  const failing = ((health.checks ?? []) as Rec[]).filter((c) => c.state !== 'pass' && c.id !== 'agent_builder')
  const title: Record<string, string> = {
    offline: 'Offline: the panel makes no calls to the kit.',
    not_configured: 'Ask the NOC isn’t set up on this laptop.',
    down: 'The kit can’t be used right now.',
    degraded: 'The kit answers, with warnings.',
    loopback_only: 'Ask the NOC works on the presenter laptop only (http://localhost:8787).',
  }
  const bad = health.mode === 'down' || health.mode === 'not_configured'
  return (
    <div role="status" className={clsx('mb-4 rounded-md border px-4 py-2.5 text-cap text-ink-1', bad ? 'border-critical/60 bg-critical/10' : 'border-warn/60 bg-warn/10')}>
      <div className="font-medium">{title[health.mode] ?? health.mode}</div>
      {failing.length > 0 && (
        <ul className="mt-1 list-disc pl-5 text-ink-2">
          {failing.map((c) => <li key={c.id}><span className="mono">{c.id}</span>: {c.detail}</li>)}
        </ul>
      )}
      {bad && health.fallback && <div className="mt-1 text-ink-2">{health.fallback}</div>}
    </div>
  )
}

function StateCard({ turn, ctx, onStop }: { turn: Rec; ctx: Ctx; onStop: (id: string) => void }) {
  const c = copyFor(turn, ctx)
  if (turn.ui_state === 'answered' && !turn.reply?.empty) return null
  return (
    <div className={clsx('mt-2 rounded-md border px-3 py-2 text-cap', TONE_CLASS[c.tone])} role={c.tone === 'critical' ? 'alert' : 'status'}>
      <div className="flex items-center gap-2">
        <span className="font-medium text-ink-1">{c.title}</span>
        {turn.can_stop && (
          <Button size="sm" variant="ghost" className="ml-auto" icon={<CircleStop size={13} />} onClick={() => onStop(turn.turn_id)}
            title="Stop following this message on this laptop. The kit has no cancel.">Stop following</Button>
        )}
      </div>
      {c.body && <div className="mt-0.5 text-ink-2">{c.body}</div>}
      {((turn.waiting_approvals ?? []) as Rec[]).map((w) => (
        <div key={w.approval_id} className="mt-1.5 flex flex-wrap items-center gap-2">
          <StatusPill status="pending" size="sm">waits for a person</StatusPill>
          <span className="mono text-ink-1">{w.approval_id}</span><CopyText value={w.approval_id} />
          {w.change ? <span className="mono text-ink-1">{w.change}</span> : <span className="text-ink-3">(not a twin change string)</span>}
          {w.item_id && <span className="mono text-ink-3">{w.item_id}</span>}
        </div>
      ))}
    </div>
  )
}

export function Thread({ turns, ctx, onStop }: { turns: Rec[]; ctx: Ctx; onStop: (id: string) => void }) {
  if (!turns.length) return <div className="py-8 text-center text-cap text-ink-3">No messages yet. Describe an incident or ask a question below.</div>
  return (
    <ol className="flex flex-col gap-4">
      {turns.map((t) => (
        <li key={t.turn_id} className="flex flex-col gap-2" data-turn={t.turn_id}>
          <div className="self-end max-w-[85%] rounded-lg border border-line bg-s2 px-3 py-2">
            <div className="mb-0.5 flex items-center gap-2 text-micro text-ink-3">
              <span>You</span><span>{time(t.created_at)}</span>
              <span title={`Sent with one extra line, “ref: ${t.tag}”, so a timed-out send can be found in the kit's audit log.`}>+ ref line</span>
            </div>
            <div className="whitespace-pre-wrap break-words text-body text-ink-1">{t.text}</div>
          </div>
          <div className="max-w-[92%] rounded-lg border border-line bg-s1 px-3 py-2">
            <div className="mb-1 flex flex-wrap items-center gap-2 text-micro text-ink-3">
              <span className="font-medium text-ink-2">noc_lead</span>
              <LabelChip label={t.label} title={LABEL_TITLE[t.label]} />
              {t.case_id && <span className="mono">{t.case_id}</span>}
              {t.case_id && <span title="The kit uses the case id as the trace id.">trace {t.case_id}</span>}
              {t.reply?.received_at && <span>{time(t.reply.received_at)}</span>}
              {t.reply?.redactions > 0 && <span style={amber} className="rounded border px-1" title="Addresses, keys or account numbers in the kit's text were replaced with [redacted].">{t.reply.redactions} redacted</span>}
            </div>
            {t.reply?.text ? <div className="whitespace-pre-wrap break-words text-body text-ink-1">{t.reply.text}</div> : null}
            {t.reply?.truncated && <div className="text-micro text-ink-3">(cut at 20 000 characters)</div>}
            <StateCard turn={t} ctx={ctx} onStop={onStop} />
          </div>
        </li>
      ))}
    </ol>
  )
}

export function Composer({ disabled, why, onSend, placeholder }: {
  disabled: boolean; why?: string | null; placeholder: string; onSend: (text: string) => Promise<void>
}) {
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<{ message: string; reasons?: string[] } | null>(null)
  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (!text.trim() || busy || disabled) return
    setBusy(true)
    setErr(null)
    try {
      await onSend(text)
      setText('')
    } catch (x) {
      setErr(x instanceof AskError ? { message: x.detail.message, reasons: x.detail.reasons } : { message: String((x as Error)?.message ?? x) })
    } finally {
      setBusy(false)
    }
  }
  return (
    <form onSubmit={submit} className="mt-4 flex flex-col gap-2">
      <textarea value={text} onChange={(e) => setText(e.target.value)} maxLength={MAX} rows={3} disabled={disabled || busy}
        placeholder={placeholder} aria-label="Message to noc_lead"
        className="w-full resize-y rounded-md border border-line bg-s1 px-3 py-2 text-body text-ink-1 placeholder:text-ink-3 disabled:opacity-50"
        onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) void submit(e) }} />
      <div className="flex flex-wrap items-center gap-2 text-micro text-ink-3">
        <span>Your text is stored verbatim in the team’s AWS audit log. Don’t paste secrets, addresses or personal data.</span>
        <span className="ml-auto mono">{text.length}/{MAX}</span>
        <Button type="submit" variant="primary" size="sm" icon={<Send size={13} />} disabled={disabled || busy || !text.trim()}>{busy ? 'Sending…' : 'Send'}</Button>
      </div>
      {disabled && why && <div className="text-cap text-ink-3">{why}</div>}
      {err && (
        <div role="alert" className="rounded-md border border-critical/60 bg-critical/10 px-3 py-2 text-cap text-ink-1">
          {err.message}
          {err.reasons && <ul className="mt-1 list-disc pl-5">{err.reasons.map((r) => <li key={r}>{r}</li>)}</ul>}
        </div>
      )}
    </form>
  )
}

const decisionPill = (e: Rec) => {
  if (e.kind === 'apply_call' && e.approval) return <StatusPill status={e.approval === 'APPLIED' ? 'approved' : e.decision === 'ALLOW' ? 'approved' : 'pending'} size="sm">{e.decision === 'ALLOW' ? 'applied' : `waits: ${e.approval}`}</StatusPill>
  if (e.kind === 'approval') return <StatusPill status={e.decision === 'APPROVED' ? 'approved' : 'rejected'} size="sm">{String(e.decision ?? '').toLowerCase()}</StatusPill>
  if (e.decision === 'DENY') return <StatusPill status="denied" size="sm">deny</StatusPill>
  if (e.decision === 'ALLOW') return <StatusPill status="good" size="sm">allow</StatusPill>
  return null
}

export function Activity({ data }: { data: Rec | null }) {
  if (!data) return <div className="text-cap text-ink-3">Loading…</div>
  if (!data.available) return <div className="text-cap text-ink-3">Activity unavailable: {data.message}</div>
  const events = (data.events ?? []) as Rec[]
  return (
    <div className="flex flex-col gap-1.5">
      <div className="text-micro text-ink-3">{data.source}. {data.lag_note}</div>
      {!events.length && <div className="text-cap text-ink-3">No governed calls logged for this case yet.</div>}
      <ol className="flex flex-col gap-1">
        {events.map((e, i) => (
          <li key={`${e.time}-${i}`} className="flex flex-wrap items-center gap-x-2 gap-y-0.5 border-b border-line/50 py-1 text-cap">
            <span className="mono text-micro text-ink-3">{time(e.time)}</span>
            <span className="text-ink-2">{KIND_LABEL[e.kind] ?? e.kind}</span>
            {e.agent && <span className="mono text-ink-1">{e.agent}</span>}
            {e.to && <span className="text-ink-2">→ <span className="mono text-ink-1">{e.to}</span></span>}
            {e.tool && e.kind !== 'ask_agent' && <span className="mono text-ink-3">{e.tool}</span>}
            {e.by && <span className="text-ink-3">by {e.by}</span>}
            {decisionPill(e)}
            {e.approval_id && <span className="mono text-ink-2">{e.approval_id}</span>}
            {e.change && <span className="mono text-ink-1">{e.change}</span>}
            {e.rule && <span className="mono text-micro text-ink-3">{e.rule}</span>}
            {e.error && <span className="text-ink-3">{e.error}</span>}
          </li>
        ))}
      </ol>
    </div>
  )
}

export function CaseFacts({ view, suite, activity }: { view: Rec; suite: Rec | null; activity: Rec | null }) {
  const applied = ((activity?.events ?? []) as Rec[]).some((e) => e.kind === 'apply_call' && e.decision === 'ALLOW')
  return (
    <div className="flex flex-col gap-2 text-cap">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-ink-3">case / trace id</span>
        <span className="mono text-ink-1">{view.case_id}</span><CopyText value={view.case_id} />
        <LabelChip label={view.label} />
      </div>
      <div className="text-micro text-ink-3">{view.trace_id_note} Started {time(view.started_at)}.</div>
      {suite && suite.state === 'warn' && (
        <div className="rounded-md border border-warn/60 bg-warn/10 px-3 py-2 text-ink-1">{suite.detail}</div>
      )}
      <div className="flex flex-col gap-1">
        <span className="text-ink-3">Rebuild this case’s trace from CloudWatch (terminal, read-only):</span>
        <span className="flex items-center gap-2"><code className="mono break-all text-micro text-ink-1">{view.trace_cli}</code><CopyText value={view.trace_cli} /></span>
      </div>
      {applied && <Link className="text-ink-1 underline" to={`/twin-view?case=${encodeURIComponent(view.case_id)}`}>Open the applied change in Twin View</Link>}
    </div>
  )
}
