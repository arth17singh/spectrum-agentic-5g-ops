import { useEffect, useRef, useState } from 'react'
import { Button } from '@/ui'
import { kit, type DecideOk, type Detail, type Failure } from './api'

const ARM_MS = 1500

/** The review-and-decide sheet. Key events never leave it: the shell's global hotkeys (Space = next demo
 *  beat, digits = beats, hold R = reset, P = pause) listen on window and must not fire here (review F8). */
export function ConfirmSheet({ row, pseudonym, onClose, onDone }: {
  row: Detail
  pseudonym: string
  onClose: () => void
  onDone: (r: DecideOk) => void
}) {
  const [reason, setReason] = useState('')
  const [pin, setPin] = useState('')
  const [armed, setArmed] = useState(false)
  const [busy, setBusy] = useState(false)
  const [fail, setFail] = useState<Failure | null>(null)
  const reasonRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    const t = window.setTimeout(() => setArmed(true), ARM_MS)
    reasonRef.current?.focus()
    return () => clearTimeout(t)
  }, [])

  const fixture = row.label !== 'LIVE'
  const r = reason.trim()
  const valid = r.length >= 3 && r.length <= 300 && /^\d{6,12}$/.test(pin)
  const go = async (decision: 'approve' | 'reject') => {
    if (!valid || !armed || busy) return
    setBusy(true)
    setFail(null)
    const res = await kit.decide(row.approval_id, { decision, reason: r, change_sha256: row.change_sha256, pin })
    setBusy(false)
    setPin('')
    if (res.ok) onDone(res.data)
    else setFail(res.fail)
  }

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/55 p-2 outline-none sm:p-4" data-sheet="kitapprovals" tabIndex={-1}
      onKeyDown={(e) => { e.stopPropagation(); if (e.key === 'Escape' && !busy) onClose() }}>
      <div role="dialog" aria-modal="true" aria-labelledby="kit-sheet-title"
        className="flex max-h-[calc(100dvh-1rem)] w-full max-w-[640px] min-w-0 flex-col rounded-[10px] border border-line bg-s1 shadow-2xl sm:max-h-[calc(100dvh-2rem)]">
        <header className="shrink-0 border-b border-line px-4 py-3 sm:px-5">
          <h2 id="kit-sheet-title" className="text-title font-medium break-all text-ink-1">Review and decide {row.approval_id}</h2>
          {fixture ? (
            <p className="mt-0.5 text-cap font-medium" style={{ color: 'var(--warn-ink)' }} data-wording="fixture">
              FIXTURE: a practice decision on fake rows. Nothing is sent to AWS: no table write, no CloudWatch line.
            </p>
          ) : (
            <p className="mt-0.5 text-cap text-ink-3" data-wording="live">A real decision in the kit's approvals table (team13-approvals), recorded as {pseudonym} and audited in CloudWatch.</p>
          )}
        </header>
        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-4 sm:px-5">
          <div>
            <div className="text-micro font-semibold tracking-[0.06em] text-ink-3 uppercase">The exact change</div>
            <pre className="mt-1 overflow-x-auto rounded-md border border-line bg-s2 px-3 py-2 font-mono text-[15px] leading-6 text-ink-1">{row.change_text ?? '(no change text)'}</pre>
            <div className="mt-1 flex flex-wrap gap-x-4 text-micro text-ink-3">
              <span>item {row.item_id ?? '—'}</span><span>case {row.case_id ?? '—'}</span><span>asked by {row.agent ?? '—'} (Gateway-verified token)</span>
              <span className="font-mono">sha256 {row.change_sha256.slice(0, 12)}…</span>
            </div>
            {!row.canonical && <div className="mt-1 text-cap" style={{ color: 'var(--warn-ink)' }}>Not in the canonical form{row.canonical_text ? ` (reads as ${row.canonical_text})` : ': the kit example tool, not the twin'}.</div>}
          </div>
          {row.warnings.length > 0 && (
            <ul className="space-y-1 rounded-md border px-3 py-2 text-cap text-ink-1" style={{ borderColor: 'color-mix(in oklab, var(--warn) 55%, transparent)', background: 'color-mix(in oklab, var(--warn) 10%, transparent)' }}>
              {row.warnings.map((w) => <li key={w}>{w}</li>)}
            </ul>
          )}
          {!row.prediction && row.prediction_note && <div className="text-cap" style={{ color: 'var(--warn-ink)' }}>{row.prediction_note}</div>}
          <label className="block">
            <span className="text-cap font-medium text-ink-2">Your reason (required, 3 to 300 characters; no emails, links or account ids)</span>
            <textarea ref={reasonRef} value={reason} onChange={(e) => setReason(e.target.value)} maxLength={300} rows={2}
              className="mt-1 w-full rounded-md border border-line bg-s2 px-3 py-2 text-body text-ink-1 outline-none focus:border-ink-3"
              placeholder="checked: coverage gain predicted, trust 0.91" />
            <span className="text-micro text-ink-3">{r.length}/300</span>
          </label>
          <label className="block">
            <span className="text-cap font-medium text-ink-2">Approver PIN</span>
            <input type="password" inputMode="numeric" autoComplete="off" value={pin} maxLength={12}
              onChange={(e) => setPin(e.target.value.replace(/\D/g, ''))}
              className="mt-1 block w-40 rounded-md border border-line bg-s2 px-3 py-1.5 font-mono text-body tracking-[0.3em] text-ink-1 outline-none focus:border-ink-3" />
          </label>
          <p className="text-cap text-ink-3">{row.note}</p>
          {fail && (
            <div role="alert" className="rounded-md border border-critical/60 bg-critical/10 px-3 py-2 text-cap text-ink-1">
              <span className="font-medium text-critical-ink">{fail.message}</span>
              {fail.by && <span> ({fail.status} by {fail.by})</span>}
            </div>
          )}
        </div>
        <footer className="flex shrink-0 flex-wrap items-center justify-end gap-2 border-t border-line px-4 py-3 sm:px-5">
          <Button variant="ghost" onClick={onClose} disabled={busy}>Cancel</Button>
          <Button variant="danger" onClick={() => go('reject')} disabled={!armed || !valid || busy} data-action="reject">Reject</Button>
          <Button variant="primary" onClick={() => go('approve')} disabled={!armed || !valid || busy} data-action="approve">
            <span className="sm:hidden">Approve</span>
            <span className="hidden sm:inline">Approve {row.approval_id} as {pseudonym}</span>
          </Button>
        </footer>
      </div>
    </div>
  )
}
