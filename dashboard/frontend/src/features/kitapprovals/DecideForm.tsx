import { useEffect, useState } from 'react'
import { Button } from '@/ui'
import { kit, type DecideOk, type Failure } from './api'

const ARM_MS = 1500

/** Reason + approver PIN + Approve / Reject: the same POST /api/kitapprovals/{id}/decide the Approvals page sends
 *  (same kit update, identity and CloudWatch control line). Used inline on Live Fix and on the paired phone page.
 *  Keys never leave the form (the shell's global hotkeys listen on window). */
export function DecideForm({ approvalId, changeSha256, onDone, compact }: {
  approvalId: string
  changeSha256: string
  onDone: (r: DecideOk) => void
  compact?: boolean
}) {
  const [reason, setReason] = useState('')
  const [pin, setPin] = useState('')
  const [armed, setArmed] = useState(false)
  const [busy, setBusy] = useState(false)
  const [fail, setFail] = useState<Failure | null>(null)
  useEffect(() => { const t = window.setTimeout(() => setArmed(true), ARM_MS); return () => clearTimeout(t) }, [])
  const r = reason.trim()
  const valid = r.length >= 3 && r.length <= 300 && /^\d{6,12}$/.test(pin)
  const go = async (decision: 'approve' | 'reject') => {
    if (!valid || !armed || busy) return
    setBusy(true); setFail(null)
    const res = await kit.decide(approvalId, { decision, reason: r, change_sha256: changeSha256, pin })
    setBusy(false); setPin('')
    if (res.ok) onDone(res.data)
    else setFail(res.fail)
  }
  return (
    <div className="space-y-2" onKeyDown={(e) => e.stopPropagation()} data-testid="decide-form">
      <div className={compact ? 'grid grid-cols-1 gap-2 sm:grid-cols-[1fr_9rem]' : 'space-y-2'}>
        <label className="block text-micro text-ink-3">Reason (recorded with the decision)
          <input value={reason} onChange={(e) => setReason(e.target.value)} maxLength={300} placeholder="e.g. checked the change against the symptoms"
            className="mt-0.5 h-9 w-full rounded-md border border-line bg-s2 px-2 text-cap text-ink-1" />
        </label>
        <label className="block text-micro text-ink-3">Approver PIN
          <input value={pin} onChange={(e) => setPin(e.target.value.replace(/\D/g, '').slice(0, 12))} inputMode="numeric" type="password"
            autoComplete="off" className="mt-0.5 h-9 w-full rounded-md border border-line bg-s2 px-2 font-mono text-cap text-ink-1" />
        </label>
      </div>
      <div className="flex flex-wrap gap-2">
        <Button variant="primary" disabled={!valid || !armed || busy} onClick={() => go('approve')}>Approve</Button>
        <Button disabled={!valid || !armed || busy} onClick={() => go('reject')}>Deny</Button>
      </div>
      {fail && <p role="alert" className="text-cap" style={{ color: 'var(--critical-ink)' }}>{fail.message}</p>}
    </div>
  )
}
