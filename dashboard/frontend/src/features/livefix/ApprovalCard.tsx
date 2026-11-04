import QRCode from 'qrcode'
import { ShieldCheck } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ConfidenceBadge, SideEffectsTable } from '@/components/livefix'
import { accessToken, withToken } from '@/core/access'
import { useLive } from '@/core/store'
import type { Rec } from '@/core/types'
import { Button } from '@/ui'
import { DecideForm } from '../kitapprovals/DecideForm'
import { kit, PIN_COMMAND, type Detail, type Status as KitStatus } from '../kitapprovals/api'
import { lf, toConfidence, toSideEffects } from './api'

const SIDE_NOTE = 'Predicted on the preflight copy of the twin (users placed differently from the grader’s twin), at the '
  + 'grade step: without the change → with it. A prediction, not the network; each cell is judged against its own healthy level.'

function parseChange(text: unknown): { cell: number; dp: number; dt: number }[] {
  if (typeof text !== 'string') return []
  return text.split('|').map((p) => /cell=(\d+);d_power_db=([+-][\d.]+);d_tilt_deg=([+-][\d.]+)/.exec(p.trim()))
    .filter((m): m is RegExpExecArray => !!m).map((m) => ({ cell: Number(m[1]), dp: Number(m[2]), dt: Number(m[3]) }))
}
const sgn = (v: number) => `${v > 0 ? '+' : ''}${v.toFixed(1)}`

function PhoneQr({ approvalId }: { approvalId: string }) {
  const token: string | null = useLive((s) => s.config?.access_token) ?? accessToken()
  const [ph, setPh] = useState<Rec | null>(null)
  const [qr, setQr] = useState('')
  useEffect(() => { lf('/phone').then(setPh).catch(() => setPh(null)) }, [])
  const url = ph?.ready ? withToken(`http://${ph.lan_ip}:${ph.port}/m/kit?id=${encodeURIComponent(approvalId)}`, token) : ''
  useEffect(() => { if (url) QRCode.toDataURL(url, { margin: 1, width: 132 }).then(setQr).catch(() => setQr('')) }, [url])
  if (!ph) return null
  return (
    <div className="flex items-center gap-3 rounded-md border border-line bg-s1 p-2" data-testid="livefix-phone">
      {ph.ready && qr ? <img src={qr} alt="QR code: approve this change on a paired phone" width={92} height={92} className="rounded" />
        : <div className="grid size-[92px] place-items-center rounded border border-dashed border-line text-center text-micro text-ink-3">no phone link</div>}
      <div className="min-w-0 max-w-[220px] text-micro text-ink-2">
        <div className="text-cap font-medium text-ink-1">Approve on a phone</div>
        {ph.ready ? 'Scan on the same Wi-Fi: the same approval, PIN and audit record as here.' : ph.note}
      </div>
    </div>
  )
}

/** Inline approval: the change per cell (before → after), the agent's reason, the confidence and its tier, and
 *  Approve / Deny with the approver PIN. The decision goes through /api/kitapprovals, the Approvals page's code
 *  path (same kit update, identity and CloudWatch line); nothing is ever approved automatically. */
export function ApprovalCard({ run, act }: { run: Rec; act: Rec | null }) {
  const a = run.approval
  const [row, setRow] = useState<Detail | null>(null)
  const [status, setStatus] = useState<KitStatus | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  const [asked, setAsked] = useState<string | null>(null)
  const id: string | undefined = a?.approval_id
  const live = run.mode === 'live'
  useEffect(() => {
    if (!id || !live) return
    let stop = false
    const load = async () => {
      const [d, s] = await Promise.all([kit.detail(id), kit.status()])
      if (stop) return
      if (d.ok) setRow(d.data); else setMsg(d.fail.message)
      if (s.ok) setStatus(s.data)
    }
    load()
    const t = setInterval(() => { if (!document.hidden) load() }, 3000) // a phone or CLI decision shows up here
    return () => { stop = true; clearInterval(t) }
  }, [id, live])
  if (!a) return null
  const before: Rec = Object.fromEntries((run.cells_at_proposal ?? run.cells ?? []).map((c: Rec) => [c.cell, c]))
  const cells = parseChange(a.change_text)
  const awsTime = (act?.events ?? []).find((e: Rec) => e.kind === 'approval' && e.approval_id === id)?.time
  const conf = run.confidence ? toConfidence(run.confidence, run.lessons) : null
  // The tier the card enforces: the stricter of the confidence tier and the side-effect tier (backend run.tier).
  const tier: string = run.tier?.tier ?? conf?.tier ?? 'one_person'
  const why: string[] = run.tier?.why ?? []
  const hold = live && tier === 'hold'
  const sideRows = toSideEffects(run.side_effects)
  const superseded: string[] = run.superseded ?? []
  const decided = live ? !!row && row.status !== 'PENDING' : run.state !== 'awaiting_approval'
  if (decided) {
    const st = row?.status ?? a.status
    return (
      <div className="rounded-md border border-line bg-s1 px-3 py-2 text-cap" data-testid="livefix-approval" data-state="decided">
        <ShieldCheck size={14} className="mr-1 inline text-ink-2" aria-hidden />
        <span className="font-medium text-ink-1">{String(st).toLowerCase()}</span>{' '}
        <span className="font-mono text-micro text-ink-3">{id}</span>
        {row?.decided_by && <> by <span className="text-ink-1">{row.decided_by}</span></>}
        {row?.decided_at && <> at {new Date(row.decided_at * 1000).toLocaleTimeString()}</>}
        {awsTime && <> · CloudWatch {String(awsTime).slice(11, 19)}Z</>}
        {row?.decision_reason && <span className="text-ink-3"> · “{row.decision_reason}”</span>}
        {conf && <span className="text-ink-3"> · confidence {conf.score.toFixed(2)}</span>}
        <span className="text-ink-3"> · tier {tier.replace('_', '-')}</span>
        {sideRows.some((r) => r.worse) && <span style={{ color: 'var(--warn-ink)' }}> · predicted to make {sideRows.filter((r) => r.worse).map((r) => `cell ${r.cell}`).join(', ')} worse</span>}
        <span className="text-ink-3"> · {a.change_text}</span>
      </div>
    )
  }
  const askAlternative = () => {
    setMsg(null)
    lf(`/runs/${encodeURIComponent(run.run_id)}/alternative`, {})
      .then(() => setAsked('Asked noc_lead for another change on the same case; it will stop at Cedar’s hold again.'))
      .catch((e) => setMsg(String((e as Error).message)))
  }
  return (
    <div className="rounded-[10px] border-2 p-3" style={{ borderColor: 'color-mix(in oklab, var(--warn) 65%, transparent)' }} data-testid="livefix-approval" data-state="pending">
      <div className="flex flex-wrap items-center gap-2 text-cap">
        <ShieldCheck size={16} className="text-ink-2" aria-hidden />
        <span className="font-medium text-ink-1">Cedar held the change: a person decides</span>
        <span className="font-mono text-micro text-ink-3">{id}</span>
        {live && <span className="rounded border px-1.5 text-micro font-semibold" data-testid="tier"
          style={{ color: tier === 'hold' ? 'var(--critical-ink)' : tier === 'two_person' ? 'var(--warn-ink)' : 'var(--ink-2)', borderColor: 'currentColor' }}>
          {tier === 'hold' ? 'ON HOLD' : tier === 'two_person' ? 'TWO PEOPLE' : 'ONE PERSON'}</span>}
        {superseded.length > 0 && <span className="text-micro text-ink-3">replaces held {superseded.join(', ')} (still PENDING in the table: deny it there)</span>}
      </div>
      {hold && (
        <div role="alert" className="mt-2 rounded-md border p-2 text-cap" data-testid="hold"
          style={{ borderColor: 'var(--critical)', background: 'color-mix(in oklab, var(--critical) 10%, transparent)' }}>
          <p className="font-medium" style={{ color: 'var(--critical-ink)' }}>On hold: {why.join('; ') || 'predicted to do harm'}.</p>
          <p className="mt-0.5 text-ink-2">Approve is off on this card and the phone link is hidden. Deny it, or ask the agents for a change that does not degrade other cells.</p>
          <div className="mt-1.5 flex flex-wrap items-center gap-2">
            <Button size="sm" onClick={askAlternative} disabled={!run.can_follow_up || !!asked} data-testid="ask-alternative">Ask for an alternative</Button>
            {asked && <span className="text-micro text-ink-3">{asked}</span>}
            {!run.can_follow_up && <span className="text-micro text-ink-3">No follow-ups left in this run.</span>}
          </div>
        </div>
      )}
      <div className="mt-2 grid grid-cols-1 gap-3 lg:grid-cols-2">
        <div>
          <table className="w-full text-cap">
            <thead><tr className="text-left text-micro text-ink-3"><th className="font-normal">cell</th><th className="font-normal">power (dBm)</th><th className="font-normal">tilt (°)</th></tr></thead>
            <tbody>
              {cells.map((c) => {
                const b = before[c.cell] ?? {}
                const p = typeof b.power_dbm === 'number' ? b.power_dbm : null
                const t = typeof b.tilt_deg === 'number' ? b.tilt_deg : null
                return (
                  <tr key={c.cell} className="border-t border-line">
                    <td className="py-1 text-ink-2">cell {c.cell}</td>
                    <td className="tnum text-ink-1">{p !== null ? `${p.toFixed(1)} → ${(p + c.dp).toFixed(1)}` : '—'} <span className="text-micro text-ink-3">({sgn(c.dp)})</span></td>
                    <td className="tnum text-ink-1">{t !== null ? `${t.toFixed(1)} → ${(t + c.dt).toFixed(1)}` : '—'} <span className="text-micro text-ink-3">({sgn(c.dt)})</span></td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          <p className="mt-1 text-micro text-ink-3">Before = the twin now; after = before plus the requested change, before the SDK safety filter.</p>
          {a.reason && <p className="mt-1 text-cap text-ink-2">Agent's reason: {a.reason}</p>}
        </div>
        {conf && run.confidence?.score != null
          ? <ConfidenceBadge score={conf.score} tier={conf.tier} breakdown={conf.breakdown} explain={run.confidence.explain}
              calibrated={!!run.confidence.calibrated} calibrationNote={run.confidence.label} />
          : <p className="text-micro text-ink-3">{live ? 'Confidence not computed (the twin-side helper is not running).' : ''}</p>}
      </div>
      {live && (
        <div className="mt-3">
          <div className="text-cap font-medium text-ink-1">Side effects, cell by cell</div>
          {sideRows.length > 0
            ? <SideEffectsTable className="mt-1" rows={sideRows} note={SIDE_NOTE} />
            : <p className="text-micro text-ink-3">Side effects not computed (the twin-side helper is not running): the per-cell prediction is missing, so check the change by hand.</p>}
        </div>
      )}
      {!live && <p className="mt-2 text-cap text-ink-3">RECORDED: in the judged run nobody approved this change. Press Continue (recorded) to see the scored outcome.</p>}
      {live && (
        <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-[1fr_auto]">
          <div>
            {status && !status.pin.set && <p className="mb-2 text-cap" style={{ color: 'var(--warn-ink)' }}>No approver PIN is set on this laptop. In a terminal: <code className="font-mono">{PIN_COMMAND}</code></p>}
            {status?.approver && <p className="mb-1 text-micro text-ink-3">Recorded as {status.approver.pseudonym}, audited in CloudWatch.</p>}
            {tier === 'two_person' && (
              <p className="mb-2 text-cap" style={{ color: 'var(--warn-ink)' }} data-testid="two-person">
                Two-person tier{why.length ? ` (${why.join('; ')})` : ''}: a second person should review this change. The
                kit records one decision per change; a second approver's co-signature is not built yet.
              </p>
            )}
            {row && <DecideForm approvalId={row.approval_id} changeSha256={row.change_sha256} compact
              approveDisabled={hold ? 'Approve is off: the change is on hold (see above). Deny is still available.' : null}
              onDone={() => { kit.detail(row.approval_id).then((d) => { if (d.ok) setRow(d.data) }) }} />}
            {msg && <p role="alert" className="mt-1 text-cap" style={{ color: 'var(--critical-ink)' }}>{msg}</p>}
            <p className="mt-2 text-micro text-ink-3">
              {hold ? 'The hold is this card’s rule: the Approvals page and the kit CLI do not know about it. ' : 'Also: '}
              <Link to={a.approve_link} className="underline decoration-dotted">the Approvals page</Link> or <code className="font-mono">checkpoint_kit.py approve {id} "why"</code></p>
          </div>
          {hold
            ? <div className="grid w-[230px] place-items-center rounded-md border border-dashed border-line p-2 text-center text-micro text-ink-3" data-testid="livefix-phone-off">Phone approval is off while the change is on hold.</div>
            : <PhoneQr approvalId={id!} />}
        </div>
      )}
    </div>
  )
}
