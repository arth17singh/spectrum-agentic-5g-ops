import { ShieldCheck } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useUi } from '@/core/ui-store'
import { DecideForm } from '../kitapprovals/DecideForm'
import { kit, type Detail } from '../kitapprovals/api'

// /m/kit?id=chg-...&t=<access token>: one REAL kit approval on a paired phone (make demo, same Wi-Fi). The decision
// is the same POST /api/kitapprovals/{id}/decide as on the laptop: same PIN, identity and CloudWatch audit line.

export default function KitPhoneApprove() {
  const theme = useUi((s) => s.theme)
  const [params] = useSearchParams()
  const id = params.get('id') ?? ''
  const valid = /^chg-[0-9a-f]{12}$/.test(id)
  const [row, setRow] = useState<Detail | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    document.body.style.overflow = 'auto'
    return () => { document.body.style.overflow = '' }
  }, [theme])
  useEffect(() => {
    if (!valid) return
    let stop = false
    const load = () => kit.detail(id).then((d) => { if (stop) return; if (d.ok) { setRow(d.data); setMsg(null) } else setMsg(d.fail.message) })
    load()
    const t = setInterval(load, 4000)
    return () => { stop = true; clearInterval(t) }
  }, [id, valid])
  const shown = valid ? msg : 'This link names no approval.'
  return (
    <main className="mx-auto min-h-dvh max-w-[520px] bg-s0 px-4 py-5 text-ink-1" data-page="kit-phone">
      <h1 className="flex items-center gap-2 text-title font-medium"><ShieldCheck size={18} aria-hidden />Approve a change</h1>
      <p className="mt-1 text-cap text-ink-3">A real decision in the kit's approvals table, audited in CloudWatch.</p>
      {shown && <p role="alert" className="mt-3 text-cap" style={{ color: 'var(--critical-ink)' }}>{shown}</p>}
      {row && (
        <div className="mt-4 space-y-3">
          <div className="text-micro text-ink-3"><span className="font-mono">{row.approval_id}</span> · case {row.case_id} · asked by {row.agent ?? '—'}</div>
          <pre className="overflow-x-auto rounded-md border border-line bg-s2 px-3 py-2 font-mono text-cap whitespace-pre-wrap">{(row.change_text ?? '').split('|').join('\n')}</pre>
          {row.agent_reason && <p className="text-cap text-ink-2">Agent's reason: {row.agent_reason}</p>}
          {row.status === 'PENDING'
            ? <DecideForm approvalId={row.approval_id} changeSha256={row.change_sha256} onDone={() => { kit.detail(id).then((d) => { if (d.ok) setRow(d.data) }) }} />
            : <p className="text-body font-medium">{row.status.toLowerCase()}{row.decided_by ? ` by ${row.decided_by}` : ''}</p>}
        </div>
      )}
    </main>
  )
}
