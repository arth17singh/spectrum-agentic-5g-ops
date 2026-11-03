import clsx from 'clsx'
import { Check, ShieldCheck, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Wordmark } from '@/app/TopBar'
import { api } from '@/core/api'
import { EMPTY, useLive } from '@/core/store'
import type { Approval, Rec } from '@/core/types'
import { useSimNow, useUi } from '@/core/ui-store'
import { startLive } from '@/core/ws'
import { LiveDot, SimBanner } from '@/ui'
import { ImpactBars, SlaCountdown, TierPill } from '../oversight/parts'

const NO_HUMANS: Rec = {}

// Phone approval page (/m/approve). Standalone: it opens its own live connection. Defaults to the
// on-call lead so the console (A. Rivera) and the phone together satisfy four-eyes.

const REJECT_REASONS = ['Too large for tonight', 'Wait for field team', 'Prediction looks wrong']

export default function MobileApprove() {
  const theme = useUi((s) => s.theme)
  useEffect(() => {
    startLive()
    document.documentElement.dataset.theme = theme
    document.body.style.overflow = 'auto'
    return () => { document.body.style.overflow = '' }
  }, [theme])
  const ready = useLive((s) => s.ready)
  const conn = useLive((s) => s.conn)
  const items: Approval[] = useLive((s) => s.oversight.items ?? EMPTY)
  const humans: Rec = useLive((s) => s.oversight.humans ?? NO_HUMANS)
  const [picked, setPicked] = useState(() => localStorage.getItem('spectrum.mobile.by') ?? 'j.okafor')
  // The saved id may be stale or hand-edited, and the server refuses unknown approvers: fall back to a known one.
  const by = Object.hasOwn(humans, picked) ? picked : Object.hasOwn(humans, 'j.okafor') ? 'j.okafor' : Object.keys(humans)[0] ?? picked
  const pending = items.filter((i) => i.status === 'pending')
  const recent = items.filter((i) => i.status !== 'pending' && i.approvals.some((a) => a.device === 'mobile')).slice(0, 3)

  // buzz when a new request arrives (not on first load)
  const seen = useRef<Set<string> | null>(null)
  useEffect(() => {
    if (!ready) return
    const ids = new Set(pending.map((p) => p.id))
    if (seen.current) {
      for (const id of ids) if (!seen.current.has(id)) { navigator.vibrate?.([140, 70, 140]); break }
    }
    seen.current = ids
  }, [pending, ready])

  return (
    <div className="min-h-full bg-page px-4 pt-4 pb-10 text-ink-1">
      <header className="mb-4 flex items-center gap-3">
        <Wordmark />
        <span className="ml-auto"><LiveDot state={!ready ? 'connecting' : conn} label={!ready ? 'Connecting' : conn === 'live' ? 'Live' : conn} /></span>
      </header>
      <div className="mb-4"><SimBanner approvals compact /></div>
      <label className="mb-5 flex items-center gap-2 rounded-[10px] border border-line bg-s1 px-3 py-2.5 text-body">
        <span className="text-ink-3">Signing as</span>
        <select value={by} onChange={(e) => { setPicked(e.target.value); localStorage.setItem('spectrum.mobile.by', e.target.value) }}
          className="ml-auto bg-transparent text-right font-medium text-ink-1">
          {Object.values(humans).map((h: Rec) => <option key={h.id} value={h.id}>{h.name}, {h.level}</option>)}
        </select>
      </label>

      <h1 className="mb-3 text-[22px] leading-7 font-semibold">
        {pending.length ? `${pending.length} decision${pending.length === 1 ? '' : 's'} waiting` : 'Nothing waiting'}
      </h1>
      {!pending.length && <p className="text-body text-ink-3">When the swarm needs a human, the request appears here and this phone vibrates.</p>}
      <div className="space-y-4">
        {pending.map((a) => <Card key={a.id} a={a} by={by} name={humans[by]?.name ?? by} />)}
      </div>

      {recent.length > 0 && (
        <section className="mt-8">
          <h2 className="mb-2 text-cap font-medium text-ink-3">Signed from a phone</h2>
          <ul className="space-y-2">
            {recent.map((a) => {
              const ap = a.approvals.find((x) => x.device === 'mobile')!
              return (
                <li key={a.id} className="flex items-center gap-2 rounded-md border border-line bg-s1 px-3 py-2 text-cap">
                  <ShieldCheck size={15} className="text-good-ink" />
                  <span className="mono">{a.id}</span>
                  <span className="truncate text-ink-2">{ap.name}</span>
                  <span className="tnum mono ml-auto text-ink-3">#{ap.ledger_seq}</span>
                </li>
              )
            })}
          </ul>
        </section>
      )}
    </div>
  )
}

function Card({ a, by, name }: { a: Approval; by: string; name: string }) {
  const now = useSimNow(2)
  const [rejecting, setRejecting] = useState(false)
  const [busy, setBusy] = useState(false)
  const [done, setDone] = useState<{ seq?: number; decision: string } | null>(null)
  const signed = a.approvals.some((x) => x.by === by)
  const go = async (decision: 'approve' | 'reject', reason = '') => {
    setBusy(true)
    try {
      const res = await api.decide(a.id, decision, { by, device: 'mobile', reason })
      const mine = res.item.approvals.find((x: Rec) => x.by === by)
      setDone({ seq: mine?.ledger_seq ?? res.item.history[res.item.history.length - 1]?.ledger_seq, decision })
      navigator.vibrate?.(60)
    } finally {
      setBusy(false)
    }
  }
  return (
    <article className="overflow-hidden rounded-[14px] border border-line bg-s1">
      <div className="flex items-center gap-2 border-b border-line px-4 py-3">
        <TierPill tier={a.tier} />
        <span className="mono text-cap text-ink-3">{a.id}</span>
        <span className="ml-auto"><SlaCountdown a={a} now={now} /></span>
      </div>
      <div className="px-4 py-4">
        <h2 className="text-[19px] leading-6 font-semibold">{a.title}</h2>
        <p className="mt-2 text-body text-ink-2">{a.summary}</p>
        {a.blast_radius_pct != null && (
          <div className="mt-4 flex items-baseline gap-2">
            <span className="tnum text-[28px] leading-8 font-semibold">{a.blast_radius_pct}%</span>
            <span className="text-cap text-ink-3">of cells touched, rule is 5% without a human</span>
          </div>
        )}
        <div className="mt-4">
          <ImpactBars preflight={a.preflight} compact only={['congestion_pct', 'dropped_pct', 'energy_kw']} />
        </div>
        {a.dissent.length > 0 && (
          <div className="mt-4 rounded-md border-l-2 border-warn bg-warn/10 px-3 py-2 text-cap">
            <span className="mono font-medium">{a.dissent[0].agent}</span><span className="text-ink-2">: {a.dissent[0].text}</span>
          </div>
        )}
        {a.required > 1 && (
          <div className="mt-3 text-cap text-ink-2">Four-eyes: {a.approvals.length} of {a.required} approvals{a.approvals[0] ? `, ${a.approvals[0].name} already signed` : ''}.</div>
        )}
      </div>
      <div className="border-t border-line p-4">
        {done ? (
          <div className="flex items-center gap-2 rounded-md bg-good/10 px-3 py-3 text-body">
            <ShieldCheck size={18} className="text-good-ink" />
            <span>{done.decision === 'approve' ? 'Approved' : 'Rejected'} and signed by {name}</span>
            {done.seq && <span className="mono ml-auto text-cap text-ink-3">#{done.seq}</span>}
          </div>
        ) : signed ? (
          <div className="text-body text-ink-2">{name} already approved. A different person must give the second approval.</div>
        ) : rejecting ? (
          <div className="space-y-2">
            <div className="text-cap text-ink-3">Pick a reason</div>
            {REJECT_REASONS.map((r) => (
              <button key={r} type="button" disabled={busy} onClick={() => go('reject', r)}
                className="block h-12 w-full rounded-[10px] border border-critical/50 px-4 text-left text-body text-critical-ink active:bg-critical/15">{r}</button>
            ))}
            <button type="button" onClick={() => setRejecting(false)} className="h-11 w-full text-body text-ink-3">Cancel</button>
          </div>
        ) : (
          <div className="grid grid-cols-[1fr_auto] gap-3">
            <button type="button" disabled={busy} onClick={() => go('approve')}
              className={clsx('flex h-14 items-center justify-center gap-2 rounded-[12px] bg-ink-1 text-[17px] font-semibold text-page active:opacity-80', busy && 'opacity-60')}>
              <Check size={20} /> Approve
            </button>
            <button type="button" disabled={busy} onClick={() => setRejecting(true)}
              className="flex h-14 items-center justify-center gap-2 rounded-[12px] border border-critical/50 px-5 text-[17px] font-medium text-critical-ink active:bg-critical/15">
              <X size={20} /> Reject
            </button>
          </div>
        )}
      </div>
    </article>
  )
}
