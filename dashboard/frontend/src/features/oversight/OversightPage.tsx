import clsx from 'clsx'
import { ArrowUpRight, BellRing, Check, ChevronsUp, Loader2, PenLine, ShieldCheck, X } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '@/core/api'
import { useOnBeat } from '@/core/director'
import { clockTime, duration, simOf } from '@/core/format'
import { EMPTY, useLive } from '@/core/store'
import type { Approval, Rec } from '@/core/types'
import { toast, useSimNow, useUi } from '@/core/ui-store'
import { Button, LedgerLink, Meter, PageHeader, Panel, Segmented, SignatureBadge, StatusPill, Tabs } from '@/ui'
import manifest from './manifest'
import { Block, FourEyes, ImpactBars, SlaCountdown, TierPill } from './parts'

const NO_HUMANS: Rec = {}

const SRC = { aws: 'Step Functions task tokens', mock: 'Simulated approval queue' }

export default function OversightPage() {
  const items: Approval[] = useLive((s) => s.oversight.items ?? EMPTY)
  const humans: Rec = useLive((s) => s.oversight.humans ?? NO_HUMANS)
  const onDuty = useLive((s) => s.oversight.on_duty ?? 'a.rivera')
  const [sel, setSel] = useState<string | null>(null)
  const pinned = useRef(false)
  const topPending = items.find((i) => i.status === 'pending')?.id

  // follow new requests unless the operator picked one by hand
  useEffect(() => {
    if (!sel || !items.some((i) => i.id === sel) || (!pinned.current && topPending && topPending !== sel)) {
      setSel(topPending ?? items[0]?.id ?? null)
    }
  }, [topPending, items, sel])
  useOnBeat(9, () => { pinned.current = false; if (topPending) setSel(topPending) })

  const selected = items.find((i) => i.id === sel) ?? null
  const duty = humans[onDuty]
  return (
    <div>
      <PageHeader title={manifest.title} description={manifest.description} tracks={manifest.tracks}
        actions={duty && (
          <span className="flex h-8 items-center gap-2 rounded-md border border-line px-2.5 text-cap">
            <span className="size-2 rounded-full bg-good" />
            <span className="text-ink-3">On duty</span>
            <span className="font-medium text-ink-1">{duty.name}</span>
            <span className="text-ink-3">{duty.level} {duty.role}</span>
          </span>
        )} />
      <Stats items={items} />
      <div className="mt-4 grid grid-cols-12 gap-4">
        <div className="col-span-12 xl:col-span-4"><Queue items={items} sel={sel} onSelect={(id) => { pinned.current = true; setSel(id) }} /></div>
        <div className="col-span-12 xl:col-span-8">
          {selected ? <Workspace a={selected} /> : (
            <Panel id="decision" plane="operate" title="Decision"><div className="py-16 text-center text-cap text-ink-3">Nothing waiting. Agents are acting within their autonomy level.</div></Panel>
          )}
        </div>
        <div className="col-span-12"><Ladder a={selected} /></div>
        <div className="col-span-12"><History items={items} /></div>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ stats

function Stats({ items }: { items: Approval[] }) {
  const pending = items.filter((i) => i.status === 'pending')
  const inReview = pending.filter((i) => i.approvals.length > 0)
  const ttd = items.map((i) => i.ttd_s).filter((x): x is number => x != null).sort((a, b) => a - b)
  const median = ttd.length ? ttd[Math.floor(ttd.length / 2)] : null
  const escalations = items.reduce((n, i) => n + i.history.filter((h) => h.event.startsWith('escalated to')).length, 0)
  const decided = items.filter((i) => i.status !== 'pending').length
  const tiles: [string, string, string?][] = [
    ['Waiting for a human', String(pending.length), pending.length ? 'oldest first by SLA' : 'queue clear'],
    ['In review', String(inReview.length), 'four-eyes, one approval in'],
    ['Median time to decision', median != null ? duration(median) : '—', `${decided} decided today`],
    ['Escalations', String(escalations), 'auto on SLA breach or by hand'],
    ['Signed decisions', String(items.reduce((n, i) => n + i.approvals.length, 0)), 'Ed25519, one key per person'],
  ]
  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
      {tiles.map(([l, v, s]) => (
        <div key={l} className="rounded-[10px] border border-line bg-s1 px-4 py-3">
          <div className="text-micro text-ink-3">{l}</div>
          <div className="tnum text-[22px] leading-7 font-semibold text-ink-1">{v}</div>
          {s && <div className="text-micro text-ink-3">{s}</div>}
        </div>
      ))}
    </div>
  )
}

// ------------------------------------------------------------------ queue

function Queue({ items, sel, onSelect }: { items: Approval[]; sel: string | null; onSelect: (id: string) => void }) {
  const now = useSimNow(4)
  const [tab, setTab] = useState<'pending' | 'decided'>('pending')
  const pending = items.filter((i) => i.status === 'pending').sort((a, b) => a.sla_deadline_t - b.sla_deadline_t)
  const decided = items.filter((i) => i.status !== 'pending')
  const list = tab === 'pending' ? pending : decided
  return (
    <Panel id="queue" plane="operate" title="Queue" source={SRC} flush>
      <div className="px-4 pt-3"><Tabs value={tab} onChange={setTab} tabs={[{ value: 'pending', label: 'Pending', count: pending.length }, { value: 'decided', label: 'Decided', count: decided.length }]} /></div>
      <ul className="max-h-[640px] overflow-auto py-1">
        {list.length === 0 && <li className="px-4 py-10 text-center text-cap text-ink-3">{tab === 'pending' ? 'No decisions waiting.' : 'No decisions yet.'}</li>}
        {list.map((a) => {
          const left = a.sla_deadline_t - now
          return (
            <li key={a.id}>
              <button type="button" onClick={() => onSelect(a.id)}
                className={clsx('row-in relative w-full border-b border-line/60 px-4 py-3 text-left transition-colors', a.id === sel ? 'bg-s2' : 'hover:bg-s2/60')}>
                {a.id === sel && <span className="absolute inset-y-2 left-0 w-[2px] rounded-full bg-operate" />}
                <div className="flex items-center gap-2">
                  <span className="mono text-cap text-ink-2">{a.id}</span>
                  <TierPill tier={a.tier} size="sm" />
                  <span className="ml-auto"><SlaCountdown a={a} now={now} /></span>
                </div>
                <div className="mt-1 line-clamp-2 text-body text-ink-1">{a.title}</div>
                <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-micro text-ink-3">
                  <span>from <span className="mono text-ink-2">{a.requested_by}</span></span>
                  {a.blast_radius_pct != null && <span>blast radius <span className="tnum text-ink-2">{a.blast_radius_pct}%</span></span>}
                  <span className="tnum">{a.approvals.length}/{a.required} approvals</span>
                  {a.status === 'pending' ? <span>{a.level}, {a.assignee.name}</span> : <StatusPill size="sm" status={a.status === 'approved' ? 'approved' : a.status === 'rejected' ? 'rejected' : 'neutral'} />}
                </div>
                {a.status === 'pending' && (
                  <div className="mt-2 h-0.5 overflow-hidden rounded-full bg-s3">
                    <div className="h-full rounded-full transition-[width] duration-300" style={{
                      width: `${Math.max(0, Math.min(1, left / a.sla_s)) * 100}%`,
                      background: left < 15 ? 'var(--critical)' : left < 60 ? 'var(--warn)' : 'var(--operate)',
                    }} />
                  </div>
                )}
              </button>
            </li>
          )
        })}
      </ul>
    </Panel>
  )
}

// ------------------------------------------------------------------ workspace

type Mode = 'idle' | 'changes' | 'reject' | 'escalate'
const REASONS: Record<'reject' | 'escalate', string[]> = {
  reject: ['Blast radius too large for tonight', 'Wait for the field team', 'Prediction does not match what I see'],
  escalate: ['Needs on-call review', 'Touches a life-safety site', 'Outside my authority'],
}

function Workspace({ a }: { a: Approval }) {
  const now = useSimNow(4)
  const navigate = useNavigate()
  const inspect = useUi((s) => s.inspect)
  const policies: Rec[] = useLive((s) => s.trust.policies ?? EMPTY)
  const humans: Rec = useLive((s) => s.oversight.humans ?? NO_HUMANS)
  const incident = useLive((s) => s.swarm.incident)
  const [by, setBy] = useState('a.rivera')
  const [mode, setMode] = useState<Mode>('idle')
  const [reason, setReason] = useState('')
  const [limits, setLimits] = useState<Record<string, number>>({})
  const [ceremony, setCeremony] = useState<{ step: number; by: string; seq?: number } | null>(null)
  useEffect(() => { setMode('idle'); setReason(''); setLimits({}) }, [a.id])

  const pending = a.status === 'pending'
  const steps: Rec[] = a.plan?.steps ?? []
  const alreadySigned = a.approvals.some((x) => x.by === by)
  const requestSeq = a.history[0]?.ledger_seq
  const cellSteps = steps.filter((s) => Array.isArray(s.params?.cells) && s.state !== 'done')

  const decide = async (decision: 'approve' | 'approve_with_changes' | 'reject' | 'escalate') => {
    const changes = decision === 'approve_with_changes'
      ? Object.fromEntries(Object.entries(limits).map(([n, k]) => {
        const st = steps.find((s) => String(s.n) === n)
        return [`step_${n}`, { cells: (st?.params?.cells ?? []).slice(0, k) }]
      }))
      : undefined
    const signing = decision === 'approve' || decision === 'approve_with_changes'
    if (signing) setCeremony({ step: 0, by })
    try {
      const res = await api.decide(a.id, decision, { by, device: 'console', reason, changes })
      const item: Approval = res.item
      const mine = item.approvals.find((x) => x.by === by)
      if (signing) {
        setCeremony({ step: 1, by })
        setTimeout(() => setCeremony({ step: 2, by, seq: mine?.ledger_seq }), 380)
        setTimeout(() => setCeremony({ step: 3, by, seq: mine?.ledger_seq }), 760)
        setTimeout(() => setCeremony(null), 6000)
      } else {
        toast({ severity: 'notice', title: `${a.id} ${decision === 'reject' ? 'rejected' : 'escalated'} and signed into the ledger` })
      }
      setMode('idle')
      setReason('')
    } catch (e) {
      setCeremony(null)
      toast({ severity: 'critical', title: `Decision failed: ${(e as Error).message}` })
    }
  }

  return (
    <Panel id="decision" plane="operate" title={<span className="flex items-center gap-2"><span className="mono">{a.id}</span><TierPill tier={a.tier} size="sm" /></span>}
      subtitle={pending ? `${a.level}, assigned to ${a.assignee.name}` : `decided ${a.status}`} source={SRC}
      actions={pending ? <SlaCountdown a={a} now={now} big /> : <StatusPill status={a.status === 'approved' ? 'approved' : a.status === 'rejected' ? 'rejected' : 'neutral'} />}>
      <h2 className="text-section font-semibold text-ink-1">{a.title}</h2>
      <p className="mt-1.5 max-w-[78ch] text-body text-ink-2">{a.summary}</p>

      <div className="mt-5 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Block title="Predicted impact (twin pre-flight)" right={a.incident && <span className="text-micro text-ink-3">{a.incident}</span>}>
          <ImpactBars preflight={a.preflight} />
          {a.blast_radius_pct != null && (
            <div className="mt-4">
              <Meter value={a.blast_radius_pct} max={25} warnAt={5} critAt={20} label="Blast radius, share of cells touched" right={`${a.blast_radius_pct}% (rule: 5% without a human)`} />
            </div>
          )}
        </Block>
        <div className="space-y-5">
          <Block title="Why it needs you">
            <ul className="space-y-1.5">
              {a.reasons.map((r) => (
                <li key={r.policy} className="flex items-start gap-2 text-cap">
                  <button type="button" onClick={() => { const p = policies.find((x) => x.id === r.policy); if (p) inspect({ kind: 'policy', id: p.id, title: `${p.id} ${p.name}`, data: p }) }}
                    className="mono shrink-0 rounded border border-line px-1.5 text-micro leading-[18px] text-ink-1 hover:bg-s2" title="Open the Cedar policy">{r.policy}</button>
                  <span className="text-ink-1">{r.text}</span>
                </li>
              ))}
            </ul>
          </Block>
          <Block title="Dissent">
            {a.dissent.length ? a.dissent.map((d) => (
              <div key={d.agent} className="rounded-md border-l-2 border-warn bg-warn/10 px-3 py-2 text-cap">
                <button type="button" className="mono font-medium text-ink-1 hover:underline" onClick={() => inspect({ kind: 'agent', id: d.agent })}>{d.agent}</button>
                <span className="text-ink-2">: {d.text}</span>
              </div>
            )) : <div className="text-cap text-ink-3">No agent objected.</div>}
          </Block>
          <Block title="What the models proposed">
            {a.model_opinions.length ? (
              <ul className="space-y-1.5">
                {a.model_opinions.map((m) => (
                  <li key={m.model} className="flex items-center gap-2 text-cap">
                    <span className="mono w-16 shrink-0 text-ink-1">{m.model}</span>
                    <span className="w-20 shrink-0 text-micro text-ink-3">{m.role}</span>
                    <span className="text-ink-2">{m.proposal}</span>
                  </li>
                ))}
              </ul>
            ) : <div className="text-cap text-ink-3">Not model-generated.</div>}
          </Block>
        </div>
      </div>

      {steps.length > 0 && (
        <Block title="Plan" className="mt-6">
          <ol className="divide-y divide-line/60 rounded-md border border-line">
            {steps.map((s) => (
              <li key={s.n} className="grid grid-cols-[24px_150px_1fr_auto_auto] items-center gap-3 px-3 py-2 text-cap">
                <span className="tnum text-ink-3">{s.n}</span>
                <span className="mono truncate text-ink-1">{s.tool}</span>
                <span className="truncate text-ink-2">{s.target}{s.agent ? <span className="text-ink-3"> by {s.agent}</span> : null}</span>
                <span className="mono text-micro text-ink-3">{s.policy}</span>
                <StatusPill size="sm" status={s.state === 'done' ? 'passed' : s.verdict === 'needs_human' ? 'pending' : s.verdict === 'denied' ? 'denied' : 'verified'}>
                  {s.state === 'done' ? (s.verdict === 'auto' ? 'Done, auto' : 'Done') : s.verdict === 'needs_human' ? 'Needs you' : s.verdict}
                </StatusPill>
              </li>
            ))}
          </ol>
        </Block>
      )}

      <div className="mt-5 flex flex-wrap items-center gap-x-4 gap-y-2 text-cap">
        <span className="text-ink-3">Evidence</span>
        {requestSeq && <button type="button" className="flex items-center gap-1 text-ink-1 hover:underline" onClick={() => inspect({ kind: 'ledger', id: String(requestSeq) })}>request record #{requestSeq} <ArrowUpRight size={12} /></button>}
        {a.incident && <button type="button" className="flex items-center gap-1 text-ink-1 hover:underline" onClick={() => navigate('/swarm')}>{a.incident} in Swarm Tower <ArrowUpRight size={12} /></button>}
        {a.incident && incident?.trace_id && <button type="button" className="flex items-center gap-1 text-ink-1 hover:underline" onClick={() => navigate('/observability')}>trace {incident.trace_id.slice(0, 8)} <ArrowUpRight size={12} /></button>}
        <button type="button" className="flex items-center gap-1 text-ink-1 hover:underline" onClick={() => inspect({ kind: 'approval', id: a.id, title: a.title, data: a })}>raw request <ArrowUpRight size={12} /></button>
      </div>

      {/* decision bar */}
      <div className="mt-6 rounded-[10px] border border-line bg-s2 p-4">
        <div className="mb-3 flex flex-wrap items-center gap-3">
          <FourEyes a={a} />
          {pending && (
            <span className="ml-auto flex items-center gap-2 text-cap">
              <span className="text-ink-3">Signing as</span>
              <Segmented size="sm" value={by} onChange={setBy} ariaLabel="Approver"
                options={Object.values(humans).map((h: Rec) => ({ value: h.id, label: h.name, title: `${h.level} ${h.role}` }))} />
            </span>
          )}
        </div>

        {pending && mode === 'idle' && (
          <div className="flex flex-wrap gap-2">
            <Button variant="primary" icon={<Check size={14} />} disabled={alreadySigned} onClick={() => decide('approve')}>
              {alreadySigned ? `${humans[by]?.name} already approved` : a.required > 1 && a.approvals.length === 0 ? 'Approve (1 of 2)' : 'Approve'}
            </Button>
            <Button icon={<PenLine size={14} />} disabled={alreadySigned || !cellSteps.length} onClick={() => setMode('changes')}
              title={cellSteps.length ? undefined : 'This plan has no adjustable steps'}>Approve with changes</Button>
            <Button variant="danger" icon={<X size={14} />} onClick={() => setMode('reject')}>Reject</Button>
            <Button variant="ghost" icon={<ChevronsUp size={14} />} disabled={a.level_index >= 2} onClick={() => setMode('escalate')}>Escalate to {['L2', 'L3', '—'][a.level_index]}</Button>
          </div>
        )}

        {pending && mode === 'changes' && (
          <div className="space-y-3">
            <div className="text-cap text-ink-2">Limit how many cells each step may touch. The change is signed with your approval.</div>
            {cellSteps.map((s) => {
              const total = s.params.cells.length
              const k = limits[s.n] ?? total
              return (
                <label key={s.n} className="flex items-center gap-3 text-cap">
                  <span className="mono w-28 text-ink-1">{s.n}. {s.tool}</span>
                  <input type="range" min={1} max={total} value={k} onChange={(e) => setLimits((x) => ({ ...x, [s.n]: Number(e.target.value) }))} className="flex-1 accent-[var(--release)]" />
                  <span className="tnum w-24 text-right text-ink-1">{k} of {total} cells</span>
                </label>
              )
            })}
            <div className="flex gap-2">
              <Button variant="primary" icon={<Check size={14} />} onClick={() => decide('approve_with_changes')}>Approve with these limits</Button>
              <Button variant="ghost" onClick={() => setMode('idle')}>Cancel</Button>
            </div>
          </div>
        )}

        {pending && (mode === 'reject' || mode === 'escalate') && (
          <div className="space-y-2">
            <div className="text-cap text-ink-2">{mode === 'reject' ? 'Why are you rejecting this plan? The reason is signed and kept in the ledger.' : 'Why does this need the next level? They are paged immediately.'}</div>
            <div className="flex flex-wrap gap-1.5">
              {REASONS[mode].map((r) => (
                <button key={r} type="button" onClick={() => setReason(r)} className={clsx('rounded-md border px-2 py-1 text-micro', reason === r ? 'border-ink-2 text-ink-1' : 'border-line text-ink-2 hover:text-ink-1')}>{r}</button>
              ))}
            </div>
            <textarea value={reason} onChange={(e) => setReason(e.target.value)} rows={2} placeholder="Reason (required)"
              className="w-full rounded-md border border-line bg-s1 px-3 py-2 text-body text-ink-1 placeholder:text-ink-3" />
            <div className="flex gap-2">
              <Button variant={mode === 'reject' ? 'danger' : 'secondary'} disabled={!reason.trim()} onClick={() => decide(mode)}>
                {mode === 'reject' ? 'Reject plan' : `Escalate to ${['L2', 'L3', '—'][a.level_index]}`}
              </Button>
              <Button variant="ghost" onClick={() => setMode('idle')}>Cancel</Button>
            </div>
          </div>
        )}

        {ceremony && <Ceremony c={ceremony} name={humans[ceremony.by]?.name ?? ceremony.by} />}

        {a.approvals.length > 0 && !ceremony && (
          <ul className="mt-3 space-y-1.5 border-t border-line pt-3">
            {a.approvals.map((ap) => (
              <li key={ap.ledger_seq} className="flex flex-wrap items-center gap-x-3 gap-y-1 text-cap">
                <ShieldCheck size={14} className="text-good-ink" />
                <span className="text-ink-1">Approved by {ap.name}</span>
                <span className="text-ink-3">on {ap.device}, {clockTime(ap.at, false)}</span>
                <SignatureBadge keyId={`kms:spectrum/human/${ap.by}`} ledgerSeq={ap.ledger_seq} />
                <LedgerLink seq={ap.ledger_seq} />
              </li>
            ))}
          </ul>
        )}
        {!pending && a.approvals.length === 0 && (
          <div className="text-cap text-ink-3">{a.history[a.history.length - 1]?.event}</div>
        )}
      </div>
    </Panel>
  )
}

function Ceremony({ c, name }: { c: { step: number; by: string; seq?: number }; name: string }) {
  const steps = [
    'Canonicalising the decision',
    `Signing with kms:spectrum/human/${c.by} (Ed25519)`,
    c.seq ? `Appended to the ledger as record #${c.seq.toLocaleString()}` : 'Appending to the ledger',
  ]
  return (
    <div className="mt-3 rounded-md border border-good/40 bg-good/10 p-3">
      <div className="mb-2 text-cap font-medium text-ink-1">{c.step >= 3 ? `Signed by ${name}` : `Signing as ${name}…`}</div>
      <ol className="space-y-1">
        {steps.map((s, i) => (
          <li key={s} className={clsx('flex items-center gap-2 text-cap', i < c.step ? 'text-ink-1' : 'text-ink-3')}>
            {i < c.step ? <Check size={13} className="text-good-ink" /> : i === c.step ? <Loader2 size={13} className="animate-spin" /> : <span className="size-[13px]" />}
            <span className={i === 1 ? 'mono text-[12px]' : ''}>{s}</span>
          </li>
        ))}
      </ol>
    </div>
  )
}

// ------------------------------------------------------------------ escalation ladder

function Ladder({ a }: { a: Approval | null }) {
  const ladder: Rec[] = useLive((s) => s.oversight.ladder ?? EMPTY)
  const humans: Rec = useLive((s) => s.oversight.humans ?? NO_HUMANS)
  const now = useSimNow(4)
  const escalatedAt = (lvl: string) => a?.history.find((h) => h.event.startsWith(`escalated to ${lvl}`))?.ts
  const started = a ? ((a as any).level_started_t as number | undefined) ?? a.sla_deadline_t - a.sla_s : 0
  return (
    <Panel id="ladder" plane="operate" title="Escalation ladder" subtitle={a ? `for ${a.id}` : 'no request selected'} source={{ aws: 'SNS + Incident Manager', mock: 'Simulated paging' }}>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-[1fr_auto_1fr_auto_1fr]">
        {ladder.map((l, i) => {
          const h = humans[l.who]
          const current = a && a.status === 'pending' && a.level_index === i
          const past = a && a.level_index > i
          const elapsed = current ? now - started : 0
          const k = current ? Math.min(1, elapsed / l.sla_s) : past ? 1 : 0
          const left = l.sla_s - elapsed
          return (
            <div key={l.level} className="contents">
              {i > 0 && <div className="hidden items-center text-ink-3 md:flex">→</div>}
              <div className={clsx('rounded-md border p-3', current ? 'border-operate bg-s2' : 'border-line')}>
                <div className="flex items-center gap-2">
                  <span className="tnum rounded bg-s3 px-1.5 text-cap font-semibold text-ink-1">{l.level}</span>
                  <span className="text-body font-medium text-ink-1">{h?.name}</span>
                  <span className="text-micro text-ink-3">{l.label}</span>
                  {current && <BellRing size={14} className="ml-auto text-warn" aria-label="paged" />}
                </div>
                <div className="mt-2 h-1 overflow-hidden rounded-full bg-s3">
                  <div className="h-full rounded-full transition-[width] duration-300"
                    style={{ width: `${k * 100}%`, background: past ? 'var(--ink-3)' : left < 15 ? 'var(--critical)' : left < 60 ? 'var(--warn)' : 'var(--operate)' }} />
                </div>
                <div className="mt-1.5 flex justify-between text-micro">
                  <span className="text-ink-3">SLA {duration(l.sla_s)}</span>
                  <span className="tnum text-ink-2">
                    {current ? (left > 0 ? `${duration(left)} left` : 'escalating') : past ? `escalated ${clockTime(escalatedAt(ladder[i + 1]?.level) ?? '', false)}` : a?.status === 'pending' ? 'standby' : '—'}
                  </span>
                </div>
              </div>
            </div>
          )
        })}
      </div>
      <p className="mt-3 text-micro text-ink-3">Unanswered requests climb a level when the SLA runs out. Each step pages the next person (SNS / Incident Manager) and writes a signed escalation record; the agents keep waiting, they never act on silence.</p>
    </Panel>
  )
}

// ------------------------------------------------------------------ history

function History({ items }: { items: Approval[] }) {
  const inspect = useUi((s) => s.inspect)
  const rows = useMemo(() => {
    const out: { ts: string; a: Approval; decision: string; by: string; device: string; seq?: number; key: string }[] = []
    for (const a of items) {
      for (const ap of a.approvals) out.push({ ts: ap.at, a, decision: 'approved', by: ap.name, device: ap.device, seq: ap.ledger_seq, key: `kms:spectrum/human/${ap.by}` })
      for (const h of a.history) {
        if (h.event.startsWith('rejected') || (h.event.startsWith('escalated by'))) {
          out.push({ ts: h.ts, a, decision: h.event.startsWith('rejected') ? 'rejected' : 'escalated', by: h.by, device: 'console', seq: h.ledger_seq, key: `kms:spectrum/human/${h.by}` })
        } else if (h.event.startsWith('escalated to') && h.event.includes('SLA')) {
          out.push({ ts: h.ts, a, decision: 'auto-escalated', by: 'SLA breach', device: 'system', seq: h.ledger_seq, key: 'kms:spectrum/oversight' })
        }
      }
    }
    return out.sort((x, y) => simOf(y.ts) - simOf(x.ts))
  }, [items])
  return (
    <Panel id="history" plane="prove" title="Decision history" subtitle="every human decision is signed and ledgered" source={{ aws: 'DynamoDB ledger + KMS', mock: 'Real Ed25519 signatures' }} flush>
      <div className="max-h-[320px] overflow-auto">
        <table className="w-full text-cap">
          <thead className="sticky top-0 bg-s1">
            <tr className="border-b border-line text-left text-micro text-ink-3">
              {['Time', 'Request', 'Title', 'Decision', 'By', 'Device', 'Signature', 'Ledger', 'Time to decide'].map((h) => <th key={h} className="px-3 py-2 font-medium">{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={`${r.a.id}-${i}`} className="border-b border-line/50 hover:bg-s2/50">
                <td className="tnum mono px-3 py-1.5 text-ink-3">{clockTime(r.ts, false)}</td>
                <td className="px-3"><button type="button" className="mono text-ink-1 hover:underline" onClick={() => inspect({ kind: 'approval', id: r.a.id, title: r.a.title, data: r.a })}>{r.a.id}</button></td>
                <td className="max-w-[340px] truncate px-3 text-ink-1">{r.a.title}</td>
                <td className="px-3"><StatusPill size="sm" status={r.decision === 'approved' ? 'approved' : r.decision === 'rejected' ? 'rejected' : 'warning'}>{r.decision}</StatusPill></td>
                <td className="px-3 text-ink-1">{r.by}</td>
                <td className="px-3 text-ink-2">{r.device}</td>
                <td className="px-3"><SignatureBadge keyId={r.key} ledgerSeq={r.seq} compact /></td>
                <td className="px-3"><LedgerLink seq={r.seq} /></td>
                <td className="tnum px-3 text-ink-2">{r.a.ttd_s != null && r.decision === 'approved' ? duration(r.a.ttd_s) : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}
