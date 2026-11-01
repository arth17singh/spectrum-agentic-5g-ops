import clsx from 'clsx'
import { ArrowRight, Lock } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { LineChart, type Series } from '@/charts/LineChart'
import { Sparkline } from '@/charts/Sparkline'
import { api } from '@/core/api'
import { useFlag } from '@/core/flags'
import { clockTime, compact, usd } from '@/core/format'
import { simClockLabel, toast, useUi } from '@/core/ui-store'
import { EMPTY, useLive } from '@/core/store'
import type { A2AMessage, Agent, Rec } from '@/core/types'
import { Button, Meter, Panel, Segmented, SignatureBadge, StatusPill } from '@/ui'

const SRC_SWARM = { aws: 'Bedrock AgentCore', mock: 'Simulated swarm' }
const NONE: Rec = {}

// ------------------------------------------------------------------ A2A message stream

const INTENT_COLOR: Record<string, string> = {
  inform: 'var(--ink-3)', request: 'var(--release)', propose: 'var(--operate)', object: 'var(--warn)', delegate: 'var(--learn)', accept: 'var(--good)',
}

export function MessageStream() {
  const msgs = useLive((s) => s.swarm.messages ?? EMPTY) as A2AMessage[]
  const inspect = useUi((s) => s.inspect)
  const [filter, setFilter] = useState<'all' | 'coordination' | 'blocked'>('all')
  const rows = useMemo(() => {
    const out: A2AMessage[] = []
    for (let i = msgs.length - 1; i >= 0 && out.length < 120; i--) {
      const m = msgs[i]
      if (filter === 'coordination' && m.intent === 'inform' && m.to === 'coordinator' && m.verdict !== 'blocked') continue
      if (filter === 'blocked' && m.verdict !== 'blocked') continue
      out.push(m)
    }
    return out
  }, [msgs, filter])
  const blocked = msgs.filter((m) => m.verdict === 'blocked').length
  return (
    <Panel id="messages" plane="operate" title="Agent-to-agent messages" subtitle="every message signed by its sender, checked and ledgered" source={{ aws: 'AgentCore A2A + EventBridge', mock: 'Simulated A2A bus' }}
      actions={<Segmented size="sm" value={filter} onChange={setFilter} ariaLabel="Message filter" options={[
        { value: 'coordination', label: 'Coordination' }, { value: 'all', label: 'All' }, { value: 'blocked', label: `Blocked ${blocked}` }]} />} flush>
      <div className="max-h-[360px] overflow-auto">
        <table className="w-full table-fixed text-cap">
          <colgroup>
            <col className="w-[104px]" /><col className="w-[190px]" /><col className="w-[84px]" /><col /><col className="w-[128px]" /><col className="w-[196px]" />
          </colgroup>
          <thead className="sticky top-0 z-10 bg-s1">
            <tr className="border-b border-line text-left text-micro text-ink-3">
              {['Time', 'From → to', 'Intent', 'Message', 'Delegation chain', 'Signature and verdict'].map((h) => <th key={h} className="px-3 py-2 font-medium">{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((m) => (
              <tr key={m.id} onClick={() => inspect({ kind: 'message', id: m.id, title: m.summary, data: m })}
                className={clsx('row-in cursor-pointer border-b border-line/60 hover:bg-s2/60', m.verdict === 'blocked' && 'bg-critical/[0.07]')}>
                <td className="tnum mono px-3 py-1.5 whitespace-nowrap text-ink-3">{clockTime(m.ts)}</td>
                <td className="mono truncate px-3 whitespace-nowrap text-ink-1">{m.from} <ArrowRight size={11} className="inline text-ink-3" /> {m.to}</td>
                <td className="px-3"><span className="inline-flex items-center gap-1.5 text-ink-2"><span className="size-1.5 rounded-full" style={{ background: INTENT_COLOR[m.intent] ?? 'var(--ink-3)' }} />{m.intent}</span></td>
                <td className="truncate px-3 text-ink-1" title={m.summary}>{m.summary}</td>
                <td className="truncate px-3 whitespace-nowrap" title={m.chain.join(' → ')}>
                  <span className="mono rounded bg-s3 px-1 text-micro text-ink-2">{m.chain.join(' → ')}</span>
                </td>
                <td className="px-3 py-1">
                  <span className="flex flex-wrap items-center gap-1">
                    <SignatureBadge state={m.verified ? 'verified' : 'failed'} keyId={m.key_id} compact />
                    <StatusPill status={m.verdict === 'blocked' ? 'blocked' : 'verified'} size="sm">{m.verdict === 'blocked' ? 'Blocked' : 'Delivered'}</StatusPill>
                    {!m.authority_ok && <StatusPill status="critical" size="sm" title={`claims ${m.claimed_authority}, signed by ${m.from}`}>authority mismatch</StatusPill>}
                    {m.guardrail && <StatusPill status="critical" size="sm">{m.guardrail.replace('_', ' ')}</StatusPill>}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {!rows.length && <div className="p-6 text-center text-cap text-ink-3">{filter === 'blocked' ? 'No blocked messages. Signed, authorised messages are delivered.' : 'No messages yet.'}</div>}
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ agent roster

const CLASS_ORDER: Record<string, number> = { coordinator: 0, domain: 1, site: 2 }

export function Roster() {
  const agents = useLive((s) => s.swarm.agents ?? EMPTY) as Agent[]
  const inspect = useUi((s) => s.inspect)
  const hist = useRef<Map<string, number[]>>(new Map())
  const [cls, setCls] = useState<'all' | 'site' | 'domain'>('all')
  // per-agent trust history, sampled client-side as agent updates arrive
  useEffect(() => {
    for (const a of agents) {
      const h = hist.current.get(a.id) ?? []
      if (h[h.length - 1] !== a.trust) h.push(a.trust)
      if (h.length > 40) h.shift()
      hist.current.set(a.id, h)
    }
  }, [agents])
  const rows = useMemo(() => [...agents]
    .filter((a) => cls === 'all' || (cls === 'site' ? a.class === 'site' : a.class !== 'site'))
    .sort((a, b) => (a.status === 'quarantined' ? -1 : 0) - (b.status === 'quarantined' ? -1 : 0) || CLASS_ORDER[a.class] - CLASS_ORDER[b.class] || a.id.localeCompare(b.id)), [agents, cls])
  return (
    <Panel id="roster" plane="operate" title="Agent roster" subtitle="identity, model, autonomy, earned trust" source={SRC_SWARM}
      actions={<Segmented size="sm" value={cls} onChange={setCls} options={[{ value: 'all', label: 'All' }, { value: 'domain', label: 'Coordinator + domain' }, { value: 'site', label: 'Site' }]} />} flush>
      <div className="max-h-[420px] overflow-auto">
        <table className="w-full text-cap">
          <thead className="sticky top-0 z-10 bg-s1">
            <tr className="border-b border-line text-left text-micro text-ink-3">
              {['Agent', 'Class', 'Model', 'Autonomy', 'Trust', 'State', 'Msgs/min', 'Tokens 1h', 'Cost 1h', 'Last action', ''].map((h) => <th key={h} className="px-3 py-2 font-medium">{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((a) => {
              const q = a.status === 'quarantined'
              return (
                <tr key={a.id} className={clsx('border-b border-line/60 hover:bg-s2/60', q && 'bg-s2')}>
                  <td className="px-3 py-1.5">
                    <button type="button" className="mono flex items-center gap-1.5 font-medium text-ink-1 hover:underline" onClick={() => inspect({ kind: 'agent', id: a.id })}>
                      {q && <Lock size={11} className="text-critical-ink" />}{a.id}
                    </button>
                  </td>
                  <td className="px-3 text-ink-2">{a.class}</td>
                  <td className="px-3 whitespace-nowrap">
                    <span className="mono text-ink-1">{a.model}</span> <span className="text-ink-3">{a.provider}</span>
                    {a.class === 'site' && a.model !== 'nlm-v6' && <span className="ml-1 rounded bg-warn/15 px-1 text-micro text-warn">canary</span>}
                  </td>
                  <td className="px-3"><span className="tnum rounded border border-line px-1.5 text-micro font-medium text-ink-1">L{a.autonomy}</span></td>
                  <td className="px-3">
                    <span className="flex items-center gap-2">
                      <span className="tnum w-6 text-right" style={{ color: a.trust < 50 ? 'var(--critical-ink)' : 'var(--ink-1)' }}>{a.trust.toFixed(0)}</span>
                      <span className="w-14"><Sparkline values={hist.current.get(a.id) ?? [a.trust]} domain={[0, 100]} height={16} color={a.trust < 50 ? 'var(--critical)' : 'var(--operate)'} /></span>
                    </span>
                  </td>
                  <td className="px-3">
                    {q ? <StatusPill status="quarantined" size="sm" /> : (
                      <span className="inline-flex items-center gap-1.5 text-ink-2">
                        <span className={clsx('size-2 rounded-full', a.state === 'thinking' && 'live-dot')} style={{ background: a.state === 'thinking' ? 'var(--release)' : a.state === 'acting' ? 'var(--operate)' : 'var(--surface-3)', color: 'var(--release)' }} />
                        {a.state}
                      </span>
                    )}
                  </td>
                  <td className="tnum px-3 text-ink-2">{a.msgs_min.toFixed(1)}</td>
                  <td className="tnum px-3 text-ink-2">{compact(a.tokens_1h)}</td>
                  <td className="tnum px-3 text-ink-2">{usd(a.usd_1h)}</td>
                  <td className="mono max-w-[200px] truncate px-3 text-ink-3" title={a.last_action}>{a.last_action || '—'}</td>
                  <td className="px-3 text-right">
                    {q ? <Button size="sm" variant="ghost" onClick={() => api.agent(a.id, 'release').then(() => toast({ severity: 'notice', title: `${a.id} released with a new identity` }))}>Release</Button>
                      : <Button size="sm" variant="ghost" onClick={() => api.agent(a.id, 'quarantine').then(() => toast({ severity: 'warn', title: `${a.id} quarantined (identity revoked)` }))}>Quarantine</Button>}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ circuit breakers

export function Breakers() {
  const b = useLive((s) => s.swarm.breakers ?? NONE)
  const frozen = useLive((s) => s.swarm.frozen)
  return (
    <Panel id="breakers" plane="operate" title="Circuit breakers" subtitle="hard limits that pause the swarm when tripped" source={SRC_SWARM}
      actions={frozen ? <StatusPill status="critical" size="sm">Swarm frozen</StatusPill> : (b.tripped ?? []).length ? <StatusPill status="critical" size="sm">Tripped</StatusPill> : <StatusPill status="good" size="sm">All clear</StatusPill>}>
      <div className="space-y-4">
        <Meter label="Blast-radius budget, this incident" value={b.blast_budget_used ?? 0} max={b.blast_budget ?? 20} warnAt={(b.blast_budget ?? 20) * 0.75} critAt={b.blast_budget ?? 20}
          right={`${(b.blast_budget_used ?? 0).toFixed(1)}% of ${b.blast_budget ?? 20}%`} color="var(--operate)" />
        <Meter label="Network changes, last hour" value={b.changes_hour ?? 0} max={b.changes_limit ?? 30} warnAt={(b.changes_limit ?? 30) * 0.75} critAt={b.changes_limit ?? 30}
          right={`${b.changes_hour ?? 0} of ${b.changes_limit ?? 30}`} color="var(--operate)" />
        <Meter label="A2A message rate" value={b.msg_rate ?? 0} max={b.msg_limit ?? 40} warnAt={(b.msg_limit ?? 40) * 0.75} critAt={b.msg_limit ?? 40}
          right={`${(b.msg_rate ?? 0).toFixed(1)} of ${b.msg_limit ?? 40} msg/s`} color="var(--operate)" />
        <Meter label="Automatic rollbacks, last hour" value={b.rollbacks ?? 0} max={3} warnAt={2} critAt={3} right={`${b.rollbacks ?? 0} of 3`} color="var(--operate)" />
      </div>
      <p className="mt-3 text-micro text-ink-3">Routine handover-offset tweaks are pre-approved micro-changes and do not count toward the change budget. Tripping any breaker freezes the swarm and writes a ledger record.</p>
    </Panel>
  )
}

// ------------------------------------------------------------------ autonomy

const LEVELS = [
  { value: 0, label: 'L0', title: 'Manual: humans do everything' },
  { value: 1, label: 'L1', title: 'Assisted: agents suggest' },
  { value: 2, label: 'L2', title: 'Partial: humans approve every action' },
  { value: 3, label: 'L3', title: 'Conditional: humans approve high-risk actions' },
  { value: 4, label: 'L4', title: 'High: humans are notified' },
  { value: 5, label: 'L5', title: 'Full autonomy' },
]
const CLASSES: { id: 'coordinator' | 'domain' | 'site'; label: string; color: string }[] = [
  { id: 'coordinator', label: 'Coordinator', color: 'var(--s1)' },
  { id: 'domain', label: 'Domain agents', color: 'var(--s2)' },
  { id: 'site', label: 'Site agents', color: 'var(--s3)' },
]

export function AutonomyPanel() {
  const on = useFlag('swarm.autonomy')
  const autonomy = useLive((s) => s.swarm.autonomy ?? NONE)
  const series = useLive((s) => s.swarm.trust_series ?? EMPTY) as Rec[]
  const classTrust = useLive((s) => s.swarm.class_trust ?? NONE)
  const [reason, setReason] = useState('')
  if (!on) return null
  const lines: Series[] = CLASSES.map((c) => ({ id: c.id, label: c.label, color: c.color, points: series.map((p) => ({ x: p.t, y: p[c.id] })) }))
  const ann = series.filter((p) => p.event).map((p) => ({ x: p.t, label: p.event, kind: 'config' }))
  const setLevel = (cls: string, level: number) => {
    const why = reason.trim() || 'changed by operator'
    api.autonomy(cls, level, why).then(() => toast({ severity: 'notice', title: `${cls} autonomy set to L${level} (ledgered)` }))
  }
  return (
    <Panel id="autonomy" plane="operate" title="Autonomy" subtitle="TM Forum Autonomous Networks levels, earned from track record" source={SRC_SWARM}>
      <div className="space-y-2.5">
        {CLASSES.map((c) => (
          <div key={c.id} className="flex items-center gap-3">
            <span className="flex w-32 items-center gap-1.5 text-cap text-ink-1"><span className="size-2 rounded-full" style={{ background: c.color }} />{c.label}</span>
            <Segmented size="sm" value={autonomy[c.id] ?? 3} onChange={(v) => setLevel(c.id, v)} ariaLabel={`${c.label} autonomy`} options={LEVELS} />
            <span className="tnum ml-auto text-cap text-ink-2" title="Average trust score">{(classTrust[c.id] ?? 0).toFixed(0)}</span>
          </div>
        ))}
        <input value={reason} onChange={(e) => setReason(e.target.value)} placeholder="Reason for the change (recorded in the ledger)"
          className="h-7 w-full rounded-md border border-line bg-s2 px-2 text-cap text-ink-1 placeholder:text-ink-3" />
      </div>
      <div className="mt-4 mb-1 text-micro text-ink-3">Earned trust by agent class. Bands: L3 from 70, L4 from 85, L5 from 95.</div>
      <LineChart series={lines} height={150} yDomain={[40, 100]} yTicks={3}
        bands={[{ y0: 70, y1: 85 }, { y0: 85, y1: 95 }]} annotations={ann}
        xFormat={(x) => simClockLabel(x).slice(0, 5)} yFormat={(y) => y.toFixed(0)} tipTitle={(x) => simClockLabel(x)} />
      <p className="mt-2 text-micro text-ink-3">Trust rises with verified successful actions and drops on denials. A quarantine automatically lowers its class one level until a human restores it.</p>
    </Panel>
  )
}
