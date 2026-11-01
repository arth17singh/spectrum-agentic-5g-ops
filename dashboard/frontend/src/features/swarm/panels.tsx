import clsx from 'clsx'
import { ExternalLink, Radio } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { DeltaBars } from '@/charts/DeltaBars'
import { HexLegend, HexMap, type HexSite, type Layer, type SiteState } from '@/charts/HexMap'
import { Scatter, type Dot } from '@/charts/Scatter'
import { Sparkline } from '@/charts/Sparkline'
import { SwarmGraph, SwarmLegend } from '@/charts/SwarmGraph'
import { api } from '@/core/api'
import { clockTime, duration, num, simOf, usd } from '@/core/format'
import { EMPTY, useLive } from '@/core/store'
import type { Agent, Rec } from '@/core/types'
import { useSimNow, useUi } from '@/core/ui-store'
import { Button, EmptyState, LedgerLink, Meter, Panel, Segmented, StatusPill } from '@/ui'

const NONE: Rec = {}
const SRC_TWIN = { aws: 'ECS twin + Kinesis', mock: 'Simulated twin' }
const SRC_SWARM = { aws: 'Bedrock AgentCore', mock: 'Simulated swarm' }

// ------------------------------------------------------------------ network KPIs strip

export function KpiStrip() {
  const kpis = useLive((s) => s.twin.kpis ?? NONE)
  const series = useLive((s) => s.twin.health_series ?? EMPTY)
  const tail = series.slice(-90)
  const items: { k: string; label: string; v: string; good?: boolean | null }[] = [
    { k: 'health_pct', label: 'Network health', v: `${(kpis.health_pct ?? 0).toFixed(1)}%`, good: kpis.health_pct >= 95 ? true : kpis.health_pct < 85 ? false : null },
    { k: 'congestion_pct', label: 'Stadium congestion', v: `${(kpis.congestion_pct ?? 0).toFixed(0)}% PRB`, good: kpis.congestion_pct > 85 ? false : null },
    { k: 'dropped_pct', label: 'Dropped calls, stadium', v: `${(kpis.dropped_pct ?? 0).toFixed(2)}%`, good: kpis.dropped_pct > 1 ? false : null },
    { k: 'throughput_gbps', label: 'Throughput', v: `${(kpis.throughput_gbps ?? 0).toFixed(1)} Gbps` },
    { k: 'users', label: 'Connected users', v: num(kpis.users) },
    { k: 'energy_kw', label: 'Radio power', v: `${(kpis.energy_kw ?? 0).toFixed(0)} kW` },
    { k: 'sleeping_cells', label: 'Cells asleep', v: String(kpis.sleeping_cells ?? 0) },
  ]
  return (
    <div className="grid grid-cols-7 gap-px overflow-hidden rounded-[10px] border border-line bg-line">
      {items.map((it) => (
        <div key={it.k} className="bg-s1 px-4 py-3">
          <div className="truncate text-micro text-ink-3">{it.label}</div>
          <div className="tnum text-[18px] leading-6 font-semibold" style={{ color: it.good === false ? 'var(--critical-ink)' : 'var(--ink-1)' }}>{it.v}</div>
          <Sparkline values={tail.map((p: Rec) => p[it.k] as number)} color={it.good === false ? 'var(--critical)' : 'var(--operate)'} height={20} className="mt-1" />
        </div>
      ))}
    </div>
  )
}

// ------------------------------------------------------------------ twin map

export function TwinPanel() {
  const twin = useLive((s) => s.twin)
  const agents = useLive((s) => s.swarm.agents ?? EMPTY)
  const actions = useLive((s) => s.twin.actions ?? EMPTY)
  const inspect = useUi((s) => s.inspect)
  const [layer, setLayer] = useState<Layer>('load')
  const [flashing, setFlashing] = useState<Set<string>>(new Set())
  const lastTs = useRef<string>((actions[actions.length - 1] as Rec | undefined)?.ts ?? '')
  const sites: HexSite[] = twin.sites ?? []

  // flash sites touched by newly executed actions
  useEffect(() => {
    const fresh = (actions as Rec[]).filter((a) => a.ts > lastTs.current)
    if (!fresh.length) return
    lastTs.current = fresh[fresh.length - 1].ts
    const ids = new Set<string>()
    for (const a of fresh as Rec[]) {
      if (a.actor === 'preflight') continue
      const p = a.params ?? {}
      for (const c of p.cells ?? []) ids.add(String(c).split('-').slice(0, 2).join('-'))
      for (const sid of [...(p.from_sites ?? []), ...(p.to_sites ?? [])]) ids.add(sid)
      if (p.hub) for (const s of sites) if (s.hub === p.hub) ids.add(s.id)
    }
    if (!ids.size) return
    setFlashing(ids)
    const t = setTimeout(() => setFlashing(new Set()), 1500)
    return () => clearTimeout(t)
  }, [actions, sites])

  const agentState = useMemo(() => Object.fromEntries((agents as Agent[]).map((a) => [a.id, { state: a.state, status: a.status }])), [agents])
  const alarms: Rec[] = (twin.alarms ?? []).slice(-1)
  return (
    <Panel id="twin" plane="operate" title="Network twin" source={SRC_TWIN}
      actions={<Segmented size="sm" value={layer} onChange={setLayer} ariaLabel="Map layer" options={[
        { value: 'load', label: 'Load' }, { value: 'throughput', label: 'Throughput' }, { value: 'energy', label: 'Energy' }, { value: 'agents', label: 'Agents' }]} />}>
      <HexMap sites={sites} state={(twin.state ?? []) as SiteState[]} cols={twin.layout?.cols ?? 8} rows={twin.layout?.rows ?? 5}
        layer={layer} agentState={agentState} flashing={flashing}
        onSite={(s, st) => inspect({ kind: 'site', id: s.id, title: `Site ${s.id}`, data: {
          ...s, kpis: st ? { prb_pct: st[1], users: st[2], throughput_mbps: st[3], drop_pct: st[4], power_kw: st[5], backhaul: st[6] } : null,
          agent: (agents as Agent[]).find((a) => a.site === s.id) ?? null,
          recent_actions: (actions as Rec[]).filter((a) => JSON.stringify(a.params ?? {}).includes(s.id)).slice(-5),
        } })} />
      <HexLegend layer={layer} />
      {alarms.map((a) => (
        <div key={a.id} className="mt-2 flex items-center gap-2 rounded-md border border-line bg-s2 px-2.5 py-1.5 text-cap">
          <StatusPill status={a.severity === 'critical' ? 'critical' : 'warning'} size="sm">{a.kind}</StatusPill>
          <span className="mono text-ink-3">{clockTime(a.ts, false)}</span>
          <span className="truncate text-ink-1">{a.text}</span>
          <span className="mono ml-auto text-micro text-ink-3">{a.id}</span>
        </div>
      ))}
    </Panel>
  )
}

// ------------------------------------------------------------------ swarm graph

export function SwarmPanel() {
  const agents = useLive((s) => s.swarm.agents ?? EMPTY) as Agent[]
  const breakers = useLive((s) => s.swarm.breakers ?? NONE)
  const frozen = useLive((s) => s.swarm.frozen)
  const busy = agents.filter((a) => a.state === 'thinking' || a.state === 'acting').length
  const q = agents.filter((a) => a.status === 'quarantined').length
  return (
    <Panel id="swarm" plane="operate" title="Swarm" subtitle={`${agents.length} agents, ${busy} active${q ? `, ${q} quarantined` : ''}`} source={SRC_SWARM}
      actions={frozen ? <StatusPill status="critical" size="sm">Frozen</StatusPill> : <span className="tnum flex items-center gap-1 text-micro text-ink-3"><Radio size={12} />{breakers.msg_rate ?? 0} msg/s</span>}>
      <SwarmGraph height={430} />
      <SwarmLegend />
    </Panel>
  )
}

// ------------------------------------------------------------------ incident card

const PHASE_STATE: Record<string, 'done' | 'active' | 'pending' | 'failed'> = { done: 'done', active: 'active', pending: 'pending' }

export function IncidentCard() {
  const inc = useLive((s) => s.swarm.incident)
  const incidents = useLive((s) => s.swarm.incidents ?? EMPTY) as Rec[]
  const now = useSimNow(4)
  const navigate = useNavigate()
  const inspect = useUi((s) => s.inspect)
  const live = inc && inc.status !== 'resolved'
  const elapsed = inc ? (inc.status === 'resolved' ? inc.mttr_s : now - simOf(inc.started_at)) : 0
  const history = incidents.filter((i) => !inc || i.id !== inc.id).slice(-4).reverse()
  return (
    <Panel id="incident" plane="operate" title="Incident" source={SRC_SWARM}
      actions={inc && <StatusPill status={live ? 'critical' : 'passed'} size="sm">{live ? inc.severity : 'resolved'}</StatusPill>}>
      {inc ? (
        <div>
          <div className="mono text-micro text-ink-3">{inc.id}</div>
          <div className="text-body font-medium text-ink-1">{inc.title}</div>
          <div className="mt-3 flex items-end gap-6">
            <div>
              <div className="text-micro text-ink-3">{live ? 'Elapsed' : 'Time to resolve'}</div>
              <div className={clsx('tnum text-[30px] leading-9 font-semibold', live ? 'text-ink-1' : 'text-good-ink')}>{duration(elapsed)}</div>
            </div>
            <div className="pb-1 text-cap">
              <div className="text-ink-3">Affected</div>
              <div className="tnum text-ink-1">{inc.affected_sites?.length ?? 0} sites, {num(inc.affected_users)} users</div>
            </div>
          </div>
          <ol className="mt-4 grid grid-cols-2 gap-x-3 gap-y-1.5">
            {(inc.phases ?? []).map((p: Rec, i: number) => {
              const st = PHASE_STATE[p.state] ?? 'pending'
              return (
                <li key={p.label} className="flex items-center gap-2 text-cap">
                  <span className={clsx('grid size-[18px] shrink-0 place-items-center rounded-full text-micro font-semibold tnum',
                    st === 'done' ? 'bg-good/20 text-good-ink' : st === 'active' ? 'bg-release text-white' : 'bg-s3 text-ink-3')}>{i + 1}</span>
                  <span className={st === 'pending' ? 'text-ink-3' : 'text-ink-1'}>{p.label}</span>
                  {p.ts && st !== 'pending' && <span className="mono ml-auto text-micro text-ink-3">{clockTime(p.ts, false).slice(3)}</span>}
                </li>
              )
            })}
          </ol>
          <dl className="mt-4 grid grid-cols-2 gap-x-3 gap-y-1.5 text-cap">
            <dt className="text-ink-3">Owner</dt><dd className="mono text-ink-1">{inc.owner}</dd>
            <dt className="text-ink-3">Model spend</dt><dd className="tnum text-ink-1">{usd(inc.cost_usd)} <span className="text-ink-3">{num(inc.tokens)} tok</span></dd>
            <dt className="text-ink-3">Ledger from</dt><dd><LedgerLink seq={inc.ledger_from} /></dd>
            <dt className="text-ink-3">Started</dt><dd className="mono text-ink-1">{clockTime(inc.started_at, false)}</dd>
          </dl>
          <div className="mt-3 flex flex-wrap gap-2">
            <Button size="sm" variant="ghost" icon={<ExternalLink size={12} />} onClick={() => navigate('/observability')}>Trace</Button>
            <Button size="sm" variant="ghost" icon={<ExternalLink size={12} />} onClick={() => navigate('/audit')}>Dossier</Button>
            {inc.approval_id && <Button size="sm" variant="ghost" icon={<ExternalLink size={12} />} onClick={() => navigate('/oversight')}>{inc.approval_id}</Button>}
          </div>
        </div>
      ) : (
        <EmptyState title="The network is nominal" body="Inject the stadium let-out scenario to watch the swarm detect, negotiate, pre-flight and ask a human."
          action={<Button size="sm" variant="primary" onClick={() => api.inject('incident')}>Inject fibre cut</Button>} />
      )}
      {history.length > 0 && (
        <div className="mt-4 border-t border-line pt-3">
          <div className="mb-1.5 text-micro text-ink-3">Earlier tonight</div>
          <ul className="space-y-1">
            {history.map((h) => (
              <li key={h.id}>
                <button type="button" onClick={() => inspect({ kind: 'raw', id: h.id, title: h.title, data: h })} className="flex w-full items-center gap-2 text-left text-cap hover:text-ink-1">
                  <span className="mono w-16 shrink-0 text-ink-3">{h.id}</span>
                  <span className="truncate text-ink-2">{h.title}</span>
                  <span className="tnum ml-auto shrink-0 text-ink-3">{h.mttr_s ? duration(h.mttr_s) : '—'}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </Panel>
  )
}

// ------------------------------------------------------------------ negotiation

const KIND_STATUS: Record<string, 'running' | 'warning' | 'passed' | 'neutral'> = { propose: 'running', object: 'warning', accept: 'passed' }

export function NegotiationPanel() {
  const inc = useLive((s) => s.swarm.incident)
  const neg = inc?.negotiation
  const agents = useLive((s) => s.swarm.agents ?? EMPTY) as Agent[]
  const label = (id: string) => agents.find((a) => a.id === id)?.label ?? id
  if (!neg) {
    return (
      <Panel id="negotiation" plane="operate" title="Negotiation" subtitle="conflicting objectives, brokered" source={SRC_SWARM}>
        <EmptyState title="No open negotiation" body="When agents disagree about a plan (energy against experience, for example), their offers, the trade-off and any dissent appear here." />
      </Panel>
    )
  }
  const dots: Dot[] = (neg.options ?? []).map((o: Rec) => ({
    id: o.id, x: o.kwh, y: o.users_served / 1000, color: o.chosen ? 'var(--operate)' : 'var(--neutral-series)', ring: o.chosen,
    label: o.id, showLabel: true, title: `Option ${o.id}: ${o.label}`,
    tip: [{ label: 'extra energy', value: `${o.kwh} kWh` }, { label: 'users served', value: num(o.users_served) }],
  }))
  return (
    <Panel id="negotiation" plane="operate" title="Negotiation" subtitle={`${neg.participants.map(label).join(', ')}`} source={SRC_SWARM}
      actions={neg.resolution ? <StatusPill status="passed" size="sm">Agreed on {neg.resolution}</StatusPill> : <StatusPill status="pending" size="sm">In progress</StatusPill>}>
      <div className="grid grid-cols-[1fr_1.1fr] gap-4">
        <ol className="space-y-2">
          {(neg.rounds ?? []).map((r: Rec, i: number) => (
            <li key={i} className="row-in rounded-md border border-line bg-s2 px-3 py-2">
              <div className="flex items-center gap-2 text-cap">
                <span className="tnum text-ink-3">{i + 1}</span>
                <span className="font-medium text-ink-1">{label(r.agent)}</span>
                <StatusPill status={KIND_STATUS[r.kind] ?? 'neutral'} size="sm">{r.kind}</StatusPill>
                <span className="ml-auto rounded bg-s3 px-1.5 text-micro text-ink-2">option {r.option}</span>
              </div>
              <p className="mt-1 text-cap text-ink-2">{r.text}</p>
            </li>
          ))}
        </ol>
        <div>
          <div className="mb-1 text-micro text-ink-3">Trade-off: extra energy tonight against users served</div>
          <Scatter dots={dots} height={190} xLabel="extra kWh" yLabel="users served (K)" xFormat={(v) => `${v}`} yFormat={(v) => `${v.toFixed(0)}K`} />
          <ul className="mt-2 space-y-0.5 text-cap">
            {(neg.options ?? []).map((o: Rec) => (
              <li key={o.id} className={clsx('flex items-center gap-2', o.chosen ? 'text-ink-1' : 'text-ink-2')}>
                <span className={clsx('grid size-4 place-items-center rounded-full text-micro font-semibold', o.chosen ? 'bg-operate text-white' : 'bg-s3 text-ink-2')}>{o.id}</span>
                <span className="truncate">{o.label}</span>
                <span className="tnum ml-auto shrink-0 text-ink-3">+{o.kwh} kWh, {num(o.users_served)}</span>
              </li>
            ))}
          </ul>
          {(neg.dissent ?? []).map((d: Rec, i: number) => (
            <div key={i} className="mt-2 rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-cap">
              <span className="font-medium text-ink-1">Dissent recorded, {label(d.agent)}: </span>
              <span className="text-ink-2">{d.text}</span>
            </div>
          ))}
        </div>
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ plan + pre-flight

const VERDICT: Record<string, { status: 'passed' | 'pending' | 'denied' | 'approved' | 'neutral'; word: string }> = {
  auto: { status: 'passed', word: 'auto' }, needs_human: { status: 'pending', word: 'needs human' }, denied: { status: 'denied', word: 'denied' },
  approved: { status: 'approved', word: 'approved' },
}

export function PlanPanel() {
  const inc = useLive((s) => s.swarm.incident)
  const items = useLive((s) => s.oversight.items ?? EMPTY) as Rec[]
  const navigate = useNavigate()
  const plan = inc?.plan
  const pf = inc?.preflight
  const appr = inc?.approval_id ? items.find((i) => i.id === inc.approval_id) : null
  if (!plan) {
    return (
      <Panel id="plan" plane="operate" title="Plan and pre-flight" subtitle="every action is simulated on a fork of the twin first" source={SRC_TWIN}>
        <EmptyState title="No plan yet" body="The coordinator’s plan appears here with each step’s policy verdict and the twin’s predicted impact before anything touches the network." />
      </Panel>
    )
  }
  return (
    <Panel id="plan" plane="operate" title="Plan and pre-flight" subtitle={plan.id} source={SRC_TWIN}
      actions={appr ? <StatusPill status={appr.status === 'approved' ? 'approved' : appr.status === 'rejected' ? 'rejected' : 'pending'} size="sm">{appr.id} {appr.status}</StatusPill> : null}>
      <div className="grid grid-cols-[1.15fr_1fr] gap-5">
        <div>
          <ol className="space-y-1.5">
            {plan.steps.map((st: Rec) => {
              const v = VERDICT[st.verdict] ?? VERDICT.auto
              return (
                <li key={st.n} className={clsx('flex items-center gap-2.5 rounded-md border px-3 py-2 text-cap', st.state === 'done' ? 'border-line bg-s2' : 'border-line')}>
                  <span className="tnum w-4 text-ink-3">{st.n}</span>
                  <span className="mono font-medium text-ink-1">{st.tool}</span>
                  <span className="truncate text-ink-2">{st.target}</span>
                  <span className="mono ml-auto shrink-0 text-micro text-ink-3">{st.policy}</span>
                  <StatusPill status={st.state === 'done' && st.verdict !== 'auto' ? 'approved' : v.status} size="sm">{st.state === 'done' ? (st.verdict === 'auto' ? 'executed' : 'executed') : v.word}</StatusPill>
                </li>
              )
            })}
          </ol>
          {pf && (
            <div className="mt-4">
              <Meter value={pf.blast_radius_pct} max={25} warnAt={5} label={<>Blast radius <span className="text-ink-3">(policy P-003: above 5% needs a human)</span></>}
                right={`${pf.blast_radius_pct}% of cells, ${pf.cells_touched} cells`} color="var(--operate)" height={8} />
            </div>
          )}
          <div className="mt-4 flex items-center gap-2">
            {appr?.status === 'pending' ? (
              <Button size="sm" variant="primary" onClick={() => navigate('/oversight')}>Review {appr.id} in the Oversight Desk</Button>
            ) : appr ? (
              <span className="text-cap text-ink-2">Decision by {appr.approvals?.map((a: Rec) => a.name).join(' and ') || 'a human'}{appr.ttd_s ? ` after ${duration(appr.ttd_s)}` : ''}.</span>
            ) : (
              <span className="text-cap text-ink-3">Policy check runs after the pre-flight.</span>
            )}
            {pf?.ledger_seq && <span className="ml-auto text-micro text-ink-3">pre-flight <LedgerLink seq={pf.ledger_seq} /></span>}
          </div>
        </div>
        <div>
          <div className="mb-2 text-micro text-ink-3">Predicted impact from the twin pre-flight</div>
          {pf ? (
            <DeltaBars rows={[
              { id: 'c', label: 'Stadium congestion', before: pf.before.congestion_pct, after: pf.after.congestion_pct, max: 100, unit: '%', better: 'lower', format: (v) => v.toFixed(0) },
              { id: 'd', label: 'Dropped calls, stadium', before: pf.before.dropped_pct, after: pf.after.dropped_pct, unit: '%', better: 'lower', format: (v) => v.toFixed(2) },
              { id: 'h', label: 'Network health', before: pf.before.health_pct, after: pf.after.health_pct, max: 100, unit: '%', better: 'higher', format: (v) => v.toFixed(0) },
              { id: 't', label: 'Throughput', before: pf.before.throughput_gbps, after: pf.after.throughput_gbps, unit: ' Gbps', better: 'higher', format: (v) => v.toFixed(1) },
              { id: 'e', label: 'Radio power', before: pf.before.energy_kw, after: pf.after.energy_kw, unit: ' kW', better: 'lower', format: (v) => v.toFixed(0) },
            ]} />
          ) : <div className="text-cap text-ink-3">Running…</div>}
        </div>
      </div>
    </Panel>
  )
}
