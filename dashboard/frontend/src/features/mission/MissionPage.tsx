import clsx from 'clsx'
import { ArrowUpRight, ShieldCheck } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { ConsortiumMap, type Packet } from '@/charts/ConsortiumMap'
import { LineChart } from '@/charts/LineChart'
import { Sparkline } from '@/charts/Sparkline'
import { SeverityIcon, PlaneChip } from '@/app/StreamDrawer'
import { bytes, clockTime, duration, pct, simOf, usd } from '@/core/format'
import { opColor, PLANES } from '@/core/palette'
import { EMPTY, useLive } from '@/core/store'
import type { Envelope, Operator, Rec } from '@/core/types'
import { useSimNow, useUi } from '@/core/ui-store'
import { PageHeader, Panel, StatusPill } from '@/ui'
import manifest from './manifest'
import { MiniTwin } from './MiniTwin'

export default function MissionPage() {
  return (
    <div>
      <PageHeader title={manifest.title} description={manifest.description} tracks={manifest.tracks} />
      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-12 xl:col-span-5"><HeroHealth /></div>
        <div className="col-span-12 xl:col-span-7"><Consortium /></div>
        <div className="col-span-12"><PlaneTiles /></div>
        <div className="col-span-12 xl:col-span-7">
          <Panel id="mini-twin" plane="operate" title="Network twin" subtitle="40 sites, 120 cells, stadium district" source={{ aws: 'ECS twin + Kinesis', mock: 'Simulated network' }}>
            <MiniTwin height={260} />
          </Panel>
        </div>
        <div className="col-span-12 xl:col-span-5"><CriticalFeed /></div>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ hero

function HeroHealth() {
  const twin = useLive((s) => s.twin)
  const sw = useLive((s) => s.swarm)
  const items: Rec[] = useLive((s) => s.oversight.items ?? EMPTY)
  const now = useSimNow(2)
  const series: Rec[] = (twin.health_series ?? EMPTY).slice(-300)
  const k = twin.kpis ?? {}
  const inc = sw.incident
  const active = inc && inc.status !== 'resolved'
  const incidents: Rec[] = sw.incidents ?? EMPTY
  const lastResolved = [...incidents].reverse().find((i) => i.status === 'resolved' && i.mttr_s)
  const pending = items.filter((i) => i.status === 'pending')
  const health = k.health_pct ?? 0
  const status = health >= 95 ? 'good' : health >= 80 ? 'warning' : 'critical'
  return (
    <Panel id="health" plane="operate" title="Network health" subtitle="share of sites inside their KPI SLO" source={{ aws: 'CloudWatch metrics', mock: 'Simulated KPIs' }}
      actions={<StatusPill status={status} size="sm">{status === 'good' ? 'Nominal' : status === 'warning' ? 'Degraded' : 'Impaired'}</StatusPill>}>
      <div className="flex items-end gap-5">
        <div>
          <div className="text-hero font-semibold tracking-[-0.02em] text-ink-1">{health.toFixed(1)}<span className="text-[24px] text-ink-3">%</span></div>
          <div className="text-cap text-ink-3">{k.users?.toLocaleString() ?? '—'} users on {(k.throughput_gbps ?? 0).toFixed(1)} Gbps</div>
        </div>
        <div className="mb-1 grid flex-1 grid-cols-3 gap-3 border-l border-line pl-5">
          <SubStat label="Incidents" value={active ? '1 active' : 'none'} tone={active ? 'critical' : undefined}
            sub={active ? `${inc.id}, ${duration(now - simOf(inc.started_at))}` : `${incidents.length} resolved today`} />
          <SubStat label="Last MTTR" value={lastResolved ? duration(lastResolved.mttr_s) : '—'} sub={lastResolved ? lastResolved.id : 'no incidents'} />
          <SubStat label="Waiting on a human" value={String(pending.length)} tone={pending.length ? 'warn' : undefined}
            sub={pending.length ? pending[0].id : 'queue clear'} />
        </div>
      </div>
      <div className="mt-4">
        <LineChart height={120} legend={false} yDomain={[0, 100]} yTicks={2}
          series={[
            { id: 'h', label: 'Health', color: 'var(--operate)', area: true, points: series.map((p) => ({ x: p.t, y: p.health_pct })) },
            { id: 'c', label: 'Stadium congestion', color: 'var(--serious)', width: 1.5, points: series.map((p) => ({ x: p.t, y: p.congestion_pct })) },
          ]}
          xTicks={5} xFormat={(x) => (now - x < 3 ? 'now' : `−${Math.round((now - x) / 60 * 10) / 10}m`)} yFormat={(y) => `${y.toFixed(0)}%`} tipTitle={(x) => `${Math.round(now - x)} s ago`} />
        <div className="mt-1 flex gap-4 text-micro text-ink-3">
          <span className="flex items-center gap-1.5"><span className="h-0.5 w-3 rounded" style={{ background: 'var(--operate)' }} />Network health</span>
          <span className="flex items-center gap-1.5"><span className="h-0.5 w-3 rounded" style={{ background: 'var(--serious)' }} />Stadium district PRB load</span>
        </div>
      </div>
    </Panel>
  )
}

function SubStat({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: 'warn' | 'critical' }) {
  return (
    <div className="min-w-0">
      <div className="truncate text-micro text-ink-3">{label}</div>
      <div className={clsx('truncate text-section font-semibold', tone === 'critical' ? 'text-critical-ink' : tone === 'warn' ? 'text-warn' : 'text-ink-1')}>{value}</div>
      {sub && <div className="truncate text-micro text-ink-3">{sub}</div>}
    </div>
  )
}

// ------------------------------------------------------------------ consortium

function Consortium() {
  const f = useLive((s) => s.federation)
  const stream = useLive((s) => s.stream)
  const motion = useUi((s) => s.motion)
  const [packets, setPackets] = useState<Packet[]>([])
  const lastSeq = useRef<number | null>(null)
  const pid = useRef(0)
  const ops: Operator[] = f.operators ?? EMPTY

  useEffect(() => {
    const maxSeq = stream.length ? stream[stream.length - 1].seq : 0
    if (lastSeq.current == null) { lastSeq.current = maxSeq; return }
    const fresh: Packet[] = []
    const t = performance.now()
    for (let i = stream.length - 1; i >= 0 && stream[i].seq > lastSeq.current; i--) {
      const e = stream[i]
      if (e.type === 'fl.upload') fresh.push({ id: ++pid.current, from: e.data.op, to: 'enclave', color: opColor(e.data.op), born: t })
      if (e.type === 'fl.round.phase' && e.data.phase === 'pull') {
        for (const op of e.data.current?.participants ?? []) fresh.push({ id: ++pid.current, from: 'enclave', to: op, color: 'var(--learn)', born: t })
      }
    }
    lastSeq.current = maxSeq
    if (fresh.length) setPackets((p) => [...p.filter((x) => t - x.born < 1500), ...fresh])
  }, [stream])

  const spr = f.steps_per_round ?? 240
  const nodes = ops.map((o) => ({
    id: o.id, short: o.short, name: o.name, color: opColor(o.id), status: o.status, eps: o.eps, samples: o.samples,
    progress: ['training', 'securing', 'uploading'].includes(o.status) ? Math.min(1, o.step / spr) : 0,
    bytesUp: f.sovereignty?.per_op?.[o.id]?.adapter ?? o.bytes_up ?? 0,
  }))
  const phaseText: Record<string, string> = { pull: 'sending adapter', train: 'local training', aggregate: 'aggregating', done: 'between rounds', idle: 'idle' }
  const sub = f.state === 'running' ? `round ${f.round}, ${phaseText[f.phase] ?? f.phase}` : f.state === 'ready' ? `round ${(f.round ?? 0) + 1} queued` : `round ${f.round}, ${f.state}`
  const sv = f.sovereignty ?? {}
  return (
    <Panel id="consortium" plane="learn" title="Consortium" subtitle="five operators, one model, no shared data" source={{ aws: 'Nitro Enclave + VPC Flow Logs', mock: 'Simulated federation' }}>
      <ConsortiumMap nodes={nodes} packets={packets} centreLabel="Enclave aggregator" centreSub={sub} attested={f.config?.secagg_mode === 'enclave'}
        format={bytes} height={300} reducedMotion={motion === 'reduced'} />
      <div className="mt-2 grid grid-cols-3 gap-3 border-t border-line pt-3">
        <div className="flex items-center gap-2">
          <ShieldCheck size={18} className="shrink-0 text-good-ink" />
          <div className="min-w-0">
            <div className="text-micro text-ink-3">Raw data moved across boundaries</div>
            <div className="tnum text-section font-semibold text-good-ink">0 B</div>
          </div>
        </div>
        <div>
          <div className="text-micro text-ink-3">Masked adapter updates moved</div>
          <div className="tnum text-section font-semibold text-ink-1">{bytes(sv.adapter_bytes ?? 0)}</div>
        </div>
        <div>
          <div className="text-micro text-ink-3">Federated loss vs best alone</div>
          <FedGain />
        </div>
      </div>
    </Panel>
  )
}

function FedGain() {
  const rounds: Rec[] = useLive((s) => s.federation.rounds_log ?? EMPTY)
  const last = rounds[rounds.length - 1]
  if (!last) return <div className="text-section font-semibold text-ink-3">—</div>
  const best = Math.min(...(Object.values(last.local_only) as number[]))
  return (
    <div className="tnum text-section font-semibold text-ink-1">
      {last.val_loss.toFixed(3)} <span className="text-cap font-medium text-good-ink">−{(((best - last.val_loss) / best) * 100).toFixed(0)}%</span>
    </div>
  )
}

// ------------------------------------------------------------------ plane tiles

function PlaneTiles() {
  const navigate = useNavigate()
  const fl = useLive((s) => s.federation)
  const pv = useLive((s) => s.proving)
  const rl = useLive((s) => s.release)
  const sw = useLive((s) => s.swarm)
  const lg = useLive((s) => s.ledger)
  const tr = useLive((s) => s.trust)
  const tw = useLive((s) => s.twin)
  const frames: Rec[] = useLive((s) => s.observability.frames ?? EMPTY)
  const now = useSimNow(1)

  const tiles = useMemo(() => {
    const ops: Operator[] = fl.operators ?? []
    const maxEps = Math.max(0, ...ops.map((o) => o.eps))
    const rounds: Rec[] = fl.rounds_log ?? []
    const models: Rec[] = pv.models ?? []
    const ours = models.filter((m) => m.family === 'ours' && /^nlm-v\d+$/.test(m.id)).sort((a, b) => Number(a.id.slice(5)) - Number(b.id.slice(5)))
    const prod = models.find((m) => m.id === (pv.production ?? 'nlm-v6'))
    const cand = pv.candidate ? models.find((m) => m.id === pv.candidate) : null
    const active = pv.active
    const agents: Rec[] = sw.agents ?? []
    const quarantined = agents.filter((a) => a.status === 'quarantined').length
    const inc = sw.incident
    const incActive = inc && inc.status !== 'resolved'
    const lastResolved = [...(sw.incidents ?? [])].reverse().find((i: Rec) => i.status === 'resolved' && i.mttr_s)
    const lv = lg.last_verify
    const lastAnchor = (lg.anchors ?? [])[(lg.anchors ?? []).length - 1]
    const series: Rec[] = tr.series ?? []
    return [
      {
        plane: 'learn', route: '/federation',
        rows: [
          ['Round', `${fl.round ?? 0} of ${fl.config?.rounds ?? 20}`, fl.state === 'running' ? fl.phase : fl.state],
          ['Highest ε', maxEps.toFixed(2), `target ${fl.config?.target_eps ?? 3}, δ 1e-5`],
          ['Validation loss', (fl.global?.val_loss ?? 0).toFixed(3), `accuracy ${pct(fl.global?.accuracy, true)}`],
        ],
        trend: rounds.map((r) => r.val_loss), trendLabel: 'loss per round',
      },
      {
        plane: 'verify', route: '/proving',
        rows: [
          ['In production', pv.production ?? 'nlm-v6', prod ? `accuracy ${pct(prod.accuracy, true)}` : ''],
          ['Candidate', cand ? cand.id : 'none', active ? (active.result ? (active.result.passed ? 'gate passed' : 'gate blocked') : `${active.state}`) : 'awaiting checkpoint'],
          ['Safety', ((cand ?? prod)?.safety ?? 0).toFixed(3), `${usd((cand ?? prod)?.usd_per_1k)} per 1K tasks`],
        ],
        trend: ours.map((m) => m.accuracy), trendLabel: 'accuracy by version',
      },
      {
        plane: 'release', route: '/release',
        rows: [
          ['Serving', rl.prod ?? 'nlm-v6', '4 targets, 2 vendors'],
          ['Rollout', rl.candidate ? `${rl.candidate}` : 'none', rl.candidate ? rl.stages?.[rl.stage_index]?.label : 'stable'],
          ['Guard p95', `${rl.guards?.p95_ms ?? '—'} ms`, `errors ${(rl.guards?.error_rate ?? 0).toFixed(2)}%`],
        ],
        trend: frames.slice(-60).map((fr) => fr.golden?.release?.p95 ?? 0), trendLabel: 'p95 latency, last minute',
      },
      {
        plane: 'operate', route: '/swarm',
        rows: [
          ['Agents', `${agents.length - quarantined} active`, quarantined ? `${quarantined} quarantined` : 'none quarantined'],
          ['Incident', incActive ? inc.id : 'none', incActive ? `${duration(now - simOf(inc.started_at))} elapsed` : lastResolved ? `last MTTR ${duration(lastResolved.mttr_s)}` : 'nominal'],
          ['Autonomy', `site L${sw.autonomy?.site ?? '–'}`, `coordinator L${sw.autonomy?.coordinator ?? '–'}`],
        ],
        trend: (tw.health_series ?? []).slice(-120).map((p: Rec) => p.health_pct), trendLabel: 'network health',
        alert: !!incActive,
      },
      {
        plane: 'prove', route: '/audit',
        rows: [
          ['Ledger', `#${(lg.height ?? 0).toLocaleString()}`, 'every record signed'],
          ['Chain', lv ? (lv.ok ? 'verified' : `broken at #${lv.broken?.seq}`) : 'not yet verified', lv ? `${lv.total?.toLocaleString()} records checked` : 'run verify in Audit'],
          ['Last anchor', lastAnchor ? `#${lastAnchor.n}` : '—', lastAnchor ? 'S3 Object Lock, compliance' : ''],
        ],
        trend: series.slice(1).slice(-30).map((b) => b.permit + b.deny + b.needs_human), trendLabel: 'policy decisions per 10 s',
        alert: lv && !lv.ok,
      },
    ]
  }, [fl, pv, rl, sw, lg, tr, tw, frames, now])

  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-5">
      {tiles.map((t) => {
        const p = PLANES.find((x) => x.id === t.plane)!
        return (
          <button key={t.plane} type="button" onClick={() => navigate(t.route)} data-panel={`tile-${t.plane}`} data-lit="false"
            className="group relative flex min-w-0 flex-col overflow-hidden rounded-[10px] border border-line bg-s1 p-4 text-left transition-colors hover:bg-s2">
            <span className="absolute inset-x-0 top-0 h-[2px]" style={{ background: p.color }} />
            <div className="mb-3 flex items-center gap-2">
              <span className="text-title font-semibold text-ink-1">{p.label}</span>
              {t.alert && <span className="size-2 rounded-full bg-critical" aria-label="attention" />}
              <ArrowUpRight size={14} className="ml-auto text-ink-3 opacity-0 transition-opacity group-hover:opacity-100" />
            </div>
            <dl className="space-y-2">
              {t.rows.map(([label, value, sub]) => (
                <div key={label as string} className="min-w-0">
                  <dt className="text-micro text-ink-3">{label}</dt>
                  <dd className="flex min-w-0 items-baseline gap-2">
                    <span className="tnum truncate text-body font-semibold text-ink-1">{value as ReactNode}</span>
                    <span className="truncate text-micro text-ink-3">{sub as ReactNode}</span>
                  </dd>
                </div>
              ))}
            </dl>
            <div className="mt-3 border-t border-line pt-2">
              <Sparkline values={t.trend.length > 1 ? t.trend : [0, 0]} color={p.color} height={26} />
              <div className="text-micro text-ink-3">{t.trendLabel}</div>
            </div>
          </button>
        )
      })}
    </div>
  )
}

// ------------------------------------------------------------------ critical feed

function CriticalFeed() {
  const stream = useLive((s) => s.stream)
  const inspect = useUi((s) => s.inspect)
  const rows = useMemo(() => {
    const out: Envelope[] = []
    for (let i = stream.length - 1; i >= 0 && out.length < 8; i--) {
      const e = stream[i]
      if (e.severity === 'warn' || e.severity === 'critical') out.push(e)
    }
    return out
  }, [stream])
  return (
    <Panel id="critical" title="Needs attention" subtitle="latest warnings and critical events" flush>
      {rows.length === 0 ? (
        <div className="px-4 py-10 text-center text-cap text-ink-3">No warnings. The swarm is handling routine work on its own.</div>
      ) : (
        <ul className="divide-y divide-line/60">
          {rows.map((e) => (
            <li key={e.seq}>
              <button type="button" className="row-in flex w-full items-start gap-3 px-4 py-2.5 text-left hover:bg-s2"
                onClick={() => (e.ledger ? inspect({ kind: 'ledger', id: String(e.ledger.seq) }) : inspect({ kind: 'raw', id: `event ${e.seq}`, title: e.summary, data: e }))}>
                <span className="mt-0.5"><SeverityIcon s={e.severity} /></span>
                <span className="min-w-0 flex-1">
                  <span className="line-clamp-2 text-cap text-ink-1">{e.summary}</span>
                  <span className="mt-0.5 flex items-center gap-2 text-micro text-ink-3">
                    <span className="tnum mono">{clockTime(e.ts, false)}</span>
                    <PlaneChip plane={e.plane} />
                    <span className="mono">{e.actor}</span>
                  </span>
                </span>
                {e.ledger && <span className="mono shrink-0 text-micro text-ink-3">#{e.ledger.seq}</span>}
              </button>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  )
}
