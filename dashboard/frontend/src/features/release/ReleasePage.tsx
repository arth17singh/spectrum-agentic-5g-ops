import clsx from 'clsx'
import { ArrowRight, CircleCheck, CircleDashed, FastForward, Pause, Play, RotateCcw, TriangleAlert } from 'lucide-react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { DataTable } from '@/charts/common'
import { LineChart } from '@/charts/LineChart'
import { api } from '@/core/api'
import { Gate, useFlag } from '@/core/flags'
import { clockTime, ms, pct } from '@/core/format'
import { useLive } from '@/core/store'
import type { Rec } from '@/core/types'
import { toast } from '@/core/ui-store'
import { Button, CodeBlock, EmptyState, Hash, LedgerLink, Meter, PageHeader, Panel, SignatureBadge, StatusPill, type Status } from '@/ui'
import manifest from './manifest'

const SRC = { aws: 'SageMaker + Bedrock CMI + ECS', mock: 'Simulated rollout controller' }

export default function ReleasePage() {
  const r = useLive((s) => s.release)
  const attacks = useFlag('demo.attacks')
  const cand: string | null = r.candidate
  const idx: number = r.stage_index ?? -1
  const last = idx >= (r.stages?.length ?? 4) - 1
  const act = (p: Promise<unknown>, msg: string, sev: 'notice' | 'warn' = 'notice') =>
    p.then(() => toast({ severity: sev, title: msg })).catch((e) => toast({ severity: 'critical', title: e.message }))
  return (
    <div>
      <PageHeader
        title={manifest.title}
        description={manifest.description}
        tracks={manifest.tracks}
        actions={
          <>
            <Button variant="primary" icon={<FastForward size={14} />} disabled={!cand || last} onClick={() => act(api.release('advance'), `Advanced ${cand} to ${r.stages?.[idx + 1]?.label}`)}>
              {cand && !last ? `Advance to ${r.stages?.[idx + 1]?.label}` : 'Advance'}
            </Button>
            <Button icon={r.paused ? <Play size={14} /> : <Pause size={14} />} disabled={!cand} onClick={() => act(api.release('pause'), r.paused ? 'Rollout resumed' : 'Rollout paused')}>
              {r.paused ? 'Resume' : 'Pause'}
            </Button>
            <Button variant="danger" icon={<RotateCcw size={13} />} disabled={!cand} onClick={() => act(api.release('rollback'), `${cand} rolled back (ledgered)`, 'warn')}>Roll back</Button>
            {attacks && (
              <Button variant="ghost" icon={<TriangleAlert size={14} />} disabled={!cand || idx < 1} title="Demo only: make the canary error-prone to show automatic rollback"
                onClick={() => act(api.release('regression'), 'Regression injected into the canary; guards will trip', 'warn')}>
                Inject regression (demo)
              </Button>
            )}
          </>
        }
      />
      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-12"><Rollout /></div>
        <div className="col-span-12"><Targets /></div>
        <div className="col-span-12 xl:col-span-7"><Agreement /></div>
        <div className="col-span-12 xl:col-span-5"><Disagreements /></div>
        <div className="col-span-12 xl:col-span-5"><Timeline /></div>
        <div className="col-span-12 xl:col-span-7"><ManifestViewer /></div>
        <Gate flag="release.dialects"><div className="col-span-12"><Dialects /></div></Gate>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ rollout track

function Rollout() {
  const r = useLive((s) => s.release)
  const navigate = useNavigate()
  const stages: Rec[] = r.stages ?? []
  const idx: number = r.stage_index ?? -1
  const g = r.guards ?? {}
  const L = r.guard_limits ?? {}
  return (
    <Panel id="rollout" plane="release" title="Rollout" subtitle={r.candidate ? `${r.candidate} replacing ${r.prod}` : `${r.prod} serving on every target`} source={SRC}
      actions={r.candidate ? (r.paused ? <StatusPill status="pending">Paused</StatusPill> : <StatusPill status="running">{stages[idx]?.label}</StatusPill>) : <StatusPill status="good">Stable</StatusPill>}>
      {!r.candidate ? (
        <EmptyState title="No candidate in rollout" body="A model that passes the Proving Ground gate is deployed here in shadow automatically."
          action={<Button size="sm" icon={<ArrowRight size={13} />} onClick={() => navigate('/proving')}>Open the Proving Ground</Button>} />
      ) : (
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1.3fr_1fr]">
          <ol className="flex items-stretch gap-2">
            {stages.map((s, i) => {
              const done = i < idx
              const cur = i === idx
              return (
                <li key={s.id} className="flex flex-1 items-center gap-2">
                  <div className={clsx('flex-1 rounded-md border px-3 py-2.5', cur ? 'border-release bg-s2' : 'border-line')}>
                    <div className="flex items-center gap-1.5 text-micro text-ink-3">
                      <span className="tnum">{i + 1}</span>
                      {done ? <CircleCheck size={12} className="text-good-ink" /> : cur ? <span className="size-2 rounded-full bg-release live-dot" style={{ color: 'var(--release)' }} /> : <CircleDashed size={12} />}
                    </div>
                    <div className={clsx('mt-1 text-body font-medium', i > idx ? 'text-ink-3' : 'text-ink-1')}>{s.label}</div>
                    <div className="tnum text-micro text-ink-3">{s.traffic}% of live traffic{s.id === 'shadow' ? ', decisions compared only' : ''}</div>
                    <div className="mt-2 h-1 overflow-hidden rounded-full bg-s3">
                      <div className="h-full rounded-full" style={{ width: done || cur ? '100%' : '0%', background: done ? 'var(--good)' : 'var(--release)' }} />
                    </div>
                  </div>
                  {i < stages.length - 1 && <ArrowRight size={14} className="shrink-0 text-ink-3" />}
                </li>
              )
            })}
          </ol>
          <div className="grid grid-cols-2 gap-x-6 gap-y-3">
            <Meter label="Error rate" right={`${g.error_rate?.toFixed(2)}% of ${L.error_rate_max}%`} value={g.error_rate ?? 0} max={L.error_rate_max * 1.5} warnAt={L.error_rate_max * 0.7} critAt={L.error_rate_max} />
            <Meter label="p95 latency" right={`${ms(g.p95_ms)} of ${ms(L.p95_max_ms)}`} value={g.p95_ms ?? 0} max={L.p95_max_ms * 1.4} warnAt={L.p95_max_ms * 0.85} critAt={L.p95_max_ms} />
            <Meter label="Safety incidents" right={`${g.safety_incidents} (max ${L.safety_incidents_max})`} value={g.safety_incidents ?? 0} max={3} critAt={1} color="var(--good)" />
            <Meter label="Shadow agreement" right={`${(g.agreement ?? 0).toFixed(1)}% (min ${L.agreement_min}%)`} value={g.agreement ?? 0} max={100} color={(g.agreement ?? 0) >= L.agreement_min ? 'var(--good)' : 'var(--warn)'} />
            <p className="col-span-2 text-micro text-ink-3">Breaching the error or latency guard during a canary rolls the candidate back automatically and writes the reason to the ledger.</p>
          </div>
        </div>
      )}
    </Panel>
  )
}

// ------------------------------------------------------------------ deployment matrix

const HEALTH: Record<string, Status> = { healthy: 'good', degraded: 'serious' }

function Targets() {
  const r = useLive((s) => s.release)
  const targets: Rec[] = r.targets ?? []
  return (
    <Panel id="targets" plane="release" title="Deployment targets" subtitle="one signed manifest, four targets, two vendors" source={SRC} flush>
      <div className="overflow-x-auto">
        <table className="w-full text-cap">
          <thead>
            <tr className="border-b border-line text-left text-micro text-ink-3">
              {['Target', 'Serving', 'Candidate', 'Traffic', 'State', 'p95', 'Error rate', 'Health', 'Stack', 'Manifest', 'Signature'].map((h) => <th key={h} className="px-3 py-2 font-medium">{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {targets.map((t) => (
              <tr key={t.id} className="border-b border-line/60 hover:bg-s2/50">
                <td className="px-3 py-2.5"><div className="font-medium text-ink-1">{t.name}</div><div className="text-micro text-ink-3">{t.detail}</div></td>
                <td className="mono px-3">{t.version}</td>
                <td className="mono px-3 text-ink-2">{t.candidate ?? '—'}</td>
                <td className="px-3">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-16 overflow-hidden rounded-full bg-s3"><div className="h-full rounded-full bg-release" style={{ width: `${t.traffic}%` }} /></div>
                    <span className="tnum text-ink-2">{t.traffic}%</span>
                  </div>
                </td>
                <td className="px-3"><StatusPill size="sm" status={t.state === 'canary' ? 'running' : t.state === 'shadow' ? 'pending' : 'neutral'}>{t.state}</StatusPill></td>
                <td className="tnum px-3" style={{ color: t.p95_ms > 600 ? 'var(--critical-ink)' : undefined }}>{ms(t.p95_ms)}</td>
                <td className="tnum px-3" style={{ color: t.error_rate > 1 ? 'var(--critical-ink)' : undefined }}>{pct(t.error_rate)}</td>
                <td className="px-3"><StatusPill size="sm" status={HEALTH[t.health] ?? 'neutral'}>{t.health}</StatusPill></td>
                <td className="mono px-3 text-micro text-ink-2">{t.stack}</td>
                <td className="px-3"><Hash value={t.manifest?.iac_hash} /></td>
                <td className="px-3"><SignatureBadge keyId="kms:spectrum/release" compact /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ shadow agreement

function Agreement() {
  const r = useLive((s) => s.release)
  const pts: Rec[] = r.agreement ?? []
  const t0 = pts[0]?.t ?? 0
  const series = [
    { id: 'agree', label: 'Identical decision', color: 'var(--release)', points: pts.map((p) => ({ x: p.t, y: p.agreement })) },
    { id: 'nonreg', label: 'Same or better (twin-judged)', color: 'var(--good)', points: pts.map((p) => ({ x: p.t, y: p.non_regression })) },
  ]
  return (
    <Panel id="agreement" plane="release" title="Shadow agreement" subtitle={r.candidate ? `${r.candidate} decisions compared with ${r.prod} on live incidents` : 'starts when a candidate enters shadow'} source={SRC}
      table={<DataTable columns={['Sim time', 'Identical', 'Same or better']} rows={[...pts].reverse().slice(0, 60).map((p) => [`${Math.round(p.t - t0)}s`, `${p.agreement}%`, `${p.non_regression}%`])} />}>
      {pts.length < 2 ? (
        <EmptyState title="No shadow traffic yet" body="Every live decision is replayed through the candidate; agreement appears here within seconds of shadow starting." />
      ) : (
        <LineChart series={series} height={220} yDomain={[60, 100]} thresholds={[{ y: r.guard_limits?.agreement_min ?? 90, label: 'guard 90%', color: 'var(--warn)' }]}
          xFormat={(x) => `${Math.round(x - t0)}s`} yFormat={(y) => `${Math.round(y)}%`} tipTitle={(x) => `${Math.round(x - t0)} s into shadow`} />
      )}
    </Panel>
  )
}

function Disagreements() {
  const r = useLive((s) => s.release)
  const rows: Rec[] = r.disagreements ?? []
  const t = r.tally ?? { better: 0, worse: 0, tie: 0 }
  return (
    <Panel id="disagreements" plane="release" title="Where they disagree" subtitle="the twin replays both decisions and scores the outcome" source={SRC} flush>
      <div className="grid grid-cols-3 gap-4 border-b border-line px-4 py-3">
        <div><div className="text-micro text-ink-3">Candidate better</div><div className="tnum text-[20px] font-semibold text-good-ink">{t.better}</div></div>
        <div><div className="text-micro text-ink-3">Candidate worse</div><div className="tnum text-[20px] font-semibold text-critical-ink">{t.worse}</div></div>
        <div><div className="text-micro text-ink-3">Identical</div><div className="tnum text-[20px] font-semibold">{t.tie}</div></div>
      </div>
      <div className="max-h-[200px] overflow-auto px-4 py-2">
        {rows.length === 0 && <div className="py-6 text-center text-cap text-ink-3">No disagreements yet.</div>}
        <ul>
          {rows.map((d, i) => (
            <li key={i} className="row-in border-b border-line/50 py-2 text-cap">
              <div className="flex items-center gap-2">
                <span className="mono text-ink-3">{clockTime(d.ts, false)}</span>
                <span className="mono text-ink-2">{d.site}</span>
                <span className="ml-auto"><StatusPill size="sm" status={d.winner === 'candidate' ? 'good' : 'serious'}>{d.winner === 'candidate' ? 'candidate better' : 'production better'}: {d.twin_delta}</StatusPill></span>
              </div>
              <div className="mt-0.5 grid grid-cols-2 gap-2 text-ink-1">
                <span><span className="text-ink-3">{r.prod}: </span>{d.prod}</span>
                <span><span className="text-ink-3">{r.candidate ?? 'candidate'}: </span>{d.candidate}</span>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ timeline

function Timeline() {
  const r = useLive((s) => s.release)
  const hist: Rec[] = [...(r.history ?? [])].reverse()
  return (
    <Panel id="timeline" plane="release" title="Version history" subtitle="every promotion and rollback is signed" source={{ aws: 'Spectrum ledger (DynamoDB)', mock: 'Ledger' }}>
      <ol className="relative max-h-[300px] overflow-auto">
        {hist.map((h, i) => {
          const bad = /roll/i.test(h.event)
          return (
            <li key={i} className="relative flex gap-3 pb-3 pl-5">
              <span className="absolute top-1.5 left-[3px] size-2 rounded-full" style={{ background: bad ? 'var(--critical)' : /100%|promoted/.test(h.event) ? 'var(--good)' : 'var(--release)' }} />
              {i < hist.length - 1 && <span className="absolute top-4 bottom-0 left-[6.5px] w-px bg-line" />}
              <div className="min-w-0 flex-1">
                <div className="text-cap text-ink-1"><span className="mono">{h.version}</span> <span className="text-ink-2">{h.event.replace(h.version, '').trim()}</span></div>
                <div className="mono text-micro text-ink-3">{h.ts?.slice(0, 10)} {clockTime(h.ts, false)}</div>
              </div>
              <LedgerLink seq={h.ledger_seq} />
            </li>
          )
        })}
      </ol>
    </Panel>
  )
}

// ------------------------------------------------------------------ manifest

function ManifestViewer() {
  const r = useLive((s) => s.release)
  const targets: Rec[] = r.targets ?? []
  const [sel, setSel] = useState('sagemaker')
  const t = targets.find((x) => x.id === sel) ?? targets[0]
  if (!t) return null
  const { signature, ...body } = t.manifest ?? {}
  return (
    <Panel id="manifest" plane="release" title="Deployment manifest" subtitle="the same artefact deploys everywhere; only the target changes" source={{ aws: 'AWS CDK + KMS', mock: 'Simulated manifest, real signature' }}
      actions={
        <select value={sel} onChange={(e) => setSel(e.target.value)} aria-label="Target" className="h-7 rounded-md border border-line bg-s1 px-2 text-cap text-ink-1">
          {targets.map((x) => <option key={x.id} value={x.id}>{x.name}</option>)}
        </select>
      }>
      <CodeBlock code={JSON.stringify(body, null, 2)} maxHeight={220} />
      <div className="mt-3 flex flex-wrap items-center gap-4 text-cap">
        <SignatureBadge keyId="kms:spectrum/release" alg="Ed25519" />
        <span className="text-ink-3">signature</span><Hash value={signature} head={12} tail={6} />
        <span className="text-ink-3">IaC</span><Hash value={body.iac_hash} />
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ vendor dialects

function Dialects() {
  const r = useLive((s) => s.release)
  const d = r.dialects
  if (!d) return null
  return (
    <Panel id="dialects" plane="release" title="Vendor dialect check" subtitle="one intent, compiled to each vendor's management interface" source={{ aws: 'Vendor EMS adapters on ECS', mock: 'Simulated adapters' }}>
      <div className="mb-3 text-cap"><span className="text-ink-3">Intent </span><span className="text-ink-1">{d.intent}</span></div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {(['vendor-a', 'vendor-b'] as const).map((v) => (
          <div key={v}>
            <div className="mb-1 flex items-center gap-2 text-cap">
              <span className="font-medium text-ink-1">{v === 'vendor-a' ? 'Vendor A (JSON over REST)' : 'Vendor B (CLI over SSH)'}</span>
              <StatusPill size="sm" status={d.valid?.[v] ? 'verified' : 'denied'}>{d.valid?.[v] ? 'Schema valid' : 'Invalid'}</StatusPill>
            </div>
            <CodeBlock code={d[v]} lang={v === 'vendor-a' ? 'json' : 'text'} />
          </div>
        ))}
      </div>
    </Panel>
  )
}
