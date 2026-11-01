import clsx from 'clsx'
import { CircleCheck, CircleDashed, CircleX, Loader2, ShieldCheck, ShieldX } from 'lucide-react'
import { DataTable, Legend } from '@/charts/common'
import { Heatmap } from '@/charts/Heatmap'
import { ParallelCoords, type PCAxis, type PCLine } from '@/charts/ParallelCoords'
import { Scatter, type Dot } from '@/charts/Scatter'
import { clockTime, ms, pct, usd } from '@/core/format'
import { useLive } from '@/core/store'
import type { Rec } from '@/core/types'
import { useUi } from '@/core/ui-store'
import { Panel, Stat, StatusPill } from '@/ui'
import { isVersion, MOCK_LABEL, paretoIds, REDTEAM_EXAMPLES, usdPerIncident, versionNum, versions, type ModelRow } from './model'

export const SRC = { aws: 'Step Functions + SageMaker', mock: MOCK_LABEL }
export const SRC_BEDROCK = { aws: 'SageMaker + Bedrock evaluation', mock: MOCK_LABEL }

const CAND = 'var(--verify)'
const PROD = 'var(--s1)'
const OURS = 'var(--s3)'
const BASE = 'var(--neutral-series)'
export const modelTone = (m: ModelRow, sel: string, prod: string) =>
  m.id === sel ? CAND : m.id === prod ? PROD : m.family === 'ours' ? OURS : BASE

export function ToneLegend() {
  return <Legend items={[{ label: 'Candidate', color: CAND, kind: 'dot' }, { label: 'Production', color: PROD, kind: 'dot' }, { label: 'Other federated NLMs', color: OURS, kind: 'dot' }, { label: 'Baselines', color: BASE, kind: 'dot' }]} />
}

// ------------------------------------------------------------------ pipeline

export function Pipeline() {
  const p = useLive((s) => s.proving)
  const a = p.active
  const stages: Rec[] = a?.stages ?? (p.stages ?? []).map((s: Rec) => ({ id: s.id, label: s.label, state: 'queued', progress: 0, duration: null }))
  const runs: Rec[] = [...(p.runs ?? [])].reverse().slice(0, 4)
  return (
    <Panel id="pipeline" plane="verify" title="Verification pipeline" subtitle={a ? `${a.id}, ${a.model}` : 'no evaluation has run yet in this session'} source={{ aws: 'AWS Step Functions', mock: 'Simulated state machine' }}
      actions={a && <StatusPill status={a.state === 'running' ? 'running' : a.state === 'passed' ? 'passed' : 'blocked'}>{a.state === 'running' ? 'Running' : a.state === 'passed' ? 'Gate passed' : 'Blocked'}</StatusPill>}>
      <ol className="grid grid-cols-9 gap-2">
        {stages.map((s, i) => {
          const running = s.state === 'running'
          const done = s.state === 'passed'
          const failed = s.state === 'failed'
          return (
            <li key={s.id} className={clsx('min-w-0 rounded-md border px-2.5 py-2', running ? 'border-verify/70 bg-s2' : 'border-line')}>
              <div className="flex items-center gap-1.5 text-micro text-ink-3">
                <span className="tnum">{i + 1}</span>
                {done ? <CircleCheck size={12} className="text-good-ink" /> : running ? <Loader2 size={12} className="animate-spin text-verify" /> : failed ? <CircleX size={12} className="text-critical-ink" /> : <CircleDashed size={12} />}
                <span className="tnum ml-auto">{s.duration != null ? `${s.duration.toFixed(1)}s` : running ? `${Math.round(s.progress * 100)}%` : ''}</span>
              </div>
              <div className={clsx('mt-1 truncate text-cap font-medium', s.state === 'queued' ? 'text-ink-3' : 'text-ink-1')}>{s.label}</div>
              <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-s3">
                <div className="h-full rounded-full transition-[width] duration-300" style={{ width: `${(done ? 1 : s.progress) * 100}%`, background: done ? 'var(--good)' : 'var(--verify)' }} />
              </div>
              <div className="tnum mt-1 h-4 truncate text-micro text-ink-3">{s.detail ?? (done ? 'passed' : '')}</div>
            </li>
          )
        })}
      </ol>
      <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-1 text-micro text-ink-3">
        {a && <span>Execution <span className="mono text-ink-2">{a.execution_arn}</span></span>}
        {a && <span>Started <span className="mono text-ink-2">{clockTime(a.started_at)}</span></span>}
        {runs.length > 0 && (
          <span className="flex flex-wrap items-center gap-2">Recent runs
            {runs.map((r) => (
              <span key={r.id} className="flex items-center gap-1 rounded border border-line px-1.5 py-0.5 text-ink-2">
                <span className="mono">{r.id}</span> {r.model} {r.result?.passed ? <CircleCheck size={11} className="text-good-ink" /> : <CircleX size={11} className="text-critical-ink" />}
              </span>
            ))}
          </span>
        )}
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ scorecard

export function Scorecard({ m, prod, models, gate }: { m: ModelRow; prod: ModelRow; models: ModelRow[]; gate: Rec }) {
  const vs = versions(models).filter((v) => !isVersion(m.id) || versionNum(v.id) <= versionNum(m.id))
  const trend = (k: keyof ModelRow) => vs.map((v) => v[k] as number)
  const d = (a: number, b: number, digits = 1, scale = 100) => `${a - b >= 0 ? '+' : '−'}${Math.abs((a - b) * scale).toFixed(digits)}`
  const same = m.id === prod.id
  const frontier = models.filter((x) => x.family === 'baseline').sort((a, b) => b.accuracy - a.accuracy)[0]
  return (
    <Panel id="scorecard" plane="verify" title="Scorecard" subtitle={same ? `${m.id} (production)` : `${m.id} against production ${prod.id}`} source={SRC}>
      <div className="grid grid-cols-2 gap-6 lg:grid-cols-4">
        <Stat label="Accuracy, weighted task suite" value={`${(m.accuracy * 100).toFixed(1)}%`} delta={same ? undefined : `${d(m.accuracy, prod.accuracy)} pts`} deltaGood={same ? null : m.accuracy >= prod.accuracy}
          trend={trend('accuracy')} trendColor={CAND} hint={`gate ≥ ${pct(gate.accuracy_min, true)}, tool-call validity ${pct(m.tool_validity, true)}`} flashKey={m.accuracy} />
        <Stat label="Safety, red-team pass rate" value={`${(m.safety * 100).toFixed(1)}%`} delta={same ? undefined : `${d(m.safety, prod.safety)} pts`} deltaGood={same ? null : m.safety >= prod.safety}
          trend={trend('safety')} trendColor={CAND}
          hint={`${Object.values(m.redteam).reduce((a, [p, n]) => a + (n - p), 0)} of 64 attacks succeeded, gate ≥ ${pct(gate.safety_min, true)}`} flashKey={m.safety} />
        <Stat label="Performance, p95 latency" value={ms(m.p95_ms)} delta={same ? undefined : `${m.p95_ms - prod.p95_ms >= 0 ? '+' : '−'}${Math.abs(m.p95_ms - prod.p95_ms)} ms`} deltaGood={same ? null : m.p95_ms <= prod.p95_ms}
          trend={trend('p95_ms')} trendColor={CAND} hint={`p50 ${ms(m.p50_ms)}, ${m.throughput_rps} req/s at concurrency 16`} flashKey={m.p95_ms} />
        <Stat label="Price per 1K tasks" value={usd(m.usd_per_1k)} delta={same ? undefined : m.usd_per_1k === prod.usd_per_1k ? 'no change' : usd(m.usd_per_1k - prod.usd_per_1k)} deltaGood={same ? null : m.usd_per_1k <= prod.usd_per_1k}
          trend={trend('usd_per_1k')} trendColor={CAND}
          hint={<>{usd(usdPerIncident(m))} per incident (modelled), {frontier ? `${Math.round(frontier.usd_per_1k / Math.max(1e-6, m.usd_per_1k))}× cheaper than ${frontier.name.replace(' (Bedrock)', '')}` : ''}</>} flashKey={m.usd_per_1k} />
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ pareto

export function Pareto({ models, sel, prod }: { models: ModelRow[]; sel: string; prod: string }) {
  const front = paretoIds(models)
  const dots: Dot[] = models.map((m) => ({
    id: m.id, x: m.usd_per_1k, y: m.accuracy * 100, color: modelTone(m, sel, prod), label: m.id,
    showLabel: front.includes(m.id) || m.id === sel, ring: m.id === sel, title: m.name,
    tip: [{ label: 'accuracy', value: pct(m.accuracy, true) }, { label: '$ per 1K tasks', value: usd(m.usd_per_1k) }, { label: 'p95', value: ms(m.p95_ms) }, { label: 'safety', value: pct(m.safety, true) }],
  }))
  return (
    <Panel id="pareto" plane="verify" title="Pareto frontier" subtitle="cost against accuracy, log scale" source={SRC_BEDROCK}
      table={<DataTable columns={['Model', 'Accuracy', '$ / 1K', 'On frontier']} rows={[...models].sort((a, b) => a.usd_per_1k - b.usd_per_1k).map((m) => [m.name, pct(m.accuracy, true), usd(m.usd_per_1k), front.includes(m.id) ? 'yes' : ''])} />}>
      <Scatter dots={dots} height={250} xLog frontier={front} xFormat={(v) => usd(v)} yFormat={(v) => `${v.toFixed(0)}%`} xLabel="$ per 1K tasks" yLabel="accuracy" />
      <div className="mt-2"><ToneLegend /></div>
      <p className="mt-2 text-micro text-ink-3">The line joins models no other model beats on both cost and accuracy. Up and to the left is better.</p>
    </Panel>
  )
}

// ------------------------------------------------------------------ parallel coordinates

export function Parallel({ models, sel, prod }: { models: ModelRow[]; sel: string; prod: string }) {
  const axes: PCAxis[] = [
    { id: 'accuracy', label: 'Accuracy', domain: [0.4, 1], format: (v) => `${Math.round(v * 100)}%` },
    { id: 'safety', label: 'Safety', domain: [0.7, 1], format: (v) => `${Math.round(v * 100)}%` },
    { id: 'tool', label: 'Tool-call validity', domain: [0.6, 1], format: (v) => `${Math.round(v * 100)}%` },
    { id: 'p95', label: 'p95 latency', invert: true, domain: [0, 3000], format: (v) => (v === 0 ? '0 ms' : ms(v)) },
    { id: 'tput', label: 'Throughput', domain: [0, 110], format: (v) => `${Math.round(v)}/s` },
    { id: 'price', label: '$ per 1K', invert: true, domain: [Math.log10(0.01), Math.log10(20)], format: (v) => usd(10 ** v) },
  ]
  const lines: PCLine[] = models.map((m) => ({
    id: m.id, label: m.id, color: modelTone(m, sel, prod), emphasis: m.id === sel || m.id === prod, showLabel: m.id === sel, muted: m.family === 'baseline',
    values: { accuracy: m.accuracy, safety: m.safety, tool: m.tool_validity, p95: m.p95_ms, tput: m.throughput_rps, price: Math.log10(m.usd_per_1k) },
  }))
  return (
    <Panel id="parallel" plane="verify" title="Model comparison" subtitle="six axes, better is always up" source={SRC_BEDROCK}>
      <ParallelCoords axes={axes} lines={lines} height={260} />
      <div className="mt-2"><ToneLegend /></div>
    </Panel>
  )
}

// ------------------------------------------------------------------ task heatmap

export function TaskHeatmap({ models, tasks }: { models: ModelRow[]; tasks: Rec[] }) {
  const matrix = models.map((m) => tasks.map((t) => m.tasks[t.id]))
  return (
    <Panel id="tasks" plane="verify" title="Task accuracy" subtitle="1,840 network-ops tasks across seven categories" source={SRC}
      table={<DataTable columns={['Model', ...tasks.map((t) => t.label)]} rows={models.map((m) => [m.id, ...tasks.map((t) => pct(m.tasks[t.id], true))])} />}>
      <Heatmap matrix={matrix} scheme="sequential" domain={[0.3, 1]} rowLabels={models.map((m) => m.id)} colLabels={tasks.map((t) => t.label)}
        cellText format={(v) => v.toFixed(2)} height={models.length * 26 + 18} gap={2} ariaLabel="Task accuracy by model" />
      <div className="mt-2 flex items-center gap-2 text-micro text-ink-3">
        <span>0.30</span><span className="h-2 w-28 rounded-sm" style={{ background: 'linear-gradient(90deg, var(--seq-lo), var(--seq-hi))' }} /><span>1.00</span>
        <span className="ml-3">Vendor A and B config tasks are validated by a parser, not a judge.</span>
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ red-team matrix

export function RedTeam({ models, attacks, sel }: { models: ModelRow[]; attacks: Rec[]; sel: string }) {
  const inspect = useUi((s) => s.inspect)
  return (
    <Panel id="redteam" plane="verify" title="Safety red-team" subtitle="64 attacks in seven classes; click a cell for an example" source={SRC} flush>
      <div className="overflow-x-auto px-4 py-3">
        <table className="w-full text-cap">
          <thead>
            <tr>
              <th className="pb-2 text-left text-micro font-medium text-ink-3">Attack class</th>
              {models.map((m) => (
                <th key={m.id} className={clsx('px-1 pb-2 text-center text-micro font-medium', m.id === sel ? 'text-verify' : 'text-ink-3')}>
                  <span className="block max-w-[64px] truncate" title={m.name}>{m.id.replace('claude-', '').replace('llama-3.2-1b', 'llama 1B')}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {attacks.map((a) => (
              <tr key={a.id} className="border-t border-line/60">
                <td className="py-1.5 pr-2 text-ink-2">{a.label}</td>
                {models.map((m) => {
                  const [pass, n] = m.redteam[a.id]
                  const ok = pass === n
                  return (
                    <td key={m.id} className="px-0.5 text-center">
                      <button type="button"
                        onClick={() => inspect({ kind: 'raw', id: `${m.id} ${a.id}`, title: `${m.id}: ${a.label}`, data: { model: m.id, attack: a.label, passed: pass, total: n, example: REDTEAM_EXAMPLES[a.id], verdict: ok ? 'model refused every attack in this class' : `${n - pass} attack(s) succeeded`, scorer: 'judge + deterministic validators v0.9' } })}
                        className={clsx('inline-flex w-full items-center justify-center gap-1 rounded py-1 hover:bg-s2', m.id === sel && 'bg-verify/10')}
                        title={`${pass}/${n} attacks refused`}>
                        {ok ? <ShieldCheck size={13} className="text-good-ink" /> : <ShieldX size={13} className="text-critical-ink" />}
                        <span className={clsx('tnum text-micro', ok ? 'text-ink-3' : 'text-critical-ink')}>{pass}/{n}</span>
                      </button>
                    </td>
                  )
                })}
              </tr>
            ))}
            <tr className="border-t border-line">
              <td className="py-1.5 pr-2 font-medium text-ink-1">Pass rate</td>
              {models.map((m) => <td key={m.id} className="tnum text-center text-micro font-medium text-ink-1">{pct(m.safety, true)}</td>)}
            </tr>
          </tbody>
        </table>
      </div>
    </Panel>
  )
}
