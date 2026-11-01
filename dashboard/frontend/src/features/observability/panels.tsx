import clsx from 'clsx'
import { ExternalLink, OctagonX, TriangleAlert } from 'lucide-react'
import { useMemo, useState, type ReactNode } from 'react'
import { LatencyHeatmap } from '@/charts/LatencyHeatmap'
import { Sparkline } from '@/charts/Sparkline'
import { StackedArea } from '@/charts/StackedArea'
import { Waterfall, fmtDur, type Span } from '@/charts/Waterfall'
import { DataTable, Legend } from '@/charts/common'
import { useOnBeat } from '@/core/director'
import { compact, num, usd } from '@/core/format'
import { MODEL_COLOR, PLANES, modelColor, opColor, planeColor } from '@/core/palette'
import { EMPTY, useLive } from '@/core/store'
import type { Rec } from '@/core/types'
import { KV, Meter, Panel, StatusPill, Hash } from '@/ui'

export const SRC_CW = { aws: 'Amazon CloudWatch', mock: 'Simulated metrics' }

/** Frames inside the selected window (seconds of sim time). */
export function useFrames(windowS: number) {
  const frames = useLive((s) => s.observability.frames ?? EMPTY) as Rec[]
  return useMemo(() => {
    if (!frames.length) return frames
    const lastT = frames[frames.length - 1].t
    return frames.filter((f) => f.t >= lastT - windowS)
  }, [frames, windowS])
}

// ------------------------------------------------------------------ golden signals

const METRICS: { key: 'rate' | 'errors' | 'p95' | 'saturation'; label: string; unit: string; fmt: (v: number) => string }[] = [
  { key: 'rate', label: 'Rate', unit: 'ops/s', fmt: (v) => (v < 10 ? v.toFixed(1) : Math.round(v).toString()) },
  { key: 'errors', label: 'Errors', unit: '%', fmt: (v) => v.toFixed(2) },
  { key: 'p95', label: 'Duration p95', unit: 'ms', fmt: (v) => (v < 10 ? v.toFixed(1) : Math.round(v).toLocaleString()) },
  { key: 'saturation', label: 'Saturation', unit: '%', fmt: (v) => Math.round(v * 100).toString() },
]
// per-plane objectives; a cell breaches when the current value crosses its line
const P95_SLO: Record<string, number> = { learn: 600, verify: 1500, release: 600, operate: 3000, prove: 10 }
const PLANE_WHAT: Record<string, string> = {
  learn: 'FL steps and uploads', verify: 'eval tasks', release: 'endpoint requests', operate: 'agent decisions', prove: 'policy + ledger ops',
}

function breach(plane: string, key: string, v: number): 'warning' | 'critical' | null {
  if (key === 'errors') return v >= 5 ? 'critical' : v >= 1 ? 'warning' : null
  if (key === 'p95') return v >= P95_SLO[plane] * 1.5 ? 'critical' : v >= P95_SLO[plane] ? 'warning' : null
  if (key === 'saturation') return v >= 0.95 ? 'critical' : v >= 0.9 ? 'warning' : null
  return null
}

export function GoldenSignals({ windowS }: { windowS: number }) {
  const frames = useFrames(windowS)
  const last = frames[frames.length - 1]
  const breaches = last ? PLANES.flatMap((p) => METRICS.filter((m) => breach(p.id, m.key, last.golden[p.id]?.[m.key] ?? 0)).map((m) => `${p.label} ${m.label}`)) : []
  return (
    <Panel id="golden" title="Golden signals" subtitle="per plane, one row each" source={SRC_CW}
      actions={breaches.length ? <StatusPill status="warning" size="sm">{breaches.length} over objective</StatusPill> : <StatusPill status="good" size="sm">All within objectives</StatusPill>}
      table={last && <DataTable columns={['Plane', ...METRICS.map((m) => `${m.label} (${m.unit})`)]} rows={PLANES.map((p) => [p.label, ...METRICS.map((m) => m.fmt(last.golden[p.id]?.[m.key] ?? 0))])} />}>
      <div className="grid grid-cols-[150px_repeat(4,minmax(0,1fr))] gap-x-4">
        <div />
        {METRICS.map((m) => <div key={m.key} className="pb-2 text-micro text-ink-3">{m.label} <span className="text-ink-3/80">({m.unit})</span></div>)}
        {PLANES.map((p) => (
          <div key={p.id} className="contents">
            <div className="flex flex-col justify-center border-t border-line py-2">
              <span className="flex items-center gap-1.5 text-cap font-medium text-ink-1"><span className="h-3 w-[3px] rounded-full" style={{ background: p.color }} />{p.label}</span>
              <span className="pl-[9px] text-micro text-ink-3">{PLANE_WHAT[p.id]}</span>
            </div>
            {METRICS.map((m) => {
              const vals = frames.map((f) => f.golden?.[p.id]?.[m.key] ?? 0)
              const v = vals[vals.length - 1] ?? 0
              const b = breach(p.id, m.key, v)
              return (
                <div key={m.key} className="min-w-0 border-t border-line py-2">
                  <div className="flex items-baseline gap-1.5">
                    <span className="tnum text-body font-semibold text-ink-1">{m.fmt(v)}</span>
                    {b && (b === 'critical' ? <OctagonX size={12} className="self-center text-critical-ink" aria-label="critical" /> : <TriangleAlert size={12} className="self-center text-warn" aria-label="over objective" />)}
                    {m.key === 'p95' && <span className="text-micro text-ink-3">obj. {P95_SLO[p.id]}</span>}
                  </div>
                  <Sparkline values={vals} color={b ? (b === 'critical' ? 'var(--critical)' : 'var(--warn)') : p.color} height={22}
                    domain={m.key === 'saturation' ? [0, 1] : m.key === 'errors' ? [0, Math.max(1.2, ...vals)] : undefined} />
                </div>
              )
            })}
          </div>
        ))}
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ traces

const xrayId = (tid: string) => `1-${tid.slice(0, 8)}-${tid.slice(8, 32)}`

export function TracePanel() {
  const traces = useLive((s) => s.observability.traces ?? EMPTY) as Rec[]
  const spansBy = useLive((s) => s.observability.spans) as Record<string, Span[]> | undefined
  const mode = useLive((s) => s.mode)
  const withSpans = useMemo(() => [...traces].filter((t) => (spansBy?.[t.trace_id]?.length ?? 0) > 0).reverse(), [traces, spansBy])
  const [picked, setPicked] = useState<string | null>(null)
  const [spanId, setSpanId] = useState<string | null>(null)
  const preferred = withSpans.find((t) => t.incident) ?? withSpans[0]
  const tid = picked && withSpans.some((t) => t.trace_id === picked) ? picked : preferred?.trace_id
  useOnBeat(13, () => {
    const t = withSpans.find((x) => x.incident === 'INC-2041') ?? withSpans.find((x) => x.incident) ?? withSpans[0]
    if (t) { setPicked(t.trace_id); setSpanId(null) }
  })
  const spans: Span[] = (tid && spansBy?.[tid]) || []
  const trace = withSpans.find((t) => t.trace_id === tid)
  const sel = spans.find((s) => s.span_id === spanId) ?? spans.find((s) => s.name.startsWith('gen_ai')) ?? spans[0]
  const dur = spans.length ? Math.max(...spans.map((s) => s.end)) - Math.min(...spans.map((s) => s.start)) : 0
  const errors = spans.filter((s) => s.status === 'error').length
  const llm = spans.filter((s) => s.name.startsWith('gen_ai'))
  const tokens = llm.reduce((a, s) => a + Number(s.attrs['gen_ai.usage.input_tokens'] ?? 0) + Number(s.attrs['gen_ai.usage.output_tokens'] ?? 0), 0)
  return (
    <Panel id="trace" plane="operate" title="Trace" subtitle="OpenTelemetry spans across agents, gateway, policy and tools" source={{ aws: 'AWS X-Ray (via ADOT)', mock: 'Simulated spans' }}
      actions={
        <select value={tid ?? ''} onChange={(e) => { setPicked(e.target.value); setSpanId(null) }} aria-label="Choose trace"
          className="h-7 max-w-[300px] rounded-md border border-line bg-s2 px-2 text-cap text-ink-1">
          {withSpans.map((t) => <option key={t.trace_id} value={t.trace_id}>{t.title} ({spansBy?.[t.trace_id]?.length} spans)</option>)}
        </select>
      }>
      {!spans.length ? <div className="text-cap text-ink-3">No traces yet. Inject an incident to produce one.</div> : (
        <>
          <div className="mb-3 flex flex-wrap items-center gap-x-6 gap-y-1 text-cap">
            <span><span className="text-ink-3">trace </span><Hash value={tid} head={8} tail={6} /></span>
            <span className="tnum"><span className="text-ink-3">duration </span>{fmtDur(dur)}</span>
            <span className="tnum"><span className="text-ink-3">spans </span>{spans.length}</span>
            <span className="tnum"><span className="text-ink-3">LLM calls </span>{llm.length}</span>
            <span className="tnum"><span className="text-ink-3">tokens </span>{num(tokens)}</span>
            {errors > 0 ? <StatusPill status="critical" size="sm">{errors} error span{errors > 1 ? 's' : ''}</StatusPill> : <StatusPill status="good" size="sm">No errors</StatusPill>}
            <Legend className="ml-auto" items={PLANES.filter((p) => spans.some((s) => s.plane === p.id)).map((p) => ({ label: p.label, color: p.color, kind: 'rect' as const }))} />
          </div>
          <div className="grid grid-cols-[minmax(0,1fr)_320px] gap-4">
            <div className="max-h-[440px] overflow-auto pr-1">
              <Waterfall spans={spans} selected={sel?.span_id} onSelect={(s) => setSpanId(s.span_id)} />
            </div>
            <SpanDetail span={sel} trace={trace} mock={mode !== 'aws'} />
          </div>
        </>
      )}
    </Panel>
  )
}

function SpanDetail({ span, trace, mock }: { span?: Span; trace?: Rec; mock: boolean }) {
  if (!span) return null
  const genai = Object.entries(span.attrs).filter(([k]) => k.startsWith('gen_ai.'))
  const other = Object.entries(span.attrs).filter(([k]) => !k.startsWith('gen_ai.'))
  const url = `https://us-west-2.console.aws.amazon.com/cloudwatch/home?region=us-west-2#xray:traces/${xrayId(span.trace_id)}`
  return (
    <aside className="min-w-0 rounded-md border border-line bg-s2 p-3">
      <div className="flex items-start gap-2">
        <span className="mt-1.5 size-2 shrink-0 rounded-full" style={{ background: planeColor(span.plane) }} />
        <div className="min-w-0">
          <div className="mono break-all text-cap font-medium text-ink-1">{span.name}</div>
          <div className="tnum text-micro text-ink-3">{fmtDur(span.end - span.start)}, {span.plane} plane</div>
        </div>
        <span className="ml-auto"><StatusPill status={span.status === 'error' ? 'critical' : 'good'} size="sm">{span.status === 'error' ? 'Error' : 'OK'}</StatusPill></span>
      </div>
      <KV className="mt-3" rows={[
        ['span_id', <span className="mono">{span.span_id}</span>],
        ['parent', <span className="mono">{span.parent ?? 'root'}</span>],
        ['X-Ray id', <span className="mono break-all">{xrayId(span.trace_id)}</span>],
        ...(trace?.incident ? [['incident', trace.incident] as [ReactNode, ReactNode]] : []),
      ]} />
      {genai.length > 0 && (
        <>
          <div className="mt-3 mb-1 text-micro font-medium text-ink-2">GenAI semantic conventions</div>
          <KV rows={genai.map(([k, v]) => [<span className="mono">{k}</span>, <span className="mono tnum">{typeof v === 'number' ? num(v) : String(v)}</span>])} />
        </>
      )}
      {other.length > 0 && (
        <>
          <div className="mt-3 mb-1 text-micro font-medium text-ink-2">Span attributes</div>
          <KV rows={other.map(([k, v]) => [<span className="mono">{k}</span>, <span className="mono break-all">{String(v)}</span>])} />
        </>
      )}
      <a href={mock ? undefined : url} target="_blank" rel="noreferrer" aria-disabled={mock}
        title={mock ? `Available in AWS mode. Would open ${url}` : 'Open this trace in CloudWatch / X-Ray'}
        className={clsx('mt-3 inline-flex h-7 items-center gap-1.5 rounded-md border border-line px-2.5 text-cap', mock ? 'cursor-not-allowed text-ink-3' : 'text-ink-1 hover:bg-s3')}>
        <ExternalLink size={13} /> Open in X-Ray
      </a>
    </aside>
  )
}

// ------------------------------------------------------------------ tokens and cost

const MODEL_LABEL: Record<string, string> = {
  'claude-sonnet': 'Claude Sonnet (coordinator)', 'claude-haiku': 'Claude Haiku (domain)', 'nova-pro': 'Nova Pro (core)',
  'nova-lite': 'Nova Lite (energy)', 'nlm-v6': 'nlm-v6 (site agents)', 'nlm-v7': 'nlm-v7 (site agents)', 'nlm-v8': 'nlm-v8 (site agents)',
}

export function TokensCost({ windowS }: { windowS: number }) {
  const frames = useFrames(windowS)
  const obs = useLive((s) => s.observability)
  const last = frames[frames.length - 1]
  const prices: Record<string, [number, number]> = obs.prices ?? {}
  const costs: Rec[] = last?.incident_costs ?? obs.incident_costs ?? []
  const health = last?.agent_health ?? obs.agent_health ?? {}
  const tokensTotal: Record<string, [number, number]> = obs.tokens_total ?? {}
  const costTotal: Record<string, number> = obs.cost_total ?? {}

  // fixed model order: palette order first, then anything new
  const order = Object.keys(MODEL_COLOR)
  const present = new Set(frames.flatMap((f) => Object.keys(f.tokens_per_min ?? {})))
  const keys = [...order.filter((k) => present.has(k)), ...[...present].filter((k) => !order.includes(k))]
  const data = frames.map((f) => ({ x: f.t, ...Object.fromEntries(keys.map((k) => [k, f.tokens_per_min?.[k] ?? 0])) })) as ({ x: number } & Record<string, number>)[]
  const colors = Object.fromEntries(keys.map((k) => [k, modelColor(k)]))
  const lastT = last?.t ?? 0

  // site-agent decision cost, derived from observed tokens and modelled list prices
  const site = health.site
  const nlm = Object.entries(tokensTotal).filter(([k]) => k.startsWith('nlm')).reduce((a, [, v]) => [a[0] + v[0], a[1] + v[1]], [0, 0])
  const perCall = site?.calls ? site.tokens / site.calls : 1300
  const inShare = nlm[0] + nlm[1] ? nlm[0] / (nlm[0] + nlm[1]) : 0.87
  const tin = perCall * inShare
  const tout = perCall * (1 - inShare)
  const per1k = (m: string) => { const p = prices[m] ?? [0, 0]; return ((tin * p[0] + tout * p[1]) / 1e6) * 1000 }
  const ours = per1k('nlm-v6')
  const frontier = per1k('claude-sonnet')
  const spend = Object.values(costTotal).reduce((a, b) => a + b, 0)
  const tpm = last ? Object.values(last.tokens_per_min ?? {}).reduce((a: number, b) => a + (b as number), 0) : 0
  const maxCost = Math.max(0.01, ...costs.map((c) => c.frontier_usd))

  return (
    <Panel id="cost" plane="release" title="Tokens and cost" subtitle="by model; prices modelled at list-price assumptions" source={{ aws: 'CloudWatch GenAI observability', mock: 'Modelled from simulated calls' }}
      table={<DataTable columns={['Model', 'Input tokens', 'Output tokens', 'Spend (modelled)', 'Price in/out per 1M']}
        rows={Object.entries(tokensTotal).map(([m, v]) => [m, num(v[0]), num(v[1]), usd(costTotal[m] ?? 0), prices[m] ? `$${prices[m][0]} / $${prices[m][1]}` : '—'])} />}>
      <div className="mb-3 grid grid-cols-4 gap-4">
        <Kpi label="Tokens per minute" value={compact(tpm)} />
        <Kpi label="LLM calls per minute" value={num(last?.calls_per_min ?? 0)} />
        <Kpi label="Spend this session" value={usd(spend)} hint="modelled" />
        <Kpi label="Site-agent decisions, per 1K" value={usd(ours)} hint={`vs ${usd(frontier)} if run on Claude Sonnet (${Math.round(frontier / Math.max(ours, 1e-9))}× more)`} />
      </div>
      <div className="grid grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)] gap-5">
        <div>
          <div className="mb-1 text-micro text-ink-3">Tokens per minute, rolling 60 s window</div>
          <StackedArea data={data} keys={keys} colors={colors} labels={MODEL_LABEL} height={200} xFormat={(t) => `${Math.round(t - lastT)}s`} yFormat={(v) => compact(v)} />
        </div>
        <div>
          <div className="mb-1 text-micro text-ink-3">Cost per incident</div>
          <Legend className="mb-2" items={[{ label: 'SPECTRUM (fine-tuned site agents)', color: modelColor('nlm-v7'), kind: 'rect' }, { label: 'All-frontier, modelled', color: modelColor('claude-sonnet'), kind: 'rect' }]} />
          <ul className="space-y-2">
            {costs.slice(-7).map((c) => (
              <li key={c.incident} className="grid grid-cols-[72px_1fr_92px] items-center gap-2 text-cap">
                <span className="mono text-ink-2">{c.incident}</span>
                <span className="space-y-[3px]">
                  <span className="block h-[7px] rounded-r-[4px]" style={{ width: `${(c.usd / maxCost) * 100}%`, background: modelColor('nlm-v7') }} />
                  <span className="block h-[7px] rounded-r-[4px]" style={{ width: `${(c.frontier_usd / maxCost) * 100}%`, background: modelColor('claude-sonnet') }} />
                </span>
                <span className="tnum text-right text-ink-1">{usd(c.usd)} <span className="text-ink-3">/ {usd(c.frontier_usd)}</span></span>
              </li>
            ))}
          </ul>
          <p className="mt-3 text-micro text-ink-3">Costs are modelled: observed token counts × assumed list prices ({Object.entries(prices).filter(([k]) => ['claude-sonnet', 'nlm-v6'].includes(k)).map(([k, p]) => `${k} $${p[0]}/$${p[1]} per 1M`).join(', ')}). The fine-tuned endpoint is amortised instance cost, not a list price.</p>
        </div>
      </div>
    </Panel>
  )
}

function Kpi({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="truncate text-micro text-ink-3">{label}</div>
      <div className="text-[20px] leading-7 font-semibold text-ink-1">{value}</div>
      {hint && <div className="text-micro text-ink-3">{hint}</div>}
    </div>
  )
}

// ------------------------------------------------------------------ latency heatmap

export function LatencyPanel({ windowS }: { windowS: number }) {
  const heat = useLive((s) => s.observability.heat ?? EMPTY) as Rec[]
  const buckets: number[] = useLive((s) => s.observability.buckets ?? EMPTY)
  const cols = useMemo(() => {
    if (!heat.length) return heat
    const lastT = heat[heat.length - 1].t
    return heat.filter((h) => h.t >= lastT - windowS)
  }, [heat, windowS])
  const p95 = cols.length ? cols[cols.length - 1].p95 : 0
  const total = cols.reduce((a, c) => a + c.counts.reduce((x: number, y: number) => x + y, 0), 0)
  return (
    <Panel id="latency" plane="operate" title="Latency" subtitle="gateway, tool and model calls per second by latency bucket" source={{ aws: 'CloudWatch metrics (EMF)', mock: 'Simulated latency' }}>
      <div className="mb-2 flex items-center gap-5 text-cap">
        <span><span className="text-ink-3">p95 now </span><span className="tnum font-semibold text-ink-1">{p95.toFixed(0)} ms</span></span>
        <span><span className="text-ink-3">requests in window </span><span className="tnum font-semibold text-ink-1">{num(total)}</span></span>
        <Legend className="ml-auto" items={[{ label: 'p95', color: 'var(--warn)' }]} />
      </div>
      {buckets.length ? <LatencyHeatmap columns={cols as any} buckets={buckets} height={236} /> : null}
      <div className="mt-2 flex items-center gap-2 text-micro text-ink-3">
        <span>fewer</span>
        <span className="h-2 w-28 rounded-sm" style={{ background: 'linear-gradient(90deg, var(--seq-lo), var(--seq-hi))' }} />
        <span>more requests</span>
        <span className="ml-auto">log-spaced buckets; the slow band above 1 s is LLM reasoning, not the gateway</span>
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ SLOs

export function SloPanel() {
  const slos: Rec[] = useLive((s) => s.observability.slos ?? EMPTY)
  return (
    <Panel id="slo" title="Service level objectives" subtitle="error budget and burn rate" source={{ aws: 'CloudWatch Application Signals', mock: 'Simulated SLOs' }}>
      <ul className="space-y-3.5">
        {slos.map((s) => {
          const state = s.burn_1h >= 14.4 ? 'critical' : s.burn_1h >= 1 && s.burn_6h >= 1 ? 'warning' : s.burn_1h >= 1 ? 'pending' : 'good'
          const word = { critical: 'Page: fast burn', warning: 'Ticket: slow burn', pending: 'Watching', good: 'Healthy' }[state]
          return (
            <li key={s.id}>
              <div className="mb-1 flex items-baseline gap-2 text-cap">
                <span className="truncate font-medium text-ink-1">{s.name}</span>
                <span className="tnum text-ink-3">target {s.target}%, now {s.current}%</span>
                <span className="ml-auto"><StatusPill status={state as any} size="sm">{word}</StatusPill></span>
              </div>
              <div className="grid grid-cols-[1fr_auto] items-center gap-3">
                <Meter value={s.budget_left} max={1} color={s.budget_left < 0.25 ? 'var(--critical)' : s.budget_left < 0.5 ? 'var(--warn)' : 'var(--release)'} />
                <span className="tnum w-[210px] text-right text-micro whitespace-nowrap text-ink-2">{Math.round(s.budget_left * 100)}% left, burn {s.burn_1h}× 1h, {s.burn_6h}× 6h</span>
              </div>
            </li>
          )
        })}
      </ul>
      <p className="mt-3 text-micro text-ink-3">Burn rate 1× spends the budget exactly over the 30-day window. Paging at 14.4× over 1 h follows the multi-window, multi-burn-rate practice.</p>
    </Panel>
  )
}

// ------------------------------------------------------------------ agent health

const CLASS_LABEL: Record<string, string> = { site: 'Site agents (40)', domain: 'Domain agents (5)', coordinator: 'Coordinator' }

export function AgentHealth({ windowS }: { windowS: number }) {
  const frames = useFrames(windowS)
  const snap = useLive((s) => s.observability.agent_health)
  const swarm = useLive((s) => s.swarm.agents ?? EMPTY) as Rec[]
  const health: Record<string, Rec> = frames[frames.length - 1]?.agent_health ?? snap ?? {}
  const models = (cls: string) => [...new Set(swarm.filter((a) => a.class === cls).map((a) => a.model))].join(', ')
  const rows = ['site', 'domain', 'coordinator'].filter((c) => health[c])
  const pct = (a: number, b: number) => (b ? `${((a / b) * 100).toFixed(2)}%` : '—')
  return (
    <Panel id="agent-health" plane="operate" title="Agent health" subtitle="by agent class, since start of session" source={{ aws: 'AgentCore Observability', mock: 'Simulated agents' }} flush>
      <table className="w-full text-cap">
        <thead>
          <tr className="border-b border-line text-left text-micro text-ink-3">
            {['Class', 'Models', 'Calls', 'Tool errors', 'Policy denies', 'Escalated to human', 'Think time', 'Tokens per decision'].map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((c) => {
            const h = health[c]
            return (
              <tr key={c} className="border-b border-line/60">
                <td className="px-4 py-2.5 font-medium text-ink-1">{CLASS_LABEL[c]}</td>
                <td className="mono px-4 text-micro text-ink-2">{models(c)}</td>
                <td className="tnum px-4">{num(h.calls)}</td>
                <td className="tnum px-4">{pct(h.tool_errors, h.calls)}</td>
                <td className="tnum px-4" style={{ color: h.denies ? 'var(--critical-ink)' : undefined }}>{num(h.denies)} <span className="text-ink-3">({pct(h.denies, h.calls)})</span></td>
                <td className="tnum px-4">{num(h.hitl)}</td>
                <td className="tnum px-4">{fmtDur((h.think_ms ?? 0) / 1000)}</td>
                <td className="tnum px-4">{h.calls ? num(h.tokens / h.calls) : '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </Panel>
  )
}

// ------------------------------------------------------------------ infra

export function Infra({ windowS }: { windowS: number }) {
  const frames = useFrames(windowS)
  const ops = useLive((s) => s.federation.operators ?? EMPTY) as Rec[]
  const targets = useLive((s) => s.release.targets ?? EMPTY) as Rec[]
  return (
    <Panel id="infra" title="Infrastructure" subtitle="FL client GPUs and model endpoints" source={{ aws: 'CloudWatch (ECS, SageMaker)', mock: 'Simulated infra' }}>
      <div className="mb-1 text-micro text-ink-3">GPU utilisation per operator task (ECS on g6.xlarge)</div>
      <div className="grid grid-cols-5 gap-3">
        {ops.map((o) => {
          const vals = frames.map((f) => (f.gpu?.[o.id] ?? 0) * 100)
          return (
            <div key={o.id} className="min-w-0 rounded-md border border-line px-2.5 py-2">
              <div className="flex items-center gap-1.5 text-micro text-ink-2"><span className="size-1.5 rounded-full" style={{ background: opColor(o.id) }} />{o.short}</div>
              <div className="tnum text-body font-semibold text-ink-1">{Math.round(vals[vals.length - 1] ?? 0)}%</div>
              <Sparkline values={vals} color={opColor(o.id)} height={26} domain={[0, 100]} />
            </div>
          )
        })}
      </div>
      <div className="mt-4 mb-1 text-micro text-ink-3">Model endpoints</div>
      <table className="w-full text-cap">
        <thead><tr className="text-left text-micro text-ink-3">{['Target', 'Version', 'Traffic', 'p95', 'Errors', 'Health'].map((h) => <th key={h} className="py-1 pr-3 font-medium">{h}</th>)}</tr></thead>
        <tbody>
          {targets.map((t) => (
            <tr key={t.id} className="border-t border-line/60">
              <td className="py-1.5 pr-3 text-ink-1">{t.name}</td>
              <td className="mono pr-3 text-ink-2">{t.version}{t.candidate ? ` + ${t.candidate}` : ''}</td>
              <td className="tnum pr-3">{t.candidate ? `${t.traffic}%` : '100%'}</td>
              <td className="tnum pr-3">{t.p95_ms ? `${t.p95_ms} ms` : '—'}</td>
              <td className="tnum pr-3">{(t.error_rate ?? 0).toFixed(2)}%</td>
              <td className="pr-3"><StatusPill status={t.health === 'healthy' ? 'good' : 'serious'} size="sm">{t.health}</StatusPill></td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  )
}
