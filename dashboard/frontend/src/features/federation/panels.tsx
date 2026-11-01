import { useMemo } from 'react'
import { Gantt, type GanttRow } from '@/charts/Gantt'
import { LineChart, type Series } from '@/charts/LineChart'
import { Ridgeline } from '@/charts/Ridgeline'
import { Scatter, type Dot } from '@/charts/Scatter'
import { DataTable } from '@/charts/common'
import { opColor } from '@/core/palette'
import { useLive } from '@/core/store'
import type { FlRound, Operator, Rec } from '@/core/types'
import { useSimNow } from '@/core/ui-store'
import { Panel, StatusPill } from '@/ui'

export const useFed = () => useLive((s) => s.federation)
const SRC_FL = { aws: 'ECS tasks + Flower', mock: 'Simulated training' }

// ------------------------------------------------------------------ round timeline

export function RoundTimeline() {
  const f = useFed()
  const now = useSimNow(8)
  const rounds: FlRound[] = f.rounds_log ?? []
  const ops: Operator[] = f.operators ?? []
  const prev = rounds[rounds.length - 1]
  const cur = f.current
  const t0 = prev ? prev.t0 : cur?.t0 ?? now - 30
  const t1 = cur ? Math.max(now, cur.t0 + 26) : Math.max(now, (prev?.t1 ?? now) + 4)
  const segColor: Record<string, (op: string) => string> = {
    pull: () => 'var(--ink-3)', train: (op) => opColor(op), secure: (op) => `color-mix(in oklab, ${opColor(op)} 55%, var(--surface-1))`,
    upload: (op) => opColor(op),
  }
  const label: Record<string, string> = { pull: 'pull global adapter', train: 'local DP-SGD (clip + noise per step)', secure: 'mask + encrypt to enclave', upload: 'upload masked update' }
  const rows: GanttRow[] = ops.map((o) => {
    const segs = [...(prev ? prev.phases[o.id] ?? [] : []), ...(cur ? cur.phases[o.id] ?? [] : [])].map((s) => ({
      start: s.start, end: s.end, label: label[s.phase] ?? s.phase, color: segColor[s.phase]?.(o.id) ?? 'var(--ink-3)',
      hatched: s.phase === 'upload', opacity: s.phase === 'pull' ? 0.5 : 0.95,
    }))
    return { id: o.id, label: o.short, color: opColor(o.id), segs }
  })
  rows.push({
    id: 'agg', label: 'Enclave', color: 'var(--learn)',
    segs: [...(prev?.aggregator ?? []), ...(cur?.aggregator ?? [])].map((s) => ({ start: s.start, end: s.end, label: 'secure aggregation (robust)', color: 'var(--learn)' })),
  })
  const dividers = [prev && { x: prev.t0, label: `Round ${prev.round}` }, cur && { x: cur.t0, label: `Round ${cur.round}` }].filter(Boolean) as { x: number; label: string }[]
  return (
    <Panel id="round-timeline" plane="learn" title="Round timeline" subtitle={cur ? `round ${cur.round}, ${f.phase}` : f.state === 'ready' ? `round ${(f.round ?? 0) + 1} queued` : 'between rounds'}
      source={SRC_FL}>
      <Gantt rows={rows} domain={[t0, t1]} now={now} dividers={dividers} xFormat={(v) => `${Math.round(v - t0)}s`} />
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-micro text-ink-3">
        <span className="flex items-center gap-1.5"><span className="h-2 w-3 rounded-sm bg-ink-3/50" /> pull</span>
        <span className="flex items-center gap-1.5"><span className="h-2 w-3 rounded-sm" style={{ background: 'var(--s1)' }} /> local DP-SGD</span>
        <span className="flex items-center gap-1.5"><span className="h-2 w-3 rounded-sm" style={{ background: 'color-mix(in oklab, var(--s1) 55%, var(--surface-1))' }} /> mask + encrypt</span>
        <span className="flex items-center gap-1.5"><span className="h-2 w-3 rounded-sm" style={{ background: 'repeating-linear-gradient(45deg, var(--s1) 0 2px, color-mix(in oklab, var(--s1) 50%, white) 2px 4px)' }} /> upload</span>
        <span className="flex items-center gap-1.5"><span className="h-2 w-3 rounded-sm" style={{ background: 'var(--learn)' }} /> aggregate in enclave</span>
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ convergence

export function Convergence() {
  const f = useFed()
  const rounds: FlRound[] = f.rounds_log ?? []
  const ops: Operator[] = f.operators ?? []
  const ann = (f.annotations ?? []).map((a: Rec) => ({ x: a.round, label: a.label, kind: a.kind }))
  const series: Series[] = [
    ...ops.map((o) => ({ id: o.id, label: `${o.short} alone`, color: opColor(o.id), width: 1.25, opacity: 0.55, dashed: true,
      points: rounds.map((r) => ({ x: r.round, y: r.local_only[o.id] })) })),
    { id: 'fed', label: 'Federated', color: 'var(--learn)', width: 2.5, endLabel: true, area: true, points: rounds.map((r) => ({ x: r.round, y: r.val_loss })) },
  ]
  const acc: Series[] = [{ id: 'acc', label: 'Federated accuracy', color: 'var(--learn)', endLabel: false, points: rounds.map((r) => ({ x: r.round, y: r.accuracy * 100 })) }]
  const last = rounds[rounds.length - 1]
  const bestAlone = last ? Math.min(...Object.values(last.local_only)) : 0
  return (
    <Panel id="convergence" plane="learn" title="Convergence" subtitle="validation loss per round" source={SRC_FL}
      table={<DataTable columns={['Round', 'Federated loss', 'Accuracy', ...ops.map((o) => `${o.short} alone`)]}
        rows={[...rounds].reverse().map((r) => [r.round, r.val_loss.toFixed(3), `${(r.accuracy * 100).toFixed(1)}%`, ...ops.map((o) => r.local_only[o.id]?.toFixed(3))])} />}>
      {last && (
        <div className="mb-2 flex gap-6">
          <div><div className="text-micro text-ink-3">Federated loss</div><div className="tnum text-[18px] font-semibold">{last.val_loss.toFixed(3)}</div></div>
          <div><div className="text-micro text-ink-3">Best operator alone</div><div className="tnum text-[18px] font-semibold text-ink-2">{bestAlone.toFixed(3)}</div></div>
          <div><div className="text-micro text-ink-3">Federation gain</div><div className="tnum text-[18px] font-semibold text-good-ink">−{(((bestAlone - last.val_loss) / bestAlone) * 100).toFixed(0)}%</div></div>
        </div>
      )}
      <LineChart series={series} height={190} annotations={ann} xFormat={(x) => `r${x}`} yFormat={(y) => y.toFixed(2)} tipTitle={(x) => `Round ${x}`} legend={false} />
      <div className="mt-1 mb-1 text-micro text-ink-3">Task accuracy (root-cause exact match)</div>
      <LineChart series={acc} height={80} xFormat={(x) => `r${x}`} yFormat={(y) => `${y.toFixed(0)}%`} yTicks={2} />
      <div className="mt-2 text-micro text-ink-3">Dashed lines: what each operator reaches training alone on its own data. Diamonds mark checkpoints and config changes.</div>
    </Panel>
  )
}

// ------------------------------------------------------------------ privacy budget

export function PrivacyBudget() {
  const f = useFed()
  const rounds: FlRound[] = f.rounds_log ?? []
  const ops: Operator[] = f.operators ?? []
  const target = f.config?.target_eps ?? 3
  const spr = f.steps_per_round ?? 240
  const series: Series[] = ops.map((o) => {
    const pts = [{ x: 0, y: 0 }, ...rounds.map((r) => ({ x: r.round, y: r.eps[o.id] }))]
    if (f.current && f.current.participants?.includes(o.id)) pts.push({ x: f.current.round - 1 + Math.min(1, o.step / spr), y: o.eps })
    return { id: o.id, label: o.short, color: opColor(o.id), points: pts, endLabel: true }
  })
  const worst = [...ops].sort((a, b) => b.eps - a.eps)[0]
  return (
    <Panel id="privacy" plane="learn" title="Privacy budget" subtitle="cumulative ε per operator, δ = 1e-5" source={{ aws: 'Opacus RDP accountant', mock: 'Approximate accountant' }}
      table={<DataTable columns={['Operator', 'ε', 'of target', 'Sampling rate q', 'Status']} rows={ops.map((o) => [o.short, o.eps.toFixed(3), `${((o.eps / target) * 100).toFixed(0)}%`, o.q.toFixed(5), o.status])} />}>
      {worst && (
        <div className="mb-2 flex items-center gap-3">
          <div className="tnum text-[18px] font-semibold">ε = {worst.eps.toFixed(2)}</div>
          <div className="text-cap text-ink-3">highest, {worst.short}, {((worst.eps / target) * 100).toFixed(0)}% of target {target}</div>
          {worst.eps >= 0.9 * target && <StatusPill status={worst.eps >= target ? 'critical' : 'warning'} size="sm">{worst.eps >= target ? 'Budget spent' : 'Near budget'}</StatusPill>}
        </div>
      )}
      <LineChart series={series} height={210} yDomain={[0, target * 1.15]} thresholds={[{ y: target, label: `target ε ${target}`, color: 'var(--critical)' }]}
        xFormat={(x) => `r${Math.round(x)}`} yFormat={(y) => y.toFixed(1)} legend={false} tipTitle={(x) => `Round ${x.toFixed(2)}`} />
      <p className="mt-2 text-micro text-ink-3">Operators with less data sample a larger share per step (higher q), so they spend budget faster. An operator that reaches its target leaves the federation automatically.</p>
    </Panel>
  )
}

// ------------------------------------------------------------------ DP-SGD internals

export function DpInternals() {
  const f = useFed()
  const gh = useLive((s) => s.live.gradhist)
  const ops: Operator[] = f.operators ?? []
  const C = f.config?.clip_norm ?? 1
  const sigma = f.config?.noise_multiplier ?? 1.1
  const rows = ops.map((o) => {
    const live = gh?.hists?.[o.id]
    const counts: number[] = live?.counts ?? o.grad_hist ?? []
    return { id: o.id, label: o.short, color: opColor(o.id), counts: counts.length ? counts : new Array(40).fill(0),
      note: `${((live?.clip_frac ?? o.clip_frac) * 100).toFixed(1)}% clipped` }
  })
  const qs = ops.map((o) => o.q)
  return (
    <Panel id="dp" plane="learn" title="DP-SGD internals" subtitle="per-sample gradient L2 norms before clipping, latest step" source={{ aws: 'Opacus per-sample grads', mock: 'Simulated distributions' }}>
      <div className="grid grid-cols-[1fr_180px] gap-4">
        <Ridgeline rows={rows} range={[0, 3 * C]} marker={C} markerLabel={`clip C = ${C}`} />
        <dl className="space-y-2.5 border-l border-line pl-4 text-cap">
          {[
            ['Clip norm C', C.toFixed(2)],
            ['Noise multiplier σ', sigma.toFixed(2)],
            ['Noise std σ·C', (sigma * C).toFixed(2)],
            ['Batch size', String(f.config?.batch_size ?? 64)],
            ['Sampling rate q', qs.length ? `${Math.min(...qs).toFixed(4)}–${Math.max(...qs).toFixed(4)}` : '—'],
            ['Steps per round', String(f.steps_per_round ?? 240)],
            ['Learning rate', (f.config?.lr ?? 0).toExponential(0)],
            ['LoRA', `r${f.config?.lora_rank}, α${f.config?.lora_alpha}`],
          ].map(([k, v]) => (
            <div key={k}><dt className="text-micro text-ink-3">{k}</dt><dd className="tnum font-medium text-ink-1">{v}</dd></div>
          ))}
        </dl>
      </div>
      <p className="mt-2 text-micro text-ink-3">Each sample’s gradient is clipped to norm C, then Gaussian noise with std σ·C is added to the batch sum. The hatched area is the share of samples being clipped.</p>
    </Panel>
  )
}

// ------------------------------------------------------------------ update norms (poison detection)

export function UpdateNorms() {
  const f = useFed()
  const rounds: FlRound[] = (f.rounds_log ?? []).slice(-10)
  const ops: Operator[] = f.operators ?? []
  const normal = rounds.flatMap((r) => Object.entries(r.update_norms).filter(([op]) => r.decisions[op] !== 'rejected' && r.update_norms[op] < r.threshold * 1.01).map(([, v]) => v))
  const top = Math.max(1.5, ...normal) * 1.35
  const dots: Dot[] = []
  let flagged: { op: string; round: number; ratio: number; decision: string } | null = null
  for (const r of rounds) {
    for (const [op, v] of Object.entries(r.update_norms)) {
      const bad = v > r.threshold
      const o = ops.find((x) => x.id === op)
      if (bad) flagged = { op, round: r.round, ratio: v / r.median_norm, decision: r.decisions[op] }
      dots.push({
        id: `${r.round}-${op}`, x: r.round, y: bad ? top * 0.96 : v, color: bad ? 'var(--critical)' : opColor(op),
        shape: bad ? 'cross' : 'circle', ring: bad, label: bad ? `${o?.short} r${r.round}: ${(v / r.median_norm).toFixed(1)}× median, ${r.decisions[op]}` : o?.short,
        showLabel: bad, title: `${o?.short}, round ${r.round}`,
        tip: [{ label: 'update L2 norm', value: v.toFixed(3) }, { label: 'robust threshold', value: r.threshold.toFixed(3) }, { label: 'decision', value: r.decisions[op] }],
      })
    }
  }
  const lastThr = rounds[rounds.length - 1]?.threshold
  return (
    <Panel id="update-norms" plane="learn" title="Client update norms" subtitle="robust aggregation rejects outliers inside the enclave" source={{ aws: 'Nitro Enclave aggregator', mock: 'Simulated aggregator' }}
      actions={<span className="rounded border border-line px-1.5 py-0.5 text-micro text-ink-2">{(f.config?.strategy ?? '').replace('_', ' ')}, {f.config?.secagg_mode}</span>}>
      <Scatter dots={dots} height={220} xFormat={(x) => `r${Math.round(x)}`} yFormat={(y) => y.toFixed(2)} yDomain={[0, top]}
        hlines={lastThr ? [{ y: lastThr, label: `threshold median + 4·MAD = ${lastThr.toFixed(2)}`, color: 'var(--warn)' }] : []} xLabel="round" yLabel="L2 norm" />
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-micro text-ink-3">
        {ops.map((o) => <span key={o.id} className="flex items-center gap-1"><span className="size-2 rounded-full" style={{ background: opColor(o.id) }} />{o.short}</span>)}
        <span className="flex items-center gap-1 text-critical-ink">× outlier (plotted at the top edge)</span>
      </div>
      {flagged && (
        <div className="mt-2 rounded-md border border-critical/40 bg-critical/10 px-3 py-2 text-cap text-ink-1">
          {ops.find((o) => o.id === flagged!.op)?.short} round {flagged.round}: update norm {flagged.ratio.toFixed(1)}× the median.{' '}
          {flagged.decision === 'rejected' ? 'Rejected by robust aggregation inside the attested enclave; the global model never saw it.'
            : 'In pure masking mode the server is blind to individual updates, so it was aggregated. Switch to enclave mode for robust aggregation.'}
        </div>
      )}
    </Panel>
  )
}

export function useOpsById() {
  const f = useFed()
  return useMemo(() => Object.fromEntries(((f.operators ?? []) as Operator[]).map((o) => [o.id, o])), [f.operators])
}
