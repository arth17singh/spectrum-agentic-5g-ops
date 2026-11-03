import { LineChart, type Pt, type Series } from '@/charts/LineChart'
import type { Rec } from '@/core/types'
import { Panel } from '@/ui'
import { RunLink, Verdict } from './RealRuns'
import { num } from './format'

// Training curves drawn from the result `details` of two declared units, as recorded (never recomputed):
//   t2.fl_convergence  error per FL round for each silo and their mean, with reference lines (Twin View)
//   t1.curves          SFT train/dev loss per step and RL mean episode reward per timestep (Scorecard)
// Until such a run exists the panels say "not run yet"; a null series says why it is empty.

const SILO: Record<string, string> = { A: 'var(--learn)', B: 'var(--verify)', C: 'var(--operate)', mean: 'var(--ink-1)' }

const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
const pairs = (xs: unknown): Pt[] =>
  Array.isArray(xs) ? xs.filter((p) => Array.isArray(p) && isNum(p[0]) && isNum(p[1])).map((p) => ({ x: p[0] as number, y: p[1] as number })) : []
/** A reference level: a number, {value}, or a per-silo object whose `mean` is compared with the mean curve. */
const refValue = (r: unknown): number | null => {
  if (isNum(r)) return r
  if (r && typeof r === 'object') {
    const o = r as Rec
    if (isNum(o.value)) return o.value
    if (isNum(o.mean)) return o.mean
  }
  return null
}
const kfmt = (x: number) => (Math.abs(x) >= 1000 ? `${Math.round(x / 1000)}k` : String(Math.round(x)))

export const Empty = ({ what }: { what: string }) => (
  <div className="flex min-h-24 items-center justify-center rounded-md border border-dashed border-line px-3 py-4 text-center text-micro text-ink-3">{what}</div>
)

/** Twin View, Consortium: error against FL round. */
export function ConvergenceChart({ c }: { c: Rec | null | undefined }) {
  if (!c) return <Empty what="FL convergence (t2.fl_convergence): not run yet" />
  const d: Rec = c.details ?? {}
  const rounds: number[] = Array.isArray(d.rounds) ? d.rounds : []
  const curve: Rec = d.curve ?? {}
  const series: Series[] = []
  for (const k of ['A', 'B', 'C', 'mean']) {
    const ys: unknown[] = Array.isArray(curve[k]) ? curve[k] : []
    const pts = rounds.map((r, i) => ({ x: r, y: ys[i] })).filter((p): p is Pt => isNum(p.x) && isNum(p.y))
    if (pts.length) series.push({ id: k, label: k === 'mean' ? 'mean of silos' : `silo ${k}`, color: SILO[k], points: pts, width: k === 'mean' ? 2.5 : 1.5 })
  }
  const r0 = rounds.length ? Math.min(...rounds) : 0
  const r1 = rounds.length ? Math.max(...rounds) : 1
  const flat = (id: string, label: string, color: string, v: number | null) =>
    v == null ? null : ({ id, label, color, dashed: true, width: 1, points: [{ x: r0, y: v }, { x: r1, y: v }] } as Series)
  const ref: Rec = d.reference ?? {}
  for (const k of ['A', 'B', 'C']) {
    const s = flat(`lo-${k}`, `local-only ${k}`, SILO[k], refValue((ref.local_only ?? {})[k]))
    if (s) series.push(s)
  }
  for (const [k, label, color] of [['central', 'central (pooled), mean', 'var(--ink-2)'], ['fedavg_local', 'local FedAvg (sim), mean', 'var(--ink-3)']] as const) {
    const s = flat(k, label, color, refValue(ref[k]))
    if (s) series.push(s)
  }
  const fr: Rec | null = d.final_round_check ?? null
  return (
    <div data-testid="fl-convergence">
      <div className="mb-1 flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="text-cap font-medium text-ink-1">Error per FL round</span>
        <Verdict s={c.state} why={c.why} />
        {c.synthetic && <span className="text-micro font-semibold text-warn-ink">SYNTHETIC</span>}
        <RunLink id={c.run_id} />
      </div>
      {series.length ? (
        <LineChart series={series} height={240} xFormat={(x) => String(Math.round(x))} yFormat={(y) => y.toFixed(2)}
          tipTitle={(x) => `round ${Math.round(x)}`} margin={{ l: 44 }} />
      ) : <Empty what="The result records no curve values." />}
      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-micro text-ink-3">
        <span>y: {d.metric ?? 'scaled MAE'} on each silo's own test segment, {d.lower_is_better === false ? 'higher' : 'lower'} is better · x: aggregation round</span>
        <span>dashed: references from separate runs</span>
      </div>
      {d.session && typeof d.session === 'object' && (
        <div className="mt-1 text-micro text-ink-2" data-testid="fl-session">
          FL session {d.session.session_id_short ?? ''}: {d.session.rounds_completed ?? '?'} rounds · {d.session.clients ?? '?'} clients ·
          secure aggregation {d.session.secure_aggregation === true ? 'on' : d.session.secure_aggregation === false ? 'off' : 'not recorded'} · DP {d.session.dp ?? 'not recorded'}
        </div>
      )}
      {fr && (
        <div className="mt-1 text-micro" style={{ color: fr.matches_published ? 'var(--good-ink)' : 'var(--critical-ink)' }} data-testid="final-round-check">
          {fr.matches_published ? '✓' : '✗'} final round {fr.round} {fr.matches_published ? 'matches' : 'does NOT match'} the published model
          {fr.published_run_id ? <> (<RunLink id={fr.published_run_id} />)</> : null}{isNum(fr.max_abs_diff) ? `, max |diff| ${num(fr.max_abs_diff)}` : ''}
        </div>
      )}
    </div>
  )
}

/** Scorecard: SFT loss per step and RL reward per timestep. */
export function TrainingCurves({ c }: { c: Rec | null | undefined }) {
  const d: Rec = c?.details ?? {}
  const sft: Rec | null = c ? d.sft ?? null : null
  const rl: Rec | null = c ? d.rl ?? null : null
  const sftSeries: Series[] = []
  if (sft) {
    const tr = pairs(sft.series?.train_loss), dv = pairs(sft.series?.dev_loss)
    if (tr.length) sftSeries.push({ id: 'train', label: 'train loss', color: 'var(--learn)', points: tr })
    if (dv.length) sftSeries.push({ id: 'dev', label: 'dev loss', color: 'var(--verify)', points: dv, width: 2 })
  }
  const rw = rl ? pairs(rl.series?.mean_episode_reward) : []
  const served = rl ? (/^(\d+(?:\.\d+)?)k$/i.exec(String(rl.served_checkpoint ?? '')) ?? null) : null
  const ann: { x: number; label: string; kind?: string }[] = []
  if (rl && isNum(rl.crashed_at_timesteps)) ann.push({ x: rl.crashed_at_timesteps, label: `training crashed at ${rl.crashed_at_timesteps.toLocaleString()} timesteps` })
  if (served) ann.push({ x: Number(served[1]) * 1000, label: `served checkpoint: ${rl!.served_checkpoint}`, kind: 'checkpoint' })
  const rwX = [...rw.map((p) => p.x), ...ann.map((a) => a.x)]
  const rwDomain: [number, number] | undefined = rwX.length ? [Math.min(...rwX), Math.max(...rwX) * 1.02] : undefined
  const why = (part: Rec | null, name: string) => (!c ? 't1.curves: not run yet' : part == null ? `${name}: not recorded in the result${d[`${name}_reason`] ? ` (${d[`${name}_reason`]})` : ''}` : null)
  return (
    <Panel id="realrun-curves" plane="learn" title="Training curves" subtitle="as recorded by the t1.curves run · SFT loss per step, RL reward per timestep">
      {c && <div className="mb-2 flex flex-wrap items-baseline gap-2"><Verdict s={c.state} why={c.why} />{c.synthetic && <span className="text-micro font-semibold text-warn-ink">SYNTHETIC</span>}<RunLink id={c.run_id} /></div>}
      <div className="grid gap-4 md:grid-cols-2" data-testid="training-curves">
        <div className="min-w-0">
          <div className="mb-1 text-cap font-medium text-ink-1">SFT loss</div>
          {why(sft, 'sft') ? <Empty what={why(sft, 'sft')!} /> : sftSeries.length ? (
            <>
              <LineChart series={sftSeries} height={200} xFormat={kfmt} yFormat={(y) => y.toFixed(2)} tipTitle={(x) => `step ${Math.round(x)}`} />
              <div className="mt-1 text-micro text-ink-3">x: optimizer step · y: loss, lower is better{sft!.run_id ? <> · <RunLink id={sft!.run_id} /></> : null}{sft!.source ? ` · ${sft!.source}` : ''}</div>
            </>
          ) : <Empty what={`SFT: no loss values recorded${sft!.reason ?? Object.values(sft!.series_reasons ?? {})[0] ? ` (${sft!.reason ?? Object.values(sft!.series_reasons ?? {})[0]})` : ''}`} />}
        </div>
        <div className="min-w-0">
          <div className="mb-1 text-cap font-medium text-ink-1">RL mean episode reward</div>
          {why(rl, 'rl') ? <Empty what={why(rl, 'rl')!} /> : rw.length ? (
            <>
              <LineChart series={[{ id: 'reward', label: 'mean episode reward', color: 'var(--operate)', points: rw }]} height={200}
                xFormat={kfmt} yFormat={(y) => y.toFixed(1)} tipTitle={(x) => `${Math.round(x).toLocaleString()} timesteps`} annotations={ann} legend={false}
                xDomain={rwDomain} />
              <div className="mt-1 text-micro text-ink-3">
                x: timesteps · y: reward, higher is better
                {isNum(rl!.crashed_at_timesteps) ? ` · crashed at ${rl!.crashed_at_timesteps.toLocaleString()}` : ''}
                {rl!.served_checkpoint ? ` · ◆ served checkpoint ${rl!.served_checkpoint}` : ''}
                {rl!.train_run_id ? <> · <RunLink id={rl!.train_run_id} /></> : null}{rl!.eval_run_id ? <> · eval <RunLink id={rl!.eval_run_id} /></> : null}
              </div>
            </>
          ) : <Empty what={`RL: no reward values recorded${rl!.reason ?? Object.values(rl!.series_reasons ?? {})[0] ? ` (${rl!.reason ?? Object.values(rl!.series_reasons ?? {})[0]})` : ''}`} />}
        </div>
      </div>
    </Panel>
  )
}
