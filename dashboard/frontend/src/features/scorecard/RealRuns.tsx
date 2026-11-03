import clsx from 'clsx'
import { CircleCheck, CircleDashed, CircleX, ExternalLink, TriangleAlert } from 'lucide-react'
import { Link } from 'react-router-dom'
import type { Rec } from '@/core/types'
import { Panel } from '@/ui'
import { num, targetText, utc, value } from './format'
import { shortReason, useOverview } from './overview'

// Real runs, read from the local byte-exact mirror of the S3 evidence prefix (make evidence-sync) through
// GET /api/realrun/overview. Every number links to its run id in the Evidence explorer. A cell with no run
// says "not run yet"; nothing is filled in.

export interface Job {
  name: string; status?: string; instance_type?: string; creation_time?: string; end_time?: string
  billable_s?: number | null; cost_usd?: number | null; cost_basis?: string | null; failure_reason?: string | null; time_source?: string
}
interface Cell {
  run_id: string; model_id: string; state: string; value: number | null; metric: string; target: Rec
  pass_rate: number | null; mean_score: number | null; n_items: number | null
  p50_ms: number | null; p95_ms: number | null; cost_per_1k_usd: number | null; instance_type?: string; price_source?: string
  cost_method?: string; ci: Rec | null; why?: string[]; checked_at?: string; declared_at?: string; declared_s3_time?: string; result_s3_time?: string
  jobs: Job[]; other_runs: string[]; secondary: Rec[]
  superseded_runs?: { run_id: string; reason: string; by: string[]; value: number | null }[]
}
export interface Overview {
  source: Rec
  runs: Rec[]
  t1: { columns: { variant: string; label: string; seen: boolean }[]; units: { unit: string; cells: Record<string, Cell | null> }[]; bar: Rec[] }
  rl: Rec | null
  t3: Rec[]
  t3_live: Rec[]
  t2: Rec[]
  t2_consortium: Rec
  judged: Rec | null
  empty: boolean
}

export const RunLink = ({ id, className }: { id: string; className?: string }) => (
  <Link to={`/evidence?run=${encodeURIComponent(id)}`} className={clsx('mono inline-flex items-center gap-1 text-micro text-ink-3 underline decoration-dotted underline-offset-2 hover:text-ink-1', className)}
    title="Open this run in the Evidence explorer: declaration, job, records, result, check_unit">
    {id}<ExternalLink size={10} aria-hidden />
  </Link>
)

const pct = (v?: number | null) => (typeof v === 'number' && Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : '—')
const usd = (v?: number | null, d = 3) => (typeof v === 'number' && Number.isFinite(v) ? `$${v.toFixed(d)}` : '—')
const secs = (v?: number | null) => (typeof v === 'number' ? `${Math.round(v)} s` : '—')

export const JobLine = ({ j }: { j: Job }) => (
  <div className="text-micro text-ink-3" title={`${j.time_source ?? ''}${j.cost_basis ? `\ncost: ${j.cost_basis}` : ''}`}>
    <span className="mono">{j.instance_type ?? '?'}</span> · {j.status ?? '?'} · AWS {utc(j.creation_time)} → {j.end_time ? utc(j.end_time).slice(11) : '…'} · billable {secs(j.billable_s)} · {usd(j.cost_usd)}
  </div>
)

const STATE: Record<string, { Icon: typeof CircleCheck; color: string; word: string }> = {
  pass: { Icon: CircleCheck, color: 'var(--good-ink)', word: 'PASS' },
  fail: { Icon: CircleX, color: 'var(--critical-ink)', word: 'MISS' },
  'not checked': { Icon: CircleDashed, color: 'var(--ink-3)', word: 'NO RESULT YET' },
  inconsistent: { Icon: TriangleAlert, color: 'var(--warn-ink)', word: 'INCONSISTENT' },
  incomplete: { Icon: TriangleAlert, color: 'var(--warn-ink)', word: 'INCOMPLETE' },
  'check failed': { Icon: CircleX, color: 'var(--critical-ink)', word: 'FAIL (check_unit)' },
  unverified: { Icon: TriangleAlert, color: 'var(--warn-ink)', word: 'UNVERIFIED' },
}
/** A status. `why` (from the server) explains an override: complete:false or a failed check_unit part. */
export const Verdict = ({ s, why }: { s: string; why?: string[] }) => {
  const x = STATE[s] ?? STATE.inconsistent
  return (
    <span className="inline-flex flex-col">
      <span className="inline-flex items-center gap-1 text-micro font-semibold tracking-[0.04em]" style={{ color: x.color }} data-verdict={s}><x.Icon size={13} aria-hidden />{x.word}</span>
      {(why ?? []).map((w) => <span key={w} className="max-w-[46ch] text-micro font-normal break-words" style={{ color: x.color }}>{w}</span>)}
    </span>
  )
}

const NotRun = ({ what = 'not run yet' }: { what?: string }) => (
  <div className="flex h-full min-h-16 items-center justify-center rounded-md border border-dashed border-line px-2 text-micro text-ink-3">{what}</div>
)

export const Superseded = ({ reason, by }: { reason: string; by?: string[] }) => (
  <span className="text-micro text-warn-ink" title={reason}>superseded: {shortReason(reason)}{by?.length ? ` (by ${by.join(', ')})` : ''}</span>
)

export default function RealRuns() {
  const { ov, err } = useOverview()
  if (err && !ov) return <div role="alert" className="mt-4 rounded-md border border-line bg-s1 p-3 text-cap text-ink-2">Real-run views unavailable: <span className="mono">{err}</span></div>
  if (!ov) return <div className="mt-4 text-cap text-ink-3">Loading the real runs…</div>
  if (!ov.source?.has_mirror) {
    return <div className="mt-4 rounded-md border border-line bg-s1 p-3 text-cap text-ink-2" data-testid="no-mirror">No local evidence mirror yet. Run <span className="mono">make evidence-sync</span>, then start the console with <span className="mono">make demo-real</span>.</div>
  }
  return (
    <div className="mt-4 space-y-4" data-testid="real-runs">
      <div className="text-micro text-ink-3">
        Read from the byte-exact mirror of the S3 evidence prefix, synced {utc(ov.source.synced_at)}. Times marked AWS come from SageMaker (DescribeTrainingJob) or S3 LastModified; declared/checked times are what our code wrote.
      </div>
      <Track1 t1={ov.t1} />
      <div className="grid grid-cols-12 items-start gap-4">
        <div className="col-span-12 xl:col-span-5"><Bar t1={ov.t1} /></div>
        <div className="col-span-12 xl:col-span-7">{ov.rl ? <RL rl={ov.rl} /> : <Panel title="Track 1 RL gate"><NotRun /></Panel>}</div>
      </div>
      <Track3Live rows={ov.t3_live ?? []} />
      <Track3 rows={ov.t3} />
      {ov.t2.length > 0 && <Track2 rows={ov.t2} />}
    </div>
  )
}

// ------------------------------------------------------------------ Track 1 matrix

function Track1({ t1 }: { t1: Overview['t1'] }) {
  return (
    <Panel id="realrun-t1" plane="verify" title="Track 1: base vs tuned vs Bedrock" subtitle="latest run per unit and model · target declared before the run · PASS/MISS recomputed here" flush>
      <div className="divide-y divide-line md:hidden" data-testid="t1-cards">
        {t1.units.map((u) => (
          <div key={u.unit} className="space-y-2 px-3 py-3">
            <div className="text-cap font-medium text-ink-1">{u.unit.replace('t1.', '')}</div>
            {t1.columns.map((c) => (
              <div key={c.variant}>
                <div className="mb-0.5 text-micro text-ink-3">{c.label}</div>
                {u.cells[c.variant] ? <T1Cell c={u.cells[c.variant]!} /> : <NotRun />}
              </div>
            ))}
          </div>
        ))}
      </div>
      <div className="hidden overflow-x-auto md:block">
        <table className="w-full min-w-[760px] text-cap" data-testid="t1-matrix">
          <thead>
            <tr className="border-b border-line text-left text-micro text-ink-3">
              <th className="px-3 py-2 font-medium">Unit</th>
              {t1.columns.map((c) => <th key={c.variant} className="px-3 py-2 font-medium">{c.label}{!c.seen && <span className="ml-1 text-ink-3">(no runs)</span>}</th>)}
            </tr>
          </thead>
          <tbody>
            {t1.units.map((u) => (
              <tr key={u.unit} className="border-b border-line/60 align-top">
                <td className="px-3 py-2 font-medium whitespace-nowrap text-ink-1">{u.unit.replace('t1.', '')}</td>
                {t1.columns.map((c) => <td key={c.variant} className="w-[30%] px-3 py-2">{u.cells[c.variant] ? <T1Cell c={u.cells[c.variant]!} /> : <NotRun />}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}

function T1Cell({ c }: { c: Cell }) {
  const unit = c.target?.unit
  return (
    <div className="space-y-0.5" data-run={c.run_id} data-state={c.state}>
      <div className="flex flex-wrap items-baseline gap-x-2">
        <Verdict s={c.state} why={c.why} />
        <span className={clsx('tnum text-[15px] font-semibold', c.state === 'fail' ? 'text-critical-ink' : 'text-ink-1')}>{c.state === 'not checked' ? '—' : value(c.value, unit)}</span>
        <span className="tnum text-micro text-ink-3">target {targetText(c.target)} · {c.metric}</span>
      </div>
      <div className="tnum text-micro text-ink-2">
        {c.pass_rate != null && c.metric !== 'pass_rate' && <>pass rate {pct(c.pass_rate)} · </>}
        n = {c.n_items ?? '—'} · p95 {num(c.p95_ms)} ms · {usd(c.cost_per_1k_usd)}/1k calls
      </div>
      <div className="text-micro text-ink-3">{c.ci ? <>95% CI [{num(c.ci.ci_low)}, {num(c.ci.ci_high)}]{c.ci.method ? ` · ${c.ci.method}` : ''}</> : 'no CI recorded for this unit'}</div>
      {c.price_source && <div className="truncate text-micro text-ink-3" title={`${c.cost_method ?? ''}\n${c.price_source}`}>price: {c.instance_type} on-demand × latency (AWS Price List), not a bill</div>}
      {c.jobs.map((j) => <JobLine key={j.name} j={j} />)}
      <RunLink id={c.run_id} />
      {c.other_runs.length > 0 && <div className="text-micro text-ink-3">also: {c.other_runs.map((r) => <RunLink key={r} id={r} className="mr-2" />)}</div>}
      {(c.superseded_runs ?? []).map((x) => (
        <div key={x.run_id} className="mt-1 rounded border border-dashed border-line px-2 py-1 opacity-60" data-superseded={x.run_id}>
          <div className="tnum text-micro text-ink-2 line-through">{value(x.value, unit)}</div>
          <Superseded reason={x.reason} /> <RunLink id={x.run_id} />
        </div>
      ))}
    </div>
  )
}

// ------------------------------------------------------------------ the bar: interface vs execution

function Bar({ t1 }: { t1: Overview['t1'] }) {
  const rows = t1.bar
  const B = ({ v, tone }: { v: number | null | undefined; tone: string }) => (
    <div className="flex items-center gap-2">
      <div className="h-3 flex-1 rounded-sm bg-s2"><div className="h-3 rounded-sm" style={{ width: `${Math.max(0, Math.min(1, v ?? 0)) * 100}%`, background: tone }} /></div>
      <span className="tnum w-14 text-right text-micro text-ink-1">{pct(v)}</span>
    </div>
  )
  return (
    <Panel id="realrun-bar" plane="verify" title="The Track 1 bar" subtitle="interface (format) vs execution (does the fix work on the twin) · pass rate per item">
      <div className="space-y-3" data-testid="t1-bar">
        {rows.map((r) => {
          const i = r['t1.interface'], e = r['t1.execution']
          return (
            <div key={r.variant}>
              <div className="mb-1 text-cap font-medium text-ink-1">{r.label}</div>
              {!i && !e ? <NotRun /> : (
                <div className="space-y-1">
                  <div className="grid grid-cols-[72px_1fr] items-center gap-2 text-micro text-ink-3"><span>interface</span>{i ? <B v={i.pass_rate} tone="var(--ink-3)" /> : <span>not run yet</span>}</div>
                  <div className="grid grid-cols-[72px_1fr] items-center gap-2 text-micro text-ink-3"><span>execution</span>{e ? <B v={e.pass_rate} tone="var(--critical-ink)" /> : <span>not run yet</span>}</div>
                  <div className="flex flex-wrap gap-x-3">{i && <RunLink id={i.run_id} />}{e && <RunLink id={e.run_id} />}</div>
                  {e && <div className="text-micro text-ink-3">execution declared metric: mean score {num(e.mean_score)} (n = {e.n_items})</div>}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ RL gate

function RL({ rl }: { rl: Rec }) {
  const tr: Rec | null = rl.training
  const m: Rec | null = tr?.model
  const tj: Job | undefined = tr?.jobs?.[0]
  return (
    <Panel id="realrun-rl" plane="verify" title="Track 1 RL: the SDK gate" subtitle={`${(rl.items ?? []).filter((i: Rec) => i.passed).length} of ${(rl.items ?? []).length} items pass · run passes only if all pass`}>
      <div className="mb-2 flex flex-wrap items-baseline gap-2"><Verdict s={rl.result?.recomputed ?? 'inconsistent'} why={rl.result?.why} /><span className="tnum text-cap text-ink-1">{num(rl.result?.value)} {rl.result?.metric}</span><span className="text-micro text-ink-3">target {targetText(rl.result?.target)}</span><RunLink id={rl.run_id} /></div>
      <ul className="grid gap-1.5 sm:grid-cols-2" data-testid="rl-items">
        {(rl.items ?? []).map((i: Rec) => (
          <li key={i.id} className={clsx('rounded-md border px-2.5 py-1.5', i.passed ? 'border-line' : 'border-critical/60 bg-critical/10')}>
            <div className="flex items-center gap-1.5 text-cap font-medium text-ink-1">{i.passed ? <CircleCheck size={14} style={{ color: 'var(--good-ink)' }} /> : <CircleX size={14} style={{ color: 'var(--critical-ink)' }} />}{i.id}</div>
            {i.measured?.text && <div className="tnum text-micro text-ink-2">{i.measured.text}</div>}
            {i.rule && <div className="text-micro text-ink-3">{i.rule}</div>}
          </li>
        ))}
      </ul>
      <div className="mt-3 rounded-md border border-line bg-s2/60 p-2.5 text-micro text-ink-2" data-testid="rl-lineage">
        <div className="mb-1 text-cap font-medium text-ink-1">Lineage</div>
        {tr ? <>
          <div>Training run <RunLink id={tr.run_id} /> ({tr.model_id}), declared {utc(tr.declared_at)}{m?.declared_timesteps ? `, ${Number(m.declared_timesteps).toLocaleString()} steps declared` : ''}.</div>
          {tj && <div>Job {tj.status}{tj.failure_reason ? `: ${tj.failure_reason}` : ''}. <JobLine j={tj} /></div>}
          {m?.last_logged && <div>Last logged step before the failure: <span className="tnum font-medium text-ink-1">{Number(m.last_logged.value).toLocaleString()}</span> timesteps (SageMaker job log, {utc(m.last_logged.timestamp)}).</div>}
          {m && <div>Evaluated checkpoint: <span className="tnum font-medium text-ink-1">{Number(m.steps ?? rl.checkpoint_steps).toLocaleString()}</span> steps, sha256 <span className="mono">{String(m.artifact_sha256 ?? '').slice(0, 12)}…</span> (S3 time {utc(m.aws_time)}).</div>}
          {rl.release && <div className="text-ink-3">{rl.release}</div>}
        </> : <div>No training run linked.</div>}
        {(rl.reward_check ?? []).map((r: Rec) => <div key={r.run_id}>Reward check <RunLink id={r.run_id} />: {r.result ? `${num(r.result.value)} ${r.result.metric}, ${r.result.recomputed ?? '?'}` : 'no result'}</div>)}
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ Track 3 / Track 2

function CIBar({ ci, lo = -0.2, hi = 1 }: { ci: Rec | null; lo?: number; hi?: number }) {
  if (!ci) return <span className="text-micro text-ink-3">no interval (this is the baseline)</span>
  const x = (v: number) => `${((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * 100}%`
  return (
    <div className="flex items-center gap-2">
      <div className="relative h-3 w-40 shrink-0 rounded-sm bg-s2" aria-hidden>
        <div className="absolute top-0 h-3 w-px bg-ink-3" style={{ left: x(0) }} />
        <div className="absolute top-1 h-1 rounded-sm" style={{ left: x(ci.ci_low), width: `calc(${x(ci.ci_high)} - ${x(ci.ci_low)})`, background: 'var(--good-ink)' }} />
        <div className="absolute top-0 h-3 w-0.5" style={{ left: x(ci.value), background: 'var(--ink-1)' }} />
      </div>
      <span className="tnum text-micro text-ink-1">{num(ci.value)} [{num(ci.ci_low)}, {num(ci.ci_high)}]</span>
    </div>
  )
}

function Track3({ rows }: { rows: Rec[] }) {
  return (
    <Panel id="realrun-t3" plane="verify" title="Track 3 baselines on the twin" subtitle="mean recovery per case · gain over no-action with a paired-bootstrap 95% interval" flush>
      {rows.length === 0 ? <div className="p-4"><NotRun /></div> : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-cap" data-testid="t3-table">
            <thead><tr className="border-b border-line text-left text-micro text-ink-3">{['Policy', 'Verdict', 'Mean recovery', 'Success', 'Gain vs no-action (95% CI)', 'n', 'AWS job'].map((h) => <th key={h} className="px-3 py-2 font-medium whitespace-nowrap">{h}</th>)}</tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.run_id} className={clsx('border-b border-line/60 align-top', !r.current && 'opacity-60')}>
                  <td className="px-3 py-2"><div className="font-medium text-ink-1">{r.model_id}{r.mode === 'masked' || /masked/.test(r.run_id) ? ' (masked)' : ''}</div><RunLink id={r.run_id} />{r.superseded ? <div><Superseded reason={r.superseded.reason} /></div> : !r.current && <div className="text-micro text-ink-3">earlier run of the same policy</div>}</td>
                  <td className="px-3 py-2"><Verdict s={r.recomputed ?? 'inconsistent'} why={r.why} /><div className="tnum text-micro text-ink-3">{r.metric} {targetText(r.target)}</div></td>
                  <td className="tnum px-3 py-2 text-ink-1">{num(r.mean_recovery)}</td>
                  <td className="tnum px-3 py-2 text-ink-2">{pct(r.success_rate)}</td>
                  <td className="px-3 py-2"><CIBar ci={r.ci} /></td>
                  <td className="tnum px-3 py-2 text-ink-2">{r.n_items}</td>
                  <td className="px-3 py-2">{(r.jobs ?? []).map((j: Job) => <JobLine key={j.name} j={j} />)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  )
}

function Track3Live({ rows }: { rows: Rec[] }) {
  return (
    <Panel id="realrun-t3-live" plane="operate" title="Track 3 live units" subtitle="through the deployed checkpoint and Gateway · every t3 unit that is not an offline eval" flush>
      {rows.length === 0 ? <div className="p-4"><NotRun /></div> : (
        <div className="divide-y divide-line" data-testid="t3-live">
          {rows.map((r) => (
            <div key={r.run_id} className={clsx('grid gap-2 px-4 py-3 md:grid-cols-[220px_1fr]', r.superseded && 'opacity-60')}>
              <div>
                <div className="font-medium text-ink-1">{r.unit}</div>
                <Verdict s={r.state} why={r.why} /> <span className="tnum text-cap text-ink-1">{num(r.value)} {r.metric}</span>
                <div className="tnum text-micro text-ink-3">target {targetText(r.target)} · n = {r.n_items ?? '—'}{r.n_declared != null && r.n_declared !== r.n_items ? ` of ${r.n_declared} declared` : ''}</div>
                <RunLink id={r.run_id} />
                {r.superseded && <div><Superseded reason={r.superseded.reason} /></div>}
              </div>
              <div className="min-w-0 text-micro text-ink-2">
                <div className="text-ink-3">{r.model_id} · checked {utc(r.checked_at)}{r.synthetic ? ' · SYNTHETIC' : ''}</div>
                <ul className="mt-1 space-y-0.5">{(r.facts ?? []).map((f: string) => <li key={f} className="mono break-all">{f}</li>)}</ul>
                {r.ci && <div className="mt-1"><CIBar ci={r.ci} /></div>}
                {(r.jobs ?? []).map((j: Job) => <JobLine key={j.name} j={j} />)}
              </div>
            </div>
          ))}
        </div>
      )}
    </Panel>
  )
}

/** "t2.local_only.B.disjoint" -> "local-only (silo B)"; read from the unit name, never a guess at the model. */
function t2Method(r: Rec): string {
  const p = String(r.unit ?? '').split('.')
  const k = p[1] ?? ''
  if (k === 'local_only') return `local-only (silo ${p[2] ?? '?'})`
  if (k === 'central') return 'central (pooled reference)'
  if (k === 'fedavg_local') return 'local FedAvg (our simulation)'
  if (k.startsWith('fl')) return 'federated (FL service)'
  if (k === 'compare') return `comparison: ${p.slice(2, -1).join('.')}`
  if (k === 'silo_data') return 'silo data check'
  return String(r.method ?? r.unit ?? '—')
}

function Track2({ rows }: { rows: Rec[] }) {
  return (
    <Panel id="realrun-t2" plane="learn" title="Track 2 baselines (simulated silos)" subtitle="per-silo, pooled and federated models on the shared test of simulated silos · score = scaled MAE, lower is better" flush>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[720px] text-cap" data-testid="t2-table">
          <thead><tr className="border-b border-line text-left text-micro text-ink-3">{['Run', 'Method', 'Mean score', 'No-op score', 'n', 'AWS job'].map((h) => <th key={h} className="px-3 py-2 font-medium whitespace-nowrap">{h}</th>)}</tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.run_id} className="border-b border-line/60 align-top">
                <td className="px-3 py-2"><RunLink id={r.run_id} /></td>
                <td className="px-3 py-2 text-ink-1">{t2Method(r)}</td>
                <td className="tnum px-3 py-2 text-ink-1">{num(r.mean_score)}</td>
                <td className="tnum px-3 py-2 text-ink-3">{num(r.mean_score_noop)}</td>
                <td className="tnum px-3 py-2 text-ink-2">{r.n_items ?? '—'}</td>
                <td className="px-3 py-2">{(r.jobs ?? []).map((j: Job) => <JobLine key={j.name} j={j} />)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}

