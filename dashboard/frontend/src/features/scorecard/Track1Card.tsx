import clsx from 'clsx'
import { Link } from 'react-router-dom'
import type { Rec } from '@/core/types'
import { Panel } from '@/ui'
import { num } from './format'
import { useOverview } from './overview'
import { RunLink, Verdict, type Overview } from './RealRuns'
import { T1_CHECKLIST } from './track1Checklist'

// The Track 1 scorecard in the participant guide's shape: areas x models, the bar, and the RL checklist.
// Every value is the model's own recorded score from its result (GET /api/realrun/overview); nothing is computed
// except the difference to base, which uses the paired gain and 95% CI recorded in the run when there is one.

type Cell = NonNullable<Overview['t1']['units'][number]['cells'][string]>

const AREAS: { unit: string; label: string; scale: string }[] = [
  { unit: 't1.knowledge', label: 'Knowledge', scale: 'pass rate, 0–100%' },
  { unit: 't1.execution', label: 'Execution', scale: 'mean score, −1 to 1 (can go negative; not a percent); pass rate beside it' },
  { unit: 't1.interface', label: 'Interface', scale: 'pass rate, 0–100%' },
  { unit: 't1.safety', label: 'Safety & trustworthiness', scale: 'pass rate, 0–100%' },
  { unit: 't1.price_latency', label: 'Price & latency', scale: 'p95 latency per call; cost per 1,000 calls modelled from AWS list prices, not a bill' },
]

const pct = (v?: number | null) => (typeof v === 'number' && Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : '—')
const signed = (v: number, f: (x: number) => string) => `${v >= 0 ? '+' : '−'}${f(Math.abs(v))}`
const pp = (x: number) => `${(x * 100).toFixed(1)} pts`
const sc = (x: number) => x.toFixed(2)

function own(unit: string, c: Cell): string {
  if (unit === 't1.execution') return c.mean_score == null ? '—' : c.mean_score.toFixed(2)
  if (unit === 't1.price_latency') return c.p95_ms == null ? '—' : `${Math.round(c.p95_ms)} ms`
  return pct(c.pass_rate)
}

function Delta({ unit, c, base }: { unit: string; c: Cell; base: Cell | null }) {
  if (unit === 't1.price_latency' || !base) return null
  const g = c.ci as Rec | null
  const f = unit === 't1.execution' ? sc : pp
  if (g && typeof g.value === 'number') {
    return <div className="tnum text-micro text-ink-2">Δ vs base {signed(g.value, f)} <span className="text-ink-3">[{signed(g.ci_low, f)}, {signed(g.ci_high, f)}] 95% CI</span></div>
  }
  return null
}

function Misses({ c }: { c: Cell }) {
  const m = (c.secondary ?? []).filter((s: Rec) => s.verdict === 'FAIL' || s.passed === false)
  if (!m.length) return null
  return <div className="text-micro text-warn-ink">misses: {m.map((s: Rec) => `${s.name}${typeof s.measured === 'number' ? ` (${num(s.measured)})` : ''}`).join('; ')}</div>
}

function T1Value({ unit, c, base }: { unit: string; c: Cell; base: Cell | null }) {
  return (
    <div className="space-y-0.5" data-run={c.run_id} data-state={c.state}>
      <div className="flex flex-wrap items-baseline gap-x-2">
        <Link to={`/evidence?run=${encodeURIComponent(c.run_id)}`} data-run-link={c.run_id} data-testid="t1sc-value"
          title={`${c.run_id}: open its evidence chain`}
          className={clsx('tnum rounded-sm text-[15px] font-semibold underline decoration-ink-3 underline-offset-2 hover:decoration-current focus-visible:outline-2 focus-visible:outline-[color:var(--focus)]',
            c.state === 'fail' ? 'text-critical-ink' : 'text-ink-1')}>{own(unit, c)}</Link>
        <Verdict s={c.state} why={c.why} />
      </div>
      {unit === 't1.execution' && <div className="tnum text-micro text-ink-2">pass rate {pct(c.pass_rate)}</div>}
      {unit === 't1.price_latency' && <div className="tnum text-micro text-ink-2" title={`${c.cost_method ?? ''}\n${c.price_source ?? ''}`}>{c.cost_per_1k_usd != null && c.price_source ? `$${c.cost_per_1k_usd.toFixed(3)} / 1k calls` : 'no recorded price'}</div>}
      <Delta unit={unit} c={c} base={base} />
      <Misses c={c} />
    </div>
  )
}

function BarLine({ t1 }: { t1: Overview['t1'] }) {
  const row = (v: string) => t1.bar.find((r) => r.variant === v) as Rec | undefined
  const b = row('base'), t = row('tuned')
  const be = b?.['t1.execution'], te = t?.['t1.execution'], bi = b?.['t1.interface'], ti = t?.['t1.interface']
  if (!be || !te || !bi || !ti) return <div className="text-cap text-ink-3">The bar needs base and tuned runs for execution and interface; not all are recorded yet.</div>
  return (
    <div className="space-y-1 text-cap text-ink-1" data-testid="t1sc-bar">
      <div>
        <span className="font-medium">Execution pass rate:</span> base {pct(be.pass_rate)} → tuned <span className="font-semibold">{pct(te.pass_rate)}</span>{' '}
        <span className="text-ink-2">(mean score {sc(be.mean_score)} → {sc(te.mean_score)}; the guide's "about 30% today" is a pass rate)</span>
      </div>
      <div>
        <span className="font-medium">Interface pass rate:</span> base {pct(bi.pass_rate)} → tuned <span className="font-semibold">{pct(ti.pass_rate)}</span>{' '}
        <span className={ti.pass_rate > 0.9 ? 'text-good-ink' : 'text-critical-ink'}>(bar: above 90%: {ti.pass_rate > 0.9 ? 'kept' : 'missed'})</span>
      </div>
    </div>
  )
}

const CHIP: Record<string, string> = {
  'recorded': 'border-good-ink text-good-ink',
  'recorded, differs': 'border-warn-ink text-warn-ink',
  'log only': 'border-warn-ink text-warn-ink',
  'not recorded': 'border-line text-ink-3',
}

export default function Track1Card() {
  const { ov } = useOverview()
  const t1 = ov?.t1
  if (!t1 || !t1.units.length) return null
  const cols = t1.columns.filter((c) => c.seen)
  const byUnit = new Map(t1.units.map((u) => [u.unit, u.cells]))
  return (
    <Panel id="t1-scorecard" plane="verify" className="mb-4" title="Track 1 scorecard"
      subtitle="the participant guide's shape · each model measured on the same items, prompt and parser · every value opens its evidence chain">
      <div className="max-w-full overflow-x-auto rounded-md border border-line" data-testid="t1sc-table">
        <table className="w-full min-w-[720px] text-cap">
          <thead>
            <tr className="border-b border-line text-left text-micro text-ink-3">
              <th className="px-3 py-2 font-medium">Area · scale</th>
              {cols.map((c) => <th key={c.variant} className="px-3 py-2 font-medium">{c.label.replace('openai.', '').replace('qwen.', '').replace(/-v?1:0$/, '')}</th>)}
            </tr>
          </thead>
          <tbody>
            {AREAS.map((a) => {
              const cells = byUnit.get(a.unit) ?? {}
              const base = cells.base ?? null
              return (
                <tr key={a.unit} className="border-b border-line/60 align-top last:border-b-0">
                  <td className="px-3 py-2"><div className="font-medium text-ink-1">{a.label}</div><div className="max-w-[180px] text-micro text-ink-3">{a.scale}</div></td>
                  {cols.map((c) => {
                    const cell = cells[c.variant]
                    return <td key={c.variant} className="px-3 py-2">{cell ? <T1Value unit={a.unit} c={cell} base={c.variant === 'base' ? null : base} /> : <span className="text-micro text-ink-3">not run yet</span>}</td>
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-2 text-micro text-ink-2" data-testid="t1sc-same"
        title="Checked in the run declarations: test_set_sha256, prompt_sha256, parser and grader are equal across the four models in each row. The harness commit differs: base runs for execution, interface, safety and price ran at 4c41542, the tuned runs at c4b4b5e (knowledge: both c4b4b5e).">
        Same test items, prompt and parser for all four models in every row (test-set and prompt hashes are equal in the run declarations).
        PASS/MISS is each run's own declared target; tuned rows are declared as a gain over base, so their target is the gain's lower bound.
      </div>
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <div>
          <div className="mb-1 text-micro font-medium tracking-[0.04em] text-ink-3 uppercase">The bar: raise execution, keep interface high</div>
          <BarLine t1={t1} />
        </div>
        <div>
          <div className="mb-1 text-micro font-medium tracking-[0.04em] text-ink-3 uppercase">RL checklist</div>
          <ul className="space-y-1.5" data-testid="t1sc-checklist">
            {T1_CHECKLIST.map((i) => (
              <li key={i.id} data-check={i.id} data-state={i.state} className="text-cap">
                <div className="flex flex-wrap items-baseline gap-x-2">
                  <span className={clsx('rounded border px-1.5 text-micro whitespace-nowrap', CHIP[i.state])}>{i.state}</span>
                  <span className="text-ink-1">{i.label}</span>
                </div>
                <div className="text-micro text-ink-3 [overflow-wrap:anywhere]">
                  {i.evidence}{' '}
                  {i.run && <RunLink id={i.run} />}
                  {i.ref && <span className="mono text-ink-2">{i.ref}</span>}
                </div>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </Panel>
  )
}
