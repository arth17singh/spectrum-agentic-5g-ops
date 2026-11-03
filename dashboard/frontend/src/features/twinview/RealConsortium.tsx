import { FlaskConical } from 'lucide-react'
import type { Rec } from '@/core/types'
import { Panel } from '@/ui'
import { JobLine, RunLink, type Job } from '../scorecard/RealRuns'
import { ConvergenceChart } from '../scorecard/Curves'
import { num } from '../scorecard/format'
import { sessionFacts, useOverview } from '../scorecard/overview'

// The real Track 2 results (GET /api/realrun/overview -> t2_consortium), read from the t2 result records in the
// evidence mirror. Simulated silos: three simulated operators on the in-process Maveric twin, not partner data.
// Score = mean per-row scaled MAE on the shared test; LOWER is better.

const d = (v?: number | null) => (typeof v === 'number' && Number.isFinite(v) ? `${v > 0 ? '+' : ''}${num(Number(v.toFixed(4)))}` : '—')
const tone = (v?: number | null) => (typeof v !== 'number' ? 'text-ink-3' : v > 0 ? 'text-critical-ink' : 'text-good-ink')

function Fact({ label, f }: { label: string; f: Rec | null }) {
  return (
    <div className="min-w-0">
      <div className="text-micro text-ink-3">{label}</div>
      {f ? <div className="mono truncate text-cap text-ink-1" title={`${String(f.value)} (from the ${f.from} record)`}>{String(f.value)}</div>
        : <div className="text-micro text-ink-3">not in the records yet</div>}
    </div>
  )
}

function Split({ s, note, conv }: { s: Rec; note: string; conv?: Rec | null }) {
  const fed: Rec | undefined = (s.federated ?? []).find((x: Rec) => x.run_id === s.current_federated)
  const f: Rec = { ...(fed?.facts ?? {}) }
  // the convergence run records the session itself; its facts replace "not in the records yet"
  const sess = sessionFacts(conv)
  if (sess) {
    const from = `t2.fl_convergence run ${conv!.run_id}`
    if (sess.rounds_completed != null) f.rounds = { value: sess.rounds_completed, from }
    if (sess.clients != null) f.clients = { value: sess.clients, from }
    if (sess.secure_aggregation != null) f.secure_aggregation = { value: sess.secure_aggregation ? 'yes' : 'no', from }
    if (sess.dp != null) f.epsilon = { value: sess.dp, from }
    if (sess.session_id_short) f.session_id = { value: sess.session_id_short, from }
  }
  const dc: Rec | null = s.distance_to_central
  return (
    <div className="space-y-4" data-split={s.split}>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[560px] text-cap" data-testid="t2-per-silo">
          <thead><tr className="border-b border-line text-left text-micro text-ink-3">
            {['Silo (own test segment)', 'Local-only', 'Federated', 'Federated − local', 'Central (reference)', 'Federated − central'].map((h) => <th key={h} className="px-3 py-2 font-medium whitespace-nowrap">{h}</th>)}
          </tr></thead>
          <tbody>
            {(s.per_silo ?? []).map((r: Rec) => (
              <tr key={r.silo} className="border-b border-line/60">
                <td className="px-3 py-2 font-medium text-ink-1">{r.silo}<div className="font-normal">{r.local_run_id && <RunLink id={r.local_run_id} />}</div></td>
                <td className="tnum px-3 text-ink-1">{num(r.local_only)}</td>
                <td className="tnum px-3 text-ink-1">{num(r.federated)}</td>
                <td className={`tnum px-3 font-medium ${tone(r.federated_minus_local)}`}>{d(r.federated_minus_local)}</td>
                <td className="tnum px-3 text-ink-2">{num(r.central)}</td>
                <td className={`tnum px-3 ${tone(r.federated_minus_central)}`}>{d(r.federated_minus_central)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="text-micro text-ink-3">{note} A positive difference means the federated model is worse on that silo. These are differences of recorded means; paired intervals come from the t2.compare runs below once they land.</div>
      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-md border border-line p-3">
          <div className="mb-2 text-cap font-medium text-ink-1">Distance to central</div>
          {dc ? <>
            <div className={`tnum text-[18px] font-semibold ${tone(dc.value)}`}>{d(dc.value)}</div>
            <div className="tnum text-micro text-ink-3">federated {num(dc.federated)} vs central {num(dc.central)} on the whole shared test</div>
            <div className="text-micro text-ink-3">{dc.how}</div>
          </> : <div className="text-micro text-ink-3">needs a federated run and the central reference</div>}
          {s.central && <div className="mt-1"><RunLink id={s.central.run_id} /></div>}
          {s.fedavg_local && <div className="mt-2 text-micro text-ink-3">Local FedAvg (our simulation, not the FL service): {num(s.fedavg_local.score)} <RunLink id={s.fedavg_local.run_id} /></div>}
        </div>
        <div className="rounded-md border border-line p-3">
          <div className="mb-2 text-cap font-medium text-ink-1">FL session (from the run's records)</div>
          {fed ? <>
            <div className="grid grid-cols-2 gap-3">
              <Fact label="Rounds" f={f.rounds} />
              <Fact label="Clients" f={f.clients} />
              <Fact label="Secure aggregation" f={f.secure_aggregation} />
              <Fact label="DP" f={f.epsilon ? { ...f.epsilon } : { value: 'none (no DP)', from: 'result' }} />
              <Fact label="Session" f={f.session_id} />
              <Fact label="Final aggregate" f={f.final_aggregation_file} />
            </div>
            <div className="mt-2 tnum text-micro text-ink-2">federated score {num(fed.score)} (no-op {num(fed.noop)})</div>
            <RunLink id={fed.run_id} />
            {(fed.jobs ?? []).map((j: Job) => <JobLine key={j.name} j={j} />)}
            {fed.note && <div className="mt-1 text-micro text-ink-3">{fed.note}</div>}
          </> : <div className="text-micro text-ink-3">federated run: not run yet</div>}
        </div>
      </div>
      {(s.compare ?? []).length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[560px] text-cap" data-testid="t2-compare">
            <thead><tr className="border-b border-line text-left text-micro text-ink-3">{['Comparison', 'Verdict', 'Mean diff (y − x)', '95% CI', 'x vs y'].map((h) => <th key={h} className="px-3 py-2 font-medium">{h}</th>)}</tr></thead>
            <tbody>
              {(s.compare as Rec[]).map((c) => (
                <tr key={c.run_id} className="border-b border-line/60 align-top">
                  <td className="px-3 py-2 text-ink-1">{c.name}<div><RunLink id={c.run_id} /></div></td>
                  <td className="px-3 text-ink-2">{c.kind === 'measure' ? 'measure (no pass bar)' : c.state === 'pass' ? 'PASS' : c.state === 'fail' ? 'MISS' : c.state}</td>
                  <td className="tnum px-3 text-ink-1">{num(c.mean_diff)}</td>
                  <td className="tnum px-3 text-ink-2">{c.ci ? `[${num(c.ci.ci_low)}, ${num(c.ci.ci_high)}]` : '—'}</td>
                  <td className="mono px-3 text-micro text-ink-3">{c.x_model} vs {c.y_model}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

export default function RealConsortium() {
  const { ov, err } = useOverview()
  const c: Rec | undefined = ov?.t2_consortium
  return (
    <Panel id="consortium-real" plane="learn" className="mb-4" title="Track 2 results (real runs)"
      subtitle="per-silo local-only vs federated on the shared test · distance to central"
      actions={<span className="flex items-center gap-1 rounded border border-line px-1.5 py-0.5 text-micro font-semibold tracking-[0.04em] text-ink-2" title="Three simulated operators on the in-process Maveric twin, run in one account; not partner data."><FlaskConical size={11} aria-hidden />SIMULATED SILOS</span>}>
      {err && !ov && <div className="text-cap text-critical-ink">Unavailable: <span className="mono">{err}</span></div>}
      {ov && !(c?.splits ?? []).length && <div className="text-cap text-ink-3">No Track 2 score runs in the evidence mirror yet.</div>}
      {ov && <div className="mb-4"><ConvergenceChart c={ov.curves?.t2_convergence} /></div>}
      {(c?.splits ?? []).map((s: Rec) => <Split key={s.split} s={s} note={c?.score_note ?? ''} conv={ov?.curves?.t2_convergence} />)}
    </Panel>
  )
}
