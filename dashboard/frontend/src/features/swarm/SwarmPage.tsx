import clsx from 'clsx'
import { CircleCheck, CircleX, ExternalLink, RefreshCw } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { get } from '@/core/api'
import type { Rec } from '@/core/types'
import { AwsTime, AwsTimeSummary, Button, PageHeader, Panel, RealChip } from '@/ui'
import { CELL_POS, hexPath } from './hex'
import manifest from './manifest'

// Swarm Tower: the live Track 3 swarm run and the Track 2 training silos, from GET /api/swarmtower (the evidence
// records, read-only). Nothing on this page is simulated and there are no simulator controls.
//   refused     the change stopped at the approval gate: ready but NOT applied; it waits for a person. The run is
//               proposal-only, so nobody approved and nothing was applied on the twin.
//   state       the lead agent's task state when the case was read (finished, or still working at the time limit)
//   recovered   from the scoring run: replaying the swarm's proposals, every graded KPI ends within tolerance of
//               the healthy network at the grade step


const sign = (v: number, d = 1) => `${v > 0 ? '+' : ''}${v.toFixed(d)}`
const pct = (v: unknown) => (typeof v === 'number' ? `${(v * 100).toFixed(0)}%` : '—')
const num = (v: unknown, d = 3) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : '—')

function Cells({ changes, fault, size = 300 }: { changes: Rec[]; fault: Rec[]; size?: number }) {
  const R = size / 7.2
  const ox = R * 1.3, oy = R * 2.0
  const byCell = (list: Rec[]) => Object.fromEntries(list.map((c) => [c.cell, c]))
  const ch = byCell(changes), fl = byCell(fault)
  return (
    <svg viewBox={`0 0 ${size} ${size * 0.95}`} className="h-auto w-full max-w-[360px]" role="img" aria-label="The 4 Maveric cells with this step's changes">
      {CELL_POS.map(([dx, dy], cell) => {
        const cx = ox + dx * R * 1.12, cy = oy + dy * R * 1.12
        const c = ch[cell], f = fl[cell]
        return (
          <g key={cell}>
            <path d={hexPath(cx, cy, R)} style={{ fill: c ? 'color-mix(in oklab, var(--operate) 22%, var(--surface-2))' : 'var(--surface-2)', stroke: f ? 'var(--critical)' : 'var(--hairline)' }} strokeWidth={f ? 2.5 : 1.2} />
            <text x={cx} y={cy - R * 0.45} textAnchor="middle" style={{ fill: 'var(--ink-1)', fontSize: 12, fontWeight: 600 }}>cell {cell}</text>
            {c ? (
              <>
                <text x={cx} y={cy + 2} textAnchor="middle" className="tnum" style={{ fill: 'var(--ink-1)', fontSize: 11 }}>power {sign(c.d_power_db)} dB</text>
                <text x={cx} y={cy + 16} textAnchor="middle" className="tnum" style={{ fill: 'var(--ink-1)', fontSize: 11 }}>tilt {sign(c.d_tilt_deg)}°</text>
              </>
            ) : <text x={cx} y={cy + 8} textAnchor="middle" style={{ fill: 'var(--ink-3)', fontSize: 11 }}>no change</text>}
          </g>
        )
      })}
    </svg>
  )
}

function Cluster({ n, size = 18 }: { n: number; size?: number }) {
  // one silo: n clusters of 4 cells, each cluster a tiny 4-hex group
  const R = size / 4
  const cols = Math.min(n, 4)
  const w = cols * size * 1.15, h = Math.ceil(n / 4) * size * 1.05
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width={w} height={h} role="img" aria-label={`${n} clusters of 4 cells`}>
      {Array.from({ length: n }, (_, k) => {
        const bx = (k % 4) * size * 1.15 + R * 1.2, by = Math.floor(k / 4) * size * 1.05 + R * 1.4
        return CELL_POS.map(([dx, dy], c) => <path key={`${k}-${c}`} d={hexPath(bx + dx * R, by + dy * R, R * 0.95)} style={{ fill: 'color-mix(in oklab, var(--learn) 30%, var(--surface-2))', stroke: 'var(--hairline)' }} strokeWidth={0.6} />)
      })}
    </svg>
  )
}

const OUTCOME_TONE: Record<string, string> = { held: 'var(--warn-ink)', none: 'var(--ink-3)', problem: 'var(--critical-ink)', other: 'var(--ink-2)' }

export default function SwarmPage() {
  const [d, setD] = useState<Rec | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [pick, setPick] = useState(0)
  const [step, setStep] = useState(0)
  const [busy, setBusy] = useState(false)
  const load = useCallback(() => get<Rec>('/api/swarmtower').then((x) => { setD(x); setErr(null) }).catch((e) => setErr(String(e))), [])
  useEffect(() => { load() }, [load])
  const incidents: Rec[] = d?.incidents ?? []
  const inc: Rec | undefined = incidents[pick]
  const s: Rec | undefined = inc?.steps?.find((x: Rec) => x.step === step) ?? inc?.steps?.[0]
  const sc: Rec | null = d?.scoring ?? null
  const prov: Rec | null = d?.provenance ?? null

  return (
    <div className="min-w-0" data-slice="swarmtower">
      <PageHeader title={manifest.title} description={manifest.description} tracks={manifest.tracks}
        actions={<>
          {prov?.label === 'VERIFIED' || prov?.state === 'real'
            ? <RealChip title={prov?.note ?? 'Read from the AWS-recorded evidence records.'} />
            : prov ? <span className="rounded border border-line px-1.5 py-0.5 text-micro font-semibold text-warn-ink" title={prov.note}>{prov.label}</span> : null}
          <Button icon={<RefreshCw size={14} className={clsx(busy && 'animate-spin')} />} disabled={busy} onClick={() => { setBusy(true); load().finally(() => setBusy(false)) }}>Refresh</Button>
        </>} />
      {err && <div role="alert" className="mb-4 rounded-md border border-critical/60 bg-critical/10 p-3 text-cap text-ink-1">{err}</div>}
      {!d && !err && <div className="text-cap text-ink-3">Loading the swarm records…</div>}
      {d && !d.run && <div className="mb-4 rounded-md border border-dashed border-line p-4 text-cap text-ink-3">No live swarm run in the evidence records yet.</div>}

      {d?.run && (
        <div className="mb-4 rounded-[10px] border border-line bg-s1 px-4 py-3 text-cap" data-testid="swarm-headline">
          <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
            <span className="text-ink-1">Live swarm run <Link to={`/evidence?run=${encodeURIComponent(d.run.run_id)}`} className="mono text-micro text-ink-3 underline decoration-dotted">{d.run.run_id}</Link></span>
            <span className="text-ink-2">{d.run.n_items} cases (18 incidents × 3 steps) · {pct(d.run.pass_rate)} read to the lead's answer</span>
            {d.aws_time_summary && <AwsTimeSummary s={d.aws_time_summary} />}
          </div>
          {sc && (
            <div className="mt-1 flex flex-wrap items-baseline gap-x-4 gap-y-1 text-ink-2">
              <span>Scored on the twin: <b className="text-ink-1">{sc.success_count} of {sc.n_items}</b> incidents recovered</span>
              <span>gain over config revert <b className="tnum text-ink-1">{sign(sc.gain?.value ?? 0, 3)}</b> (95% CI {num(sc.gain?.ci_low)} to {num(sc.gain?.ci_high)})</span>
              <Link to={`/evidence?run=${encodeURIComponent(sc.run_id)}`} className="mono text-micro text-ink-3 underline decoration-dotted">{sc.run_id}</Link>
            </div>
          )}
          <div className="mt-1 text-micro text-ink-3">{d.run.approval_policy ? 'Proposal-only: every change waited at the approval gate; nobody approved, so nothing was applied on the twin. ' : ''}Agents: {(d.run.models ?? []).map((m: Rec) => `${m.role} ${m.model}`).join(' · ')}</div>
        </div>
      )}

      {inc && (
        <Panel id="swarm-network" plane="operate" title="Swarm network" subtitle="the 4 cells of the team's Maveric twin · one incident, one operator step at a time">
          <div className="grid min-w-0 gap-4 lg:grid-cols-[280px_1fr]">
            <ol className="max-h-[220px] min-w-0 space-y-1 overflow-y-auto pr-1 lg:max-h-[520px]" data-testid="incident-list" aria-label="Incidents">
              {incidents.map((x, i) => (
                <li key={x.incident_id}>
                  <button type="button" onClick={() => { setPick(i); setStep(0) }}
                    className={clsx('flex w-full min-w-0 items-center gap-2 rounded-md border px-2 py-1.5 text-left text-cap', i === pick ? 'border-line bg-s2' : 'border-transparent hover:bg-s2/60')}>
                    {x.recovered ? <CircleCheck size={14} style={{ color: 'var(--good-ink)' }} aria-label="recovered" /> : <CircleX size={14} style={{ color: 'var(--critical-ink)' }} aria-label="not recovered" />}
                    <span className="mono min-w-0 truncate text-micro text-ink-1">{x.incident_id}</span>
                    <span className="ml-auto shrink-0 text-micro text-ink-3">{x.recovered ? 'recovered' : 'not recovered'}</span>
                  </button>
                </li>
              ))}
            </ol>
            <div className="min-w-0 [overflow-wrap:anywhere]">
              <div className="mb-2 flex flex-wrap items-center gap-2">
                <span className="mono text-cap text-ink-1">{inc.incident_id}</span>
                <span className="text-micro" style={{ color: inc.recovered ? 'var(--good-ink)' : 'var(--critical-ink)' }}>{inc.recovered ? 'recovered' : 'not recovered'}{typeof inc.recovery === 'number' ? ` (recovery ${num(inc.recovery, 2)})` : ''}</span>
              </div>
              <div className="mb-2 text-micro text-ink-3">{inc.preset} · seed {inc.seed} · {inc.tier} · fault {inc.template}: {(inc.fault ?? []).map((f: Rec) => `cell ${f.cell}${f.d_power_db ? ` power ${sign(f.d_power_db)} dB` : ''}${f.d_tilt_deg ? ` tilt ${sign(f.d_tilt_deg)}°` : ''}`).join(', ') || 'not recorded'}</div>
              <div className="mb-3 flex items-center gap-1" role="group" aria-label="Operator step">
                {(inc.steps ?? []).map((x: Rec) => (
                  <Button key={x.step} size="sm" variant={x.step === s?.step ? 'primary' : 'secondary'} onClick={() => setStep(x.step)}>Step {x.step}</Button>
                ))}
              </div>
              {s && (
                <div className="grid min-w-0 gap-3 md:grid-cols-[minmax(0,360px)_1fr]">
                  <Cells changes={s.cells ?? []} fault={step === 0 ? inc.fault ?? [] : []} />
                  <dl className="grid min-w-0 grid-cols-[110px_1fr] gap-x-3 gap-y-1.5 text-cap" data-testid="step-detail">
                    <dt className="text-ink-3">Outcome</dt><dd style={{ color: OUTCOME_TONE[s.outcome?.code] ?? 'var(--ink-2)' }}>{s.outcome?.words}</dd>
                    {s.approval_id && <><dt className="text-ink-3">Approval</dt><dd className="mono text-micro text-ink-2">{s.approval_id} (waiting; not approved)</dd></>}
                    <dt className="text-ink-3">Lead agent</dt><dd className="text-ink-2">{s.state_words}</dd>
                    <dt className="text-ink-3">Case read</dt><dd className="text-ink-2">{s.passed ? 'yes' : 'no'} · latency {typeof s.latency_ms === 'number' ? `${(s.latency_ms / 1000).toFixed(0)} s` : '—'}</dd>
                    <dt className="text-ink-3">AWS time</dt><dd><AwsTime aws={s.aws_time} source={s.aws_time_source} runner={s.runner_time} withDate /></dd>
                    <dt className="text-ink-3">Case</dt><dd className="mono text-micro text-ink-2">{s.case_id} {s.in_trace ? <Link to="/trace" className="underline decoration-dotted">in Trace <ExternalLink size={10} className="inline" /></Link> : <span className="font-sans text-ink-3">(not a Trace case: the swarm run recorded one item per case, no decision records)</span>}</dd>
                    <dt className="text-ink-3">Change</dt><dd className="text-micro text-ink-2">{(s.cells ?? []).length ? (s.cells as Rec[]).map((c) => `cell ${c.cell}: power ${sign(c.d_power_db)} dB, tilt ${sign(c.d_tilt_deg)}°`).join(' · ') : 'none'}</dd>
                    {s.reply_excerpt && <><dt className="text-ink-3">Lead said</dt><dd className="text-micro text-ink-3">{String(s.reply_excerpt).replace(/\*\*/g, '').slice(0, 260)}…</dd></>}
                  </dl>
                </div>
              )}
              {step === 0 && (inc.fault ?? []).length > 0 && <div className="mt-1 text-micro text-ink-3">Red outline: the cell the fault was injected on (its offset is the incident's starting fault).</div>}
            </div>
          </div>
        </Panel>
      )}

      {d?.silos && (
        <Panel id="swarm-silos" plane="learn" className="mt-4" title="Track 2 training silos"
          subtitle="where Track 2 trained; built from the same 4-cell Maveric model">
          <div className="mb-3 text-cap text-ink-2" data-testid="silos-note">
            Where Track 2 trained, built from the same 4-cell Maveric model. The swarm did not run inside these silos.{' '}
            <Link to="/twin-view?mode=consortium" className="underline decoration-dotted">Federated results on Twin View</Link>.
          </div>
          <div className="grid min-w-0 gap-3 md:grid-cols-3">
            {(d.silos.silos as Rec[]).map((x) => (
              <div key={x.silo} className="min-w-0 rounded-md border border-line p-3" data-silo={x.silo}>
                <div className="mb-1 flex items-baseline gap-2"><span className="text-cap font-medium text-ink-1">Silo {x.silo}</span><span className="text-micro text-ink-3">{x.size} · {x.clusters} cluster{x.clusters === 1 ? '' : 's'} × 4 = {x.cells} cells</span></div>
                <div className="mb-2 overflow-hidden"><Cluster n={x.clusters ?? 0} /></div>
                <dl className="grid grid-cols-[120px_1fr] gap-x-2 gap-y-0.5 text-micro">
                  <dt className="text-ink-3">users per cluster</dt><dd className="text-ink-1">{x.users_per_cluster} {x.mobility}</dd>
                  <dt className="text-ink-3">noise floor</dt><dd className="tnum text-ink-1">{x.noise_floor_dbm} dBm</dd>
                  <dt className="text-ink-3">start power</dt><dd className="tnum text-ink-1">{Array.isArray(x.power_start_dbm) ? `${x.power_start_dbm[0]}–${x.power_start_dbm[1]} dBm` : '—'}</dd>
                  <dt className="text-ink-3">start tilt</dt><dd className="tnum text-ink-1">{Array.isArray(x.tilt_start_deg) ? `${x.tilt_start_deg[0]}–${x.tilt_start_deg[1]}°` : '—'}</dd>
                  <dt className="text-ink-3">leans to</dt><dd className="text-ink-1">{x.mix?.lean ?? x.mix?.summary ?? '—'}</dd>
                  <dt className="text-ink-3">train / test rows</dt><dd className="tnum text-ink-1">{x.train_rows?.toLocaleString?.() ?? '—'} / {x.test_rows?.toLocaleString?.() ?? '—'}</dd>
                </dl>
              </div>
            ))}
          </div>
          <div className="mt-2 text-micro text-ink-3 [overflow-wrap:anywhere]">
            {d.silos.manifest?.label} · {d.silos.manifest?.twin} · generator commit <span className="mono">{String(d.silos.manifest?.generator_commit ?? '').slice(0, 12)}</span> · config <span className="mono">{String(d.silos.manifest?.config_sha256 ?? '').slice(0, 12)}</span>
          </div>
        </Panel>
      )}
    </div>
  )
}
