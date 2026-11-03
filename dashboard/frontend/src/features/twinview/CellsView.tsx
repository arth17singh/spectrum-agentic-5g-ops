import clsx from 'clsx'
import { ArrowDown, ArrowRight, ArrowUp, CircleAlert, RotateCcw, RotateCw, TriangleAlert, UserCheck } from 'lucide-react'
import { useEffect, useState } from 'react'
import { get } from '@/core/api'
import type { Rec } from '@/core/types'
import { AwsTime, AwsTimeSummary, CodeBlock, CopyText, EmptyState, Panel, Segmented, StatusPill } from '@/ui'
import { fixed, pct, signed, utc, type Load } from './format'
import { ProvenanceChip } from './shared'

// Cells mode. Everything comes from GET /api/twinview/cases and /api/twinview/case/{id}. The per-cell state
// is drawn only when the change record carries `twin_state`; otherwise the window says it was not recorded.

const caseKey = (c: Rec) => `${c.run_id ?? ''}\u0000${c.case_id}`

const RUNG_COLOR: Record<string, string> = { R1: 'var(--verify)', R2: 'var(--learn)', R3: 'var(--operate)' }

export function RungBadge({ r }: { r: Rec }) {
  const color = r.code ? RUNG_COLOR[r.code] : 'var(--warn)'
  return (
    <span data-rung={r.code ?? 'unknown'} title={r.basis}
      className="inline-flex shrink-0 items-center gap-1 rounded-[4px] border px-1.5 py-0.5 text-micro font-semibold"
      style={{ borderColor: `color-mix(in oklab, ${color} 60%, transparent)`, color: r.code ? 'var(--ink-1)' : 'var(--warn-ink)', background: `color-mix(in oklab, ${color} 12%, transparent)` }}>
      {r.code ? `${r.code} ${r.label}` : r.label}
    </span>
  )
}

export function CellsView({ cases }: { cases: Rec }) {
  const rows: Rec[] = cases.cases ?? []
  const [sel, setSel] = useState<string | null>(null)
  const picked = rows.find((c) => caseKey(c) === sel) ?? rows[0]
  if (rows.length === 0) {
    return (
      <Panel id="twin-empty" plane="operate" title="Cases" subtitle="none yet">
        <EmptyState title="No applied changes yet"
          body="No change record has been written to the evidence store. When the swarm applies a change through ran___apply_change, its case appears here with the twin state before and after." />
      </Panel>
    )
  }
  return (
    <div className="grid grid-cols-12 items-start gap-4">
      <div className="col-span-12 min-w-0 xl:col-span-3">
        <Panel id="twin-cases" plane="operate" title="Cases" subtitle={`${rows.length} from ${cases.n_changes} changes`} flush>
          <ul>
            {rows.map((c) => {
              const k = caseKey(c)
              const on = picked && caseKey(picked) === k
              return (
                <li key={k}>
                  <button type="button" onClick={() => setSel(k)} aria-pressed={on}
                    className={clsx('w-full border-b border-line/60 px-3 py-2 text-left', on ? 'bg-s2' : 'hover:bg-s2/60')}>
                    <div className="flex min-w-0 flex-wrap items-center gap-1.5">
                      <span className="mono min-w-0 text-cap break-all text-ink-1">{c.case_id}</span>
                      <RungBadge r={c.rung} />
                    </div>
                    <div className="mono mt-0.5 text-micro break-all text-ink-3">{c.run_id ?? 'no run id'}</div>
                    <div className="tnum mt-0.5 text-micro text-ink-3">{c.n_changes} change{c.n_changes === 1 ? '' : 's'} · {c.n_with_state} with cell state · {utc(c.last_time, true)}</div>
                  </button>
                </li>
              )
            })}
          </ul>
        </Panel>
      </div>
      <div className="col-span-12 min-w-0 xl:col-span-9">{picked && <CaseDetail key={caseKey(picked)} summary={picked} />}</div>
    </div>
  )
}

function CaseDetail({ summary }: { summary: Rec }) {
  const [d, setD] = useState<Load<Rec>>({ state: 'loading' })
  const [wi, setWi] = useState(0)
  const [phase, setPhase] = useState<'before' | 'after'>('after')
  const { case_id, run_id } = summary
  const stamp = `${summary.n_changes}|${summary.last_time}`

  useEffect(() => {
    let alive = true
    const q = run_id ? `?run_id=${encodeURIComponent(run_id)}` : ''
    get<Rec>(`/api/twinview/case/${encodeURIComponent(case_id)}${q}`)
      .then((data) => { if (alive) setD({ state: 'ok', data }) })
      .catch((e) => { if (alive) setD({ state: 'error', message: String(e?.message ?? e) }) })
    return () => { alive = false }
  }, [case_id, run_id, stamp])

  if (d.state === 'loading') return <div className="text-cap text-ink-3">Loading {case_id}…</div>
  if (d.state === 'error') return <div className="rounded-md border border-critical/60 bg-critical/10 p-3 text-cap text-critical-ink" role="alert">Could not load {case_id}: {d.message}</div>
  const c = d.data
  const ws: Rec[] = c.windows ?? []
  const w = ws[Math.min(wi, ws.length - 1)]
  return (
    <div className="flex flex-col gap-4">
      <Panel id="twin-schematic" plane="operate" title="Network twin"
        subtitle={`${c.scenario ?? '?'} · seed ${c.seed ?? '?'}`}>
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <RungBadge r={c.rung} />
          {c.provenance && <ProvenanceChip p={c.provenance} />}
          <AwsTimeSummary s={c.provenance?.aws_time_summary} />
          <span className="min-w-0 text-micro text-ink-3">{c.rung?.basis}</span>
        </div>
        <Timeline windows={ws} index={wi} onPick={setWi} />
        {w && (
          <>
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <Segmented size="sm" ariaLabel="Which state" value={phase} onChange={setPhase}
                options={[{ value: 'before', label: `Before (step ${w.before?.step ?? w.step ?? '?'})`, title: 'The state the agents observed when they chose the change' },
                  { value: 'after', label: `After (step ${w.after?.step ?? '?'})`, title: 'After the change and one twin step: what was measured' }]} />
              <CoverageLegend />
            </div>
            <Schematic w={w} phase={phase} />
          </>
        )}
      </Panel>
      {w && <ChangePanel w={w} />}
      <MissingPanel c={c} />
    </div>
  )
}

// ------------------------------------------------------------------ timeline of windows

function Timeline({ windows, index, onPick }: { windows: Rec[]; index: number; onPick: (i: number) => void }) {
  return (
    <ol className="flex flex-wrap items-stretch gap-1.5" aria-label="Windows">
      {windows.map((w, i) => (
        <li key={w.change_id ?? i} className="flex min-w-0 items-center gap-1.5">
          {w.gap_steps > 0 && <span className="text-micro text-ink-3" title="steps with no change applied; no record was written for them">+{w.gap_steps} step{w.gap_steps === 1 ? '' : 's'} idle</span>}
          <button type="button" onClick={() => onPick(i)} aria-pressed={i === index}
            className={clsx('flex min-w-0 flex-col rounded-md border px-2 py-1 text-left', i === index ? 'border-focus bg-s2' : 'border-line hover:bg-s2/60')}>
            <span className="tnum text-micro text-ink-3">step {w.step ?? '?'} → {w.after?.step ?? (typeof w.step === 'number' ? w.step + 1 : '?')}</span>
            <AwsTime aws={w.aws_time} source={w.aws_time_source} runner={w.runner_time} compact />
            <span className="flex flex-wrap items-center gap-1 text-cap text-ink-1">
              {(w.cells as Rec[]).map((x) => <span key={x.cell} className="mono">{x.name}</span>)}
              {w.probe && <StatusPill status="neutral" size="sm">probe</StatusPill>}
              {w.scored === false && !w.probe && <StatusPill status="warning" size="sm">{w.unscored_reason ?? 'unscored'}</StatusPill>}
              {!w.approval?.by && <span title="approver not recorded"><CircleAlert size={12} style={{ color: 'var(--warn-ink)' }} aria-label="approver not recorded" /></span>}
            </span>
          </button>
          {i < windows.length - 1 && <ArrowRight size={12} className="shrink-0 text-ink-3" aria-hidden />}
        </li>
      ))}
    </ol>
  )
}

// ------------------------------------------------------------------ the four cells

/** Coverage share bands (share of the cell's served users with RSRP above the twin's coverage threshold). */
function covColor(v: unknown): string | null {
  if (typeof v !== 'number') return null
  return v >= 0.95 ? 'var(--good)' : v >= 0.8 ? 'var(--warn)' : 'var(--critical)'
}

function CoverageLegend() {
  return (
    <span className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-micro text-ink-3">
      <span>colour = coverage share (serving RSRP above −97 dBm, RSRP only):</span>
      {[['var(--good)', '≥ 95%'], ['var(--warn)', '80–95%'], ['var(--critical)', '< 80%'], ['var(--surface-3)', 'not recorded']].map(([c, l]) => (
        <span key={l} className="flex items-center gap-1"><span className="inline-block size-2.5 rounded-sm" style={{ background: c }} />{l}</span>
      ))}
    </span>
  )
}

function Schematic({ w, phase }: { w: Rec; phase: 'before' | 'after' }) {
  const snap: Rec | null = phase === 'before' ? w.before : w.after
  if (!snap) {
    return (
      <div className="mt-3 rounded-md border border-dashed border-line p-4 text-cap text-ink-2" data-state="no-twin-state">
        <span className="font-medium text-ink-1">Per-cell state not recorded for this window.</span> The change record has no <span className="mono">twin_state</span>, so power, tilt, users and per-cell KPIs cannot be drawn. The network KPIs below come from the record.
      </div>
    )
  }
  const touched = new Map<number, Rec>((w.cells as Rec[]).map((x) => [x.cell, x]))
  return (
    <div className="mt-3">
      <div className="mb-2 flex flex-wrap gap-x-4 gap-y-0.5 text-cap text-ink-2">
        <span>network coverage <span className="tnum mono text-ink-1">{pct(snap.coverage_ratio)}</span></span>
        <span>SLA (SINR above 0 dB) <span className="tnum mono text-ink-1">{pct(snap.sla_ratio)}</span></span>
        <span className="text-ink-3">schematic layout, not to scale</span>
      </div>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {(snap.per_cell as Rec[]).map((cell) => <CellCard key={cell.cell} cell={cell} change={touched.get(cell.cell)} phase={phase} />)}
      </div>
    </div>
  )
}

function CellCard({ cell, change, phase }: { cell: Rec; change?: Rec; phase: 'before' | 'after' }) {
  const col = covColor(cell.coverage_share)
  const down = cell.alarm || cell.asleep
  return (
    <div data-cell={cell.cell} data-touched={change ? 'true' : 'false'}
      className={clsx('relative min-w-0 rounded-md border p-2.5', change ? 'border-2' : 'border-line')}
      style={{ background: col ? `color-mix(in oklab, ${col} 16%, var(--surface-1))` : 'var(--surface-2)', borderColor: change ? 'var(--operate)' : undefined }}>
      <div className="flex min-w-0 flex-wrap items-baseline gap-x-2">
        <span className="mono text-body font-medium text-ink-1">{cell.name}</span>
        <span className="text-micro text-ink-3">cell {cell.cell}</span>
        {down && <StatusPill status="critical" size="sm">{cell.alarm ? 'alarm' : 'asleep'}</StatusPill>}
        <span className="tnum ml-auto text-cap font-medium" style={{ color: col ?? 'var(--ink-3)' }} title="Coverage share: this cell's served users with serving RSRP > −97 dBm (RSRP only, no SINR condition). Empty when the cell serves no users.">{cell.coverage_share == null ? 'cov. —' : `cov. ${pct(cell.coverage_share)}`}</span>
      </div>
      <div className="tnum mono mt-1 grid grid-cols-3 gap-x-2 gap-y-0.5 text-cap text-ink-1">
        <span title="transmit power"><span className="text-ink-3">P </span>{fixed(cell.power_dbm)} dBm</span>
        <span title="electrical tilt"><span className="text-ink-3">tilt </span>{fixed(cell.tilt_deg)}°</span>
        <span title="users served"><span className="text-ink-3">UEs </span>{cell.ue_count ?? '—'}</span>
        <span title="SLA share: this cell's served users with SINR > 0 dB (strictly greater). Empty when the cell serves no users."><span className="text-ink-3">SLA </span>{pct(cell.sla_share)}</span>
        <span className="col-span-2" title="10th percentile SINR of served users"><span className="text-ink-3">p10 SINR </span>{fixed(cell.sinr_p10_db)} dB</span>
        <span className="col-span-3" title="mean RSRP of served users"><span className="text-ink-3">RSRP mean </span>{fixed(cell.rsrp_mean_dbm)} dBm</span>
      </div>
      {change && <ChangeArrow x={change} phase={phase} />}
    </div>
  )
}

function ChangeArrow({ x, phase }: { x: Rec; phase: 'before' | 'after' }) {
  const req: Rec = x.requested ?? {}
  const app: Rec | null = x.applied
  const p = req.d_power_db ?? 0
  const t = req.d_tilt_deg ?? 0
  const PIcon = p >= 0 ? ArrowUp : ArrowDown
  const TIcon = t >= 0 ? RotateCw : RotateCcw
  return (
    <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 rounded border px-2 py-1 text-cap"
      style={{ borderColor: 'color-mix(in oklab, var(--operate) 50%, transparent)', background: 'color-mix(in oklab, var(--operate) 10%, transparent)' }}>
      <span className="text-micro font-semibold text-ink-2">{phase === 'before' ? 'swarm will change' : 'swarm changed'}</span>
      {p !== 0 && <span className="tnum mono flex items-center gap-0.5 text-ink-1"><PIcon size={13} aria-hidden />ΔP {signed(p)} dB</span>}
      {t !== 0 && <span className="tnum mono flex items-center gap-0.5 text-ink-1"><TIcon size={13} aria-hidden />Δtilt {signed(t)}°</span>}
      {x.differs && app && (
        <span className="flex items-center gap-1 text-micro" style={{ color: 'var(--warn-ink)' }} title="The SDK safety filter passed a different change than the one requested">
          <TriangleAlert size={11} aria-hidden />twin applied ΔP {signed(app.d_power_db)} dB, Δtilt {signed(app.d_tilt_deg)}°
        </span>
      )}
    </div>
  )
}

// ------------------------------------------------------------------ predicted vs measured, approval

const KPI_LABEL: Record<string, string> = { coverage_ratio: 'coverage', sla_ratio: 'SLA', sinr_p10_db: 'p10 SINR (dB)', sinr_mean_db: 'mean SINR (dB)' }

function ChangePanel({ w }: { w: Rec }) {
  const pred: Rec | null = w.predicted_delta
  const meas: Rec = w.measured_delta ?? {}
  const err: Rec = w.error ?? {}
  const a: Rec = w.approval ?? {}
  return (
    <Panel id="twin-change" plane="operate" title="Change" subtitle={w.item_id ?? ''}>
      <div className="flex min-w-0 items-center gap-2">
        <div className="min-w-0 flex-1"><CodeBlock code={w.change_text ?? '—'} lang="text" className="!py-1.5 whitespace-pre-wrap break-all" /></div>
        {w.change_id && <span className="mono shrink-0 text-micro text-ink-3">{w.change_id}</span>}
      </div>
      <div className="mt-1.5"><AwsTime aws={w.aws_time} source={w.aws_time_source} runner={w.runner_time} withDate /></div>
      <table className="mt-3 w-full table-fixed text-cap">
        <thead>
          <tr className="text-left text-micro text-ink-3">
            <th className="py-1 pr-2 font-medium">KPI change vs no change</th>
            <th className="px-1 py-1 text-right font-medium">Predicted</th>
            <th className="px-1 py-1 text-right font-medium">Measured</th>
            <th className="py-1 pl-1 text-right font-medium">Abs. error</th>
          </tr>
        </thead>
        <tbody>
          {Object.keys(KPI_LABEL).map((k) => (
            <tr key={k} className="border-t border-line/60">
              <td className="break-words py-1 pr-2 text-ink-2">{KPI_LABEL[k]}</td>
              <td className="tnum mono px-1 text-right text-ink-1">{pred ? signed(pred[k]) : '—'}</td>
              <td className="tnum mono px-1 text-right text-ink-1">{signed(meas[k])}</td>
              <td className="tnum mono py-1 pl-1 text-right text-ink-2">{typeof err[k] === 'number' ? Number(err[k].toPrecision(3)) : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-0.5 text-micro text-ink-3">
        {!pred && <span>no prediction recorded{w.probe ? ' (probe: a fixed change with no model)' : w.unscored_reason ? ` (${w.unscored_reason})` : ''}</span>}
        {pred && <span>direction {w.direction_agree == null ? '—' : w.direction_agree ? 'agrees' : 'disagrees'}</span>}
        {w.predictor && <span>predictor <span className="mono">{w.predictor.model_id} {w.predictor.version}</span></span>}
        {w.policy && <span>policy <span className="mono">{w.policy.model_id} {w.policy.version}</span></span>}
        {(w.trust_before != null || w.trust_after != null) && <span>trust <span className="tnum mono">{w.trust_before ?? '—'} → {w.trust_after ?? '—'}</span></span>}
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-line/60 pt-2 text-cap" data-approval={a.by ? 'named' : 'unnamed'}>
        <UserCheck size={14} aria-hidden style={{ color: a.by ? 'var(--good-ink)' : 'var(--warn-ink)' }} />
        {a.by ? (
          <span className="text-ink-1">Approved by <span className="mono">{a.by}</span>{a.at ? <> at <span className="tnum mono">{utc(a.at, true)}</span></> : ''}</span>
        ) : a.source === 'decision' ? (
          <span className="text-ink-1">Cedar allowed the apply at <span className="tnum mono">{utc(a.allowed_at, true)}</span>; <span style={{ color: 'var(--warn-ink)' }}>the approver is not recorded</span></span>
        ) : (
          <span style={{ color: 'var(--warn-ink)' }}>No approval record found for this change</span>
        )}
        {a.source && <span className="text-micro text-ink-3">from {a.source}</span>}
        {a.reason && <span className="w-full text-micro text-ink-2">“{a.reason}”</span>}
      </div>
    </Panel>
  )
}

function MissingPanel({ c }: { c: Rec }) {
  const miss: Rec[] = c.missing ?? []
  if (!miss.length && !c.show_record) return null
  return (
    <Panel id="twin-fields" plane="operate" title="What the records carry" subtitle={miss.length ? `${miss.length} gap${miss.length === 1 ? '' : 's'}` : 'complete'}>
      {miss.length > 0 && (
        <ul className="mb-3 flex flex-col gap-1">
          {miss.map((m) => (
            <li key={m.field} className="flex items-start gap-1.5 text-cap text-ink-2">
              <TriangleAlert size={13} className="mt-0.5 shrink-0" style={{ color: 'var(--warn-ink)' }} aria-hidden />
              <span><span className="mono break-all text-ink-1">{m.field}</span> missing{m.n_windows ? ` in ${m.n_windows} window${m.n_windows === 1 ? '' : 's'}` : ''}: {m.effect}.</span>
            </li>
          ))}
        </ul>
      )}
      <div className="flex min-w-0 items-center gap-2">
        <div className="min-w-0 flex-1"><CodeBlock code={c.show_record} lang="text" className="!py-1.5 whitespace-pre-wrap break-all" /></div>
        <CopyText value={c.show_record} />
      </div>
      <div className="mt-1 text-micro text-ink-3">Run this from the repository root to print the same records outside the dashboard.</div>
    </Panel>
  )
}
