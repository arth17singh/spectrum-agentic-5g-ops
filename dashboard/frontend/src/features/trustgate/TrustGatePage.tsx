import clsx from 'clsx'
import { CircleCheck, CircleX, Lock, OctagonX, RefreshCw, TriangleAlert, User } from 'lucide-react'
import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { get } from '@/core/api'
import { Button, EmptyState, PageHeader, Panel, RealChip } from '@/ui'
import { LadderPanel, SyntheticTag } from './LadderPanel'
import manifest from './manifest'
import { TrustChart } from './TrustChart'
import type { ChangeRow, ChangesResponse, LadderResponse, ModelsResponse, ModelSummary, PauseEvent, Thresholds } from './types'

// Every value comes from the evidence records via /api/trustgate/*. Nothing is simulated or recomputed here.
// Records marked synthetic (fixtures) are flagged as such, and the REAL chip is shown only when none are.

const utc = (iso?: string | null) => (iso ? `${iso.slice(11, 19)}Z` : '—')
const f2 = (v: number | null | undefined) => (v == null ? '—' : v.toFixed(2))
const signed = (v: number | null) => (v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(3)}`)
const keyOf = (m: { model_id: string; version: string }) => `${m.model_id}\u0000${m.version}`

export default function TrustGatePage() {
  const [models, setModels] = useState<ModelsResponse | null>(null)
  const [ladder, setLadder] = useState<LadderResponse | null>(null)
  const [detail, setDetail] = useState<ChangesResponse | null>(null)
  const [sel, setSel] = useState<string | null>(null)
  const [fail, setFail] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const fetchAll = useCallback(() => {
    Promise.all([get<ModelsResponse>('/api/trustgate/models'), get<LadderResponse>('/api/trustgate/ladder')])
      .then(([m, l]) => { setModels(m); setLadder(l); setFail(null) })
      .catch((e) => setFail(String(e)))
      .finally(() => setBusy(false))
  }, [])
  const refresh = () => { setBusy(true); fetchAll() }
  useEffect(() => { fetchAll(); const id = setInterval(fetchAll, 15000); return () => clearInterval(id) }, [fetchAll])

  const list = models?.models ?? []
  const current = list.find((m) => keyOf(m) === sel) ?? list[0] ?? null
  const curKey = current ? keyOf(current) : null
  // refetch the selected model's rows whenever the summary changes (new records, a pause)
  const stamp = current ? `${current.n_changes}/${current.pauses.length}/${current.trust_latest}` : ''
  useEffect(() => {
    if (!current) return
    let live = true
    get<ChangesResponse>(`/api/trustgate/changes?model_id=${encodeURIComponent(current.model_id)}&version=${encodeURIComponent(current.version)}`)
      .then((d) => { if (live) setDetail(d) })
      .catch((e) => { if (live) { setDetail(null); setFail(String(e)) } })
    return () => { live = false }
  }, [curKey, stamp]) // eslint-disable-line react-hooks/exhaustive-deps

  const synthetic = list.some((m) => m.synthetic) || (ladder?.ladders ?? []).some((l) => l.synthetic) || (models?.probes ?? []).some((p) => p.synthetic)

  return (
    <div>
      <PageHeader title={manifest.title} description={manifest.description} tracks={manifest.tracks}
        actions={<>
          {models?.verification?.label === 'VERIFIED'
            ? <RealChip title={models.verification.note} />
            : <SyntheticTag what={models?.verification?.label ?? 'NOT YET CHECKED'} />}
          <Button icon={<RefreshCw size={14} className={clsx(busy && 'animate-spin')} />} disabled={busy} onClick={refresh}>Refresh</Button>
        </>} />
      {synthetic && (
        <div className="mb-4 flex items-start gap-2 rounded-md border border-warn/50 bg-warn/10 px-3 py-2 text-cap text-warn-ink" role="note">
          <TriangleAlert size={14} className="mt-0.5 shrink-0" aria-hidden />
          Inputs: {models?.verification?.n_synthetic_inputs ?? 'some'} records on this screen are marked synthetic (fixtures, e.g. a dry-run policy client). That describes the inputs; whether the runs are verified is check_unit's verdict: {models?.verification?.label ?? 'not yet checked'}.
        </div>
      )}
      {(fail || models?.error) && (
        <div className="mb-4 rounded-md border border-critical/60 bg-critical/10 p-3 text-cap text-critical-ink" role="alert">
          Could not read the trust records: {fail ?? models?.error}
        </div>
      )}

      {models == null ? <div className="text-cap text-ink-3">Loading trust records…</div> : list.length === 0 ? (
        <Panel title="Trust per model version" plane="verify">
          <EmptyState title="No scored change from a predictor yet"
            body="The trust line starts with the first applied change that a Track 2 predictor predicted. Nothing is drawn until then." />
        </Panel>
      ) : (
        <div className="grid grid-cols-12 items-start gap-4">
          <div className="col-span-12 min-w-0 space-y-4 lg:col-span-3">
            <Panel id="model-list" plane="verify" title="Model versions" subtitle={`${list.length}`} flush>
              <ul>
                {list.map((m) => (
                  <li key={keyOf(m)}>
                    <button type="button" onClick={() => setSel(keyOf(m))} aria-pressed={keyOf(m) === curKey}
                      className={clsx('block w-full border-b border-line/60 px-4 py-3 text-left', keyOf(m) === curKey ? 'bg-s2' : 'hover:bg-s2/60')}>
                      <div className="mono truncate text-body text-ink-1">{m.model_id}</div>
                      <div className="mono text-cap text-ink-2">{m.version}</div>
                      <div className="mt-1.5 flex flex-wrap items-center gap-1.5"><PauseBadge m={m} />{m.synthetic && <SyntheticTag what="SYNTHETIC" />}</div>
                      <div className="tnum mt-1 text-micro text-ink-3">trust {f2(m.trust_latest)} · {m.n_scored} scored{m.n_unscored ? ` · ${m.n_unscored} unscored` : ''}</div>
                    </button>
                  </li>
                ))}
              </ul>
            </Panel>
            <ProbesPanel probes={models.probes} />
          </div>
          <div className="col-span-12 min-w-0 space-y-4 lg:col-span-9">
            {current && <TrustPanel m={current} />}
            {current && <ChangesPanel d={detail} m={current} />}
          </div>
        </div>
      )}

      <div className="mt-4"><LadderPanel ladders={ladder?.ladders ?? null} error={ladder?.error ?? null} /></div>
    </div>
  )
}

// ------------------------------------------------------------------ status

function PauseBadge({ m }: { m: ModelSummary }) {
  if (m.paused && m.paused_by_kind === 'trust-gate')
    return <span className="inline-flex items-center gap-1 rounded-[4px] bg-critical/15 px-1.5 py-0.5 text-micro font-bold text-critical-ink" title="Paused automatically by the trust gate"><OctagonX size={12} aria-hidden />AUTO-PAUSED · trust gate</span>
  if (m.paused)
    return <span className="inline-flex items-center gap-1 rounded-[4px] border border-line bg-s3 px-1.5 py-0.5 text-micro font-semibold text-ink-1" title="Paused by a person"><User size={12} aria-hidden />PAUSED · {m.paused_by ?? 'a person'}</span>
  return <span className="inline-flex items-center gap-1 rounded-[4px] bg-good/15 px-1.5 py-0.5 text-micro font-semibold text-good-ink"><CircleCheck size={12} aria-hidden />{m.pauses.length ? 'RESUMED' : 'NOT PAUSED'}</span>
}

// ------------------------------------------------------------------ trust line

function TrustPanel({ m }: { m: ModelSummary }) {
  return (
    <Panel id="trust-line" plane="verify" title={<span className="mono block max-w-[55vw] truncate md:max-w-none" title={`${m.model_id} ${m.version}`}>{m.model_id} {m.version}</span>} subtitle="trust after each scored change"
      actions={<PauseBadge m={m} />}>
      {m.status_conflict && (
        <div className="mb-3 flex items-center gap-2 rounded-md border border-warn/50 bg-warn/10 px-3 py-2 text-cap text-warn-ink">
          <TriangleAlert size={14} aria-hidden />The model record says “{m.model_status}” but the newest control record says {m.paused ? 'pause' : 'resume'}. The control record is shown.
        </div>
      )}
      <TrustChart points={m.points} pauses={m.pauses} pauseBelow={m.pause_below} />
      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <PauseEvents events={m.pauses} />
        <RuleBox th={m.thresholds} />
      </div>
    </Panel>
  )
}

function PauseEvents({ events }: { events: PauseEvent[] }) {
  return (
    <div>
      <h4 className="mb-1.5 text-cap font-medium text-ink-2">Pause and resume events</h4>
      {events.length === 0 ? <div className="text-cap text-ink-3">None. This model version has never been paused.</div> : (
        <ul className="space-y-2">
          {events.map((e, i) => {
            const auto = e.by_kind === 'trust-gate'
            return (
              <li key={i} className={clsx('rounded-md border p-2.5 text-cap', auto && e.action === 'pause' ? 'border-critical/60 bg-critical/10' : 'border-line bg-s2/40')}>
                <div className="flex flex-wrap items-center gap-2">
                  {auto
                    ? <span className="inline-flex items-center gap-1 font-bold text-critical-ink"><OctagonX size={13} aria-hidden />{e.action === 'pause' ? 'AUTO-PAUSE BY TRUST GATE' : 'RESUMED BY TRUST GATE'}</span>
                    : <span className="inline-flex items-center gap-1 font-semibold text-ink-1"><User size={13} aria-hidden />{e.action === 'pause' ? 'PAUSED BY A PERSON' : 'RESUMED BY A PERSON'}</span>}
                  <span className="mono text-ink-2">{e.by}</span>
                  <span className="tnum ml-auto text-ink-3">{utc(e.time)}</span>
                </div>
                <div className="mt-1 text-ink-2">{e.reason || 'No reason recorded.'}</div>
                {e.policy_name && <div className="mono mt-0.5 text-micro text-ink-3">{e.policy_name}</div>}
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

function RuleBox({ th }: { th: Thresholds | null }) {
  if (!th) return (
    <div>
      <h4 className="mb-1.5 text-cap font-medium text-ink-2">Pause rule</h4>
      <div className="text-cap text-ink-3">No declared thresholds were found in the result or model records, so τ, the scales and the pause line are not shown.</div>
    </div>
  )
  const rows: [string, ReactNode][] = []
  if (th.tau != null) rows.push(['τ (error limit per change)', th.tau])
  if (th.pause_below != null) rows.push(['Pause line (EWMA trust)', th.pause_below])
  if (th.watch_below != null) rows.push(['Watch line', th.watch_below])
  if (th.alpha != null) rows.push(['EWMA α', th.alpha])
  if (th.scales) rows.push(['Scales per KPI', Object.entries(th.scales).map(([k, v]) => `${k} ${v}`).join(' · ')])
  return (
    <div>
      <h4 className="mb-1.5 text-cap font-medium text-ink-2">Pause rule</h4>
      <p className="text-cap text-ink-2">
        After {th.n_min ?? 3} or more scored changes, pause when {(typeof th.window === 'object' ? th.window.needed : undefined) ?? th.k_of ?? 2} of the last {(typeof th.window === 'object' ? th.window.of_last : th.window) ?? 3} have e above τ, or when the EWMA trust falls below the pause line.
        Here e is the mean over KPIs of |predicted Δ − measured Δ| ÷ scale.
      </p>
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5 text-cap">
        {rows.map(([k, v]) => <div key={k} className="contents"><dt className="text-ink-3">{k}</dt><dd className="tnum text-ink-1">{v}</dd></div>)}
      </dl>
      {th.source && <div className="mono mt-1 text-micro text-ink-3">declared in {th.source}</div>}
    </div>
  )
}

// ------------------------------------------------------------------ predicted vs measured

function ChangesPanel({ d, m }: { d: ChangesResponse | null; m: ModelSummary }) {
  const rows = d && d.model_id === m.model_id && d.version === m.version ? d.rows : null
  const keys = [...new Set((rows ?? []).flatMap((r) => r.kpis.map((k) => k.key)))]
  const tau = m.thresholds?.tau
  return (
    <Panel id="predicted-vs-measured" plane="verify" title="Predicted vs measured" flush
      subtitle={`${m.n_changes} change${m.n_changes === 1 ? '' : 's'}, effects as change against doing nothing`}
      actions={m.n_direction_known > 0 ? <span className="tnum text-cap text-ink-2">direction agrees on {m.n_direction_agree} of {m.n_direction_known} scored</span> : undefined}>
      {rows == null ? <div className="p-4 text-cap text-ink-3">Loading changes…</div> : rows.length === 0 ? <EmptyState title="No changes recorded for this model version" /> : (
        <div className="max-w-full overflow-x-auto" data-testid="changes-scroll">
          <table className="w-full text-cap">
            <thead>
              <tr className="border-b border-line text-left text-micro text-ink-3">
                {['Time', 'Change'].map((h) => <th key={h} className="px-3 py-2 font-medium whitespace-nowrap">{h}</th>)}
                {keys.map((k) => <th key={k} className="px-3 py-2 font-medium whitespace-nowrap" title="predicted Δ → measured Δ (absolute error)">{k}<div className="font-normal">pred → meas (err)</div></th>)}
                <th className="px-3 py-2 font-medium whitespace-nowrap">Error e{tau != null ? <div className="font-normal">τ = {tau}</div> : null}</th>
                <th className="px-3 py-2 font-medium whitespace-nowrap">Direction</th>
                <th className="px-3 py-2 font-medium whitespace-nowrap">Trust</th>
                <th className="px-3 py-2 font-medium whitespace-nowrap">Counted</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => <ChangeTr key={`${r.run_id}-${r.change_id}`} r={r} keys={keys} />)}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  )
}

function ChangeTr({ r, keys }: { r: ChangeRow; keys: string[] }) {
  const by = new Map(r.kpis.map((k) => [k.key, k]))
  return (
    <tr className={clsx('border-b border-line/60 align-top', !r.scored && 'opacity-60')} data-scored={r.scored}>
      <td className="tnum px-3 py-2 whitespace-nowrap text-ink-2">{utc(r.time)}</td>
      <td className="px-3 py-2">
        <div className="mono whitespace-nowrap text-ink-1">{r.change_id ?? '—'}</div>
        {r.change_text && <div className="mono max-w-[26ch] truncate text-micro text-ink-3" title={r.change_text}>{r.change_text}</div>}
      </td>
      {keys.map((k) => {
        const v = by.get(k)
        return (
          <td key={k} className="tnum px-3 py-2 whitespace-nowrap">
            {v ? <><span className="text-ink-2">{signed(v.predicted_delta)}</span><span className="text-ink-3"> → </span><span className="text-ink-1">{signed(v.measured_delta)}</span>
              <span className="text-ink-3"> ({v.error == null ? '—' : v.error.toFixed(3)})</span></> : '—'}
          </td>
        )
      })}
      <td className={clsx('tnum px-3 py-2 font-semibold', r.over_tau ? 'text-critical-ink' : 'text-ink-1')}>
        {f2(r.error_summary)}{r.over_tau ? <span className="ml-1 text-micro font-bold">&gt; τ</span> : null}
      </td>
      <td className="px-3 py-2 whitespace-nowrap">
        {r.direction_agree == null ? <span className="text-ink-3">—</span>
          : r.direction_agree ? <span className="inline-flex items-center gap-1 text-good-ink"><CircleCheck size={13} aria-hidden />agrees</span>
          : <span className="inline-flex items-center gap-1 font-medium text-critical-ink"><CircleX size={13} aria-hidden />disagrees</span>}
      </td>
      <td className="tnum px-3 py-2 whitespace-nowrap text-ink-2">{r.scored ? `${f2(r.trust_before)} → ${f2(r.trust_after)}` : '—'}</td>
      <td className="px-3 py-2 whitespace-nowrap">
        {r.scored ? <span className="text-ink-2">scored</span>
          : <span className="inline-flex items-center gap-1 rounded border border-line px-1.5 text-micro text-ink-2" title="Shown, but never counted toward trust">unscored: {r.unscored_reason ?? 'no reason recorded'}</span>}
      </td>
    </tr>
  )
}

// ------------------------------------------------------------------ probes

function ProbesPanel({ probes }: { probes: ChangeRow[] }) {
  return (
    <Panel id="probes" plane="verify" title="Probes" subtitle="fixed changes, no model" flush>
      {probes.length === 0 ? <div className="p-4 text-cap text-ink-3">No probes recorded.</div> : (
        <ul>
          {probes.map((p) => (
            <li key={`${p.run_id}-${p.change_id}`} className="border-b border-line/60 px-4 py-2 text-cap">
              <div className="flex items-baseline gap-2"><span className="mono text-ink-1">{p.change_id}</span><span className="tnum ml-auto text-ink-3">{utc(p.time)}</span></div>
              <div className="tnum text-ink-2">{p.kpis.map((k) => `${k.key.replace('_ratio', '').replace('sinr_', '')} ${signed(k.measured_delta)}`).join(' · ')}</div>
            </li>
          ))}
        </ul>
      )}
      <div className="flex items-center gap-1.5 px-4 py-2 text-micro text-ink-3"><Lock size={11} aria-hidden />Probes have no prediction, so they are never scored.</div>
    </Panel>
  )
}
