import clsx from 'clsx'
import { ChevronDown, ChevronRight, CircleCheck, CircleDashed, CircleX, FlaskConical, RefreshCw, TriangleAlert } from 'lucide-react'
import { Fragment, useCallback, useEffect, useState, type ReactNode } from 'react'
import { get } from '@/core/api'
import type { Rec } from '@/core/types'
import { Button, CodeBlock, CopyText, EmptyState, PageHeader, Panel, RealChip } from '@/ui'
import { gainText, stateAttr, targetText, utc, value } from './format'
import manifest from './manifest'

// Everything here is read from the declared/ and results/ records through GET /api/scorecard/units.
// The server recomputes pass/fail from the measured value and the DECLARED target; a stored
// "passed" that disagrees shows as inconsistent, never as a pass. Rows marked SYNTHETIC are test
// fixtures, not measurements.

interface Problem { code: string; detail: string }
interface Row {
  track: string; unit: string; run_id: string
  target: { metric: string; op: string; value: number; unit?: string } | null
  measured: number | null
  state: 'pass' | 'fail' | 'not checked' | 'inconsistent'
  recomputed: 'pass' | 'fail' | null
  stored_passed: boolean | null
  n_items: number | null; declared_n_items: number | null
  checked_at: string | null; declared_at: string | null
  checker: { script?: string; code_sha?: string } | null
  baseline_run_id: string | null
  gain: { value: number; ci_low: number; ci_high: number; baseline_run_id: string; paired_n: number } | null
  gain_missing: boolean
  synthetic: boolean
  problems: Problem[]
}
interface TrackGroup { track: string; declared: number; passed: number; failed: number; not_checked: number; inconsistent: number; units: Row[] }
interface Card {
  tracks: TrackGroup[]
  totals: { declared: number; passed: number; failed: number; not_checked: number; inconsistent: number }
  synthetic_rows: number
  orphans: { run_id: string; reason: string; unit?: string; synthetic?: boolean }[]
  empty: boolean
  source?: Rec | null
}

const STATE = {
  pass: { Icon: CircleCheck, color: 'var(--good-ink)', word: 'Pass' },
  fail: { Icon: CircleX, color: 'var(--critical-ink)', word: 'Fail' },
  'not checked': { Icon: CircleDashed, color: 'var(--ink-3)', word: 'Not checked' },
  inconsistent: { Icon: TriangleAlert, color: 'var(--warn-ink)', word: 'Inconsistent' },
} as const

function StateCell({ r }: { r: Row }) {
  const s = STATE[r.state]
  const title = r.state === 'inconsistent'
    ? `The record contradicts itself, so no verdict is given.${r.recomputed ? ` Recomputed from the measured value and the declared target: ${r.recomputed}.` : ''}`
    : r.state === 'not checked' ? 'Declared, but no result exists yet. Not a pass.' : undefined
  return (
    <span className="inline-flex items-center gap-1.5 font-medium whitespace-nowrap" style={{ color: s.color }} title={title}>
      <s.Icon size={15} aria-hidden />{s.word}
    </span>
  )
}

const Synthetic = () => (
  <span className="inline-flex items-center gap-1 rounded border border-line px-1 text-micro font-semibold tracking-[0.04em] text-ink-2"
    title="This record is a synthetic test fixture, not a measurement.">
    <FlaskConical size={11} aria-hidden />SYNTHETIC
  </span>
)

export default function ScorecardPage() {
  const [card, setCard] = useState<Card | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [open, setOpen] = useState<string | null>(null)

  const fetchCard = useCallback(() =>
    get<Card>('/api/scorecard/units')
      .then((c) => { setCard(c); setErr(null) })
      .catch((e) => setErr(String(e))), [])
  const refresh = () => { setBusy(true); fetchCard().finally(() => setBusy(false)) }
  useEffect(() => {
    fetchCard()
    const t = setInterval(fetchCard, 10_000)
    return () => clearInterval(t)
  }, [fetchCard])

  const synthetic = card?.synthetic_rows ?? 0
  const total = card?.totals.declared ?? 0

  return (
    <div>
      <PageHeader title={manifest.title} description={manifest.description} tracks={manifest.tracks}
        actions={<>
          {synthetic > 0
            ? <span className="flex items-center gap-1 rounded border border-line px-1.5 py-0.5 text-micro font-semibold tracking-[0.04em] text-ink-2" title="Some rows are synthetic fixtures"><FlaskConical size={11} />{synthetic === total ? 'SYNTHETIC' : 'PARTLY SYNTHETIC'}</span>
            : <RealChip muted={!card || !!err} title="Read from the persisted declared and result records. Nothing is simulated." />}
          <Button icon={<RefreshCw size={14} className={clsx(busy && 'animate-spin')} />} disabled={busy} onClick={refresh}>Refresh</Button>
        </>} />

      {err && (
        <div role="alert" className="mb-4 rounded-md border border-critical/60 bg-critical/10 p-3 text-cap text-ink-1">
          The scorecard could not be read: <span className="mono">{err}</span>.{card ? ' Showing the last data that loaded.' : ''}
        </div>
      )}
      {synthetic > 0 && (
        <div role="note" className="mb-4 flex items-start gap-2 rounded-md border border-line bg-s1 p-3 text-cap text-ink-2" data-testid="synthetic-banner">
          <FlaskConical size={15} className="mt-0.5 shrink-0" aria-hidden />
          <span>{synthetic === total ? 'Every row' : `${synthetic} of ${total} rows`} on this page {synthetic === 1 ? 'is a synthetic test fixture' : 'are synthetic test fixtures'}, not {synthetic === 1 ? 'a measurement' : 'measurements'}. Rows marked SYNTHETIC must not be quoted as results.</span>
        </div>
      )}

      {!card && !err && <div className="text-cap text-ink-3">Loading the scorecard…</div>}
      {card && <Summary card={card} />}
      {card && card.empty && (
        <Panel id="scorecard-empty" plane="verify" className="mt-4" title="No units yet">
          <EmptyState title="Nothing has been declared"
            body="A unit appears here when its declaration (target, item count, test-set hash) is written, before the run. Until then there is nothing to score, and nothing is filled in for show." />
        </Panel>
      )}
      {card && !card.empty && (
        <Panel id="scorecard-units" plane="verify" className="mt-4" title="Scored units"
          subtitle={`${total} declared · pass or fail is recomputed from measured value and declared target`} flush>
          <div className="overflow-x-auto">
            <table className="w-full text-cap" data-testid="scorecard-table">
              <thead>
                <tr className="border-b border-line text-left text-micro text-ink-3">
                  <th className="w-7" />
                  {['State', 'Unit', 'Declared target', 'Measured', 'Gain over baseline (95% interval)', 'Items', 'Checked (UTC)'].map((h) => <th key={h} className="px-3 py-2 font-medium whitespace-nowrap">{h}</th>)}
                </tr>
              </thead>
              {card.tracks.map((t) => (
                <tbody key={t.track} data-track={t.track}>
                  <tr className="bg-s2/70">
                    <th colSpan={8} scope="colgroup" className="px-3 py-1.5 text-left text-cap font-medium text-ink-1">
                      {t.track}
                      <span className="tnum ml-3 font-normal text-ink-2" data-testid="track-count">{t.passed} of {t.declared} unit{t.declared === 1 ? '' : 's'} passed</span>
                      <span className="ml-3 text-micro font-normal text-ink-3">
                        {[t.failed && `${t.failed} failed`, t.not_checked && `${t.not_checked} not checked`, t.inconsistent && `${t.inconsistent} inconsistent`].filter(Boolean).join(' · ')}
                      </span>
                    </th>
                  </tr>
                  {t.units.map((r) => (
                    <Fragment key={r.run_id}>
                      <UnitRow r={r} open={open === r.run_id} onToggle={() => setOpen(open === r.run_id ? null : r.run_id)} />
                      {open === r.run_id && (
                        <tr className="border-b border-line bg-page/40"><td colSpan={8} className="p-0"><Detail r={r} /></td></tr>
                      )}
                    </Fragment>
                  ))}
                </tbody>
              ))}
            </table>
          </div>
        </Panel>
      )}

      {card && card.orphans.length > 0 && (
        <Panel id="scorecard-orphans" plane="verify" className="mt-4" title="Results with no declaration" subtitle="not scored: nothing was promised before the run">
          <ul className="space-y-1 text-cap">
            {card.orphans.map((o) => (
              <li key={o.run_id} className="flex flex-wrap items-center gap-2 text-ink-2">
                <TriangleAlert size={13} style={{ color: 'var(--warn-ink)' }} aria-hidden />
                <span className="mono text-ink-1">{o.run_id}</span>{o.unit && <span>({o.unit})</span>}<span>{o.reason}</span>{o.synthetic && <Synthetic />}
              </li>
            ))}
          </ul>
        </Panel>
      )}
    </div>
  )
}

function Summary({ card }: { card: Card }) {
  const tiles: [string, ReactNode, string?][] = [
    ['Units declared', card.totals.declared],
    ['Passed', card.totals.passed],
    ['Failed', card.totals.failed],
    ['Not checked', card.totals.not_checked, 'declared, no result yet'],
    ['Inconsistent', card.totals.inconsistent, 'record contradicts itself'],
  ]
  return (
    <div className="grid grid-cols-2 gap-4 rounded-[10px] border border-line bg-s1 px-5 py-4 md:grid-cols-5" data-testid="scorecard-summary">
      {tiles.map(([label, v, sub]) => (
        <div key={label} className="min-w-0 border-l border-line pl-4 first:border-l-0 first:pl-0">
          <div className="text-micro text-ink-3">{label}</div>
          <div className="tnum mt-0.5 text-[20px] leading-7 font-semibold text-ink-1">{v}</div>
          {sub && <div className="truncate text-micro text-ink-3">{sub}</div>}
        </div>
      ))}
    </div>
  )
}

function UnitRow({ r, open, onToggle }: { r: Row; open: boolean; onToggle: () => void }) {
  const unit = r.target?.unit
  return (
    <tr onClick={onToggle} aria-expanded={open} data-state={stateAttr(r.state)} data-run={r.run_id} data-synthetic={r.synthetic ? 'true' : 'false'}
      className={clsx('cursor-pointer border-b border-line/60', open ? 'bg-s2' : 'hover:bg-s2/60')}>
      <td className="pl-3 text-ink-3">{open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}</td>
      <td className="px-3 py-2"><StateCell r={r} /></td>
      <td className="px-3 py-2">
        <div className="flex items-center gap-2"><span className="font-medium text-ink-1">{r.unit}</span>{r.synthetic && <Synthetic />}</div>
        <div className="mono text-micro text-ink-3">{r.run_id}</div>
      </td>
      <td className="px-3 whitespace-nowrap">
        <span className="tnum font-medium text-ink-1">{targetText(r.target)}</span>
        {r.target && <div className="text-micro text-ink-3">{r.target.metric}</div>}
      </td>
      <td className="px-3 whitespace-nowrap">
        {r.state === 'not checked'
          ? <span className="text-ink-3">not checked</span>
          : <span className={clsx('tnum font-medium', r.state === 'fail' && 'text-critical-ink', r.state === 'inconsistent' && 'text-warn-ink', r.state === 'pass' && 'text-ink-1')}>{value(r.measured, unit)}</span>}
        {r.state === 'inconsistent' && r.problems.length > 0 && <div className="max-w-[26ch] truncate text-micro text-warn-ink" title={r.problems.map((p) => p.detail).join('\n')}>{r.problems[0].detail}</div>}
      </td>
      <td className="px-3 whitespace-nowrap">
        {r.gain
          ? <span title={`Gain over baseline run ${r.gain.baseline_run_id}, interval from ${r.gain.paired_n} paired items`}>
              <span className="tnum font-medium text-ink-1">{gainText(r.gain)}</span>
              <div className="text-micro text-ink-3">vs <span className="mono">{r.gain.baseline_run_id}</span>, n = {r.gain.paired_n}</div>
            </span>
          : r.state === 'not checked' ? <span className="text-ink-3">—</span>
          : r.gain_missing ? <span className="text-warn-ink" title={`Baseline ${r.baseline_run_id} was declared but no gain was recorded`}>baseline declared, no gain recorded</span>
          : <span className="text-ink-3" title="This unit has no baseline">no baseline</span>}
      </td>
      <td className="tnum px-3 whitespace-nowrap text-ink-2">{r.n_items ?? '—'}{r.n_items != null && r.declared_n_items != null && r.n_items !== r.declared_n_items ? <span className="text-warn-ink"> of {r.declared_n_items}</span> : null}</td>
      <td className="mono px-3 whitespace-nowrap text-ink-3">{utc(r.checked_at)}</td>
    </tr>
  )
}

function Detail({ r }: { r: Row }) {
  const [d, setD] = useState<Rec | null>(null)
  const [failed, setFailed] = useState(false)
  useEffect(() => {
    let alive = true
    get(`/api/scorecard/run/${encodeURIComponent(r.run_id)}`).then((x) => alive && setD(x)).catch(() => alive && setFailed(true))
    return () => { alive = false }
  }, [r.run_id, r.checked_at, r.state])
  const cmd = d?.command ?? `python3 tools/show_record.py --run-id ${r.run_id}`
  return (
    <div className="grid grid-cols-12 gap-4 px-4 py-4">
      <div className="col-span-12 lg:col-span-5">
        {r.problems.length > 0 && (
          <div className="mb-3 rounded-md border p-3" style={{ borderColor: 'var(--warn)', background: 'color-mix(in oklab, var(--warn) 8%, transparent)' }}>
            <div className="mb-1 flex items-center gap-1.5 text-cap font-medium text-ink-1"><TriangleAlert size={14} style={{ color: 'var(--warn-ink)' }} />Why this is inconsistent</div>
            <ul className="list-disc space-y-1 pl-5 text-cap text-ink-2">{r.problems.map((p) => <li key={p.code}>{p.detail}</li>)}</ul>
          </div>
        )}
        <dl className="grid grid-cols-[130px_1fr] gap-x-3 gap-y-1.5 text-cap">
          <dt className="text-ink-3">Stored result says</dt><dd className="text-ink-1">{r.stored_passed == null ? '—' : r.stored_passed ? 'passed' : 'failed'}</dd>
          <dt className="text-ink-3">Recomputed here</dt><dd className="text-ink-1">{r.recomputed ?? 'cannot be recomputed'} <span className="text-ink-3">({value(r.measured, r.target?.unit)} {targetText(r.target)})</span></dd>
          <dt className="text-ink-3">Declared (UTC)</dt><dd className="mono text-ink-1">{utc(r.declared_at)}</dd>
          <dt className="text-ink-3">Checked (UTC)</dt><dd className="mono text-ink-1">{utc(r.checked_at)}</dd>
          <dt className="text-ink-3">Checker</dt><dd className="mono truncate text-ink-1" title={r.checker?.code_sha}>{r.checker?.script ?? '—'}</dd>
          {d && d.item_records != null && <><dt className="text-ink-3">Item records</dt><dd className="tnum text-ink-1">{d.item_records}{r.n_items != null && d.item_records !== r.n_items ? <span className="text-warn-ink"> (result says {r.n_items})</span> : null}</dd></>}
        </dl>
        <div className="mt-3">
          <div className="mb-1 text-micro text-ink-3">Show every record for this run</div>
          <div className="flex items-center gap-2 rounded-md border border-line bg-s2 px-3 py-2">
            <code className="mono min-w-0 flex-1 truncate text-[12px] text-ink-1" data-testid="show-record-command">{cmd}</code>
            <CopyText value={cmd} />
          </div>
        </div>
      </div>
      <div className="col-span-12 grid min-w-0 gap-3 md:grid-cols-2 lg:col-span-7">
        {failed && <div className="col-span-full text-cap text-critical-ink">The raw records could not be fetched.</div>}
        {!d && !failed && <div className="col-span-full text-cap text-ink-3">Fetching the run’s records…</div>}
        {d && <>
          <div className="min-w-0"><div className="mb-1 text-micro text-ink-3">declared/{r.run_id}.json</div><CodeBlock code={JSON.stringify(d.declared, null, 2)} maxHeight={240} /></div>
          <div className="min-w-0"><div className="mb-1 text-micro text-ink-3">results/{r.run_id}.json</div>
            {d.result ? <CodeBlock code={JSON.stringify(d.result, null, 2)} maxHeight={240} /> : <div className="rounded-md border border-line p-3 text-cap text-ink-3">No result file. This unit is declared and not checked.</div>}</div>
        </>}
      </div>
    </div>
  )
}

