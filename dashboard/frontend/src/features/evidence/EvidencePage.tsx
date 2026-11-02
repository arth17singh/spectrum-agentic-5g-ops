import clsx from 'clsx'
import {
  ChevronDown, ChevronRight, CircleCheck, CircleDashed, CircleX, Clock3, FileCheck2, FolderTree, KeyRound, Loader2,
  RefreshCw, ShieldCheck, ShieldX,
} from 'lucide-react'
import { Fragment, useEffect, useState, type ReactNode } from 'react'
import { ChainStrip, glyphFor, type BlockState } from '@/charts/ChainStrip'
import { api } from '@/core/api'
import { bytes, shortHash } from '@/core/format'
import { EMPTY, useLive } from '@/core/store'
import type { LedgerRecord, Rec } from '@/core/types'
import { toast } from '@/core/ui-store'
import { Button, CodeBlock, Hash, PageHeader, Panel, RealChip, SignatureBadge, StatusPill } from '@/ui'
import manifest from './manifest'
import { useBrowserVerify, type BrowserVerify } from './verify'

// Every value on this screen comes from the persisted evidence ledger (GET /api/evidence, evidence.*
// events). Nothing is simulated; checks that are not implemented yet say "not yet checked".

const CHECKS: { id: string; label: string; short: string; about: string }[] = [
  { id: 'declared', label: 'Declared', short: 'Decl', about: 'A valid declared/<run_id>.json exists and matches the schema.' },
  { id: 'declared_before_outputs', label: 'Declared before outputs', short: 'Before', about: 'The declaration’s source time is earlier than every output for the run.' },
  { id: 'test_set_hash', label: 'Test-set hash present', short: 'Hash', about: 'The declaration names the SHA-256 of the frozen test set.' },
  { id: 'grader_recorded', label: 'Grader recorded', short: 'Grader', about: 'The declaration names the grader.' },
  { id: 'unchanged', label: 'Unchanged since ingest', short: 'Intact', about: 'No artifact changed or disappeared after the ledger recorded it.' },
  { id: 'times_agree_aws', label: 'Times agree with AWS', short: 'AWS', about: 'Not yet checked: needs AWS-side timestamps (later work).' },
  { id: 'independent_grader', label: 'Independent grader', short: 'Indep', about: 'Not yet checked: needs a second grader run (later work).' },
]

const STATE: Record<string, { Icon: typeof CircleCheck; color: string; word: string }> = {
  pass: { Icon: CircleCheck, color: 'var(--good-ink)', word: 'Pass' },
  fail: { Icon: CircleX, color: 'var(--critical-ink)', word: 'Fail' },
  pending: { Icon: Clock3, color: 'var(--warn)', word: 'Waiting' },
  not_checked: { Icon: CircleDashed, color: 'var(--ink-3)', word: 'Not yet checked' },
}

/** Real wall-clock UTC, never the demo's simulated clock. */
const utc = (iso?: string | null, withDate = true) =>
  !iso ? '—' : withDate ? `${iso.slice(0, 10)} ${iso.slice(11, 19)}Z` : `${iso.slice(11, 19)}Z`

export default function EvidencePage() {
  const ev: Rec = useLive((s) => s.evidence)
  const st: Rec | undefined = ev.status
  const runs: Rec[] = ev.runs ?? EMPTY
  const bv = useBrowserVerify()
  const [open, setOpen] = useState<string | null>(null)
  const [scanning, setScanning] = useState(false)

  // a newer chain (or a reload that found a break) invalidates the browser verdict
  const records: number = st?.records ?? 0
  const state: string = st?.state ?? 'ok'
  useEffect(() => { if (bv.phase === 'ok' || bv.phase === 'broken') bv.reset() }, [records, state]) // eslint-disable-line react-hooks/exhaustive-deps

  const scan = () => {
    setScanning(true)
    api.evidenceScan()
      .then((r: Rec) => {
        const s = r.scan ?? {}
        toast({ severity: s.error || s.skipped ? 'warn' : 'notice', title: s.error ? `Scan failed: ${s.error}` : s.skipped ? `Scan skipped: ${s.skipped}` : `Evidence scan: ${s.new ?? 0} new, ${s.changed ?? 0} changed, ${s.missing ?? 0} missing` })
        if (r.evidence) useLive.setState({ evidence: r.evidence })
      })
      .catch((e) => toast({ severity: 'critical', title: String(e) }))
      .finally(() => setScanning(false))
  }

  if (!st) return <div className="text-cap text-ink-3">Loading the evidence ledger…</div>

  return (
    <div>
      <PageHeader title={manifest.title} description={manifest.description} tracks={manifest.tracks}
        actions={<>
          <RealChip title="Every number on this screen is read from the persisted evidence ledger. Nothing is simulated." />
          <Button icon={<RefreshCw size={14} className={clsx(scanning && 'animate-spin')} />} disabled={scanning} onClick={scan}>Scan now</Button>
          <Button variant="primary" icon={<ShieldCheck size={14} />} disabled={bv.phase === 'running' || records === 0} onClick={bv.run}>
            {bv.phase === 'running' ? 'Verifying…' : 'Verify in browser'}
          </Button>
        </>} />
      {state !== 'ok' && <ProblemBanner st={st} />}
      <StatusStrip st={st} />
      {bv.phase !== 'idle' && <BrowserResult bv={bv} fp={st.key?.fingerprint} />}

      <Panel id="runs" plane="prove" className="mt-4" title="Runs" subtitle={runs.length ? `${runs.length} run id${runs.length === 1 ? '' : 's'} in the ledger${ev.judged?.length ? `, ${ev.judged.length} listed in judging.json` : ''}` : 'none yet'}
        actions={<RealChip />} flush>
        {runs.length === 0 ? <EmptyRuns st={st} /> : (
          <div className="overflow-x-auto">
            <table className="w-full text-cap">
              <thead>
                <tr className="border-b border-line text-left text-micro text-ink-3">
                  <th className="w-8" />
                  {['Run id', 'Unit', 'Declared target', 'Model', 'Grader', 'Test set', 'Artifacts'].map((h) => <th key={h} className="px-3 py-2 font-medium whitespace-nowrap">{h}</th>)}
                  <th className="px-3 py-2 font-medium">Checks</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <Fragment key={r.run_id}>
                    <RunRow r={r} open={open === r.run_id} onToggle={() => setOpen(open === r.run_id ? null : r.run_id)} />
                    {open === r.run_id && (
                      <tr className="border-b border-line bg-page/40">
                        <td colSpan={9} className="p-0"><RunDetail runId={r.run_id} summary={r} bv={bv} records={records} fp={st.key?.fingerprint} /></td>
                      </tr>
                    )}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <div className="mt-4 grid grid-cols-12 gap-4">
        <div className="col-span-12 xl:col-span-7"><RecentRecords recent={ev.recent ?? EMPTY} /></div>
        <div className="col-span-12 xl:col-span-5"><ChecksLegend /></div>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ header strip

function StatusStrip({ st }: { st: Rec }) {
  const v: Rec = st.verify ?? {}
  const a: Rec | null = st.last_anchor
  const tile = (label: string, value: ReactNode, sub?: ReactNode) => (
    <div className="min-w-0 border-l border-line pl-4 first:border-l-0 first:pl-0">
      <div className="text-micro text-ink-3">{label}</div>
      <div className="mt-0.5 truncate text-[20px] leading-7 font-semibold text-ink-1">{value}</div>
      {sub && <div className="truncate text-micro text-ink-3">{sub}</div>}
    </div>
  )
  const chain = st.state === 'ok'
    ? <span className="flex items-center gap-1.5 text-good-ink"><ShieldCheck size={18} />Intact</span>
    : st.state === 'key_mismatch'
      ? <span className="flex items-center gap-1.5 text-critical-ink"><KeyRound size={18} />Key mismatch</span>
      : <span className="flex items-center gap-1.5 text-critical-ink"><ShieldX size={18} />{v.broken ? `Broken at #${v.broken.seq.toLocaleString()}` : 'Anchor broken'}</span>
  const src: Rec = st.source ?? {}
  const last: Rec | null = st.last_scan
  return (
    <div className="grid grid-cols-2 gap-4 rounded-[10px] border border-line bg-s1 px-5 py-4 md:grid-cols-5">
      {tile('Chain status', chain, <span title={`verified ${v.at ?? ''} in ${v.elapsed_ms ?? '—'} ms`}>checked with public_keys.json</span>)}
      {tile('Records', <span className="tnum">{(st.records ?? 0).toLocaleString()}</span>, 'signed, appended to ledger.jsonl')}
      {tile('Last anchor', a ? <span className="tnum">#{a.n} <span className="text-[14px] font-normal text-ink-2">{a.range[0]}–{a.range[1]}</span></span> : <span className="text-ink-3">None yet</span>,
        a ? <span className="mono">root {shortHash(a.root, 10, 6)}</span> : `first Merkle root at record ${st.anchor_every ?? 32}`)}
      {tile('Signing key', <span className="mono text-[15px]">{shortHash(st.key?.fingerprint?.replace('SHA256:', ''), 10, 6)}</span>,
        <span title={st.key?.public_key_hex}>Ed25519 · {st.key?.key_id}</span>)}
      {tile('Source', src.kind === 's3' ? 'S3, read-only' : 'Local folder',
        src.error ? <span className="text-critical-ink" title={src.error}>{src.error}</span>
          : <span title={`${src.uri}\nsource time: ${src.time_basis}`}>{last ? `scanned ${utc(last.at, false)}, every ${st.poll_s} s` : `polled every ${st.poll_s} s`}</span>)}
    </div>
  )
}

function ProblemBanner({ st }: { st: Rec }) {
  const b: Rec | null = st.verify?.broken
  return (
    <div className="mb-4 rounded-md border border-critical/60 bg-critical/10 p-4" role="alert">
      <div className="flex items-start gap-3">
        <ShieldX size={18} className="mt-0.5 shrink-0 text-critical-ink" />
        <div className="min-w-0 flex-1">
          <div className="text-body font-medium text-ink-1">
            {st.state === 'key_mismatch' ? 'Evidence ledger: signing key mismatch.' : `Evidence ledger BROKEN${b ? ` at #${b.seq.toLocaleString()}` : ''}:`}{' '}
            <span className="font-normal text-ink-2">{b?.reason ?? st.problem}</span>
          </div>
          <p className="mt-1 text-cap text-ink-2">Nothing has been repaired. New records are stopped until a person investigates; the files on disk are left exactly as found.</p>
          {b?.stored_hash && (
            <div className="mt-3 grid grid-cols-1 gap-x-6 gap-y-2 text-cap lg:grid-cols-2">
              <div><div className="text-micro text-ink-3">Stored hash (signed at write time)</div><div className="mono break-all text-ink-1">{b.stored_hash}</div></div>
              <div><div className="text-micro text-ink-3">Recomputed from the record as it is on disk</div><div className="mono break-all text-critical-ink">{b.recomputed_hash}</div></div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function BrowserResult({ bv, fp }: { bv: BrowserVerify; fp?: string }) {
  const k = bv.total ? bv.checked / bv.total : bv.phase === 'ok' ? 1 : 0
  return (
    <div className="mt-3 rounded-[10px] border border-line bg-s1 px-5 py-3">
      <div className="mb-1.5 flex flex-wrap items-baseline gap-x-3 gap-y-1 text-cap">
        {bv.phase === 'running' && <span className="flex items-center gap-1.5 font-medium text-ink-1"><Loader2 size={14} className="animate-spin" /> Verifying in this browser…</span>}
        {bv.phase === 'ok' && <span className="tnum flex items-center gap-1.5 font-medium text-good-ink"><ShieldCheck size={14} /> Verified {bv.total.toLocaleString()} / {bv.total.toLocaleString()} records in this browser in {bv.ms.toFixed(1)} ms</span>}
        {bv.phase === 'broken' && bv.broken && <span className="tnum flex items-center gap-1.5 font-medium text-critical-ink"><ShieldX size={14} /> BROKEN at #{bv.broken.seq.toLocaleString()}: {bv.broken.reason}</span>}
        {bv.phase === 'error' && <span className="font-medium text-critical-ink">Verification could not run: {bv.error}</span>}
        {(bv.phase === 'ok' || bv.phase === 'broken') && (
          <span className="text-micro text-ink-3">
            recomputed SHA-256 and links from genesis; {bv.signatures === 'verified' ? <>Ed25519 signatures checked with the published key <span className="mono">{shortHash(fp?.replace('SHA256:', ''), 8, 4)}</span></> : 'signatures NOT checked here (this browser has no WebCrypto Ed25519; the server checked them)'}; {bv.anchors} Merkle anchor{bv.anchors === 1 ? '' : 's'} recomputed
          </span>
        )}
      </div>
      <div className="h-1 overflow-hidden rounded-full bg-s3">
        <div className="h-full rounded-full transition-[width] duration-100" style={{ width: `${k * 100}%`, background: bv.phase === 'broken' || bv.phase === 'error' ? 'var(--critical)' : bv.phase === 'ok' ? 'var(--good)' : 'var(--release)' }} />
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ runs

function CheckIcons({ checks }: { checks: Rec }) {
  return (
    <span className="flex items-center gap-1">
      {CHECKS.map((c) => {
        const s = STATE[checks?.[c.id]?.state ?? 'not_checked']
        return (
          <span key={c.id} title={`${c.label}: ${s.word}\n${checks?.[c.id]?.detail ?? ''}`} className="grid size-[18px] place-items-center">
            <s.Icon size={14} style={{ color: s.color }} aria-label={`${c.label}: ${s.word}`} />
          </span>
        )
      })}
    </span>
  )
}

function RunRow({ r, open, onToggle }: { r: Rec; open: boolean; onToggle: () => void }) {
  const d: Rec | null = r.declaration
  const known = CHECKS.filter((c) => ['pass', 'fail'].includes(r.checks?.[c.id]?.state))
  const passed = known.filter((c) => r.checks[c.id].state === 'pass').length
  return (
    <tr onClick={onToggle} aria-expanded={open} className={clsx('cursor-pointer border-b border-line/60', open ? 'bg-s2' : 'hover:bg-s2/60')}>
      <td className="pl-3 text-ink-3">{open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}</td>
      <td className="px-3 py-2">
        <div className="mono whitespace-nowrap text-ink-1">{r.run_id}</div>
        <div className="mt-0.5 flex items-center gap-1.5 text-micro text-ink-3">
          {r.judged && <span className="rounded border border-prove/50 px-1 text-ink-2">judged</span>}
          {r.declaration_status === 'invalid' && <StatusPill status="critical" size="sm">invalid declaration</StatusPill>}
          {r.declaration_status === 'missing' && <StatusPill status="warning" size="sm">no declaration</StatusPill>}
          <span className="tnum">{r.records} ledger record{r.records === 1 ? '' : 's'}</span>
        </div>
      </td>
      <td className="px-3 whitespace-nowrap text-ink-2">{d?.unit ?? '—'}</td>
      <td className="px-3 whitespace-nowrap">{d ? <><span className="text-ink-2">{d.target.metric}</span> <span className="tnum font-medium text-ink-1">≥ {d.target.value}</span></> : <span className="text-ink-3">—</span>}</td>
      <td className="mono max-w-[180px] truncate px-3 text-ink-2" title={d?.model_id}>{d?.model_id ?? '—'}</td>
      <td className="mono max-w-[170px] truncate px-3 text-ink-2" title={d?.grader}>{d?.grader ?? '—'}</td>
      <td className="px-3 whitespace-nowrap">{d ? <Hash value={d.test_set_sha256} head={8} tail={4} /> : <span className="text-ink-3">—</span>}</td>
      <td className="tnum px-3 text-right text-ink-1">{r.artifacts}</td>
      <td className="px-3">
        <div className="flex items-center gap-2 whitespace-nowrap">
          <CheckIcons checks={r.checks} />
          <span className="tnum text-micro text-ink-3">{passed}/{known.length}</span>
        </div>
      </td>
    </tr>
  )
}

function RunDetail({ runId, summary, bv, records, fp }: { runId: string; summary: Rec; bv: ReturnType<typeof useBrowserVerify>; records: number; fp?: string }) {
  const [d, setD] = useState<Rec | null>(null)
  const [sel, setSel] = useState<number | null>(null)
  useEffect(() => {
    let alive = true
    api.evidenceRun(runId).then((x) => alive && setD(x)).catch(() => alive && setD({ records: [] }))
    return () => { alive = false }
  }, [runId, records])
  const recs: LedgerRecord[] = d?.records ?? EMPTY
  const states: Record<number, BlockState> = {}
  if (bv.phase === 'ok' || bv.phase === 'broken') for (const r of recs) states[r.seq] = bv.states[r.seq] ?? (bv.broken && r.seq > bv.broken.seq ? 'unverified' : 'idle')
  const picked = recs.find((r) => r.seq === sel) ?? null
  return (
    <div className="grid grid-cols-12 gap-4 px-4 py-4">
      <div className="col-span-12 2xl:col-span-4">
        <div className="mb-2 text-micro font-medium text-ink-3">Checks for this run</div>
        <ul className="grid grid-cols-1 gap-px overflow-hidden rounded-md border border-line bg-line md:grid-cols-2 2xl:grid-cols-1">
          {CHECKS.map((c) => {
            const k = summary.checks?.[c.id] ?? { state: 'not_checked' }
            const s = STATE[k.state] ?? STATE.not_checked
            return (
              <li key={c.id} className="flex min-w-0 items-start gap-2 bg-s1 px-3 py-2 md:last:col-span-2 2xl:last:col-span-1">
                <s.Icon size={14} className="mt-0.5 shrink-0" style={{ color: s.color }} aria-hidden />
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline justify-between gap-2 text-cap">
                    <span className="text-ink-1">{c.label}</span>
                    <span className="shrink-0 text-micro" style={{ color: s.color }}>{s.word}</span>
                  </div>
                  {k.detail && <div className="mono truncate text-micro text-ink-3" title={k.detail}>{k.detail}</div>}
                </div>
              </li>
            )
          })}
        </ul>
        {summary.declaration_status === 'invalid' && (
          <p className="mt-2 text-micro text-critical-ink">The declaration file was recorded as found, with its errors, so the mistake itself is on the record.</p>
        )}
      </div>
      <div className="col-span-12 min-w-0 2xl:col-span-8">
        <div className="mb-2 flex items-center gap-2">
          <span className="text-micro font-medium text-ink-3">Signed chain for this run</span>
          <span className="text-micro text-ink-3">{recs.length} record{recs.length === 1 ? '' : 's'}; links are checked against the full ledger</span>
          <Button size="sm" className="ml-auto" icon={<ShieldCheck size={13} />} disabled={bv.phase === 'running'} onClick={(e) => { e.stopPropagation(); bv.run() }}>
            {bv.phase === 'running' ? 'Verifying…' : 'Verify in browser'}
          </Button>
        </div>
        {!d ? <div className="text-cap text-ink-3">Fetching the run’s records…</div> : (
          <>
            <ChainStrip records={recs} states={states} scan={null} selected={sel} onSelect={setSel} follow={false} />
            <div className="mt-2 max-h-[260px] overflow-auto rounded-md border border-line">
              <table className="w-full text-cap">
                <thead className="sticky top-0 bg-s1">
                  <tr className="border-b border-line text-left text-micro text-ink-3">
                    {['#', 'Recorded (UTC)', 'Record', 'File', 'File SHA-256', 'Size', 'Record hash', 'Sig'].map((h) => <th key={h} className="px-2.5 py-1.5 font-medium whitespace-nowrap">{h}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {recs.map((r) => {
                    const G = glyphFor(r.type)
                    const p = r.payload
                    const bst = states[r.seq]
                    return (
                      <tr key={r.seq} onClick={() => setSel(r.seq === sel ? null : r.seq)} className={clsx('cursor-pointer border-b border-line/50', sel === r.seq ? 'bg-s3' : 'hover:bg-s2/60')}>
                        <td className="tnum mono px-2.5 py-1.5 text-ink-1">{r.seq}</td>
                        <td className="tnum mono px-2.5 whitespace-nowrap text-ink-3">{utc(r.ts, false)}</td>
                        <td className="px-2.5 whitespace-nowrap">
                          <span className="flex items-center gap-1.5 text-ink-2"><G size={12} className="text-ink-3" />{r.type.replace('evidence.', '')}{p.kind ? <span className="text-ink-3">· {p.kind}</span> : null}</span>
                          {p.status === 'invalid' && <StatusPill status="critical" size="sm">invalid</StatusPill>}
                        </td>
                        <td className="mono max-w-[240px] truncate px-2.5 text-ink-2" title={p.path}>{p.path.split('/').slice(-1)[0]}</td>
                        <td className="px-2.5 whitespace-nowrap"><Hash value={p.sha256 ?? p.last_sha256} head={6} tail={4} /></td>
                        <td className="tnum px-2.5 whitespace-nowrap text-ink-3">{p.size != null ? bytes(p.size) : '—'}</td>
                        <td className="px-2.5 whitespace-nowrap"><Hash value={r.hash} head={6} tail={4} /></td>
                        <td className="px-2.5">
                          {bst === 'fail' ? <StatusPill status="tampered" size="sm" /> : <SignatureBadge compact state={bst === 'ok' ? 'verified' : 'signed'} keyId={r.key_id} />}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            {picked && <RecordBody r={picked} fp={fp} />}
          </>
        )}
      </div>
    </div>
  )
}

function RecordBody({ r, fp }: { r: LedgerRecord; fp?: string }) {
  return (
    <div className="mt-3 grid grid-cols-1 gap-3 lg:grid-cols-2">
      <div className="min-w-0">
        <div className="mb-1 flex items-center justify-between text-micro text-ink-3"><span>Record #{r.seq} as stored</span><span className="mono">hash = SHA-256(canonical body ‖ prev_hash)</span></div>
        <CodeBlock code={JSON.stringify(r.payload, null, 2)} className="max-h-[180px]" />
      </div>
      <dl className="grid min-w-0 grid-cols-[96px_1fr] content-start gap-x-3 gap-y-1.5 text-cap">
        <dt className="text-ink-3">Time</dt><dd className="mono text-ink-1">{r.ts}</dd>
        <dt className="text-ink-3">Hash</dt><dd className="min-w-0"><Hash value={r.hash} full className="text-micro" /></dd>
        <dt className="text-ink-3">Prev hash</dt><dd className="min-w-0"><Hash value={r.prev_hash} full className="text-micro" /></dd>
        <dt className="text-ink-3">Signature</dt><dd className="min-w-0"><Hash value={r.sig} head={16} tail={8} /></dd>
        <dt className="text-ink-3">Key</dt><dd className="mono truncate text-ink-2">{r.key_id} · {shortHash(fp?.replace('SHA256:', ''), 8, 4)}</dd>
      </dl>
    </div>
  )
}

// ------------------------------------------------------------------ empty + side panels

function EmptyRuns({ st }: { st: Rec }) {
  const layout = [
    ['declared/<run_id>.json', 'the declaration: unit, target, model, grader, test-set SHA-256. Written before the run.'],
    ['outputs/<run_id>/…', 'per-item outputs from the run'],
    ['records/<run_id>/…, logs/<run_id>/…', 'run records and logs'],
    ['judging.json', 'the run ids handed in for judging (at most 20)'],
  ]
  return (
    <div className="grid gap-6 px-6 py-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
      <div className="flex items-start gap-4">
        <span className="grid size-10 shrink-0 place-items-center rounded-lg border border-line bg-s2 text-ink-2"><FileCheck2 size={20} /></span>
        <div className="min-w-0">
          <div className="text-title font-medium text-ink-1">No real runs yet</div>
          <p className="mt-1 max-w-[52ch] text-cap text-ink-2">
            When a run is declared, a row appears here for its run id. Every file is hashed and signed into the ledger as it
            arrives, so the screen can show what was declared, when, and whether anything changed afterwards. Nothing is
            filled in for show: this stays empty until real evidence exists.
          </p>
          <p className="mt-3 text-micro text-ink-3">
            Watching <span className="mono text-ink-2" title={st.source?.uri}>{st.source?.kind === 's3' ? st.source?.uri : 'evidence/'}</span> every {st.poll_s} s{st.last_scan ? `, last scan ${utc(st.last_scan.at, false)}` : ''}. Press Scan now after adding files.
          </p>
        </div>
      </div>
      <div className="min-w-0 rounded-md border border-line bg-s2/60">
        <div className="flex items-center gap-2 border-b border-line px-3 py-2 text-micro font-medium text-ink-3"><FolderTree size={13} /> What the ledger picks up</div>
        <dl className="grid grid-cols-[minmax(0,auto)_minmax(0,1fr)] gap-x-4 gap-y-2 px-3 py-3 text-cap">
          {layout.map(([k, v]) => (
            <Fragment key={k}>
              <dt className="mono whitespace-nowrap text-ink-1">{k}</dt>
              <dd className="text-ink-3">{v}</dd>
            </Fragment>
          ))}
        </dl>
      </div>
    </div>
  )
}

function RecentRecords({ recent }: { recent: LedgerRecord[] }) {
  const rows = [...recent].reverse().slice(0, 12)
  return (
    <Panel id="evidence-recent" plane="prove" title="Latest ledger records" subtitle="newest first, as persisted" actions={<RealChip />} flush className="h-full">
      {rows.length === 0 ? <div className="px-4 py-6 text-center text-cap text-ink-3">The ledger is empty. The first record is written when the first artifact is found.</div> : (
        <div className="max-h-[320px] overflow-auto">
          <table className="w-full text-cap">
            <thead className="sticky top-0 bg-s1">
              <tr className="border-b border-line text-left text-micro text-ink-3">
                {['#', 'Recorded (UTC)', 'Record', 'Path', 'Hash'].map((h) => <th key={h} className="px-3 py-1.5 font-medium whitespace-nowrap">{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const G = glyphFor(r.type)
                const warn = r.type === 'evidence.changed' || r.type === 'evidence.missing' || r.payload?.status === 'invalid'
                return (
                  <tr key={r.seq} className="border-b border-line/50">
                    <td className="tnum mono px-3 py-1.5 text-ink-1">{r.seq}</td>
                    <td className="tnum mono px-3 whitespace-nowrap text-ink-3">{utc(r.ts, false)}</td>
                    <td className="px-3 whitespace-nowrap"><span className={clsx('flex items-center gap-1.5', warn ? 'text-critical-ink' : 'text-ink-2')}><G size={12} />{r.type.replace('evidence.', '')}{r.payload?.status === 'invalid' ? ' (invalid)' : ''}</span></td>
                    <td className="mono max-w-[360px] truncate px-3 text-ink-2" title={r.payload?.path}>{r.payload?.path}</td>
                    <td className="mono px-3 text-ink-2">{shortHash(r.hash, 8, 4)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  )
}

function ChecksLegend() {
  return (
    <Panel id="evidence-checks" plane="prove" title="What each check means" subtitle="per run" className="h-full">
      <ul className="space-y-2">
        {CHECKS.map((c) => {
          const s = c.id === 'times_agree_aws' || c.id === 'independent_grader' ? STATE.not_checked : STATE.pass
          return (
            <li key={c.id} className="flex items-start gap-2 text-cap">
              <s.Icon size={14} className="mt-0.5 shrink-0" style={{ color: c.id === 'times_agree_aws' || c.id === 'independent_grader' ? 'var(--ink-3)' : 'var(--ink-2)' }} aria-hidden />
              <div><span className="text-ink-1">{c.label}.</span> <span className="text-ink-3">{c.about}</span></div>
            </li>
          )
        })}
      </ul>
      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 border-t border-line pt-3 text-micro text-ink-3">
        {Object.entries(STATE).map(([k, s]) => <span key={k} className="flex items-center gap-1"><s.Icon size={12} style={{ color: s.color }} />{s.word}</span>)}
      </div>
    </Panel>
  )
}
