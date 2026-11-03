import clsx from 'clsx'
import { CircleCheck, CircleDashed, CircleHelp, CircleX } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { get } from '@/core/api'
import type { Rec } from '@/core/types'
import { CodeBlock, Panel } from '@/ui'
import { JobLine, Verdict, type Job, type Overview } from '../scorecard/RealRuns'
import { num, targetText, utc } from '../scorecard/format'

// Run id -> declaration -> SageMaker job -> records -> result -> check_unit, read from the local mirror of
// the S3 evidence prefix (GET /api/realrun/*). check_unit is not run from the browser: the sync caches its
// --json output (make evidence-sync CHECK=1), and its time is shown.

const PART: Record<string, { Icon: typeof CircleCheck; color: string }> = {
  PASS: { Icon: CircleCheck, color: 'var(--good-ink)' },
  FAIL: { Icon: CircleX, color: 'var(--critical-ink)' },
  'CANNOT CHECK': { Icon: CircleHelp, color: 'var(--warn-ink)' },
  SKIP: { Icon: CircleDashed, color: 'var(--ink-3)' },
}
const PART_TITLE: Record<string, string> = {
  a: 'declared before the run', b: 'code in our repo', c: 'records keyed to the run, times agree with AWS',
  d: 'unchanged since the ledger recorded it', e: 'result recomputes from the items',
}

export default function RunExplorer() {
  const [params, setParams] = useSearchParams()
  const [ov, setOv] = useState<Overview | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const run = params.get('run') ?? ''
  useEffect(() => { get<Overview>('/api/realrun/overview').then(setOv).catch((e) => setErr(String(e))) }, [])
  const runs = (ov?.runs ?? []).slice().sort((a, b) => Number(b.headline) - Number(a.headline) || String(a.run_id).localeCompare(String(b.run_id)))
  const pick = (id: string) => setParams(id ? { run: id } : {}, { replace: true })

  return (
    <Panel id="run-explorer" plane="prove" className="mt-4" title="Run explorer"
      subtitle={ov?.source?.has_mirror ? `${runs.length} runs in the S3 evidence mirror, synced ${utc(ov.source.synced_at)}` : 'needs make evidence-sync'}>
      {err && <div className="text-cap text-critical-ink">Explorer unavailable: <span className="mono">{err}</span></div>}
      {ov && !ov.source?.has_mirror && <div className="text-cap text-ink-2">No local evidence mirror. Run <span className="mono">make evidence-sync</span> and start with <span className="mono">make demo-real</span>.</div>}
      {ov?.source?.has_mirror && (
        <>
          <label className="flex flex-wrap items-center gap-2 text-cap text-ink-2">
            Run id
            <select value={run} onChange={(e) => pick(e.target.value)} data-testid="run-select"
              className="mono max-w-full min-w-0 flex-1 rounded-md border border-line bg-s2 px-2 py-1 text-cap text-ink-1">
              <option value="">choose a run…</option>
              {runs.map((r) => <option key={r.run_id} value={r.run_id}>{r.run_id}{r.headline ? '' : '  (check / data)'}{r.result ? '' : '  (no result yet)'}</option>)}
            </select>
          </label>
          {run && <Chain key={run} id={run} />}
        </>
      )}
    </Panel>
  )
}

function Step({ n, title, children, ok }: { n: number; title: string; children: React.ReactNode; ok?: boolean | null }) {
  return (
    <li className="relative border-l border-line pb-4 pl-5 last:pb-0">
      <span className={clsx('absolute -left-[9px] top-0 flex h-[18px] w-[18px] items-center justify-center rounded-full border text-[10px] font-semibold',
        ok === true ? 'border-good-ink text-good-ink' : ok === false ? 'border-critical-ink text-critical-ink' : 'border-line text-ink-3')}
        style={{ background: 'var(--s1)' }}>{n}</span>
      <div className="mb-1 text-cap font-medium text-ink-1">{title}</div>
      <div className="min-w-0 space-y-1 text-micro text-ink-2">{children}</div>
    </li>
  )
}

function Chain({ id }: { id: string }) {
  const [d, setD] = useState<Rec | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    let alive = true
    get(`/api/realrun/run/${encodeURIComponent(id)}`).then((x) => alive && setD(x)).catch((e) => alive && setErr(String(e)))
    return () => { alive = false }
  }, [id])
  if (err) return <div className="mt-3 text-cap text-critical-ink">{err}</div>
  if (!d) return <div className="mt-3 text-cap text-ink-3">Reading {id}…</div>
  const jobs: Job[] = d.jobs ?? []
  const firstJob = jobs.map((j) => j.creation_time).filter(Boolean).sort()[0]
  const ac: Rec = d.attest_counts ?? {}
  const cu: Rec | null = d.check_unit
  return (
    <div className="mt-4 grid grid-cols-12 gap-4" data-testid="run-chain" data-run={id}>
      <ol className="col-span-12 lg:col-span-6">
        <Step n={1} title="Declaration" ok={d.declared ? d.declared_before_job ?? null : false}>
          {d.declared ? <>
            <div>unit <span className="mono text-ink-1">{d.unit}</span> · model <span className="mono text-ink-1">{d.model_id}</span> · target <span className="tnum text-ink-1">{targetText(d.target)}</span> {d.target?.metric} · {d.n_items_declared ?? '—'} items</div>
            <div>declared_at (our clock) <span className="mono">{utc(d.declared_at)}</span> · stored in S3 <span className="mono">{utc(d.declared_s3_time)}</span> (LastModified)</div>
            <div>{firstJob ? <>first job CreationTime <span className="mono">{utc(firstJob)}</span> (SageMaker): declaration {d.declared_before_job ? <b className="text-good-ink">stored before</b> : <b className="text-critical-ink">NOT before</b>} the job</> : 'no SageMaker job for this run'}</div>
          </> : <div className="text-critical-ink">No declaration: nothing was promised before this run, so it is not scored.</div>}
        </Step>
        <Step n={2} title={`SageMaker job${jobs.length === 1 ? '' : 's'}`} ok={jobs.length ? jobs.every((j) => j.status === 'Completed') : null}>
          {jobs.length === 0 ? <div>none recorded</div> : jobs.map((j) => (
            <div key={j.name}><div className="mono text-ink-1">{j.name}</div><JobLine j={j} />{j.failure_reason && <div className="text-critical-ink">{j.failure_reason}</div>}{j.cost_basis && <div className="text-ink-3">cost: {j.cost_basis}</div>}</div>
          ))}
        </Step>
        <Step n={3} title={`Records (${d.files_total} files)`} ok={ac.MISMATCH ? false : ac.match ? true : null}>
          <div>{Object.entries(d.record_kinds ?? {}).map(([k, v]) => <span key={k} className="mr-3"><span className="mono text-ink-1">{k}</span> × {String(v)}</span>)}</div>
          <div>sha256 vs the job log's attestation: <b className="text-good-ink">{ac.match ?? 0} match</b>{ac.MISMATCH ? <b className="text-critical-ink"> · {ac.MISMATCH} MISMATCH</b> : null}{ac['not attested'] ? <> · {ac['not attested']} not attested (the job record is written after the job, by the launcher)</> : null}{!Object.keys(ac).length && ' not available for this run'}</div>
          {d.attest_source && <div className="text-ink-3">{d.attest_source}, compared {utc(d.attest_checked_at)}</div>}
          <div className="max-h-48 overflow-auto rounded-md border border-line">
            <table className="w-full text-micro">
              <tbody>
                {(d.files ?? []).map((f: Rec) => (
                  <tr key={f.path} className="border-b border-line/50">
                    <td className="mono max-w-[260px] truncate px-2 py-0.5 text-ink-2" title={f.path}>{f.path}</td>
                    <td className="mono px-2 text-ink-3" title={f.sha256}>{String(f.sha256).slice(0, 10)}…</td>
                    <td className="mono px-2 whitespace-nowrap text-ink-3">{utc(f.s3_time).slice(11)}</td>
                    <td className={clsx('px-2', f.attest === 'match' ? 'text-good-ink' : f.attest === 'MISMATCH' ? 'text-critical-ink' : 'text-ink-3')}>{f.attest ?? ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {d.files_total > (d.files ?? []).length && <div className="text-ink-3">showing {(d.files ?? []).length} of {d.files_total}; every file was checked</div>}
        </Step>
        <Step n={4} title="Result" ok={d.result ? d.result.recomputed === 'pass' : null}>
          {d.result ? <>
            <div className="flex flex-wrap items-baseline gap-2"><Verdict s={d.result.recomputed ?? 'inconsistent'} /><span className="tnum text-cap text-ink-1">{num(d.result.value)} {d.result.metric}</span><span>n = {d.result.n_items}</span></div>
            <div>checked_at (our clock) <span className="mono">{utc(d.result.checked_at)}</span> · stored in S3 <span className="mono">{utc(d.result.s3_time)}</span></div>
            {d.result.stored_passed !== (d.result.recomputed === 'pass') && d.result.recomputed && <div className="text-warn-ink">stored passed={String(d.result.stored_passed)} disagrees with the recomputed verdict</div>}
          </> : <div>No result yet.</div>}
        </Step>
        <Step n={5} title="Verify: check_unit parts a–e" ok={cu ? cu.exit_code === 0 : null}>
          {cu ? <>
            <div>exit {cu.exit_code}: {cu.verdict} · cached by the sync {utc(cu.checked_at)} ({cu.elapsed_s} s, read-only AWS)</div>
            <ul className="space-y-1" data-testid="check-parts">
              {(cu.parts ?? []).map((p: Rec) => {
                const s = PART[p.state] ?? PART.SKIP
                return (
                  <li key={p.id}>
                    <div className="flex items-center gap-1.5 text-ink-1"><s.Icon size={13} style={{ color: s.color }} aria-hidden /><b>[{p.id}]</b> {p.title ?? PART_TITLE[p.id]} <span style={{ color: s.color }}>{p.state}</span></div>
                    {(p.lines ?? []).slice(0, p.state === 'PASS' ? 2 : 6).map((l: string, i: number) => <div key={i} className="mono pl-5 text-[11px] break-all text-ink-3">{l}</div>)}
                  </li>
                )
              })}
            </ul>
            {cu.window && <div className="text-ink-3">AWS window {utc(cu.window[0])} → {utc(cu.window[1])}</div>}
          </> : <div>Not yet verified: no cached check_unit output for this run. Run <span className="mono">make evidence-sync CHECK=1 RUN={id}</span> (or <span className="mono">python3 tools/check_unit.py {id} --s3 …</span>).</div>}
        </Step>
      </ol>
      <div className="col-span-12 grid min-w-0 gap-3 lg:col-span-6">
        <div className="min-w-0"><div className="mb-1 text-micro text-ink-3">declared/{id}.json</div>{d.declaration ? <CodeBlock code={JSON.stringify(d.declaration, null, 2)} maxHeight={260} /> : <div className="text-micro text-ink-3">absent</div>}</div>
        <div className="min-w-0"><div className="mb-1 text-micro text-ink-3">results/{id}.json</div>{d.result_doc ? <CodeBlock code={JSON.stringify(d.result_doc, null, 2)} maxHeight={260} /> : <div className="text-micro text-ink-3">absent</div>}</div>
        {Object.entries(d.record_samples ?? {}).slice(0, 2).map(([k, v]) => (
          <div key={k} className="min-w-0"><div className="mb-1 text-micro text-ink-3">records/{id}/{k}/ (first record)</div><CodeBlock code={JSON.stringify((v as Rec[])[0], null, 2)} maxHeight={200} /></div>
        ))}
      </div>
    </div>
  )
}
