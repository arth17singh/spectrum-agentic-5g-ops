import clsx from 'clsx'
import { FileText, Radio, ShieldCheck, ShieldX } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { ChainStrip } from '@/charts/ChainStrip'
import type { Proof } from '@/charts/MerkleTree'
import { api } from '@/core/api'
import { useOnBeat } from '@/core/director'
import { Gate, useFlag } from '@/core/flags'
import { clockTime, relative, shortHash, simOf } from '@/core/format'
import { EMPTY, useLive } from '@/core/store'
import type { LedgerRecord, Rec } from '@/core/types'
import { useSimNow } from '@/core/ui-store'
import { Button, Hash, PageHeader, Panel, StatusPill } from '@/ui'
import manifest from './manifest'
import { AnchorsTable, ProofPanel, RecordDetail, RecordsTable, TamperPanel } from './panels'
import { ProvenancePanel } from './Provenance'
import { useVerify } from './verify'

const STRIP = 36

export default function AuditPage() {
  const records: LedgerRecord[] = useLive((s) => s.ledger.records ?? EMPTY)
  const height: number = useLive((s) => s.ledger.height ?? 0)
  const tampered: number[] = useLive((s) => s.ledger.tampered ?? EMPTY)
  const visible = useMemo(() => records.slice(-STRIP), [records])
  const getVisible = useCallback(() => useLive.getState().ledger.records?.slice(-STRIP) ?? [], [])
  const v = useVerify(getVisible)
  const [selected, setSelected] = useState<number | null>(null)
  const [proof, setProof] = useState<Proof | null>(null)
  const dossierOn = useFlag('audit.dossier')

  const lastAnchorEnd: number | undefined = useLive((s) => s.ledger.anchors?.[s.ledger.anchors.length - 1]?.range?.[1])
  // default to the newest anchored record, so the inclusion proof is complete out of the box
  const sel = selected ?? lastAnchorEnd ?? records[records.length - 1]?.seq ?? null
  useEffect(() => {
    if (!sel) return
    let alive = true
    setProof(null)
    api.proof(sel).then((p) => alive && setProof(p)).catch(() => {})
    return () => { alive = false }
  }, [sel, tampered.length, Math.floor(height / 32)])

  // beat 12: the backend tampers on arrival; run the verification so the break is shown
  useOnBeat(12, () => { setTimeout(() => v.run(), 1200) })

  // a new tamper or restore invalidates the previous verdict
  useEffect(() => { if (v.phase === 'ok' || v.phase === 'broken') v.reset() }, [tampered.length]) // eslint-disable-line react-hooks/exhaustive-deps

  const focusMode = v.focus != null
  const stripRecords = focusMode ? v.focus! : visible
  const brokenSeq: number | undefined = v.result?.broken?.seq

  const openDossier = () => {
    const inc = useLive.getState().swarm.incident?.id ?? 'INC-2040'
    window.open(`/dossier/${inc}`, '_blank', 'noopener')
  }

  return (
    <div>
      <PageHeader title={manifest.title} description={manifest.description} tracks={manifest.tracks}
        actions={<>
          <Button variant="primary" icon={<ShieldCheck size={14} />} disabled={v.phase === 'scanning'} onClick={v.run}>
            {v.phase === 'scanning' ? 'Verifying…' : 'Verify chain'}
          </Button>
          {dossierOn && <Button icon={<FileText size={14} />} onClick={openDossier}>Incident dossier</Button>}
        </>} />
      <LedgerStatus phase={v.phase} />

      <Panel id="chain" plane="prove" className="mt-4" title="Hash chain" subtitle={focusMode ? `around the break at #${brokenSeq?.toLocaleString()}` : 'newest on the right'}
        source={{ aws: 'Amazon DynamoDB + KMS', mock: 'In-memory store, real SHA-256 + Ed25519' }}
        actions={focusMode ? <Button size="sm" variant="ghost" icon={<Radio size={13} />} onClick={v.reset}>Back to live</Button> : (
          <span className="flex items-center gap-1.5 text-micro text-ink-3"><span className="live-dot size-1.5 rounded-full bg-good text-good" /> appending live</span>
        )}>
        <VerifyBar v={v} />
        <ChainStrip records={stripRecords} states={v.phase === 'idle' ? {} : v.states} scan={v.scan} selected={focusMode ? brokenSeq : sel}
          onSelect={setSelected} follow={!focusMode && v.phase !== 'scanning'} />
        {v.phase === 'broken' && v.result?.broken && <BreakBanner v={v} onShowProof={() => setSelected(brokenSeq!)} />}
      </Panel>

      <div className="mt-4 grid grid-cols-12 gap-4">
        <div className="col-span-12 xl:col-span-7"><RecordsTable selected={sel} onSelect={setSelected} /></div>
        <div className="col-span-12 xl:col-span-5"><RecordDetail seq={sel} proof={proof} /></div>
        <div className="col-span-12 xl:col-span-7"><AnchorsTable onPick={setSelected} /></div>
        <div className="col-span-12 xl:col-span-5"><ProofPanel seq={sel} proof={proof} /></div>
        <Gate flag="audit.provenance"><div className="col-span-12"><ProvenancePanel /></div></Gate>
        <Gate flag="demo.tamper"><div className="col-span-12 xl:col-span-7"><TamperPanel onVerify={v.run} /></div></Gate>
        <Gate flag="audit.dossier"><div className="col-span-12 xl:col-span-5"><DossierCard onOpen={openDossier} /></div></Gate>
      </div>
    </div>
  )
}

function LedgerStatus({ phase }: { phase: string }) {
  const l = useLive((s) => s.ledger)
  const now = useSimNow(1)
  const anchors: Rec[] = l.anchors ?? EMPTY
  const last = anchors[anchors.length - 1]
  const lv = l.last_verify
  const tile = (label: string, value: React.ReactNode, sub?: React.ReactNode) => (
    <div className="min-w-0 border-l border-line pl-4 first:border-l-0 first:pl-0">
      <div className="text-micro text-ink-3">{label}</div>
      <div className="mt-0.5 truncate text-[20px] leading-7 font-semibold text-ink-1">{value}</div>
      {sub && <div className="truncate text-micro text-ink-3">{sub}</div>}
    </div>
  )
  return (
    <div className="grid grid-cols-2 gap-4 rounded-[10px] border border-line bg-s1 px-5 py-4 md:grid-cols-5">
      {tile('Ledger height', <span className="tnum">{(l.height ?? 0).toLocaleString()}</span>, 'records, every one signed')}
      {tile('Chain head', <span className="mono text-[16px]">{shortHash(l.head, 10, 6)}</span>, <Hash value={l.head} head={20} tail={8} />)}
      {tile('Anchors', <span className="tnum">{(l.anchors_total ?? anchors.length).toLocaleString()}</span>, last ? `last ${relative(now, simOf(last.written_at))}, Object Lock COMPLIANCE` : 'none yet')}
      {tile('Last verification', lv ? (
        <span className={clsx('flex items-center gap-1.5', lv.ok ? 'text-good-ink' : 'text-critical-ink')}>
          {lv.ok ? <ShieldCheck size={18} /> : <ShieldX size={18} />}{lv.ok ? 'Intact' : `Broken at #${lv.broken?.seq?.toLocaleString()}`}
        </span>
      ) : <span className="text-ink-3">{phase === 'scanning' ? 'Running…' : 'Not run yet'}</span>, lv ? `${lv.total?.toLocaleString()} records in ${lv.elapsed_ms} ms, ${clockTime(lv.at, false)}` : 'press Verify chain')}
      {tile('Algorithms', <span className="text-[16px]">SHA-256, Ed25519</span>, 'Merkle roots every 32 records')}
    </div>
  )
}

function VerifyBar({ v }: { v: ReturnType<typeof useVerify> }) {
  if (v.phase === 'idle') {
    return <p className="mb-2 text-micro text-ink-3">Each block commits to the hash of the one before it. Changing any field of any record breaks every link after it.</p>
  }
  const k = v.total ? v.progress / v.total : 0
  return (
    <div className="mb-3">
      <div className="mb-1.5 flex items-baseline gap-3 text-cap">
        {v.phase === 'scanning' && <span className="tnum font-medium text-ink-1">Verifying {v.progress.toLocaleString()} / {v.total.toLocaleString()}</span>}
        {v.phase === 'ok' && <span className="tnum flex items-center gap-1.5 font-medium text-good-ink"><ShieldCheck size={14} /> Verified {v.total.toLocaleString()} / {v.total.toLocaleString()} in {((v.result?.elapsed_ms ?? 0) / 1000).toFixed(2)} s</span>}
        {v.phase === 'broken' && <span className="tnum flex items-center gap-1.5 font-medium text-critical-ink"><ShieldX size={14} /> Stopped at #{v.progress.toLocaleString()} of {v.total.toLocaleString()}</span>}
        <span className="text-micro text-ink-3">
          server walked hash links, signatures and {v.result?.anchors_checked ?? '…'} anchors; this browser recomputed {v.clientChecked} SHA-256 hashes in {v.clientMs.toFixed(1)} ms
        </span>
      </div>
      <div className="h-1 overflow-hidden rounded-full bg-s3">
        <div className="h-full rounded-full transition-[width] duration-100" style={{ width: `${k * 100}%`, background: v.phase === 'broken' ? 'var(--critical)' : v.phase === 'ok' ? 'var(--good)' : 'var(--release)' }} />
      </div>
    </div>
  )
}

function BreakBanner({ v, onShowProof }: { v: ReturnType<typeof useVerify>; onShowProof: () => void }) {
  const b = v.result!.broken
  const anchor = b.anchor
  return (
    <div className="mt-3 rounded-md border border-critical/60 bg-critical/10 p-4">
      <div className="flex items-start gap-3">
        <ShieldX size={18} className="mt-0.5 shrink-0 text-critical-ink" />
        <div className="min-w-0 flex-1">
          <div className="text-body font-medium text-ink-1">
            Chain broken at #{b.seq.toLocaleString()}: payload modified after signing.
            {anchor && <> Anchor #{anchor.n} in S3 Object Lock still holds the original root.</>}
          </div>
          <div className="mt-3 grid grid-cols-1 gap-x-6 gap-y-2 text-cap lg:grid-cols-2">
            <div>
              <div className="text-micro text-ink-3">Stored hash (signed at write time)</div>
              <div className="mono break-all text-ink-1">{b.stored_hash}</div>
            </div>
            <div>
              <div className="text-micro text-ink-3">Recomputed from current contents{v.clientRecomputed ? ' (this browser)' : ''}</div>
              <div className="mono break-all text-critical-ink">{v.clientRecomputed ?? b.recomputed_hash}</div>
            </div>
          </div>
          {b.diff?.length > 0 && (
            <div className="mt-3">
              <div className="mb-1 text-micro text-ink-3">What changed</div>
              <table className="text-cap">
                <tbody>
                  {b.diff.map((d: Rec) => (
                    <tr key={d.path}>
                      <td className="mono pr-4 text-ink-2">payload.{d.path}</td>
                      <td className="mono pr-2 text-good-ink line-through decoration-1">{JSON.stringify(d.before)}</td>
                      <td className="pr-2 text-ink-3">→</td>
                      <td className="mono rounded bg-critical/20 px-1 text-critical-ink">{JSON.stringify(d.after)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <StatusPill status={b.checks.hash_matches ? 'verified' : 'tampered'} size="sm">hash {b.checks.hash_matches ? 'matches' : 'mismatch'}</StatusPill>
            <StatusPill status={b.checks.link_matches ? 'verified' : 'tampered'} size="sm">link {b.checks.link_matches ? 'intact' : 'broken'}</StatusPill>
            <StatusPill status={b.checks.signature_valid ? 'verified' : 'tampered'} size="sm">signature over stored hash {b.checks.signature_valid ? 'valid' : 'invalid'}</StatusPill>
            <Button size="sm" className="ml-auto" onClick={onShowProof}>Show inclusion proof</Button>
          </div>
        </div>
      </div>
    </div>
  )
}

function DossierCard({ onOpen }: { onOpen: () => void }) {
  const inc = useLive((s) => s.swarm.incident)
  return (
    <Panel id="dossier" plane="prove" title="Incident dossier" subtitle="printable, signed, ledgered">
      <p className="text-cap text-ink-2">
        A regulator-ready report for {inc ? <span className="font-medium text-ink-1">{inc.id}, {inc.title.toLowerCase()}</span> : 'the latest incident'}:
        timeline, every decision and approval with its signature, policy and model versions, the ledger range with anchor proofs,
        and a verification result. Generating it hashes the report and writes that hash to the ledger.
      </p>
      <Button className="mt-3" icon={<FileText size={14} />} onClick={onOpen}>Generate and open</Button>
    </Panel>
  )
}
