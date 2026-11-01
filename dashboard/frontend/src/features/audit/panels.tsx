import clsx from 'clsx'
import { ArrowUpRight, FlaskConical, Loader2, ShieldCheck, ShieldX } from 'lucide-react'
import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { glyphFor } from '@/charts/ChainStrip'
import { MerkleTree, type Proof } from '@/charts/MerkleTree'
import { api } from '@/core/api'
import { canonical, recordHash } from '@/core/canonical'
import { clockTime, relative, shortHash, simOf } from '@/core/format'
import { PLANES, planeColor } from '@/core/palette'
import { EMPTY, useLive } from '@/core/store'
import type { LedgerRecord, Rec } from '@/core/types'
import { toast, useSimNow, useUi } from '@/core/ui-store'
import { Button, CodeBlock, EmptyState, Hash, KV, Panel, SignatureBadge, StatusPill } from '@/ui'

const NONE: Record<string, number> = {}
const SRC_LEDGER = { aws: 'Amazon DynamoDB + KMS', mock: 'In-memory store, real Ed25519' }
const SRC_ANCHOR = { aws: 'Amazon S3 Object Lock', mock: 'Simulated bucket, real Merkle roots' }

// ------------------------------------------------------------------ records table

export function RecordsTable({ selected, onSelect }: { selected: number | null; onSelect: (seq: number) => void }) {
  const records: LedgerRecord[] = useLive((s) => s.ledger.records ?? EMPTY)
  const typeCounts: Record<string, number> = useLive((s) => s.ledger.type_counts ?? NONE)
  const tampered: number[] = useLive((s) => s.ledger.tampered ?? EMPTY)
  const [type, setType] = useState('')
  const [plane, setPlane] = useState('')
  const [actor, setActor] = useState('')
  const rows = useMemo(() => {
    const a = actor.trim().toLowerCase()
    const out: LedgerRecord[] = []
    for (let i = records.length - 1; i >= 0 && out.length < 250; i--) {
      const r = records[i]
      if (type && r.type !== type) continue
      if (plane && r.plane !== plane) continue
      if (a && !r.actor.toLowerCase().includes(a)) continue
      out.push(r)
    }
    return out
  }, [records, type, plane, actor])
  const types = Object.entries(typeCounts).sort((x, y) => y[1] - x[1])
  return (
    <Panel id="records" plane="prove" title="Records" subtitle={`${records.length.toLocaleString()} most recent in this console`} source={SRC_LEDGER} flush className="h-full">
      <div className="flex flex-wrap items-center gap-2 border-b border-line px-4 py-2">
        <select value={type} onChange={(e) => setType(e.target.value)} aria-label="Record type" className="h-7 rounded-md border border-line bg-s2 px-2 text-cap text-ink-1">
          <option value="">All types</option>
          {types.map(([t, n]) => <option key={t} value={t}>{t} ({n.toLocaleString()})</option>)}
        </select>
        <select value={plane} onChange={(e) => setPlane(e.target.value)} aria-label="Plane" className="h-7 rounded-md border border-line bg-s2 px-2 text-cap text-ink-1">
          <option value="">All planes</option>
          {PLANES.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
        </select>
        <input value={actor} onChange={(e) => setActor(e.target.value)} placeholder="Actor, e.g. coordinator" className="h-7 w-44 rounded-md border border-line bg-s2 px-2 text-cap text-ink-1 placeholder:text-ink-3" />
        <span className="tnum ml-auto text-micro text-ink-3">{rows.length} shown</span>
      </div>
      <div className="max-h-[360px] overflow-auto">
        <table className="w-full text-cap">
          <thead className="sticky top-0 bg-s1">
            <tr className="border-b border-line text-left text-micro text-ink-3">
              {['#', 'Time', 'Type', 'Actor', 'Hash', 'Signature'].map((h) => <th key={h} className="px-3 py-1.5 font-medium">{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const G = glyphFor(r.type)
              const bad = tampered.includes(r.seq)
              return (
                <tr key={r.seq} onClick={() => onSelect(r.seq)}
                  className={clsx('cursor-pointer border-b border-line/50', selected === r.seq ? 'bg-s3' : 'hover:bg-s2/60')}>
                  <td className="tnum mono px-3 py-1.5 text-ink-1">
                    <span className="flex items-center gap-1.5"><span className="size-1.5 rounded-full" style={{ background: planeColor(r.plane) }} />{r.seq.toLocaleString()}</span>
                  </td>
                  <td className="tnum mono px-3 text-ink-3">{clockTime(r.ts)}</td>
                  <td className="px-3"><span className="flex items-center gap-1.5 text-ink-2"><G size={12} className="text-ink-3" />{r.type}</span></td>
                  <td className="mono max-w-[140px] truncate px-3 text-ink-2">{r.actor}</td>
                  <td className="mono px-3 text-ink-2">{shortHash(r.hash, 8, 4)}</td>
                  <td className="px-3">{bad ? <StatusPill status="tampered" size="sm" /> : <SignatureBadge compact keyId={r.key_id} />}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ record detail

function Step({ ok, n, label, detail }: { ok: boolean | null; n: number; label: string; detail?: ReactNode }) {
  return (
    <li className="flex items-start gap-2 py-1.5">
      {ok == null ? <Loader2 size={14} className="mt-0.5 animate-spin text-ink-3" /> : ok ? <ShieldCheck size={14} className="mt-0.5 text-good-ink" /> : <ShieldX size={14} className="mt-0.5 text-critical-ink" />}
      <div className="min-w-0">
        <div className="text-cap text-ink-1"><span className="tnum text-ink-3">{n}.</span> {label}</div>
        {detail && <div className="mono truncate text-micro text-ink-3">{detail}</div>}
      </div>
    </li>
  )
}

export function RecordDetail({ seq, proof }: { seq: number | null; proof: Proof | null }) {
  const [d, setD] = useState<Rec | null>(null)
  const [client, setClient] = useState<string | null>(null)
  const inspect = useUi((s) => s.inspect)
  const height = useLive((s) => s.ledger.height)
  const tampered: number[] = useLive((s) => s.ledger.tampered ?? EMPTY)
  useEffect(() => {
    if (!seq) return
    let alive = true
    setD(null); setClient(null)
    api.record(seq).then(async (x) => { if (!alive) return; setD(x); setClient(await recordHash(x.record)) }).catch(() => {})
    return () => { alive = false }
  }, [seq, tampered.length])
  if (!seq) return <Panel title="Record" plane="prove"><EmptyState title="Select a record" body="Pick a block in the chain or a row in the table to see its proof." /></Panel>
  const r: LedgerRecord | undefined = d?.record
  const v = d?.verification
  const pk = d?.public_key ?? {}
  const body = r ? canonical({ seq: r.seq, ts: r.ts, type: r.type, actor: r.actor, plane: r.plane, payload: r.payload, key_id: r.key_id }) : ''
  return (
    <Panel id="record" plane="prove" title={`Record #${seq.toLocaleString()}`} subtitle={r?.type} className="h-full"
      actions={<>
        {v && <StatusPill status={v.ok ? 'verified' : 'tampered'} size="sm" />}
        <button type="button" aria-label="Open in inspector" title="Open in inspector" onClick={() => inspect({ kind: 'ledger', id: String(seq) })} className="grid size-7 place-items-center rounded-md text-ink-3 hover:bg-s2 hover:text-ink-1"><ArrowUpRight size={15} /></button>
      </>}>
      {!r ? <div className="text-cap text-ink-3">Fetching #{seq} from the datastore…</div> : (
        <div className="space-y-3">
          <ol className="rounded-md border border-line bg-s2 px-3 py-0.5">
            <Step n={1} ok={client == null ? null : client === r.hash} label="SHA-256 recomputed in this browser" detail={client ?? '…'} />
            <Step n={2} ok={v.checks.hash_matches} label="Equals the stored hash" detail={r.hash} />
            <Step n={3} ok={v.checks.link_matches} label={`Links to #${(seq - 1).toLocaleString()}`} detail={`prev ${r.prev_hash}`} />
            <Step n={4} ok={v.checks.signature_valid} label={`${r.alg} signature verifies with ${r.key_id.replace('kms:spectrum/', '')}`} detail={pk.fingerprint} />
            <Step n={5} ok={proof ? (proof.anchored ? !!proof.matches : null) : null}
              label={proof?.anchored ? `Included in anchor #${proof.anchor!.n} in S3 Object Lock` : proof ? `Anchored at #${proof.pending_until?.toLocaleString()} (${Math.max(0, (proof.pending_until ?? 0) - (height ?? 0))} records to go)` : 'Checking inclusion…'}
              detail={proof?.anchor?.root} />
          </ol>
          <div>
            <div className="mb-1 flex items-center justify-between text-micro text-ink-3">
              <span>Canonical body (sorted keys, no whitespace)</span><span className="mono">hash = SHA-256(body ‖ prev_hash)</span>
            </div>
            <CodeBlock code={body} lang="text" className="max-h-[96px] break-all whitespace-pre-wrap" />
          </div>
          <KV rows={[
            ['Actor', <span className="mono">{r.actor}</span>],
            ['Time', <span className="mono">{r.ts}</span>],
            ['Signing key', <span className="mono break-all">{r.key_id}</span>],
            ['Public key', <Hash value={pk.public_key_hex} head={12} tail={8} />],
            ['Signature', <Hash value={r.sig} head={12} tail={8} />],
          ]} />
        </div>
      )}
    </Panel>
  )
}

// ------------------------------------------------------------------ anchors + proof

export function AnchorsTable({ onPick }: { onPick: (seq: number) => void }) {
  const anchors: Rec[] = useLive((s) => s.ledger.anchors ?? EMPTY)
  const total = useLive((s) => s.ledger.anchors_total ?? anchors.length)
  const now = useSimNow(1)
  const rows = [...anchors].reverse().slice(0, 40)
  return (
    <Panel id="anchors" plane="prove" title="Merkle anchors" subtitle={`${(total ?? 0).toLocaleString()} roots, one per 32 records`} source={SRC_ANCHOR} flush className="h-full">
      <div className="max-h-[392px] overflow-auto">
        <table className="w-full text-cap">
          <thead className="sticky top-0 bg-s1">
            <tr className="border-b border-line text-left text-micro text-ink-3">
              {['Anchor', 'Records', 'Merkle root', 'Object', 'Version', 'Lock', 'Retain until', 'Written'].map((h) => <th key={h} className="px-2.5 py-1.5 font-medium whitespace-nowrap">{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((a) => (
              <tr key={a.n} onClick={() => onPick(a.range[1])} className="cursor-pointer border-b border-line/50 hover:bg-s2/60">
                <td className="tnum mono px-2.5 py-1.5 text-ink-1">#{a.n}</td>
                <td className="tnum mono px-2.5 whitespace-nowrap text-ink-2">{a.range[0].toLocaleString()}–{a.range[1].toLocaleString()}</td>
                <td className="mono px-2.5 whitespace-nowrap text-ink-1">{shortHash(a.root, 6, 4)}</td>
                <td className="mono max-w-[130px] truncate px-2.5 text-ink-3" title={a.s3_uri}>…/anchors/{a.s3_uri.split('/').pop()}</td>
                <td className="mono px-2.5 text-ink-3" title={a.version_id}>{a.version_id.slice(0, 6)}</td>
                <td className="px-2.5"><span className="rounded border border-prove/50 px-1 text-micro font-medium text-ink-1" title="S3 Object Lock mode: cannot be overwritten or deleted, even by the root account, until the retention date">{a.object_lock}</span></td>
                <td className="tnum px-2.5 whitespace-nowrap text-ink-2">{a.retain_until}</td>
                <td className="px-2.5 whitespace-nowrap text-ink-3">{relative(now, simOf(a.written_at))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}

export function ProofPanel({ seq, proof }: { seq: number | null; proof: Proof | null }) {
  return (
    <Panel id="proof" plane="prove" title="Inclusion proof" subtitle={seq ? `record #${seq.toLocaleString()}` : undefined} source={SRC_ANCHOR} className="h-full">
      {!seq ? <EmptyState title="Select a record" /> : !proof ? <div className="text-cap text-ink-3">Fetching Merkle path…</div> : !proof.anchored ? (
        <EmptyState title="Not anchored yet" body={`Records are anchored in batches of 32. This one is included in the next root at #${proof.pending_until?.toLocaleString()}.`} />
      ) : (
        <>
          <MerkleTree proof={proof} />
          <p className={clsx('mt-1 text-micro', proof.matches ? 'text-ink-3' : 'text-critical-ink')}>
            {proof.matches
              ? `Recomputed in this browser from the record and ${proof.path?.length} sibling hashes; equal to the root written to ${proof.anchor?.s3_uri} under Object Lock.`
              : 'The record’s current contents no longer hash to the root stored in S3 Object Lock. The anchor cannot be overwritten, so it still proves the original.'}
          </p>
        </>
      )}
    </Panel>
  )
}

// ------------------------------------------------------------------ tamper

export function TamperPanel({ onVerify }: { onVerify: () => void }) {
  const tampered: number[] = useLive((s) => s.ledger.tampered ?? EMPTY)
  return (
    <Panel id="tamper" plane="prove" title="Tamper simulation" subtitle="demo only"
      actions={<span className="flex items-center gap-1 text-micro text-ink-3"><FlaskConical size={12} /> demo-only control</span>}>
      <p className="max-w-[60ch] text-cap text-ink-2">
        Acts as an attacker with write access to the ledger table: it flips one field of a signed, already-anchored record
        (by default the latest human approval) without touching its hash or signature. Then run the verification.
      </p>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Button variant="danger" onClick={() => api.tamper().then((r: Rec) => toast({ severity: 'warn', title: `Record #${r.seq} modified directly in the datastore (${r.field})` }))}>
          Modify a record in the datastore
        </Button>
        <Button onClick={onVerify}>Verify chain</Button>
        <Button variant="ghost" disabled={!tampered.length} onClick={() => api.restoreLedger().then(() => toast({ severity: 'notice', title: 'Tampered records restored from originals' }))}>Restore originals</Button>
        {tampered.length > 0 && <span className="text-micro text-critical-ink">Modified: {tampered.map((s) => `#${s.toLocaleString()}`).join(', ')}</span>}
      </div>
    </Panel>
  )
}
