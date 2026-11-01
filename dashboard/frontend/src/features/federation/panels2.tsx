import clsx from 'clsx'
import { Lock, Upload, RotateCcw, ArrowRightCircle, Save, Eye, EyeOff, Sigma } from 'lucide-react'
import { useMemo, useState } from 'react'
import { BarList } from '@/charts/BarList'
import { Heatmap } from '@/charts/Heatmap'
import { Sankey } from '@/charts/Sankey'
import { Sparkline } from '@/charts/Sparkline'
import { api } from '@/core/api'
import { useFlag } from '@/core/flags'
import { bytes, clockTime, num } from '@/core/format'
import { opColor } from '@/core/palette'
import { EMPTY, useLive } from '@/core/store'
import type { Checkpoint, Operator, Rec } from '@/core/types'
import { toast, useUi } from '@/core/ui-store'
import { Button, Hash, KV, Panel, SignatureBadge, StatusPill, type Status } from '@/ui'
import { SeverityIcon } from '@/app/StreamDrawer'
import { useFed } from './panels'

// ------------------------------------------------------------------ secure aggregation visualizer

export function SecAggViz() {
  const f = useFed()
  const sa = f.secagg
  const privileged = useFlag('demo.privileged_view')
  const [showTrue, setShowTrue] = useState(false)
  const ops: Operator[] = f.operators ?? []
  if (!sa) return <Panel id="secagg" plane="learn" title="Secure aggregation"><div className="text-cap text-ink-3">Waiting for the first aggregation…</div></Panel>
  const mats: number[][][] = showTrue && privileged ? sa.true : sa.masked
  return (
    <Panel id="secagg" plane="learn" title="Secure aggregation" subtitle={`round ${sa.round}, what the untrusted host sees`}
      source={{ aws: 'Nitro Enclave + KMS', mock: 'Real pairwise-mask maths' }}
      actions={privileged ? (
        <button type="button" onClick={() => setShowTrue((v) => !v)} className={clsx('flex h-7 items-center gap-1.5 rounded-md border px-2 text-micro', showTrue ? 'border-warn/60 text-warn' : 'border-line text-ink-2 hover:text-ink-1')}
          title="Demo only: reveal the unmasked updates to explain what masking hides">
          {showTrue ? <EyeOff size={13} /> : <Eye size={13} />} {showTrue ? 'Privileged view (demo)' : 'Show true updates'}
        </button>
      ) : null}>
      <div className="grid grid-cols-[1fr_auto_200px] items-center gap-4">
        <div className="grid grid-cols-5 gap-2">
          {sa.ops.map((op: string, i: number) => (
            <div key={op}>
              <div className="mb-1 flex items-center gap-1 text-micro text-ink-2"><span className="size-1.5 rounded-full" style={{ background: opColor(op) }} />{ops.find((o) => o.id === op)?.short ?? op}</div>
              <Heatmap matrix={mats[i]} square gap={0.5} scheme="diverging" domain={showTrue ? undefined : [-1, 1]} ariaLabel={`${op} ${showTrue ? 'true' : 'masked'} update`} format={(v) => v.toFixed(3)} />
              <div className="mt-1 text-center text-micro text-ink-3">{showTrue ? 'true update' : 'masked'}</div>
            </div>
          ))}
        </div>
        <div className="grid place-items-center text-ink-3"><Sigma size={22} /></div>
        <div>
          <div className="mb-1 text-micro text-ink-2">Aggregate (mean)</div>
          <Heatmap matrix={sa.aggregate} square gap={0.5} scheme="diverging" ariaLabel="Aggregated update" format={(v) => v.toFixed(4)} />
        </div>
      </div>
      <div className="mt-3 grid grid-cols-4 gap-3 rounded-md border border-line bg-s2 px-3 py-2 text-cap">
        <div><div className="text-micro text-ink-3">max |Σ masked − Σ true|</div><div className="tnum mono font-medium text-good-ink">{sa.max_abs_error.toFixed(6)}</div></div>
        <div><div className="text-micro text-ink-3">pairwise masks</div><div className="tnum font-medium">{sa.pairwise_masks}</div></div>
        <div><div className="text-micro text-ink-3">dropout recovery</div><div className="font-medium">{sa.dropout_recovery}</div></div>
        <div><div className="text-micro text-ink-3">arithmetic</div><div className="font-medium">{sa.fixed_point}</div></div>
      </div>
      <p className="mt-2 text-micro text-ink-3">Each operator adds random masks shared pairwise with the others. Individually the updates are indistinguishable from noise; the masks cancel exactly in the sum, so the server learns only the aggregate.</p>
    </Panel>
  )
}

// ------------------------------------------------------------------ HE panel

export function HePanel() {
  const f = useFed()
  const he = f.he ?? {}
  return (
    <Panel id="he" plane="learn" title="Homomorphic encryption" subtitle="stretch: CKKS micro-demo on a 4,096-weight slice" source={{ aws: 'TenSEAL / OpenFHE on ECS', mock: 'Simulated CKKS stats' }}>
      <div className="grid grid-cols-2 gap-4">
        <KV rows={[
          ['Scheme', he.scheme], ['Poly modulus', num(he.poly_modulus_degree)], ['Coeff modulus', `[${(he.coeff_modulus_bits ?? []).join(', ')}] bits`],
          ['Scale', `2^${he.scale_bits}`], ['Security', `${he.security_bits}-bit`],
          ['Decrypt error', <span className="tnum mono text-good-ink">{he.error ?? '—'}</span>],
        ]} />
        <div>
          <div className="mb-1 text-micro text-ink-3">Ciphertext expansion</div>
          <BarList items={[{ id: 'p', label: 'plaintext slice', value: he.plain_kib ?? 16, color: 'var(--ink-3)' }, { id: 'c', label: 'ciphertext', value: he.cipher_kib ?? 512, color: 'var(--learn)' }]} format={(v) => `${v} KiB`} />
          <div className="mt-3 mb-1 text-micro text-ink-3">Noise budget remaining (bits)</div>
          <BarList items={(he.noise_budget ?? []).map((n: Rec, i: number) => ({ id: String(i), label: n.op, value: n.bits, color: 'var(--learn)' }))} format={(v) => `${v}`} max={100} barHeight={6} />
          <div className="mt-2 tnum text-micro text-ink-3">encrypt {he.timings_ms?.encrypt} ms, aggregate {he.timings_ms?.aggregate} ms, decrypt {he.timings_ms?.decrypt} ms</div>
        </div>
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ sovereignty

export function Sovereignty() {
  const f = useFed()
  const ops: Operator[] = f.operators ?? []
  const sv = f.sovereignty ?? { per_op: {} }
  const ckBytes = (f.checkpoints ?? []).length * 22_675_456
  const nodes = [
    ...ops.map((o) => ({ id: o.id, label: o.short, color: opColor(o.id), sub: `${num(o.samples)} private incidents` })),
    { id: 'agg', label: 'Enclave aggregator', color: 'var(--learn)' },
    { id: 'reg', label: 'Model registry (S3)', color: 'var(--verify)' },
  ]
  const links = [
    ...ops.map((o) => ({ source: o.id, target: 'agg', value: Math.max(1, sv.per_op?.[o.id]?.adapter ?? 0), label: `${o.short} → enclave: masked adapter updates`, color: opColor(o.id) })),
    { source: 'agg', target: 'reg', value: Math.max(1, ckBytes), label: 'signed checkpoints', color: 'var(--learn)' },
    { source: 'x', target: 'y', value: 0, label: 'Raw incident data across boundaries: 0 B', color: 'var(--good)', zero: true },
  ]
  return (
    <Panel id="sovereignty" plane="learn" title="Data sovereignty" subtitle="bytes that crossed an organisational boundary" source={{ aws: 'VPC Flow Logs + S3 access logs', mock: 'Simulated flow evidence' }}>
      <div className="mb-2 grid grid-cols-3 gap-4">
        <div><div className="text-micro text-ink-3">Adapter updates moved</div><div className="tnum text-[18px] font-semibold">{bytes(sv.adapter_bytes)}</div></div>
        <div><div className="text-micro text-ink-3">Metrics moved</div><div className="tnum text-[18px] font-semibold">{bytes(sv.metric_bytes)}</div></div>
        <div><div className="text-micro text-ink-3">Raw data moved</div><div className="tnum text-[18px] font-semibold text-good-ink">0 B</div></div>
      </div>
      <Sankey nodes={nodes} links={links} height={210} format={bytes} />
    </Panel>
  )
}

// ------------------------------------------------------------------ attestation

export function Attestation() {
  const f = useFed()
  const at = useLive((s) => s.trust.attestation)
  const mode = f.config?.secagg_mode
  if (!at) return null
  return (
    <Panel id="attestation" plane="learn" title="Enclave attestation" subtitle={at.enclave_image} source={{ aws: 'AWS Nitro Enclaves + KMS', mock: 'Simulated attestation' }}
      actions={mode === 'enclave' ? <StatusPill status="verified" size="sm">Attested</StatusPill> : <StatusPill status="neutral" size="sm">Enclave mode off</StatusPill>}>
      <div className={clsx(mode !== 'enclave' && 'opacity-50')}>
        <KV rows={[
          ['PCR0 image', <Hash value={at.pcr0} head={14} tail={6} />],
          ['PCR1 kernel', <Hash value={at.pcr1} head={14} tail={6} />],
          ['PCR2 application', <Hash value={at.pcr2} head={14} tail={6} />],
          ['KMS key', <span className="mono break-all text-micro">{at.kms_key}</span>],
          ['Key policy condition', <span className="mono">{at.condition}</span>],
          ['Chain of trust', `${at.root} → enclave certificate → attestation document`],
        ]} />
        <p className="mt-3 text-micro text-ink-3">Operators release their mask seeds only to a KMS key whose policy requires this exact PCR0. Change one byte of the aggregator image and the key stops working, for us too.</p>
      </div>
      {mode !== 'enclave' && <Button size="sm" className="mt-3" onClick={() => api.flConfig({ secagg_mode: 'enclave' })}>Switch to enclave mode</Button>}
    </Panel>
  )
}

// ------------------------------------------------------------------ fleet table

const OP_STATUS: Record<string, Status> = {
  training: 'running', pulling: 'running', securing: 'running', uploading: 'running', waiting: 'pending', idle: 'neutral',
  dropped: 'denied', budget_exhausted: 'critical', quarantined: 'quarantined',
}

export function FleetTable() {
  const f = useFed()
  const ops: Operator[] = f.operators ?? []
  const frames = useLive((s) => s.observability.frames ?? EMPTY)
  const poison = useFlag('demo.poison')
  const spr = f.steps_per_round ?? 240
  const inspect = useUi((s) => s.inspect)
  const target = f.config?.target_eps ?? 3
  return (
    <Panel id="fleet" plane="learn" title="Client fleet" subtitle="one ECS task per operator, own IAM role, own S3 bucket" source={{ aws: 'Amazon ECS + S3', mock: 'Simulated clients' }} flush>
      <div className="overflow-x-auto">
        <table className="w-full text-cap">
          <thead>
            <tr className="border-b border-line text-left text-micro text-ink-3">
              {['Operator', 'Vendor', 'Region', 'Private samples', 'Data commitment', 'Status', 'Step', 'Local loss', 'ε', 'Update norm', 'GPU', 'Uploaded', ''].map((h) => (
                <th key={h} className="px-3 py-2 font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ops.map((o) => (
              <tr key={o.id} className={clsx('border-b border-line/60 hover:bg-s2/50', !o.participating && 'opacity-50')}>
                <td className="px-3 py-2">
                  <button type="button" className="flex items-center gap-2 text-left" onClick={() => inspect({ kind: 'operator', id: o.id, title: o.name, data: o })}>
                    <span className="size-2 rounded-full" style={{ background: opColor(o.id) }} />
                    <span className="font-medium text-ink-1">{o.short}</span><span className="text-ink-3">{o.name}</span>
                  </button>
                </td>
                <td className="px-3">{o.vendor}</td>
                <td className="mono px-3 text-ink-2">{o.region}</td>
                <td className="tnum px-3">{num(o.samples)}</td>
                <td className="px-3"><Hash value={o.commitment} /></td>
                <td className="px-3"><StatusPill status={OP_STATUS[o.status] ?? 'neutral'} size="sm">{o.status.replace('_', ' ')}</StatusPill></td>
                <td className="px-3">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-16 overflow-hidden rounded-full bg-s3"><div className="h-full rounded-full" style={{ width: `${(o.step / spr) * 100}%`, background: opColor(o.id) }} /></div>
                    <span className="tnum text-ink-2">{Math.round(o.step)}</span>
                  </div>
                </td>
                <td className="tnum px-3">{o.loss.toFixed(3)}</td>
                <td className="tnum px-3" style={{ color: o.eps >= 0.9 * target ? 'var(--warn)' : undefined }}>{o.eps.toFixed(2)}</td>
                <td className="tnum px-3">{o.update_norm ? o.update_norm.toFixed(2) : '—'}</td>
                <td className="w-24 px-3"><Sparkline values={frames.slice(-40).map((fr: Rec) => (fr.gpu?.[o.id] ?? 0) * 100)} color={opColor(o.id)} height={18} domain={[0, 100]} /></td>
                <td className="tnum px-3 text-ink-2">{bytes(o.bytes_up)}</td>
                <td className="px-3">
                  <div className="flex justify-end gap-1">
                    <Button size="sm" variant="ghost" onClick={() => api.flClient(o.id, 'pause')}>{o.participating ? 'Pause' : 'Resume'}</Button>
                    <Button size="sm" variant="ghost" onClick={() => api.flClient(o.id, 'drop')} disabled={o.status === 'dropped'}>Drop</Button>
                    {poison && <Button size="sm" variant="danger" onClick={() => api.flClient(o.id, 'poison').then(() => toast({ severity: 'warn', title: `${o.short} will send a poisoned update next round` }))}>Poison</Button>}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ checkpoint rail

export function CheckpointRail() {
  const f = useFed()
  const cks: Checkpoint[] = f.checkpoints ?? []
  const [sel, setSel] = useState<string | null>(null)
  const ck = cks.find((c) => c.id === sel) ?? cks[cks.length - 1]
  const prev = ck ? cks[cks.indexOf(ck) - 1] : undefined
  return (
    <Panel id="checkpoints" plane="learn" title="Checkpoints" subtitle="signed adapter snapshots in S3" source={{ aws: 'Amazon S3 + KMS', mock: 'Simulated store, real signatures' }}
      actions={<Button size="sm" icon={<Save size={13} />} onClick={() => api.fl('checkpoint').then(() => toast({ severity: 'notice', title: 'Checkpoint saved, signed and ledgered' }))}>Save checkpoint</Button>}>
      <div className="relative mb-4 flex items-center gap-0 overflow-x-auto pb-1">
        {cks.map((c, i) => (
          <div key={c.id} className="flex items-center">
            {i > 0 && <div className="h-px w-6 bg-axis" />}
            <button type="button" onClick={() => setSel(c.id)}
              className={clsx('flex shrink-0 flex-col items-start rounded-md border px-2 py-1.5 text-left', c.id === ck?.id ? 'border-learn bg-s2' : 'border-line hover:bg-s2')}>
              <span className="flex items-center gap-1 text-cap font-medium text-ink-1">
                {c.id}{c.promoted && <ArrowRightCircle size={12} className="text-verify" aria-label="promoted" />}
              </span>
              <span className="tnum text-micro text-ink-3">loss {c.val_loss.toFixed(3)}</span>
            </button>
          </div>
        ))}
      </div>
      {ck && (
        <div className="grid grid-cols-[1fr_auto] gap-4">
          <KV rows={[
            ['Round', <span className="tnum">{ck.round} {ck.manual ? '(manual)' : '(auto)'}</span>],
            ['Validation loss', <span className="tnum">{ck.val_loss.toFixed(4)} {prev && <span className={ck.val_loss <= prev.val_loss ? 'text-good-ink' : 'text-critical-ink'}>({(ck.val_loss - prev.val_loss).toFixed(4)} vs {prev.id})</span>}</span>],
            ['Accuracy', <span className="tnum">{(ck.accuracy * 100).toFixed(1)}%</span>],
            ['Privacy spent', <span className="tnum">ε max {ck.eps_max.toFixed(2)} (δ = 1e-5)</span>],
            ['Adapter', <span className="flex items-center gap-2"><Hash value={ck.adapter_hash} /> <span className="text-ink-3">{bytes(ck.size_bytes)}</span></span>],
            ['Object', <span className="mono break-all text-micro text-ink-2">{ck.s3_uri}</span>],
            ['Signature', <SignatureBadge keyId="kms:spectrum/aggregator" ledgerSeq={ck.ledger_seq} />],
            ['Saved', <span className="mono">{clockTime(ck.created_at)}</span>],
          ]} />
          <div className="flex flex-col gap-2">
            <Button size="sm" icon={<Upload size={13} />} disabled={ck.promoted} onClick={() => api.checkpoint(ck.id, 'promote').then(() => toast({ severity: 'notice', title: `${ck.id} sent to the Proving Ground` }))}>
              {ck.promoted ? 'Promoted' : 'Promote to Proving Ground'}
            </Button>
            <Button size="sm" variant="ghost" icon={<RotateCcw size={13} />} onClick={() => api.checkpoint(ck.id, 'restore').then(() => toast({ severity: 'warn', title: `Global model restored to ${ck.id}` }))}>Restore</Button>
          </div>
        </div>
      )}
    </Panel>
  )
}

// ------------------------------------------------------------------ FL log

export function FlLog() {
  const stream = useLive((s) => s.stream)
  const rows = useMemo(() => stream.filter((e) => e.plane === 'learn').slice(-60).reverse(), [stream])
  const inspect = useUi((s) => s.inspect)
  return (
    <Panel id="fl-log" plane="learn" title="Training log" subtitle="learn-plane events, newest first" flush>
      <div className="max-h-[220px] overflow-auto px-4 py-2">
        {rows.map((e) => (
          <div key={e.seq} className="row-in flex items-center gap-3 py-0.5 text-cap">
            <SeverityIcon s={e.severity} />
            <span className="tnum mono text-ink-3">{clockTime(e.ts)}</span>
            <span className="mono w-20 shrink-0 text-ink-2">{e.actor}</span>
            <span className="truncate text-ink-1">{e.summary}</span>
            {e.ledger && <button type="button" className="mono ml-auto shrink-0 text-ink-3 hover:text-ink-1" onClick={() => inspect({ kind: 'ledger', id: String(e.ledger!.seq) })}>#{e.ledger.seq}</button>}
          </div>
        ))}
      </div>
    </Panel>
  )
}

export function LockNote() {
  return <span className="inline-flex items-center gap-1 text-micro text-ink-3"><Lock size={11} /> enclave</span>
}
