import clsx from 'clsx'
import { ChevronDown, ChevronRight, CircleCheck, CircleX, ShieldCheck, ShieldX, Upload } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { BarList } from '@/charts/BarList'
import { Tooltip, useSize } from '@/charts/common'
import { Histogram } from '@/charts/Histogram'
import { LineChart } from '@/charts/LineChart'
import { Scatter, type Dot } from '@/charts/Scatter'
import { api } from '@/core/api'
import { pct, usd } from '@/core/format'
import { opColor } from '@/core/palette'
import { useLive } from '@/core/store'
import type { Rec } from '@/core/types'
import { toast } from '@/core/ui-store'
import { Button, CodeBlock, Hash, KV, Panel, SignatureBadge, StatusPill } from '@/ui'
import { gateChecks, latencyBins, MOCK_LABEL, throughputCurve, type ModelRow } from './model'
import { SRC } from './panels'

// ------------------------------------------------------------------ latency + throughput

export function Latency({ m, prod, models }: { m: ModelRow; prod: ModelRow; models: ModelRow[] }) {
  const { bins, p99 } = latencyBins(m.p50_ms, m.p95_ms)
  const ref = models.find((x) => x.id === 'claude-haiku')
  const curves = [
    { id: m.id, label: `${m.id} (candidate)`, color: 'var(--verify)', points: throughputCurve(m.throughput_rps), endLabel: false },
    ...(prod.id !== m.id ? [{ id: prod.id, label: `${prod.id} (production)`, color: 'var(--s1)', points: throughputCurve(prod.throughput_rps) }] : []),
    ...(ref ? [{ id: ref.id, label: 'Claude Haiku (Bedrock)', color: 'var(--neutral-series)', points: throughputCurve(ref.throughput_rps) }] : []),
  ]
  return (
    <Panel id="latency" plane="verify" title="Performance" subtitle={`${m.id}: latency distribution and throughput under load`} source={{ aws: 'SageMaker load test', mock: MOCK_LABEL }}>
      <div className="grid grid-cols-2 gap-6">
        <div>
          <div className="mb-1 text-micro text-ink-3">Request latency, share of requests</div>
          <Histogram bins={bins} color="var(--verify)" height={200} xFormat={(v) => `${Math.round(v)}`}
            markers={[{ x: m.p50_ms, label: `p50 ${m.p50_ms}` }, { x: m.p95_ms, label: `p95 ${m.p95_ms}` }, { x: p99, label: `p99 ${Math.round(p99)}` }]} yLabel="requests" />
          <div className="mt-1 text-right text-micro text-ink-3">milliseconds</div>
        </div>
        <div>
          <div className="mb-1 text-micro text-ink-3">Throughput against concurrency (requests/s)</div>
          <LineChart series={curves} height={200} xFormat={(x) => `${x}`} yFormat={(y) => `${Math.round(y)}`} tipTitle={(x) => `concurrency ${x}`} />
          <div className="mt-1 text-right text-micro text-ink-3">concurrent requests</div>
        </div>
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ federated evaluation

export function FedEval({ m, rows }: { m: ModelRow; rows: Rec[] }) {
  const { ref, width } = useSize()
  const [hover, setHover] = useState<{ r: Rec; x: number; y: number } | null>(null)
  if (!rows.length) {
    return <Panel id="fed-eval" plane="verify" title="Federated evaluation"><div className="text-cap text-ink-3">No private scores yet for {m.id}. They arrive in the “Federated eval” stage.</div></Panel>
  }
  const N = rows.reduce((a, r) => a + r.n, 0)
  const mean = rows.reduce((a, r) => a + r.score * r.n, 0) / N
  const ci = 1.96 * Math.sqrt((mean * (1 - mean)) / N)
  const lo = Math.min(...rows.map((r) => r.score), mean - ci) - 0.02
  const hi = Math.max(...rows.map((r) => r.score), mean + ci) + 0.02
  const W = Math.max(10, width - 70)
  const x = (v: number) => ((v - lo) / (hi - lo)) * W
  const rowH = 22
  return (
    <Panel id="fed-eval" plane="verify" title="Federated evaluation" subtitle={`${m.id} scored on each operator's private held-out set`} source={{ aws: 'Operator-side SageMaker jobs', mock: MOCK_LABEL }}>
      <div ref={ref} className="relative">
        {width > 0 && (
          <svg width={width} height={(rows.length + 1) * rowH + 22} role="img" aria-label="Private scores per operator">
            {[0, 0.25, 0.5, 0.75, 1].map((k) => {
              const v = lo + (hi - lo) * k
              return (
                <g key={k} transform={`translate(${60 + x(v)},0)`}>
                  <line y1={0} y2={(rows.length + 1) * rowH} style={{ stroke: 'var(--hairline)' }} />
                  <text y={(rows.length + 1) * rowH + 14} textAnchor="middle" style={{ fill: 'var(--ink-3)', fontSize: 11 }} className="tnum">{(v * 100).toFixed(0)}%</text>
                </g>
              )
            })}
            {rows.map((r, i) => (
              <g key={r.operator} transform={`translate(0,${i * rowH + rowH / 2})`}
                onPointerMove={(e) => { const rc = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect(); setHover({ r, x: e.clientX - rc.left, y: e.clientY - rc.top }) }}
                onPointerLeave={() => setHover(null)}>
                <rect x={0} y={-rowH / 2} width={width} height={rowH} fill="transparent" />
                <text x={0} dy="0.32em" style={{ fill: 'var(--ink-2)', fontSize: 11 }}>{`Op-${r.operator.slice(3).toUpperCase()}`}</text>
                <circle cx={60 + x(r.score)} r={4.5} style={{ fill: opColor(r.operator), stroke: 'var(--surface-1)' }} strokeWidth={2} />
              </g>
            ))}
            <g transform={`translate(0,${rows.length * rowH + rowH / 2})`}>
              <text x={0} dy="0.32em" style={{ fill: 'var(--ink-1)', fontSize: 11, fontWeight: 500 }}>Weighted</text>
              <line x1={60 + x(mean - ci)} x2={60 + x(mean + ci)} style={{ stroke: 'var(--verify)' }} strokeWidth={2} />
              <line x1={60 + x(mean - ci)} x2={60 + x(mean - ci)} y1={-5} y2={5} style={{ stroke: 'var(--verify)' }} strokeWidth={2} />
              <line x1={60 + x(mean + ci)} x2={60 + x(mean + ci)} y1={-5} y2={5} style={{ stroke: 'var(--verify)' }} strokeWidth={2} />
              <rect x={60 + x(mean) - 5} y={-5} width={10} height={10} transform={`rotate(45 ${60 + x(mean)} 0)`} style={{ fill: 'var(--verify)', stroke: 'var(--surface-1)' }} strokeWidth={2} />
            </g>
          </svg>
        )}
        {hover && <Tooltip x={hover.x} y={hover.y} containerWidth={width} title={hover.r.operator} rows={[{ label: 'private score', value: pct(hover.r.score, true) }, { label: 'samples', value: hover.r.n.toLocaleString() }, { label: 'data shared', value: '0 B' }]} />}
      </div>
      <table className="mt-2 w-full text-cap">
        <thead><tr className="text-left text-micro text-ink-3">{['Operator', 'Score', 'Samples', 'Data shared', 'Signature'].map((h) => <th key={h} className="pb-1 font-medium">{h}</th>)}</tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.operator} className="border-t border-line/60">
              <td className="py-1"><span className="mr-1.5 inline-block size-2 rounded-full" style={{ background: opColor(r.operator) }} />{r.operator}</td>
              <td className="tnum">{pct(r.score, true)}</td>
              <td className="tnum text-ink-2">{r.n.toLocaleString()}</td>
              <td className="tnum text-good-ink">{r.data_left_bytes} B</td>
              <td><SignatureBadge state={r.verified ? 'verified' : 'failed'} keyId={r.key_id} /></td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-2 flex items-baseline justify-between text-cap">
        <span className="text-ink-2">Weighted mean <span className="tnum font-semibold text-ink-1">{pct(mean, true)}</span> <span className="tnum text-ink-3">±{(ci * 100).toFixed(1)} pts (95% CI, n = {N.toLocaleString()})</span></span>
      </div>
      <p className="mt-1 text-micro text-ink-3">Each operator scored the model on data we never see. Only signed scores were shared.</p>
    </Panel>
  )
}

// ------------------------------------------------------------------ scaling curve

export function Scaling({ models, sel }: { models: ModelRow[]; sel: string }) {
  const ours = models.filter((m) => ['nlm-0.5b', 'nlm-v6', 'nlm-v7-3b'].includes(m.id) || (m.id === sel && m.params_b === 1))
  const pts = models.filter((m) => ['nlm-0.5b', 'nlm-v6', 'nlm-v7-3b'].includes(m.id))
  const sonnet = models.find((m) => m.id === 'claude-sonnet')
  const haiku = models.find((m) => m.id === 'claude-haiku')
  const dots: Dot[] = ours.map((m) => ({
    id: m.id, x: m.params_b ?? 1, y: m.accuracy * 100, color: m.id === sel ? 'var(--verify)' : 'var(--s3)', label: `${m.id}, ${m.params_b}B`, showLabel: true, ring: m.id === sel,
    tip: [{ label: 'accuracy', value: pct(m.accuracy, true) }, { label: '$ per 1K', value: usd(m.usd_per_1k) }],
  }))
  return (
    <Panel id="scaling" plane="verify" title="NLM scaling" subtitle="parameters against accuracy, federated fine-tunes" source={SRC}>
      <Scatter dots={dots} height={210} xLog xDomain={[0.5, 3]} yDomain={[70, 95]} frontier={pts.map((m) => m.id)} xFormat={(v) => `${v}B`} yFormat={(v) => `${v.toFixed(0)}%`} xLabel="parameters (log)" yLabel="accuracy"
        hlines={[...(sonnet ? [{ y: sonnet.accuracy * 100, label: `Claude Sonnet ${pct(sonnet.accuracy, true)}`, color: 'var(--neutral-series)' }] : []),
          ...(haiku ? [{ y: haiku.accuracy * 100, label: `Claude Haiku ${pct(haiku.accuracy, true)}`, color: 'var(--neutral-series)' }] : [])]} />
      <div className="mt-3 mb-1 text-micro text-ink-3">Price per 1K tasks at each size</div>
      <BarList items={pts.map((m) => ({ id: m.id, label: `${m.params_b}B (${m.id})`, value: m.usd_per_1k, color: 'var(--s3)' }))} format={(v) => usd(v)} />
    </Panel>
  )
}

// ------------------------------------------------------------------ gate

export function Gate({ m, gate, run }: { m: ModelRow; gate: Rec; run: Rec | null }) {
  const navigate = useNavigate()
  const release = useLive((s) => s.release)
  const checks = gateChecks(m, gate)
  const pass = checks.every((c) => c.ok)
  const evaluated = run && run.model === m.id && run.result
  const inRelease = release.candidate === m.id || release.prod === m.id
  const cedar = `permit (\n  principal == Pipeline::"proving-ground",\n  action == Action::"deploy",\n  resource == Model::"${m.id}"\n)\nwhen {\n${checks.map((c, i) => `  ${c.cedar}${i < checks.length - 1 ? ' &&' : ''}`).join('\n')}\n};`
  return (
    <Panel id="gate" plane="verify" title="Deploy gate" subtitle="policy P-012 in Verified Permissions" source={{ aws: 'Amazon Verified Permissions', mock: 'Cedar mirror' }}
      actions={<StatusPill status={pass ? 'passed' : 'blocked'}>{pass ? 'Pass: eligible for shadow' : 'Blocked'}</StatusPill>}>
      <ul className="mb-3 space-y-1.5">
        {checks.map((c) => (
          <li key={c.clause} className="flex items-center gap-2 text-cap">
            {c.ok ? <CircleCheck size={14} className="text-good-ink" /> : <CircleX size={14} className="text-critical-ink" />}
            <span className="mono text-ink-1">{c.clause}</span>
            <span className={clsx('tnum ml-auto', c.ok ? 'text-ink-2' : 'text-critical-ink')}>{c.value}</span>
          </li>
        ))}
      </ul>
      <CodeBlock code={cedar} lang="cedar" />
      <div className="mt-3 flex items-center gap-2">
        <Button variant="primary" size="sm" icon={<Upload size={13} />} disabled={!pass}
          onClick={() => {
            if (inRelease) { navigate('/release'); return }
            api.release('shadow').then(() => { toast({ severity: 'notice', title: `${m.id} deployed in shadow on all targets` }); navigate('/release') })
          }}>
          {inRelease ? 'Open in Release Center' : 'Promote to Release'}
        </Button>
        <span className="text-micro text-ink-3">{evaluated ? `gate decision ledgered by ${run.id}` : 'live check against the latest scores; run an evaluation to ledger it'}</span>
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ AI-BOM viewer

export function AiBom({ m, bom }: { m: ModelRow; bom: Rec | undefined }) {
  const [res, setRes] = useState<Rec | null>(null)
  const [busy, setBusy] = useState(false)
  if (!bom) {
    return <Panel id="aibom" plane="verify" title="AI-BOM"><div className="text-cap text-ink-3">{m.id} has no AI bill of materials. Baselines are third-party models; our models get one in the “Sign AI-BOM” stage.</div></Panel>
  }
  const { signature, digest, signed_at, ...body } = bom
  return (
    <Panel id="aibom" plane="verify" title="AI-BOM" subtitle={`${bom.spec}, ${m.id}`} source={{ aws: 'AWS KMS Sign', mock: 'Ed25519 keyring' }}
      actions={<Button size="sm" disabled={busy} onClick={() => { setBusy(true); api.verifyAibom(m.id).then(setRes).finally(() => setBusy(false)) }}>{busy ? 'Verifying…' : 'Verify signature'}</Button>}>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_340px]">
        <div className="max-h-[360px] overflow-auto rounded-md border border-line bg-s2 p-2"><Tree value={body} name="ai-bom" depth={0} /></div>
        <div>
          <KV rows={[
            ['Digest', <Hash value={digest} head={10} tail={6} />],
            ['Signature', <Hash value={signature} head={10} tail={6} />],
            ['Signer', <span className="mono break-all text-micro">{bom.signer?.key}</span>],
            ['Algorithm', bom.signer?.alg],
            ['Signed', <span className="mono">{signed_at?.slice(0, 19).replace('T', ' ')}</span>],
          ]} />
          {res && (
            <ul className="mt-3 space-y-1 rounded-md border border-line bg-s2 px-3 py-2 text-cap">
              <li className="flex items-center gap-2">{res.digest_matches ? <ShieldCheck size={14} className="text-good-ink" /> : <ShieldX size={14} className="text-critical-ink" />} Canonical digest recomputed and matches</li>
              <li className="flex items-center gap-2">{res.signature_valid ? <ShieldCheck size={14} className="text-good-ink" /> : <ShieldX size={14} className="text-critical-ink" />} Signature valid for the signer key</li>
            </ul>
          )}
          <p className="mt-3 text-micro text-ink-3">Base model, adapter, training run, DP parameters, operator data commitments and eval datasets, in one signed document. A change to any field breaks the digest.</p>
        </div>
      </div>
    </Panel>
  )
}

function Tree({ value, name, depth }: { value: unknown; name: string; depth: number }): ReactNode {
  const [open, setOpen] = useState(depth < 2)
  const isObj = value !== null && typeof value === 'object'
  if (!isObj) {
    const s = String(value)
    return (
      <div className="flex gap-2 py-0.5 text-cap" style={{ paddingLeft: depth * 14 + 16 }}>
        <span className="text-ink-3">{name}</span>
        <span className="mono min-w-0 truncate text-ink-1" title={s}>{/^[0-9a-f]{40,}$/.test(s) ? `${s.slice(0, 12)}…${s.slice(-6)}` : s}</span>
      </div>
    )
  }
  const entries = Array.isArray(value) ? value.map((v, i) => [String(i), v] as const) : Object.entries(value as Rec)
  return (
    <div>
      <button type="button" onClick={() => setOpen((o) => !o)} className="flex items-center gap-1 py-0.5 text-cap text-ink-2 hover:text-ink-1" style={{ paddingLeft: depth * 14 }}>
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}<span className="font-medium">{name}</span>
        <span className="text-micro text-ink-3">{Array.isArray(value) ? `[${entries.length}]` : `{${entries.length}}`}</span>
      </button>
      {open && entries.map(([k, v]) => <Tree key={k} value={v} name={k} depth={depth + 1} />)}
    </div>
  )
}

