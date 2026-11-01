import clsx from 'clsx'
import { ArrowRight, Check, KeyRound, Play, ShieldCheck, ShieldX } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { AreaStack } from '@/charts/AreaStack'
import { BarList } from '@/charts/BarList'
import { api, post } from '@/core/api'
import { clockTime, compact, simOf } from '@/core/format'
import { EMPTY, useLive } from '@/core/store'
import type { Agent, Rec } from '@/core/types'
import { useUi } from '@/core/ui-store'
import { Button, CodeBlock, EmptyState, Hash, KV, Panel, StatusPill, type Status } from '@/ui'

const SRC_AVP = { aws: 'Amazon Verified Permissions', mock: 'Cedar mirror, deterministic' }
const NONE: Record<string, any> = {}

export const DECISION_STATUS: Record<string, Status> = { permit: 'verified', deny: 'denied', needs_human: 'pending' }
export const DECISION_WORD: Record<string, string> = { permit: 'Permit', deny: 'Deny', needs_human: 'Needs human' }

// ------------------------------------------------------------------ posture

export function Posture() {
  const msgs: Rec[] = useLive((s) => s.swarm.messages ?? EMPTY)
  const agents: Agent[] = useLive((s) => s.swarm.agents ?? EMPTY)
  const decisions: Rec[] = useLive((s) => s.trust.decisions ?? EMPTY)
  const stats = useLive((s) => s.trust.stats ?? NONE)
  const guard = useLive((s) => s.trust.guardrail_counts ?? NONE)
  const policies: Rec[] = useLive((s) => s.trust.policies ?? EMPTY)
  const signed = msgs.length ? msgs.filter((m) => m.sig).length / msgs.length : 1
  const verified = msgs.filter((m) => m.verified).length
  const revoked = agents.filter((a) => a.status === 'quarantined').length
  const actions = new Set(decisions.map((d) => d.action))
  const covered = new Set(decisions.filter((d) => !(d.policies ?? []).includes('default-deny')).map((d) => d.action))
  const lat = decisions.length ? decisions.reduce((a, d) => a + (d.latency_us ?? 0), 0) / decisions.length : 0
  const total = (stats.permit ?? 0) + (stats.deny ?? 0) + (stats.needs_human ?? 0)
  const tile = (label: string, value: React.ReactNode, sub: React.ReactNode, tone?: 'good' | 'bad') => (
    <div className="min-w-0 border-l border-line pl-4 first:border-l-0 first:pl-0">
      <div className="text-micro text-ink-3">{label}</div>
      <div className={clsx('mt-0.5 text-[20px] leading-7 font-semibold', tone === 'good' ? 'text-good-ink' : tone === 'bad' ? 'text-critical-ink' : 'text-ink-1')}>{value}</div>
      <div className="truncate text-micro text-ink-3">{sub}</div>
    </div>
  )
  return (
    <div className="grid grid-cols-2 gap-4 rounded-[10px] border border-line bg-s1 px-5 py-4 md:grid-cols-3 xl:grid-cols-6">
      {tile('Signed agent messages', `${Math.round(signed * 100)}%`, `${verified} of ${msgs.length} recent verified`, 'good')}
      {tile('Tool coverage by policy', `${covered.size} / ${actions.size || 0}`, 'tools seen with an explicit policy', covered.size === actions.size ? 'good' : 'bad')}
      {tile('Policies in force', policies.length, `${policies.filter((p) => p.effect === 'forbid').length} forbid, ${policies.filter((p) => p.effect === 'permit').length} permit`)}
      {tile('Decisions', compact(total), `${stats.deny ?? 0} denied, ${stats.needs_human ?? 0} sent to a human`)}
      {tile('Identities', `${agents.length - revoked} active`, revoked ? `${revoked} revoked (quarantine)` : 'none revoked', revoked ? 'bad' : undefined)}
      {tile('Decision latency', `${Math.round(lat)} µs`, `mean of ${decisions.length}; guardrail blocks ${guard.prompt_attack ?? 0}`)}
    </div>
  )
}

// ------------------------------------------------------------------ policy store + test bench

const PRESETS: { id: string; label: string; body: Rec }[] = [
  { id: 'hospital', label: 'site-17 puts a hospital cell to sleep', body: { principal: 'site-17', action: 'cell_sleep', resource: { id: 's-06-0', site: 's-06', tags: ['hospital'] }, context: { target_count: 1 } } },
  { id: 'local', label: 'site-14 raises power on its own cell (2%)', body: { principal: 'site-14', action: 'set_tx_power', resource: { id: 's-14-1', site: 's-14', tags: ['stadium'] }, context: { blast_radius_pct: 2 } } },
  { id: 'big', label: 'coordinator wakes 3 cells, 12.5% blast radius', body: { principal: 'coordinator', action: 'cell_wake', resource: { id: '3 cells', site: null, tags: [] }, context: { blast_radius_pct: 12.5, plan_id: 'plan-2041' } } },
  { id: 'approved', label: 'same, after a human approved', body: { principal: 'coordinator', action: 'cell_wake', resource: { id: '3 cells', site: null, tags: [] }, context: { blast_radius_pct: 12.5, plan_id: 'plan-2041', human_approved: true } } },
  { id: 'mass', label: 'site-21 sleeps 120 cells', body: { principal: 'site-21', action: 'cell_sleep', resource: { id: 'region-west', site: 's-21', tags: [] }, context: { target_count: 120, blast_radius_pct: 100 } } },
  { id: 'quarantined', label: 'a quarantined agent tunes handover', body: { principal: 'site-17', action: 'set_handover_offset', resource: { id: 's-17-0', site: 's-17', tags: [] }, context: { blast_radius_pct: 0.8 }, status: 'quarantined' } },
  { id: 'deputy', label: 'site-22 reroutes transport (not its job)', body: { principal: 'site-22', action: 'reroute', resource: { id: 'T3', site: null, tags: [] }, context: {} } },
]

export function PolicyStore() {
  const policies: Rec[] = useLive((s) => s.trust.policies ?? EMPTY)
  const storeId = useLive((s) => s.trust.policy_store_id)
  const [sel, setSel] = useState('P-003')
  const [preset, setPreset] = useState(PRESETS[2].id)
  const [res, setRes] = useState<Rec | null>(null)
  const [busy, setBusy] = useState(false)
  const p = policies.find((x) => x.id === sel) ?? policies[0]
  const run = () => {
    const pr = PRESETS.find((x) => x.id === preset)!
    setBusy(true)
    post('/api/trust/test', pr.body).then(setRes).finally(() => setBusy(false))
  }
  useEffect(() => { setRes(null) }, [preset])
  return (
    <Panel id="policy" plane="prove" title="Policy store" subtitle={<span className="mono">{storeId}</span>} source={SRC_AVP} flush>
      <div className="grid grid-cols-[minmax(260px,1fr)_1.5fr]">
        <ul className="max-h-[520px] overflow-auto border-r border-line py-1">
          {policies.map((x) => (
            <li key={x.id}>
              <button type="button" onClick={() => setSel(x.id)}
                className={clsx('flex w-full items-center gap-2 px-4 py-2 text-left', x.id === p?.id ? 'bg-s3' : 'hover:bg-s2/60')}>
                <span className="mono w-11 shrink-0 text-cap text-ink-3">{x.id}</span>
                <span className="min-w-0 flex-1 truncate text-cap text-ink-1">{x.name}</span>
                <StatusPill status={x.effect === 'forbid' ? 'denied' : 'verified'} size="sm">{x.effect}</StatusPill>
                <span className="tnum w-12 shrink-0 text-right text-micro text-ink-3" title="decisions this policy determined">{compact(x.hits)}</span>
              </button>
            </li>
          ))}
        </ul>
        <div className="min-w-0 p-4">
          {p && (
            <>
              <div className="mb-2 flex items-center gap-2">
                <span className="text-title font-medium text-ink-1">{p.name}</span>
                <span className="text-micro text-ink-3">version {p.version}</span>
                <span className="ml-auto"><Hash value={p.hash} /></span>
              </div>
              <CodeBlock code={p.cedar} lang="cedar" />
              <div className="mt-1.5 text-micro text-ink-3">Evaluated outside the model, by the gateway, for every tool call. The model cannot see or edit it.</div>
            </>
          )}
          <div className="mt-5 rounded-md border border-line bg-s2 p-3">
            <div className="mb-2 text-cap font-medium text-ink-1">Test a request against the live policy set</div>
            <div className="flex flex-wrap items-center gap-2">
              <select value={preset} onChange={(e) => setPreset(e.target.value)} aria-label="Scenario" className="h-8 min-w-0 flex-1 rounded-md border border-line bg-s1 px-2 text-cap text-ink-1">
                {PRESETS.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
              </select>
              <Button icon={<Play size={13} />} disabled={busy} onClick={run}>Evaluate</Button>
            </div>
            <CodeBlock className="mt-2" code={JSON.stringify(PRESETS.find((x) => x.id === preset)!.body, null, 1).replace(/\n\s*/g, ' ')} />
            {res && (
              <div className="mt-3 flex flex-wrap items-center gap-2 text-cap">
                <StatusPill status={DECISION_STATUS[res.decision]}>{DECISION_WORD[res.decision]}</StatusPill>
                <span className="text-ink-3">determined by</span>
                {res.policies.map((pid: string) => (
                  <button key={pid} type="button" onClick={() => policies.some((x) => x.id === pid) && setSel(pid)} className="mono rounded border border-line px-1.5 text-ink-1 hover:bg-s3">{pid}</button>
                ))}
                <span className="tnum ml-auto text-micro text-ink-3">evaluated in {res.eval_us} µs, not ledgered</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ decision stream

export function DecisionStream() {
  const decisions: Rec[] = useLive((s) => s.trust.decisions ?? EMPTY)
  const series: Rec[] = useLive((s) => s.trust.series ?? EMPTY)
  const inspect = useUi((s) => s.inspect)
  const { xs, rows } = useMemo(() => {
    const B = 15
    const map = new Map<number, Rec>()
    for (const b of series) { const k = Math.floor(b.t / B) * B; map.set(k, { permit: b.permit, deny: b.deny, needs_human: b.needs_human }) }
    const first = decisions.length ? simOf(decisions[0].ts) : Infinity
    for (const k of [...map.keys()]) if (k >= first) map.delete(k)
    for (const d of decisions) {
      const k = Math.floor(simOf(d.ts) / B) * B
      const r = map.get(k) ?? { permit: 0, deny: 0, needs_human: 0 }
      r[d.decision] = (r[d.decision] ?? 0) + 1
      map.set(k, r)
    }
    const keys = [...map.keys()].sort((a, b) => a - b)
    if (keys.length) {
      // fill gaps so quiet periods read as zero, not as interpolation
      const full: number[] = []
      for (let k = keys[0]; k <= keys[keys.length - 1]; k += B) full.push(k)
      return { xs: full, rows: full.map((k) => map.get(k) ?? { permit: 0, deny: 0, needs_human: 0 }) }
    }
    return { xs: [], rows: [] }
  }, [decisions, series])
  const fmtT = (t: number) => {
    const d = new Date(Date.parse('2026-10-31T21:30:00-07:00') + t * 1000 - 7 * 3600e3)
    return `${String(d.getUTCHours()).padStart(2, '0')}:${String(d.getUTCMinutes()).padStart(2, '0')}`
  }
  const recent = [...decisions].reverse().slice(0, 80)
  return (
    <Panel id="decisions" plane="prove" title="Policy decisions" subtitle="every tool call, per 15 s" source={SRC_AVP}>
      <AreaStack xs={xs} rows={rows} height={150} xFormat={fmtT}
        series={[{ id: 'permit', label: 'Permit', color: 'var(--s1)' }, { id: 'needs_human', label: 'Needs human', color: 'var(--s4)' }, { id: 'deny', label: 'Deny', color: 'var(--s8)' }]} />
      <div className="mt-1 mb-1 text-micro text-ink-3">Denied and escalated only</div>
      <AreaStack xs={xs} rows={rows} height={80} xFormat={fmtT} legend={false}
        series={[{ id: 'needs_human', label: 'Needs human', color: 'var(--s4)' }, { id: 'deny', label: 'Deny', color: 'var(--s8)' }]} />
      <div className="mt-3 max-h-[300px] overflow-auto">
        <table className="w-full text-cap">
          <thead className="sticky top-0 bg-s1">
            <tr className="border-b border-line text-left text-micro text-ink-3">
              {['Time', 'Principal', 'Action', 'Resource', 'Decision', 'Determining policies', 'Latency'].map((h) => <th key={h} className="py-1.5 pr-3 font-medium">{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {recent.map((d) => (
              <tr key={d.id} onClick={() => inspect({ kind: 'decision', id: d.id, title: `${d.decision} ${d.principal} ${d.action}`, data: d })}
                className={clsx('row-in cursor-pointer border-b border-line/50 hover:bg-s2/60', d.decision === 'deny' && 'bg-critical/5')}>
                <td className="tnum mono py-1 pr-3 text-ink-3">{clockTime(d.ts)}</td>
                <td className="mono pr-3 text-ink-1">{d.principal}</td>
                <td className="mono pr-3 text-ink-2">{d.action}</td>
                <td className="mono max-w-[140px] truncate pr-3 text-ink-2">{d.resource}</td>
                <td className="pr-3"><StatusPill status={DECISION_STATUS[d.decision] ?? 'neutral'} size="sm">{DECISION_WORD[d.decision] ?? d.decision}</StatusPill></td>
                <td className="pr-3"><span className="mono text-ink-2">{(d.policies ?? []).join(', ')}</span></td>
                <td className="tnum pr-3 text-ink-3">{d.latency_us} µs</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}

export function TopDenied() {
  const decisions: Rec[] = useLive((s) => s.trust.decisions ?? EMPTY)
  const { byAction, byPrincipal } = useMemo(() => {
    const a: Record<string, number> = {}
    const p: Record<string, number> = {}
    for (const d of decisions) if (d.decision === 'deny') { a[d.action] = (a[d.action] ?? 0) + 1; p[d.principal] = (p[d.principal] ?? 0) + 1 }
    const top = (m: Record<string, number>) => Object.entries(m).sort((x, y) => y[1] - x[1]).slice(0, 6).map(([k, v]) => ({ id: k, label: k, value: v }))
    return { byAction: top(a), byPrincipal: top(p) }
  }, [decisions])
  return (
    <Panel id="denied" plane="prove" title="Top denied" subtitle={`last ${decisions.length} decisions`} source={SRC_AVP}>
      {!byAction.length ? <EmptyState title="Nothing denied recently" body="Every recent tool call was permitted or sent to a human. Run the prompt-injection attack to see the gateway refuse one." /> : (
        <div className="grid grid-cols-2 gap-6">
          <div><div className="mb-2 text-micro text-ink-3">By action</div><BarList items={byAction} color="var(--s8)" /></div>
          <div><div className="mb-2 text-micro text-ink-3">By principal</div><BarList items={byPrincipal} color="var(--s8)" /></div>
        </div>
      )}
    </Panel>
  )
}

// ------------------------------------------------------------------ identities

export function Identities() {
  const agents: Agent[] = useLive((s) => s.swarm.agents ?? EMPTY)
  const [fp, setFp] = useState<Record<string, Rec>>({})
  const inspect = useUi((s) => s.inspect)
  const keys = agents.map((a) => a.key_id).join('|')
  useEffect(() => {
    let alive = true
    Promise.all(agents.map((a) => api.key(a.key_id).then((k) => [a.key_id, k] as const).catch(() => null))).then((xs) => {
      if (alive) setFp(Object.fromEntries(xs.filter(Boolean) as [string, Rec][]))
    })
    return () => { alive = false }
  }, [keys]) // eslint-disable-line react-hooks/exhaustive-deps
  const order = { quarantined: 0, coordinator: 1, domain: 2, site: 3 } as Record<string, number>
  const rows = [...agents].sort((a, b) => (order[a.status === 'quarantined' ? 'quarantined' : a.class] - order[b.status === 'quarantined' ? 'quarantined' : b.class]) || a.id.localeCompare(b.id))
  return (
    <Panel id="identities" plane="prove" title="Agent identities" subtitle={`${agents.length} agents, one signing key each`} source={{ aws: 'AgentCore Identity + AWS KMS', mock: 'Ed25519 keyring' }} flush>
      <div className="max-h-[380px] overflow-auto">
        <table className="w-full text-cap">
          <thead className="sticky top-0 bg-s1">
            <tr className="border-b border-line text-left text-micro text-ink-3">
              {['Agent', 'Class', 'Key', 'Fingerprint', 'Autonomy', 'Trust', 'Status'].map((h) => <th key={h} className="px-3 py-1.5 font-medium">{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((a) => {
              const k = fp[a.key_id]
              const revoked = a.status === 'quarantined' || !!k?.revoked_at
              return (
                <tr key={a.id} onClick={() => inspect({ kind: 'agent', id: a.id })} className={clsx('cursor-pointer border-b border-line/50 hover:bg-s2/60', revoked && 'bg-critical/5')}>
                  <td className="mono px-3 py-1.5 text-ink-1">{a.id}</td>
                  <td className="px-3 text-ink-2">{a.class}</td>
                  <td className="mono max-w-[200px] truncate px-3 text-ink-3"><KeyRound size={11} className="mr-1 inline" />{a.key_id.replace('kms:spectrum/', '')}</td>
                  <td className="mono px-3 text-ink-2">{k ? k.fingerprint.slice(0, 23) : '…'}</td>
                  <td className="tnum px-3">L{a.autonomy}</td>
                  <td className="tnum px-3" style={{ color: a.trust < 50 ? 'var(--critical-ink)' : undefined }}>{Math.round(a.trust)}</td>
                  <td className="px-3">{revoked ? <StatusPill status="denied" size="sm">Revoked</StatusPill> : <StatusPill status="verified" size="sm">Active</StatusPill>}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ delegation

export function Delegations() {
  const dels: Rec[] = useLive((s) => s.trust.delegations ?? EMPTY)
  const list = [...dels].reverse().slice(0, 4)
  return (
    <Panel id="delegation" plane="prove" title="Delegation-chain checks" subtitle="an action is allowed only if every hop is allowed" source={SRC_AVP}>
      {!list.length ? (
        <>
          <EmptyState title="No delegated actions yet" body="When an agent asks another agent to act for it, each hop is authorised separately and the result is the AND. Run the prompt-injection attack to see a confused-deputy attempt refused." />
          <ChainViz chain={['site-17', 'transport']} hops={[{ principal: 'site-17', decision: 'deny', policies: ['P-002'] }, { principal: 'transport', decision: 'permit', policies: ['P-004'] }]} final="deny" action="cell_sleep" illustrative />
        </>
      ) : list.map((d, i) => (
        <div key={i} className={clsx(i > 0 && 'mt-4 border-t border-line pt-4')}>
          <div className="mb-2 flex items-center gap-2 text-micro text-ink-3"><span className="mono">{clockTime(d.ts)}</span> request <span className="mono text-ink-2">{d.action}</span> on <span className="mono text-ink-2">{d.resource}</span></div>
          <ChainViz chain={d.chain} hops={d.hops} final={d.final} action={d.action} />
        </div>
      ))}
    </Panel>
  )
}

function ChainViz({ hops, final, illustrative }: { chain: string[]; hops: Rec[]; final: string; action: string; illustrative?: boolean }) {
  return (
    <div className={clsx('flex flex-wrap items-center gap-2', illustrative && 'opacity-60')}>
      {hops.map((h, i) => (
        <div key={h.principal + i} className="flex items-center gap-2">
          {i > 0 && <ArrowRight size={14} className="text-ink-3" />}
          <div className="rounded-md border border-line bg-s2 px-3 py-2">
            <div className="mono text-cap text-ink-1">{h.principal}</div>
            <div className="mt-1 flex items-center gap-1.5">
              <StatusPill status={DECISION_STATUS[h.decision] ?? 'neutral'} size="sm">{DECISION_WORD[h.decision] ?? h.decision}</StatusPill>
              <span className="mono text-micro text-ink-3">{(h.policies ?? []).join(', ')}</span>
            </div>
          </div>
        </div>
      ))}
      <span className="mx-1 text-cap text-ink-3">AND =</span>
      <StatusPill status={DECISION_STATUS[final] ?? 'neutral'}>{final === 'deny' ? 'Denied (P-006)' : DECISION_WORD[final]}</StatusPill>
      {illustrative && <span className="text-micro text-ink-3">illustration</span>}
    </div>
  )
}

// ------------------------------------------------------------------ guardrails

export function Guardrails() {
  const counts = useLive((s) => s.trust.guardrail_counts ?? NONE)
  const blocks: Rec[] = useLive((s) => s.trust.guardrail_blocks ?? EMPTY)
  const labels: [string, string][] = [['prompt_attack', 'Prompt attack'], ['pii', 'PII'], ['denied_topic', 'Denied topic'], ['grounding', 'Grounding']]
  return (
    <Panel id="guardrails" plane="prove" title="Guardrails" subtitle="on telemetry, tool inputs and agent-to-agent messages" source={{ aws: 'Amazon Bedrock Guardrails', mock: 'Pattern filter' }}>
      <div className="grid grid-cols-4 gap-3">
        {labels.map(([k, l]) => (
          <div key={k} className="rounded-md border border-line bg-s2 px-3 py-2">
            <div className="text-micro text-ink-3">{l}</div>
            <div className={clsx('tnum text-[20px] font-semibold', (counts[k] ?? 0) > 0 ? 'text-critical-ink' : 'text-ink-1')}>{counts[k] ?? 0}</div>
          </div>
        ))}
      </div>
      <div className="mt-3 space-y-2">
        {!blocks.length && <div className="text-cap text-ink-3">No blocked content yet.</div>}
        {[...blocks].reverse().slice(0, 5).map((b, i) => (
          <div key={i} className="rounded-md border border-critical/40 bg-critical/5 px-3 py-2">
            <div className="flex items-center gap-2 text-micro text-ink-3">
              <ShieldX size={12} className="text-critical-ink" /> <span className="text-critical-ink">{b.action}</span> {b.filter.replace('_', ' ')} from <span className="mono text-ink-2">{b.source}</span>
              <span className="mono ml-auto">{clockTime(b.ts)}</span>
            </div>
            <div className="mono mt-1 truncate text-cap text-ink-1" title={b.excerpt}>“{b.excerpt}”</div>
          </div>
        ))}
      </div>
    </Panel>
  )
}

// ------------------------------------------------------------------ attestation chain

export function AttestationChain() {
  const at = useLive((s) => s.trust.attestation)
  if (!at) return null
  const steps: { label: string; detail: React.ReactNode }[] = [
    { label: at.root, detail: 'AWS-operated root certificate' },
    { label: 'Enclave certificate', detail: 'issued to this Nitro Enclave instance' },
    { label: 'Attestation document', detail: <span className="mono">PCR0 {at.pcr0.slice(0, 12)}…</span> },
    { label: 'KMS key policy', detail: <span className="mono">{at.condition}</span> },
    { label: 'Mask seeds released', detail: 'only to the measured aggregator image' },
  ]
  return (
    <Panel id="attestation-chain" plane="prove" title="Enclave attestation chain" subtitle={at.enclave_image} source={{ aws: 'AWS Nitro Enclaves + KMS', mock: 'Simulated attestation' }}
      actions={<StatusPill status={at.verified ? 'verified' : 'denied'} size="sm">{at.verified ? 'Verified' : 'Failed'}</StatusPill>}>
      <ol className="flex flex-wrap items-stretch gap-2">
        {steps.map((s, i) => (
          <li key={s.label} className="flex items-center gap-2">
            {i > 0 && <ArrowRight size={14} className="text-ink-3" />}
            <div className="rounded-md border border-line bg-s2 px-3 py-2">
              <div className="flex items-center gap-1.5 text-cap text-ink-1"><Check size={12} className="text-good-ink" />{s.label}</div>
              <div className="mt-0.5 text-micro text-ink-3">{s.detail}</div>
            </div>
          </li>
        ))}
      </ol>
      <KV className="mt-4" rows={[
        ['PCR0 (image)', <Hash value={at.pcr0} head={16} tail={8} />],
        ['PCR1 (kernel)', <Hash value={at.pcr1} head={16} tail={8} />],
        ['PCR2 (application)', <Hash value={at.pcr2} head={16} tail={8} />],
        ['KMS key', <span className="mono break-all">{at.kms_key}</span>],
      ]} />
      <div className="mt-2 flex items-center gap-1.5 text-micro text-ink-3"><ShieldCheck size={12} /> Rebuilding the aggregator changes PCR0, and the key stops releasing seeds, for us too.</div>
    </Panel>
  )
}
