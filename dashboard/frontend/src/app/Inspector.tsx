import { ArrowLeft, Check, Loader2, ShieldCheck, ShieldX, X } from 'lucide-react'
import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { api } from '@/core/api'
import { recordHash } from '@/core/canonical'
import { clockTime, shortHash, usd } from '@/core/format'
import { EMPTY, useLive } from '@/core/store'
import type { LedgerRecord, Rec } from '@/core/types'
import { useUi, type InspectTarget } from '@/core/ui-store'
import { Button, CodeBlock, Hash, KV, StatusPill, Tabs, LedgerLink, SignatureBadge } from '@/ui'

// Right-hand inspector: every badge, hash and row in the console can open its raw artefact here.

export function Inspector() {
  const stack = useUi((s) => s.inspector)
  const close = useUi((s) => s.closeInspector)
  const back = useUi((s) => s.backInspector)
  const t = stack[stack.length - 1]
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && stack.length) close() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [stack.length, close])
  if (!t) return null
  return (
    <aside aria-label="Inspector" className="absolute inset-y-0 right-0 z-40 flex w-[440px] flex-col border-l border-line bg-s1 shadow-[-12px_0_32px_rgba(0,0,0,0.25)]">
      <header className="flex h-12 shrink-0 items-center gap-2 border-b border-line px-3">
        {stack.length > 1 && <button type="button" onClick={back} aria-label="Back" className="grid size-7 place-items-center rounded text-ink-3 hover:bg-s2 hover:text-ink-1"><ArrowLeft size={15} /></button>}
        <div className="min-w-0">
          <div className="text-micro text-ink-3">{KIND_LABEL[t.kind]}</div>
          <div className="truncate text-title font-medium text-ink-1">{t.title ?? t.id}</div>
        </div>
        <button type="button" onClick={close} aria-label="Close inspector" className="ml-auto grid size-7 place-items-center rounded text-ink-3 hover:bg-s2 hover:text-ink-1"><X size={16} /></button>
      </header>
      <div className="min-h-0 flex-1 overflow-auto p-4">
        <Body t={t} key={`${t.kind}-${t.id}`} />
      </div>
    </aside>
  )
}

const KIND_LABEL: Record<string, string> = {
  ledger: 'Ledger record', agent: 'Agent', message: 'A2A message', policy: 'Cedar policy', decision: 'Policy decision',
  checkpoint: 'Checkpoint', model: 'Model', approval: 'Approval', operator: 'Operator', site: 'Cell site', trace: 'Trace', raw: 'Raw data',
}

function Body({ t }: { t: InspectTarget }) {
  switch (t.kind) {
    case 'ledger': return <LedgerBody seq={Number(t.id)} />
    case 'agent': return <AgentBody id={t.id} />
    case 'message': return <MessageBody m={t.data!} />
    case 'policy': return <PolicyBody p={t.data!} />
    default: return <RawBody data={t.data ?? {}} />
  }
}

function Section({ title, children, right }: { title: string; children: ReactNode; right?: ReactNode }) {
  return (
    <section className="mb-5">
      <div className="mb-2 flex items-center justify-between">
        <h4 className="text-cap font-medium text-ink-2">{title}</h4>
        {right}
      </div>
      {children}
    </section>
  )
}

function Check1({ ok, label, detail }: { ok: boolean | null; label: string; detail?: ReactNode }) {
  return (
    <li className="flex items-start gap-2 py-1">
      {ok == null ? <Loader2 size={14} className="mt-0.5 animate-spin text-ink-3" /> : ok ? <ShieldCheck size={14} className="mt-0.5 text-good-ink" /> : <ShieldX size={14} className="mt-0.5 text-critical-ink" />}
      <div className="min-w-0">
        <div className="text-cap text-ink-1">{label}</div>
        {detail && <div className="text-micro text-ink-3">{detail}</div>}
      </div>
    </li>
  )
}

function LedgerBody({ seq }: { seq: number }) {
  const [data, setData] = useState<{ record: LedgerRecord; verification: Rec; public_key: Rec } | null>(null)
  const [client, setClient] = useState<string | null>(null)
  const [proof, setProof] = useState<Rec | null>(null)
  const [tab, setTab] = useState<'proof' | 'payload' | 'raw'>('proof')
  useEffect(() => {
    api.record(seq).then(async (d) => {
      setData(d)
      setClient(await recordHash(d.record))
    })
    api.proof(seq).then(setProof).catch(() => {})
  }, [seq])
  if (!data) return <div className="text-cap text-ink-3">Loading record #{seq}…</div>
  const r = data.record
  const v = data.verification
  const ok = v.ok
  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <StatusPill status={ok ? 'verified' : 'tampered'} />
        <span className="mono text-cap text-ink-2">{r.type}</span>
        <span className="text-cap text-ink-3">by</span>
        <span className="mono text-cap text-ink-1">{r.actor}</span>
        <span className="ml-auto tnum mono text-micro text-ink-3">{clockTime(r.ts)}</span>
      </div>
      <Tabs value={tab} onChange={setTab} tabs={[{ value: 'proof', label: 'Proof' }, { value: 'payload', label: 'Payload' }, { value: 'raw', label: 'Raw' }]} />
      <div className="mt-4">
        {tab === 'proof' && (
          <>
            <Section title="Verification, step by step">
              <ol className="rounded-md border border-line bg-s2 px-3 py-1">
                <Check1 ok={client == null ? null : client === r.hash} label="1. Recompute SHA-256 in this browser"
                  detail={<>SHA-256(canonical(body) ‖ prev_hash) = <span className="mono">{shortHash(client, 10, 6)}</span></>} />
                <Check1 ok={v.checks.hash_matches} label="2. Matches the stored hash" detail={<span className="mono">{shortHash(r.hash, 10, 6)}</span>} />
                <Check1 ok={v.checks.link_matches} label="3. Links to the previous record" detail={<span className="mono">prev {shortHash(r.prev_hash, 10, 6)}</span>} />
                <Check1 ok={v.checks.signature_valid} label={`4. ${r.alg} signature is valid`} detail={<span className="mono">{data.public_key.fingerprint}</span>} />
                <Check1 ok={proof ? (proof.anchored ? proof.matches : null) : null}
                  label={proof?.anchored ? `5. Included in anchor #${proof.anchor.n} (S3 Object Lock)` : '5. Awaiting the next Merkle anchor'}
                  detail={proof?.anchored ? <>Merkle path of {proof.path.length} hashes, root <span className="mono">{shortHash(proof.anchor.root, 8, 6)}</span></> : proof ? `anchors every 32 records (next at #${proof.pending_until})` : undefined} />
              </ol>
            </Section>
            <Section title="Signature">
              <KV rows={[
                ['Algorithm', r.alg],
                ['Signing key', <span className="mono break-all">{r.key_id}</span>],
                ['Public key', <Hash value={data.public_key.public_key_hex} head={12} tail={8} />],
                ['Fingerprint', <span className="mono">{data.public_key.fingerprint}</span>],
                ['Signature', <Hash value={r.sig} head={12} tail={8} />],
                ['Key status', data.public_key.revoked_at ? <StatusPill status="denied" size="sm">Revoked {clockTime(data.public_key.revoked_at, false)}</StatusPill> : <StatusPill status="verified" size="sm">Active</StatusPill>],
              ]} />
            </Section>
            {proof?.anchored && (
              <Section title="Anchor">
                <KV rows={[
                  ['Records', `#${proof.anchor.range[0]}–#${proof.anchor.range[1]}`],
                  ['Merkle root', <Hash value={proof.anchor.root} head={10} tail={6} />],
                  ['Object', <span className="mono break-all">{proof.anchor.s3_uri}</span>],
                  ['Object Lock', `${proof.anchor.object_lock}, retain until ${proof.anchor.retain_until}`],
                  ['Version id', <span className="mono">{shortHash(proof.anchor.version_id, 10, 4)}</span>],
                ]} />
              </Section>
            )}
          </>
        )}
        {tab === 'payload' && <CodeBlock code={JSON.stringify(r.payload, null, 2)} maxHeight={520} />}
        {tab === 'raw' && <CodeBlock code={JSON.stringify(r, null, 2)} maxHeight={520} />}
      </div>
    </div>
  )
}

function AgentBody({ id }: { id: string }) {
  const agent = useLive((s) => (s.swarm.agents ?? []).find((a: Rec) => a.id === id))
  const allMsgs = useLive((s) => s.swarm.messages ?? EMPTY)
  const msgs = useMemo(() => allMsgs.filter((m: Rec) => m.from === id || m.to === id).slice(-8).reverse(), [allMsgs, id])
  const [key, setKey] = useState<Rec | null>(null)
  const inspect = useUi((s) => s.inspect)
  useEffect(() => { if (agent) api.key(agent.key_id).then(setKey).catch(() => {}) }, [agent?.key_id])
  if (!agent) return <RawBody data={{ id }} />
  return (
    <div>
      <div className="mb-4 flex items-center gap-2">
        <StatusPill status={agent.status === 'quarantined' ? 'quarantined' : 'verified'}>{agent.status === 'quarantined' ? 'Quarantined' : 'Active'}</StatusPill>
        <span className="text-cap text-ink-2">{agent.class} agent</span>
        <span className="ml-auto text-cap text-ink-3">autonomy</span><span className="tnum font-medium">L{agent.autonomy}</span>
      </div>
      <Section title="Identity">
        <KV rows={[
          ['Key', <span className="mono break-all">{agent.key_id}</span>],
          ['Fingerprint', <span className="mono">{key?.fingerprint ?? '…'}</span>],
          ['Status', key?.revoked_at ? <StatusPill status="denied" size="sm">Revoked</StatusPill> : <StatusPill status="verified" size="sm">Active</StatusPill>],
          ['Model', `${agent.model} on ${agent.provider}`],
          ['Trust score', `${agent.trust.toFixed(0)} / 100`],
          ['Allowed sites', agent.allowed_sites ? agent.allowed_sites.join(', ') : 'n/a'],
        ]} />
      </Section>
      <Section title="Last hour">
        <KV rows={[['Tokens', agent.tokens_1h.toLocaleString()], ['Cost', usd(agent.usd_1h)], ['Last action', agent.last_action || '—']]} />
      </Section>
      <Section title="Recent messages">
        <ul className="space-y-1">
          {msgs.map((m: Rec) => (
            <li key={m.id}>
              <button type="button" onClick={() => inspect({ kind: 'message', id: m.id, title: m.summary, data: m })} className="flex w-full items-center gap-2 rounded px-1 py-1 text-left text-cap hover:bg-s2">
                <span className="mono shrink-0 text-ink-3">{m.from === id ? `→ ${m.to}` : `← ${m.from}`}</span>
                <span className="truncate text-ink-1">{m.summary}</span>
                {m.verdict === 'blocked' && <StatusPill status="blocked" size="sm" />}
              </button>
            </li>
          ))}
        </ul>
      </Section>
      <div className="flex gap-2">
        {agent.status === 'quarantined'
          ? <Button size="sm" onClick={() => api.agent(id, 'release')}>Release with new identity</Button>
          : <Button size="sm" variant="danger" onClick={() => api.agent(id, 'quarantine')}>Quarantine</Button>}
      </div>
    </div>
  )
}

function MessageBody({ m }: { m: Rec }) {
  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <StatusPill status={m.verdict === 'blocked' ? 'blocked' : 'verified'}>{m.verdict === 'blocked' ? 'Blocked' : 'Delivered'}</StatusPill>
        <span className="mono text-cap text-ink-2">{m.intent}</span>
        <span className="mono text-cap text-ink-1">{m.from} → {m.to}</span>
      </div>
      <p className="mb-4 text-body text-ink-1">{m.summary}</p>
      <Section title="Checks">
        <ol className="rounded-md border border-line bg-s2 px-3 py-1">
          <Check1 ok={m.verified} label="Signature valid for the sender’s key" detail={<span className="mono">{m.key_id}</span>} />
          <Check1 ok={m.authority_ok} label="Claimed authority matches the signer" detail={m.authority_ok ? `${m.from} speaks for itself` : `claims to speak for ${m.claimed_authority} but is signed by ${m.from}`} />
          <Check1 ok={!m.guardrail} label="Guardrails (prompt attack filter)" detail={m.guardrail ? `blocked: ${m.guardrail}` : 'no findings'} />
        </ol>
      </Section>
      <Section title="Envelope">
        <KV rows={[
          ['Delegation chain', m.chain.join(' → ')],
          ['Digest', <Hash value={m.digest} head={10} tail={6} />],
          ['Signature', <Hash value={m.sig} head={10} tail={6} />],
          ['Ledger', <LedgerLink seq={m.ledger_seq} />],
        ]} />
      </Section>
      <CodeBlock code={JSON.stringify(m.body, null, 2)} />
    </div>
  )
}

function PolicyBody({ p }: { p: Rec }) {
  return (
    <div>
      <div className="mb-3 flex items-center gap-2">
        <StatusPill status={p.effect === 'forbid' ? 'denied' : 'verified'}>{p.effect}</StatusPill>
        <span className="text-cap text-ink-2">version {p.version}</span>
        <span className="ml-auto text-cap text-ink-3">{(p.hits ?? 0).toLocaleString()} decisions</span>
      </div>
      <CodeBlock code={p.cedar} lang="cedar" />
      <div className="mt-4"><KV rows={[['Policy hash', <Hash value={p.hash} head={10} tail={6} />], ['Store', 'Amazon Verified Permissions']]} /></div>
    </div>
  )
}

function RawBody({ data }: { data: Rec }) {
  return (
    <div>
      {data.sig && <div className="mb-3"><SignatureBadge keyId={data.key_id} ledgerSeq={data.ledger_seq} /></div>}
      <CodeBlock code={JSON.stringify(data, null, 2)} maxHeight={640} />
      <div className="mt-2 flex items-center gap-1 text-micro text-ink-3"><Check size={12} /> Values shown exactly as received from the backend.</div>
    </div>
  )
}
