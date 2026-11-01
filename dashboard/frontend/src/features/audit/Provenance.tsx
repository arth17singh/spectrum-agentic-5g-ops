import {
  Bot, Box, CircleCheck, Cpu, FileCheck2, FlaskConical, Gavel, Network, RefreshCw, Save, ShieldCheck, UserCheck, Wrench, Building2,
} from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { DagGraph, type DagNode } from '@/charts/DagGraph'
import { api } from '@/core/api'
import { useOnBeat } from '@/core/director'
import { clockTime, shortHash } from '@/core/format'
import { opColor } from '@/core/palette'
import type { Rec } from '@/core/types'
import { useUi } from '@/core/ui-store'
import { Button, EmptyState, Panel, StatusPill } from '@/ui'

const KIND: Record<string, { label: string; color: string; Icon: typeof Bot }> = {
  action: { label: 'Action', color: 'var(--operate)', Icon: Wrench },
  policy: { label: 'Policy decision', color: 'var(--prove)', Icon: Gavel },
  approval: { label: 'Human approval', color: 'var(--prove)', Icon: UserCheck },
  agent: { label: 'Agent', color: 'var(--operate)', Icon: Bot },
  model: { label: 'Model', color: 'var(--verify)', Icon: Cpu },
  aibom: { label: 'AI-BOM', color: 'var(--verify)', Icon: FileCheck2 },
  eval: { label: 'Deploy gate', color: 'var(--verify)', Icon: FlaskConical },
  checkpoint: { label: 'Checkpoint', color: 'var(--learn)', Icon: Save },
  fl_round: { label: 'Federated round', color: 'var(--learn)', Icon: Network },
  operator: { label: 'Operator', color: 'var(--learn)', Icon: Building2 },
  attestation: { label: 'Attestation', color: 'var(--learn)', Icon: ShieldCheck },
}

function detailOf(n: Rec): string {
  switch (n.type) {
    case 'operator': return `ε ${Number(n.eps).toFixed(2)}, δ 1e-5`
    case 'model': return n.accuracy != null ? `acc ${(n.accuracy * 100).toFixed(1)}%, safety ${(n.safety * 100).toFixed(1)}%` : ''
    case 'fl_round': return `σ ${n.sigma}, ${Object.keys(n.decisions ?? {}).length} operators`
    case 'agent': return n.model ? `${n.model}` : ''
    case 'action': return n.ts ? clockTime(n.ts) : ''
    default: return ''
  }
}

export function ProvenancePanel() {
  const [g, setG] = useState<Rec | null>(null)
  const [sel, setSel] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const inspect = useUi((s) => s.inspect)
  const load = useCallback(() => {
    setLoading(true)
    api.provenance().then((x: Rec) => { setG(x); setSel(lastLeaf(x)) }).finally(() => setLoading(false))
  }, [])
  useEffect(() => { load() }, [load])
  useOnBeat(11, () => {
    load()
    setTimeout(() => document.querySelector('[data-panel="provenance"]')?.scrollIntoView({ block: 'center', behavior: 'smooth' }), 300)
  })
  const all: Rec[] = g?.nodes ?? []
  const ops = all.filter((n) => n.type === 'operator')
  const nodes: DagNode[] = all.filter((n) => n.type !== 'operator').map((n: Rec) => {
    const k = KIND[n.type] ?? { label: n.type, color: 'var(--ink-3)', Icon: Box }
    const status = n.status === 'quarantined' ? 'quarantined' : n.status === 'rejected' ? 'rejected' : 'verified'
    return {
      id: n.id,
      color: k.color,
      render: (
        <span className="flex h-full flex-col justify-between">
          <span className="flex items-center gap-1.5 text-micro text-ink-3">
            <k.Icon size={12} />{k.label}
            <span className="ml-auto">{status === 'verified' ? <CircleCheck size={12} className="text-good-ink" aria-label="verified" /> : <StatusPill status={status} size="sm" />}</span>
          </span>
          <span className="truncate text-cap font-medium text-ink-1">{n.label}</span>
          <span className="mono truncate text-[10px] leading-3 text-ink-3">
            {n.hash ? shortHash(n.hash, 8, 4) : detailOf(n)}{n.ledger_seq ? `  #${n.ledger_seq.toLocaleString()}` : ''}
          </span>
        </span>
      ),
    }
  })
  // the five operators collapse into one group node so the lineage stays readable
  if (ops.length) {
    nodes.push({
      id: 'operators', color: 'var(--learn)', height: 34 + ops.length * 19,
      render: (
        <span className="flex h-full flex-col">
          <span className="mb-1 flex items-center gap-1.5 text-micro text-ink-3"><Building2 size={12} />Operators, private data</span>
          {ops.map((o) => (
            <span key={o.id} className="flex items-center gap-1.5 text-[11px] leading-[19px] whitespace-nowrap">
              <span className="size-1.5 shrink-0 rounded-full" style={{ background: opColor(o.label) }} />
              <span className="mono shrink-0 text-ink-1">{o.label}</span>
              <span className="tnum shrink-0 text-ink-2">ε {Number(o.eps).toFixed(2)}</span>
              <span className="mono ml-auto truncate text-[10px] text-ink-3" title={`data commitment ${o.commitment}`}>{String(o.commitment ?? '').slice(0, 8)}</span>
              {o.status === 'rejected' ? <StatusPill status="rejected" size="sm" /> : <CircleCheck size={11} className="shrink-0 text-good-ink" />}
            </span>
          ))}
        </span>
      ),
    })
  }
  const edges = (g?.edges ?? []).filter((e: Rec) => !String(e.to).startsWith('op-'))
  const opEdge = (g?.edges ?? []).find((e: Rec) => String(e.to).startsWith('op-'))
  if (opEdge) edges.push({ from: opEdge.from, to: 'operators', rel: 'contributed (DP)' })
  const open = (id: string) => {
    setSel(id)
    if (id === 'operators') { inspect({ kind: 'raw', id, title: 'Operators that contributed (data commitments only)', data: { operators: ops } }); return }
    const n = g?.nodes.find((x: Rec) => x.id === id)
    if (!n) return
    if (n.type === 'agent') inspect({ kind: 'agent', id: id.replace(/^agent-/, '') })
    else if (n.ledger_seq) inspect({ kind: 'ledger', id: String(n.ledger_seq) })
    else inspect({ kind: 'raw', id, title: `${KIND[n.type]?.label ?? n.type}: ${n.label}`, data: n })
  }
  return (
    <Panel id="provenance" plane="prove" title="Provenance" subtitle="why did the network do that? from the latest executed action back to training"
      source={{ aws: 'DynamoDB ledger + S3 model registry', mock: 'Built from the ledger' }}
      actions={<Button size="sm" variant="ghost" icon={<RefreshCw size={13} className={loading ? 'animate-spin' : ''} />} onClick={load}>Refresh</Button>}>
      {!g ? <div className="text-cap text-ink-3">Building provenance graph…</div> : !g.nodes?.length ? <EmptyState title="No executed actions yet" /> : (
        <>
          <DagGraph root={g.root} nodes={nodes} edges={edges} selected={sel} onSelect={open} />
          <p className="mt-2 text-micro text-ink-3">
            Every box is a signed ledger record or a hash-committed artefact. The highlighted path runs from the selected box back to the action; click any box for its proof.
          </p>
        </>
      )}
    </Panel>
  )
}

/** select the deepest node (an operator or the attestation) so the full path lights up */
function lastLeaf(g: Rec): string | null {
  if (g.nodes?.some((n: Rec) => n.type === 'operator')) return 'operators'
  const att = g.nodes?.find((n: Rec) => n.type === 'attestation')
  return att?.id ?? g.nodes?.[g.nodes.length - 1]?.id ?? null
}
