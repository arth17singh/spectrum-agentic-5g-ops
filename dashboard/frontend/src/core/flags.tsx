import { create } from 'zustand'
import type { ReactNode } from 'react'

// Feature flags (DASHBOARD_SPEC.md §7). Resolution order, last wins:
// code default -> remote (/api/config/flags, AWS AppConfig in aws mode) -> localStorage -> URL ?ff=a:on,b:off
// A flag is effectively on only if every flag it depends on is also on.

export type Stage = 'stable' | 'beta' | 'experimental' | 'demo-only'

export interface FlagDef {
  id: string
  label: string
  description: string
  stage: Stage
  default: boolean
  dependsOn?: string[]
  group: string
}

export const FLAGS: FlagDef[] = [
  // screens
  { id: 'screen.mission', group: 'Screens', label: 'Mission Control', description: 'Overview and cold open', stage: 'stable', default: true },
  { id: 'screen.federation', group: 'Screens', label: 'Federation Center', description: 'Federated DP-SGD fine-tuning', stage: 'stable', default: true },
  { id: 'screen.proving', group: 'Screens', label: 'Proving Ground', description: 'Price, performance, accuracy, safety', stage: 'stable', default: true },
  { id: 'screen.release', group: 'Screens', label: 'Release Center', description: 'Multi-vendor shadow and canary', stage: 'stable', default: true },
  { id: 'screen.swarm', group: 'Screens', label: 'Swarm Tower', description: 'Agent swarm on the network twin', stage: 'stable', default: true },
  { id: 'screen.oversight', group: 'Screens', label: 'Oversight Desk', description: 'Human approvals and escalation', stage: 'stable', default: true },
  { id: 'screen.trust', group: 'Screens', label: 'Trust Center', description: 'Policy, identity, guardrails', stage: 'stable', default: true },
  { id: 'screen.audit', group: 'Screens', label: 'Audit & Proof', description: 'Ledger, anchors, provenance', stage: 'stable', default: true },
  { id: 'screen.observability', group: 'Screens', label: 'Observability', description: 'Traces, cost, latency, SLOs', stage: 'stable', default: true },
  { id: 'screen.settings', group: 'Screens', label: 'Settings & Flags', description: 'This page', stage: 'stable', default: true },
  // shell
  { id: 'shell.ribbon', group: 'Shell', label: 'Lifecycle Ribbon', description: 'Five-plane lifecycle band under the top bar', stage: 'stable', default: true },
  { id: 'shell.stream', group: 'Shell', label: 'Live stream drawer', description: 'Bottom event stream', stage: 'stable', default: true },
  { id: 'shell.palette', group: 'Shell', label: 'Command palette', description: '⌘K search and actions', stage: 'stable', default: true },
  { id: 'shell.killswitch', group: 'Shell', label: 'Kill switch', description: 'Hold-to-freeze the swarm', stage: 'stable', default: true },
  // demo
  { id: 'demo.director', group: 'Demo', label: 'Demo Director', description: 'Beat-by-beat scenario driver', stage: 'demo-only', default: true },
  { id: 'demo.spotlight', group: 'Demo', label: 'Spotlight', description: 'Dim everything except the beat’s panels', stage: 'demo-only', default: true, dependsOn: ['demo.director'] },
  { id: 'demo.tamper', group: 'Demo', label: 'Tamper simulation', description: 'Modify a ledger record to prove detection', stage: 'demo-only', default: true, dependsOn: ['screen.audit'] },
  { id: 'demo.poison', group: 'Demo', label: 'Poison injection', description: 'Compromise an operator’s next update', stage: 'demo-only', default: true, dependsOn: ['screen.federation'] },
  { id: 'demo.attacks', group: 'Demo', label: 'Attack scenarios', description: 'Telemetry prompt-injection trigger', stage: 'demo-only', default: true, dependsOn: ['screen.swarm'] },
  { id: 'demo.privileged_view', group: 'Demo', label: 'Privileged SecAgg view', description: 'Show true (unmasked) updates for explanation', stage: 'demo-only', default: true, dependsOn: ['fl.secagg_viz'] },
  // federation
  { id: 'fl.controls', group: 'Federation', label: 'Training controls', description: 'Start, pause, stop, settings', stage: 'stable', default: true, dependsOn: ['screen.federation'] },
  { id: 'fl.secagg_viz', group: 'Federation', label: 'Secure aggregation visualizer', description: 'Masked vs aggregate heatmaps', stage: 'beta', default: true, dependsOn: ['screen.federation'] },
  { id: 'fl.he', group: 'Federation', label: 'Homomorphic encryption panel', description: 'CKKS micro-demo statistics', stage: 'experimental', default: true, dependsOn: ['screen.federation'] },
  { id: 'fl.attestation', group: 'Federation', label: 'Enclave attestation', description: 'PCR measurements and KMS condition', stage: 'beta', default: true, dependsOn: ['screen.federation'] },
  { id: 'fl.sovereignty', group: 'Federation', label: 'Data sovereignty flows', description: 'Bytes per boundary', stage: 'beta', default: true, dependsOn: ['screen.federation'] },
  { id: 'fl.checkpoints', group: 'Federation', label: 'Checkpoint rail', description: 'Signed checkpoints, restore, promote', stage: 'stable', default: true, dependsOn: ['screen.federation'] },
  // verify / release
  { id: 'eval.parallel', group: 'Verification', label: 'Parallel coordinates', description: 'Model comparison across six axes', stage: 'beta', default: true, dependsOn: ['screen.proving'] },
  { id: 'eval.fed_eval', group: 'Verification', label: 'Federated evaluation', description: 'Private per-operator scores', stage: 'beta', default: true, dependsOn: ['screen.proving'] },
  { id: 'eval.scaling', group: 'Verification', label: 'Scaling curve', description: 'Params vs accuracy', stage: 'beta', default: true, dependsOn: ['screen.proving'] },
  { id: 'release.dialects', group: 'Release', label: 'Vendor dialect check', description: 'Same intent, two vendor syntaxes', stage: 'beta', default: true, dependsOn: ['screen.release'] },
  // operate
  { id: 'swarm.negotiation', group: 'Swarm', label: 'Negotiation panel', description: 'Agent trade-offs and dissent', stage: 'beta', default: true, dependsOn: ['screen.swarm'] },
  { id: 'swarm.autonomy', group: 'Swarm', label: 'Autonomy controls', description: 'TM Forum L0–L5 dial and earned autonomy', stage: 'beta', default: true, dependsOn: ['screen.swarm'] },
  { id: 'oversight.mobile', group: 'Oversight', label: 'Mobile approvals', description: '/m/approve page and QR code', stage: 'beta', default: true, dependsOn: ['screen.oversight'] },
  // prove / obs
  { id: 'audit.provenance', group: 'Audit', label: 'Provenance explorer', description: 'Action-to-training lineage graph', stage: 'beta', default: true, dependsOn: ['screen.audit'] },
  { id: 'audit.dossier', group: 'Audit', label: 'Incident dossier', description: 'Printable signed report', stage: 'beta', default: true, dependsOn: ['screen.audit'] },
  { id: 'obs.heatmap', group: 'Observability', label: 'Latency heatmap', description: 'Time × latency buckets', stage: 'beta', default: true, dependsOn: ['screen.observability'] },
]

export type Source = 'default' | 'remote' | 'local' | 'url'

interface FlagState {
  remote: Record<string, boolean>
  local: Record<string, boolean>
  url: Record<string, boolean>
  setLocal: (id: string, v: boolean | null) => void
  applyPreset: (p: Record<string, boolean>) => void
  setRemote: (r: Record<string, boolean>) => void
}

const parseUrl = (): Record<string, boolean> => {
  const out: Record<string, boolean> = {}
  const ff = new URLSearchParams(location.search).get('ff')
  for (const part of ff?.split(',') ?? []) {
    const [k, v] = part.split(':')
    if (k) out[k] = v !== 'off' && v !== '0' && v !== 'false'
  }
  return out
}

const loadLocal = (): Record<string, boolean> => {
  try {
    return JSON.parse(localStorage.getItem('spectrum.flags') ?? '{}')
  } catch {
    return {}
  }
}

export const useFlags = create<FlagState>((set, get) => ({
  remote: {},
  local: loadLocal(),
  url: parseUrl(),
  setLocal: (id, v) => {
    const local = { ...get().local }
    if (v === null) delete local[id]
    else local[id] = v
    localStorage.setItem('spectrum.flags', JSON.stringify(local))
    set({ local })
  },
  applyPreset: (p) => {
    localStorage.setItem('spectrum.flags', JSON.stringify(p))
    set({ local: p })
  },
  setRemote: (r) => set({ remote: Object.fromEntries(Object.entries(r).filter(([k]) => !k.startsWith('_'))) }),
}))

const DEFS = new Map(FLAGS.map((f) => [f.id, f]))

export function rawValue(id: string, s: FlagState = useFlags.getState()): { value: boolean; source: Source } {
  if (id in s.url) return { value: s.url[id], source: 'url' }
  if (id in s.local) return { value: s.local[id], source: 'local' }
  if (id in s.remote) return { value: s.remote[id], source: 'remote' }
  return { value: DEFS.get(id)?.default ?? true, source: 'default' }
}

export function isOn(id: string, s: FlagState = useFlags.getState(), seen = new Set<string>()): boolean {
  if (seen.has(id)) return true
  seen.add(id)
  if (!rawValue(id, s).value) return false
  return (DEFS.get(id)?.dependsOn ?? []).every((d) => isOn(d, s, seen))
}

export function useFlag(id: string): boolean {
  return useFlags((s) => isOn(id, s))
}

export function Gate({ flag, children, fallback = null }: { flag: string; children: ReactNode; fallback?: ReactNode }) {
  return useFlag(flag) ? <>{children}</> : <>{fallback}</>
}

export const PRESETS: Record<string, { label: string; description: string; values: Record<string, boolean> }> = {
  demo: { label: 'Demo (judges)', description: 'Everything on, demo controls visible', values: {} },
  full: { label: 'Full', description: 'Every flag on', values: Object.fromEntries(FLAGS.map((f) => [f.id, true])) },
  minimal: {
    label: 'Minimal', description: 'Core screens only',
    values: Object.fromEntries(FLAGS.map((f) => [f.id, f.stage === 'stable'])),
  },
  ops: {
    label: 'Ops', description: 'No demo-only controls',
    values: Object.fromEntries(FLAGS.filter((f) => f.stage === 'demo-only').map((f) => [f.id, false])),
  },
}
