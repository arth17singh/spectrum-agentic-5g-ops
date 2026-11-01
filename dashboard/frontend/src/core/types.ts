// Wire types shared with the backend (spectrum/*). Engine payloads are large and evolve quickly,
// so slices are typed where screens depend on them and left open (`Rec`) elsewhere.

export type Plane = 'learn' | 'verify' | 'release' | 'operate' | 'prove' | 'system'
export type Severity = 'debug' | 'info' | 'notice' | 'warn' | 'critical'
export type Rec = Record<string, any>

export interface Envelope {
  seq: number
  ts: string
  sim_t: number
  plane: Plane
  type: string
  severity: Severity
  actor: string
  summary: string
  data: Rec
  ledger?: { seq: number; hash: string }
}

export interface LedgerRecord {
  seq: number
  ts: string
  type: string
  actor: string
  plane: Plane
  payload: Rec
  key_id: string
  prev_hash: string
  hash: string
  alg: string
  sig: string
}

export interface Operator {
  id: string
  short: string
  name: string
  vendor: string
  region: string
  samples: number
  q: number
  status: string
  participating: boolean
  step: number
  loss: number
  clip_frac: number
  eps: number
  update_norm: number
  gpu: number
  bytes_up: number
  local_only: number
  commitment: string
  grad_hist: number[]
  phase: string
}

export interface FlRound {
  round: number
  t0: number
  t1: number
  val_loss: number
  accuracy: number
  local_only: Record<string, number>
  eps: Record<string, number>
  update_norms: Record<string, number>
  threshold: number
  median_norm: number
  decisions: Record<string, string>
  phases: Record<string, { phase: string; start: number; end?: number }[]>
  aggregator: { phase: string; start: number; end?: number }[]
  participants: string[]
  ledger_seq: number
  sigma: number
  clip: number
}

export interface Checkpoint {
  id: string
  round: number
  val_loss: number
  accuracy: number
  eps_max: number
  delta: number
  size_bytes: number
  adapter_hash: string
  manual: boolean
  s3_uri: string
  created_at: string
  promoted: boolean
  ledger_seq: number
  sig: string
}

export interface Agent {
  id: string
  class: 'site' | 'domain' | 'coordinator'
  label: string
  model: string
  provider: string
  autonomy: number
  trust: number
  state: 'idle' | 'thinking' | 'acting' | 'quarantined'
  status: 'active' | 'quarantined'
  msgs_min: number
  tokens_1h: number
  usd_1h: number
  last_action: string
  key_id: string
  site?: string
  allowed_sites?: string[]
}

export interface A2AMessage {
  id: string
  from: string
  to: string
  intent: string
  summary: string
  body: Rec
  chain: string[]
  claimed_authority: string
  ts: string
  digest: string
  sig: string
  key_id: string
  verified: boolean
  authority_ok: boolean
  guardrail: string | null
  verdict: 'delivered' | 'blocked'
  ledger_seq: number
}

export interface Approval {
  id: string
  title: string
  tier: 'high' | 'critical'
  requested_by: string
  incident: string | null
  summary: string
  plan: Rec
  preflight: Rec
  reasons: { policy: string; text: string }[]
  dissent: { agent: string; text: string }[]
  model_opinions: { model: string; role: string; proposal: string }[]
  created_at: string
  status: 'pending' | 'approved' | 'rejected' | 'expired'
  required: number
  approvals: { by: string; name: string; at: string; device: string; sig: string; ledger_seq: number }[]
  history: { ts: string; event: string; by: string; ledger_seq?: number }[]
  blast_radius_pct: number
  level: string
  level_index: number
  assignee: { id: string; name: string; role: string; level: string }
  sla_deadline_t: number
  sla_s: number
  ttd_s?: number
}

export interface Snapshot {
  seq: number
  clock: { t: number; iso: string; speed: number; paused: boolean }
  director: Rec
  federation: Rec
  proving: Rec
  release: Rec
  twin: Rec
  swarm: Rec
  trust: Rec
  ledger: Rec
  oversight: Rec
  observability: Rec
  stream: Envelope[]
}
