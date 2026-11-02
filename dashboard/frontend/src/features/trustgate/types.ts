// Shapes returned by /api/trustgate/*. The backend copies recorded values; the page never recomputes them.

export interface KpiRow { key: string; predicted_delta: number | null; measured_delta: number | null; error: number | null }

export interface ChangeRow {
  change_id: string | null
  item_id: string | null
  case_id: string | null
  run_id: string | null
  time: string | null
  t: number
  change_text: string | null
  probe: boolean
  scored: boolean
  unscored_reason: string | null
  kpis: KpiRow[]
  error_summary: number | null
  direction_agree: boolean | null
  over_tau: boolean | null
  trust_before: number | null
  trust_after: number | null
  twin: { kind?: string; version?: string; scenario?: string; seed?: number; step?: number }
  synthetic: boolean
}

export interface TrustPoint {
  time: string | null
  t: number
  change_id: string | null
  trust_before: number | null
  trust_after: number | null
  error_summary: number | null
  direction_agree: boolean | null
  over_tau: boolean | null
}

export interface PauseEvent {
  time: string
  t: number
  action: 'pause' | 'resume'
  by: string
  by_kind: 'trust-gate' | 'person'
  reason: string
  policy_name: string | null
  run_id: string | null
  synthetic: boolean
}

export interface Thresholds {
  tau?: number
  scales?: Record<string, number>
  pause_below?: number
  watch_below?: number
  alpha?: number
  warmup?: number
  n_min?: number
  window?: number
  k_of?: number
  source?: string
}

export interface ModelSummary {
  model_id: string
  version: string
  paused: boolean
  paused_by: string | null
  paused_by_kind: 'trust-gate' | 'person' | null
  paused_at: string | null
  model_status: string | null
  status_basis: 'control' | 'model' | 'none'
  status_conflict: boolean
  model_kind: string | null
  has_model_record: boolean
  thresholds: Thresholds | null
  pause_below: number | null
  n_changes: number
  n_scored: number
  n_unscored: number
  n_direction_agree: number
  n_direction_known: number
  trust_latest: number | null
  pauses: PauseEvent[]
  points: TrustPoint[]
  synthetic: boolean
}

export interface ModelDetail extends ModelSummary { rows: ChangeRow[] }

export interface ModelsResponse { source: unknown; error: string | null; models: ModelSummary[]; probes: ChangeRow[] }
export interface ChangesResponse extends ModelDetail { source: unknown; error: string | null }

export type RungLabel = 'MAVERIC' | 'REHEARSAL (Viavi-like Maveric)' | 'VIAVI'

export interface Rung {
  name: string | null
  label: RungLabel
  twin_kind: string | null
  n_cases: number | null
  gain: number | null
  gain_ci: [number, number] | null
  label_ok: boolean
  label_problem: string | null
}

export interface Ladder {
  run_id: string | null
  checked_at: string | null
  model_id: string
  version: string
  rungs: Rung[]
  top_label: RungLabel
  has_viavi: boolean
  label_ok: boolean
  retention: number | null
  retention_ci: [number, number] | null
  passed: boolean | null
  target: { metric: string; op: string; value: number; unit: string } | null
  synthetic: boolean
}

export interface LadderResponse { source: unknown; error: string | null; ladders: Ladder[] }
