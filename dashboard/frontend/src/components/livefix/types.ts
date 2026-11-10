// Shared prop types for the Live Fix components. Every component here is prop-driven: no fetching, no stores.
// The faults are the incident suite's templates (t3/evals/incidents.py TEMPLATES) in operator language.

export type CellId = 0 | 1 | 2 | 3
export const CELL_IDS: readonly CellId[] = [0, 1, 2, 3]

export type FaultKey = 'tilt_drift' | 'pa_fault' | 'bad_config' | 'hardware' | 'cascade'
export type AttackKey = 'poisoned_alarm' | 'rogue_agent'

export interface FaultSpec {
  key: FaultKey
  label: string
  /** what the twin is actually told to do, shown under the label */
  twin: string
  /** the suite template it maps to */
  template: string
  cells: 1 | 2
}

export const FAULTS: readonly FaultSpec[] = [
  { key: 'tilt_drift', label: 'Antenna tilt drift', twin: 'tilt +5°', template: 'tilt+5', cells: 1 },
  { key: 'pa_fault', label: 'Power amplifier fault', twin: 'power −6 dB', template: 'power-6', cells: 1 },
  { key: 'bad_config', label: 'Bad config push', twin: 'power −3 dB, tilt +3°', template: 'power-3+tilt+3', cells: 1 },
  { key: 'hardware', label: 'Hardware failure', twin: 'power −6 dB, tilt +5°', template: 'power-6+tilt+5', cells: 1 },
  { key: 'cascade', label: 'Cascade, two cells', twin: 'cell −6 dB + neighbour tilt +5°', template: 'two-cell:power-6|tilt+5', cells: 2 },
]

export const ATTACKS: readonly { key: AttackKey; label: string; detail: string }[] = [
  { key: 'poisoned_alarm', label: 'Poisoned alarm', detail: 'alarm text carries a prompt injection' },
  { key: 'rogue_agent', label: 'Rogue agent', detail: 'a monitor agent tries to apply a change itself' },
]

export type Health = 'good' | 'degraded' | 'critical' | 'unknown'

export interface CellKpis {
  coverage?: number | null
  sla?: number | null
  throughput_mbps?: number | null
  [k: string]: number | null | undefined
}

export interface CellState {
  id: CellId
  power: number
  tilt: number
  health: Health
  kpis?: CellKpis
  /** the same cell before the fault (or before the fix), for before→after deltas */
  before?: { power: number; tilt: number; kpis?: CellKpis }
}

export type StageKey = 'green' | 'fault' | 'diagnosing' | 'awaiting_approval' | 'applied' | 'recovered' | 'evidence'
/** warn: reached, but not cleanly (e.g. the network recovered while a cell is still degraded) */
export type StageState = 'done' | 'active' | 'pending' | 'failed' | 'skipped' | 'warn'

export interface Stage {
  key: StageKey
  state: StageState
  /** ISO time AWS produced for this stage, if any */
  awsTime?: string | null
  awsTimeSource?: string | null
  /** seconds since the fault was injected */
  elapsedS?: number | null
  detail?: string
}

export type ConfidenceTier = 'one_person' | 'two_person' | 'hold'

export interface ConfidenceBreakdown {
  /** preflight on a copy of the twin: passed + predicted KPI */
  preflight?: { passed: boolean; predicted?: number | null; metric?: string } | null
  /** federated predictor's forecast of recovery, 0..1 */
  federated?: number | null
  /** k of n agents proposed the same change */
  agree?: { k: number; n: number } | null
  /** id of a verified lesson that supports the change */
  lesson?: string | null
}

export interface Lesson {
  id: string
  symptoms: string
  change: string
  source: string
  verified: boolean
  /** why a lesson was not learned (cell did not recover, etc.) */
  reason?: string
}

export interface BeforeAfter {
  recoveredBefore: number
  recoveredAfter: number
  n: number
  ciLow?: number | null
  /** declared target for ciLow, e.g. 0 */
  target?: number | null
  stepsBefore?: number | null
  stepsAfter?: number | null
  harmfulBefore?: number | null
  harmfulAfter?: number | null
  runId?: string
}

export interface ReliabilityBin {
  /** bin lower and upper edges, 0..1 */
  lo: number
  hi: number
  /** mean stated confidence in the bin */
  meanConf: number
  /** observed success rate in the bin */
  observed: number
  count: number
}

export const STAGE_LABEL: Record<StageKey, string> = {
  green: 'All green',
  fault: 'Fault injected',
  diagnosing: 'Agents diagnosing',
  awaiting_approval: 'Cedar holds: awaiting approval',
  applied: 'Change applied',
  recovered: 'Recovered',
  evidence: 'Evidence written',
}

export const STAGE_ORDER: StageKey[] = ['green', 'fault', 'diagnosing', 'awaiting_approval', 'applied', 'recovered', 'evidence']
