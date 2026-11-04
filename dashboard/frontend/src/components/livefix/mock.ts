// PREVIEW DATA ONLY. Illustrative values for LiveFixPreview; never wire these into a real screen. Every number a
// judge sees on the real Live Fix page must come from the backend (livefix slice) and its evidence records.
import type { BeforeAfter, CellState, Lesson, ReliabilityBin, Stage } from './types'

export const MOCK_CELLS_GREEN: CellState[] = [
  { id: 0, power: 0, tilt: 2, health: 'good', kpis: { coverage: 0.96 } },
  { id: 1, power: 1, tilt: 1, health: 'good', kpis: { coverage: 0.95 } },
  { id: 2, power: -1, tilt: 2, health: 'good', kpis: { coverage: 0.97 } },
  { id: 3, power: 0, tilt: 3, health: 'good', kpis: { coverage: 0.94 } },
]

export const MOCK_CELLS_FAULT: CellState[] = MOCK_CELLS_GREEN.map((c) =>
  c.id === 2
    ? { ...c, power: -7, health: 'critical', kpis: { coverage: 0.61 }, before: { power: c.power, tilt: c.tilt, kpis: c.kpis } }
    : c.id === 1 ? { ...c, health: 'degraded', kpis: { coverage: 0.88 }, before: { power: c.power, tilt: c.tilt, kpis: c.kpis } } : c,
)

export const MOCK_STAGES: Stage[] = [
  { key: 'green', state: 'done', elapsedS: 0 },
  { key: 'fault', state: 'done', elapsedS: 0.4, awsTime: '2026-11-04T06:30:01.120Z', awsTimeSource: 'cloudwatch_lambda' },
  { key: 'diagnosing', state: 'done', elapsedS: 31.8, awsTime: '2026-11-04T06:30:33.010Z', awsTimeSource: 'gateway_date', detail: 'lead → monitor ×2 → optimizer' },
  { key: 'awaiting_approval', state: 'active', elapsedS: 33.2, detail: 'ran___apply_change held' },
]

export const MOCK_BINS: ReliabilityBin[] = [
  { lo: 0, hi: 0.2, meanConf: 0.12, observed: 0, count: 2 },
  { lo: 0.2, hi: 0.4, meanConf: 0.33, observed: 0.25, count: 4 },
  { lo: 0.4, hi: 0.6, meanConf: 0.51, observed: 0.5, count: 8 },
  { lo: 0.6, hi: 0.8, meanConf: 0.71, observed: 0.64, count: 14 },
  { lo: 0.8, hi: 1, meanConf: 0.9, observed: 0.85, count: 26 },
]

export const MOCK_LESSONS: Lesson[] = [
  { id: 'L-07', symptoms: 'coverage drop, neighbour load up, power low', change: 'power +6 dB', source: 'dev incident inc-off-vis-1187', verified: true },
  { id: 'L-08', symptoms: 'coverage hole, tilt high', change: 'tilt −5°', source: 'dev incident inc-vvl-vis-1203', verified: true },
  { id: 'L-09', symptoms: 'SLA dip on two cells', change: 'tilt −2° on cell 1', source: 'dev incident inc-off-vis-1220', verified: false, reason: 'the cell did not recover' },
]

export const MOCK_BEFORE_AFTER: BeforeAfter = {
  recoveredBefore: 13, recoveredAfter: 15, n: 18, ciLow: -0.02, target: 0, stepsBefore: 1.9, stepsAfter: 1.5,
  harmfulBefore: 4, harmfulAfter: 2, runId: 't3.lessons-PREVIEW',
}

export const MOCK_ROUNDS = Array.from({ length: 20 }, (_, i) => 0.62 + 0.2 * (1 - Math.exp(-i / 5)))
