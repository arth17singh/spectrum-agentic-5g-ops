import { useState } from 'react'
import { AttackResult } from './AttackResult'
import { CellHealthMap } from './CellHealthMap'
import { ConfidenceBadge } from './ConfidenceBadge'
import { FaultPanel } from './FaultPanel'
import { FederatedForecastLine, FederatedInset } from './FederatedForecastNote'
import { LessonsCard } from './LessonsCard'
import { LoopTimeline } from './LoopTimeline'
import { MOCK_BEFORE_AFTER, MOCK_BINS, MOCK_CELLS_FAULT, MOCK_CELLS_GREEN, MOCK_LESSONS, MOCK_ROUNDS, MOCK_STAGES } from './mock'
import { ReliabilityChart } from './ReliabilityChart'
import type { AttackKey } from './types'

/**
 * Route-free preview of every Live Fix component with illustrative data (see mock.ts). Mount it anywhere to eyeball
 * the components in light and dark mode; it is not a screen and is not registered as one.
 */
export function LiveFixPreview() {
  const [faulted, setFaulted] = useState(false)
  const [attack, setAttack] = useState<AttackKey | null>(null)
  return (
    <div className="flex flex-col gap-6 p-4">
      <div className="self-start rounded-[4px] bg-s3 px-1.5 py-0.5 text-micro font-medium text-ink-2">PREVIEW: illustrative data</div>
      <FaultPanel onInject={() => setFaulted(true)} onAttack={(k) => setAttack(k)} />
      <div className="grid gap-6 lg:grid-cols-2">
        <CellHealthMap cells={faulted ? MOCK_CELLS_FAULT : MOCK_CELLS_GREEN} faultCells={faulted ? [2] : []} highlight={faulted ? 2 : null} />
        <div className="flex flex-col gap-3">
          <ConfidenceBadge
            score={0.91}
            tier="one_person"
            breakdown={{ preflight: { passed: true, predicted: 0.97, metric: 'cov' }, federated: 0.94, agree: { k: 3, n: 3 }, lesson: 'L-07' }}
            explain="Preflight on a twin copy recovers cell 2; all three agents propose the same change."
          />
          <ConfidenceBadge score={0.48} tier="two_person" breakdown={{ preflight: { passed: true, predicted: 0.81 }, federated: 0.62, agree: { k: 1, n: 3 }, lesson: null }} explain="Agents disagree on the cascade; two approvers needed." />
          <FederatedForecastLine predicted={0.94} change="cell 2 power +6 dB" />
          <FederatedInset rounds={MOCK_ROUNDS} siloGains={{ A: 0.004, B: -0.006, C: 0.041 }} />
        </div>
      </div>
      <LoopTimeline stages={MOCK_STAGES} />
      <div className="grid gap-6 lg:grid-cols-2">
        <ReliabilityChart bins={MOCK_BINS} brier={0.142} ece={0.061} source="preview bins" />
        <LessonsCard lessons={MOCK_LESSONS} beforeAfter={MOCK_BEFORE_AFTER} synthetic />
      </div>
      {attack && (
        <AttackResult attack={attack} decision="DENY" agent={attack === 'rogue_agent' ? 'ran_monitor' : 'noc_lead'} tool="ran___apply_change" policy="forbid-monitor-writes" awsTime="2026-11-04T06:31:10.400Z" awsTimeSource="cloudwatch" />
      )}
    </div>
  )
}
