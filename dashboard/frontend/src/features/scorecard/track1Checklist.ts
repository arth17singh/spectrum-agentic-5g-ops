// The participant guide's Track 1 RL checklist, with the evidence we hold for each item.
// Static on purpose: smoke, train-local and check-aws leave no run record, so nothing can be derived for them.
// An item is "recorded" only when a run record backs it ("recorded, differs" when the record is not literally what the guide names); a log line is "log only"; nothing else is ever marked done.

export type CheckState = 'recorded' | 'recorded, differs' | 'log only' | 'not recorded'

export interface CheckItem {
  id: string
  label: string
  state: CheckState
  evidence: string
  run?: string
  ref?: string
}

export const T1_CHECKLIST: CheckItem[] = [
  { id: 'smoke', label: 'make smoke passes', state: 'not recorded',
    evidence: 'No record, output or log of a make smoke run in the evidence mirror, the repo or GAPS.md.' },
  { id: 'train-local', label: 'make train-local passes', state: 'not recorded',
    evidence: 'No record, output or log of a make train-local run in the evidence mirror, the repo or GAPS.md.' },
  { id: 'check-aws', label: 'make check-aws ends "all checks passed"', state: 'log only',
    evidence: 'A log line only: 2 Nov ~15:45 PST, make check-aws printed "all checks passed". No run record or output file.',
    ref: 'GAPS.md#g-001' },
  { id: 'sagemaker', label: 'A SageMaker job reached Completed against the hosted twin', state: 'recorded',
    evidence: 'CPU evaluation job, Completed, against the team-hosted Maveric twin over http. The 100k-step train job (t1.rl.train-20261103T023805Z-a09e) ended Failed; this evaluation used its 50k-step checkpoint.',
    run: 't1.rl.eval-20261103T043115Z-bf52' },
  { id: 'evaluate', label: 'make evaluate wrote reports/evaluation.md', state: 'recorded, differs',
    evidence: 'evaluation.md was written by the SDK evaluate (ran_rl.evaluation.evaluate --with-baseline) inside that job, at eval/evaluation.md, not reports/. Gates: 5 of 6 pass; UPLIFT FAILS (objective 3.20 against the 31.24 needed).',
    run: 't1.rl.eval-20261103T043115Z-bf52' },
]
