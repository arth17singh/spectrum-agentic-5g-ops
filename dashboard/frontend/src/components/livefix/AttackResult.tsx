import clsx from 'clsx'
import { OctagonX, ShieldCheck } from 'lucide-react'
import { AwsTime } from '@/ui'
import type { AttackKey } from './types'
import { ATTACKS } from './types'

export interface AttackResultProps {
  attack: AttackKey
  /** DENY is the expected outcome; ALLOW is shown as a failure */
  decision: 'DENY' | 'ALLOW' | 'ERROR'
  /** the Cedar policy (or rule) that decided */
  policy?: string | null
  /** which agent made the call and which tool */
  agent?: string
  tool?: string
  awsTime?: string | null
  awsTimeSource?: string | null
  /** a short excerpt of the refused input (redacted) */
  excerpt?: string
  caseId?: string
  className?: string
}

export function AttackResult({ attack, decision, policy, agent, tool, awsTime, awsTimeSource, excerpt, caseId, className }: AttackResultProps) {
  const a = ATTACKS.find((x) => x.key === attack)
  const refused = decision === 'DENY'
  const color = refused ? 'var(--good)' : 'var(--critical)'
  return (
    <div
      className={clsx('flex flex-col gap-1.5 rounded-lg border p-3', className)}
      style={{ borderColor: `color-mix(in oklab, ${color} 50%, var(--hairline))`, background: `color-mix(in oklab, ${color} 7%, transparent)` }}
      role="status"
    >
      <div className="flex flex-wrap items-center gap-2">
        {refused ? <ShieldCheck size={16} className="text-good-ink" aria-hidden /> : <OctagonX size={16} className="text-critical-ink" aria-hidden />}
        <span className="text-title font-medium text-ink-1">{a?.label ?? attack}</span>
        <span className={clsx('text-cap font-semibold', refused ? 'text-good-ink' : 'text-critical-ink')}>
          {refused ? 'Refused by Cedar' : decision === 'ALLOW' ? 'NOT refused' : 'Error'}
        </span>
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-cap text-ink-2">
        {agent && <span><span className="text-ink-3">agent </span><span className="mono">{agent}</span></span>}
        {tool && <span><span className="text-ink-3">tool </span><span className="mono">{tool}</span></span>}
        {policy && <span><span className="text-ink-3">policy </span><span className="mono">{policy}</span></span>}
        {caseId && <span><span className="text-ink-3">case </span><span className="mono">{caseId}</span></span>}
      </div>
      {excerpt && <div className="mono truncate text-micro text-ink-3" title={excerpt}>“{excerpt}”</div>}
      <div className="text-micro"><AwsTime aws={awsTime} source={awsTimeSource ?? 'none'} compact /></div>
    </div>
  )
}
