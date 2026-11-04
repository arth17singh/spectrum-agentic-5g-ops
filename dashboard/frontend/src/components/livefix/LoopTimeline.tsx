import clsx from 'clsx'
import { Check, Loader2, Minus, X } from 'lucide-react'
import { AwsTime } from '@/ui'
import { STAGE_LABEL, STAGE_ORDER, type Stage } from './types'

export interface LoopTimelineProps {
  stages: Stage[]
  /** horizontal on wide screens (default) or always vertical */
  orientation?: 'auto' | 'vertical'
  className?: string
}

const DOT = {
  done: { bg: 'var(--good)', Icon: Check },
  active: { bg: 'var(--release)', Icon: Loader2 },
  pending: { bg: 'var(--surface-3)', Icon: null },
  failed: { bg: 'var(--critical)', Icon: X },
  skipped: { bg: 'var(--surface-3)', Icon: Minus },
} as const

export function LoopTimeline({ stages, orientation = 'auto', className }: LoopTimelineProps) {
  const byKey = new Map(stages.map((s) => [s.key, s]))
  const list = STAGE_ORDER.map((k) => byKey.get(k) ?? { key: k, state: 'pending' as const })
  return (
    <ol
      className={clsx('grid gap-2', orientation === 'auto' ? 'grid-cols-1 md:grid-cols-7' : 'grid-cols-1', className)}
      aria-label="Fault-to-fix loop"
    >
      {list.map((s, i) => {
        const d = DOT[s.state]
        return (
          <li key={s.key} className={clsx('flex min-w-0 gap-2', orientation === 'auto' && 'md:flex-col')}>
            <div className={clsx('flex items-center', orientation === 'auto' && 'md:w-full')}>
              <span
                className="grid size-6 shrink-0 place-items-center rounded-full text-page"
                style={{ background: d.bg }}
                aria-label={s.state}
              >
                {d.Icon && <d.Icon size={13} className={s.state === 'active' ? 'animate-spin' : undefined} aria-hidden />}
              </span>
              {i < list.length - 1 && orientation === 'auto' && (
                <span className="ml-1 hidden h-px flex-1 md:block" style={{ background: s.state === 'done' ? 'var(--good)' : 'var(--hairline)' }} />
              )}
            </div>
            <div className="min-w-0">
              <div className={clsx('text-cap font-medium', s.state === 'pending' || s.state === 'skipped' ? 'text-ink-3' : 'text-ink-1')}>
                {STAGE_LABEL[s.key]}
              </div>
              {typeof s.elapsedS === 'number' && (
                <div className="tnum mono text-micro text-ink-2">+{s.elapsedS.toFixed(1)} s</div>
              )}
              {s.awsTime !== undefined && (s.state === 'done' || s.state === 'failed') && (
                <div className="text-micro"><AwsTime aws={s.awsTime} source={s.awsTimeSource ?? 'none'} compact /></div>
              )}
              {s.detail && <div className="truncate text-micro text-ink-3" title={s.detail}>{s.detail}</div>}
            </div>
          </li>
        )
      })}
    </ol>
  )
}
