import clsx from 'clsx'
import { Network } from 'lucide-react'
import { pct } from './hex'

export interface FederatedForecastLineProps {
  /** predicted recovery 0..1 for the proposed change */
  predicted: number
  /** the change being forecast, e.g. "cell 2 power +6 dB" */
  change?: string
  /** who asked, e.g. "optimizer" */
  actor?: string
  operators?: number
  className?: string
}

/** One activity-stream line: the swarm asked the federated model for a forecast. */
export function FederatedForecastLine({ predicted, change, actor = 'optimizer', operators = 3, className }: FederatedForecastLineProps) {
  return (
    <div className={clsx('flex items-baseline gap-2 text-cap', className)}>
      <Network size={13} className="shrink-0 translate-y-0.5 text-learn" aria-hidden />
      <span className="text-ink-2">
        <span className="font-medium text-ink-1">{actor}</span> → forecast with <span className="font-medium text-ink-1">federated model</span>{' '}
        <span className="text-ink-3">(trained across {operators} operators, no data shared)</span>
        {change && <> for <span className="mono">{change}</span></>} → predicted recovery{' '}
        <span className="tnum font-semibold text-ink-1">{pct(predicted)}</span>
      </span>
    </div>
  )
}

export interface FederatedInsetProps {
  /** per-round metric of the federated session (e.g. shared-test loss or score), round 1..N */
  rounds: number[]
  /** label for the metric */
  metric?: string
  /** lower is better? (loss) */
  lowerIsBetter?: boolean
  /** per-silo gain over local-only training, CI low; e.g. {A: 0.01, B: -0.004, C: 0.041} */
  siloGains?: Record<string, number>
  width?: number
  className?: string
}

/** A tiny inset: the federated session's round curve and each silo's gain over training alone. */
export function FederatedInset({ rounds, metric = 'shared-test score', lowerIsBetter = false, siloGains, width = 220, className }: FederatedInsetProps) {
  const W = width, H = 64, p = 4
  const lo = Math.min(...rounds), hi = Math.max(...rounds)
  const span = hi - lo || 1
  const pts = rounds.map((v, i) => {
    const x = p + (i / Math.max(1, rounds.length - 1)) * (W - 2 * p)
    const y = p + (1 - (v - lo) / span) * (H - 2 * p)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
  return (
    <div className={clsx('flex flex-col gap-1 rounded-lg border border-line bg-s2 p-2', className)}>
      <div className="flex items-baseline gap-2 text-micro">
        <span className="font-medium text-ink-1">Federated session</span>
        <span className="text-ink-3">{rounds.length} rounds · {metric}{lowerIsBetter ? ' (lower is better)' : ''}</span>
      </div>
      {rounds.length > 1 && (
        <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" style={{ maxWidth: W }} role="img" aria-label={`${metric} over ${rounds.length} federated rounds, from ${rounds[0].toFixed(3)} to ${rounds[rounds.length - 1].toFixed(3)}`}>
          <polyline points={pts} fill="none" style={{ stroke: 'var(--learn)' }} strokeWidth={1.6} strokeLinejoin="round" />
        </svg>
      )}
      {siloGains && (
        <div className="flex flex-wrap gap-x-3 text-micro">
          {Object.entries(siloGains).map(([s, g]) => (
            <span key={s} className="tnum">
              <span className="text-ink-3">silo {s} gain CI low </span>
              <span className={clsx('font-medium', g > 0 ? 'text-good-ink' : 'text-ink-2')}>{g >= 0 ? '+' : ''}{g.toFixed(3)}</span>
            </span>
          ))}
        </div>
      )}
      <div className="text-micro text-ink-3">Trained before the demo on the FL service; replayed here, not trained live.</div>
    </div>
  )
}
