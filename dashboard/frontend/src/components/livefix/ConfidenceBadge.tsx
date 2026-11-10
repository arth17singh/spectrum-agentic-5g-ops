import clsx from 'clsx'
import { pct } from './hex'
import type { ConfidenceBreakdown, ConfidenceTier } from './types'

export interface ConfidenceBadgeProps {
  /** 0..1, computed from evidence (not the model's own opinion) */
  score: number
  tier: ConfidenceTier
  breakdown?: ConfidenceBreakdown
  /** one line on why */
  explain?: string
  /** is this score's calibration measured yet? shows an honest tag when false */
  calibrated?: boolean
  /** the measured calibration result, shown in place of the default tag (e.g. a missed ECE target) */
  calibrationNote?: string
  className?: string
}

const TIER: Record<ConfidenceTier, { word: string; color: string; ink: string }> = {
  one_person: { word: 'one-person approval', color: 'var(--good)', ink: 'var(--good-ink)' },
  two_person: { word: 'two-person approval', color: 'var(--warn)', ink: 'var(--warn-ink)' },
  hold: { word: 'held: no change proposed', color: 'var(--critical)', ink: 'var(--critical-ink)' },
}

function Chip({ label, value, tone }: { label: string; value: string; tone: 'good' | 'warn' | 'bad' | 'neutral' }) {
  const c = tone === 'good' ? 'var(--good)' : tone === 'warn' ? 'var(--warn)' : tone === 'bad' ? 'var(--critical)' : 'var(--ink-3)'
  return (
    <span
      className="inline-flex items-baseline gap-1 rounded-[4px] px-1.5 py-0.5 text-micro"
      style={{ background: `color-mix(in oklab, ${c} 13%, transparent)` }}
    >
      <span className="text-ink-3">{label}</span>
      <span className="tnum font-medium text-ink-1">{value}</span>
    </span>
  )
}

export function ConfidenceBadge({ score, tier, breakdown, explain, calibrated = false, calibrationNote, className }: ConfidenceBadgeProps) {
  const t = TIER[tier]
  const b = breakdown ?? {}
  const s = Math.max(0, Math.min(1, score))
  return (
    <div className={clsx('flex flex-col gap-2 rounded-lg border border-line bg-s2 p-3', className)}>
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-baseline gap-1.5">
          <span className="text-cap text-ink-3">Confidence</span>
          <span className="tnum text-h1 font-semibold" style={{ color: t.ink }}>{s.toFixed(2)}</span>
        </div>
        <div className="h-1.5 w-28 overflow-hidden rounded-full bg-s3" aria-hidden>
          <div className="h-full rounded-full" style={{ width: `${s * 100}%`, background: t.color }} />
        </div>
        <span
          className="rounded-[4px] px-1.5 py-0.5 text-cap font-medium"
          style={{ color: t.ink, background: `color-mix(in oklab, ${t.color} 14%, transparent)` }}
        >
          {t.word}
        </span>
        {!calibrated && (calibrationNote
          ? <span className="text-micro font-medium" style={{ color: 'var(--warn-ink)' }} data-testid="calibration-note">{calibrationNote}</span>
          : <span className="text-micro text-ink-3" title="The reliability of this score has not been measured on held-out incidents yet">calibration not yet measured</span>)}
      </div>
      <div className="flex flex-wrap gap-1.5">
        {b.preflight && (
          <Chip
            label="preflight"
            value={`${b.preflight.passed ? 'pass' : 'fail'}${typeof b.preflight.predicted === 'number' ? ` · ${b.preflight.metric ?? 'cov'} ${pct(b.preflight.predicted)}` : ''}`}
            tone={b.preflight.passed ? 'good' : 'bad'}
          />
        )}
        {typeof b.federated === 'number' && (
          <Chip label="federated forecast" value={pct(b.federated)} tone={b.federated >= 0.8 ? 'good' : b.federated >= 0.5 ? 'warn' : 'bad'} />
        )}
        {b.agree && (
          <Chip label="agents agree" value={`${b.agree.k} of ${b.agree.n}`} tone={b.agree.k === b.agree.n ? 'good' : b.agree.k * 2 > b.agree.n ? 'warn' : 'bad'} />
        )}
        <Chip label="lesson" value={b.lesson ?? 'none'} tone={b.lesson ? 'good' : 'neutral'} />
      </div>
      {explain && <p className="text-cap text-ink-2">{explain}</p>}
    </div>
  )
}
