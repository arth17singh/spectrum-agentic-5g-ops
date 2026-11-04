import clsx from 'clsx'
import { CircleCheck, CircleSlash } from 'lucide-react'
import { pct } from './hex'
import type { BeforeAfter, Lesson } from './types'

export interface LessonsCardProps {
  lessons: Lesson[]
  beforeAfter?: BeforeAfter | null
  /** true while the numbers are not from a recorded AWS run */
  synthetic?: boolean
  className?: string
}

function Row({ label, before, after, better }: { label: string; before: string; after: string; better: boolean | null }) {
  return (
    <div className="flex items-baseline justify-between gap-3 text-cap">
      <span className="text-ink-3">{label}</span>
      <span className="tnum">
        <span className="text-ink-2">{before}</span>
        <span className="mx-1 text-ink-3">→</span>
        <span className={clsx('font-medium', better === null ? 'text-ink-1' : better ? 'text-good-ink' : 'text-warn-ink')}>{after}</span>
      </span>
    </div>
  )
}

export function LessonsCard({ lessons, beforeAfter, synthetic = false, className }: LessonsCardProps) {
  const ba = beforeAfter
  const passed = ba && typeof ba.ciLow === 'number' ? ba.ciLow > (ba.target ?? 0) : null
  const verified = lessons.filter((l) => l.verified).length
  return (
    <div className={clsx('flex flex-col gap-3', className)}>
      {synthetic && (
        <div className="self-start rounded-[4px] bg-s3 px-1.5 py-0.5 text-micro font-medium text-ink-2">
          SYNTHETIC / OFFLINE: not from a recorded AWS run
        </div>
      )}
      {ba && (
        <div
          className="rounded-lg border p-3"
          style={{
            borderColor: passed === false ? 'var(--warn)' : passed ? 'color-mix(in oklab, var(--good) 50%, var(--hairline))' : 'var(--hairline)',
            background: passed === false ? 'color-mix(in oklab, var(--warn) 8%, transparent)' : 'transparent',
          }}
        >
          <div className="mb-2 flex items-baseline gap-2">
            <span className="text-title font-medium text-ink-1">Held-out faults, before → after learning</span>
            {passed !== null && (
              <span className={clsx('ml-auto text-cap font-semibold', passed ? 'text-good-ink' : 'text-warn-ink')}>{passed ? 'PASS' : 'FAIL'}</span>
            )}
          </div>
          <Row label={`recovered (of ${ba.n})`} before={String(ba.recoveredBefore)} after={String(ba.recoveredAfter)} better={ba.recoveredAfter > ba.recoveredBefore ? true : ba.recoveredAfter < ba.recoveredBefore ? false : null} />
          {typeof ba.stepsBefore === 'number' && typeof ba.stepsAfter === 'number' && (
            <Row label="mean steps to fix" before={ba.stepsBefore.toFixed(2)} after={ba.stepsAfter.toFixed(2)} better={ba.stepsAfter < ba.stepsBefore ? true : ba.stepsAfter > ba.stepsBefore ? false : null} />
          )}
          {typeof ba.harmfulBefore === 'number' && typeof ba.harmfulAfter === 'number' && (
            <Row label="harmful proposals" before={String(ba.harmfulBefore)} after={String(ba.harmfulAfter)} better={ba.harmfulAfter < ba.harmfulBefore ? true : ba.harmfulAfter > ba.harmfulBefore ? false : null} />
          )}
          {typeof ba.ciLow === 'number' && (
            <div className="mt-1 text-micro text-ink-3">
              paired gain CI low <span className="tnum font-medium text-ink-1">{ba.ciLow >= 0 ? '+' : ''}{ba.ciLow.toFixed(3)}</span>, target &gt; {(ba.target ?? 0).toFixed(2)}
              {ba.runId && <span className="mono"> · {ba.runId}</span>}
            </div>
          )}
        </div>
      )}
      <div>
        <div className="mb-1 flex items-baseline gap-2">
          <span className="text-title font-medium text-ink-1">Lessons</span>
          <span className="text-cap text-ink-3">{verified} verified by the twin, {lessons.length - verified} not learned</span>
        </div>
        {lessons.length === 0 ? (
          <p className="text-cap text-ink-3">No lessons yet. A fix becomes a lesson only if the twin confirms the cell recovered.</p>
        ) : (
          <ul className="flex flex-col divide-y divide-line">
            {lessons.map((l) => (
              <li key={l.id} className={clsx('flex gap-2 py-1.5', !l.verified && 'opacity-55')}>
                {l.verified
                  ? <CircleCheck size={14} className="mt-0.5 shrink-0 text-good-ink" aria-label="verified" />
                  : <CircleSlash size={14} className="mt-0.5 shrink-0 text-ink-3" aria-label="not learned" />}
                <div className="min-w-0 text-cap">
                  <div className="text-ink-1">
                    <span className="mono mr-1.5 text-ink-3">{l.id}</span>
                    {l.symptoms} <span className="text-ink-3">→</span> <span className="mono">{l.change}</span>
                  </div>
                  <div className="text-micro text-ink-3">
                    {l.verified ? `from ${l.source}` : `not learned: ${l.reason ?? 'the cell did not recover'} (${l.source})`}
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
      {ba && (
        <div className="text-micro text-ink-3">
          Recovery rate {pct(ba.recoveredBefore / ba.n)} → {pct(ba.recoveredAfter / ba.n)}; lessons come only from dev incidents, graded on held-out ones.
        </div>
      )}
    </div>
  )
}
