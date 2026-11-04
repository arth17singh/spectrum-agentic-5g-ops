import clsx from 'clsx'
import type { ReliabilityBin } from './types'

export interface ReliabilityChartProps {
  bins: ReliabilityBin[]
  brier?: number | null
  ece?: number | null
  /** where the outcomes came from, e.g. "18 held-out incidents × 3 steps, graded by the twin" */
  source?: string
  width?: number
  className?: string
}

/** Stated confidence (x) against the observed success rate (y); a calibrated agent sits on the diagonal. */
export function ReliabilityChart({ bins, brier, ece, source, width = 320, className }: ReliabilityChartProps) {
  const n = bins.reduce((a, b) => a + b.count, 0)
  const m = { l: 36, r: 12, t: 12, b: 32 }
  const W = width, H = width * 0.82
  const pw = W - m.l - m.r, ph = H - m.t - m.b
  const x = (v: number) => m.l + v * pw
  const y = (v: number) => m.t + (1 - v) * ph
  const maxC = Math.max(1, ...bins.map((b) => b.count))
  const ticks = [0, 0.25, 0.5, 0.75, 1]
  return (
    <figure className={clsx('flex flex-col gap-1', className)}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        style={{ maxWidth: W }}
        role="img"
        aria-label={`Reliability diagram over ${n} outcomes${typeof ece === 'number' ? `, calibration error ${ece.toFixed(3)}` : ''}`}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(0)} x2={x(1)} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--hairline)' }} strokeWidth={0.6} />
            <text x={m.l - 5} y={y(t) + 3.5} textAnchor="end" className="tnum" style={{ fill: 'var(--ink-3)', fontSize: 10 }}>{t}</text>
            <text x={x(t)} y={H - m.b + 13} textAnchor="middle" className="tnum" style={{ fill: 'var(--ink-3)', fontSize: 10 }}>{t}</text>
          </g>
        ))}
        <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} style={{ stroke: 'var(--ink-3)' }} strokeDasharray="4 3" strokeWidth={1} />
        {bins.map((b, i) => {
          const bw = (b.hi - b.lo) * pw
          return (
            <g key={i}>
              {/* bin bar: observed success; opacity by how many outcomes sit in it */}
              <rect
                x={x(b.lo) + 1.5}
                y={y(b.observed)}
                width={Math.max(0, bw - 3)}
                height={Math.max(0, y(0) - y(b.observed))}
                style={{ fill: 'var(--verify)', opacity: b.count ? 0.25 + 0.55 * (b.count / maxC) : 0.1 }}
              />
              {b.count > 0 && (
                <>
                  <circle cx={x(b.meanConf)} cy={y(b.observed)} r={3.5} style={{ fill: 'var(--ink-1)' }} />
                  <text x={x(b.lo) + bw / 2} y={y(b.observed) - 5} textAnchor="middle" className="tnum" style={{ fill: 'var(--ink-2)', fontSize: 9.5 }}>
                    n={b.count}
                  </text>
                </>
              )}
            </g>
          )
        })}
        <line x1={x(0)} x2={x(1)} y1={y(0)} y2={y(0)} style={{ stroke: 'var(--axis)' }} />
        <line x1={x(0)} x2={x(0)} y1={y(0)} y2={y(1)} style={{ stroke: 'var(--axis)' }} />
        <text x={m.l + pw / 2} y={H - 3} textAnchor="middle" style={{ fill: 'var(--ink-2)', fontSize: 10.5 }}>stated confidence</text>
        <text x={10} y={m.t + ph / 2} textAnchor="middle" transform={`rotate(-90 10 ${m.t + ph / 2})`} style={{ fill: 'var(--ink-2)', fontSize: 10.5 }}>
          actually recovered
        </text>
      </svg>
      <figcaption className="flex flex-wrap gap-x-3 text-micro text-ink-3">
        <span className="tnum">n={n}{n < 50 ? ' (small sample: coarse bins)' : ''}</span>
        {typeof brier === 'number' && <span className="tnum">Brier {brier.toFixed(3)}</span>}
        {typeof ece === 'number' && <span className="tnum">calibration error {ece.toFixed(3)}</span>}
        {source && <span>{source}</span>}
      </figcaption>
    </figure>
  )
}
