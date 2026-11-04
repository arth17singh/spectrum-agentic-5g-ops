import clsx from 'clsx'
import { cellCentre, hexPath, pct, sign } from './hex'
import type { CellId, CellState, Health } from './types'

export interface CellHealthMapProps {
  cells: CellState[]
  /** cells the fault was injected on: outlined and pulsing */
  faultCells?: CellId[]
  /** a cell to emphasise (e.g. the one the agents are changing) */
  highlight?: CellId | null
  size?: number
  className?: string
}

const TONE: Record<Health, { fill: string; stroke: string; ink: string; word: string }> = {
  good: { fill: 'color-mix(in oklab, var(--good) 26%, var(--surface-2))', stroke: 'var(--good)', ink: 'var(--good-ink)', word: 'healthy' },
  degraded: { fill: 'color-mix(in oklab, var(--warn) 28%, var(--surface-2))', stroke: 'var(--warn)', ink: 'var(--warn-ink)', word: 'degraded' },
  critical: { fill: 'color-mix(in oklab, var(--critical) 32%, var(--surface-2))', stroke: 'var(--critical)', ink: 'var(--critical-ink)', word: 'down' },
  unknown: { fill: 'var(--surface-2)', stroke: 'var(--hairline)', ink: 'var(--ink-3)', word: 'no data' },
}

function delta(now: number, was: number | undefined, unit: string, d = 1) {
  if (was === undefined || Math.abs(now - was) < 1e-9) return null
  return `${sign(was, d)}→${sign(now, d)}${unit}`
}

export function CellHealthMap({ cells, faultCells = [], highlight, size = 360, className }: CellHealthMapProps) {
  const R = size / 6.6
  const ox = R * 1.25, oy = R * 1.95
  const byId = new Map(cells.map((c) => [c.id, c]))
  return (
    <svg
      viewBox={`0 0 ${size} ${size * 0.98}`}
      className={clsx('h-auto w-full max-w-[420px]', className)}
      role="img"
      aria-label={`The 4 Maveric cells: ${cells.map((c) => `cell ${c.id} ${TONE[c.health].word}`).join(', ')}`}
    >
      {[0, 1, 2, 3].map((id) => {
        const c = byId.get(id as CellId)
        const [cx, cy] = cellCentre(id, R, ox, oy)
        const t = TONE[c?.health ?? 'unknown']
        const fault = faultCells.includes(id as CellId)
        const hi = highlight === id
        const cov = c?.kpis?.coverage
        const covWas = c?.before?.kpis?.coverage
        const dp = c ? delta(c.power, c.before?.power, ' dB') : null
        const dt = c ? delta(c.tilt, c.before?.tilt, '°') : null
        return (
          <g key={id}>
            {fault && (
              <path d={hexPath(cx, cy, R * 1.06)} fill="none" style={{ stroke: 'var(--critical)' }} strokeWidth={2}>
                <animate attributeName="opacity" values="1;0.15;1" dur="1.2s" repeatCount="indefinite" />
              </path>
            )}
            <path
              d={hexPath(cx, cy, R)}
              style={{ fill: t.fill, stroke: hi ? 'var(--focus)' : t.stroke, transition: 'fill 600ms ease, stroke 600ms ease' }}
              strokeWidth={hi ? 3 : fault ? 2.4 : 1.4}
            />
            <text x={cx} y={cy - R * 0.5} textAnchor="middle" style={{ fill: 'var(--ink-1)', fontSize: 12.5, fontWeight: 600 }}>cell {id}</text>
            <text x={cx} y={cy - R * 0.5 + 14} textAnchor="middle" style={{ fill: t.ink, fontSize: 10.5, fontWeight: 600 }}>{t.word}</text>
            {c && (
              <>
                <text x={cx} y={cy + 6} textAnchor="middle" className="tnum" style={{ fill: 'var(--ink-1)', fontSize: 11 }}>
                  {dp ?? `power ${sign(c.power)} dB`}
                </text>
                <text x={cx} y={cy + 20} textAnchor="middle" className="tnum" style={{ fill: 'var(--ink-1)', fontSize: 11 }}>
                  {dt ?? `tilt ${sign(c.tilt)}°`}
                </text>
                {typeof cov === 'number' && (
                  <text x={cx} y={cy + 35} textAnchor="middle" className="tnum" style={{ fill: 'var(--ink-2)', fontSize: 10.5 }}>
                    {typeof covWas === 'number' && Math.abs(covWas - cov) > 1e-6 ? `cov ${pct(covWas)}→${pct(cov)}` : `cov ${pct(cov)}`}
                  </text>
                )}
              </>
            )}
          </g>
        )
      })}
    </svg>
  )
}
