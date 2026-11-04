import clsx from 'clsx'
import { Bug, Loader2, ShieldAlert, Zap } from 'lucide-react'
import { useState } from 'react'
import { cellCentre, hexPath } from './hex'
import { ATTACKS, CELL_IDS, FAULTS, type AttackKey, type CellId, type FaultKey } from './types'

export interface FaultPanelProps {
  onInject: (fault: FaultKey, cell: CellId) => void
  onAttack: (attack: AttackKey) => void
  disabled?: boolean
  /** the action in flight, if any: shows a spinner on that button */
  busy?: FaultKey | AttackKey | boolean | null
  /** controlled cell selection (optional) */
  cell?: CellId
  onCellChange?: (cell: CellId) => void
  className?: string
}

/** A small hex picker: click a hexagon to choose where the fault lands. */
function CellPicker({ value, onChange, disabled }: { value: CellId; onChange: (c: CellId) => void; disabled?: boolean }) {
  const R = 26, ox = R * 1.3, oy = R * 2.0
  return (
    <svg viewBox="0 0 132 126" className="h-auto w-[132px] shrink-0" role="radiogroup" aria-label="Choose the cell to break">
      {CELL_IDS.map((c) => {
        const [cx, cy] = cellCentre(c, R, ox, oy)
        const on = c === value
        return (
          <g
            key={c}
            role="radio"
            aria-checked={on}
            aria-label={`cell ${c}`}
            tabIndex={disabled ? -1 : 0}
            onClick={() => !disabled && onChange(c)}
            onKeyDown={(e) => { if (!disabled && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); onChange(c) } }}
            className={clsx(!disabled && 'cursor-pointer', 'outline-none')}
          >
            <path
              d={hexPath(cx, cy, R)}
              style={{
                fill: on ? 'color-mix(in oklab, var(--critical) 22%, var(--surface-2))' : 'var(--surface-2)',
                stroke: on ? 'var(--critical)' : 'var(--hairline)',
              }}
              strokeWidth={on ? 2.2 : 1.2}
            />
            <text x={cx} y={cy + 4} textAnchor="middle" style={{ fill: on ? 'var(--critical-ink)' : 'var(--ink-2)', fontSize: 11, fontWeight: 600 }}>
              cell {c}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

export function FaultPanel({ onInject, onAttack, disabled, busy, cell, onCellChange, className }: FaultPanelProps) {
  const [own, setOwn] = useState<CellId>(0)
  const picked = cell ?? own
  const setPicked = (c: CellId) => { setOwn(c); onCellChange?.(c) }
  const anyBusy = !!busy
  const spin = (k: string) => busy === k
  return (
    <div className={clsx('flex flex-col gap-4', className)}>
      <section>
        <div className="mb-2 flex items-baseline gap-2">
          <Zap size={14} className="text-critical-ink" aria-hidden />
          <h4 className="text-title font-medium text-ink-1">Break the network</h4>
          <span className="text-cap text-ink-3">on our in-process Maveric copy, cell {picked}</span>
        </div>
        <div className="flex flex-wrap items-start gap-4">
          <CellPicker value={picked} onChange={setPicked} disabled={disabled || anyBusy} />
          <div className="grid min-w-0 flex-1 grid-cols-1 gap-2 sm:grid-cols-2">
            {FAULTS.map((f) => (
              <button
                key={f.key}
                type="button"
                disabled={disabled || anyBusy}
                onClick={() => onInject(f.key, picked)}
                className={clsx(
                  'flex min-h-14 flex-col items-start justify-center rounded-lg border border-critical/40 bg-s2 px-3 py-2 text-left transition-colors',
                  'hover:border-critical hover:bg-critical/10 disabled:cursor-not-allowed disabled:opacity-40',
                  f.cells === 2 && 'sm:col-span-2',
                )}
                title={`suite template ${f.template}`}
              >
                <span className="flex items-center gap-1.5 text-body font-medium text-ink-1">
                  {spin(f.key) && <Loader2 size={13} className="animate-spin" aria-hidden />}
                  {f.label}
                </span>
                <span className="mono text-micro text-ink-3">
                  {f.cells === 2 ? `cell ${picked} + neighbour: ${f.twin}` : f.twin}
                </span>
              </button>
            ))}
          </div>
        </div>
      </section>
      <section>
        <div className="mb-2 flex items-baseline gap-2">
          <ShieldAlert size={14} className="text-warn-ink" aria-hidden />
          <h4 className="text-title font-medium text-ink-1">Attack the agents</h4>
          <span className="text-cap text-ink-3">Cedar should refuse both</span>
        </div>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {ATTACKS.map((a) => (
            <button
              key={a.key}
              type="button"
              disabled={disabled || anyBusy}
              onClick={() => onAttack(a.key)}
              className="flex min-h-12 flex-col items-start justify-center rounded-lg border border-warn/40 bg-s2 px-3 py-2 text-left transition-colors hover:border-warn hover:bg-warn/10 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <span className="flex items-center gap-1.5 text-body font-medium text-ink-1">
                {spin(a.key) ? <Loader2 size={13} className="animate-spin" aria-hidden /> : <Bug size={13} aria-hidden />}
                {a.label}
              </span>
              <span className="text-micro text-ink-3">{a.detail}</span>
            </button>
          ))}
        </div>
      </section>
    </div>
  )
}
