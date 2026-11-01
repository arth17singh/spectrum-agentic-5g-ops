import { ArrowDown, ArrowUp } from 'lucide-react'

/** Before/after pairs per metric (pre-flight predictions). Neutral bar = now, blue bar = predicted.
 *  The delta is coloured by whether the change is an improvement for that metric. */
export interface DeltaRow { id: string; label: string; before: number; after: number; max?: number; unit?: string; better: 'lower' | 'higher'; format?: (v: number) => string }

export function DeltaBars({ rows }: { rows: DeltaRow[] }) {
  return (
    <div className="space-y-3">
      {rows.map((r) => {
        const max = r.max ?? Math.max(r.before, r.after, 1e-9) * 1.1
        const f = r.format ?? ((v: number) => v.toFixed(1))
        const d = r.after - r.before
        const good = r.better === 'lower' ? d < 0 : d > 0
        const flat = Math.abs(d) < 1e-6
        return (
          <div key={r.id}>
            <div className="mb-1 flex items-baseline gap-2 text-cap">
              <span className="text-ink-2">{r.label}</span>
              <span className="ml-auto tnum text-ink-3">{f(r.before)}{r.unit}</span>
              <span className="text-ink-3">→</span>
              <span className="tnum font-medium text-ink-1">{f(r.after)}{r.unit}</span>
              {!flat && (
                <span className="tnum flex w-16 items-center justify-end gap-0.5 font-medium" style={{ color: good ? 'var(--good-ink)' : 'var(--critical-ink)' }}>
                  {d < 0 ? <ArrowDown size={12} /> : <ArrowUp size={12} />}{f(Math.abs(d))}
                </span>
              )}
              {flat && <span className="w-16 text-right text-ink-3">no change</span>}
            </div>
            <div className="space-y-[2px]">
              <div className="h-[7px] rounded-r-[4px]" style={{ width: `${Math.min(100, (r.before / max) * 100)}%`, background: 'var(--neutral-series)' }} title={`now ${f(r.before)}${r.unit ?? ''}`} />
              <div className="h-[7px] rounded-r-[4px] transition-[width] duration-500" style={{ width: `${Math.min(100, (r.after / max) * 100)}%`, background: 'var(--s1)' }} title={`predicted ${f(r.after)}${r.unit ?? ''}`} />
            </div>
          </div>
        )
      })}
      <div className="flex gap-4 text-micro text-ink-3">
        <span className="flex items-center gap-1.5"><span className="h-2 w-3 rounded-sm" style={{ background: 'var(--neutral-series)' }} />now</span>
        <span className="flex items-center gap-1.5"><span className="h-2 w-3 rounded-sm" style={{ background: 'var(--s1)' }} />predicted by the twin</span>
      </div>
    </div>
  )
}
