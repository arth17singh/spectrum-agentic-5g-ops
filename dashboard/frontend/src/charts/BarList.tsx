import type { ReactNode } from 'react'

/** Ranked horizontal bars; value at the tip, single series color unless an entity color is given. */
export function BarList({ items, format = (v) => String(v), max, color = 'var(--s1)', barHeight = 8 }: {
  items: { id: string; label: ReactNode; value: number; color?: string; sub?: ReactNode }[]
  format?: (v: number) => string; max?: number; color?: string; barHeight?: number
}) {
  const m = max ?? Math.max(1, ...items.map((i) => i.value))
  return (
    <ul className="space-y-2">
      {items.map((it) => (
        <li key={it.id} className="grid grid-cols-[minmax(0,140px)_1fr_auto] items-center gap-3 text-cap">
          <span className="truncate text-ink-2" title={typeof it.label === 'string' ? it.label : undefined}>{it.label}</span>
          <span className="relative h-2.5">
            <span className="absolute inset-y-0 left-0 rounded-r-[4px]" style={{ width: `${(it.value / m) * 100}%`, height: barHeight, top: (10 - barHeight) / 2, background: it.color ?? color }} />
          </span>
          <span className="tnum text-right text-ink-1">{format(it.value)}{it.sub && <span className="ml-1 text-ink-3">{it.sub}</span>}</span>
        </li>
      ))}
    </ul>
  )
}
