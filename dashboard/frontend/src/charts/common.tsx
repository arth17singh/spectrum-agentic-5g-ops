import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { useUi } from '@/core/ui-store'

// Shared chart plumbing: sizing, tooltip, legend, axis ticks. Rules (dataviz method): hairline solid
// grid, 2px lines, >= 8px markers with a 2px surface ring, text in ink tokens, legend for >= 2 series.

export function useSize<T extends HTMLElement = HTMLDivElement>() {
  const ref = useRef<T>(null)
  const [size, setSize] = useState({ width: 0, height: 0 })
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => {
      const { width, height } = e.contentRect
      setSize((s) => (Math.abs(s.width - width) > 0.5 || Math.abs(s.height - height) > 0.5 ? { width, height } : s))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return { ref, ...size }
}

/** Re-render canvas charts when the theme changes (canvas needs concrete colors). */
export function useThemeKey() {
  return useUi((s) => s.theme)
}

export interface TipRow { color?: string; label: ReactNode; value: ReactNode; dashed?: boolean }

export function Tooltip({ x, y, title, rows, containerWidth }: {
  x: number; y: number; title?: ReactNode; rows: TipRow[]; containerWidth: number
}) {
  const flip = x > containerWidth - 200
  return (
    <div
      role="tooltip"
      className="pointer-events-none absolute z-20 min-w-[140px] rounded-md border border-line bg-s2 px-2.5 py-2 text-cap shadow-[0_8px_24px_rgba(0,0,0,0.25)]"
      style={{ left: flip ? undefined : x + 14, right: flip ? containerWidth - x + 14 : undefined, top: Math.max(0, y - 10) }}
    >
      {title && <div className="mb-1 text-micro text-ink-3">{title}</div>}
      <div className="space-y-0.5">
        {rows.map((r, i) => (
          <div key={i} className="flex items-center gap-2">
            {r.color && (
              <svg width="12" height="8" aria-hidden className="shrink-0">
                <line x1="0" x2="12" y1="4" y2="4" style={{ stroke: r.color }} strokeWidth="2" strokeDasharray={r.dashed ? '3 2' : undefined} />
              </svg>
            )}
            <span className="tnum font-medium text-ink-1">{r.value}</span>
            <span className="truncate text-ink-3">{r.label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export function Legend({ items, className }: { items: { label: ReactNode; color: string; kind?: 'line' | 'rect' | 'dot'; dashed?: boolean }[]; className?: string }) {
  return (
    <div className={`flex flex-wrap items-center gap-x-3 gap-y-1 text-cap text-ink-2 ${className ?? ''}`}>
      {items.map((it, i) => (
        <span key={i} className="inline-flex items-center gap-1.5">
          {it.kind === 'rect' ? (
            <span className="size-2.5 rounded-[2px]" style={{ background: it.color }} />
          ) : it.kind === 'dot' ? (
            <span className="size-2 rounded-full" style={{ background: it.color }} />
          ) : (
            <svg width="14" height="8" aria-hidden>
              <line x1="0" x2="14" y1="4" y2="4" style={{ stroke: it.color }} strokeWidth="2" strokeLinecap="round" strokeDasharray={it.dashed ? '3 2' : undefined} />
            </svg>
          )}
          {it.label}
        </span>
      ))}
    </div>
  )
}

export const AXIS_TEXT = { fill: 'var(--ink-3)', fontSize: 11 } as const

/** A minimal accessible table twin for any chart. */
export function DataTable({ columns, rows, maxHeight = 260 }: { columns: string[]; rows: (string | number | ReactNode)[][]; maxHeight?: number }) {
  return (
    <div className="overflow-auto" style={{ maxHeight }}>
      <table className="w-full text-cap">
        <thead className="sticky top-0 bg-s1">
          <tr>{columns.map((c) => <th key={c} className="border-b border-line py-1.5 pr-3 text-left font-medium text-ink-3">{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-line/60">
              {r.map((v, j) => <td key={j} className="tnum py-1 pr-3 text-ink-1">{v}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function useAnimationFrame(cb: (t: number) => void, active = true) {
  const cbRef = useRef(cb)
  cbRef.current = cb
  useEffect(() => {
    if (!active) return
    let id = 0
    const loop = (t: number) => {
      cbRef.current(t)
      id = requestAnimationFrame(loop)
    }
    id = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(id)
  }, [active])
}
