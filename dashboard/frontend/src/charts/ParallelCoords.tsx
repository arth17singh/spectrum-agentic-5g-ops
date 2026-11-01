import { scaleLinear } from 'd3-scale'
import { useState } from 'react'
import { AXIS_TEXT, Tooltip, useSize } from './common'

export interface PCAxis {
  id: string
  label: string
  /** lower is better: the axis is flipped so "up" always means better */
  invert?: boolean
  domain: [number, number]
  format: (v: number) => string
}
export interface PCLine { id: string; label: string; color: string; values: Record<string, number>; emphasis?: boolean; muted?: boolean; showLabel?: boolean }

/** Parallel coordinates. Every axis points "better = up"; hovering a line highlights it and dims the rest. */
export function ParallelCoords({ axes, lines, height = 240 }: { axes: PCAxis[]; lines: PCLine[]; height?: number }) {
  const { ref, width } = useSize()
  const [hover, setHover] = useState<{ id: string; x: number; y: number } | null>(null)
  const m = { l: 52, r: 96, t: 34, b: 18 }
  const w = Math.max(10, width - m.l - m.r)
  const h = Math.max(10, height - m.t - m.b)
  const xs = scaleLinear().domain([0, axes.length - 1]).range([0, w])
  const ys = Object.fromEntries(axes.map((a) => [a.id, scaleLinear().domain(a.invert ? [a.domain[1], a.domain[0]] : a.domain).range([h, 0]).clamp(true)]))
  const path = (l: PCLine) => axes.map((a, i) => `${i ? 'L' : 'M'}${xs(i)},${ys[a.id](l.values[a.id])}`).join('')
  const active = hover?.id
  const ordered = [...lines].sort((a, b) => Number(a.id === active || !!a.emphasis) - Number(b.id === active || !!b.emphasis))
  const hl = lines.find((l) => l.id === active)
  return (
    <div ref={ref} className="relative" style={{ height }}>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label="Models compared across six axes">
          <g transform={`translate(${m.l},${m.t})`}>
            {axes.map((a, i) => (
              <g key={a.id} transform={`translate(${xs(i)},0)`}>
                <line y1={0} y2={h} style={{ stroke: 'var(--axis)' }} />
                <text y={-20} textAnchor="middle" style={{ fill: 'var(--ink-2)', fontSize: 11, fontWeight: 500 }}>{a.label}</text>
                {a.invert && <text y={-8} textAnchor="middle" style={AXIS_TEXT}>lower is better</text>}
                {ys[a.id].ticks(4).map((t) => (
                  <g key={t} transform={`translate(0,${ys[a.id](t)})`}>
                    <line x1={-3} x2={3} style={{ stroke: 'var(--axis)' }} />
                    <text x={-6} dy="0.32em" textAnchor="end" style={{ ...AXIS_TEXT, fontSize: 10 }} className="tnum">{a.format(t)}</text>
                  </g>
                ))}
              </g>
            ))}
            {ordered.map((l) => {
              const on = l.id === active
              const dim = active != null && !on
              return (
                <g key={l.id}>
                  <path d={path(l)} fill="none" style={{ stroke: l.color }} strokeWidth={on || l.emphasis ? 2.5 : 1.5}
                    strokeOpacity={dim ? 0.12 : l.muted && !on ? 0.4 : 1} strokeLinejoin="round" />
                  <path d={path(l)} fill="none" stroke="transparent" strokeWidth={12}
                    onPointerMove={(e) => {
                      const r = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
                      setHover({ id: l.id, x: e.clientX - r.left, y: e.clientY - r.top })
                    }}
                    onPointerLeave={() => setHover(null)} />
                  {(l.showLabel || on) && (
                    <text x={w + 8} y={ys[axes[axes.length - 1].id](l.values[axes[axes.length - 1].id])} dy="0.32em"
                      style={{ fill: 'var(--ink-1)', fontSize: 11, fontWeight: 500 }}>{l.label}</text>
                  )}
                </g>
              )
            })}
          </g>
        </svg>
      )}
      {hover && hl && (
        <Tooltip x={hover.x} y={hover.y} containerWidth={width} title={hl.label}
          rows={axes.map((a) => ({ label: a.label, value: a.format(hl.values[a.id]) }))} />
      )}
    </div>
  )
}
