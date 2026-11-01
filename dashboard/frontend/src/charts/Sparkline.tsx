import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import { useSize } from './common'

export function Sparkline({ values, color = 'var(--ink-3)', height = 24, className, domain }: {
  values: number[]; color?: string; height?: number; className?: string; domain?: [number, number]
}) {
  const { ref, width } = useSize()
  const vals = values.filter((v) => Number.isFinite(v))
  const lo = domain?.[0] ?? Math.min(...vals)
  const hi = domain?.[1] ?? Math.max(...vals)
  const x = scaleLinear().domain([0, Math.max(1, vals.length - 1)]).range([1, Math.max(2, width - 4)])
  const y = scaleLinear().domain(lo === hi ? [lo - 1, hi + 1] : [lo, hi]).range([height - 3, 3])
  const d = line<number>().x((_, i) => x(i)).y((v) => y(v))(vals) ?? ''
  const last = vals[vals.length - 1]
  return (
    <div ref={ref} className={className} style={{ height }}>
      {width > 0 && vals.length > 1 && (
        <svg width={width} height={height} aria-hidden>
          <path d={d} fill="none" style={{ stroke: color }} strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" />
          <circle cx={x(vals.length - 1)} cy={y(last)} r={2.5} style={{ fill: color }} />
        </svg>
      )}
    </div>
  )
}
