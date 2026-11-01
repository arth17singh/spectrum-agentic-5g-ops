import { sankey, sankeyLinkHorizontal, sankeyLeft } from 'd3-sankey'
import { useMemo, useState } from 'react'
import { Tooltip, useSize } from './common'

export interface SNode { id: string; label: string; color: string; sub?: string }
export interface SLink { source: string; target: string; value: number; label: string; color: string; zero?: boolean }

/** Flows between boundaries. Zero-value links are drawn as a thin outlined lane so "0 B" is visible. */
export function Sankey({ nodes, links, height = 220, format }: { nodes: SNode[]; links: SLink[]; height?: number; format: (v: number) => string }) {
  const { ref, width } = useSize()
  const [hover, setHover] = useState<{ l: SLink; x: number; y: number } | null>(null)
  const graph = useMemo(() => {
    if (!width) return null
    const gen = sankey<any, any>().nodeId((d: any) => d.id).nodeAlign(sankeyLeft).nodeWidth(10).nodePadding(10)
      .extent([[0, 4], [width - 150, height - 4]])
    const live = links.filter((l) => !l.zero && l.value > 0)
    return gen({ nodes: nodes.map((n) => ({ ...n })), links: live.map((l) => ({ ...l })) })
  }, [nodes, links, width, height])
  const zero = links.filter((l) => l.zero)
  return (
    <div ref={ref} className="relative" style={{ height }}>
      {graph && (
        <svg width={width} height={height} role="img" aria-label="Bytes moved across organisational boundaries">
          {graph.links.map((l: any, i: number) => (
            <path key={i} d={sankeyLinkHorizontal()(l) ?? ''} fill="none" style={{ stroke: l.color }} strokeOpacity={0.35}
              strokeWidth={Math.max(1.5, l.width)}
              onPointerMove={(e) => {
                const r = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
                setHover({ l, x: e.clientX - r.left, y: e.clientY - r.top })
              }}
              onPointerLeave={() => setHover(null)} />
          ))}
          {graph.nodes.map((n: any) => (
            <g key={n.id}>
              <rect x={n.x0} y={n.y0} width={n.x1 - n.x0} height={Math.max(2, n.y1 - n.y0)} rx={2} style={{ fill: n.color }} />
              <text x={n.x1 + 6} y={(n.y0 + n.y1) / 2} dy="0.32em" style={{ fill: 'var(--ink-1)', fontSize: 11 }}>
                {n.label}{n.sub ? <tspan style={{ fill: 'var(--ink-3)' }}> {n.sub}</tspan> : null}
              </text>
            </g>
          ))}
          {zero.map((z, i) => (
            <g key={`z-${i}`} transform={`translate(0,${height - 18 - i * 16})`}>
              <line x1={10} x2={width - 160} y1={0} y2={0} style={{ stroke: 'var(--good)' }} strokeWidth={1} strokeDasharray="2 3" />
              <text x={width - 154} y={0} dy="0.32em" style={{ fill: 'var(--good-ink)', fontSize: 11, fontWeight: 500 }}>{z.label}</text>
            </g>
          ))}
        </svg>
      )}
      {hover && (
        <Tooltip x={hover.x} y={hover.y} containerWidth={width} title={hover.l.label}
          rows={[{ label: 'moved', value: format(hover.l.value), color: hover.l.color }]} />
      )}
    </div>
  )
}
