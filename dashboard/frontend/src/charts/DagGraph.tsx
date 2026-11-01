import clsx from 'clsx'
import { useMemo, type ReactNode } from 'react'
import { useSize } from './common'

// Left-to-right DAG. Layers come from BFS depth from the root; when the layers don't fit the width
// they wrap into bands (still read left to right, band by band). Edges carry their relation; the
// path from the selected node back to the root is highlighted.

export interface DagNode { id: string; render: ReactNode; color?: string; height?: number }
export interface DagEdge { from: string; to: string; rel: string }

const W = 180
const H = 70
const COL = 248
const VGAP = 14
const BAND_GAP = 56

export function DagGraph({ root, nodes, edges, selected, onSelect }: {
  root: string; nodes: DagNode[]; edges: DagEdge[]; selected?: string | null; onSelect?: (id: string) => void
}) {
  const { ref, width } = useSize()
  const layout = useMemo(() => {
    const depth: Record<string, number> = { [root]: 0 }
    const q = [root]
    while (q.length) {
      const n = q.shift()!
      for (const e of edges) if (e.from === n && depth[e.to] == null) { depth[e.to] = depth[n] + 1; q.push(e.to) }
    }
    const hOf = (id: string) => nodes.find((n) => n.id === id)?.height ?? H
    const cols: string[][] = []
    for (const n of nodes) (cols[depth[n.id] ?? 0] ??= []).push(n.id)
    const perRow = Math.max(3, Math.floor((Math.max(width, W) + COL - W) / COL))
    const colH = cols.map((ids) => (ids ?? []).reduce((a, id) => a + hOf(id), 0) + VGAP * Math.max(0, (ids ?? []).length - 1))
    const bands = Math.ceil(cols.length / perRow)
    const bandH: number[] = []
    for (let b = 0; b < bands; b++) bandH.push(Math.max(H, ...colH.slice(b * perRow, (b + 1) * perRow)))
    const bandY = bandH.map((_, b) => bandH.slice(0, b).reduce((a, x) => a + x + BAND_GAP, 0))
    const pos: Record<string, { x: number; y: number; h: number; band: number; bandTop: number }> = {}
    cols.forEach((ids, c) => {
      const band = Math.floor(c / perRow)
      let y = bandY[band] + (bandH[band] - colH[c]) / 2
      for (const id of ids ?? []) {
        const h = hOf(id)
        pos[id] = { x: (c % perRow) * COL, y, h, band, bandTop: bandY[band] }
        y += h + VGAP
      }
    })
    const parent: Record<string, string> = {}
    for (const e of edges) if (parent[e.to] == null) parent[e.to] = e.from
    const usedCols = Math.min(perRow, cols.length)
    return { pos, width: usedCols * COL - (COL - W), height: bandY[bands - 1] + bandH[bands - 1], parent }
  }, [root, nodes, edges, width])

  const onPath = useMemo(() => {
    const s = new Set<string>()
    let cur = selected ?? null
    while (cur) { s.add(cur); cur = layout.parent[cur] ?? null }
    return s
  }, [selected, layout])

  return (
    <div ref={ref} className="overflow-x-auto pb-2">
      {width > 0 && (
        <div className="relative" style={{ width: layout.width + 2, height: layout.height + 4 }}>
          <svg className="absolute inset-0" width={layout.width + 2} height={layout.height + 4} aria-hidden>
            {edges.map((e, i) => {
              const a = layout.pos[e.from]
              const b = layout.pos[e.to]
              if (!a || !b) return null
              const hot = onPath.has(e.from) && onPath.has(e.to)
              let d: string
              let lx: number
              let ly: number
              if (a.band === b.band) {
                const x1 = a.x + W, y1 = a.y + a.h / 2, x2 = b.x, y2 = b.y + b.h / 2, mx = (x1 + x2) / 2
                d = `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`
                lx = mx; ly = (y1 + y2) / 2 - 5
              } else {
                // wrap to the next band: drop from the source, run left through the gap, enter the target from above
                const x1 = a.x + W / 2, y1 = a.y + a.h, x2 = b.x + W / 2, y2 = b.y
                const gy = b.bandTop - BAND_GAP / 2
                const r = 10
                d = `M${x1},${y1} V${gy - r} Q${x1},${gy} ${x1 - r},${gy} H${x2 + r} Q${x2},${gy} ${x2},${gy + r} V${y2}`
                lx = (x1 + x2) / 2; ly = gy - 5
              }
              return (
                <g key={i}>
                  <path d={d} fill="none" style={{ stroke: hot ? 'var(--prove)' : 'var(--axis)' }} strokeWidth={hot ? 2 : 1.25} />
                  <text x={lx} y={ly} textAnchor="middle" style={{ fill: hot ? 'var(--ink-1)' : 'var(--ink-3)', fontSize: 10, paintOrder: 'stroke', stroke: 'var(--surface-1)', strokeWidth: 3 }}>{e.rel}</text>
                </g>
              )
            })}
          </svg>
          {nodes.map((n) => {
            const p = layout.pos[n.id]
            if (!p) return null
            return (
              <button
                key={n.id}
                type="button"
                onClick={() => onSelect?.(n.id)}
                className={clsx('absolute overflow-hidden rounded-md border bg-s2 px-2.5 py-2 text-left transition-colors hover:bg-s3',
                  onPath.has(n.id) ? 'border-prove' : 'border-line', selected === n.id && 'bg-s3')}
                style={{ left: p.x, top: p.y, width: W, height: p.h }}
              >
                <span className="absolute inset-y-0 left-0 w-[3px]" style={{ background: n.color ?? 'var(--ink-3)' }} />
                {n.render}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
