// Flat-topped hex geometry for the 4-cell Maveric cluster (same layout as Swarm Tower).
const SQ3 = Math.sqrt(3)

export const hexPath = (cx: number, cy: number, R: number) =>
  [0, 60, 120, 180, 240, 300].map((d, i) => {
    const a = (d * Math.PI) / 180
    return `${i ? 'L' : 'M'}${(cx + R * Math.cos(a)).toFixed(1)},${(cy + R * Math.sin(a)).toFixed(1)}`
  }).join('') + 'Z'

/** two on top, two below, offset: index = cell id */
export const CELL_POS = [[0, 0], [1.5, SQ3 / 2], [0, SQ3], [1.5, -SQ3 / 2]] as const

/** centre of a cell for a cluster drawn with radius R and origin (ox, oy) */
export const cellCentre = (cell: number, R: number, ox: number, oy: number, gap = 1.12) => {
  const [dx, dy] = CELL_POS[cell]
  return [ox + dx * R * gap, oy + dy * R * gap] as const
}

export const sign = (v: number, d = 1) => `${v > 0 ? '+' : ''}${v.toFixed(d)}`
export const pct = (v: number | null | undefined, d = 0) =>
  typeof v === 'number' && Number.isFinite(v) ? `${(v * 100).toFixed(d)}%` : '—'
