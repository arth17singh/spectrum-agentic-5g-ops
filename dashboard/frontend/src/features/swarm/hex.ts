// The 4-cell Maveric cluster as flat-topped hexagons: shared by Swarm Tower and Live Fix.
export const SQ3 = Math.sqrt(3)
export const hexPath = (cx: number, cy: number, R: number) =>
  [0, 60, 120, 180, 240, 300].map((d, i) => {
    const a = (d * Math.PI) / 180
    return `${i ? 'L' : 'M'}${(cx + R * Math.cos(a)).toFixed(1)},${(cy + R * Math.sin(a)).toFixed(1)}`
  }).join('') + 'Z'
// 4 cells as a compact flat-topped hex cluster: two on top, two below, offset
export const CELL_POS = [[0, 0], [1.5, SQ3 / 2], [0, SQ3], [1.5, -SQ3 / 2]] as const
