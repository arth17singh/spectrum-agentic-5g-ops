import { scaleLinear } from 'd3-scale'
import { useEffect, useRef, useState } from 'react'
import { cssVar } from '@/core/palette'
import { Tooltip, useSize, useThemeKey } from './common'

/** Canvas heatmap. Diverging (blue - gray - red, symmetric around 0) or sequential (single blue hue). */
export function Heatmap({ matrix, scheme = 'diverging', domain, height, gap = 1, rowLabels, colLabels, format = (v) => v.toFixed(3), cellText, ariaLabel, square }: {
  matrix: number[][]; scheme?: 'diverging' | 'sequential'; domain?: [number, number]; height?: number; gap?: number
  rowLabels?: string[]; colLabels?: string[]; format?: (v: number) => string; cellText?: boolean; ariaLabel?: string; square?: boolean
}) {
  const { ref, width } = useSize()
  const canvas = useRef<HTMLCanvasElement>(null)
  const theme = useThemeKey()
  const [hover, setHover] = useState<{ r: number; c: number; x: number; y: number } | null>(null)
  const rows = matrix.length
  const cols = matrix[0]?.length ?? 0
  const left = rowLabels ? 112 : 0
  const top = colLabels ? 18 : 0
  const cw = cols ? (width - left) / cols : 0
  const H = height ?? (square ? Math.round(cw * rows) + top : rows * 22 + top)
  const ch = rows ? (H - top) / rows : 0

  useEffect(() => {
    const cv = canvas.current
    if (!cv || !width || !rows) return
    const dpr = window.devicePixelRatio || 1
    cv.width = width * dpr
    cv.height = H * dpr
    const ctx = cv.getContext('2d')!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, width, H)
    const flat = matrix.flat()
    let d = domain
    if (!d) {
      if (scheme === 'diverging') {
        const m = Math.max(1e-9, ...flat.map((v) => Math.abs(v)))
        d = [-m, m]
      } else d = [Math.min(...flat), Math.max(...flat)]
    }
    const color = scheme === 'diverging'
      ? scaleLinear<string>().domain([d[0], (d[0] + d[1]) / 2, d[1]]).range([cssVar('--div-neg'), cssVar('--div-mid'), cssVar('--div-pos')]).clamp(true)
      : scaleLinear<string>().domain(d).range([cssVar('--seq-lo'), cssVar('--seq-hi')]).clamp(true)
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = matrix[r][c]
        ctx.fillStyle = Number.isFinite(v) ? color(v) : cssVar('--surface-2')
        ctx.fillRect(left + c * cw + gap / 2, top + r * ch + gap / 2, Math.max(0.5, cw - gap), Math.max(0.5, ch - gap))
        if (cellText && cw > 34 && ch > 16 && Number.isFinite(v)) {
          const k = (v - d[0]) / (d[1] - d[0] || 1)
          ctx.fillStyle = (scheme === 'sequential' ? (theme === 'projector' ? k > 0.55 : k < 0.55) : Math.abs(k - 0.5) > 0.3) ? '#ffffff' : '#0b0b0b'
          ctx.font = '500 11px "IBM Plex Sans", sans-serif'
          ctx.textAlign = 'center'
          ctx.textBaseline = 'middle'
          ctx.fillText(format(v), left + c * cw + cw / 2, top + r * ch + ch / 2)
        }
      }
    }
    ctx.fillStyle = cssVar('--ink-3')
    ctx.font = '11px "IBM Plex Sans", sans-serif'
    ctx.textBaseline = 'middle'
    ctx.textAlign = 'left'
    rowLabels?.forEach((l, r) => ctx.fillText(l, 0, top + r * ch + ch / 2, left - 8))
    ctx.textAlign = 'center'
    colLabels?.forEach((l, c) => ctx.fillText(l, left + c * cw + cw / 2, 8, cw - 2))
  }, [matrix, width, H, scheme, domain, theme, rows, cols, cw, ch, left, top, gap, rowLabels, colLabels, cellText, format])

  return (
    <div ref={ref} className="relative" style={{ height: H }}>
      <canvas
        ref={canvas}
        role="img"
        aria-label={ariaLabel ?? 'Heatmap'}
        style={{ width, height: H, display: 'block' }}
        onPointerMove={(e) => {
          const rct = e.currentTarget.getBoundingClientRect()
          const px = e.clientX - rct.left
          const py = e.clientY - rct.top
          const c = Math.floor((px - left) / cw)
          const r = Math.floor((py - top) / ch)
          if (r >= 0 && r < rows && c >= 0 && c < cols) setHover({ r, c, x: px, y: py })
          else setHover(null)
        }}
        onPointerLeave={() => setHover(null)}
      />
      {hover && (
        <Tooltip x={hover.x} y={hover.y} containerWidth={width}
          title={`${rowLabels?.[hover.r] ?? `row ${hover.r}`} · ${colLabels?.[hover.c] ?? `col ${hover.c}`}`}
          rows={[{ label: 'value', value: format(matrix[hover.r][hover.c]) }]} />
      )}
    </div>
  )
}
