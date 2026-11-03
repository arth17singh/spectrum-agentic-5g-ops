import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import { useState } from 'react'
import { AXIS_TEXT, Legend, Tooltip, useSize } from '@/charts/common'
import type { PauseEvent, TrustPoint } from './types'

// The trust line of one model version. Two kinds of pause look different on purpose: an automatic pause
// by the trust gate is a solid red line with an octagon and the word AUTO; a pause by a person is a dashed
// grey line with a hollow diamond and the word PERSON. A resume is a dashed green line with a hollow circle.

const utcClock = (sec: number) => new Date(sec * 1000).toISOString().slice(11, 19)
const f2 = (v: number | null) => (v == null ? '—' : v.toFixed(2))

export function TrustChart({ points, pauses, pauseBelow, height = 260 }: {
  points: TrustPoint[]
  pauses: PauseEvent[]
  pauseBelow: number | null
  height?: number
}) {
  const { ref, width } = useSize()
  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(null)
  const m = { l: 40, r: 16, t: 26, b: 24 }
  const w = Math.max(10, width - m.l - m.r)
  const h = Math.max(10, height - m.t - m.b)

  const pts = points.filter((p) => p.trust_after != null) as (TrustPoint & { trust_after: number })[]
  const ts = [...pts.map((p) => p.t), ...pauses.map((p) => p.t)]
  const lo = ts.length ? Math.min(...ts) : 0
  const hi = ts.length ? Math.max(...ts) : 1
  const pad = Math.max(5, (hi - lo) * 0.04)
  const x = scaleLinear().domain([lo - pad, hi + pad]).range([0, w])
  const y = scaleLinear().domain([0, 1]).range([h, 0])
  const path = line<TrustPoint & { trust_after: number }>().x((p) => x(p.t)).y((p) => y(p.trust_after))(pts)

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    if (!pts.length) return
    const r = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - r.left
    let best = 0
    for (let i = 1; i < pts.length; i++) if (Math.abs(x(pts[i].t) - px) < Math.abs(x(pts[best].t) - px)) best = i
    setHover({ i: best, x: x(pts[best].t) + m.l, y: y(pts[best].trust_after) + m.t })
  }
  const hp = hover ? pts[hover.i] : null

  return (
    <div>
      <Legend className="mb-2" items={[
        { label: 'Trust after each scored change', color: 'var(--release)' },
        { label: 'Pause line', color: 'var(--critical)', dashed: true },
        { label: 'Change with e above τ', color: 'var(--critical)', kind: 'dot' },
      ]} />
      <div ref={ref} className="relative" style={{ height }}>
        {width > 0 && (
          <svg width={width} height={height} role="img" aria-label="Trust over time with pause markers">
            <g transform={`translate(${m.l},${m.t})`}>
              {[0, 0.25, 0.5, 0.75, 1].map((t) => (
                <g key={t} transform={`translate(0,${y(t)})`}>
                  <line x1={0} x2={w} style={{ stroke: 'var(--hairline)' }} />
                  <text x={-8} dy="0.32em" textAnchor="end" style={AXIS_TEXT} className="tnum">{t.toFixed(2)}</text>
                </g>
              ))}
              <line x1={0} x2={w} y1={h} y2={h} style={{ stroke: 'var(--axis)' }} />
              {x.ticks(Math.max(2, Math.floor(w / 110))).map((t) => (
                <text key={t} x={x(t)} y={h + 16} textAnchor="middle" style={AXIS_TEXT} className="tnum">{utcClock(t)}Z</text>
              ))}
              {pauseBelow != null && (
                <g>
                  <line x1={0} x2={w} y1={y(pauseBelow)} y2={y(pauseBelow)} style={{ stroke: 'var(--critical)' }} strokeDasharray="5 3" />
                  <text x={w} y={y(pauseBelow) + 12} textAnchor="end" style={{ ...AXIS_TEXT, fill: 'var(--critical-ink)' }}>pause below {pauseBelow.toFixed(2)}</text>
                </g>
              )}
              {pauses.map((p, i) => <PauseMarker key={i} p={p} x={x(p.t)} h={h} />)}
              {path && <path d={path} fill="none" style={{ stroke: 'var(--release)' }} strokeWidth={2} strokeLinejoin="round" />}
              {pts.map((p, i) => (
                <circle key={i} cx={x(p.t)} cy={y(p.trust_after)} r={p.over_tau ? 5 : 3.5}
                  style={{ fill: p.over_tau ? 'var(--critical)' : 'var(--release)', stroke: 'var(--surface-1)' }} strokeWidth={2} />
              ))}
              <rect width={w} height={h} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
            </g>
          </svg>
        )}
        {hp && hover && (
          <Tooltip x={hover.x} y={hover.y} containerWidth={width} title={`${utcClock(hp.t)}Z · ${hp.change_id ?? ''}`} rows={[
            { label: 'trust after', value: f2(hp.trust_after) },
            { label: 'error e', value: f2(hp.error_summary) },
            { label: 'direction', value: hp.direction_agree == null ? '—' : hp.direction_agree ? 'agrees' : 'disagrees' },
          ]} />
        )}
        {!pts.length && <div className="absolute inset-0 grid place-items-center text-cap text-ink-3">No scored change with a recorded trust value yet.</div>}
      </div>
    </div>
  )
}

function PauseMarker({ p, x, h }: { p: PauseEvent; x: number; h: number }) {
  const auto = p.by_kind === 'trust-gate'
  const resume = p.action === 'resume'
  const color = resume ? 'var(--good)' : auto ? 'var(--critical)' : 'var(--ink-2)'
  const ink = resume ? 'var(--good-ink)' : auto ? 'var(--critical-ink)' : 'var(--ink-1)'
  const word = resume ? (auto ? 'RESUME · AUTO' : 'RESUME · PERSON') : auto ? 'AUTO-PAUSE' : 'PAUSE · PERSON'
  return (
    <g transform={`translate(${x},0)`}>
      <title>{`${word} by ${p.by} at ${p.time}\n${p.reason}`}</title>
      <line y1={-6} y2={h} style={{ stroke: color }} strokeWidth={auto && !resume ? 2.5 : 1.5} strokeDasharray={auto && !resume ? undefined : '4 3'} />
      {auto && !resume && <polygon points="-6,-12 6,-12 11,-6 11,2 6,8 -6,8 -11,2 -11,-6" transform="translate(0,-4) scale(0.75)" style={{ fill: color }} />}
      {!auto && !resume && <rect x={-5} y={-12} width={10} height={10} transform="rotate(45 0 -7)" style={{ fill: 'var(--surface-1)', stroke: color }} strokeWidth={2} />}
      {resume && <circle cy={-7} r={5} style={{ fill: 'var(--surface-1)', stroke: color }} strokeWidth={2} />}
      <text x={9} y={-4} style={{ ...AXIS_TEXT, fill: ink, fontWeight: 600 }}>{word}</text>
    </g>
  )
}
