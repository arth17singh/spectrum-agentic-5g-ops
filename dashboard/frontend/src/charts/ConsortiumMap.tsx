import { Lock } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useAnimationFrame, useSize } from './common'

// Five operators on a ring around an attested enclave. Packets travel along the spokes when
// adapters move (operator -> enclave on upload, enclave -> operators when a round starts).

export interface ConsortiumNode {
  id: string
  short: string
  name: string
  color: string
  /** 0..1 local training progress, drawn as an arc around the node */
  progress: number
  status: string
  eps: number
  samples: number
  bytesUp: number
}

export interface Packet { id: number; from: string; to: string; color: string; born: number }

const DUR = 1300

export function ConsortiumMap({ nodes, packets, centreLabel, centreSub, attested, height = 330, format, reducedMotion }: {
  nodes: ConsortiumNode[]; packets: Packet[]; centreLabel: string; centreSub: string; attested: boolean; height?: number
  format: (b: number) => string; reducedMotion?: boolean
}) {
  const { ref, width } = useSize()
  const [now, setNow] = useState(performance.now())
  const active = packets.some((p) => now - p.born < DUR)
  useAnimationFrame((t) => setNow(t), active && !reducedMotion)
  // nudge one frame when new packets arrive so the loop starts
  const last = useRef(0)
  useEffect(() => {
    const newest = packets[packets.length - 1]?.born ?? 0
    if (newest !== last.current) { last.current = newest; setNow(performance.now()) }
  }, [packets])

  const W = width || 600
  const cx = W / 2
  const cy = height / 2 - 6
  const rx = Math.min(W * 0.36, 250)
  const ry = height * 0.34
  const pos = new Map<string, { x: number; y: number }>()
  nodes.forEach((n, i) => {
    const a = -Math.PI / 2 + (i / nodes.length) * Math.PI * 2
    pos.set(n.id, { x: cx + rx * Math.cos(a), y: cy + ry * Math.sin(a) })
  })
  pos.set('enclave', { x: cx, y: cy })
  const R = 24

  return (
    <div ref={ref} className="relative" style={{ height }}>
      {width > 0 && (
        <svg width={W} height={height} role="img" aria-label="Consortium: five operators sending masked adapter updates to an attested enclave">
          {/* spokes */}
          {nodes.map((n) => {
            const p = pos.get(n.id)!
            const mx = p.x + (cx - p.x) * 0.45
            const my = p.y + (cy - p.y) * 0.45
            return (
              <g key={`s-${n.id}`}>
                <line x1={p.x} y1={p.y} x2={cx} y2={cy} style={{ stroke: 'var(--hairline)' }} strokeWidth={1.5} />
                <g transform={`translate(${mx},${my})`}>
                  <rect x={-34} y={-9} width={68} height={18} rx={4} style={{ fill: 'var(--surface-1)', stroke: 'var(--hairline)' }} />
                  <text textAnchor="middle" dy="0.34em" className="tnum" style={{ fill: 'var(--ink-2)', fontSize: 10.5 }}>{format(n.bytesUp)}</text>
                </g>
              </g>
            )
          })}
          {/* packets */}
          {!reducedMotion && packets.map((pk) => {
            const k = (now - pk.born) / DUR
            if (k < 0 || k > 1) return null
            const a = pos.get(pk.from)
            const b = pos.get(pk.to)
            if (!a || !b) return null
            const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2
            const x = a.x + (b.x - a.x) * e
            const y = a.y + (b.y - a.y) * e
            return <rect key={pk.id} x={x - 4} y={y - 4} width={8} height={8} rx={1.5} style={{ fill: pk.color }} opacity={1 - Math.max(0, k - 0.85) * 6} />
          })}
          {/* enclave */}
          <g transform={`translate(${cx},${cy})`}>
            <rect x={-84} y={-38} width={168} height={76} rx={10} style={{ fill: 'var(--surface-2)', stroke: 'var(--learn)' }} strokeWidth={1.5} />
            <text x={8} y={-12} textAnchor="middle" style={{ fill: 'var(--ink-1)', fontSize: 12, fontWeight: 600 }}>{centreLabel}</text>
            <text y={4} textAnchor="middle" style={{ fill: 'var(--ink-3)', fontSize: 10.5 }}>{centreSub}</text>
            <text y={22} textAnchor="middle" style={{ fill: attested ? 'var(--good-ink)' : 'var(--ink-3)', fontSize: 10.5, fontWeight: 500 }}>
              {attested ? 'PCR0 attested' : 'enclave off'}
            </text>
          </g>
          <foreignObject x={cx - 66} y={cy - 24} width={16} height={16}>
            <Lock size={13} style={{ color: 'var(--learn)' }} />
          </foreignObject>
          {/* operators */}
          {nodes.map((n) => {
            const p = pos.get(n.id)!
            const below = p.y >= cy
            const circ = 2 * Math.PI * (R + 4)
            const dim = n.status === 'dropped' || n.status === 'budget_exhausted'
            return (
              <g key={n.id} transform={`translate(${p.x},${p.y})`} opacity={dim ? 0.45 : 1}>
                <circle r={R + 4} fill="none" style={{ stroke: 'var(--surface-3)' }} strokeWidth={3} />
                {n.progress > 0 && (
                  <circle r={R + 4} fill="none" style={{ stroke: n.color, transition: 'stroke-dashoffset 300ms linear' }} strokeWidth={3}
                    strokeDasharray={circ} strokeDashoffset={circ * (1 - n.progress)} transform="rotate(-90)" strokeLinecap="round" />
                )}
                <circle r={R} style={{ fill: 'var(--surface-1)', stroke: n.color }} strokeWidth={2} />
                <text textAnchor="middle" dy="-0.15em" style={{ fill: 'var(--ink-1)', fontSize: 12, fontWeight: 600 }}>{n.short}</text>
                <text textAnchor="middle" dy="1.1em" className="tnum" style={{ fill: 'var(--ink-3)', fontSize: 10 }}>ε {n.eps.toFixed(2)}</text>
                <text textAnchor="middle" y={below ? R + 20 : -R - 22} style={{ fill: 'var(--ink-2)', fontSize: 11 }}>{n.name}</text>
                <text textAnchor="middle" y={below ? R + 34 : -R - 9} className="tnum" style={{ fill: 'var(--ink-3)', fontSize: 10 }}>
                  {n.samples.toLocaleString()} private incidents
                </text>
              </g>
            )
          })}
        </svg>
      )}
    </div>
  )
}
