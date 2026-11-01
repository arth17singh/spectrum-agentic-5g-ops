import { useEffect, useMemo, useRef, useState } from 'react'
import { cssVar } from '@/core/palette'
import { EMPTY, useLive } from '@/core/store'
import type { A2AMessage, Agent, Rec } from '@/core/types'
import { useUi } from '@/core/ui-store'
import { Tooltip, useSize, useThemeKey } from './common'
import { hexCenter } from './HexMap'

// Radial swarm graph on canvas: coordinator at the centre, domain agents on ring 1, 40 site agents
// on ring 2 in the same angular order as their hex positions. Each A2A message becomes a particle
// travelling sender -> receiver, coloured by verdict; the edge lingers for 4 s. Policy decisions
// pulse the principal's node (amber = needs a human, red = denied).

interface NodePos { id: string; x: number; y: number; r: number; cls: Agent['class'] }
interface Particle { from: string; to: string; color: string; start: number; dur: number; msg: A2AMessage }
interface EdgeT { a: string; b: string; color: string; t: number; msg: A2AMessage }

const DOMAIN_ORDER = ['transport', 'energy', 'cx', 'core', 'sentinel']

export function SwarmGraph({ height = 420 }: { height?: number }) {
  const { ref, width } = useSize()
  const canvas = useRef<HTMLCanvasElement>(null)
  const theme = useThemeKey()
  const motion = useUi((s) => s.motion)
  const inspect = useUi((s) => s.inspect)
  const agentIds = useLive((s) => (s.swarm.agents ?? EMPTY).map((a: Agent) => a.id).join(','))
  const sites = useLive((s) => s.twin.sites ?? EMPTY)
  const [hover, setHover] = useState<{ id: string; x: number; y: number } | null>(null)
  const parts = useRef<Particle[]>([])
  const edges = useRef<Map<string, EdgeT>>(new Map())
  const pulses = useRef<{ id: string; color: string; t: number }[]>([])
  const mouse = useRef<{ x: number; y: number } | null>(null)

  const colors = useMemo(() => ({
    ink1: cssVar('--ink-1'), ink2: cssVar('--ink-2'), ink3: cssVar('--ink-3'), line: cssVar('--hairline'), surface: cssVar('--surface-1'),
    surface2: cssVar('--surface-2'), surface3: cssVar('--surface-3'), operate: cssVar('--operate'), release: cssVar('--release'),
    warn: cssVar('--warn'), crit: cssVar('--critical'), critInk: cssVar('--critical-ink'), learn: cssVar('--learn'), prove: cssVar('--prove'),
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [theme])

  // layout
  const nodes = useMemo(() => {
    const out = new Map<string, NodePos>()
    if (!width) return out
    const cx = width / 2
    const cy = height / 2
    const R2 = Math.min(width, height) / 2 - 22
    const R1 = R2 * 0.46
    out.set('coordinator', { id: 'coordinator', x: cx, y: cy, r: 13, cls: 'coordinator' })
    DOMAIN_ORDER.forEach((id, i) => {
      const a = -Math.PI / 2 + (i / DOMAIN_ORDER.length) * Math.PI * 2
      out.set(id, { id, x: cx + R1 * Math.cos(a), y: cy + R1 * Math.sin(a), r: 9, cls: 'domain' })
    })
    // site agents follow the hex map's angular order around its centre
    const pts = (sites as Rec[]).map((s) => ({ id: `site-${s.id.slice(2)}`, ...hexCenter(s.col, s.row, 1) }))
    if (pts.length) {
      const mx = pts.reduce((a, p) => a + p.x, 0) / pts.length
      const my = pts.reduce((a, p) => a + p.y, 0) / pts.length
      const sorted = pts.map((p) => ({ id: p.id, ang: Math.atan2(p.y - my, p.x - mx) })).sort((a, b) => a.ang - b.ang)
      const start = sorted[0].ang
      sorted.forEach((p, i) => {
        const a = start + (i / sorted.length) * Math.PI * 2
        out.set(p.id, { id: p.id, x: cx + R2 * Math.cos(a), y: cy + R2 * Math.sin(a), r: 4.5, cls: 'site' })
      })
    }
    return out
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, height, sites, agentIds])

  // feed particles from the live store (no React re-render per message)
  useEffect(() => {
    const seen = new Set<string>((useLive.getState().swarm.messages ?? []).map((m: A2AMessage) => m.id))
    const seenDec = new Set<string>((useLive.getState().trust.decisions ?? []).map((d: Rec) => d.id))
    return useLive.subscribe((s, prev) => {
      const now = performance.now()
      if (s.swarm.messages !== prev.swarm.messages) {
        for (const m of (s.swarm.messages ?? []) as A2AMessage[]) {
          if (seen.has(m.id)) continue
          seen.add(m.id)
          const color = m.verdict === 'blocked' ? colors.crit : m.intent === 'object' ? colors.warn : colors.ink2
          parts.current.push({ from: m.from, to: m.to, color, start: now, dur: m.verdict === 'blocked' ? 1300 : 900, msg: m })
          edges.current.set(`${m.from}>${m.to}`, { a: m.from, b: m.to, color, t: now, msg: m })
        }
      }
      if (s.trust.decisions !== prev.trust.decisions) {
        for (const d of (s.trust.decisions ?? []) as Rec[]) {
          if (seenDec.has(d.id)) continue
          seenDec.add(d.id)
          if (d.decision === 'needs_human') pulses.current.push({ id: d.principal, color: colors.warn, t: now })
          if (d.decision === 'deny') pulses.current.push({ id: d.principal, color: colors.crit, t: now })
        }
      }
    })
  }, [colors])

  // render loop
  useEffect(() => {
    const cv = canvas.current
    if (!cv || !width) return
    const dpr = window.devicePixelRatio || 1
    cv.width = width * dpr
    cv.height = height * dpr
    const ctx = cv.getContext('2d')!
    let raf = 0
    const reduced = motion === 'reduced' || window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const draw = (now: number) => {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, width, height)
      const agents = new Map<string, Agent>(((useLive.getState().swarm.agents ?? []) as Agent[]).map((a) => [a.id, a]))
      const c = nodes.get('coordinator')
      if (!c) { raf = requestAnimationFrame(draw); return }
      // guides
      ctx.strokeStyle = colors.line
      ctx.lineWidth = 1
      const r1 = Math.hypot((nodes.get('transport')?.x ?? c.x) - c.x, (nodes.get('transport')?.y ?? c.y) - c.y)
      const site0 = [...nodes.values()].find((n) => n.cls === 'site')
      const r2 = site0 ? Math.hypot(site0.x - c.x, site0.y - c.y) : r1 * 2
      for (const rr of [r1, r2]) { ctx.beginPath(); ctx.arc(c.x, c.y, rr, 0, Math.PI * 2); ctx.stroke() }
      for (const id of DOMAIN_ORDER) {
        const n = nodes.get(id)
        if (!n) continue
        ctx.beginPath(); ctx.moveTo(c.x, c.y); ctx.lineTo(n.x, n.y); ctx.stroke()
      }
      // edges (fade over 4 s)
      for (const [k, e] of edges.current) {
        const age = now - e.t
        if (age > 4000) { edges.current.delete(k); continue }
        const a = nodes.get(e.a), b = nodes.get(e.b)
        if (!a || !b) continue
        ctx.globalAlpha = 0.85 * (1 - age / 4000)
        ctx.strokeStyle = e.color
        ctx.lineWidth = e.color === colors.crit ? 2 : 1.25
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke()
      }
      ctx.globalAlpha = 1
      // particles
      parts.current = parts.current.filter((p) => now - p.start < p.dur)
      if (!reduced) {
        for (const p of parts.current) {
          const a = nodes.get(p.from), b = nodes.get(p.to)
          if (!a || !b) continue
          const k = (now - p.start) / p.dur
          const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2
          const x = a.x + (b.x - a.x) * e, y = a.y + (b.y - a.y) * e
          ctx.fillStyle = p.color
          ctx.beginPath(); ctx.arc(x, y, p.color === colors.crit ? 4 : 3, 0, Math.PI * 2); ctx.fill()
          ctx.strokeStyle = colors.surface; ctx.lineWidth = 1.5; ctx.stroke()
        }
      }
      // pulses
      pulses.current = pulses.current.filter((p) => now - p.t < 1600)
      for (const p of pulses.current) {
        const n = nodes.get(p.id)
        if (!n) continue
        const k = (now - p.t) / 1600
        ctx.strokeStyle = p.color
        ctx.globalAlpha = 1 - k
        ctx.lineWidth = 2
        ctx.beginPath(); ctx.arc(n.x, n.y, n.r + 3 + k * 16, 0, Math.PI * 2); ctx.stroke()
      }
      ctx.globalAlpha = 1
      // nodes
      for (const n of nodes.values()) {
        const ag = agents.get(n.id)
        const quarantined = ag?.status === 'quarantined'
        const activity = ag ? Math.min(3, ag.msgs_min / 4) : 0
        const r = n.r + (n.cls === 'site' ? activity : 0)
        ctx.fillStyle = quarantined ? colors.surface3 : n.cls === 'site' ? colors.surface2 : colors.operate
        ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, Math.PI * 2); ctx.fill()
        ctx.lineWidth = 2
        ctx.strokeStyle = colors.surface
        ctx.stroke()
        const state = ag?.state
        if (quarantined) {
          // lock glyph
          ctx.strokeStyle = colors.ink2; ctx.lineWidth = 1.25
          ctx.strokeRect(n.x - 3, n.y - 1, 6, 5)
          ctx.beginPath(); ctx.arc(n.x, n.y - 1.5, 2.2, Math.PI, 0); ctx.stroke()
          ctx.strokeStyle = colors.crit; ctx.lineWidth = 1.5
          ctx.beginPath(); ctx.arc(n.x, n.y, r + 4, 0, Math.PI * 2); ctx.stroke()
        } else if (state === 'thinking') {
          ctx.strokeStyle = colors.release; ctx.lineWidth = 1.75
          ctx.setLineDash([3, 3])
          const rot = reduced ? 0 : (now / 600) % (Math.PI * 2)
          ctx.beginPath(); ctx.arc(n.x, n.y, r + 3.5, rot, rot + Math.PI * 1.6); ctx.stroke()
          ctx.setLineDash([])
        } else if (state === 'acting') {
          ctx.strokeStyle = colors.operate; ctx.lineWidth = 2
          ctx.beginPath(); ctx.arc(n.x, n.y, r + 3.5, 0, Math.PI * 2); ctx.stroke()
        } else if (n.cls === 'site') {
          ctx.strokeStyle = colors.ink3; ctx.lineWidth = 1; ctx.globalAlpha = 0.6
          ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, Math.PI * 2); ctx.stroke(); ctx.globalAlpha = 1
        }
        if (ag && ag.model !== 'nlm-v6' && n.cls === 'site') {
          // canary cohort marker: a small amber dot
          ctx.fillStyle = colors.warn
          ctx.beginPath(); ctx.arc(n.x + r, n.y - r, 2, 0, Math.PI * 2); ctx.fill()
        }
        if (n.cls !== 'site') {
          ctx.fillStyle = colors.ink1
          ctx.font = `${n.cls === 'coordinator' ? 600 : 500} 11px "IBM Plex Sans", sans-serif`
          ctx.textAlign = 'center'
          ctx.textBaseline = 'top'
          ctx.fillText(ag?.label ?? n.id, n.x, n.y + r + 5)
        }
      }
      // hovered node label
      const m = mouse.current
      if (m) {
        const hn = nearest(nodes, m.x, m.y)
        if (hn) {
          ctx.strokeStyle = colors.ink1; ctx.lineWidth = 1.5
          ctx.beginPath(); ctx.arc(hn.x, hn.y, hn.r + 6, 0, Math.PI * 2); ctx.stroke()
        }
      }
      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [width, height, nodes, colors, motion])

  const hoveredAgent = hover ? ((useLive.getState().swarm.agents ?? []) as Agent[]).find((a) => a.id === hover.id) : null

  return (
    <div ref={ref} className="relative" style={{ height }}>
      <canvas
        ref={canvas}
        role="img"
        aria-label="Agent swarm: coordinator, five domain agents and forty site agents with live messages"
        style={{ width, height, display: 'block', cursor: hover ? 'pointer' : 'default' }}
        onPointerMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect()
          const x = e.clientX - r.left, y = e.clientY - r.top
          mouse.current = { x, y }
          const n = nearest(nodes, x, y)
          setHover((h) => (n ? (h?.id === n.id && Math.abs(h.x - x) < 8 && Math.abs(h.y - y) < 8 ? h : { id: n.id, x, y }) : null))
        }}
        onPointerLeave={() => { mouse.current = null; setHover(null) }}
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect()
          const x = e.clientX - r.left, y = e.clientY - r.top
          const n = nearest(nodes, x, y)
          if (n) { inspect({ kind: 'agent', id: n.id }); return }
          // otherwise: the closest live edge opens its message
          let best: EdgeT | null = null
          let bd = 8
          for (const ed of edges.current.values()) {
            const a = nodes.get(ed.a), b = nodes.get(ed.b)
            if (!a || !b) continue
            const d = segDist(x, y, a.x, a.y, b.x, b.y)
            if (d < bd) { bd = d; best = ed }
          }
          if (best) inspect({ kind: 'message', id: best.msg.id, title: best.msg.summary, data: best.msg })
        }}
      />
      {hover && hoveredAgent && (
        <Tooltip x={hover.x} y={hover.y} containerWidth={width} title={`${hoveredAgent.id}, ${hoveredAgent.class} agent`} rows={[
          { label: 'state', value: hoveredAgent.status === 'quarantined' ? 'quarantined' : hoveredAgent.state },
          { label: 'model', value: hoveredAgent.model },
          { label: 'autonomy', value: `L${hoveredAgent.autonomy}` },
          { label: 'trust', value: hoveredAgent.trust.toFixed(0) },
          { label: 'last action', value: hoveredAgent.last_action || '—' },
        ]} />
      )}
    </div>
  )
}

function nearest(nodes: Map<string, NodePos>, x: number, y: number): NodePos | null {
  let best: NodePos | null = null
  let bd = Infinity
  for (const n of nodes.values()) {
    const d = Math.hypot(n.x - x, n.y - y)
    if (d < n.r + 7 && d < bd) { bd = d; best = n }
  }
  return best
}

function segDist(px: number, py: number, ax: number, ay: number, bx: number, by: number) {
  const dx = bx - ax, dy = by - ay
  const L = dx * dx + dy * dy || 1
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / L))
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy))
}

export function SwarmLegend() {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-micro text-ink-3">
      <span className="flex items-center gap-1.5"><span className="size-3 rounded-full" style={{ background: 'var(--operate)' }} />coordinator / domain</span>
      <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-s2 ring-1 ring-ink-3" />site agent</span>
      <span className="flex items-center gap-1.5"><svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="none" stroke="var(--release)" strokeWidth="1.75" strokeDasharray="3 3" /></svg>thinking</span>
      <span className="flex items-center gap-1.5"><svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="none" stroke="var(--operate)" strokeWidth="2" /></svg>acting</span>
      <span className="flex items-center gap-1.5"><svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="none" stroke="var(--critical)" strokeWidth="1.5" /></svg>quarantined</span>
      <span className="flex items-center gap-1.5"><span className="size-1.5 rounded-full bg-warn" />canary model</span>
      <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-ink-2" />message delivered</span>
      <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-warn" />objection / needs a human</span>
      <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-critical" />blocked / denied</span>
    </div>
  )
}
