import clsx from 'clsx'
import { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { duration, relative, simOf } from '@/core/format'
import { PLANES } from '@/core/palette'
import { useLive } from '@/core/store'
import { useSimNow, useUi } from '@/core/ui-store'

// The Lifecycle Ribbon: the one bold element. Five planes side by side form a literal spectrum;
// pulses travel between segments when an artefact moves through the lifecycle.

const ROUTES: Record<string, string> = { learn: '/federation', verify: '/proving', release: '/release', operate: '/', prove: '/audit' }

export function Ribbon() {
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const fl = useLive((s) => s.federation)
  const pv = useLive((s) => s.proving)
  const rl = useLive((s) => s.release)
  const sw = useLive((s) => s.swarm)
  const lg = useLive((s) => s.ledger)
  const pulses = useLive((s) => s.live.pulses)
  const now = useSimNow(1)
  const lit = useUi((s) => s.lit.includes('ribbon'))

  const maxEps = Math.max(0, ...((fl.operators ?? []) as any[]).map((o) => o.eps))
  const active = pv.active
  const gate = active?.result ? (active.result.passed ? 'gate pass' : 'gate blocked') : active?.state === 'running' ? 'evaluating' : null
  const stage = rl.candidate ? rl.stages?.[rl.stage_index]?.label?.toLowerCase() : null
  const inc = sw.incident
  const incActive = inc && inc.status !== 'resolved'
  const lastAnchor = (lg.anchors ?? [])[lg.anchors?.length - 1]
  const broken = lg.last_verify && !lg.last_verify.ok

  const metrics: Record<string, { main: string; sub: string; alert?: boolean }> = {
    learn: {
      main: fl.state === 'running' ? `Round ${fl.round}/${fl.config?.rounds ?? 20}` : fl.state === 'ready' ? `Round ${(fl.round ?? 0) + 1} queued` : `Round ${fl.round ?? 0} ${fl.state ?? ''}`,
      sub: `ε max ${maxEps.toFixed(2)} of ${fl.config?.target_eps ?? 3}`,
    },
    verify: {
      main: active ? `${active.model}` : `${pv.production ?? 'nlm-v6'} in prod`,
      sub: gate ?? `${(pv.models ?? []).length} models benchmarked`,
      alert: active?.state === 'blocked',
    },
    release: {
      main: rl.candidate ? `${rl.candidate} ${stage}` : `${rl.prod ?? 'nlm-v6'} serving`,
      sub: rl.candidate ? `agreement ${rl.guards?.agreement?.toFixed?.(0) ?? '–'}%` : '4 targets, 2 vendors',
    },
    operate: {
      main: incActive ? `${inc.id} active` : `${(sw.agents ?? []).length} agents, 0 incidents`,
      sub: incActive ? `elapsed ${duration(now - simOf(inc.started_at))}` : inc?.mttr_s ? `last MTTR ${duration(inc.mttr_s)}` : 'network nominal',
      alert: !!incActive,
    },
    prove: {
      main: `Ledger #${(lg.height ?? 0).toLocaleString()}`,
      sub: broken ? `chain broken at #${lg.last_verify.broken?.seq}` : lastAnchor ? `anchored ${relative(now, simOf(lastAnchor.written_at))}` : 'signed',
      alert: !!broken,
    },
  }

  return (
    <div className="flex h-9 shrink-0 overflow-x-auto border-b border-line bg-page">
    <span className="flex shrink-0 items-center bg-warn/15 px-2.5 text-micro font-semibold tracking-wide text-warn"
      title="Every figure in this strip comes from the built-in simulator, not from measurements">SIMULATED</span>
    <nav aria-label="Lifecycle (simulated)" data-panel="ribbon" data-lit={lit ? 'true' : 'false'} className="relative flex h-9 min-w-0 flex-1" style={{ ['--lit-color' as string]: 'var(--ink-1)' }}>
      {PLANES.map((p, i) => {
        const m = metrics[p.id]
        const here = pathname.startsWith(ROUTES[p.id])
        return (
          <button
            key={p.id}
            type="button"
            onClick={() => navigate(ROUTES[p.id])}
            className={clsx('group relative flex min-w-0 flex-1 items-center gap-2.5 px-3 text-left transition-colors', i > 0 && 'border-l border-page')}
            style={{ background: `color-mix(in oklab, ${p.color} ${here ? 22 : 13}%, var(--page))` }}
          >
            <span className="absolute inset-x-0 top-0 h-[2px]" style={{ background: p.color }} />
            {here && <span className="absolute inset-x-3 bottom-0 h-[2px] rounded-full bg-ink-1" />}
            <span className="text-cap font-semibold text-ink-1">{p.label}</span>
            <span className="truncate text-cap text-ink-1">{m.main}</span>
            <span className={clsx('ml-auto truncate text-micro', m.alert ? 'font-medium text-critical-ink' : 'text-ink-2')}>{m.sub}</span>
          </button>
        )
      })}
      {pulses.map((p, i) => <Pulse key={`${p.at}-${p.from}-${p.to}-${i}`} from={p.from} to={p.to} at={p.at} />)}
    </nav>
    </div>
  )
}

function Pulse({ from, to, at }: { from: string; to: string; at: number }) {
  const idx = (id: string) => PLANES.findIndex((p) => p.id === id)
  const a = (idx(from) + 0.5) * 20
  const b = (idx(to) + 0.5) * 20 + (from === to ? 0 : 0)
  const wrap = idx(to) < idx(from)
  const [k, setK] = useState(0)
  useEffect(() => {
    let id = 0
    const step = () => {
      const t = Math.min(1, (performance.now() - at) / 900)
      setK(t)
      if (t < 1) id = requestAnimationFrame(step)
    }
    id = requestAnimationFrame(step)
    return () => cancelAnimationFrame(id)
  }, [at])
  if (k >= 1) return null
  const e = 1 - Math.pow(1 - k, 3)
  // a wrap-around (Prove -> Learn) travels right and re-enters from the left edge
  const pos = wrap ? (a + e * (100 - a + b)) % 100 : a + (b - a) * e
  return (
    <span aria-hidden className="pointer-events-none absolute top-0 h-full w-6 -translate-x-1/2"
      style={{ left: `${pos}%`, background: 'linear-gradient(90deg, transparent, color-mix(in oklab, white 55%, transparent), transparent)', opacity: 1 - k * 0.4 }} />
  )
}
