import { Flag, Moon, Search, Snowflake, Sun } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '@/core/api'
import { Gate } from '@/core/flags'
import { EMPTY, useLive } from '@/core/store'
import { simClockLabel, toast, useSimNow, useUi } from '@/core/ui-store'
import { HoldButton, IconButton, Kbd, LiveDot } from '@/ui'

export function Wordmark() {
  const bars = [
    ['var(--learn)', 9], ['var(--verify)', 14], ['var(--release)', 7], ['var(--operate)', 12], ['var(--prove)', 5],
  ] as const
  return (
    <span className="flex items-center gap-2">
      <svg width="22" height="16" viewBox="0 0 22 16" aria-hidden>
        {bars.map(([c, h], i) => (
          <rect key={i} x={i * 4.4} y={(16 - h) / 2} width="3" height={h} rx="1" style={{ fill: c }} />
        ))}
      </svg>
      <span className="text-[15px] font-semibold tracking-[0.04em] text-ink-1">SPECTRUM</span>
    </span>
  )
}

export function TopBar() {
  const navigate = useNavigate()
  const conn = useLive((s) => s.conn)
  const ready = useLive((s) => s.ready)
  const clock = useLive((s) => s.clock)
  const mode = useLive((s) => s.mode)
  const rate = useLive((s) => s.msgPerSec)
  const items = useLive((s) => s.oversight.items ?? EMPTY)
  const pending = useMemo(() => items.filter((i: any) => i.status === 'pending'), [items])
  const autonomy = useLive((s) => s.swarm.autonomy)
  const frozen = useLive((s) => s.swarm.frozen)
  const now = useSimNow(4)
  const ui = useUi()
  const [scope, setScope] = useState('a')
  const urgent = pending.some((i: any) => i.sla_deadline_t - now < 60)
  const connLabel = !ready ? 'Connecting' : conn === 'live' ? 'Live' : conn === 'lagging' ? 'Lagging' : 'Offline'

  return (
    <header className="flex h-12 shrink-0 items-center gap-4 border-b border-line bg-page px-4">
      <button type="button" onClick={() => navigate('/')} className="rounded-md" aria-label="Mission Control"><Wordmark /></button>
      <select
        value={scope}
        onChange={(e) => setScope(e.target.value)}
        aria-label="Operator context"
        className="h-7 rounded-md border border-line bg-s1 px-2 text-cap text-ink-1"
      >
        <option value="a">Operator A, home NOC</option>
        <option value="c">Consortium view</option>
      </select>
      <span title={`${rate} events/s over the WebSocket`}>
        <LiveDot state={!ready ? 'connecting' : conn} label={connLabel} />
      </span>
      <div className="flex items-baseline gap-1.5" title="Simulation clock (Saturday 31 Oct 2026, Pacific time)">
        <span className="tnum mono text-body text-ink-1">{simClockLabel(now)}</span>
        <span className="text-micro text-ink-3">PDT</span>
        {clock.paused && <span className="rounded bg-warn/15 px-1 text-micro font-medium text-warn">Paused</span>}
        {clock.speed !== 1 && <span className="tnum rounded bg-s3 px-1 text-micro text-ink-2">{clock.speed}×</span>}
      </div>

      <Gate flag="shell.palette">
        <button
          type="button"
          onClick={() => ui.set({ paletteOpen: true })}
          className="ml-2 flex h-8 w-[260px] items-center gap-2 rounded-md border border-line bg-s1 px-2.5 text-cap text-ink-3 hover:text-ink-2"
        >
          <Search size={14} />
          <span className="truncate">Search agents, records, actions</span>
          <span className="ml-auto flex gap-0.5"><Kbd>⌘</Kbd><Kbd>K</Kbd></span>
        </button>
      </Gate>

      <div className="ml-auto flex items-center gap-2">
        <button
          type="button"
          onClick={() => navigate('/oversight')}
          className={`flex h-8 items-center gap-1.5 rounded-md px-2 text-cap ${pending.length ? (urgent ? 'bg-warn/15 text-warn' : 'text-ink-1 hover:bg-s2') : 'text-ink-3 hover:bg-s2'}`}
          title={`${pending.length} decision${pending.length === 1 ? '' : 's'} waiting for a human`}
        >
          <Flag size={15} />
          <span className="tnum font-medium">{pending.length}</span>
          <span className="sr-only">pending approvals</span>
        </button>
        <span className="flex h-8 items-center gap-1.5 rounded-md border border-line px-2 text-cap" title="Autonomy level by agent class (TM Forum Autonomous Networks L0–L5)">
          <span className="text-ink-3">Autonomy</span>
          <span className="tnum font-medium text-ink-1">L{autonomy?.site ?? '–'}</span>
          <span className="text-ink-3">site</span>
          <span className="tnum font-medium text-ink-1">L{autonomy?.coordinator ?? '–'}</span>
          <span className="text-ink-3">coord.</span>
        </span>
        <span className={`rounded px-1.5 py-0.5 text-micro font-semibold ${mode === 'aws' ? 'bg-release/20 text-release' : 'bg-s3 text-ink-2'}`}
          title={mode === 'aws' ? 'Backed by AWS services' : 'Mock engine: simulated data, real cryptography'}>
          {mode === 'aws' ? 'AWS' : 'MOCK'}
        </span>
        <IconButton label={ui.theme === 'ops' ? 'Projector theme (T)' : 'Ops theme (T)'} onClick={() => ui.set({ theme: ui.theme === 'ops' ? 'projector' : 'ops' })}>
          {ui.theme === 'ops' ? <Sun size={16} /> : <Moon size={16} />}
        </IconButton>
        <Gate flag="shell.killswitch">
          {frozen ? (
            <button type="button" onClick={() => api.swarm('resume').then(() => toast({ severity: 'notice', title: 'Swarm resumed' }))}
              className="flex h-8 items-center gap-1.5 rounded-md bg-critical/20 px-2.5 text-cap font-medium text-critical-ink">
              <Snowflake size={14} /> Frozen: resume
            </button>
          ) : (
            <HoldButton label="Freeze swarm" onConfirm={() => api.swarm('freeze')}>
              <Snowflake size={14} /> Freeze
            </HoldButton>
          )}
        </Gate>
      </div>
    </header>
  )
}
