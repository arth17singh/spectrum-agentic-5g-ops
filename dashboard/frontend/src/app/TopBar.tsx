import { Flag, Moon, Search, Snowflake, Sun } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useIsRealRoute } from './realRoute'
import { api } from '@/core/api'
import { Gate } from '@/core/flags'
import { EMPTY, useLive } from '@/core/store'
import { simClockLabel, toast, useSimNow, useUi } from '@/core/ui-store'
import { HoldButton, IconButton, Kbd, LiveDot } from '@/ui'

export function Wordmark({ short = false }: { short?: boolean }) {
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
      <span className={`text-[15px] font-semibold tracking-[0.04em] text-ink-1${short ? ' hidden sm:inline' : ''}`}>SPECTRUM</span>
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
  const real = useIsRealRoute() // measured-record screens: the simulator controls and clock are hidden
  const [scope, setScope] = useState('a')
  const urgent = pending.some((i: any) => i.sla_deadline_t - now < 60)
  const connLabel = !ready ? 'Connecting' : conn === 'live' ? 'Live' : conn === 'lagging' ? 'Lagging' : 'Offline'

  return (
    <header className="flex h-12 shrink-0 items-center gap-2 overflow-x-auto border-b border-line bg-page px-3 sm:gap-4 sm:px-4">
      <button type="button" onClick={() => navigate('/')} className="rounded-md" aria-label="Mission Control"><Wordmark short /></button>
      {!real && <select
        value={scope}
        onChange={(e) => setScope(e.target.value)}
        aria-label="Operator context"
        className="hidden h-7 rounded-md border border-line bg-s1 px-2 text-cap text-ink-1 sm:block"
      >
        <option value="a">Operator A, home NOC (simulated)</option>
        <option value="c">Consortium view (simulated)</option>
      </select>}
      <span title={`${rate} events/s over the WebSocket`}>
        <LiveDot state={!ready ? 'connecting' : conn} label={connLabel} />
      </span>
      {!real && <div className="hidden items-baseline gap-1.5 sm:flex" title="Simulation clock (Saturday 31 Oct 2026, Pacific time)">
        <span className="tnum mono text-body text-ink-1">{simClockLabel(now)}</span>
        <span className="text-micro text-ink-3">PDT</span>
        {clock.paused && <span className="rounded bg-warn/15 px-1 text-micro font-medium text-warn">Paused</span>}
        {clock.speed !== 1 && <span className="tnum rounded bg-s3 px-1 text-micro text-ink-2">{clock.speed}×</span>}
        <span className="text-micro font-semibold text-warn">SIM CLOCK</span>
      </div>}

      <Gate flag="shell.palette">
        <button
          type="button"
          onClick={() => ui.set({ paletteOpen: true })}
          className="flex h-8 w-8 shrink-0 items-center justify-center gap-2 rounded-md border border-line bg-s1 text-cap text-ink-3 hover:text-ink-2 sm:ml-2 sm:w-[260px] sm:justify-start sm:px-2.5"
        >
          <Search size={14} />
          <span className="hidden truncate sm:inline">Search agents, records, actions</span>
          <span className="ml-auto hidden gap-0.5 sm:flex"><Kbd>⌘</Kbd><Kbd>K</Kbd></span>
        </button>
      </Gate>

      <div className="ml-auto flex shrink-0 items-center gap-2">
        {!real && <button
          type="button"
          onClick={() => navigate('/oversight')}
          className={`flex h-8 items-center gap-1.5 rounded-md px-2 text-cap ${pending.length ? (urgent ? 'bg-warn/15 text-warn' : 'text-ink-1 hover:bg-s2') : 'text-ink-3 hover:bg-s2'}`}
          title={`${pending.length} decision${pending.length === 1 ? '' : 's'} waiting for a human`}
        >
          <Flag size={15} />
          <span className="tnum font-medium">{pending.length}</span>
          <span className="sr-only">pending approvals (simulated)</span>
        </button>}
        {!real && <span className="hidden h-8 items-center gap-1.5 rounded-md border border-line px-2 text-cap sm:flex" title="Autonomy level by agent class (TM Forum Autonomous Networks L0–L5)">
          <span className="text-ink-3">Autonomy</span>
          <span className="tnum font-medium text-ink-1">L{autonomy?.site ?? '–'}</span>
          <span className="text-ink-3">site</span>
          <span className="tnum font-medium text-ink-1">L{autonomy?.coordinator ?? '–'}</span>
          <span className="text-ink-3">coord.</span>
        </span>}
        {!real && <span className={`rounded px-1.5 py-0.5 text-micro font-semibold ${mode === 'aws' ? 'bg-release/20 text-release' : 'bg-warn/15 text-warn'}`}
          title={mode === 'aws' ? 'Simulator labelled as AWS mode: the AWS adapters are not called' : 'Mock engine: simulated data, real cryptography'}>
          {mode === 'aws' ? 'SIMULATOR (AWS label)' : 'SIMULATED'}
        </span>}
        <IconButton label={ui.theme === 'ops' ? 'Projector theme (T)' : 'Ops theme (T)'} onClick={() => ui.set({ theme: ui.theme === 'ops' ? 'projector' : 'ops' })}>
          {ui.theme === 'ops' ? <Sun size={16} /> : <Moon size={16} />}
        </IconButton>
        {!real && <Gate flag="shell.killswitch">
          {frozen ? (
            <button type="button" onClick={() => api.swarm('resume').then(() => toast({ severity: 'notice', title: 'Swarm resumed' }))}
              className="flex h-8 items-center gap-1.5 rounded-md bg-critical/20 px-2.5 text-cap font-medium text-critical-ink">
              <Snowflake size={14} /> <span className="hidden sm:inline">Frozen: </span>resume
            </button>
          ) : (
            <HoldButton label="Freeze swarm" onConfirm={() => api.swarm('freeze')}>
              <Snowflake size={14} /> <span className="hidden sm:inline">Freeze</span>
            </HoldButton>
          )}
        </Gate>}
      </div>
    </header>
  )
}
