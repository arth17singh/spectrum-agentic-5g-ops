import clsx from 'clsx'
import { ArrowRight, Search, X } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '@/core/api'
import { fireBeat } from '@/core/director'
import type { Command } from '@/core/features'
import { isOn, useFlags } from '@/core/flags'
import { EMPTY, useLive } from '@/core/store'
import { toast, useUi } from '@/core/ui-store'
import { features } from '@/features'
import { Kbd } from '@/ui'
import { useIsRealRoute } from './realRoute'
import { SeverityIcon } from './StreamDrawer'

// ------------------------------------------------------------------ global commands

export const GLOBAL_COMMANDS: Command[] = [
  { id: 'fl.start', group: 'Federation', label: 'Start or resume training', run: () => api.fl('start') },
  { id: 'fl.pause', group: 'Federation', label: 'Pause training', run: () => api.fl('pause') },
  { id: 'fl.checkpoint', group: 'Federation', label: 'Save checkpoint now', run: () => api.fl('checkpoint') },
  { id: 'eval.run', group: 'Verification', label: 'Run Proving Ground on the latest checkpoint', run: () => api.evalRun() },
  { id: 'release.advance', group: 'Release', label: 'Advance rollout to the next stage', run: () => api.release('advance') },
  { id: 'release.rollback', group: 'Release', label: 'Roll back the candidate', run: () => api.release('rollback') },
  { id: 'twin.incident', group: 'Network', label: 'Inject incident: fibre cut during stadium let-out', run: () => api.inject('incident') },
  { id: 'twin.interference', group: 'Network', label: 'Inject fault: uplink interference at s-08', run: () => api.inject('interference', 's-08') },
  { id: 'attack.injection', group: 'Security', flag: 'demo.attacks', label: 'Attack: telemetry prompt injection on site-17', run: () => api.inject('telemetry_injection', 'site-17') },
  { id: 'ledger.verify', group: 'Audit', label: 'Verify the whole ledger chain', run: () => api.verifyLedger() },
  { id: 'ledger.tamper', group: 'Audit', flag: 'demo.tamper', label: 'Simulate tampering of a ledger record', run: () => api.tamper() },
  { id: 'ledger.restore', group: 'Audit', flag: 'demo.tamper', label: 'Restore tampered records', run: () => api.restoreLedger() },
  { id: 'swarm.freeze', group: 'Swarm', label: 'Freeze the swarm (kill switch)', run: () => api.swarm('freeze') },
  { id: 'swarm.resume', group: 'Swarm', label: 'Resume the swarm', run: () => api.swarm('resume') },
  { id: 'sim.pause', group: 'Simulation', label: 'Pause or resume the simulation', run: () => api.sim({ paused: !useLive.getState().clock.paused }) },
  { id: 'demo.reset', group: 'Demo', flag: 'demo.director', label: 'Reset scenario to the pre-demo state', run: () => api.reset() },
  { id: 'ui.theme', group: 'Display', label: 'Toggle Ops / Projector theme', run: () => useUi.getState().set({ theme: useUi.getState().theme === 'ops' ? 'projector' : 'ops' }) },
]

export function CommandPalette() {
  const open = useUi((s) => s.paletteOpen)
  const set = useUi((s) => s.set)
  const inspect = useUi((s) => s.inspect)
  const navigate = useNavigate()
  const flags = useFlags()
  const agents = useLive((s) => s.swarm.agents ?? EMPTY)
  const beats = useLive((s) => s.director?.beats ?? EMPTY)
  const [q, setQ] = useState('')
  const [sel, setSel] = useState(0)
  const input = useRef<HTMLInputElement>(null)
  useEffect(() => { if (open) { setQ(''); setSel(0); setTimeout(() => input.current?.focus(), 10) } }, [open])

  const items = useMemo(() => {
    const out: { id: string; group: string; label: string; hint?: string; run: () => void }[] = []
    const qq = q.trim().toLowerCase()
    const m = qq.match(/^#?(\d{1,6})$/)
    if (m) out.push({ id: 'rec', group: 'Ledger', label: `Open ledger record #${m[1]}`, run: () => inspect({ kind: 'ledger', id: m[1] }) })
    for (const f of features) {
      if (f.bare || !isOn(f.id, flags)) continue
      out.push({ id: f.id, group: 'Go to', label: f.nav.label, hint: f.nav.hotkey ? `g ${f.nav.hotkey}` : undefined, run: () => navigate(f.route) })
    }
    for (const c of [...GLOBAL_COMMANDS, ...features.flatMap((f) => f.commands ?? [])]) {
      if (c.flag && !isOn(c.flag, flags)) continue
      out.push({ id: c.id, group: c.group, label: c.label, run: () => {
        Promise.resolve(c.run({ navigate })).then(() => toast({ severity: 'notice', title: c.label })).catch((e) => toast({ severity: 'critical', title: `${c.label} failed: ${e.message}` }))
      } })
    }
    for (const b of beats) out.push({ id: `beat-${b.n}`, group: 'Demo beats', label: `Beat ${b.n}: ${b.title}`, run: () => fireBeat(b.n, navigate) })
    if (qq.length >= 2) {
      for (const a of agents) if (a.id.includes(qq)) out.push({ id: `agent-${a.id}`, group: 'Agents', label: `${a.id} (${a.class}, ${a.model})`, run: () => inspect({ kind: 'agent', id: a.id }) })
    }
    if (!qq) return out.filter((i) => i.group !== 'Agents').slice(0, 60)
    const words = qq.split(/\s+/)
    return out.filter((i) => i.id === 'rec' || words.every((w) => `${i.group} ${i.label}`.toLowerCase().includes(w))).slice(0, 60)
  }, [q, flags, agents, beats, navigate, inspect])

  if (!open) return null
  const run = (i: number) => { const it = items[i]; if (!it) return; set({ paletteOpen: false }); it.run() }
  return (
    <div className="fixed inset-0 z-50 grid place-items-start justify-center pt-[12vh]" style={{ background: 'var(--scrim)' }} onClick={() => set({ paletteOpen: false })}>
      <div role="dialog" aria-label="Command palette" className="w-[620px] overflow-hidden rounded-[10px] border border-line bg-s2" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-2 border-b border-line px-3">
          <Search size={16} className="text-ink-3" />
          <input
            ref={input}
            value={q}
            onChange={(e) => { setQ(e.target.value); setSel(0) }}
            onKeyDown={(e) => {
              if (e.key === 'ArrowDown') { e.preventDefault(); setSel((s) => Math.min(items.length - 1, s + 1)) }
              if (e.key === 'ArrowUp') { e.preventDefault(); setSel((s) => Math.max(0, s - 1)) }
              if (e.key === 'Enter') run(sel)
              if (e.key === 'Escape') set({ paletteOpen: false })
            }}
            placeholder="Jump to a screen, run an action, open #record or an agent"
            className="h-12 flex-1 bg-transparent text-body text-ink-1 outline-none placeholder:text-ink-3"
          />
          <Kbd>Esc</Kbd>
        </div>
        <ul className="max-h-[52vh] overflow-auto py-1">
          {items.map((it, i) => (
            <li key={it.id + i}>
              <button type="button" onMouseEnter={() => setSel(i)} onClick={() => run(i)}
                className={clsx('flex w-full items-center gap-3 px-3 py-2 text-left text-body', i === sel ? 'bg-s3 text-ink-1' : 'text-ink-2')}>
                <span className="w-24 shrink-0 text-micro text-ink-3">{it.group}</span>
                <span className="truncate">{it.label}</span>
                {it.hint && <span className="ml-auto text-micro text-ink-3">{it.hint}</span>}
                {i === sel && !it.hint && <ArrowRight size={14} className="ml-auto text-ink-3" />}
              </button>
            </li>
          ))}
          {!items.length && <li className="px-3 py-6 text-center text-cap text-ink-3">No matches. Try a screen name, an agent like site-17, or #1234.</li>}
        </ul>
      </div>
    </div>
  )
}

export function Toasts() {
  const toasts = useUi((s) => s.toasts)
  const navigate = useNavigate()
  const inspect = useUi((s) => s.inspect)
  const real = useIsRealRoute()
  // A toast made from a stream event carries a plane and comes from the simulator, except the evidence ones.
  const fromSim = (t: (typeof toasts)[number]) => !!t.plane && t.route !== '/evidence'
  return (
    <div className="pointer-events-none fixed right-4 bottom-12 z-50 flex w-[380px] max-w-[calc(100vw-2rem)] flex-col-reverse gap-2">
      {toasts.filter((t) => !(real && fromSim(t))).map((t) => (
        <div key={t.id} role="status" className={clsx('pointer-events-auto row-in flex items-start gap-2.5 rounded-md border bg-s2 px-3 py-2.5 shadow-[0_8px_24px_rgba(0,0,0,0.3)]',
          t.severity === 'critical' ? 'border-critical/60 pulse-twice' : 'border-line')}>
          <div className="mt-0.5"><SeverityIcon s={t.severity} /></div>
          <div className="min-w-0 flex-1 text-cap text-ink-1">
            {fromSim(t) && <span className="mr-1.5 rounded bg-warn/15 px-1 text-micro font-semibold text-warn">SIMULATED</span>}
            {t.title}
          </div>
          {t.action && (
            <button type="button" className="shrink-0 text-cap font-medium text-ink-1 underline-offset-2 hover:underline"
              onClick={() => { if (t.route) navigate(t.route); else if (t.ledgerSeq) inspect({ kind: 'ledger', id: String(t.ledgerSeq) }) }}>
              {t.action}
            </button>
          )}
          <button type="button" aria-label="Dismiss" className="shrink-0 text-ink-3 hover:text-ink-1" onClick={() => useUi.setState((s) => ({ toasts: s.toasts.filter((x) => x.id !== t.id) }))}><X size={13} /></button>
        </div>
      ))}
    </div>
  )
}

const KEYS: [string, string][] = [
  ['→  PageDown  Space', 'Next demo beat (presentation clickers send PageDown)'],
  ['←  PageUp', 'Previous beat'],
  ['1 – 9', 'Jump to beat'],
  ['D', 'Show or hide the Demo Director'],
  ['N', 'Presenter notes'],
  ['S', 'Spotlight on or off'],
  ['Hold R', 'Reset the scenario'],
  ['P', 'Pause or resume the simulation'],
  ['T', 'Ops / Projector theme'],
  ['L', 'Live stream drawer'],
  ['⌘K', 'Command palette'],
  ['g then m f p r s o t a b ,', 'Go to a screen'],
  ['Esc', 'Close overlay, clear spotlight'],
  ['?', 'This help'],
]

export function HelpOverlay() {
  const open = useUi((s) => s.helpOpen)
  const set = useUi((s) => s.set)
  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 grid place-items-center" style={{ background: 'var(--scrim)' }} onClick={() => set({ helpOpen: false })}>
      <div role="dialog" aria-label="Keyboard shortcuts" className="w-[560px] rounded-[10px] border border-line bg-s2 p-5" onClick={(e) => e.stopPropagation()}>
        <h2 className="mb-3 text-section font-semibold">Keyboard shortcuts</h2>
        <dl className="grid grid-cols-[200px_1fr] gap-y-2 text-body">
          {KEYS.map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="flex flex-wrap gap-1">{k.split('  ').map((x) => <Kbd key={x}>{x}</Kbd>)}</dt>
              <dd className="text-ink-2">{v}</dd>
            </div>
          ))}
        </dl>
      </div>
    </div>
  )
}
