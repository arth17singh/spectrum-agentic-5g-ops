import clsx from 'clsx'
import { Check, ChevronLeft, ChevronRight, CircleDashed, RotateCcw, Timer } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Wordmark } from '@/app/TopBar'
import { api } from '@/core/api'
import { fireBeat, type Beat } from '@/core/director'
import { EMPTY, useLive } from '@/core/store'
import { simClockLabel, useSimNow, useUi } from '@/core/ui-store'
import { startLive } from '@/core/ws'
import { Kbd, LiveDot } from '@/ui'

// Second-screen presenter view (/presenter). It drives the server-side director; the projector
// console follows each beat. Clicker friendly: PageDown / PageUp / arrows / space.

const TARGET = 7 * 60
const noop = () => {}

export default function PresenterPage() {
  const theme = useUi((s) => s.theme)
  useEffect(() => {
    startLive()
    document.documentElement.dataset.theme = theme
  }, [theme])
  const ready = useLive((s) => s.ready)
  const conn = useLive((s) => s.conn)
  const director = useLive((s) => s.director)
  const list: Beat[] = director?.beats ?? EMPTY
  const cur: number = director?.beat ?? 0
  const beat = list[cur - 1]
  const next = list[cur]
  const simNow = useSimNow(2)

  // talk timer: starts when beat 1 fires (or by hand), wall-clock based
  const [t0, setT0] = useState<number | null>(() => Number(localStorage.getItem('spectrum.presenter.t0')) || null)
  const [, tick] = useState(0)
  const lastStarted = useRef<string | null>(null)
  useEffect(() => {
    if (cur === 1 && director?.started_at && director.started_at !== lastStarted.current) {
      const t = Date.now()
      setT0(t)
      localStorage.setItem('spectrum.presenter.t0', String(t))
    }
    lastStarted.current = director?.started_at ?? null
  }, [cur, director?.started_at])
  useEffect(() => { const id = setInterval(() => tick((x) => x + 1), 500); return () => clearInterval(id) }, [])
  const elapsed = t0 ? (Date.now() - t0) / 1000 : 0

  const go = (n: number) => { if (n >= 1 && n <= list.length) fireBeat(n, noop) }
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return
      if (['ArrowRight', 'PageDown', ' '].includes(e.key)) { e.preventDefault(); go(cur + 1) }
      if (['ArrowLeft', 'PageUp'].includes(e.key)) { e.preventDefault(); go(cur - 1) }
      if (/^[1-9]$/.test(e.key)) go(Number(e.key))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  // planned time at the start of each beat (sum of duration hints) vs actual elapsed
  const planned = list.slice(0, Math.max(0, cur - 1)).reduce((s, b) => s + b.dur, 0)
  const drift = t0 && cur ? elapsed - planned : 0
  const timerColor = elapsed >= TARGET ? 'var(--critical-ink)' : elapsed >= 390 ? 'var(--warn)' : 'var(--ink-1)'

  return (
    <div className="flex h-full flex-col bg-page text-ink-1">
      <header className="flex h-12 shrink-0 items-center gap-4 overflow-x-auto border-b border-line px-5 whitespace-nowrap">
        <Wordmark />
        <span className="text-cap text-ink-3">Presenter</span>
        <LiveDot state={!ready ? 'connecting' : conn} label={!ready ? 'Connecting' : conn === 'live' ? 'Live' : conn} />
        <span className="tnum mono text-cap text-ink-2">{simClockLabel(simNow)} PDT</span>
        <div className="ml-auto flex items-center gap-3">
          <Timer size={16} className="text-ink-3" />
          <span className="tnum text-[22px] leading-7 font-semibold" style={{ color: timerColor }}>{fmt(elapsed)}</span>
          <span className="text-cap text-ink-3">of 7:00</span>
          {t0 && cur > 0 && (
            <span className={clsx('tnum rounded px-1.5 text-micro font-medium', drift > 15 ? 'bg-warn/15 text-warn' : 'bg-s3 text-ink-2')}>
              {drift >= 0 ? '+' : '−'}{fmt(Math.abs(drift))} vs plan
            </span>
          )}
          <button type="button" onClick={() => { const t = Date.now(); setT0(t); localStorage.setItem('spectrum.presenter.t0', String(t)) }}
            className="flex h-7 items-center gap-1 rounded-md border border-line px-2 text-micro text-ink-2 hover:text-ink-1"><RotateCcw size={12} /> Restart timer</button>
        </div>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 md:grid-cols-[1fr_360px]">
        <main className="flex min-h-0 min-w-0 flex-col overflow-auto p-4 sm:p-8">
          <div className="text-cap text-ink-3">{beat ? `Beat ${beat.n} of ${list.length}, on ${beat.route}` : 'Ready. Press → or the clicker to start.'}</div>
          <h1 className="mt-1 text-[40px] leading-[46px] font-semibold tracking-[-0.02em]">{beat ? beat.title : 'Cold open is next'}</h1>
          <div className="mt-6 max-w-[62ch] rounded-[12px] border border-line bg-s1 p-6">
            <div className="mb-2 text-cap text-ink-3">Say this</div>
            <p className="text-[22px] leading-[32px] text-ink-1">{beat ? beat.note : list[0]?.note ?? 'Waiting for the engine…'}</p>
            {beat && <div className="mt-4 text-cap text-ink-3">Suggested {beat.dur}s</div>}
          </div>
          {beat && (
            <div className="mt-4 flex max-w-[62ch] flex-wrap items-center gap-2 text-cap text-ink-2">
              <span className="text-ink-3">Projector shows</span>
              <span className="mono rounded border border-line px-1.5 text-ink-1">{beat.route}</span>
              <span className="text-ink-3">with the spotlight on</span>
              {beat.spotlight.map((sp) => <span key={sp} className="mono rounded bg-s2 px-1.5 text-ink-1">{sp}</span>)}
            </div>
          )}

          <div className="mt-auto pt-8">
            {next && (
              <div className="mb-4 text-body text-ink-2">
                <span className="text-ink-3">Next: </span><span className="font-medium text-ink-1">{next.title}</span>
                <span className="text-ink-3"> on {next.route}</span>
              </div>
            )}
            <div className="flex gap-3">
              <button type="button" onClick={() => go(cur - 1)} disabled={cur <= 1}
                className="flex h-16 w-24 shrink-0 items-center sm:w-40 justify-center gap-2 rounded-[12px] border border-line text-[18px] text-ink-2 hover:bg-s2 disabled:opacity-30">
                <ChevronLeft size={22} /> Back
              </button>
              <button type="button" onClick={() => go(cur + 1)} disabled={cur >= list.length}
                className="flex h-16 min-w-0 flex-1 items-center justify-center gap-2 rounded-[12px] bg-ink-1 text-[20px] font-semibold text-page hover:bg-ink-2 disabled:opacity-40">
                {next ? `Next: ${next.title}` : 'End of script'} <ChevronRight size={24} />
              </button>
            </div>
            <div className="mt-3 flex gap-3 text-micro text-ink-3">
              <span className="flex items-center gap-1"><Kbd>→</Kbd><Kbd>PageDown</Kbd> next</span>
              <span className="flex items-center gap-1"><Kbd>←</Kbd><Kbd>PageUp</Kbd> back</span>
              <span className="flex items-center gap-1"><Kbd>1–9</Kbd> jump</span>
            </div>
          </div>
        </main>

        <aside className="hidden min-h-0 flex-col border-l border-line md:flex">
          <ol className="min-h-0 flex-1 overflow-auto py-2">
            {list.map((b) => (
              <li key={b.n}>
                <button type="button" onClick={() => go(b.n)}
                  className={clsx('flex w-full items-start gap-3 px-4 py-2 text-left', b.n === cur ? 'bg-s2' : 'hover:bg-s1')}>
                  <span className="tnum mt-0.5 w-5 text-right text-cap text-ink-3">{b.n}</span>
                  <span className="mt-0.5">{b.n < cur ? <Check size={14} className="text-good-ink" /> : b.n === cur ? <span className="mt-1 block size-2 rounded-full bg-release" /> : <CircleDashed size={14} className="text-ink-3" />}</span>
                  <span className="min-w-0 flex-1">
                    <span className={clsx('block truncate text-body', b.n === cur ? 'font-medium text-ink-1' : 'text-ink-2')}>{b.title}</span>
                    <span className="block text-micro text-ink-3">{b.route}, {b.dur}s</span>
                  </span>
                </button>
              </li>
            ))}
          </ol>
          <Preflight />
        </aside>
      </div>
    </div>
  )
}

function Preflight() {
  const conn = useLive((s) => s.conn)
  const [pf, setPf] = useState<any>(null)
  useEffect(() => {
    const load = () => api.preflight().then(setPf).catch(() => setPf(null))
    load()
    const id = setInterval(load, 3000)
    return () => clearInterval(id)
  }, [])
  const rows: [string, boolean, string][] = [
    ['Backend', !!pf?.backend, pf ? 'reachable' : 'unreachable'],
    ['WebSocket', conn === 'live', conn],
    ['Ledger clean', !!pf?.ledger_clean, pf ? `${pf.ledger_height?.toLocaleString()} records` : '…'],
    ['Consoles connected', (pf?.ws_clients ?? 0) > 1, `${pf?.ws_clients ?? 0} clients`],
    ['Training', pf?.fl_state === 'ready' || pf?.fl_state === 'running', pf ? `${pf.fl_state}, round ${pf.round}` : '…'],
  ]
  return (
    <div className="border-t border-line px-4 py-3">
      <div className="mb-2 text-cap font-medium text-ink-2">Preflight</div>
      <ul className="space-y-1">
        {rows.map(([l, ok, d]) => (
          <li key={l} className="flex items-center gap-2 text-cap">
            {ok ? <Check size={13} className="text-good-ink" /> : <CircleDashed size={13} className="text-warn" />}
            <span>{l}</span>
            <span className="ml-auto truncate text-micro text-ink-3">{d}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function fmt(s: number) {
  const m = Math.floor(s / 60)
  return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`
}
