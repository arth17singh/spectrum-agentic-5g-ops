import clsx from 'clsx'
import { Check, ChevronLeft, ChevronRight, CircleDashed, GripHorizontal, Pause, Play, RotateCcw, X } from 'lucide-react'
import QRCode from 'qrcode'
import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '@/core/api'
import { beats, fireBeat, nextBeat, prevBeat, type Beat } from '@/core/director'
import { useFlag } from '@/core/flags'
import { useLive } from '@/core/store'
import { toast, useUi } from '@/core/ui-store'
import { HoldButton, Kbd, Segmented, Toggle } from '@/ui'

// Demo Director: a floating, draggable panel. Every beat = navigate + backend scenario action +
// spotlight + presenter note. Clicker-friendly (PageDown/PageUp), resettable, preflight-checked.

export function Director() {
  const on = useFlag('demo.director')
  const open = useUi((s) => s.directorOpen)
  const notes = useUi((s) => s.notesOpen)
  const spotlight = useUi((s) => s.spotlight)
  const set = useUi((s) => s.set)
  const navigate = useNavigate()
  const director = useLive((s) => s.director)
  const clock = useLive((s) => s.clock)
  const [tab, setTab] = useState<'beats' | 'preflight'>('beats')
  const [pos, setPos] = useState<{ x: number; y: number }>(() => ({ x: 80, y: window.innerHeight - 560 }))
  const drag = useRef<{ dx: number; dy: number } | null>(null)
  const list: Beat[] = director?.beats ?? []
  const cur = director?.beat ?? 0
  const beat = list[cur - 1]
  const next = list[cur]

  if (!on || !open) return null
  return (
    <div
      role="dialog"
      aria-label="Demo Director"
      className="fixed z-40 w-[340px] overflow-hidden rounded-[10px] border border-line bg-s2 shadow-[0_16px_40px_rgba(0,0,0,0.35)]"
      style={{ left: pos.x, top: Math.max(56, pos.y) }}
    >
      <div
        className="flex h-9 cursor-grab items-center gap-2 border-b border-line px-3 active:cursor-grabbing"
        onPointerDown={(e) => {
          drag.current = { dx: e.clientX - pos.x, dy: e.clientY - pos.y }
          ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
        }}
        onPointerMove={(e) => drag.current && setPos({ x: e.clientX - drag.current.dx, y: e.clientY - drag.current.dy })}
        onPointerUp={() => (drag.current = null)}
      >
        <GripHorizontal size={14} className="text-ink-3" />
        <span className="text-cap font-semibold text-ink-1">Demo Director</span>
        <span className="tnum text-micro text-ink-3">beat {cur} of {list.length}</span>
        <button type="button" aria-label="Hide director (D)" onClick={() => set({ directorOpen: false })} className="ml-auto text-ink-3 hover:text-ink-1"><X size={14} /></button>
      </div>

      {notes && beat && (
        <div className="border-b border-line bg-s1 px-3 py-2.5">
          <div className="text-micro text-ink-3">Say this</div>
          <p className="mt-0.5 text-body text-ink-1">{beat.note}</p>
        </div>
      )}

      <div className="flex items-center gap-2 px-3 py-2.5">
        <button type="button" onClick={() => prevBeat(navigate)} disabled={cur <= 1} aria-label="Previous beat"
          className="grid size-9 place-items-center rounded-md border border-line text-ink-2 hover:bg-s3 disabled:opacity-30"><ChevronLeft size={16} /></button>
        <button type="button" onClick={() => nextBeat(navigate)} disabled={cur >= list.length}
          className="flex h-9 flex-1 items-center justify-between rounded-md bg-ink-1 px-3 text-body font-medium text-page hover:bg-ink-2 disabled:opacity-40">
          <span className="truncate">{next ? `Next: ${next.title}` : 'End of script'}</span>
          <ChevronRight size={16} />
        </button>
        <button type="button" aria-label={clock.paused ? 'Resume simulation (P)' : 'Pause simulation (P)'} onClick={() => api.sim({ paused: !clock.paused })}
          className="grid size-9 place-items-center rounded-md border border-line text-ink-2 hover:bg-s3">{clock.paused ? <Play size={15} /> : <Pause size={15} />}</button>
      </div>

      <div className="flex items-center gap-2 border-t border-line px-3 py-2">
        <Segmented size="sm" value={tab} onChange={setTab} options={[{ value: 'beats', label: 'Script' }, { value: 'preflight', label: 'Preflight' }]} />
        <Segmented size="sm" value={clock.speed} onChange={(v) => api.sim({ speed: v })} ariaLabel="Simulation speed"
          options={[{ value: 1, label: '1×' }, { value: 2, label: '2×' }, { value: 5, label: '5×' }]} />
        <HoldButton label="Reset scenario" className="ml-auto h-7" onConfirm={() => {
          api.reset().then(() => { useUi.setState({ lit: [] }); navigate('/'); toast({ severity: 'notice', title: 'Scenario reset to the pre-demo state' }) })
        }}>
          <RotateCcw size={13} /> Reset
        </HoldButton>
      </div>

      {tab === 'beats' ? (
        <ol className="max-h-[230px] overflow-auto border-t border-line py-1">
          {list.map((b) => (
            <li key={b.n}>
              <button type="button" onClick={() => fireBeat(b.n, navigate)}
                className={clsx('flex w-full items-center gap-2 px-3 py-1.5 text-left text-cap', b.n === cur ? 'bg-s3 text-ink-1' : 'text-ink-2 hover:bg-s3/60')}>
                <span className="tnum w-5 text-right text-ink-3">{b.n}</span>
                {b.n < cur ? <Check size={13} className="text-good-ink" /> : b.n === cur ? <span className="size-2 rounded-full bg-release" /> : <CircleDashed size={13} className="text-ink-3" />}
                <span className="truncate">{b.title}</span>
                <span className="ml-auto shrink-0 text-micro text-ink-3">{b.route}</span>
              </button>
            </li>
          ))}
        </ol>
      ) : (
        <Preflight />
      )}

      <div className="flex items-center gap-3 border-t border-line px-3 py-2 text-micro text-ink-2">
        <label className="flex items-center gap-1.5"><Toggle checked={spotlight} onChange={(v) => set({ spotlight: v, lit: v ? useUi.getState().lit : [] })} label="Spotlight" /> Spotlight</label>
        <label className="flex items-center gap-1.5"><Toggle checked={notes} onChange={(v) => set({ notesOpen: v })} label="Notes" /> Notes</label>
        <span className="ml-auto flex items-center gap-1"><Kbd>→</Kbd> next</span>
      </div>
    </div>
  )
}

function Preflight() {
  const conn = useLive((s) => s.conn)
  const [pf, setPf] = useState<any>(null)
  const [qr, setQr] = useState<string>('')
  const fonts = typeof document !== 'undefined' && document.fonts?.check('13px "IBM Plex Sans"')
  const theme = useUi((s) => s.theme)
  useEffect(() => {
    const load = () => api.preflight().then(setPf).catch(() => setPf(null))
    load()
    const id = setInterval(load, 10000)
    return () => clearInterval(id)
  }, [])
  const url = `${location.protocol}//${location.hostname === 'localhost' ? (pf?.lan_ip ?? location.hostname) : location.hostname}:${location.port}/m/approve`
  useEffect(() => { QRCode.toDataURL(url, { margin: 1, width: 120 }).then(setQr) }, [url])
  const rows: [string, boolean, string][] = [
    ['Backend reachable', !!pf?.backend, pf ? 'ok' : 'start the API: make dev'],
    ['WebSocket live', conn === 'live', conn],
    ['Ledger verified clean', !!pf?.ledger_clean, pf ? `${pf.ledger_height?.toLocaleString()} records` : '…'],
    ['Scenario reset (beat 0)', pf?.beat === 0, pf ? `beat ${pf.beat}, FL ${pf.fl_state}` : '…'],
    ['Fonts loaded', !!fonts, fonts ? 'IBM Plex' : 'fallback font in use'],
    ['Theme chosen', true, theme === 'projector' ? 'Projector (light)' : 'Ops (dark)'],
    ['Approval phones connected', (pf?.ws_clients ?? 0) > 1, `${Math.max(0, (pf?.ws_clients ?? 1) - 1)} other client(s)`],
  ]
  return (
    <div className="max-h-[300px] overflow-auto border-t border-line px-3 py-2">
      <ul className="space-y-1">
        {rows.map(([label, ok, detail]) => (
          <li key={label} className="flex items-center gap-2 text-cap">
            {ok ? <Check size={13} className="text-good-ink" /> : <CircleDashed size={13} className="text-warn" />}
            <span className="text-ink-1">{label}</span>
            <span className="ml-auto truncate text-micro text-ink-3">{detail}</span>
          </li>
        ))}
      </ul>
      <div className="mt-3 flex items-center gap-3 rounded-md border border-line bg-s1 p-2">
        {qr && <img src={qr} alt="QR code for the mobile approval page" width={84} height={84} className="rounded" />}
        <div className="min-w-0 text-micro text-ink-2">
          <div className="text-cap font-medium text-ink-1">Phone approvals</div>
          Scan to open the mobile approval page on the same Wi-Fi.
          <div className="mono mt-1 truncate text-ink-3">{url}</div>
        </div>
      </div>
    </div>
  )
}

export { beats }
