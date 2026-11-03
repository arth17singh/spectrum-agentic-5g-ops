import { create } from 'zustand'
import { useEffect, useState } from 'react'
import { useLive } from './store'
import type { Rec, Severity } from './types'

export type InspectKind = 'ledger' | 'agent' | 'message' | 'policy' | 'decision' | 'checkpoint' | 'model' | 'approval' |
  'operator' | 'site' | 'trace' | 'raw'

export interface InspectTarget {
  kind: InspectKind
  id: string
  title?: string
  data?: Rec
}

export interface Toast {
  id: number
  severity: Severity
  title: string
  plane?: string
  action?: string
  route?: string
  ledgerSeq?: number
  at: number
}

interface UiState {
  theme: 'ops' | 'projector'
  motion: 'auto' | 'reduced'
  density: 'comfortable' | 'compact'
  railExpanded: boolean
  streamOpen: boolean
  paletteOpen: boolean
  helpOpen: boolean
  directorOpen: boolean
  notesOpen: boolean
  spotlight: boolean
  lit: string[]
  inspector: InspectTarget[]
  toasts: Toast[]
  set: (p: Partial<UiState>) => void
  inspect: (t: InspectTarget) => void
  closeInspector: () => void
  backInspector: () => void
}

const saved = (() => {
  try {
    return JSON.parse(localStorage.getItem('spectrum.ui') ?? '{}')
  } catch {
    return {}
  }
})()

export const useUi = create<UiState>((set, get) => ({
  theme: saved.theme ?? 'ops',
  motion: saved.motion ?? 'auto',
  density: saved.density ?? 'comfortable',
  railExpanded: false,
  streamOpen: false,
  paletteOpen: false,
  helpOpen: false,
  directorOpen: false, // closed on every load: it covers the screen; D opens it on purpose
  notesOpen: saved.notesOpen ?? true,
  spotlight: saved.spotlight ?? true,
  lit: [],
  inspector: [],
  toasts: [],
  set: (p) => {
    set(p)
    const s = get()
    localStorage.setItem('spectrum.ui', JSON.stringify({ theme: s.theme, motion: s.motion, density: s.density,
      directorOpen: s.directorOpen, notesOpen: s.notesOpen, spotlight: s.spotlight }))
  },
  inspect: (t) => set((s) => ({ inspector: [...s.inspector.slice(-9), t] })),
  closeInspector: () => set({ inspector: [] }),
  backInspector: () => set((s) => ({ inspector: s.inspector.slice(0, -1) })),
}))

let toastId = 0
export function toast(t: Omit<Toast, 'id' | 'at'>) {
  const id = ++toastId
  useUi.setState((s) => ({ toasts: [...s.toasts, { ...t, id, at: performance.now() }].slice(-3) }))
  setTimeout(() => useUi.setState((s) => ({ toasts: s.toasts.filter((x) => x.id !== id) })), 6000)
}

/** Simulation "now", interpolated locally between 1 Hz clock syncs. */
export function useSimNow(hz = 4): number {
  const clock = useLive((s) => s.clock)
  const [now, setNow] = useState(clock.t)
  useEffect(() => {
    const calc = () => (clock.paused ? clock.t : clock.t + ((performance.now() - clock.syncedAt) / 1000) * clock.speed)
    setNow(calc())
    const id = setInterval(() => setNow(calc()), 1000 / hz)
    return () => clearInterval(id)
  }, [clock, hz])
  return now
}

const SIM_EPOCH_MS = Date.parse('2026-10-31T21:30:00-07:00')
export function simClockLabel(t: number): string {
  const d = new Date(SIM_EPOCH_MS + t * 1000)
  // render in PDT (UTC-7) regardless of the viewer's timezone
  const p = new Date(d.getTime() - 7 * 3600 * 1000)
  return `${String(p.getUTCHours()).padStart(2, '0')}:${String(p.getUTCMinutes()).padStart(2, '0')}:${String(
    p.getUTCSeconds()).padStart(2, '0')}`
}
