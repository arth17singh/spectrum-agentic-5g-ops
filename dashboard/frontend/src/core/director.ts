import { useEffect, useRef } from 'react'
import { api } from './api'
import { useLive } from './store'
import { toast, useUi } from './ui-store'

// The Demo Director state lives on the server (so a presenter laptop and the projector stay in
// sync). Firing a beat: navigate locally for snappiness, then POST; every console follows the
// resulting `director.beat` event.

export interface Beat { n: number; title: string; route: string; spotlight: string[]; dur: number; note: string }

export function beats(): Beat[] {
  return (useLive.getState().director?.beats ?? []) as Beat[]
}

export async function fireBeat(n: number, navigate: (to: string) => void) {
  const list = beats()
  const b = list[n - 1]
  if (!b) return
  navigate(b.route)
  const ui = useUi.getState()
  useUi.setState({ lit: ui.spotlight ? b.spotlight : [] })
  try {
    await api.beat(n)
  } catch (e) {
    toast({ severity: 'critical', title: `Beat ${n} failed: ${(e as Error).message}` })
  }
}

export function nextBeat(navigate: (to: string) => void) {
  const cur = useLive.getState().director?.beat ?? 0
  if (cur < beats().length) fireBeat(cur + 1, navigate)
}

export function prevBeat(navigate: (to: string) => void) {
  const cur = useLive.getState().director?.beat ?? 0
  if (cur > 1) fireBeat(cur - 1, navigate)
}

/** Run `cb` once each time the director moves to beat `n` (screens wire their hero moments here). */
export function useOnBeat(n: number, cb: () => void) {
  const beat = useLive((s) => s.director?.beat)
  const started = useLive((s) => s.director?.started_at)
  const last = useRef<string | null>(null)
  const cbRef = useRef(cb)
  cbRef.current = cb
  useEffect(() => {
    if (beat === n && started && last.current !== started) {
      last.current = started
      cbRef.current()
    }
  }, [beat, started, n])
}
