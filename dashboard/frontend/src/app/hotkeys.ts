import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '@/core/api'
import { fireBeat, nextBeat, prevBeat } from '@/core/director'
import { useLive } from '@/core/store'
import { toast, useUi } from '@/core/ui-store'
import { features } from '@/features'

// Global keyboard map (DASHBOARD_SPEC.md §2.6). Ignored while typing in inputs.

export function useHotkeys() {
  const navigate = useNavigate()
  useEffect(() => {
    let gPending = false
    let gTimer = 0
    let rTimer = 0
    const typing = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      return t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)
    }
    const down = (e: KeyboardEvent) => {
      const ui = useUi.getState()
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        ui.set({ paletteOpen: !ui.paletteOpen })
        return
      }
      if (typing(e) || e.metaKey || e.ctrlKey || e.altKey) return
      if (ui.paletteOpen) return
      const k = e.key
      if (gPending) {
        gPending = false
        clearTimeout(gTimer)
        const f = features.find((x) => x.nav.hotkey === k.toLowerCase())
        if (f) navigate(f.route)
        return
      }
      if (k === 'g') { gPending = true; gTimer = window.setTimeout(() => (gPending = false), 900); return }
      if (k === 'ArrowRight' || k === 'PageDown' || k === ' ') { e.preventDefault(); nextBeat(navigate); return }
      if (k === 'ArrowLeft' || k === 'PageUp') { e.preventDefault(); prevBeat(navigate); return }
      if (/^[1-9]$/.test(k)) { fireBeat(Number(k), navigate); return }
      if (k === 'Escape') { useUi.setState({ lit: [], helpOpen: false }); return }
      if (k === '?') { ui.set({ helpOpen: !ui.helpOpen }); return }
      switch (k.toLowerCase()) {
        case 'd': ui.set({ directorOpen: !ui.directorOpen }); break
        case 'n': ui.set({ notesOpen: !ui.notesOpen }); break
        case 's': ui.set({ spotlight: !ui.spotlight, lit: ui.spotlight ? [] : ui.lit }); break
        case 't': ui.set({ theme: ui.theme === 'ops' ? 'projector' : 'ops' }); break
        case 'l': ui.set({ streamOpen: !ui.streamOpen }); break
        case 'p': api.sim({ paused: !useLive.getState().clock.paused }); break
        case 'r':
          if (!e.repeat) rTimer = window.setTimeout(() => {
            api.reset().then(() => { useUi.setState({ lit: [] }); navigate('/'); toast({ severity: 'notice', title: 'Scenario reset to the pre-demo state' }) })
          }, 1000)
          break
      }
    }
    const up = (e: KeyboardEvent) => { if (e.key.toLowerCase() === 'r') clearTimeout(rTimer) }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => { window.removeEventListener('keydown', down); window.removeEventListener('keyup', up) }
  }, [navigate])
}
