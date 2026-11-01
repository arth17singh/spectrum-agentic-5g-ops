import { useCallback, useRef, useState } from 'react'
import { api } from '@/core/api'
import { recordHash } from '@/core/canonical'
import { useLive } from '@/core/store'
import type { LedgerRecord, Rec } from '@/core/types'
import type { BlockState } from '@/charts/ChainStrip'

// Chain verification, as an auditor would run it: the server walks every record (hash, link,
// Ed25519 signature, anchors) while this browser independently recomputes the SHA-256 of the
// blocks on screen. On a break, the strip refocuses on the broken record fetched fresh from the
// datastore, so the tampered payload is what gets hashed.

export type VerifyPhase = 'idle' | 'scanning' | 'ok' | 'broken'

export interface VerifyState {
  phase: VerifyPhase
  scan: number | null
  progress: number
  total: number
  states: Record<number, BlockState>
  result: Rec | null
  clientChecked: number
  clientMs: number
  /** records shown around the break, fetched from the server */
  focus: LedgerRecord[] | null
  brokenRecord: Rec | null
  clientRecomputed: string | null
}

const INITIAL: VerifyState = {
  phase: 'idle', scan: null, progress: 0, total: 0, states: {}, result: null, clientChecked: 0, clientMs: 0,
  focus: null, brokenRecord: null, clientRecomputed: null,
}

export function useVerify(visible: () => LedgerRecord[]) {
  const [s, setS] = useState<VerifyState>(INITIAL)
  const running = useRef(false)

  const reset = useCallback(() => setS(INITIAL), [])

  const run = useCallback(async () => {
    if (running.current) return
    running.current = true
    const recs = visible()
    const total = useLive.getState().ledger.height ?? recs.length
    setS({ ...INITIAL, phase: 'scanning', scan: 0, total, states: Object.fromEntries(recs.map((r) => [r.seq, 'pending' as BlockState])) })

    const server = api.verifyLedger() as Promise<Rec>
    const t0 = performance.now()
    const client = await Promise.all(recs.map(async (r) => ({ seq: r.seq, ok: (await recordHash(r)) === r.hash })))
    const clientMs = performance.now() - t0

    // sweep the cursor across the visible blocks (~1.6 s), ticking each one as it passes
    const D = 1600
    const start = performance.now()
    await new Promise<void>((resolve) => {
      const step = () => {
        const k = Math.min(1, (performance.now() - start) / D)
        const passed = Math.floor(k * recs.length)
        const states: Record<number, BlockState> = {}
        recs.forEach((r, i) => { states[r.seq] = i < passed ? (client[i].ok ? 'ok' : 'fail') : 'pending' })
        setS((p) => ({ ...p, scan: k, progress: Math.round(k * total * 0.97), states, clientChecked: passed, clientMs }))
        if (k < 1) requestAnimationFrame(step)
        else resolve()
      }
      requestAnimationFrame(step)
    })

    const result: Rec = await server.catch((e) => ({ ok: false, error: String(e), total: 0 }))
    if (result.ok) {
      setS((p) => ({ ...p, phase: 'ok', scan: null, progress: result.total, total: result.total, result,
        states: Object.fromEntries(recs.map((r) => [r.seq, 'ok' as BlockState])) }))
      running.current = false
      return
    }

    // broken: fetch a window around the bad record straight from the datastore
    const b = result.broken?.seq as number
    const lo = Math.max(1, b - 5)
    const hi = Math.min(result.total, b + 5)
    const fetched = await Promise.all(Array.from({ length: hi - lo + 1 }, (_, i) => api.record(lo + i).catch(() => null)))
    const focus = fetched.filter(Boolean).map((x: Rec) => x.record as LedgerRecord)
    const brokenRec = fetched.find((x: Rec | null) => x?.record.seq === b) ?? null
    const recomputed = brokenRec ? await recordHash(brokenRec.record) : null
    const states: Record<number, BlockState> = {}
    for (const r of focus) states[r.seq] = r.seq < b ? 'ok' : r.seq === b ? 'fail' : 'unverified'
    // replay a short cursor over the window, stopping at the break
    const idx = focus.findIndex((r) => r.seq === b)
    const stopAt = (idx + 0.5) / Math.max(1, focus.length)
    setS((p) => ({ ...p, focus, scan: 0, states: Object.fromEntries(focus.map((r) => [r.seq, 'pending' as BlockState])) }))
    const s2 = performance.now()
    await new Promise<void>((resolve) => {
      const step = () => {
        const k = Math.min(1, (performance.now() - s2) / 700)
        const pos = k * stopAt
        const passed = pos * focus.length
        const st: Record<number, BlockState> = {}
        focus.forEach((r, i) => { st[r.seq] = i < passed - 0.5 || (i === idx && k === 1) ? states[r.seq] : i > idx ? 'unverified' : 'pending' })
        setS((p) => ({ ...p, scan: pos, states: st }))
        if (k < 1) requestAnimationFrame(step)
        else resolve()
      }
      requestAnimationFrame(step)
    })
    setS((p) => ({ ...p, phase: 'broken', progress: b, total: result.total, result, states, brokenRecord: brokenRec, clientRecomputed: recomputed }))
    running.current = false
  }, [visible])

  return { ...s, run, reset }
}
