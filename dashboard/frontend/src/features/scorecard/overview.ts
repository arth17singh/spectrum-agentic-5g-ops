import { useCallback, useEffect, useState } from 'react'
import { get } from '@/core/api'
import type { Rec } from '@/core/types'
import type { Overview } from './RealRuns'

/** GET /api/realrun/overview, refreshed every 15 s. Shared by Scorecard, Evidence and Twin View. */
export function useOverview() {
  const [ov, setOv] = useState<Overview | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const load = useCallback(() => get<Overview>('/api/realrun/overview').then((o) => { setOv(o); setErr(null) }).catch((e) => setErr(String(e))), [])
  useEffect(() => { load(); const t = setInterval(load, 15_000); return () => clearInterval(t) }, [load])
  return { ov, err }
}

/** "Harness defect: the parser ..." -> "harness defect" */
export const shortReason = (r?: string) => {
  const m = /^([^:.]{3,60})[:.]/.exec(r ?? '')
  if (m) return m[1].toLowerCase()
  // no short lead-in: cut the reason at a word near 60 characters rather than repeating all of it
  const t = (r ?? '').trim()
  if (!t) return 'superseded'
  return (t.length <= 60 ? t : `${t.slice(0, 60).replace(/\s+\S*$/, '')}…`).toLowerCase()
}

/** The FL session facts a t2.fl_convergence run records in details.session, or null. */
export function sessionFacts(c: Rec | null | undefined): Rec | null {
  const s = c?.details?.session
  return s && typeof s === 'object' ? s : null
}
