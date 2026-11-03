import { useCallback, useEffect, useState } from 'react'
import { get } from '@/core/api'
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
  return (m ? m[1] : r || 'superseded').toLowerCase()
}
