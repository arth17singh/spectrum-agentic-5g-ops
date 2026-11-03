import clsx from 'clsx'
import { RefreshCw } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { get } from '@/core/api'
import type { Rec } from '@/core/types'
import { Button, PageHeader, Segmented } from '@/ui'
import { CellsView } from './CellsView'
import { ConsortiumView } from './ConsortiumView'
import { ViaviSites } from './ViaviSites'
import manifest from './manifest'
import type { Load } from './format'
import { Banner, ProvenanceChip } from './shared'

const POLL_MS = 10_000
type Mode = 'cells' | 'consortium' | 'viavi'

export default function TwinViewPage() {
  const [params, setParams] = useSearchParams()
  const m0 = params.get('mode')
  const mode: Mode = m0 === 'consortium' || m0 === 'viavi' ? m0 : 'cells'
  const setMode = (m: Mode) => setParams(m === 'cells' ? {} : { mode: m }, { replace: true })
  const [cases, setCases] = useState<Load<Rec>>({ state: 'loading' })
  const [cons, setCons] = useState<Load<Rec>>({ state: 'loading' })
  const [busy, setBusy] = useState(false)

  const fetchAll = useCallback(() => Promise.all([
    get<Rec>('/api/twinview/cases').then((data) => setCases({ state: 'ok', data })).catch((e) => setCases({ state: 'error', message: String(e?.message ?? e) })),
    get<Rec>('/api/twinview/consortium').then((data) => setCons({ state: 'ok', data })).catch((e) => setCons({ state: 'error', message: String(e?.message ?? e) })),
  ]), [])
  const refresh = () => { setBusy(true); fetchAll().finally(() => setBusy(false)) }

  useEffect(() => {
    fetchAll()
    const t = setInterval(fetchAll, POLL_MS)
    return () => clearInterval(t)
  }, [fetchAll])

  const cur = mode === 'cells' ? cases : cons
  const prov = cases.state === 'ok' && (cases.data.cases ?? []).length > 0 ? cases.data.provenance : null

  return (
    <div className="min-w-0" data-slice="twinview">
      <PageHeader title={manifest.title} tracks={manifest.tracks} description={manifest.description}
        actions={<>
          <Segmented ariaLabel="Mode" value={mode} onChange={setMode}
            options={[{ value: 'cells', label: 'Cells', title: 'One case: the 4 cells window by window' }, { value: 'consortium', label: 'Consortium', title: 'The operator silos behind the shared response model' }, { value: 'viavi', label: 'Viavi sites', title: 'Viavi RSG scenarios and their sites, for context' }]} />
          {mode === 'cells' && prov && <ProvenanceChip p={prov} />}
          <Button icon={<RefreshCw size={14} className={clsx(busy && 'animate-spin')} />} disabled={busy} onClick={refresh}>Refresh</Button>
        </>} />
      {mode === 'cells' && prov && <Banner p={prov} extra="Per-cell coverage and SLA shares are a proposed field (twin_state) the tools do not write yet." />}
      {mode === 'viavi' && <ViaviSites />}
      {mode !== 'viavi' && cur.state === 'loading' && <div className="text-cap text-ink-3">Loading…</div>}
      {mode !== 'viavi' && cur.state === 'error' && (
        <div className="rounded-md border border-critical/60 bg-critical/10 p-4 text-cap text-ink-1" role="alert">
          <span className="font-medium text-critical-ink">The records could not be read.</span> {cur.message}
          <div className="mt-1 text-ink-3">This is an error, not an empty result: nothing has been assumed.</div>
        </div>
      )}
      {mode !== 'viavi' && cur.state === 'ok' && (mode === 'cells' ? <CellsView cases={cur.data} /> : <ConsortiumView data={cur.data} />)}
    </div>
  )
}
