import { useState } from 'react'
import { Gate } from '@/core/flags'
import { PageHeader, Segmented } from '@/ui'
import { LogSearch } from './LogSearch'
import manifest from './manifest'
import { AgentHealth, GoldenSignals, Infra, LatencyPanel, SloPanel, TokensCost, TracePanel } from './panels'

const WINDOWS = [
  { value: 60, label: '1 min' },
  { value: 120, label: '2 min' },
  { value: 300, label: '5 min' },
]

export default function ObservabilityPage() {
  const [win, setWin] = useState(120)
  return (
    <div>
      <PageHeader title={manifest.title} description={manifest.description} tracks={manifest.tracks} />
      {/* one filter row scopes every time-based chart below */}
      <div className="mb-4 flex items-center gap-3">
        <span className="text-cap text-ink-3">Window</span>
        <Segmented value={win} onChange={setWin} options={WINDOWS} ariaLabel="Time window" />
        <span className="text-micro text-ink-3">live, 1 s resolution, sim time</span>
      </div>
      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-12"><GoldenSignals windowS={win} /></div>
        <div className="col-span-12"><TracePanel /></div>
        <div className="col-span-12 2xl:col-span-7"><TokensCost windowS={win} /></div>
        <Gate flag="obs.heatmap"><div className="col-span-12 2xl:col-span-5"><LatencyPanel windowS={win} /></div></Gate>
        <div className="col-span-12 xl:col-span-5"><SloPanel /></div>
        <div className="col-span-12 xl:col-span-7"><Infra windowS={win} /></div>
        <div className="col-span-12"><AgentHealth windowS={win} /></div>
        <div className="col-span-12"><LogSearch /></div>
      </div>
    </div>
  )
}
