import { FlaskConical, Play } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { api } from '@/core/api'
import { Gate as FlagGate } from '@/core/flags'
import { useLive } from '@/core/store'
import { toast } from '@/core/ui-store'
import { Button, PageHeader } from '@/ui'
import manifest from './manifest'
import { versions, type ModelRow } from './model'
import { Pareto, Parallel, Pipeline, RedTeam, Scorecard, TaskHeatmap } from './panels'
import { AiBom, FedEval, Gate, Latency, Scaling } from './panels2'

export default function ProvingPage() {
  const p = useLive((s) => s.proving)
  const models: ModelRow[] = p.models ?? []
  const vs = useMemo(() => versions(models), [models])
  const ours = useMemo(() => models.filter((m) => m.family === 'ours'), [models])
  const active = p.active
  const running = active?.state === 'running'
  const [sel, setSel] = useState<string | null>(null)
  // follow the run / newest candidate unless the user picked something
  const [pinned, setPinned] = useState(false)
  const auto = active?.model ?? p.candidate ?? vs[vs.length - 1]?.id ?? p.production
  useEffect(() => { if (!pinned) setSel(auto) }, [auto, pinned])
  const m = models.find((x) => x.id === sel) ?? models.find((x) => x.id === auto)
  const prod = models.find((x) => x.id === p.production) ?? models[0]
  if (!m || !prod) return <PageHeader title={manifest.title} description={manifest.description} tracks={manifest.tracks} />
  // display order: our versions newest first, then other federated sizes, then baselines
  const ordered = [...vs].reverse().concat(ours.filter((x) => !vs.includes(x)), models.filter((x) => x.family === 'baseline'))
  const fed = p.fed_evals?.[m.id] ?? []
  return (
    <div>
      <PageHeader
        title={manifest.title}
        description={manifest.description}
        tracks={manifest.tracks}
        actions={
          <>
            <label className="flex items-center gap-2 text-cap text-ink-3">
              Candidate
              <select value={m.id} onChange={(e) => { setPinned(true); setSel(e.target.value) }} aria-label="Candidate model"
                className="h-8 rounded-md border border-line bg-s1 px-2 text-cap text-ink-1">
                {ordered.map((x) => <option key={x.id} value={x.id}>{x.id}{x.id === p.production ? ' (production)' : x.pending ? ' (pending)' : x.family === 'baseline' ? ' (baseline)' : ''}</option>)}
              </select>
            </label>
            <Button icon={<FlaskConical size={14} />} disabled={running}
              onClick={() => api.evalRun().then(() => { setPinned(false); toast({ severity: 'notice', title: 'Evaluating the latest checkpoint' }) })}>
              Evaluate latest checkpoint
            </Button>
            <Button variant="primary" icon={<Play size={14} />} disabled={running || m.family !== 'ours'}
              title={m.family !== 'ours' ? 'Baselines are third-party models; pick one of ours to evaluate' : undefined}
              onClick={() => api.evalRun(m.id).then(() => toast({ severity: 'notice', title: `Evaluating ${m.id}` }))}>
              {running ? 'Evaluation running' : `Run evaluation on ${m.id}`}
            </Button>
          </>
        }
      />
      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-12"><Pipeline /></div>
        <div className="col-span-12"><Scorecard m={m} prod={prod} models={models} gate={p.gate} /></div>
        <div className="col-span-12 xl:col-span-5"><Pareto models={models} sel={m.id} prod={prod.id} /></div>
        <FlagGate flag="eval.parallel"><div className="col-span-12 xl:col-span-7"><Parallel models={models} sel={m.id} prod={prod.id} /></div></FlagGate>
        <div className="col-span-12 xl:col-span-6"><TaskHeatmap models={ordered} tasks={p.tasks ?? []} /></div>
        <div className="col-span-12 xl:col-span-6"><RedTeam models={ordered} attacks={p.attacks ?? []} sel={m.id} /></div>
        <div className="col-span-12 xl:col-span-7"><Latency m={m} prod={prod} models={models} /></div>
        <FlagGate flag="eval.fed_eval"><div className="col-span-12 xl:col-span-5"><FedEval m={m} rows={fed} /></div></FlagGate>
        <FlagGate flag="eval.scaling"><div className="col-span-12 xl:col-span-5"><Scaling models={models} sel={m.id} /></div></FlagGate>
        <div className="col-span-12 xl:col-span-7"><Gate m={m} gate={p.gate} run={active} /></div>
        <div className="col-span-12"><AiBom m={m} bom={p.aiboms?.[m.id]} /></div>
      </div>
    </div>
  )
}
