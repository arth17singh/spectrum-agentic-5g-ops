import { Bug, Pause, Play, Unplug, Zap, Siren } from 'lucide-react'
import { api } from '@/core/api'
import { Gate, useFlag } from '@/core/flags'
import { useLive } from '@/core/store'
import { toast } from '@/core/ui-store'
import { Button, PageHeader } from '@/ui'
import manifest from './manifest'
import { IncidentCard, KpiStrip, NegotiationPanel, PlanPanel, SwarmPanel, TwinPanel } from './panels'
import { AutonomyPanel, Breakers, MessageStream, Roster } from './panels2'

export default function SwarmPage() {
  const attacks = useFlag('demo.attacks')
  const paused = useLive((s) => s.swarm.paused)
  const incident = useLive((s) => s.swarm.incident)
  const active = !!incident && incident.status !== 'resolved'
  const run = (p: Promise<unknown>, msg: string) =>
    p.then(() => toast({ severity: 'notice', title: msg })).catch((e) => toast({ severity: 'critical', title: e.message }))
  return (
    <div>
      <PageHeader
        title={manifest.title}
        description={manifest.description}
        tracks={manifest.tracks}
        actions={
          <>
            <Button variant="primary" icon={<Unplug size={14} />} disabled={active} onClick={() => run(api.inject('incident'), 'Injected: fibre cut T3 during stadium let-out')}>
              Fibre cut + stadium surge
            </Button>
            <Button icon={<Zap size={14} />} onClick={() => run(api.inject('interference', 's-08'), 'Injected: uplink interference at s-08')}>Interference</Button>
            <Button icon={<Siren size={14} />} onClick={() => run(api.inject('power_outage', 's-33'), 'Injected: mains power lost at s-33')}>Power outage</Button>
            {attacks && (
              <Button variant="danger" icon={<Bug size={14} />} onClick={() => run(api.inject('telemetry_injection', 'site-17'), 'Attack: poisoned syslog planted at s-17')}>
                Telemetry injection
              </Button>
            )}
            <Button icon={paused ? <Play size={14} /> : <Pause size={14} />} onClick={() => api.swarm('pause').then((r: any) => {
              useLive.setState((s) => ({ swarm: { ...s.swarm, paused: r.paused } }))
              toast({ severity: 'notice', title: r.paused ? 'Swarm paused: no new routine actions' : 'Swarm resumed' })
            })}>
              {paused ? 'Resume swarm' : 'Pause swarm'}
            </Button>
          </>
        }
      />
      <KpiStrip />
      <div className="mt-4 grid grid-cols-12 gap-4">
        <div className="col-span-12 xl:col-span-5"><TwinPanel /></div>
        <div className="col-span-12 md:col-span-7 xl:col-span-4"><SwarmPanel /></div>
        <div className="col-span-12 md:col-span-5 xl:col-span-3"><IncidentCard /></div>
        <Gate flag="swarm.negotiation"><div className="col-span-12 xl:col-span-5"><NegotiationPanel /></div></Gate>
        <div className="col-span-12 xl:col-span-7"><PlanPanel /></div>
        <div className="col-span-12 xl:col-span-8"><MessageStream /></div>
        <div className="col-span-12 xl:col-span-4"><Breakers /></div>
        <div className="col-span-12 xl:col-span-8"><Roster /></div>
        <Gate flag="swarm.autonomy"><div className="col-span-12 xl:col-span-4"><AutonomyPanel /></div></Gate>
      </div>
    </div>
  )
}
