import clsx from 'clsx'
import { Pause, Play, Settings2, SkipForward, Square, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '@/core/api'
import { Gate, useFlag } from '@/core/flags'
import { opColor } from '@/core/palette'
import { compact } from '@/core/format'
import type { Operator } from '@/core/types'
import { toast } from '@/core/ui-store'
import { Button, Hash, PageHeader, Panel, Segmented, Slider, StatusPill, Toggle } from '@/ui'
import manifest from './manifest'
import { Convergence, DpInternals, PrivacyBudget, RoundTimeline, UpdateNorms, useFed } from './panels'
import { Attestation, CheckpointRail, FleetTable, FlLog, HePanel, SecAggViz, Sovereignty } from './panels2'

export default function FederationPage() {
  const f = useFed()
  const controls = useFlag('fl.controls')
  const [settings, setSettings] = useState(false)
  const run = (p: Promise<unknown>, msg: string) => p.then(() => toast({ severity: 'notice', title: msg })).catch((e) => toast({ severity: 'critical', title: e.message }))
  const state: string = f.state ?? 'idle'
  return (
    <div>
      <PageHeader
        title={manifest.title}
        description={manifest.description}
        tracks={manifest.tracks}
        actions={controls && (
          <>
            {state === 'running' ? (
              <Button icon={<Pause size={14} />} onClick={() => run(api.fl('pause'), 'Training paused')}>Pause</Button>
            ) : (
              <Button variant="primary" icon={<Play size={14} />} disabled={state === 'complete'} onClick={() => run(api.fl('start'), 'Training running')}>
                {state === 'ready' ? `Start round ${(f.round ?? 0) + 1}` : state === 'paused' ? 'Resume' : 'Start'}
              </Button>
            )}
            <Button icon={<SkipForward size={14} />} disabled={state !== 'running'} onClick={() => run(api.fl('next_round'), 'Skipping to aggregation')}>Next round now</Button>
            <Button variant="danger" icon={<Square size={13} />} disabled={state !== 'running' && state !== 'paused'} onClick={() => run(api.fl('stop'), 'Training stopped (ledgered)')}>Stop</Button>
            <Button icon={<Settings2 size={14} />} onClick={() => setSettings(true)}>Training settings</Button>
          </>
        )}
      />
      <RunStrip />
      <div className="mt-4 grid grid-cols-12 gap-4">
        <div className="col-span-12 xl:col-span-7"><RoundTimeline /></div>
        <div className="col-span-12 xl:col-span-5"><Convergence /></div>
        <div className="col-span-12 xl:col-span-5"><PrivacyBudget /></div>
        <div className="col-span-12 xl:col-span-7"><DpInternals /></div>
        <Gate flag="fl.secagg_viz"><div className="col-span-12 xl:col-span-7"><SecAggViz /></div></Gate>
        <div className="col-span-12 xl:col-span-5"><UpdateNorms /></div>
        <div className="col-span-12"><FleetTable /></div>
        <Gate flag="fl.checkpoints"><div className="col-span-12 xl:col-span-7"><CheckpointRail /></div></Gate>
        <Gate flag="fl.attestation"><div className="col-span-12 xl:col-span-5"><Attestation /></div></Gate>
        <Gate flag="fl.sovereignty"><div className="col-span-12 xl:col-span-7"><Sovereignty /></div></Gate>
        <Gate flag="fl.he"><div className="col-span-12 xl:col-span-5"><HePanel /></div></Gate>
        <div className="col-span-12"><FlLog /></div>
      </div>
      {settings && <SettingsDrawer onClose={() => setSettings(false)} />}
    </div>
  )
}

const STATE_PILL: Record<string, Parameters<typeof StatusPill>[0]['status']> = {
  running: 'running', paused: 'pending', ready: 'pending', stopped: 'denied', complete: 'passed', idle: 'neutral',
}

function RunStrip() {
  const f = useFed()
  const run = f.run ?? {}
  const cfg = f.config ?? {}
  const ops: Operator[] = f.operators ?? []
  const training = ops.filter((o) => o.status === 'training')
  const avgStep = training.length ? training.reduce((a, o) => a + o.step, 0) / training.length : 0
  const tps = training.length * 15 * 64 * 512 // steps/s × batch × tokens per sample
  return (
    <Panel id="run" plane="learn" title={<span className="mono text-body">{run.id}</span>} source={{ aws: 'Amazon ECS (GPU) + S3', mock: 'Simulated run' }}
      actions={<StatusPill status={STATE_PILL[f.state] ?? 'neutral'}>{f.state === 'ready' ? `Round ${(f.round ?? 0) + 1} queued` : f.state}</StatusPill>}>
      <div className="grid grid-cols-2 gap-x-8 gap-y-3 md:grid-cols-4 xl:grid-cols-8">
        <Field label="Base model" value={run.base_model} sub={<Hash value={run.base_hash} />} />
        <Field label="Adapter" value={`LoRA r${cfg.lora_rank}, α${cfg.lora_alpha}`} sub={(cfg.lora_targets ?? []).join(', ')} />
        <Field label="Round" value={<span className="tnum">{f.round} of {cfg.rounds}</span>} sub={<span className="tnum">{f.phase}</span>} />
        <Field label="Local step" value={<span className="tnum">{Math.round(avgStep)} / {f.steps_per_round}</span>} sub={`${training.length} operators training`} />
        <Field label="Throughput" value={<span className="tnum">{training.length ? `${compact(tps)} tok/s` : '—'}</span>} sub="across 5 private GPUs" />
        <Field label="Global val loss" value={<span className="tnum">{f.global?.val_loss?.toFixed(3)}</span>} sub={<span className="tnum">accuracy {((f.global?.accuracy ?? 0) * 100).toFixed(1)}%</span>} />
        <Field label="Aggregation" value={(cfg.strategy ?? '').replace('_', ' ')} sub={`secure agg: ${cfg.secagg_mode}`} />
        <Field label="Privacy" value={<span className="tnum">σ {cfg.noise_multiplier}, C {cfg.clip_norm}</span>} sub={`target ε ${cfg.target_eps}, δ 1e-5`} />
      </div>
      <div className="mt-3 flex items-center gap-1">
        {ops.map((o) => (
          <div key={o.id} className="flex-1" title={`${o.short}: ${o.status}, step ${Math.round(o.step)}`}>
            <div className="h-1 overflow-hidden rounded-full bg-s3">
              <div className="h-full rounded-full transition-[width] duration-300" style={{ width: `${(o.step / (f.steps_per_round ?? 240)) * 100}%`, background: opColor(o.id) }} />
            </div>
          </div>
        ))}
      </div>
    </Panel>
  )
}

function Field({ label, value, sub }: { label: string; value: React.ReactNode; sub?: React.ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="text-micro text-ink-3">{label}</div>
      <div className="truncate text-body font-medium text-ink-1">{value}</div>
      {sub && <div className="truncate text-micro text-ink-3">{sub}</div>}
    </div>
  )
}

function SettingsDrawer({ onClose }: { onClose: () => void }) {
  const f = useFed()
  const cfg = f.config ?? {}
  const ops: Operator[] = f.operators ?? []
  const [d, setD] = useState({ ...cfg })
  const [proj, setProj] = useState<Record<string, number>>({})
  useEffect(() => {
    const t = setTimeout(() => api.projectedEps(d.noise_multiplier, d.rounds).then(setProj).catch(() => {}), 150)
    return () => clearTimeout(t)
  }, [d.noise_multiplier, d.rounds])
  const changed = Object.keys(d).filter((k) => JSON.stringify(d[k]) !== JSON.stringify(cfg[k]))
  const apply = () => {
    const changes = Object.fromEntries(changed.map((k) => [k, d[k]]))
    api.flConfig(changes).then(() => { toast({ severity: 'notice', title: `Applied ${changed.length} change${changed.length === 1 ? '' : 's'} (ledgered)` }); onClose() })
  }
  const set = (k: string) => (v: unknown) => setD((x: any) => ({ ...x, [k]: v }))
  return (
    <div className="fixed inset-0 z-50 flex justify-end" style={{ background: 'var(--scrim)' }} onClick={onClose}>
      <aside role="dialog" aria-label="Training settings" className="flex h-full w-[440px] flex-col border-l border-line bg-s1" onClick={(e) => e.stopPropagation()}>
        <header className="flex h-12 items-center border-b border-line px-4">
          <h2 className="text-title font-medium">Training settings</h2>
          <button type="button" aria-label="Close" onClick={onClose} className="ml-auto text-ink-3 hover:text-ink-1"><X size={16} /></button>
        </header>
        <div className="flex-1 space-y-5 overflow-auto p-4">
          <section className="space-y-4">
            <h3 className="text-cap font-medium text-ink-2">Differential privacy</h3>
            <Slider label="Noise multiplier σ" value={d.noise_multiplier} min={0.3} max={3} step={0.05} onChange={(v) => set('noise_multiplier')(Math.round(v * 100) / 100)} format={(v) => v.toFixed(2)} />
            <Slider label="Clip norm C" value={d.clip_norm} min={0.1} max={5} step={0.1} onChange={(v) => set('clip_norm')(Math.round(v * 10) / 10)} format={(v) => v.toFixed(1)} />
            <Slider label="Target ε" value={d.target_eps} min={0.5} max={10} step={0.1} onChange={(v) => set('target_eps')(Math.round(v * 10) / 10)} format={(v) => v.toFixed(1)} />
            <div className="rounded-md border border-line bg-s2 p-3">
              <div className="mb-2 text-micro text-ink-3">Projected ε at round {d.rounds} with these settings</div>
              {ops.map((o) => {
                const p = proj[o.id] ?? 0
                const over = p >= d.target_eps
                return (
                  <div key={o.id} className="mb-1.5 flex items-center gap-2 text-cap">
                    <span className="w-12 text-ink-2">{o.short}</span>
                    <div className="relative h-2 flex-1 rounded-full bg-s3">
                      <div className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${Math.min(100, (p / (d.target_eps * 1.5)) * 100)}%`, background: over ? 'var(--critical)' : opColor(o.id) }} />
                      <div className="absolute inset-y-[-3px] w-px bg-ink-2" style={{ left: `${(1 / 1.5) * 100}%` }} />
                    </div>
                    <span className={clsx('tnum w-12 text-right', over ? 'text-critical-ink' : 'text-ink-1')}>{p.toFixed(2)}</span>
                  </div>
                )
              })}
              <div className="mt-1 text-micro text-ink-3">The vertical tick is the target. Red operators would exhaust their budget and leave early.</div>
            </div>
          </section>
          <section className="space-y-4">
            <h3 className="text-cap font-medium text-ink-2">Optimisation</h3>
            <Slider label="Rounds" value={d.rounds} min={1} max={50} step={1} onChange={(v) => set('rounds')(Math.round(v))} />
            <Slider label="Learning rate" value={d.lr} min={1e-5} max={1e-3} log onChange={(v) => set('lr')(Number(v.toPrecision(2)))} format={(v) => v.toExponential(1)} />
            <Slider label="Auto-checkpoint every N rounds" value={d.auto_checkpoint_every} min={0} max={10} step={1} onChange={(v) => set('auto_checkpoint_every')(Math.round(v))} />
          </section>
          <section className="space-y-3">
            <h3 className="text-cap font-medium text-ink-2">Aggregation</h3>
            <div className="text-micro text-ink-3">Strategy</div>
            <Segmented value={d.strategy} onChange={set('strategy')} options={[{ value: 'fedavg', label: 'FedAvg' }, { value: 'trimmed_mean', label: 'Trimmed mean' }, { value: 'krum', label: 'Krum' }, { value: 'norm_bounded', label: 'Norm-bounded' }]} />
            <div className="text-micro text-ink-3">Secure aggregation mode</div>
            <Segmented value={d.secagg_mode} onChange={set('secagg_mode')} options={[{ value: 'masking', label: 'Masking' }, { value: 'enclave', label: 'Masking + enclave' }, { value: 'he', label: 'CKKS (HE)' }]} />
            <p className="text-micro text-ink-3">Masking alone keeps the server blind but prevents robust aggregation. The enclave mode releases mask seeds only to the attested aggregator, which can then reject poisoned updates.</p>
            <div className="text-micro text-ink-3">Participation</div>
            {ops.map((o) => (
              <label key={o.id} className="flex items-center gap-2 text-cap">
                <Toggle checked={o.participating} onChange={() => api.flClient(o.id, 'pause')} label={`${o.short} participation`} />
                <span className="size-2 rounded-full" style={{ background: opColor(o.id) }} />{o.short} {o.name}
              </label>
            ))}
          </section>
        </div>
        <footer className="flex items-center gap-2 border-t border-line p-4">
          <span className="text-micro text-ink-3">{changed.length ? `${changed.length} unsaved change${changed.length === 1 ? '' : 's'}; applying writes a ledger record` : 'No changes'}</span>
          <Button className="ml-auto" variant="ghost" onClick={onClose}>Cancel</Button>
          <Button variant="primary" disabled={!changed.length} onClick={apply}>Apply changes</Button>
        </footer>
      </aside>
    </div>
  )
}
