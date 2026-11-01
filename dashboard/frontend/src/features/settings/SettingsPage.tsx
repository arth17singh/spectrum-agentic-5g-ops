import clsx from 'clsx'
import { FlaskConical, Link2Off, RotateCcw, Search } from 'lucide-react'
import { useMemo, useState } from 'react'
import { api } from '@/core/api'
import { FLAGS, PRESETS, isOn, rawValue, useFlags, type FlagDef, type Source, type Stage } from '@/core/flags'
import { planeColor } from '@/core/palette'
import { useLive } from '@/core/store'
import { useUi } from '@/core/ui-store'
import { features } from '@/features'
import { Button, CodeBlock, KV, LiveDot, PageHeader, Panel, Segmented, StatusPill, Toggle } from '@/ui'
import manifest from './manifest'

const STAGE_STYLE: Record<Stage, { color: string; label: string }> = {
  stable: { color: 'var(--good-ink)', label: 'Stable' },
  beta: { color: 'var(--release)', label: 'Beta' },
  experimental: { color: 'var(--serious)', label: 'Experimental' },
  'demo-only': { color: 'var(--ink-2)', label: 'Demo only' },
}

function StageBadge({ stage }: { stage: Stage }) {
  const s = STAGE_STYLE[stage]
  return (
    <span className="inline-flex h-[18px] items-center gap-1 rounded-[4px] border px-1.5 text-micro font-medium whitespace-nowrap"
      style={{ color: s.color, borderColor: `color-mix(in oklab, ${s.color} 45%, transparent)` }}>
      {stage === 'demo-only' && <FlaskConical size={11} aria-hidden />}
      {s.label}
    </span>
  )
}

const SOURCE_LABEL: Record<Source, string> = { default: 'Code default', remote: 'Remote config', local: 'This browser', url: 'URL' }

export default function SettingsPage() {
  return (
    <div>
      <PageHeader title={manifest.title} description={manifest.description} />
      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-12"><FlagsPanel /></div>
        <div className="col-span-12 xl:col-span-7"><ModulesPanel /></div>
        <div className="col-span-12 xl:col-span-5"><AddFeaturePanel /></div>
        <div className="col-span-12 xl:col-span-4"><DataSourcePanel /></div>
        <div className="col-span-12 xl:col-span-4"><SimulationPanel /></div>
        <div className="col-span-12 xl:col-span-4"><DisplayPanel /></div>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ flags

function FlagsPanel() {
  const st = useFlags()
  const [q, setQ] = useState('')
  const [stage, setStage] = useState<'all' | Stage>('all')
  const rows = useMemo(() => FLAGS.map((f) => {
    const raw = rawValue(f.id, st)
    const eff = isOn(f.id, st)
    const blockedBy = raw.value && !eff ? (f.dependsOn ?? []).filter((d) => !isOn(d, st)) : []
    return { f, raw, eff, blockedBy }
  }), [st])
  const visible = rows.filter(({ f }) => (stage === 'all' || f.stage === stage) &&
    (!q || `${f.id} ${f.label} ${f.description}`.toLowerCase().includes(q.toLowerCase())))
  const groups = [...new Set(visible.map((r) => r.f.group))]
  const counts = {
    on: rows.filter((r) => r.eff).length,
    local: rows.filter((r) => r.raw.source === 'local').length,
    remote: rows.filter((r) => r.raw.source === 'remote').length,
    url: rows.filter((r) => r.raw.source === 'url').length,
    blocked: rows.filter((r) => r.blockedBy.length).length,
  }
  const activePreset = Object.entries(PRESETS).find(([, p]) => JSON.stringify(p.values) === JSON.stringify(st.local))?.[0]

  return (
    <Panel id="flags" title="Feature flags" subtitle={`${FLAGS.length} flags, changes apply live`} source={{ aws: 'AWS AppConfig', mock: 'flags.json + this browser' }} flush>
      <div className="grid grid-cols-1 gap-4 border-b border-line px-4 py-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <div>
          <div className="mb-2 text-micro text-ink-3">How a flag’s value is decided (last one wins)</div>
          <ol className="flex items-stretch gap-1.5">
            {(['default', 'remote', 'local', 'url'] as Source[]).map((s, i) => (
              <li key={s} className="flex-1 rounded-md border border-line bg-s2 px-2.5 py-2">
                <div className="flex items-baseline gap-1.5 text-cap"><span className="tnum text-ink-3">{i + 1}</span><span className="font-medium text-ink-1">{SOURCE_LABEL[s]}</span></div>
                <div className="tnum text-micro text-ink-3">{s === 'default' ? `${FLAGS.length} defined in code` : s === 'remote' ? `${counts.remote} from /api/config/flags` : s === 'local' ? `${counts.local} overridden here` : `${counts.url} via ?ff=`}</div>
              </li>
            ))}
          </ol>
          <div className="mt-2 text-micro text-ink-3">A flag is only effective when every flag it depends on is also on. Turning off a screen hides its route, nav item, palette commands and panels together.</div>
        </div>
        <div>
          <div className="mb-2 text-micro text-ink-3">Presets</div>
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(PRESETS).map(([k, p]) => (
              <button key={k} type="button" onClick={() => st.applyPreset(p.values)}
                className={clsx('rounded-md border px-3 py-2 text-left transition-colors', activePreset === k ? 'border-ink-2 bg-s3' : 'border-line bg-s2 hover:bg-s3')}>
                <div className="flex items-center gap-2 text-cap font-medium text-ink-1">{p.label}{activePreset === k && <StatusPill status="good" size="sm">Active</StatusPill>}</div>
                <div className="text-micro text-ink-3">{p.description}</div>
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3 border-b border-line px-4 py-2.5">
        <label className="flex h-8 w-72 items-center gap-2 rounded-md border border-line bg-s2 px-2.5">
          <Search size={14} className="text-ink-3" />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter flags" aria-label="Filter flags" className="flex-1 bg-transparent text-cap text-ink-1 outline-none placeholder:text-ink-3" />
        </label>
        <Segmented value={stage} onChange={setStage} size="sm" ariaLabel="Stage"
          options={[{ value: 'all', label: 'All' }, { value: 'stable', label: 'Stable' }, { value: 'beta', label: 'Beta' }, { value: 'experimental', label: 'Experimental' }, { value: 'demo-only', label: 'Demo only' }]} />
        <span className="tnum ml-auto text-cap text-ink-2">{counts.on} of {FLAGS.length} effectively on{counts.blocked ? `, ${counts.blocked} blocked by a dependency` : ''}</span>
        {counts.local > 0 && <Button size="sm" variant="ghost" icon={<RotateCcw size={13} />} onClick={() => st.applyPreset({})}>Clear {counts.local} browser override{counts.local > 1 ? 's' : ''}</Button>}
      </div>

      <div className="max-h-[760px] overflow-auto">
        <table className="w-full text-cap">
          <thead className="sticky top-0 z-10 bg-s1">
            <tr className="border-b border-line text-left text-micro text-ink-3">
              {['Flag', 'Stage', 'What it controls', 'Depends on', 'Value from', 'On', ''].map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}
            </tr>
          </thead>
          {groups.map((g) => (
            <tbody key={g}>
              <tr><td colSpan={7} className="bg-s2/60 px-4 py-1.5 text-micro font-medium text-ink-2">{g}</td></tr>
              {visible.filter((r) => r.f.group === g).map((r) => <FlagRow key={r.f.id} {...r} />)}
            </tbody>
          ))}
        </table>
        {!visible.length && <div className="py-8 text-center text-cap text-ink-3">No flags match this filter.</div>}
      </div>
    </Panel>
  )
}

function FlagRow({ f, raw, eff, blockedBy }: { f: FlagDef; raw: { value: boolean; source: Source }; eff: boolean; blockedBy: string[] }) {
  const setLocal = useFlags((s) => s.setLocal)
  const urlWins = raw.source === 'url'
  return (
    <tr className={clsx('border-b border-line/60 hover:bg-s2/40', !eff && 'text-ink-3')}>
      <td className="px-4 py-2">
        <div className={clsx('font-medium', eff ? 'text-ink-1' : 'text-ink-2')}>{f.label}</div>
        <div className="mono text-micro text-ink-3">{f.id}</div>
      </td>
      <td className="px-4"><StageBadge stage={f.stage} /></td>
      <td className="max-w-[320px] px-4 text-ink-2">{f.description}</td>
      <td className="px-4">
        <div className="flex flex-wrap gap-1">
          {(f.dependsOn ?? []).map((d) => {
            const blocked = blockedBy.includes(d)
            return (
              <span key={d} className={clsx('mono inline-flex items-center gap-1 rounded px-1 text-micro', blocked ? 'bg-critical/15 text-critical-ink' : 'bg-s3 text-ink-2')}
                title={blocked ? `${d} is off, so this flag has no effect` : undefined}>
                {blocked && <Link2Off size={10} aria-hidden />}{d}
              </span>
            )
          })}
          {!f.dependsOn?.length && <span className="text-micro text-ink-3">—</span>}
        </div>
      </td>
      <td className="px-4 whitespace-nowrap">
        <span className={clsx('text-micro', raw.source === 'default' ? 'text-ink-3' : 'font-medium text-ink-1')}>{SOURCE_LABEL[raw.source]}</span>
      </td>
      <td className="px-4">
        <div className="flex items-center gap-2" title={urlWins ? 'Set by the URL (?ff=), which overrides this browser' : blockedBy.length ? `Blocked by ${blockedBy.join(', ')}` : undefined}>
          <Toggle checked={raw.value} onChange={(v) => setLocal(f.id, v)} label={`${f.label} ${raw.value ? 'on' : 'off'}`} disabled={urlWins} />
          {blockedBy.length > 0 && <span className="text-micro text-critical-ink">blocked</span>}
        </div>
      </td>
      <td className="px-4 text-right">
        {raw.source === 'local' && (
          <button type="button" onClick={() => setLocal(f.id, null)} className="inline-flex items-center gap-1 text-micro text-ink-3 hover:text-ink-1" title="Remove this browser’s override">
            <RotateCcw size={11} /> Reset
          </button>
        )}
      </td>
    </tr>
  )
}

// ------------------------------------------------------------------ registered modules

function ModulesPanel() {
  const st = useFlags()
  const mods = [...features].sort((a, b) => a.nav.order - b.nav.order)
  return (
    <Panel id="modules" title="Registered feature modules" subtitle="from src/features/index.ts" flush>
      <div className="overflow-x-auto">
      <table className="w-full text-cap">
        <thead>
          <tr className="border-b border-line text-left text-micro text-ink-3">
            {['Module', 'Route', 'Plane', 'Hotkey', 'State'].map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}
          </tr>
        </thead>
        <tbody>
          {mods.map((m) => {
            const on = isOn(m.id, st)
            return (
              <tr key={m.id + m.route} className="border-b border-line/60">
                <td className="px-4 py-2">
                  <span className="flex items-center gap-2 font-medium whitespace-nowrap text-ink-1"><m.nav.icon size={14} className="text-ink-2" />{m.nav.label}</span>
                  <span className="mono pl-[22px] text-micro text-ink-3">gate {m.id}{m.commands?.length ? `, ${m.commands.length} commands` : ''}</span>
                </td>
                <td className="mono px-4 text-ink-2">{m.route}{m.bare && <div className="font-sans text-micro text-ink-3">outside the shell</div>}</td>
                <td className="px-4">
                  {m.nav.plane === 'all' || m.nav.plane === 'none' ? <span className="text-micro text-ink-3">{m.nav.plane === 'all' ? 'all planes' : '—'}</span>
                    : <span className="inline-flex items-center gap-1.5 text-micro text-ink-2"><span className="size-1.5 rounded-full" style={{ background: planeColor(m.nav.plane) }} />{m.nav.plane}</span>}
                </td>
                <td className="mono px-4 text-ink-2">{m.nav.hotkey ? `g ${m.nav.hotkey}` : '—'}</td>
                <td className="px-4">{on ? <StatusPill status="good" size="sm">Mounted</StatusPill> : <StatusPill status="neutral" size="sm">Gated off</StatusPill>}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      </div>
    </Panel>
  )
}

function AddFeaturePanel() {
  return (
    <Panel id="add-feature" title="Adding or removing a screen" subtitle="no other file changes">
      <ol className="mb-3 space-y-1 text-cap text-ink-2">
        <li><span className="tnum mr-1.5 text-ink-3">1</span>Create <span className="mono text-ink-1">src/features/&lt;name&gt;/manifest.ts</span> and its page.</li>
        <li><span className="tnum mr-1.5 text-ink-3">2</span>Add one import line to <span className="mono text-ink-1">src/features/index.ts</span>.</li>
        <li><span className="tnum mr-1.5 text-ink-3">3</span>Add its flag to <span className="mono text-ink-1">src/core/flags.tsx</span>. Route, nav, palette and hotkey appear automatically.</li>
      </ol>
      <CodeBlock lang="text" code={`export default defineFeature({
  id: 'screen.capacity',          // its flag
  route: '/capacity',
  nav: { label: 'Capacity', icon: Gauge,
         plane: 'operate', hotkey: 'c', order: 11 },
  title: 'Capacity planning',
  description: 'Forecast PRB headroom per cluster.',
  component: lazy(() => import('./CapacityPage')),
  commands: [{ id: 'cap.run', group: 'Network',
    label: 'Run capacity forecast', run: () => … }],
})`} />
      <p className="mt-2 text-micro text-ink-3">Panels inside a page are wrapped in <span className="mono">&lt;Gate flag=…&gt;</span>; a gated-off panel leaves no gap in the grid.</p>
    </Panel>
  )
}

// ------------------------------------------------------------------ data source, simulation, display

function DataSourcePanel() {
  const cfg = useLive((s) => s.config)
  const conn = useLive((s) => s.conn)
  const rate = useLive((s) => s.msgPerSec)
  const ready = useLive((s) => s.ready)
  const adapters: Record<string, string> = cfg?.adapters ?? {}
  return (
    <Panel id="datasource" title="Data source" subtitle="where every panel reads from">
      <KV rows={[
        ['Mode', <span className="flex items-center gap-2"><span className={clsx('rounded px-1.5 py-0.5 text-micro font-semibold', cfg?.mode === 'aws' ? 'bg-release/20 text-release' : 'bg-s3 text-ink-2')}>{cfg?.mode === 'aws' ? 'AWS' : 'MOCK'}</span><span className="text-ink-3">set by SPECTRUM_MODE on the backend</span></span>],
        ['Region', <span className="mono">{cfg?.region ?? '—'}</span>],
        ['Backend', <span className="mono">{location.host}/api, /ws</span>],
        ['WebSocket', <LiveDot state={!ready ? 'connecting' : conn} label={`${conn}, ${rate} events/s`} />],
      ]} />
      <div className="mt-3 mb-1 text-micro text-ink-3">Adapter seams</div>
      <ul className="space-y-1">
        {Object.entries(adapters).map(([k, v]) => (
          <li key={k} className="grid grid-cols-[84px_1fr] gap-2 text-cap">
            <span className="mono text-ink-3">{k}</span><span className="text-ink-1">{v}</span>
          </li>
        ))}
      </ul>
    </Panel>
  )
}

function SimulationPanel() {
  const clock = useLive((s) => s.clock)
  const cfg = useLive((s) => s.config)
  return (
    <Panel id="simulation" title="Simulation" subtitle="mock engine clock">
      <KV rows={[
        ['Seed', <span className="tnum mono">{cfg?.seed ?? '—'}</span>],
        ['Warm-up', <span className="tnum">{cfg?.warm_ms ? `${cfg.warm_ms} ms to the pre-demo state` : '—'}</span>],
        ['Scenario day', 'Saturday 31 Oct 2026, from 21:30 PDT'],
      ]} />
      <div className="mt-4 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-cap text-ink-2">Speed</span>
          <Segmented size="sm" value={clock.speed} onChange={(v) => api.sim({ speed: v })} ariaLabel="Simulation speed"
            options={[{ value: 0.5, label: '0.5×' }, { value: 1, label: '1×' }, { value: 2, label: '2×' }, { value: 5, label: '5×' }]} />
        </div>
        <label className="flex items-center justify-between">
          <span className="text-cap text-ink-2">Run the simulation</span>
          <Toggle checked={!clock.paused} onChange={(v) => api.sim({ paused: !v })} label="Run simulation" />
        </label>
        <p className="text-micro text-ink-3">Reset lives in the Demo Director (hold R or the Reset button) so it can’t be hit by accident.</p>
      </div>
    </Panel>
  )
}

function DisplayPanel() {
  const ui = useUi()
  const cfg = useLive((s) => s.config)
  return (
    <Panel id="display" title="Display" subtitle="saved in this browser">
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-cap text-ink-2">Theme</span>
          <Segmented size="sm" value={ui.theme} onChange={(v) => ui.set({ theme: v })} options={[{ value: 'ops', label: 'Ops (dark)' }, { value: 'projector', label: 'Projector (light)' }]} />
        </div>
        <div className="flex items-center justify-between">
          <span className="text-cap text-ink-2">Density</span>
          <Segmented size="sm" value={ui.density} onChange={(v) => ui.set({ density: v })} options={[{ value: 'comfortable', label: 'Comfortable' }, { value: 'compact', label: 'Compact' }]} />
        </div>
        <div className="flex items-center justify-between">
          <span className="text-cap text-ink-2">Motion</span>
          <Segmented size="sm" value={ui.motion} onChange={(v) => ui.set({ motion: v })} options={[{ value: 'auto', label: 'Follow system' }, { value: 'reduced', label: 'Reduced' }]} />
        </div>
      </div>
      <div className="mt-4 border-t border-line pt-3">
        <div className="mb-1 text-micro text-ink-3">About</div>
        <KV rows={[
          ['Console', 'SPECTRUM 0.1.0 (prototype)'],
          ['Backend API', <span className="tnum">{cfg?.version ?? '—'}</span>],
          ['Event schema', 'envelope v1 (seq, ts, sim_t, plane, type, severity)'],
          ['Ledger format', 'SHA-256 chain, Ed25519, Merkle anchor every 32'],
        ]} />
      </div>
    </Panel>
  )
}
