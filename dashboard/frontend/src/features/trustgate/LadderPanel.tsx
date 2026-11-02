import clsx from 'clsx'
import { CircleCheck, CircleX, FlaskConical, Lock, TriangleAlert } from 'lucide-react'
import type { ReactNode } from 'react'
import { EmptyState, Panel } from '@/ui'
import type { Ladder, Rung } from './types'

// Which simulator a number came from is the one thing this panel must never blur. Every rung leads with its
// label in large type. A REHEARSAL is a harder Maveric configuration, never Viavi; a rung whose label
// contradicts its twin kind is shown as a LABEL CONFLICT and takes no label colour at all.

const STYLE: Record<string, { word: string; sub: string; color: string; ink: string; dashed?: boolean; Icon: typeof Lock }> = {
  MAVERIC: { word: 'MAVERIC', sub: 'basic simulator the model was trained in', color: 'var(--release)', ink: 'var(--ink-1)', Icon: FlaskConical },
  'REHEARSAL (Viavi-like Maveric)': {
    word: 'REHEARSAL', sub: 'Viavi-like Maveric configuration. NOT Viavi.', color: 'var(--warn)', ink: 'var(--warn-ink)', dashed: true, Icon: FlaskConical,
  },
  VIAVI: { word: 'VIAVI', sub: 'the organisers’ realistic simulator', color: 'var(--good)', ink: 'var(--good-ink)', Icon: CircleCheck },
}

const g = (v: number | null, d = 3) => (v == null ? '—' : v.toFixed(d))
const pc = (v: number | null) => (v == null ? '—' : `${(v * 100).toFixed(0)}%`)

export function LadderPanel({ ladders, error }: { ladders: Ladder[] | null; error: string | null }) {
  const groups = new Map<string, Ladder[]>()
  for (const l of ladders ?? []) groups.set(`${l.model_id} ${l.version}`, [...(groups.get(`${l.model_id} ${l.version}`) ?? []), l])
  return (
    <Panel id="ladder" plane="verify" title="Simulator ladder"
      subtitle="How much of a model’s gain survives the move to a more realistic simulator">
      <p className="mb-3 max-w-[80ch] text-cap text-ink-3">
        Retention compares the gain on a more realistic rung with the gain on the rung below it, as recorded by the checker; it is “—” when there is no measured gain on that rung. Gains, intervals and retention are read from the recorded result and are not recomputed here. Only a real Viavi measurement is ever drawn green or checked.
      </p>
      {error && <div className="mb-3 rounded-md border border-critical/60 bg-critical/10 p-3 text-cap text-critical-ink" role="alert">Could not read ladder results: {error}</div>}
      {ladders == null ? <div className="text-cap text-ink-3">Loading ladder results…</div>
        : ladders.length === 0 ? (
          <div className="grid gap-4 lg:grid-cols-2">
            <EmptyState title="No ladder result recorded yet" body="A ladder appears when a t3.ladder result exists for a model version." />
            <NoViavi />
          </div>
        ) : (
          <div className="space-y-4">
            {[...groups.entries()].map(([k, ls]) => (
              <section key={k} className="rounded-md border border-line bg-page/40 p-3">
                <h4 className="mb-2 flex flex-wrap items-baseline gap-2 text-body font-medium text-ink-1">
                  <span className="mono">{k}</span>
                  {ls.some((l) => l.synthetic) && <SyntheticTag />}
                </h4>
                <div className="space-y-4">
                  {ls.map((l) => <OneLadder key={l.run_id ?? l.top_label} l={l} />)}
                  {!ls.some((l) => l.real_viavi) && <NoViavi />}
                </div>
              </section>
            ))}
          </div>
        )}
    </Panel>
  )
}

export function SyntheticTag({ what = 'SYNTHETIC FIXTURE' }: { what?: string }) {
  return (
    <span className="rounded border border-warn/60 bg-warn/10 px-1.5 py-0.5 text-micro font-semibold tracking-[0.04em] text-warn-ink"
      title="This record is marked synthetic: it was not measured on any twin.">{what}</span>
  )
}

function NoViavi() {
  return (
    <div className="flex items-center gap-3 rounded-md border border-dashed border-line px-4 py-3" data-testid="no-viavi" data-state="not-run">
      <Lock size={18} className="shrink-0 text-ink-3" aria-hidden />
      <div className="min-w-0 flex-1">
        <div className="text-title font-semibold tracking-[0.02em] text-ink-2">VIAVI — not run (no access yet)</div>
        <div className="text-cap text-ink-3">No measurement on the organisers’ realistic simulator exists, so no gain or fidelity is shown for it.</div>
      </div>
      <div className="shrink-0 text-right">
        <div className="text-micro text-ink-3">Retention on Viavi</div>
        <div className="tnum text-[22px] leading-7 font-semibold text-ink-3">—</div>
      </div>
    </div>
  )
}

function OneLadder({ l }: { l: Ladder }) {
  const top = l.rungs[l.rungs.length - 1]
  const ts = STYLE[l.top_label]
  const lo = Math.min(0, ...l.rungs.flatMap((r) => r.gain_ci ?? (r.gain != null ? [r.gain] : [])))
  const hi = Math.max(0.0001, ...l.rungs.flatMap((r) => r.gain_ci ?? (r.gain != null ? [r.gain] : []))) * 1.08
  // Only a measured, non-synthetic Viavi gain may look like a result. Everything else is neutral.
  const toViavi = l.top_label === 'VIAVI'
  const illustrative = l.synthetic || (toViavi && !l.real_viavi)
  const inkColor = illustrative ? 'var(--ink-2)' : ts.ink
  return (
    <div data-ladder={l.top_label} data-synthetic={l.synthetic} data-real-viavi={l.real_viavi}>
      {l.synthetic && (
        <div className="mb-2 inline-flex items-center gap-1.5 rounded border border-warn/60 bg-warn/10 px-2 py-1 text-cap font-bold tracking-[0.04em] text-warn-ink" data-testid="synthetic-ribbon">
          <FlaskConical size={13} aria-hidden />SYNTHETIC — illustrative only. Not measured on any twin.
        </div>
      )}
      <div className="mb-2 flex flex-wrap items-baseline gap-x-3 gap-y-1">
        {l.label_ok ? (
          <span className="text-title font-semibold" style={{ color: inkColor }}>
            {illustrative ? 'Illustrative retention' : 'Retention'} MAVERIC → {ts.word}{l.top_label === 'REHEARSAL (Viavi-like Maveric)' ? ' (Viavi-like Maveric, not Viavi)' : ''}
            {illustrative ? ' (synthetic, not a result)' : ''}:
            <span className="tnum ml-2 text-[22px]">{pc(l.retention)}</span>
          </span>
        ) : (
          <span className="flex items-center gap-1.5 text-title font-semibold text-critical-ink"><TriangleAlert size={16} />Retention not shown: a rung label contradicts its twin kind</span>
        )}
        {l.label_ok && l.retention_ci && <span className="tnum text-cap text-ink-2">{illustrative ? 'illustrative ' : ''}interval {pc(l.retention_ci[0])} to {pc(l.retention_ci[1])}</span>}
        {l.label_ok && l.retention != null && !l.retention_ci && <span className="text-cap text-ink-3">no interval recorded</span>}
        {l.label_ok && l.passed != null && l.target && (illustrative
          ? <span className="text-cap text-ink-3">target {l.target.op} {l.target.value}: not evaluated for synthetic data</span>
          : (
            <span className={clsx('flex items-center gap-1 text-cap font-medium', l.passed ? 'text-good-ink' : 'text-critical-ink')}>
              {l.passed ? <CircleCheck size={13} /> : <CircleX size={13} />}
              target {l.target.op} {l.target.value}: {l.passed ? 'met' : 'not met'}
            </span>
          ))}
      </div>
      <div className={clsx('grid gap-3', l.rungs.length > 2 ? 'md:grid-cols-3' : 'md:grid-cols-2')}>
        {l.rungs.map((r, i) => <RungCard key={i} r={r} lo={lo} hi={hi} synthetic={l.synthetic} />)}
      </div>
      <div className="mt-1.5 text-micro text-ink-3">
        run <span className="mono">{l.run_id}</span> · checked {l.checked_at?.slice(0, 19).replace('T', ' ')}Z · top rung {top.twin_kind}
      </div>
    </div>
  )
}

function RungCard({ r, lo, hi, synthetic }: { r: Rung; lo: number; hi: number; synthetic: boolean }) {
  const base = STYLE[r.label]
  const bad = !r.label_ok
  // a Viavi rung from synthetic data is neutral and hatched: no green, no check mark
  const neutralViavi = r.label === 'VIAVI' && !bad && (synthetic || r.gain == null)
  const s = neutralViavi
    ? { ...base, word: synthetic ? 'VIAVI (synthetic)' : 'VIAVI — not run', sub: synthetic ? 'placeholder values, not a Viavi measurement' : 'no measured gain recorded', color: 'var(--ink-3)', ink: 'var(--ink-2)', dashed: true, Icon: FlaskConical }
    : base
  const color = bad ? 'var(--critical)' : s.color
  const ink = bad ? 'var(--critical-ink)' : s.ink
  return (
    <div className="rounded-md border-2 bg-s1 p-3" data-rung={r.label} data-label-ok={r.label_ok} data-neutral={neutralViavi}
      style={{ borderColor: color, borderStyle: s.dashed || bad ? 'dashed' : 'solid',
        background: s.dashed && !bad
          ? `repeating-linear-gradient(135deg, color-mix(in oklab, ${neutralViavi ? 'var(--ink-3)' : 'var(--warn)'} 12%, transparent) 0 8px, transparent 8px 16px)`
          : undefined }}>
      {neutralViavi && synthetic && <div className="mb-1 text-micro font-bold tracking-[0.04em] text-warn-ink">SYNTHETIC — illustrative only</div>}
      <div className="flex items-center gap-2">
        <s.Icon size={18} aria-hidden style={{ color: ink }} />
        <span className="text-[22px] leading-7 font-bold tracking-[0.03em]" style={{ color: ink }}>{bad ? 'LABEL CONFLICT' : s.word}</span>
      </div>
      <div className="text-cap" style={{ color: ink }}>{bad ? r.label_problem : s.sub}</div>
      <div className="mt-3 flex items-baseline gap-2">
        <span className="tnum text-[20px] font-semibold text-ink-1">{g(r.gain)}</span>
        <span className="text-cap text-ink-3">{neutralViavi ? 'illustrative gain' : 'gain'}{r.n_cases != null ? ` over ${r.n_cases} cases` : ''}</span>
      </div>
      <GainBar r={r} lo={lo} hi={hi} color={color} />
      <div className="mt-1 text-micro text-ink-3">
        {r.gain_ci ? <span className="tnum">interval {g(r.gain_ci[0])} to {g(r.gain_ci[1])}</span> : 'no interval recorded'} · twin <span className="mono">{r.twin_kind ?? '—'}</span>
      </div>
    </div>
  )
}

function GainBar({ r, lo, hi, color }: { r: Rung; lo: number; hi: number; color: string }): ReactNode {
  if (r.gain == null) return null
  const p = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`
  return (
    <div className="relative mt-2 h-4" aria-hidden>
      <div className="absolute inset-x-0 top-1/2 h-px bg-line" />
      <div className="absolute top-0 h-4 w-px bg-axis" style={{ left: p(0) }} />
      {r.gain_ci && <div className="absolute top-1/2 h-1.5 -translate-y-1/2 rounded-full opacity-45" style={{ left: p(r.gain_ci[0]), width: `calc(${p(r.gain_ci[1])} - ${p(r.gain_ci[0])})`, background: color }} />}
      <div className="absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-s1" style={{ left: p(r.gain), background: color }} />
    </div>
  )
}
