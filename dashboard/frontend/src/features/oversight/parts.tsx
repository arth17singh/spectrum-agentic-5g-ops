import clsx from 'clsx'
import { Check, Clock3, UserRound } from 'lucide-react'
import type { ReactNode } from 'react'
import { duration } from '@/core/format'
import type { Approval, Rec } from '@/core/types'

// Shared pieces for the Oversight Desk and the mobile approval page.

export const TIER: Record<string, { label: string; color: string }> = {
  high: { label: 'High risk', color: 'var(--warn)' },
  critical: { label: 'Critical, four-eyes', color: 'var(--critical)' },
}

export function TierPill({ tier, size = 'md' }: { tier: string; size?: 'sm' | 'md' }) {
  const t = TIER[tier] ?? TIER.high
  return (
    <span className={clsx('inline-flex shrink-0 items-center gap-1 rounded-[4px] font-medium whitespace-nowrap', size === 'md' ? 'h-[22px] px-1.5 text-cap' : 'h-[18px] px-1 text-micro')}
      style={{ color: tier === 'critical' ? 'var(--critical-ink)' : 'var(--warn)', background: `color-mix(in oklab, ${t.color} 15%, transparent)` }}>
      <span className="size-1.5 rounded-full" style={{ background: t.color }} />
      {t.label}
    </span>
  )
}

/** Remaining SLA at the current escalation level, amber under 60 s, red under 15 s. */
export function slaLeft(a: Approval, now: number) {
  return a.sla_deadline_t - now
}

export function SlaCountdown({ a, now, big }: { a: Approval; now: number; big?: boolean }) {
  if (a.status !== 'pending') return <span className="text-micro text-ink-3">{a.ttd_s != null ? `decided in ${duration(a.ttd_s)}` : a.status}</span>
  const left = slaLeft(a, now)
  const color = left < 15 ? 'var(--critical-ink)' : left < 60 ? 'var(--warn)' : 'var(--ink-1)'
  const m = Math.max(0, Math.floor(left / 60))
  const s = Math.max(0, Math.floor(left % 60))
  return (
    <span className={clsx('tnum inline-flex items-center gap-1 font-medium', big ? 'text-section' : 'text-cap')} style={{ color }}
      title={`SLA at ${a.level}: ${a.sla_s}s, then auto-escalates`}>
      <Clock3 size={big ? 16 : 12} />
      {left <= 0 ? 'escalating…' : `${m}:${String(s).padStart(2, '0')}`}
    </span>
  )
}

/** Four-eyes slots: one per required approver; a person can only fill one. */
export function FourEyes({ a }: { a: Approval }) {
  return (
    <div className="flex items-center gap-2">
      {Array.from({ length: a.required }, (_, i) => {
        const ap = a.approvals[i]
        return (
          <span key={i} className={clsx('inline-flex h-7 items-center gap-1.5 rounded-md border px-2 text-cap',
            ap ? 'border-good/50 text-ink-1' : 'border-dashed border-line text-ink-3')}>
            {ap ? <Check size={13} className="text-good-ink" /> : <UserRound size={13} />}
            {ap ? `${ap.name} (${ap.device})` : i === 0 ? 'Approver' : 'Second approver, different person'}
          </span>
        )
      })}
    </div>
  )
}

const METRICS: { key: string; label: string; unit: string; better: 'lower' | 'higher'; digits: number }[] = [
  { key: 'congestion_pct', label: 'Stadium congestion', unit: '%', better: 'lower', digits: 1 },
  { key: 'dropped_pct', label: 'Dropped calls', unit: '%', better: 'lower', digits: 2 },
  { key: 'health_pct', label: 'Network health', unit: '%', better: 'higher', digits: 1 },
  { key: 'throughput_gbps', label: 'Throughput', unit: ' Gbps', better: 'higher', digits: 1 },
  { key: 'energy_kw', label: 'Energy draw', unit: ' kW', better: 'lower', digits: 1 },
]

/** Pre-flight before/after bars from the twin fork. Improvement is good-green, regression red. */
export function ImpactBars({ preflight, compact, only }: { preflight: Rec; compact?: boolean; only?: string[] }) {
  const b = preflight?.before ?? {}
  const af = preflight?.after ?? {}
  const rows = METRICS.filter((m) => b[m.key] != null && af[m.key] != null && (!only || only.includes(m.key)))
  if (!rows.length) return <div className="text-cap text-ink-3">No twin pre-flight was recorded for this request.</div>
  return (
    <div className={clsx('space-y-2.5', compact && 'space-y-2')}>
      {rows.map((m) => {
        const v0 = Number(b[m.key])
        const v1 = Number(af[m.key])
        const max = Math.max(v0, v1, m.unit === '%' ? 100 : 0) || 1
        const improved = m.better === 'lower' ? v1 < v0 : v1 > v0
        const same = Math.abs(v1 - v0) < 10 ** -m.digits
        const delta = v1 - v0
        return (
          <div key={m.key} className="grid grid-cols-[minmax(0,130px)_1fr_auto] items-center gap-3 text-cap">
            <span className="truncate text-ink-2">{m.label}</span>
            <div className="space-y-1">
              <div className="h-1.5 rounded-r-[3px] bg-ink-3/50" style={{ width: `${(v0 / max) * 100}%` }} title={`before: ${v0}`} />
              <div className="h-1.5 rounded-r-[3px]" style={{ width: `${(v1 / max) * 100}%`, background: same ? 'var(--ink-3)' : improved ? 'var(--good)' : 'var(--critical)' }} title={`after: ${v1}`} />
            </div>
            <span className="tnum whitespace-nowrap text-right">
              <span className="text-ink-3">{v0.toFixed(m.digits)}</span>
              <span className="text-ink-3"> → </span>
              <span className="font-medium text-ink-1">{v1.toFixed(m.digits)}{m.unit}</span>
              {!same && <span className="ml-1.5" style={{ color: improved ? 'var(--good-ink)' : 'var(--critical-ink)' }}>{delta > 0 ? '+' : ''}{delta.toFixed(m.digits)}</span>}
            </span>
          </div>
        )
      })}
      {!compact && (
        <div className="flex gap-4 pt-1 text-micro text-ink-3">
          <span className="flex items-center gap-1.5"><span className="h-1.5 w-3 rounded-sm bg-ink-3/50" />now</span>
          <span className="flex items-center gap-1.5"><span className="h-1.5 w-3 rounded-sm bg-good" />predicted by the twin after the plan</span>
        </div>
      )}
    </div>
  )
}

export function Block({ title, children, right, className }: { title: string; children: ReactNode; right?: ReactNode; className?: string }) {
  return (
    <section className={clsx('min-w-0', className)}>
      <div className="mb-2 flex items-center gap-2">
        <h4 className="text-cap font-medium text-ink-2">{title}</h4>
        {right && <div className="ml-auto">{right}</div>}
      </div>
      {children}
    </section>
  )
}
