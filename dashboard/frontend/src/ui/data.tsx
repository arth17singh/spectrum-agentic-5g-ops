import clsx from 'clsx'
import {
  CircleAlert, CircleCheck, Clock3, FlaskConical, Lock, OctagonX, ShieldCheck, ShieldX, TriangleAlert, Copy, Check,
} from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { shortHash } from '@/core/format'
import { useUi } from '@/core/ui-store'
import { Sparkline } from '@/charts/Sparkline'

// ------------------------------------------------------------------ status vocabulary (one everywhere)

export type Status = 'verified' | 'signed' | 'pending' | 'denied' | 'quarantined' | 'tampered' | 'simulated' | 'good' |
  'warning' | 'serious' | 'critical' | 'neutral' | 'running' | 'passed' | 'blocked' | 'approved' | 'rejected'

const STATUS: Record<Status, { color: string; ink: string; Icon: typeof ShieldCheck; word: string }> = {
  verified: { color: 'var(--good)', ink: 'var(--good-ink)', Icon: ShieldCheck, word: 'Verified' },
  signed: { color: 'var(--good)', ink: 'var(--good-ink)', Icon: ShieldCheck, word: 'Signed' },
  approved: { color: 'var(--good)', ink: 'var(--good-ink)', Icon: CircleCheck, word: 'Approved' },
  passed: { color: 'var(--good)', ink: 'var(--good-ink)', Icon: CircleCheck, word: 'Passed' },
  good: { color: 'var(--good)', ink: 'var(--good-ink)', Icon: CircleCheck, word: 'OK' },
  pending: { color: 'var(--warn)', ink: 'var(--warn)', Icon: Clock3, word: 'Pending' },
  running: { color: 'var(--release)', ink: 'var(--ink-1)', Icon: Clock3, word: 'Running' },
  warning: { color: 'var(--warn)', ink: 'var(--warn)', Icon: TriangleAlert, word: 'Warning' },
  serious: { color: 'var(--serious)', ink: 'var(--serious)', Icon: CircleAlert, word: 'Degraded' },
  denied: { color: 'var(--critical)', ink: 'var(--critical-ink)', Icon: OctagonX, word: 'Denied' },
  blocked: { color: 'var(--critical)', ink: 'var(--critical-ink)', Icon: OctagonX, word: 'Blocked' },
  rejected: { color: 'var(--critical)', ink: 'var(--critical-ink)', Icon: OctagonX, word: 'Rejected' },
  critical: { color: 'var(--critical)', ink: 'var(--critical-ink)', Icon: OctagonX, word: 'Critical' },
  tampered: { color: 'var(--critical)', ink: 'var(--critical-ink)', Icon: ShieldX, word: 'Tampered' },
  quarantined: { color: 'var(--ink-3)', ink: 'var(--ink-2)', Icon: Lock, word: 'Quarantined' },
  simulated: { color: 'var(--ink-3)', ink: 'var(--ink-3)', Icon: FlaskConical, word: 'Simulated' },
  neutral: { color: 'var(--ink-3)', ink: 'var(--ink-2)', Icon: CircleCheck, word: '' },
}

export function StatusPill({ status, children, size = 'md', title }: {
  status: Status; children?: ReactNode; size?: 'sm' | 'md'; title?: string
}) {
  const s = STATUS[status]
  return (
    <span
      title={title}
      className={clsx('inline-flex shrink-0 items-center gap-1 rounded-[4px] font-medium whitespace-nowrap',
        size === 'md' ? 'h-[22px] px-1.5 text-cap' : 'h-[18px] px-1 text-micro')}
      style={{ color: s.ink, background: `color-mix(in oklab, ${s.color} 14%, transparent)` }}
    >
      <s.Icon size={size === 'md' ? 13 : 11} aria-hidden />
      {children ?? s.word}
    </span>
  )
}

export function LiveDot({ state = 'live', label }: { state?: 'live' | 'lagging' | 'offline' | 'connecting' | 'idle'; label?: string }) {
  const color = { live: 'var(--good)', lagging: 'var(--warn)', offline: 'var(--critical)', connecting: 'var(--ink-3)', idle: 'var(--ink-3)' }[state]
  return (
    <span className="inline-flex items-center gap-1.5" title={label}>
      <span className={clsx('size-2 rounded-full', state === 'live' && 'live-dot')} style={{ background: color, color }} />
      {label && <span className="text-cap text-ink-2">{label}</span>}
    </span>
  )
}

// ------------------------------------------------------------------ proof primitives

export function Hash({ value, head = 6, tail = 4, className, full }: { value?: string | null; head?: number; tail?: number; className?: string; full?: boolean }) {
  const [copied, setCopied] = useState(false)
  if (!value) return <span className="mono text-ink-3">—</span>
  return (
    <button
      type="button"
      title={`${value}\nClick to copy`}
      onClick={(e) => {
        e.stopPropagation()
        navigator.clipboard?.writeText(value)
        setCopied(true)
        setTimeout(() => setCopied(false), 1000)
      }}
      className={clsx('mono inline-flex items-center gap-1 rounded px-0.5 text-ink-2 hover:bg-s2 hover:text-ink-1', className)}
    >
      <span className={full ? 'break-all text-left' : ''}>{full ? value : shortHash(value, head, tail)}</span>
      {copied ? <Check size={11} className="text-good-ink" /> : null}
    </button>
  )
}

export function CopyText({ value }: { value: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button type="button" aria-label="Copy" className="text-ink-3 hover:text-ink-1" onClick={() => {
      navigator.clipboard?.writeText(value); setCopied(true); setTimeout(() => setCopied(false), 1000)
    }}>
      {copied ? <Check size={13} /> : <Copy size={13} />}
    </button>
  )
}

/** Signature state + algorithm + key alias. Click opens the record in the inspector when a ledger seq is known. */
export function SignatureBadge({ state = 'signed', alg = 'Ed25519', keyId, ledgerSeq, compact }: {
  state?: 'signed' | 'verified' | 'failed' | 'revoked'; alg?: string; keyId?: string; ledgerSeq?: number; compact?: boolean
}) {
  const inspect = useUi((s) => s.inspect)
  const ok = state === 'signed' || state === 'verified'
  const Icon = ok ? ShieldCheck : ShieldX
  const color = ok ? 'var(--good-ink)' : 'var(--critical-ink)'
  const alias = keyId?.replace('kms:spectrum/', '')
  return (
    <button
      type="button"
      onClick={(e) => { e.stopPropagation(); if (ledgerSeq) inspect({ kind: 'ledger', id: String(ledgerSeq) }) }}
      title={`${state === 'verified' ? 'Signature verified' : state === 'failed' ? 'Signature check failed' : state === 'revoked' ? 'Signing key revoked' : 'Signed'}\n${alg}${keyId ? `\n${keyId}` : ''}${ledgerSeq ? `\nLedger #${ledgerSeq}` : ''}`}
      className="inline-flex max-w-full items-center gap-1 rounded px-0.5 text-cap hover:bg-s2"
    >
      <Icon size={13} style={{ color }} aria-hidden />
      {!compact && <span style={{ color }}>{state === 'verified' ? 'Verified' : state === 'failed' ? 'Failed' : state === 'revoked' ? 'Revoked' : 'Signed'}</span>}
      {!compact && alias && <span className="mono truncate text-ink-3">{alias}</span>}
    </button>
  )
}

export function LedgerLink({ seq }: { seq?: number }) {
  const inspect = useUi((s) => s.inspect)
  if (!seq) return null
  return (
    <button type="button" onClick={(e) => { e.stopPropagation(); inspect({ kind: 'ledger', id: String(seq) }) }}
      className="tnum mono rounded px-0.5 text-ink-2 hover:bg-s2 hover:text-ink-1" title={`Open ledger record #${seq}`}>
      #{seq.toLocaleString()}
    </button>
  )
}

// ------------------------------------------------------------------ figures

export function Stat({ label, value, unit, delta, deltaGood, trend, trendColor, hint, className, flashKey }: {
  label: ReactNode; value: ReactNode; unit?: string; delta?: string; deltaGood?: boolean | null; trend?: number[]
  trendColor?: string; hint?: ReactNode; className?: string; flashKey?: unknown
}) {
  const ref = useRef<HTMLDivElement>(null)
  const first = useRef(true)
  useEffect(() => {
    if (first.current) { first.current = false; return }
    const el = ref.current
    if (!el) return
    el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash')
  }, [flashKey])
  return (
    <div ref={ref} className={clsx('min-w-0 rounded-md', className)}>
      <div className="truncate text-cap text-ink-3">{label}</div>
      <div className="mt-0.5 flex items-baseline gap-1.5">
        <span className="text-[22px] leading-7 font-semibold tracking-[-0.01em] text-ink-1">{value}</span>
        {unit && <span className="text-cap text-ink-3">{unit}</span>}
        {delta && (
          <span className="tnum text-cap font-medium" style={{ color: deltaGood == null ? 'var(--ink-2)' : deltaGood ? 'var(--good-ink)' : 'var(--critical-ink)' }}>
            {delta}
          </span>
        )}
      </div>
      {trend && trend.length > 1 && <Sparkline values={trend} color={trendColor ?? 'var(--ink-3)'} height={22} className="mt-1" />}
      {hint && <div className="mt-0.5 text-micro text-ink-3">{hint}</div>}
    </div>
  )
}

export function Meter({ value, max = 1, warnAt, critAt, label, right, color = 'var(--release)', height = 6 }: {
  value: number; max?: number; warnAt?: number; critAt?: number; label?: ReactNode; right?: ReactNode; color?: string; height?: number
}) {
  const k = Math.max(0, Math.min(1, value / max))
  const c = critAt != null && value >= critAt ? 'var(--critical)' : warnAt != null && value >= warnAt ? 'var(--warn)' : color
  return (
    <div className="min-w-0">
      {(label || right) && (
        <div className="mb-1 flex items-baseline justify-between gap-2 text-cap">
          <span className="truncate text-ink-2">{label}</span>
          <span className="tnum shrink-0 text-ink-1">{right}</span>
        </div>
      )}
      <div className="relative overflow-hidden rounded-full" style={{ height, background: `color-mix(in oklab, ${c} 18%, var(--surface-2))` }}>
        <div className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-500" style={{ width: `${k * 100}%`, background: c }} />
        {warnAt != null && <div className="absolute inset-y-0 w-px bg-ink-3/60" style={{ left: `${(warnAt / max) * 100}%` }} />}
      </div>
    </div>
  )
}

export function Steps({ steps }: { steps: { label: string; state: 'done' | 'active' | 'pending' | 'failed'; detail?: ReactNode }[] }) {
  return (
    <ol className="flex w-full items-stretch gap-1">
      {steps.map((s, i) => (
        <li key={s.label} className="min-w-0 flex-1">
          <div className="h-1 rounded-full" style={{
            background: s.state === 'done' ? 'var(--good)' : s.state === 'active' ? 'var(--release)' : s.state === 'failed' ? 'var(--critical)' : 'var(--surface-3)',
          }} />
          <div className="mt-1.5 flex items-baseline gap-1 text-cap">
            <span className="tnum text-ink-3">{i + 1}</span>
            <span className={clsx('truncate', s.state === 'pending' ? 'text-ink-3' : 'text-ink-1')}>{s.label}</span>
          </div>
          {s.detail && <div className="truncate text-micro text-ink-3">{s.detail}</div>}
        </li>
      ))}
    </ol>
  )
}

export function EmptyState({ title, body, action }: { title: string; body?: string; action?: ReactNode }) {
  return (
    <div className="grid h-full min-h-24 place-items-center p-6 text-center">
      <div>
        <div className="text-body font-medium text-ink-2">{title}</div>
        {body && <div className="mt-1 max-w-[42ch] text-cap text-ink-3">{body}</div>}
        {action && <div className="mt-3">{action}</div>}
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ code

const CEDAR_KW = /\b(permit|forbid|when|unless|principal|action|resource|context|in|has|like|if|then|else)\b/g

export function CodeBlock({ code, lang = 'json', className, maxHeight }: { code: string; lang?: 'json' | 'cedar' | 'text' | 'yaml'; className?: string; maxHeight?: number }) {
  const html = highlight(code, lang)
  return (
    <pre
      className={clsx('mono overflow-auto rounded-md border border-line bg-s2 p-3 text-[12px] leading-[18px] text-ink-1', className)}
      style={{ maxHeight }}
      // highlight() escapes all input before inserting spans
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}

function esc(s: string) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function highlight(code: string, lang: string): string {
  const e = esc(code)
  if (lang === 'json') {
    return e
      .replace(/("(?:[^"\\]|\\.)*")(\s*:)/g, '<span style="color:var(--ink-2)">$1</span>$2')
      .replace(/:\s*("(?:[^"\\]|\\.)*")/g, ': <span style="color:var(--good-ink)">$1</span>')
      .replace(/\b(true|false|null)\b/g, '<span style="color:var(--serious)">$1</span>')
  }
  if (lang === 'cedar') {
    return e
      .replace(/("(?:[^"\\]|\\.)*")/g, '<span style="color:var(--good-ink)">$1</span>')
      .replace(CEDAR_KW, '<span style="color:var(--learn);font-weight:500">$1</span>')
      .replace(/(\w+::)/g, '<span style="color:var(--ink-2)">$1</span>')
  }
  return e
}

export function KV({ rows, className }: { rows: [ReactNode, ReactNode][]; className?: string }) {
  return (
    <dl className={clsx('grid grid-cols-[minmax(110px,auto)_1fr] gap-x-4 gap-y-1.5 text-cap', className)}>
      {rows.map(([k, v], i) => (
        <div key={i} className="contents">
          <dt className="text-ink-3">{k}</dt>
          <dd className="min-w-0 text-ink-1">{v}</dd>
        </div>
      ))}
    </dl>
  )
}
