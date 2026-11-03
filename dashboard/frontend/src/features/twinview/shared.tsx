import { FlaskConical } from 'lucide-react'
import { RealChip } from '@/ui'
import { amber, type Prov } from './format'

/** REAL (green) only when every displayed record has an AWS time and none is synthetic; anything else is amber. */
export function ProvenanceChip({ p, tag }: { p: Prov; tag?: string }) {
  if (p.state === 'empty') return null
  if (p.state === 'real') return <RealChip title={`${p.note} Read from the persisted evidence records.`} />
  return (
    <span data-provenance={p.state} title={p.note}
      className="flex min-w-0 items-center gap-1 rounded border px-1.5 py-0.5 text-micro font-semibold tracking-[0.04em]" style={amber}>
      <FlaskConical size={11} aria-hidden className="shrink-0" />
      <span className="truncate">{tag ? `${tag} ` : ''}{p.label}</span>
    </span>
  )
}

/** An amber label for things that are not results at all (generated datasets, local previews). */
export function NoteChip({ label, title }: { label: string; title: string }) {
  return (
    <span title={title} className="flex shrink-0 items-center gap-1 rounded border px-1.5 py-0.5 text-micro font-semibold tracking-[0.04em]" style={amber}>
      <FlaskConical size={11} aria-hidden className="shrink-0" />{label}
    </span>
  )
}

export function Banner({ p, extra }: { p: Prov; extra?: string }) {
  if (p.state === 'real' || p.state === 'empty') return null
  return (
    <div className="mb-4 flex items-start gap-2 rounded-md border px-4 py-2.5 text-cap text-ink-1" role="status" data-provenance={p.state}
      style={{ borderColor: amber.borderColor, background: amber.background }}>
      <FlaskConical size={14} className="mt-0.5 shrink-0" style={{ color: 'var(--warn-ink)' }} aria-hidden />
      <span><span className="font-medium">{p.label}: {p.note}</span> Do not read these numbers as a measured result.{extra ? ` ${extra}` : ''}
        {(p.unverified_reasons ?? []).map((r) => <span key={r.reason} className="mt-0.5 block text-ink-2" data-testid="unverified-reason">{r.n}: {r.reason}</span>)}</span>
    </div>
  )
}
