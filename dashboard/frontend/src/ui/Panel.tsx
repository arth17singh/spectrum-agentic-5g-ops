import clsx from 'clsx'
import { FlaskConical, Table2 } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import { useLive } from '@/core/store'
import { useUi } from '@/core/ui-store'
import { planeColor } from '@/core/palette'

export interface PanelProps {
  /** stable id used by the Demo Director spotlight */
  id?: string
  title: ReactNode
  subtitle?: ReactNode
  /** where the data comes from now, and the AWS service it maps to */
  source?: { aws: string; mock?: string }
  plane?: string
  actions?: ReactNode
  className?: string
  bodyClassName?: string
  /** optional table twin of a chart; when provided a toggle appears in the header */
  table?: ReactNode
  children: ReactNode
  flush?: boolean
}

export function Panel({ id, title, subtitle, source, plane, actions, className, bodyClassName, table, children, flush }: PanelProps) {
  const lit = useUi((s) => (id ? s.lit.includes(id) : false))
  const [showTable, setShowTable] = useState(false)
  return (
    <section
      data-panel={id ?? ''}
      data-lit={lit ? 'true' : 'false'}
      style={{ ['--lit-color' as string]: plane ? planeColor(plane) : 'var(--focus)' }}
      className={clsx('flex min-h-0 min-w-0 flex-col rounded-[10px] border border-line bg-s1', className)}
    >
      <header className="flex h-10 shrink-0 items-center gap-3 border-b border-line px-4">
        <div className="flex min-w-0 items-baseline gap-2">
          <h3 className="shrink-0 text-title font-medium whitespace-nowrap text-ink-1">{title}</h3>
          {subtitle && <span className="truncate text-cap text-ink-3">{subtitle}</span>}
        </div>
        <div className="ml-auto flex shrink-0 items-center gap-2">
          {actions}
          {table && (
            <button
              type="button"
              onClick={() => setShowTable((v) => !v)}
              className={clsx('grid size-7 place-items-center rounded-md text-ink-3 hover:bg-s2 hover:text-ink-1',
                showTable && 'bg-s3 text-ink-1')}
              aria-label={showTable ? 'Show chart' : 'Show table'}
              title={showTable ? 'Show chart' : 'Show table'}
            >
              <Table2 size={15} />
            </button>
          )}
          {source && <SourceChip {...source} />}
        </div>
      </header>
      <div className={clsx('min-h-0 flex-1', !flush && 'p-4', bodyClassName)}>{showTable && table ? table : children}</div>
    </section>
  )
}

export function SourceChip({ aws, mock }: { aws: string; mock?: string }) {
  const mode = useLive((s) => s.mode)
  if (mode === 'aws')
    return <span className="rounded border border-line px-1.5 py-0.5 text-micro text-ink-2" title={`Live from ${aws}`}>{aws}</span>
  return (
    <span
      className="flex max-w-[220px] items-center gap-1 rounded border border-line px-1.5 py-0.5 text-micro text-ink-3"
      title={`${mock ?? 'Simulated by the mock engine'} now. In AWS mode this panel reads from ${aws}.`}
    >
      <FlaskConical size={11} aria-hidden className="shrink-0" />
      <span className="truncate text-ink-2">{aws}</span>
    </span>
  )
}

export function PageHeader({ title, description, tracks, actions }: {
  title: string; description: string; tracks?: string[]; actions?: ReactNode
}) {
  return (
    <div className="flex flex-wrap items-end gap-x-6 gap-y-2 pb-4">
      <div className="min-w-0">
        <div className="flex items-center gap-3">
          <h1 className="text-h1 font-semibold tracking-[-0.01em] text-ink-1">{title}</h1>
          <div className="flex gap-1.5">
            {tracks?.map((t) => <TrackTag key={t} track={t} />)}
          </div>
        </div>
        <p className="mt-0.5 max-w-[72ch] text-body text-ink-2">{description}</p>
      </div>
      {actions && <div className="ml-auto flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  )
}

const TRACK_TITLE: Record<string, string> = {
  T1: 'Track 1: NLM training, scaling & production readiness',
  T2: 'Track 2: Federated learning at scale',
  T3: 'Track 3: Multi-agent swarms with policy, audit and human oversight',
  Trust: 'Security & traceability',
  Obs: 'Observability',
}
const TRACK_COLOR: Record<string, string> = { T1: 'var(--verify)', T2: 'var(--learn)', T3: 'var(--operate)', Trust: 'var(--prove)', Obs: 'var(--release)' }

export function TrackTag({ track }: { track: string }) {
  return (
    <span
      title={TRACK_TITLE[track]}
      className="rounded-[4px] border px-1.5 text-micro font-medium leading-[18px]"
      style={{ borderColor: `color-mix(in oklab, ${TRACK_COLOR[track]} 60%, transparent)`, color: 'var(--ink-2)' }}
    >
      {track}
    </span>
  )
}
