import clsx from 'clsx'
import { ChevronDown, ChevronUp, CircleAlert, Info, OctagonX, TriangleAlert, Dot, ArrowUpRight } from 'lucide-react'
import { useMemo, useState } from 'react'
import { clockTime } from '@/core/format'
import { PLANES, planeColor } from '@/core/palette'
import { useLive } from '@/core/store'
import type { Envelope, Severity } from '@/core/types'
import { useUi } from '@/core/ui-store'

const SEV_ORDER: Severity[] = ['debug', 'info', 'notice', 'warn', 'critical']
export const SEV_ICON: Record<Severity, { Icon: typeof Info; color: string }> = {
  debug: { Icon: Dot, color: 'var(--ink-3)' },
  info: { Icon: Info, color: 'var(--ink-3)' },
  notice: { Icon: CircleAlert, color: 'var(--release)' },
  warn: { Icon: TriangleAlert, color: 'var(--warn)' },
  critical: { Icon: OctagonX, color: 'var(--critical-ink)' },
}

export function SeverityIcon({ s }: { s: Severity }) {
  const { Icon, color } = SEV_ICON[s]
  return <Icon size={13} style={{ color }} aria-label={s} className="shrink-0" />
}

export function StreamDrawer() {
  const open = useUi((s) => s.streamOpen)
  const set = useUi((s) => s.set)
  const inspect = useUi((s) => s.inspect)
  const stream = useLive((s) => s.stream)
  const [planes, setPlanes] = useState<string[]>([])
  const [minSev, setMinSev] = useState<Severity>('info')
  const [q, setQ] = useState('')
  const [ledgeredOnly, setLedgeredOnly] = useState(false)

  const rows = useMemo(() => {
    const min = SEV_ORDER.indexOf(minSev)
    const qq = q.trim().toLowerCase()
    const out: Envelope[] = []
    for (let i = stream.length - 1; i >= 0 && out.length < 400; i--) {
      const e = stream[i]
      if (SEV_ORDER.indexOf(e.severity) < min) continue
      if (planes.length && !planes.includes(e.plane)) continue
      if (ledgeredOnly && !e.ledger) continue
      if (qq && !(`${e.actor} ${e.type} ${e.summary}`.toLowerCase().includes(qq))) continue
      out.push(e)
    }
    return out
  }, [stream, planes, minSev, q, ledgeredOnly])

  const latest = useMemo(() => {
    for (let i = stream.length - 1; i >= 0; i--) if (stream[i].severity !== 'debug') return stream[i]
    return null
  }, [stream])

  return (
    <section aria-label="Live stream" className={clsx('shrink-0 border-t border-line bg-page transition-[height] duration-200', open ? 'h-[280px]' : 'h-8')}>
      <div className="flex h-8 items-center gap-3 px-3 text-cap">
        <button type="button" onClick={() => set({ streamOpen: !open })} className="flex items-center gap-1.5 font-medium text-ink-1">
          {open ? <ChevronDown size={14} /> : <ChevronUp size={14} />} Live stream
        </button>
        {!open && latest && (
          <button type="button" key={latest.seq} onClick={() => set({ streamOpen: true })} className="flex min-w-0 items-center gap-2 text-left">
            <SeverityIcon s={latest.severity} />
            <span className="tnum mono text-ink-3">{clockTime(latest.ts)}</span>
            <PlaneChip plane={latest.plane} />
            <span className="mono shrink-0 text-ink-2">{latest.actor}</span>
            <span className="truncate text-ink-1">{latest.summary}</span>
          </button>
        )}
        {open && (
          <div className="flex min-w-0 flex-1 items-center gap-2">
            <div className="flex gap-1">
              {PLANES.map((p) => (
                <button key={p.id} type="button" onClick={() => setPlanes((x) => (x.includes(p.id) ? x.filter((y) => y !== p.id) : [...x, p.id]))}
                  className={clsx('flex h-6 items-center gap-1 rounded border px-1.5 text-micro', planes.includes(p.id) ? 'border-ink-3 text-ink-1' : 'border-line text-ink-3')}>
                  <span className="size-1.5 rounded-full" style={{ background: p.color }} />{p.label}
                </button>
              ))}
            </div>
            <select value={minSev} onChange={(e) => setMinSev(e.target.value as Severity)} aria-label="Minimum severity" className="h-6 rounded border border-line bg-s1 px-1 text-micro text-ink-1">
              {SEV_ORDER.map((s) => <option key={s} value={s}>severity ≥ {s}</option>)}
            </select>
            <label className="flex items-center gap-1 text-micro text-ink-2">
              <input type="checkbox" checked={ledgeredOnly} onChange={(e) => setLedgeredOnly(e.target.checked)} /> only ledgered
            </label>
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter by actor, type, text" className="h-6 w-56 rounded border border-line bg-s1 px-2 text-micro text-ink-1 placeholder:text-ink-3" />
            <span className="tnum ml-auto text-micro text-ink-3">{rows.length} shown, {stream.length.toLocaleString()} buffered</span>
          </div>
        )}
      </div>
      {open && (
        <div className="h-[248px] overflow-auto px-3 pb-2">
          <table className="w-full text-cap">
            <tbody>
              {rows.map((e) => (
                <tr key={e.seq} className="row-in group border-b border-line/50 hover:bg-s1">
                  <td className="w-5 py-1"><SeverityIcon s={e.severity} /></td>
                  <td className="tnum mono w-[96px] text-ink-3">{clockTime(e.ts)}</td>
                  <td className="w-[78px]"><PlaneChip plane={e.plane} /></td>
                  <td className="mono w-[130px] truncate pr-2 text-ink-2">{e.actor}</td>
                  <td className="mono w-[150px] truncate pr-2 text-ink-3">{e.type}</td>
                  <td className="truncate pr-2 text-ink-1">{e.summary}</td>
                  <td className="w-16 text-right">
                    {e.ledger && (
                      <button type="button" onClick={() => inspect({ kind: 'ledger', id: String(e.ledger!.seq) })} className="mono text-ink-3 hover:text-ink-1">#{e.ledger.seq}</button>
                    )}
                  </td>
                  <td className="w-6">
                    <button type="button" aria-label="Inspect event" onClick={() => inspect({ kind: 'raw', id: `event ${e.seq}`, title: e.summary, data: e })} className="opacity-0 group-hover:opacity-100 text-ink-3 hover:text-ink-1">
                      <ArrowUpRight size={13} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

export function PlaneChip({ plane }: { plane: string }) {
  if (plane === 'system') return <span className="text-micro text-ink-3">system</span>
  return (
    <span className="inline-flex items-center gap-1 text-micro text-ink-2">
      <span className="size-1.5 rounded-full" style={{ background: planeColor(plane) }} />
      {plane}
    </span>
  )
}
