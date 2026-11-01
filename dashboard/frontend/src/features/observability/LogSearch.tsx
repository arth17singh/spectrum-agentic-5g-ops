import { Search } from 'lucide-react'
import { useMemo, useState } from 'react'
import { PlaneChip, SeverityIcon } from '@/app/StreamDrawer'
import { clockTime } from '@/core/format'
import { useLive } from '@/core/store'
import type { Envelope, Severity } from '@/core/types'
import { useUi } from '@/core/ui-store'
import { Panel } from '@/ui'

const SEV: Severity[] = ['debug', 'info', 'notice', 'warn', 'critical']
const EXAMPLES = ['plane:operate severity>=warn', 'actor:site-17', 'type:policy.decision deny', 'plane:learn poison', 'severity>=critical']

interface Query { planes: string[]; actors: string[]; types: string[]; minSev: number; sevExact: number | null; text: string[] }

/** Tiny query language: plane:x actor:x type:x severity>=warn severity:critical, plus free text (AND). */
export function parseQuery(q: string): Query {
  const out: Query = { planes: [], actors: [], types: [], minSev: 0, sevExact: null, text: [] }
  for (const tok of q.trim().split(/\s+/).filter(Boolean)) {
    const m = tok.match(/^(plane|actor|type|severity|sev)(:|>=)(.+)$/i)
    if (!m) { out.text.push(tok.toLowerCase()); continue }
    const [, key, op, val] = m
    const k = key.toLowerCase()
    if (k === 'plane') out.planes.push(val.toLowerCase())
    else if (k === 'actor') out.actors.push(val.toLowerCase())
    else if (k === 'type') out.types.push(val.toLowerCase())
    else {
      const i = SEV.indexOf(val.toLowerCase() as Severity)
      if (i >= 0) { if (op === '>=') out.minSev = i; else out.sevExact = i }
    }
  }
  return out
}

function matches(e: Envelope, q: Query) {
  const si = SEV.indexOf(e.severity)
  if (si < q.minSev) return false
  if (q.sevExact != null && si !== q.sevExact) return false
  if (q.planes.length && !q.planes.includes(e.plane)) return false
  if (q.actors.length && !q.actors.some((a) => e.actor.toLowerCase().includes(a))) return false
  if (q.types.length && !q.types.some((t) => e.type.toLowerCase().startsWith(t))) return false
  const hay = `${e.actor} ${e.type} ${e.summary}`.toLowerCase()
  return q.text.every((t) => hay.includes(t))
}

export function LogSearch() {
  const stream = useLive((s) => s.stream)
  const inspect = useUi((s) => s.inspect)
  const [q, setQ] = useState('plane:operate severity>=warn')
  const parsed = useMemo(() => parseQuery(q), [q])
  const rows = useMemo(() => {
    const out: Envelope[] = []
    for (let i = stream.length - 1; i >= 0 && out.length < 200; i--) if (matches(stream[i], parsed)) out.push(stream[i])
    return out
  }, [stream, parsed])
  return (
    <Panel id="logs" title="Log search" subtitle={`${stream.length.toLocaleString()} events buffered in this browser`} source={{ aws: 'CloudWatch Logs Insights', mock: 'Live event stream' }} flush>
      <div className="flex flex-wrap items-center gap-2 border-b border-line px-4 py-3">
        <label className="flex h-8 min-w-[360px] flex-1 items-center gap-2 rounded-md border border-line bg-s2 px-2.5">
          <Search size={14} className="text-ink-3" />
          <input value={q} onChange={(e) => setQ(e.target.value)} aria-label="Log query" spellCheck={false}
            className="mono flex-1 bg-transparent text-cap text-ink-1 outline-none placeholder:text-ink-3" placeholder="plane:operate severity>=warn actor:site-17 text" />
        </label>
        {EXAMPLES.map((x) => (
          <button key={x} type="button" onClick={() => setQ(x)} className="mono h-7 rounded border border-line px-2 text-micro text-ink-2 hover:bg-s2 hover:text-ink-1">{x}</button>
        ))}
        <span className="tnum ml-auto text-micro text-ink-3">{rows.length === 200 ? '200+' : rows.length} matches</span>
      </div>
      <div className="max-h-[300px] overflow-auto px-4 py-1">
        {!rows.length && <div className="py-6 text-center text-cap text-ink-3">No events match. Try one of the examples above, or widen the severity.</div>}
        <table className="w-full text-cap">
          <tbody>
            {rows.map((e) => (
              <tr key={e.seq} className="border-b border-line/50 hover:bg-s2/50">
                <td className="w-5 py-1"><SeverityIcon s={e.severity} /></td>
                <td className="tnum mono w-[96px] text-ink-3">{clockTime(e.ts)}</td>
                <td className="w-[76px]"><PlaneChip plane={e.plane} /></td>
                <td className="mono w-[130px] truncate pr-2 text-ink-2">{e.actor}</td>
                <td className="mono w-[150px] truncate pr-2 text-ink-3">{e.type}</td>
                <td className="truncate pr-2 text-ink-1">{e.summary}</td>
                <td className="w-16 text-right">
                  {e.ledger && <button type="button" onClick={() => inspect({ kind: 'ledger', id: String(e.ledger!.seq) })} className="mono text-ink-3 hover:text-ink-1">#{e.ledger.seq}</button>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}
