import clsx from 'clsx'
import {
  Circle, FileText, FlaskConical, Gauge, KeyRound, MessagesSquare, Network, RefreshCw, Rocket, ShieldCheck, ShieldX,
  Siren, Snowflake, TriangleAlert, UserCheck, Wrench, Anchor, Loader2, FileCheck2, FileWarning, FileX2,
} from 'lucide-react'
import { useEffect, useLayoutEffect, useRef } from 'react'
import { shortHash } from '@/core/format'
import { planeColor } from '@/core/palette'
import type { LedgerRecord } from '@/core/types'

// Horizontally scrolling hash chain: each block is one signed ledger record; the connector between
// blocks carries the previous hash it commits to. A scan cursor sweeps across while verifying.

export type BlockState = 'ok' | 'fail' | 'pending' | 'unverified' | 'idle'

export const TYPE_GLYPH: [string, typeof Circle][] = [
  ['tool.call', Wrench], ['a2a.', MessagesSquare], ['approval.', UserCheck], ['escalation', TriangleAlert],
  ['fl.', Network], ['eval.', FlaskConical], ['aibom', FlaskConical], ['release.', Rocket], ['identity.', KeyRound],
  ['incident.', Siren], ['dossier', FileText], ['evidence.artifact', FileCheck2], ['evidence.changed', FileWarning],
  ['evidence.missing', FileX2], ['evidence.', FileText], ['autonomy', Gauge], ['swarm.', Snowflake], ['loop.', RefreshCw],
]
export const glyphFor = (type: string) => TYPE_GLYPH.find(([p]) => type.startsWith(p))?.[1] ?? Circle

const BLOCK_W = 148
const GAP = 30

export function ChainStrip({ records, states, scan, selected, onSelect, anchorEvery = 32, follow = true }: {
  records: LedgerRecord[]
  states: Record<number, BlockState>
  /** 0..1 position of the verification cursor, or null when idle */
  scan: number | null
  selected?: number | null
  onSelect?: (seq: number) => void
  anchorEvery?: number
  /** keep the newest block in view as records arrive */
  follow?: boolean
}) {
  const scroller = useRef<HTMLDivElement>(null)
  const lastSeq = records[records.length - 1]?.seq
  const contentW = records.length * (BLOCK_W + GAP)

  useLayoutEffect(() => {
    const el = scroller.current
    if (el && follow) el.scrollLeft = el.scrollWidth
  }, [lastSeq, follow])

  // keep the cursor in view while scanning
  useEffect(() => {
    const el = scroller.current
    if (!el || scan == null) return
    const x = scan * contentW
    if (x < el.scrollLeft + 40 || x > el.scrollLeft + el.clientWidth - 40) el.scrollLeft = Math.max(0, x - el.clientWidth / 2)
  }, [scan, contentW])

  // focus a selected block (e.g. the broken record)
  useEffect(() => {
    const el = scroller.current
    if (!el || selected == null || follow) return
    const i = records.findIndex((r) => r.seq === selected)
    if (i >= 0) el.scrollLeft = Math.max(0, i * (BLOCK_W + GAP) - el.clientWidth / 2 + BLOCK_W / 2)
  }, [selected, records, follow])

  return (
    <div ref={scroller} className="relative overflow-x-auto pb-2" role="list" aria-label="Hash chain">
      <style>{`@keyframes chainin{from{transform:translateX(24px);opacity:0}to{transform:none;opacity:1}}.chain-in{animation:chainin 320ms cubic-bezier(.2,.8,.2,1)}`}</style>
      <div className="relative flex items-stretch py-2" style={{ width: contentW + 8 }}>
        {records.map((r, i) => {
          const st = states[r.seq] ?? 'idle'
          const G = glyphFor(r.type)
          const anchored = r.seq % anchorEvery === 0
          return (
            <div key={r.seq} className={clsx('flex shrink-0 items-center', i === records.length - 1 && follow && 'chain-in')} role="listitem">
              {i > 0 && (
                <div className="relative flex flex-col items-center justify-center" style={{ width: GAP }}>
                  <svg width={GAP} height="10" aria-hidden>
                    <line x1="0" x2={GAP - 5} y1="5" y2="5" style={{ stroke: st === 'fail' || states[records[i - 1].seq] === 'fail' ? 'var(--critical)' : 'var(--axis)' }} strokeWidth="1.5" />
                    <path d={`M${GAP - 6},1 L${GAP - 1},5 L${GAP - 6},9`} fill="none" style={{ stroke: 'var(--axis)' }} strokeWidth="1.5" />
                  </svg>
                </div>
              )}
              <button
                type="button"
                onClick={() => onSelect?.(r.seq)}
                title={`#${r.seq} ${r.type}\nby ${r.actor}\nhash ${r.hash}\nprev ${r.prev_hash}`}
                className={clsx('relative flex flex-col gap-1 rounded-md border px-2.5 py-2 text-left transition-colors',
                  selected === r.seq ? 'bg-s3' : 'bg-s2 hover:bg-s3',
                  st === 'fail' ? 'border-critical' : st === 'ok' ? 'border-good/60' : 'border-line')}
                style={{ width: BLOCK_W }}
              >
                <span className="absolute inset-x-0 top-0 h-[2px] rounded-t-md" style={{ background: planeColor(r.plane) }} />
                <span className="flex items-center gap-1.5">
                  <span className="tnum text-cap font-semibold text-ink-1">#{r.seq.toLocaleString()}</span>
                  {anchored && <Anchor size={11} className="text-ink-3" aria-label="anchored here" />}
                  <span className="ml-auto">
                    {st === 'ok' ? <ShieldCheck size={13} className="text-good-ink" aria-label="verified" />
                      : st === 'fail' ? <ShieldX size={13} className="text-critical-ink" aria-label="tampered" />
                        : st === 'pending' ? <Loader2 size={13} className="animate-spin text-ink-3" aria-label="checking" />
                          : st === 'unverified' ? <Circle size={12} className="text-ink-3" aria-label="not verified past the break" />
                            : <ShieldCheck size={13} className="text-ink-3" aria-label="signed" />}
                  </span>
                </span>
                <span className="flex min-w-0 items-center gap-1.5 text-micro text-ink-2">
                  <G size={12} className="shrink-0 text-ink-3" />
                  <span className="truncate">{r.type}</span>
                </span>
                <span className="mono truncate text-[11px] leading-4 text-ink-1">{shortHash(r.hash, 8, 4)}</span>
                <span className="mono truncate text-[10px] leading-3 text-ink-3">prev {shortHash(r.prev_hash, 6, 0).replace('…', '')}</span>
              </button>
            </div>
          )
        })}
        {scan != null && (
          <div aria-hidden className="pointer-events-none absolute top-0 bottom-0 w-[2px] rounded-full bg-release shadow-[0_0_12px_2px_color-mix(in_oklab,var(--release)_60%,transparent)]"
            style={{ left: Math.min(contentW, scan * contentW) }} />
        )}
      </div>
    </div>
  )
}
