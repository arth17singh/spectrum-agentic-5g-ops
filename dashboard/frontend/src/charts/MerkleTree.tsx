import { useEffect, useState } from 'react'
import { shortHash } from '@/core/format'
import { useSize } from './common'

// Merkle inclusion proof, recomputed in the browser: leaf -> (sibling, parent) at each level -> root,
// compared against the root stored in the S3 Object Lock anchor.

export interface Proof {
  seq: number
  anchored: boolean
  leaf?: string
  index?: number
  path?: { hash: string; side: 'left' | 'right' }[]
  computed_root?: string
  anchor?: { n: number; root: string; range: [number, number]; s3_uri: string }
  matches?: boolean
  pending_until?: number
}

const hex = (h: string) => new Uint8Array(h.match(/../g)!.map((b) => parseInt(b, 16)))
async function pairHash(a: string, b: string) {
  const buf = new Uint8Array(64)
  buf.set(hex(a), 0)
  buf.set(hex(b), 32)
  const d = await crypto.subtle.digest('SHA-256', buf)
  return [...new Uint8Array(d)].map((x) => x.toString(16).padStart(2, '0')).join('')
}

export function useMerkleLevels(proof: Proof | null) {
  const [levels, setLevels] = useState<string[]>([])
  useEffect(() => {
    let alive = true
    if (!proof?.anchored || !proof.leaf || !proof.path) { setLevels([]); return }
    ;(async () => {
      const out = [proof.leaf!]
      let cur = proof.leaf!
      for (const p of proof.path!) {
        cur = p.side === 'left' ? await pairHash(p.hash, cur) : await pairHash(cur, p.hash)
        out.push(cur)
      }
      if (alive) setLevels(out)
    })()
    return () => { alive = false }
  }, [proof])
  return levels
}

export function MerkleTree({ proof }: { proof: Proof }) {
  const { ref, width } = useSize()
  const levels = useMerkleLevels(proof)
  const path = proof.path ?? []
  const n = path.length
  const rowH = 46
  const top = 56
  const H = top + (n + 1) * rowH + 12
  const nodeW = 112
  const nodeH = 26
  const cx = width / 2
  const root = levels[levels.length - 1]
  const ok = root && proof.anchor ? root === proof.anchor.root : null
  // level i (0 = leaf) sits at the bottom; current node and its sibling side by side
  const yOf = (i: number) => top + (n - i) * rowH + rowH / 2
  const off = Math.min(120, Math.max(70, width / 5))
  // x of the node on the proof path at level i (the root sits in the middle)
  const pathX = (i: number) => {
    const sib = path[i]
    if (!sib) return cx
    return sib.side === 'left' ? cx + off / 2 : cx - off / 2
  }
  return (
    <div ref={ref} style={{ height: H }} className="relative">
      {width > 0 && (
        <svg width={width} height={H} role="img" aria-label="Merkle inclusion proof">
          {/* anchored root vs recomputed root */}
          <g transform={`translate(${cx},${22})`}>
            <rect x={-nodeW - 10} y={-15} width={nodeW + 4} height={30} rx={5} style={{ fill: 'var(--surface-2)', stroke: ok == null ? 'var(--hairline)' : ok ? 'var(--good)' : 'var(--critical)' }} strokeWidth={1.5} />
            <text x={-nodeW / 2 - 8} y={-3} textAnchor="middle" style={{ fill: 'var(--ink-3)', fontSize: 10 }}>recomputed root</text>
            <text x={-nodeW / 2 - 8} y={10} textAnchor="middle" className="mono" style={{ fill: 'var(--ink-1)', fontSize: 11 }}>{shortHash(root, 7, 4)}</text>
            <text x={0} y={4} textAnchor="middle" style={{ fill: ok == null ? 'var(--ink-3)' : ok ? 'var(--good-ink)' : 'var(--critical-ink)', fontSize: 14, fontWeight: 600 }}>{ok == null ? '…' : ok ? '=' : '≠'}</text>
            <rect x={6} y={-15} width={nodeW + 4} height={30} rx={5} style={{ fill: 'var(--surface-2)', stroke: 'var(--prove)' }} strokeWidth={1.5} />
            <text x={nodeW / 2 + 8} y={-3} textAnchor="middle" style={{ fill: 'var(--ink-3)', fontSize: 10 }}>anchor #{proof.anchor?.n} (S3)</text>
            <text x={nodeW / 2 + 8} y={10} textAnchor="middle" className="mono" style={{ fill: 'var(--ink-1)', fontSize: 11 }}>{shortHash(proof.anchor?.root, 7, 4)}</text>
          </g>
          {levels.map((h, i) => {
            const y = yOf(i)
            const sib = path[i]
            const curX = sib ? (sib.side === 'left' ? cx + off / 2 : cx - off / 2) : cx
            const sibX = sib ? (sib.side === 'left' ? cx - off / 2 : cx + off / 2) : cx
            const isLeaf = i === 0
            const isRoot = i === levels.length - 1
            return (
              <g key={i}>
                {sib && (
                  <>
                    <path d={`M${curX},${y - nodeH / 2} L${pathX(i + 1)},${yOf(i + 1) + nodeH / 2}`} style={{ stroke: 'var(--release)' }} strokeWidth={1.5} fill="none" />
                    <path d={`M${sibX},${y - nodeH / 2} L${pathX(i + 1)},${yOf(i + 1) + nodeH / 2}`} style={{ stroke: 'var(--axis)' }} strokeWidth={1.5} fill="none" />
                    <g transform={`translate(${sibX - nodeW / 2},${y - nodeH / 2})`}>
                      <rect width={nodeW} height={nodeH} rx={4} style={{ fill: 'var(--surface-2)', stroke: 'var(--hairline)' }} />
                      <text x={nodeW / 2} y={nodeH / 2} dy="0.32em" textAnchor="middle" className="mono" style={{ fill: 'var(--ink-2)', fontSize: 11 }}>{shortHash(sib.hash, 6, 4)}</text>
                      <title>{`sibling (${sib.side}) ${sib.hash}`}</title>
                    </g>
                  </>
                )}
                <g transform={`translate(${(sib ? curX : cx) - nodeW / 2},${y - nodeH / 2})`}>
                  <rect width={nodeW} height={nodeH} rx={4}
                    style={{ fill: isLeaf ? 'color-mix(in oklab, var(--release) 22%, var(--surface-2))' : 'var(--surface-2)', stroke: isRoot ? (ok ? 'var(--good)' : ok === false ? 'var(--critical)' : 'var(--release)') : 'var(--release)' }} strokeWidth={1.5} />
                  <text x={nodeW / 2} y={nodeH / 2} dy="0.32em" textAnchor="middle" className="mono" style={{ fill: 'var(--ink-1)', fontSize: 11 }}>{shortHash(h, 6, 4)}</text>
                  <title>{`${isLeaf ? `leaf #${proof.seq}` : isRoot ? 'root' : `level ${i}`} ${h}`}</title>
                </g>
                <text x={12} y={y} dy="0.32em" style={{ fill: 'var(--ink-3)', fontSize: 10 }}>{isLeaf ? `leaf #${proof.seq}` : isRoot ? 'root' : `level ${i}`}</text>
              </g>
            )
          })}
        </svg>
      )}
    </div>
  )
}
