import { useCallback, useRef, useState } from 'react'
import { api } from '@/core/api'
import { ed25519Supported, ed25519Verify, merkleRoot, recordHash } from '@/core/canonical'
import type { LedgerRecord, Rec } from '@/core/types'

// Independent check of the evidence ledger in this browser: fetch every record as stored on disk,
// then walk from genesis recomputing each SHA-256 (canonical.ts, the same code the Audit screen
// uses), checking each link, each Ed25519 signature against the PUBLISHED public key, and every
// Merkle anchor root. Stops at the first failure and reports its exact seq.

const GENESIS = '0'.repeat(64)

export interface BrowserVerify {
  phase: 'idle' | 'running' | 'ok' | 'broken' | 'error'
  checked: number
  total: number
  ms: number
  /** 'verified' | 'unsupported' (this browser has no WebCrypto Ed25519: signatures were not checked here) */
  signatures: 'verified' | 'unsupported' | null
  anchors: number
  broken: { seq: number; reason: string; stored?: string; recomputed?: string } | null
  /** seq -> ok, for every record checked (and the failing one) */
  states: Record<number, 'ok' | 'fail' | 'unverified'>
  error?: string
}

const INITIAL: BrowserVerify = { phase: 'idle', checked: 0, total: 0, ms: 0, signatures: null, anchors: 0, broken: null, states: {} }

export function useBrowserVerify() {
  const [s, setS] = useState<BrowserVerify>(INITIAL)
  const busy = useRef(false)
  const run = useCallback(async () => {
    if (busy.current) return
    busy.current = true
    try {
      setS({ ...INITIAL, phase: 'running' })
      const chain: Rec = await api.evidenceChain()
      const records: LedgerRecord[] = chain.records ?? []
      const anchors: Rec[] = chain.anchors ?? []
      const every: number = chain.anchor_every ?? 32
      const keys: Record<string, Rec> = chain.public_keys?.keys ?? {}
      const sigOk = await ed25519Supported()
      const t0 = performance.now()
      const states: BrowserVerify['states'] = {}
      let prev = GENESIS
      let broken: BrowserVerify['broken'] = null
      for (let i = 0; i < records.length; i++) {
        const r = records[i]
        const seq = i + 1
        let reason: string | null = null
        let recomputed: string | undefined
        if (r.seq !== seq) reason = `expected record #${seq}, found #${r.seq}`
        else {
          recomputed = await recordHash(r)
          if (recomputed !== r.hash) reason = 'contents changed after signing (hash mismatch)'
          else if (r.prev_hash !== prev) reason = 'link to the previous record is broken'
          else if (sigOk) {
            const pk = keys[r.key_id]?.public_key_hex
            if (!pk || !(await ed25519Verify(pk, r.hash, r.sig))) reason = 'signature does not verify with the published public key'
          }
        }
        if (reason) {
          broken = { seq, reason, stored: r.hash, recomputed }
          states[seq] = 'fail'
          for (const later of records.slice(i + 1)) states[later.seq] = 'unverified'
          break
        }
        states[seq] = 'ok'
        prev = r.hash
        if (i % 200 === 199) setS((p) => ({ ...p, checked: seq, total: records.length }))
      }
      let anchorsChecked = 0
      if (!broken) {
        for (const a of anchors) {
          const [start, end] = a.range as [number, number]
          const leaves = records.slice(start - 1, end).map((r) => r.hash)
          if (leaves.length !== every || (await merkleRoot(leaves)) !== a.root) {
            broken = { seq: end, reason: `Merkle anchor #${a.n} does not match records ${start}–${end}` }
            break
          }
          anchorsChecked++
        }
      }
      setS({
        phase: broken ? 'broken' : 'ok', checked: broken ? broken.seq - 1 : records.length, total: records.length,
        ms: performance.now() - t0, signatures: sigOk ? 'verified' : 'unsupported', anchors: anchorsChecked, broken, states,
      })
    } catch (e) {
      setS({ ...INITIAL, phase: 'error', error: String(e) })
    } finally {
      busy.current = false
    }
  }, [])
  const reset = useCallback(() => setS(INITIAL), [])
  return { ...s, run, reset }
}
