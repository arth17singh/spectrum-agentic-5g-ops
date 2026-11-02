// Browser twin of backend/spectrum/core.py canonical(): sorted keys, no whitespace, UTF-8.
// Ledger payloads never contain floats (the backend stores them as strings), so JSON.stringify
// number formatting can't diverge. Verified against tests/fixtures/canonical.json.

export function canonical(v: unknown): string {
  if (v === null || typeof v !== 'object') return JSON.stringify(v)
  if (Array.isArray(v)) return `[${v.map(canonical).join(',')}]`
  const o = v as Record<string, unknown>
  const keys = Object.keys(o).sort()
  return `{${keys.map((k) => `${JSON.stringify(k)}:${canonical(o[k])}`).join(',')}}`
}

export async function sha256Hex(text: string): Promise<string> {
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text))
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('')
}

/** Recompute a ledger record hash exactly as the backend does: SHA-256(canonical(body) || prev_hash). */
export async function recordHash(rec: {
  seq: number; ts: string; type: string; actor: string; plane: string; payload: unknown; key_id: string; prev_hash: string
}): Promise<string> {
  const body = { seq: rec.seq, ts: rec.ts, type: rec.type, actor: rec.actor, plane: rec.plane, payload: rec.payload,
    key_id: rec.key_id }
  return sha256Hex(canonical(body) + rec.prev_hash)
}

const hexBytes = (h: string) => new Uint8Array((h.match(/../g) ?? []).map((b) => parseInt(b, 16)))

/** Ed25519 support in WebCrypto (Chrome 137+, Safari 17+, Firefox 129+). Feature-detected, never assumed. */
export async function ed25519Supported(): Promise<boolean> {
  try {
    await crypto.subtle.importKey('raw', new Uint8Array(32), { name: 'Ed25519' }, false, ['verify'])
    return true
  } catch {
    return false
  }
}

/** Verify an Ed25519 signature (hex) over a UTF-8 message with a raw public key (hex), as the backend signs. */
export async function ed25519Verify(publicKeyHex: string, message: string, sigHex: string): Promise<boolean> {
  try {
    const key = await crypto.subtle.importKey('raw', hexBytes(publicKeyHex), { name: 'Ed25519' }, false, ['verify'])
    return await crypto.subtle.verify({ name: 'Ed25519' }, key, hexBytes(sigHex), new TextEncoder().encode(message))
  } catch {
    return false
  }
}

/** Merkle root exactly as backend merkle_levels(): SHA-256(left || right) over raw bytes, odd node paired with itself. */
export async function merkleRoot(leaves: string[]): Promise<string> {
  let cur = leaves
  while (cur.length > 1) {
    const nxt: string[] = []
    for (let i = 0; i < cur.length; i += 2) {
      const l = hexBytes(cur[i])
      const r = hexBytes(cur[i + 1] ?? cur[i])
      const buf = new Uint8Array(64)
      buf.set(l, 0)
      buf.set(r, 32)
      const d = await crypto.subtle.digest('SHA-256', buf)
      nxt.push([...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, '0')).join(''))
    }
    cur = nxt
  }
  return cur[0] ?? ''
}
