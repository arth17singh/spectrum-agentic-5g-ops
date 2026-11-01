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
