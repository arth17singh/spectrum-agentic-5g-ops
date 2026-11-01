// Cross-language check: the browser canonicaliser must hash ledger records exactly like Python does.
import { readFileSync } from 'node:fs'
import { createHash } from 'node:crypto'

function canonical(v) {
  if (v === null || typeof v !== 'object') return JSON.stringify(v)
  if (Array.isArray(v)) return `[${v.map(canonical).join(',')}]`
  return `{${Object.keys(v).sort().map((k) => `${JSON.stringify(k)}:${canonical(v[k])}`).join(',')}}`
}
const { records } = JSON.parse(readFileSync(new URL('../../tests/fixtures/canonical.json', import.meta.url)))
let bad = 0
for (const r of records) {
  const body = { seq: r.seq, ts: r.ts, type: r.type, actor: r.actor, plane: r.plane, payload: r.payload, key_id: r.key_id }
  const h = createHash('sha256').update(canonical(body) + r.prev_hash, 'utf8').digest('hex')
  const ok = h === r.hash
  if (!ok) bad++
  console.log(`${ok ? 'ok  ' : 'FAIL'} #${r.seq} ${r.type}`)
}
process.exit(bad ? 1 : 0)
