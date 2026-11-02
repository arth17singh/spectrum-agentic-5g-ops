// Static check of the feature registry (no browser, no dependencies): every manifest under
// src/features/*/ must have a unique id, route and hotkey, and the Track 3 routes are reserved for
// their slice folders. The same rules run in the app at load (src/features/index.ts), where a
// clash logs a console error that the route smoke fails on.
// usage: node scripts/check-features.mjs [features-dir]    exit 0 = fine, 1 = problems (listed)
import { readdirSync, readFileSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = process.argv[2] ?? join(dirname(fileURLToPath(import.meta.url)), '..', 'src', 'features')
const RESERVED = { '/trace': 'trace', '/trust-gate': 'trustgate', '/scorecard': 'scorecard' }

const files = []
for (const d of readdirSync(root, { withFileTypes: true })) {
  if (!d.isDirectory()) continue
  for (const name of ['manifest.ts', 'dossierManifest.ts']) {
    const p = join(root, d.name, name)
    if (existsSync(p)) files.push({ folder: d.name, file: `${d.name}/${name}`, src: readFileSync(p, 'utf8') })
  }
}

const field = (src, key) => src.match(new RegExp(`\\b${key}:\\s*['"]([^'"]+)['"]`))?.[1]
const rows = files.map((f) => ({ ...f, id: field(f.src, 'id'), route: field(f.src, 'route'), hotkey: field(f.src, 'hotkey')?.toLowerCase() }))

const problems = []
for (const r of rows) if (!r.id || !r.route) problems.push(`${r.file}: no literal id/route found`)
for (const key of ['id', 'route', 'hotkey']) {
  const seen = new Map()
  for (const r of rows) {
    if (!r[key]) continue
    if (seen.has(r[key])) problems.push(`duplicate ${key} ${JSON.stringify(r[key])}: ${seen.get(r[key])} and ${r.file}`)
    else seen.set(r[key], r.file)
  }
}
for (const r of rows) {
  const owner = RESERVED[r.route]
  if (owner && owner !== r.folder) problems.push(`${r.file}: route ${r.route} is reserved for features/${owner}`)
}

if (problems.length) {
  console.error(`check-features: ${problems.length} problem(s)\n  ` + problems.join('\n  '))
  process.exit(1)
}
console.log(`check-features: ${rows.length} manifests, ids/routes/hotkeys unique, reserved routes ok (${rows.map((r) => r.route).join(' ')})`)
