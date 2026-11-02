// Feature registry. Every src/features/<x>/manifest.ts registers itself: add a folder with a
// manifest to add a screen, delete the folder to remove it. The shell, navigation, palette and
// hotkeys all derive from this list, ordered by nav.order.
import { registryProblems, type FeatureManifest } from '@/core/features'

const modules = import.meta.glob<FeatureManifest>(['./*/manifest.ts', './audit/dossierManifest.ts'], { eager: true, import: 'default' })

// Screens whose numbers are all measured, until their own manifest says so (data: 'real').
const REAL_BY_DEFAULT = new Set(['screen.evidence'])

const folderOf = new Map<string, string>()
for (const [path, m] of Object.entries(modules)) folderOf.set(m.id, path.split('/')[1])

export const features: FeatureManifest[] = Object.values(modules)
  .map((m) => ({ ...m, data: m.data ?? (REAL_BY_DEFAULT.has(m.id) ? 'real' : 'mock') }) as FeatureManifest)
  .sort((a, b) => a.nav.order - b.nav.order || a.id.localeCompare(b.id))

/** The shell screens backed by real records, in nav order: what the demo rail lists. */
export const realFeatures = (): FeatureManifest[] => features.filter((f) => f.data === 'real' && !f.bare)

const problems = registryProblems(features, (id) => folderOf.get(id))
if (problems.length) {
  // the route smoke fails on console errors, so a clash is caught at the merge gate; dev stops outright
  console.error('[features] registry problems:\n' + problems.join('\n'))
  if (import.meta.env.DEV) throw new Error('feature registry: ' + problems.join('; '))
}
