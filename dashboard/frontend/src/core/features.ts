import type { ComponentType, LazyExoticComponent } from 'react'
import type { LucideIcon } from 'lucide-react'

// Feature modules (DASHBOARD_SPEC.md §7). Each screen lives in src/features/<name>/ and exports a
// manifest. The shell builds routes, navigation, palette commands and hotkeys from the registered
// manifests, so removing a feature is: delete the folder + its one line in src/features/index.ts.

export interface Command {
  id: string
  label: string
  group: string
  flag?: string
  keywords?: string
  run: (ctx: { navigate: (to: string) => void }) => void | Promise<unknown>
}

export interface FeatureManifest {
  id: string // the screen flag, e.g. 'screen.federation'
  route: string
  nav: {
    label: string
    icon: LucideIcon
    plane: 'learn' | 'verify' | 'release' | 'operate' | 'prove' | 'all' | 'none'
    hotkey?: string // letter after "g"
    order: number
  }
  title: string
  description: string
  tracks?: ('T1' | 'T2' | 'T3' | 'Trust' | 'Obs' | 'Bridge')[]
  /** 'real' = every number comes from measured records (listed in the demo rail); 'mock' = the simulator.
   *  Optional in the manifest; the registry fills it ('mock' unless declared). */
  data?: 'real' | 'mock'
  component: LazyExoticComponent<ComponentType> | ComponentType
  /** warms the lazy chunk so the first visit during a demo never waits on the network */
  preload?: () => Promise<unknown>
  commands?: Command[]
  /** routes outside the shell (e.g. mobile approval page) */
  bare?: boolean
}

export const defineFeature = (m: FeatureManifest) => m

/** Routes held for the Track 3 slices (features/trace, features/trustgate, features/scorecard). */
export const RESERVED_ROUTES: Record<string, string> = { '/trace': 'trace', '/trust-gate': 'trustgate', '/scorecard': 'scorecard' }

/** Everything wrong with a registry: duplicate ids, routes or hotkeys, and reserved routes taken by
 *  another folder. `folder` maps a feature id to the features/<folder> it came from. Empty = fine. */
export function registryProblems(list: FeatureManifest[], folder: (id: string) => string | undefined = () => undefined): string[] {
  const problems: string[] = []
  const dupes = (what: string, key: (f: FeatureManifest) => string | undefined) => {
    const seen = new Map<string, string>()
    for (const f of list) {
      const k = key(f)
      if (k === undefined) continue
      const other = seen.get(k)
      if (other) problems.push(`duplicate ${what} ${JSON.stringify(k)}: ${other} and ${f.id}`)
      else seen.set(k, f.id)
    }
  }
  dupes('id', (f) => f.id)
  dupes('route', (f) => f.route)
  dupes('hotkey', (f) => f.nav.hotkey?.toLowerCase())
  for (const f of list) {
    const owner = RESERVED_ROUTES[f.route]
    const from = folder(f.id)
    if (owner && from !== undefined && from !== owner) problems.push(`route ${f.route} is reserved for features/${owner}, not features/${from}`)
  }
  return problems
}
