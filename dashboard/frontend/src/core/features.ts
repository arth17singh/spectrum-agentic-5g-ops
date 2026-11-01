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
  tracks?: ('T1' | 'T2' | 'T3' | 'Trust' | 'Obs')[]
  component: LazyExoticComponent<ComponentType> | ComponentType
  /** warms the lazy chunk so the first visit during a demo never waits on the network */
  preload?: () => Promise<unknown>
  commands?: Command[]
  /** routes outside the shell (e.g. mobile approval page) */
  bare?: boolean
}

export const defineFeature = (m: FeatureManifest) => m
