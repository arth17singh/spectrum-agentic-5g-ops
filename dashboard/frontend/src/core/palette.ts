import type { Plane } from './types'

// Plane hues (identity) and entity colors. Read from CSS variables so both themes stay in sync.
export const PLANES: { id: Exclude<Plane, 'system'>; label: string; color: string }[] = [
  { id: 'learn', label: 'Learn', color: 'var(--learn)' },
  { id: 'verify', label: 'Verify', color: 'var(--verify)' },
  { id: 'release', label: 'Release', color: 'var(--release)' },
  { id: 'operate', label: 'Operate', color: 'var(--operate)' },
  { id: 'prove', label: 'Prove', color: 'var(--prove)' },
]

export const planeColor = (p: string) =>
  ({ learn: 'var(--learn)', verify: 'var(--verify)', release: 'var(--release)', operate: 'var(--operate)',
     prove: 'var(--prove)' } as Record<string, string>)[p] ?? 'var(--ink-3)'

// Operators keep the same color on every screen (color follows the entity, never its rank).
export const OPERATOR_COLOR: Record<string, string> = {
  'op-a': 'var(--s1)',
  'op-b': 'var(--s2)',
  'op-c': 'var(--s3)',
  'op-d': 'var(--s4)',
  'op-e': 'var(--s5)',
}
export const opColor = (id: string) => OPERATOR_COLOR[id] ?? 'var(--neutral-series)'

// Models: ours in the Verify hue family, baselines neutral; fixed per entity.
export const MODEL_COLOR: Record<string, string> = {
  'claude-sonnet': 'var(--s1)',
  'claude-haiku': 'var(--s3)',
  'nova-pro': 'var(--s2)',
  'nova-lite': 'var(--s5)',
  'nlm-v6': 'var(--s7)',
  'nlm-v7': 'var(--s4)',
  'nlm-v8': 'var(--s6)',
}
export const modelColor = (id: string) => MODEL_COLOR[id] ?? 'var(--neutral-series)'

/** Resolve a CSS variable to a concrete color (canvas renderers need real values). */
export function cssVar(name: string, el: Element = document.documentElement): string {
  const n = name.startsWith('var(') ? name.slice(4, -1) : name
  return getComputedStyle(el).getPropertyValue(n).trim() || '#888'
}
