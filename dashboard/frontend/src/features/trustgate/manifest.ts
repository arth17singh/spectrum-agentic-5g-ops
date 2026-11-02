import { lazy } from 'react'
import { Gauge } from 'lucide-react'
import { defineFeature, type FeatureManifest } from '@/core/features'

// `data: 'real'` is read by the shell's demo rail (added to FeatureManifest by the shared-parts
// builder). Built as a variable so this file type-checks before and after that field exists.
const manifest = {
  id: 'screen.trustgate',
  route: '/trust-gate',
  nav: { label: 'Trust Gate', icon: Gauge, plane: 'verify', order: 3.5 },
  title: 'Trust Gate',
  description: 'Each change the swarm makes has a predicted effect and a measured one. A model whose predictions stop matching reality loses trust and is paused automatically. The ladder shows how much of a model’s gain survives a move to a more realistic simulator.',
  tracks: ['T3', 'Trust'],
  data: 'real',
  component: lazy(() => import('./TrustGatePage')),
  preload: () => import('./TrustGatePage'),
} satisfies Record<string, unknown>

export default defineFeature(manifest as unknown as FeatureManifest)
