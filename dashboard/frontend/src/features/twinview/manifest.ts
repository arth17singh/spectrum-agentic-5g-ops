import { lazy } from 'react'
import { Boxes } from 'lucide-react'
import { defineFeature, type FeatureManifest } from '@/core/features'

// `data: 'real'`: the screen fills in nothing. Every number is read from the persisted records (or, in
// Consortium mode, from the silos manifest our generator wrote) and carries a provenance chip.
const manifest = {
  id: 'screen.twinview',
  route: '/twin-view',
  nav: { label: 'Twin View', icon: Boxes, plane: 'operate', order: 5.4 },
  title: 'Twin View',
  description: 'The 4-cell network twin as the agents see it and change it: each cell’s power, tilt, users and coverage in every window, the change the swarm applied, what was predicted against what was measured, and who approved it. Consortium mode shows the operator silos that trained the shared response model.',
  tracks: ['T3', 'T2'],
  data: 'real',
  component: lazy(() => import('./TwinViewPage')),
  preload: () => import('./TwinViewPage'),
} satisfies Record<string, unknown>

export default defineFeature(manifest as unknown as FeatureManifest)
