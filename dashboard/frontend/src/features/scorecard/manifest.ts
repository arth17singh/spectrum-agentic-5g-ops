import { lazy } from 'react'
import { ListChecks } from 'lucide-react'
import { defineFeature } from '@/core/features'

// `data: 'real'` is part of the shared manifest contract (W7a adds it to FeatureManifest in core/).
// It is spread in so this file type-checks both before and after that type lands.
const contract = { data: 'real' as const }

export default defineFeature({
  id: 'screen.scorecard',
  route: '/scorecard',
  nav: { label: 'Scorecard', icon: ListChecks, plane: 'verify', order: 3.5 },
  title: 'Scorecard',
  description: 'Every scored unit: the target declared before the run, what was measured, pass or fail recomputed here from the declared target, and the gain over its baseline with a 95% interval.',
  tracks: ['T3', 'Trust'],
  component: lazy(() => import('./ScorecardPage')),
  preload: () => import('./ScorecardPage'),
  ...contract,
})
