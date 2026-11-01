import { lazy } from 'react'
import { Activity } from 'lucide-react'
import { defineFeature } from '@/core/features'

export default defineFeature({
  id: 'screen.observability',
  route: '/observability',
  nav: { label: 'Observability', icon: Activity, plane: 'all', hotkey: 'b', order: 9 },
  title: 'Observability',
  description: 'Golden signals per plane, end-to-end traces across agents, gateway, policy and tools, plus token cost and latency.',
  tracks: ['Obs', 'T1'],
  component: lazy(() => import('./ObservabilityPage')),
  preload: () => import('./ObservabilityPage'),
})
