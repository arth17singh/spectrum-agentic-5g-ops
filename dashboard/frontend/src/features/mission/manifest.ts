import { lazy } from 'react'
import { LayoutDashboard } from 'lucide-react'
import { defineFeature } from '@/core/features'

export default defineFeature({
  id: 'screen.mission',
  route: '/',
  nav: { label: 'Mission Control', icon: LayoutDashboard, plane: 'all', hotkey: 'm', order: 1 },
  title: 'Mission Control',
  description: 'The whole lifecycle at a glance: five operators train privately, models are verified and released, a governed swarm runs the network, and every step is provable.',
  tracks: ['T1', 'T2', 'T3'],
  component: lazy(() => import('./MissionPage')),
  preload: () => import('./MissionPage'),
})
