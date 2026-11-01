import { lazy } from 'react'
import { Rocket } from 'lucide-react'
import { defineFeature } from '@/core/features'

export default defineFeature({
  id: 'screen.release',
  route: '/release',
  nav: { label: 'Release Center', icon: Rocket, plane: 'release', hotkey: 'r', order: 4 },
  title: 'Release Center',
  description: 'Repeatable multi-vendor deployment from one signed manifest: shadow, then canary, then full, with guard metrics and automatic rollback.',
  tracks: ['T1'],
  component: lazy(() => import('./ReleasePage')),
  preload: () => import('./ReleasePage'),
})
