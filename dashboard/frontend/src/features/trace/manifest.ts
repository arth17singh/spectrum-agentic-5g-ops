import { lazy } from 'react'
import { Route } from 'lucide-react'
import { defineFeature } from '@/core/features'

// `data: 'real'` is a field the shell registry gains in the C1 change; spreading it keeps this file
// type-correct both before and after that lands.
const real = { data: 'real' } as const

export default defineFeature({
  id: 'screen.trace',
  route: '/trace',
  nav: { label: 'Trace', icon: Route, plane: 'operate', order: 5.5 },
  title: 'Trace',
  description: 'Each Track 3 case rebuilt from the persisted Cedar decisions, model pauses and applied changes, and checked against the declared sequence: case opened, agents asked, tools called, approval, change applied, observe.',
  tracks: ['T3'],
  component: lazy(() => import('./TracePage')),
  preload: () => import('./TracePage'),
  ...real,
})
