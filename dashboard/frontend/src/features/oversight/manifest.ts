import { lazy } from 'react'
import { UserCheck } from 'lucide-react'
import { defineFeature } from '@/core/features'

export default defineFeature({
  id: 'screen.oversight',
  route: '/oversight',
  nav: { label: 'Oversight Desk', icon: UserCheck, plane: 'operate', hotkey: 'o', order: 6 },
  title: 'Oversight Desk',
  description: 'Decisions agents are not allowed to make alone. Each one arrives with the plan, the predicted impact, the dissent, and an SLA, and every decision is signed.',
  tracks: ['T3', 'Trust'],
  component: lazy(() => import('./OversightPage')),
  preload: () => import('./OversightPage'),
})
