import { lazy } from 'react'
import { FlaskConical } from 'lucide-react'
import { defineFeature } from '@/core/features'

export default defineFeature({
  id: 'screen.proving',
  route: '/proving',
  nav: { label: 'Proving Ground', icon: FlaskConical, plane: 'verify', hotkey: 'p', order: 3 },
  title: 'Proving Ground',
  description: 'An automated verification pipeline: price, performance, accuracy and safety, plus private scores from each operator, a signed AI-BOM and a deploy gate.',
  tracks: ['T1'],
  component: lazy(() => import('./ProvingPage')),
  preload: () => import('./ProvingPage'),
})
