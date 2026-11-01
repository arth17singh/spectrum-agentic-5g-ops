import { lazy } from 'react'
import { Presentation } from 'lucide-react'
import { defineFeature } from '@/core/features'

export default defineFeature({
  id: 'demo.director',
  route: '/presenter',
  nav: { label: 'Presenter', icon: Presentation, plane: 'none', order: 98 },
  title: 'Presenter',
  description: 'Second-screen demo driver',
  component: lazy(() => import('./PresenterPage')),
  preload: () => import('./PresenterPage'),
  bare: true,
})
