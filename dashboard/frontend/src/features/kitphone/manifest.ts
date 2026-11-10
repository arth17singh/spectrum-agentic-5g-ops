import { lazy } from 'react'
import { Smartphone } from 'lucide-react'
import { defineFeature } from '@/core/features'

export default defineFeature({
  id: 'screen.kitphone',
  route: '/m/kit',
  nav: { label: 'Phone approval', icon: Smartphone, plane: 'none', order: 99.5 },
  title: 'Approve a change',
  description: 'One real kit approval on a paired phone',
  component: lazy(() => import('./KitPhoneApprove')),
  preload: () => import('./KitPhoneApprove'),
  bare: true,
})
