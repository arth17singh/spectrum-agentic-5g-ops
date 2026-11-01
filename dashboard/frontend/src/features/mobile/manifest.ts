import { lazy } from 'react'
import { Smartphone } from 'lucide-react'
import { defineFeature } from '@/core/features'

export default defineFeature({
  id: 'oversight.mobile',
  route: '/m/approve',
  nav: { label: 'Mobile approvals', icon: Smartphone, plane: 'none', order: 99 },
  title: 'Approvals',
  description: 'Phone-optimised approval page',
  component: lazy(() => import('./MobileApprove')),
  preload: () => import('./MobileApprove'),
  bare: true,
})
