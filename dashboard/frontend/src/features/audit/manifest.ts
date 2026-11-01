import { lazy } from 'react'
import { Link2 } from 'lucide-react'
import { defineFeature } from '@/core/features'

export default defineFeature({
  id: 'screen.audit',
  route: '/audit',
  nav: { label: 'Audit & Proof', icon: Link2, plane: 'prove', hotkey: 'a', order: 8 },
  title: 'Audit & Proof',
  description: 'A hash-chained, signed ledger anchored to S3 Object Lock. Verify it, tamper with it, and trace any action back to the training round behind it.',
  tracks: ['Trust'],
  component: lazy(() => import('./AuditPage')),
  preload: () => import('./AuditPage'),
})
