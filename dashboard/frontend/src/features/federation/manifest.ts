import { lazy } from 'react'
import { Network } from 'lucide-react'
import { defineFeature } from '@/core/features'

export default defineFeature({
  id: 'screen.federation',
  route: '/federation',
  nav: { label: 'Federation Center', icon: Network, plane: 'learn', hotkey: 'f', order: 2 },
  title: 'Federated fine-tuning',
  description: 'Five operators fine-tune one network model on private data with DP-SGD. Only masked adapter updates leave each operator, and an attested enclave aggregates them.',
  tracks: ['T1', 'T2'],
  component: lazy(() => import('./FederationPage')),
  preload: () => import('./FederationPage'),
})
