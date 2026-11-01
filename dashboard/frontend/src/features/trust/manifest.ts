import { lazy } from 'react'
import { ShieldCheck } from 'lucide-react'
import { defineFeature } from '@/core/features'

export default defineFeature({
  id: 'screen.trust',
  route: '/trust',
  nav: { label: 'Trust Center', icon: ShieldCheck, plane: 'prove', hotkey: 't', order: 7 },
  title: 'Trust Center',
  description: 'The guarantees behind every action: Cedar policies outside the model, per-agent identities, guardrails, delegation checks and enclave attestation.',
  tracks: ['T3', 'Trust'],
  component: lazy(() => import('./TrustPage')),
  preload: () => import('./TrustPage'),
})
