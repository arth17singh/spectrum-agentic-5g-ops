import { lazy } from 'react'
import { BadgeCheck } from 'lucide-react'
import { defineFeature, type FeatureManifest } from '@/core/features'

// `data: 'real'`: rows come from the kit's approvals table (team13-approvals) through the user's own AWS
// session on this laptop; a decision is the kit's own update plus its CloudWatch control line. Laptop only.
const manifest = {
  id: 'screen.kitapprovals',
  route: '/approvals',
  nav: { label: 'Approvals', icon: BadgeCheck, plane: 'operate', hotkey: 'k', order: 5.35 },
  title: 'Approvals',
  description: 'The kit’s real approvals: a change noc_lead wants to make to the twin waits here until a person approves or rejects it with a reason and the approver PIN. Approving does not apply it; noc_lead applies it after you tell it “Approved.” in Ask the NOC.',
  tracks: ['T3'],
  data: 'real',
  component: lazy(() => import('./ApprovalsPage')),
  preload: () => import('./ApprovalsPage'),
} satisfies Record<string, unknown>

export default defineFeature(manifest as unknown as FeatureManifest)
