import { lazy } from 'react'
import { MessagesSquare } from 'lucide-react'
import { defineFeature, type FeatureManifest } from '@/core/features'

// `data: 'real'`: every reply comes from the real checkpoint kit (labelled LIVE, or RECORDED once the case is
// no longer followed). There is no demo or scripted mode.
const manifest = {
  id: 'screen.asknoc',
  route: '/ask-noc',
  nav: { label: 'Ask the NOC', icon: MessagesSquare, plane: 'operate', hotkey: 'n', order: 5.3 },
  title: 'Ask the NOC',
  description: 'Ask noc_lead through the checkpoint’s front door. Every reply comes live from the kit, with its case id, and the swarm’s governed calls are shown beside it. This panel can’t approve or apply anything: a person decides every change.',
  tracks: ['T3'],
  data: 'real',
  component: lazy(() => import('./AskNocPage')),
  preload: () => import('./AskNocPage'),
} satisfies Record<string, unknown>

export default defineFeature(manifest as unknown as FeatureManifest)
