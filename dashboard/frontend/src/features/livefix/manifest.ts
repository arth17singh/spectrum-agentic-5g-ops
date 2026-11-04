import { lazy } from 'react'
import { Wrench } from 'lucide-react'
import { defineFeature, type FeatureManifest } from '@/core/features'

// `data: 'real'`: a LIVE run calls the real kit (AgentCore Gateway, Cedar, the agents, the approvals table) and
// the team's in-process Maveric twin; a RECORDED run replays the judged swarm run's records and says so.
const manifest = {
  id: 'screen.livefix',
  route: '/live-fix',
  nav: { label: 'Live Fix', icon: Wrench, plane: 'operate', hotkey: 'l', order: 5.2 },
  title: 'Live Fix',
  description: 'Inject a fault on the twin and watch the swarm fix it: the agents see symptoms only, Cedar holds the change until a person approves it, and the twin is graded against the healthy network. Every run has its own evidence chain.',
  tracks: ['T3'],
  data: 'real',
  component: lazy(() => import('./LiveFixPage')),
  preload: () => import('./LiveFixPage'),
} satisfies Record<string, unknown>

export default defineFeature(manifest as unknown as FeatureManifest)
