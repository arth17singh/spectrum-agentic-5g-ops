import { lazy } from 'react'
import { Radar } from 'lucide-react'
import { defineFeature } from '@/core/features'

// `data: 'real'`: every number on this screen is read from the evidence records (GET /api/swarmtower).
const contract = { data: 'real' as const }

export default defineFeature({
  id: 'screen.swarm',
  route: '/swarm',
  nav: { label: 'Swarm Tower', icon: Radar, plane: 'operate', hotkey: 's', order: 5 },
  title: 'Swarm Tower',
  description: 'The live Track 3 swarm on the team’s Maveric twin, from AWS-recorded records: 18 incidents, 3 operator steps each, every proposed change held at the approval gate; and the three Track 2 silos it learned from.',
  tracks: ['T3'],
  component: lazy(() => import('./SwarmPage')),
  preload: () => import('./SwarmPage'),
  ...contract,
})
