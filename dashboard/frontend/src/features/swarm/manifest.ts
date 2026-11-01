import { lazy } from 'react'
import { Radar } from 'lucide-react'
import { defineFeature } from '@/core/features'

export default defineFeature({
  id: 'screen.swarm',
  route: '/swarm',
  nav: { label: 'Swarm Tower', icon: Radar, plane: 'operate', hotkey: 's', order: 5 },
  title: 'Swarm Tower',
  description: 'A governed agent swarm on the network twin. Agents negotiate over signed messages, every action passes the policy gateway, and large changes wait for a human.',
  tracks: ['T3'],
  component: lazy(() => import('./SwarmPage')),
  preload: () => import('./SwarmPage'),
})
