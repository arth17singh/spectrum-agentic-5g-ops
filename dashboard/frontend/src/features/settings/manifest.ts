import { lazy } from 'react'
import { Settings2 } from 'lucide-react'
import { defineFeature } from '@/core/features'

export default defineFeature({
  id: 'screen.settings',
  route: '/settings',
  nav: { label: 'Settings & Flags', icon: Settings2, plane: 'none', hotkey: ',', order: 10 },
  title: 'Settings & feature flags',
  description: 'Turn features on or off without touching the rest of the console. Flags resolve from code defaults, remote config, this browser and the URL.',
  tracks: [],
  component: lazy(() => import('./SettingsPage')),
  preload: () => import('./SettingsPage'),
})
