import { lazy } from 'react'
import { FileCheck2 } from 'lucide-react'
import { api } from '@/core/api'
import { defineFeature } from '@/core/features'
import { toast } from '@/core/ui-store'

export default defineFeature({
  id: 'screen.evidence',
  route: '/evidence',
  nav: { label: 'Evidence', icon: FileCheck2, plane: 'prove', hotkey: 'e', order: 8.5 },
  title: 'Evidence',
  description: 'The real evidence chain for judged runs: every declaration, output and log file, hashed and signed into a persisted ledger that survives restarts and demo resets. Nothing here is simulated.',
  tracks: ['Trust'],
  component: lazy(() => import('./EvidencePage')),
  preload: () => import('./EvidencePage'),
  commands: [
    {
      id: 'evidence.scan', label: 'Evidence: scan for new artifacts now', group: 'Evidence', keywords: 'ingest ledger real',
      run: () => api.evidenceScan().then((r) => toast({ severity: 'notice', title: `Evidence scan: ${r.scan?.new ?? 0} new record(s)` })),
    },
  ],
})
