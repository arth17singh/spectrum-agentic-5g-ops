import { lazy } from 'react'
import { FileText } from 'lucide-react'
import { defineFeature } from '@/core/features'

export default defineFeature({
  id: 'audit.dossier',
  route: '/dossier/:incident',
  nav: { label: 'Incident dossier', icon: FileText, plane: 'none', order: 97 },
  title: 'Incident dossier',
  description: 'Printable, signed incident report',
  component: lazy(() => import('./DossierPage')),
  preload: () => import('./DossierPage'),
  bare: true,
})
