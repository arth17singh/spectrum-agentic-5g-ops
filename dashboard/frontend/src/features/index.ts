// Feature registry. One line per feature module: delete the line (and the folder) to remove a
// screen; the shell, navigation, palette and hotkeys all derive from this list.
import type { FeatureManifest } from '@/core/features'
import mission from './mission/manifest'
import federation from './federation/manifest'
import proving from './proving/manifest'
import release from './release/manifest'
import swarm from './swarm/manifest'
import oversight from './oversight/manifest'
import trust from './trust/manifest'
import audit from './audit/manifest'
import evidence from './evidence/manifest'
import observability from './observability/manifest'
import settings from './settings/manifest'
import mobile from './mobile/manifest'
import presenter from './presenter/manifest'
import dossier from './audit/dossierManifest'

export const features: FeatureManifest[] = [mission, federation, proving, release, swarm, oversight, trust, audit, evidence, observability, settings, mobile, presenter, dossier]
