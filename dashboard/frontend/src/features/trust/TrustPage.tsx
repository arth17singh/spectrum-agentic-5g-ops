import { PageHeader } from '@/ui'
import manifest from './manifest'
import { AttestationChain, DecisionStream, Delegations, Guardrails, Identities, PolicyStore, Posture, TopDenied } from './panels'

export default function TrustPage() {
  return (
    <div>
      <PageHeader title={manifest.title} description={manifest.description} tracks={manifest.tracks} />
      <Posture />
      <div className="mt-4 grid grid-cols-12 gap-4">
        <div className="col-span-12"><PolicyStore /></div>
        <div className="col-span-12 xl:col-span-7"><DecisionStream /></div>
        <div className="col-span-12 flex flex-col gap-4 xl:col-span-5">
          <TopDenied />
          <Delegations />
        </div>
        <div className="col-span-12 xl:col-span-7"><Identities /></div>
        <div className="col-span-12 xl:col-span-5"><Guardrails /></div>
        <div className="col-span-12"><AttestationChain /></div>
      </div>
    </div>
  )
}
