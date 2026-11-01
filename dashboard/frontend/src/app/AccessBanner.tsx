import { ACCESS_HELP, useAccess } from '@/core/access'

/** Shown over every route, the phone page included, once the backend refuses this device. */
export function AccessBanner() {
  const denied = useAccess((s) => s.denied)
  if (!denied) return null
  return (
    <div role="alert" className="fixed inset-x-0 top-0 z-[60] border-b-2 border-critical bg-s2 px-4 py-3 text-body text-ink-1 shadow-[0_8px_24px_rgba(0,0,0,0.3)]">
      <span className="font-semibold text-critical-ink">This device has no valid access token.</span> {ACCESS_HELP}
    </div>
  )
}
