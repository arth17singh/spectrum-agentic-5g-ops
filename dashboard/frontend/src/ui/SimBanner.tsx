import { FlaskConical } from 'lucide-react'
import { Link } from 'react-router-dom'

export const SIM_BANNER_TEXT = 'SIMULATED: illustrative numbers from the built-in simulator, not measurements'

/** The one honesty banner for every simulator-backed screen. `approvals` adds the pointer to the real
 *  Approvals page (oversight and the phone page, where a reader could mistake the demo queue for it). */
export function SimBanner({ approvals = false, compact = false }: { approvals?: boolean; compact?: boolean }) {
  return (
    <div role="note" data-sim-banner
      className={`flex items-start gap-2 border border-warn/40 bg-warn/10 text-warn ${compact ? 'rounded-md px-2.5 py-1.5 text-micro' : 'px-4 py-1.5 text-cap border-x-0 border-t-0'}`}>
      <FlaskConical size={13} aria-hidden className="mt-0.5 shrink-0" />
      <div className="min-w-0">
        <span className="font-semibold">{SIM_BANNER_TEXT}.</span>
        {approvals && (
          <span className="text-ink-1">
            {' '}Real approvals happen on the <Link to="/approvals" className="font-medium underline underline-offset-2">Approvals page</Link>.
          </span>
        )}
      </div>
    </div>
  )
}
