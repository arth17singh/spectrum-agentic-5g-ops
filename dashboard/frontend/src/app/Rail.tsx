import clsx from 'clsx'
import { CircleHelp } from 'lucide-react'
import { NavLink } from 'react-router-dom'
import { useFlags, isOn } from '@/core/flags'
import { planeColor } from '@/core/palette'
import { useLive } from '@/core/store'
import { useUi } from '@/core/ui-store'
import { features } from '@/features'

export function Rail() {
  const expanded = useUi((s) => s.railExpanded)
  const set = useUi((s) => s.set)
  const flags = useFlags()
  const pending = useLive((s) => (s.oversight.items ?? []).filter((i: any) => i.status === 'pending').length)
  const incident = useLive((s) => s.swarm.incident && s.swarm.incident.status !== 'resolved')
  const items = features.filter((f) => !f.bare && isOn(f.id, flags)).sort((a, b) => a.nav.order - b.nav.order)
  const badge: Record<string, number | boolean> = { '/oversight': pending, '/swarm': !!incident }
  return (
    <nav
      aria-label="Screens"
      onMouseEnter={() => set({ railExpanded: true })}
      onMouseLeave={() => set({ railExpanded: false })}
      className={clsx('z-30 flex shrink-0 flex-col border-r border-line bg-page py-2 transition-[width] duration-200', expanded ? 'w-[220px]' : 'w-16')}
    >
      {items.map((f) => (
        <NavLink
          key={f.id}
          to={f.route}
          end={f.route === '/'}
          className={({ isActive }) => clsx('group relative mx-2 flex h-10 items-center gap-3 rounded-md px-3 text-body transition-colors',
            isActive ? 'bg-s2 text-ink-1' : 'text-ink-3 hover:bg-s1 hover:text-ink-1')}
          title={expanded ? undefined : f.nav.label}
        >
          {({ isActive }) => (
            <>
              <span className="absolute top-2 bottom-2 left-0 w-[2px] rounded-full" style={{ background: f.nav.plane === 'all' || f.nav.plane === 'none' ? 'transparent' : planeColor(f.nav.plane), opacity: isActive ? 1 : 0.55 }} />
              <f.nav.icon size={18} strokeWidth={isActive ? 2 : 1.75} className="shrink-0" />
              {expanded && <span className="truncate">{f.nav.label}</span>}
              {expanded && f.nav.hotkey && <span className="ml-auto text-micro text-ink-3">g {f.nav.hotkey}</span>}
              {badge[f.route] ? (
                <span className={clsx('absolute size-2 rounded-full', expanded ? 'right-2' : 'top-2 right-2')}
                  style={{ background: f.route === '/swarm' ? 'var(--critical)' : 'var(--warn)' }} />
              ) : null}
            </>
          )}
        </NavLink>
      ))}
      <button type="button" onClick={() => set({ helpOpen: true })} className="mx-2 mt-auto flex h-10 items-center gap-3 rounded-md px-3 text-ink-3 hover:bg-s1 hover:text-ink-1" title="Keyboard shortcuts (?)">
        <CircleHelp size={18} className="shrink-0" />
        {expanded && <span className="text-body">Shortcuts</span>}
      </button>
    </nav>
  )
}
