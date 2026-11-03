import { Suspense, useEffect } from 'react'
import { Navigate, Outlet, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { useAccess } from '@/core/access'
import { api } from '@/core/api'
import { Gate, isOn, useFlags } from '@/core/flags'
import { useLive } from '@/core/store'
import { useUi } from '@/core/ui-store'
import { startLive } from '@/core/ws'
import { features } from '@/features'
import { SimBanner } from '@/ui'
import { Director } from './Director'
import { useHotkeys } from './hotkeys'
import { Inspector } from './Inspector'
import { CommandPalette, HelpOverlay, Toasts } from './Overlays'
import { Rail } from './Rail'
import { isSimPath, useIsRealRoute } from './realRoute'
import { Ribbon } from './Ribbon'
import { StreamDrawer } from './StreamDrawer'
import { TopBar } from './TopBar'

function useThemeAttrs() {
  const theme = useUi((s) => s.theme)
  const motion = useUi((s) => s.motion)
  const density = useUi((s) => s.density)
  const spotlight = useUi((s) => s.spotlight)
  const lit = useUi((s) => s.lit)
  useEffect(() => {
    const el = document.documentElement
    el.dataset.theme = theme
    el.dataset.motion = motion
    el.dataset.density = density
    el.dataset.spotlight = spotlight && lit.length ? 'on' : 'off'
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', theme === 'ops' ? '#0F1216' : '#F4F6F8')
  }, [theme, motion, density, spotlight, lit])
}

/** Every console follows server-side director beats (so a presenter laptop can drive the projector). */
function useFollowDirector() {
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const started = useLive((s) => s.director?.started_at)
  const beat = useLive((s) => s.director?.beat)
  useEffect(() => {
    if (!beat || !started) return
    const b = (useLive.getState().director?.beats ?? [])[beat - 1]
    if (!b) return
    if (pathname !== b.route && !pathname.startsWith('/m/') && !pathname.startsWith('/presenter')) navigate(b.route)
    const ui = useUi.getState()
    useUi.setState({ lit: ui.spotlight ? b.spotlight : [] })
    // bring the first lit panel into view once the route has rendered
    setTimeout(() => {
      const el = document.querySelector(`[data-panel="${b.spotlight[0]}"]`)
      const main = document.querySelector('main')
      if (!el || !main) return
      const tall = el.getBoundingClientRect().height > main.clientHeight * 0.8
      el.scrollIntoView({ block: tall ? 'start' : 'center', behavior: 'smooth' })
    }, 450)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [started])
}

/** When the backend serves real evidence runs (`real_data` on the small /api/config, not the big snapshot),
 *  the first landing on "/" goes to the Scorecard, once per page load. Mission Control stays reachable. */
let landed = false
function useRealLanding() {
  const navigate = useNavigate()
  const { pathname } = useLocation()
  useEffect(() => {
    if (landed) return
    landed = true
    if (pathname !== '/') return
    api.config().then((c) => { if (c?.real_data && window.location.pathname === '/') navigate('/scorecard', { replace: true }) }).catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
}

function Layout() {
  useRealLanding()
  useHotkeys()
  useFollowDirector()
  const ready = useLive((s) => s.ready)
  useEffect(() => {
    if (!ready) return
    const warm = () => features.forEach((f) => f.preload?.().catch(() => {}))
    const id = setTimeout(warm, 800)
    return () => clearTimeout(id)
  }, [ready])
  const { pathname } = useLocation()
  const real = useIsRealRoute()
  const sim = isSimPath(pathname)
  return (
    <div className="flex h-full flex-col">
      <TopBar />
      {/* the ribbon and the stream drawer show simulator numbers: not on routes that show measured records */}
      {!real && <Gate flag="shell.ribbon"><Ribbon /></Gate>}
      {sim && <SimBanner approvals={pathname.startsWith('/oversight')} />}
      <div className="relative flex min-h-0 flex-1">
        <Rail />
        <main className="min-w-0 flex-1 overflow-auto px-6 pt-5 pb-8">
          {ready ? (
            <Suspense fallback={<div className="text-cap text-ink-3">Loading screen…</div>}>
              <Outlet />
            </Suspense>
          ) : (
            <Connecting />
          )}
        </main>
        <Inspector />
      </div>
      {!real && <Gate flag="shell.stream"><StreamDrawer /></Gate>}
      <Director />
      <CommandPalette />
      <HelpOverlay />
      <Toasts />
    </div>
  )
}

function Connecting() {
  const conn = useLive((s) => s.conn)
  const denied = useAccess((s) => s.denied)
  if (denied) return null // the access banner explains; "Backend unreachable" would mislead
  return (
    <div className="grid h-full place-items-center">
      <div className="text-center">
        <div className="text-section font-medium text-ink-1">{conn === 'offline' ? 'Backend unreachable' : 'Connecting to the SPECTRUM engine'}</div>
        <p className="mt-1 text-body text-ink-3">
          {conn === 'offline' ? 'Start it with `make dev` (or `uv run uvicorn spectrum.main:app --port 8787` in backend/). Retrying automatically.' : 'Loading the pre-warmed demo state…'}
        </p>
      </div>
    </div>
  )
}

export function App() {
  useThemeAttrs()
  const flags = useFlags()
  useEffect(() => {
    startLive()
    api.flags().then((r) => useFlags.getState().setRemote(r)).catch(() => {})
  }, [])
  const shellRoutes = features.filter((f) => !f.bare)
  const bareRoutes = features.filter((f) => f.bare)
  return (
    <Routes>
      {bareRoutes.map((f) => <Route key={f.id} path={f.route} element={<Suspense fallback={null}><f.component /></Suspense>} />)}
      <Route element={<Layout />}>
        {shellRoutes.map((f) => (
          <Route key={f.id} path={f.route} element={isOn(f.id, flags) ? <f.component /> : <Navigate to="/" replace />} />
        ))}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
