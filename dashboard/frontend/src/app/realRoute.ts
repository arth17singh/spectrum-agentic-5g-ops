import { matchPath, useLocation } from 'react-router-dom'
import { features } from '@/features'

// Routes whose numbers are all measured: the simulator strips (ribbon, stream, sim clock, ...) are
// hidden there. The registry says which shell features are real; the others are lanes whose pages
// may not be registered in every build.
const EXTRA_REAL = ['/approvals', '/ask-noc']

const inRoute = (pathname: string, route: string) => route !== '/' && !!(matchPath({ path: route, end: true }, pathname) || matchPath({ path: route + '/*' }, pathname))

export function isRealPath(pathname: string): boolean {
  return features.some((f) => f.data === 'real' && !f.bare && inRoute(pathname, f.route)) || EXTRA_REAL.some((r) => inRoute(pathname, r))
}

/** Screens that show simulator numbers and so carry the SIMULATED banner. */
export function isSimPath(pathname: string): boolean {
  if (isRealPath(pathname)) return false
  const f = features.find((x) => !x.bare && (x.route === '/' ? pathname === '/' : inRoute(pathname, x.route)))
  return !!f && f.data !== 'real' && f.id !== 'screen.settings'
}

export function useIsRealRoute(): boolean {
  return isRealPath(useLocation().pathname)
}
