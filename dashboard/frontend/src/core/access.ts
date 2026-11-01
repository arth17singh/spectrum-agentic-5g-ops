import { create } from 'zustand'

// Access token for devices other than this laptop. The backend lets the console on this laptop
// (loopback) in without one and asks every other device for it: X-Spectrum-Token on fetches, ?t= on
// the WebSocket. The Demo Director's phone link carries it as ?t=, which this module reads on load.
//
// Kept in localStorage rather than sessionStorage: a phone browser that discards and reloads the tab
// mid-demo, or a dossier opening in a new tab, must not lock the device out. A leftover copy is
// harmless because the server mints a new token on every restart, and a stale one only raises the
// banner that says how to get a fresh link. A ?t= in the URL always replaces the stored token.

const KEY = 'spectrum.token'

let token: string | null = null
try {
  token = localStorage.getItem(KEY)
} catch {
  // storage blocked: the token then lasts for this page load only
}

const here = new URL(location.href)
const fromLink = here.searchParams.get('t')
if (fromLink) {
  token = fromLink
  try {
    localStorage.setItem(KEY, fromLink)
  } catch {
    // as above
  }
  // keep the secret out of the address bar and shared links; runs before the router reads the URL
  here.searchParams.delete('t')
  history.replaceState(history.state, '', here)
}

export const accessToken = (): string | null => token

/** Adds ?t= to URLs that cannot carry a header: the WebSocket and the phone link. */
export function withToken(url: string, t: string | null = token): string {
  return t ? `${url}${url.includes('?') ? '&' : '?'}t=${encodeURIComponent(t)}` : url
}

/** localhost, 127.x.x.x or [::1]: a console opened on this laptop. */
export const isLoopbackHost = (host: string): boolean =>
  host === 'localhost' || host === '[::1]' || /^127\.\d+\.\d+\.\d+$/.test(host)

export const ACCESS_HELP = 'Open the console at http://localhost:8787, or use the phone link from the Demo Director.'

/** Set once the backend refuses this device (HTTP 401 or WebSocket close 1008); raises the banner. */
export const useAccess = create<{ denied: boolean }>(() => ({ denied: false }))
export const reportDenied = () => useAccess.setState({ denied: true })
