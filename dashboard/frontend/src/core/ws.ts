import { reportDenied, withToken } from './access'
import { api } from './api'
import { applyEvents, loadSnapshot, useLive } from './store'
import { toast } from './ui-store'
import type { Envelope } from './types'

// One WebSocket. On (re)connect: buffer events, fetch the snapshot, then replay buffered events
// newer than the snapshot. A `system.reset` event triggers a fresh snapshot.

let socket: WebSocket | null = null
let backoff = 500
let started = false
let counter = 0
let isOpen = false

const TOAST_TYPES: Record<string, string> = {
  'approval.requested': 'Review',
  escalation: 'Review',
  'attack.denied': 'Inspect',
  'identity.revoked': 'Inspect',
  'fl.poison': 'Inspect',
  'ledger.verify': 'Open',
  'incident.opened': 'Open',
  'eval.gate': 'Open',
  'guardrail.block': 'Inspect',
  'fl.budget': 'Open',
}
const TOAST_ROUTE: Record<string, string> = {
  'approval.requested': '/oversight', escalation: '/oversight', 'attack.denied': '/swarm', 'identity.revoked': '/swarm',
  'fl.poison': '/federation', 'ledger.verify': '/audit', 'incident.opened': '/swarm', 'eval.gate': '/proving',
  'guardrail.block': '/trust', 'fl.budget': '/federation',
}

function maybeToast(e: Envelope) {
  const action = TOAST_TYPES[e.type]
  if (!action) return
  if (e.severity !== 'critical' && !['approval.requested', 'incident.opened', 'eval.gate', 'fl.budget'].includes(e.type)) return
  if (e.type === 'ledger.verify' && e.severity !== 'critical') return
  toast({ severity: e.severity, title: e.summary, plane: e.plane, action, route: TOAST_ROUTE[e.type], ledgerSeq: e.ledger?.seq })
}

export function startLive() {
  if (started) return
  started = true
  connect()
  // connection health + message rate
  setInterval(() => {
    const s = useLive.getState()
    const age = performance.now() - s.lastMsgAt
    // the stream ticks at >= 1 Hz (clock + twin), so silence means trouble
    const conn = !isOpen ? 'offline' : age < 2500 ? 'live' : 'lagging'
    useLive.setState({ conn: s.ready ? conn : s.conn, msgPerSec: counter })
    counter = 0
  }, 1000)
}

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(withToken(`${proto}://${location.host}/ws`))
  socket = ws
  let buffered: Envelope[] | null = []
  let snapSeq = -1

  const resync = async () => {
    buffered = buffered ?? []
    try {
      const snap = await api.snapshot()
      snapSeq = snap.seq
      loadSnapshot(snap)
      const pending = (buffered ?? []).filter((e) => e.seq > snapSeq)
      buffered = null
      if (pending.length) applyEvents(pending)
    } catch {
      buffered = null
    }
  }

  ws.onopen = () => {
    isOpen = true
    backoff = 500
    resync()
    // on every (re)connect: a restarted backend has a new access token for the Director's phone link
    api.config().then((c) => useLive.setState({ config: c, mode: c.mode })).catch(() => {})
  }
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data)
    if (msg.kind !== 'events') return
    const events: Envelope[] = msg.events
    counter += events.length
    if (events.some((e) => e.type === 'system.reset')) {
      resync()
      return
    }
    if (buffered) {
      buffered.push(...events)
      return
    }
    applyEvents(events)
    for (const e of events) maybeToast(e)
  }
  ws.onclose = (e) => {
    if (socket !== ws) return
    isOpen = false
    useLive.setState({ conn: 'offline' })
    // 1008: the backend refused this device's access token. Say so, and only retry slowly.
    if (e.code === 1008) reportDenied()
    setTimeout(connect, e.code === 1008 ? 8000 : backoff)
    backoff = Math.min(8000, backoff * 2)
  }
  ws.onerror = () => ws.close()
}
