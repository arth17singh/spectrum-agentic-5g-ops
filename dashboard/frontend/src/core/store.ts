import { create } from 'zustand'
import type { Envelope, Rec, Snapshot } from './types'

// One store, one slice per engine. Each WebSocket batch is applied in a single `set`, and every
// touched slice (and nested array) is replaced, never mutated, so selectors re-render correctly.

export type SliceKey = 'federation' | 'proving' | 'release' | 'twin' | 'swarm' | 'trust' | 'ledger' | 'oversight' |
  'observability' | 'director'

export type Conn = 'connecting' | 'live' | 'lagging' | 'offline'

export interface LiveState {
  ready: boolean
  conn: Conn
  lastMsgAt: number
  msgPerSec: number
  mode: string
  config: Rec | null
  clock: { t: number; iso: string; speed: number; paused: boolean; syncedAt: number }
  federation: Rec
  proving: Rec
  release: Rec
  twin: Rec
  swarm: Rec
  trust: Rec
  ledger: Rec
  oversight: Rec
  observability: Rec
  director: Rec
  stream: Envelope[]
  /** live-only data that is not part of the snapshot */
  live: { gradhist: Rec | null; lastSecagg: Rec | null; pulses: { from: string; to: string; at: number }[] }
}

const empty = {} as Rec
export const EMPTY: any[] = []
export const STREAM_MAX = 3000
export const NOISY = new Set(['fl.step', 'fl.gradhist', 'twin.tick', 'obs.metrics', 'obs.span', 'release.metrics',
  'swarm.agents', 'sim.clock', 'eval.progress', 'obs.trace'])

export const useLive = create<LiveState>(() => ({
  ready: false,
  conn: 'connecting',
  lastMsgAt: 0,
  msgPerSec: 0,
  mode: 'mock',
  config: null,
  clock: { t: 0, iso: '', speed: 1, paused: false, syncedAt: performance.now() },
  federation: empty,
  proving: empty,
  release: empty,
  twin: empty,
  swarm: empty,
  trust: empty,
  ledger: empty,
  oversight: empty,
  observability: empty,
  director: empty,
  stream: [],
  live: { gradhist: null, lastSecagg: null, pulses: [] },
}))

export function loadSnapshot(s: Snapshot) {
  useLive.setState({
    ready: true,
    clock: { ...s.clock, syncedAt: performance.now() },
    federation: s.federation,
    proving: s.proving,
    release: s.release,
    twin: s.twin,
    swarm: s.swarm,
    trust: s.trust,
    ledger: s.ledger,
    oversight: s.oversight,
    observability: s.observability,
    director: s.director,
    stream: s.stream.filter((e) => !NOISY.has(e.type)),
  })
}

type Draft = Partial<LiveState>
type Handler = (e: Envelope, d: (k: SliceKey) => Rec, draft: Draft, cur: LiveState) => void

const cap = <T,>(arr: T[] | undefined, item: T, max: number): T[] => {
  const out = arr ? [...arr, item] : [item]
  return out.length > max ? out.slice(out.length - max) : out
}
const upsert = (arr: Rec[] | undefined, item: Rec, key = 'id'): Rec[] => {
  const list = arr ? [...arr] : []
  const i = list.findIndex((x) => x[key] === item[key])
  if (i >= 0) list[i] = { ...list[i], ...item }
  else list.push(item)
  return list
}

// Ribbon pulses: artefacts moving between lifecycle planes.
const PULSE: Record<string, [string, string]> = {
  'fl.promote': ['learn', 'verify'],
  'eval.gate': ['verify', 'release'],
  'release.stage': ['release', 'operate'],
  'incident.resolved': ['operate', 'prove'],
  'loop.enqueue': ['prove', 'learn'],
  'fl.checkpoint': ['learn', 'learn'],
}

const H: Record<string, Handler> = {
  // ---------------------------------------------------------------- learn
  'fl.step': (e, d) => {
    const f = d('federation')
    const rows: Rec[] = e.data.rows
    f.operators = (f.operators ?? []).map((o: Rec) => {
      const r = rows.find((x) => x.op === o.id)
      return r ? { ...o, step: r.step, loss: r.loss, eps: r.eps, clip_frac: r.clip_frac, gpu: r.gpu, status: r.status } : o
    })
    f.step_series = cap(f.step_series, {
      t: e.sim_t, round: e.data.round,
      loss: Object.fromEntries(rows.map((r) => [r.op, r.loss])), eps: Object.fromEntries(rows.map((r) => [r.op, r.eps])),
    }, 600)
  },
  'fl.gradhist': (e, _d, draft, cur) => {
    draft.live = { ...(draft.live ?? cur.live), gradhist: e.data }
  },
  'fl.round.phase': (e, d) => {
    const f = d('federation')
    f.current = e.data.current
    f.phase = e.data.phase
    f.round = e.data.round
  },
  'fl.round': (e, d) => {
    const f = d('federation')
    f.rounds_log = [...(f.rounds_log ?? []), e.data]
    f.global = { val_loss: e.data.val_loss, accuracy: e.data.accuracy }
    f.current = null
    f.phase = 'done'
  },
  'fl.secagg': (e, d, draft, cur) => {
    d('federation').secagg = e.data
    draft.live = { ...(draft.live ?? cur.live), lastSecagg: e.data }
  },
  'fl.checkpoint': (e, d) => {
    const f = d('federation')
    f.checkpoints = upsert(f.checkpoints, e.data)
    f.annotations = cap(f.annotations, { round: e.data.round, label: e.data.id, kind: 'checkpoint' }, 60)
  },
  'fl.promote': (e, d) => {
    const f = d('federation')
    f.checkpoints = upsert(f.checkpoints, { ...e.data, promoted: true })
  },
  'fl.config': (e, d) => {
    const f = d('federation')
    f.config = e.data.config
    f.annotations = cap(f.annotations, e.data.annotation, 60)
  },
  'fl.run': (e, d) => {
    const f = d('federation')
    Object.assign(f, { state: e.data.state, round: e.data.round, phase: e.data.phase, config: e.data.config })
  },
  'fl.upload': (e, d) => {
    const f = d('federation')
    const s = f.sovereignty ?? { adapter_bytes: 0, metric_bytes: 0, raw_bytes: 0, per_op: {} }
    const per = { ...(s.per_op ?? {}) }
    const cur = per[e.data.op] ?? { adapter: 0, metrics: 0, raw: 0 }
    per[e.data.op] = { ...cur, adapter: cur.adapter + e.data.bytes, metrics: cur.metrics + 6144 }
    f.sovereignty = { ...s, adapter_bytes: s.adapter_bytes + e.data.bytes, metric_bytes: s.metric_bytes + 6144, per_op: per }
    f.operators = (f.operators ?? []).map((o: Rec) => (o.id === e.data.op ? { ...o, bytes_up: o.bytes_up + e.data.bytes } : o))
  },
  'fl.client': (e, d) => {
    const f = d('federation')
    f.operators = (f.operators ?? []).map((o: Rec) => {
      if (o.id !== e.data.op) return o
      if (e.data.action === 'drop') return { ...o, status: 'dropped', participating: false }
      if (e.data.action === 'pause') return { ...o, participating: !o.participating }
      if (e.data.action === 'poison') return { ...o, poison_next: true }
      return o
    })
  },
  'fl.budget': (e, d) => {
    if (e.severity !== 'critical') return
    const f = d('federation')
    f.operators = (f.operators ?? []).map((o: Rec) => (o.id === e.data.op ? { ...o, status: 'budget_exhausted' } : o))
  },
  // ---------------------------------------------------------------- verify
  'eval.candidate': (e, d) => {
    const p = d('proving')
    p.models = upsert(p.models, e.data.model)
    p.candidate = e.data.model.id
  },
  'eval.run': (e, d) => { d('proving').active = e.data.run },
  'eval.stage': (e, d) => { d('proving').active = e.data.run },
  'eval.progress': (e, d) => { d('proving').active = e.data.run },
  'eval.gate': (e, d) => {
    const p = d('proving')
    p.active = e.data.run
    p.models = upsert(p.models, e.data.model)
    p.runs = cap(p.runs, e.data.run, 20)
  },
  'eval.federated': (e, d) => {
    const p = d('proving')
    p.fed_evals = { ...(p.fed_evals ?? {}), [e.data.model]: e.data.rows }
  },
  'aibom.signed': (e, d) => {
    const p = d('proving')
    p.aiboms = { ...(p.aiboms ?? {}), [e.data.model]: e.data.aibom }
  },
  // ---------------------------------------------------------------- release
  'release.metrics': (e, d) => { Object.assign(d('release'), e.data) },
  'release.stage': (e, d) => {
    const r = d('release')
    Object.assign(r, e.data)
    r.history = cap(r.history, { version: e.data.candidate ?? e.data.prod, event: e.summary, ts: e.ts,
      ledger_seq: e.ledger?.seq }, 40)
  },
  // ---------------------------------------------------------------- operate
  'twin.tick': (e, d) => {
    const t = d('twin')
    t.state = e.data.sites
    t.kpis = e.data.kpis
    if (e.data.hubs) t.hubs = e.data.hubs
    if (e.data.links) t.links = e.data.links
    t.health_series = cap(t.health_series, { t: e.sim_t, ...e.data.kpis }, 900)
  },
  'twin.alarm': (e, d) => { const t = d('twin'); t.alarms = cap(t.alarms, e.data, 60) },
  'twin.action': (e, d) => { const t = d('twin'); t.actions = cap(t.actions, e.data, 60) },
  'swarm.agents': (e, d) => {
    const s = d('swarm')
    const byId = new Map<string, Rec>((s.agents ?? []).map((a: Rec) => [a.id, a]))
    for (const a of e.data.agents) byId.set(a.id, { ...byId.get(a.id), ...a })
    s.agents = [...byId.values()]
    s.breakers = e.data.breakers
    if (e.data.trust_point) {
      s.trust_series = cap(s.trust_series, e.data.trust_point, 400)
      s.class_trust = e.data.class_trust
    }
  },
  'a2a.msg': (e, d) => { const s = d('swarm'); s.messages = cap(s.messages, e.data, 300) },
  'swarm.freeze': (e, d) => { d('swarm').frozen = e.data.frozen },
  'autonomy.change': (e, d) => {
    const s = d('swarm')
    s.autonomy = { ...(s.autonomy ?? {}), [e.data.class]: e.data.to }
  },
  // ---------------------------------------------------------------- prove
  'policy.decision': (e, d) => {
    const t = d('trust')
    t.decisions = cap(t.decisions, e.data, 300)
    if (e.data.decision) t.stats = { ...(t.stats ?? {}), [e.data.decision]: ((t.stats ?? {})[e.data.decision] ?? 0) + 1 }
  },
  'guardrail.block': (e, d) => {
    const t = d('trust')
    t.guardrail_blocks = cap(t.guardrail_blocks, e.data, 60)
    t.guardrail_counts = { ...(t.guardrail_counts ?? {}), prompt_attack: ((t.guardrail_counts ?? {}).prompt_attack ?? 0) + 1 }
  },
  'delegation.check': (e, d) => { const t = d('trust'); t.delegations = cap(t.delegations, e.data, 60) },
  'ledger.append': (e, d) => {
    const l = d('ledger')
    const rec = e.data.record
    l.records = cap(l.records, rec, 600)
    l.height = rec.seq
    l.head = rec.hash
    l.type_counts = { ...(l.type_counts ?? {}), [rec.type]: ((l.type_counts ?? {})[rec.type] ?? 0) + 1 }
  },
  'ledger.anchor': (e, d) => {
    const l = d('ledger')
    l.anchors = cap(l.anchors, e.data.anchor, 120)
    l.anchors_total = e.data.anchor.n
  },
  'ledger.verify': (e, d) => { d('ledger').last_verify = e.data },
  'ledger.tamper': (e, d) => { const l = d('ledger'); l.tampered = [...(l.tampered ?? []), e.data.seq] },
  'ledger.restore': (_e, d) => { d('ledger').tampered = [] },
  'approval.requested': (e, d) => { const o = d('oversight'); o.items = [e.data, ...(o.items ?? []).filter((x: Rec) => x.id !== e.data.id)] },
  'approval.decision': (e, d) => { const o = d('oversight'); o.items = upsert(o.items, e.data) },
  escalation: (e, d) => { const o = d('oversight'); o.items = upsert(o.items, e.data) },
  // ---------------------------------------------------------------- observability
  'obs.metrics': (e, d) => {
    const o = d('observability')
    const { heat, ...frame } = e.data
    o.frames = cap(o.frames, frame, 300)
    if (heat) o.heat = cap(o.heat, heat, 180)
    o.cost_total = e.data.cost_total
    o.tokens_total = e.data.tokens_total
  },
  'obs.span': (e, d) => {
    const o = d('observability')
    const tid = e.data.trace_id
    o.spans = { ...(o.spans ?? {}), [tid]: [...((o.spans ?? {})[tid] ?? []), e.data] }
  },
  'obs.trace': (e, d) => {
    const o = d('observability')
    o.traces = [...(o.traces ?? []).filter((t: Rec) => t.trace_id !== e.data.trace_id), e.data].slice(-30)
  },
  // ---------------------------------------------------------------- director / system
  'director.beat': (e, d) => { Object.assign(d('director'), e.data.director) },
}

// incident-shaped events all carry the incident view
for (const t of ['incident.opened', 'incident.phase', 'incident.resolved', 'negotiation.round', 'negotiation.resolved',
  'plan.preflight', 'plan.step']) {
  H[t] = (e, d) => {
    const s = d('swarm')
    s.incident = e.data
    if (t === 'incident.resolved' || t === 'incident.opened') {
      s.incidents = upsert(s.incidents, { id: e.data.id, title: e.data.title, severity: e.data.severity,
        status: e.data.status, mttr_s: e.data.mttr_s, started_at: e.data.started_at, cost_usd: e.data.cost_usd })
    }
  }
}

export function applyEvents(events: Envelope[]) {
  const cur = useLive.getState()
  const draft: Draft = {}
  const touched = new Map<SliceKey, Rec>()
  const d = (k: SliceKey) => {
    let s = touched.get(k)
    if (!s) {
      s = { ...(cur[k] as Rec) }
      touched.set(k, s)
    }
    return s
  }
  let stream: Envelope[] | null = null
  let pulses: LiveState['live']['pulses'] | null = null
  for (const e of events) {
    const h = H[e.type]
    if (h) {
      try {
        h(e, d, draft, cur)
      } catch (err) {
        console.warn('[spectrum] handler failed', e.type, err)
      }
    }
    if (e.type === 'sim.clock') draft.clock = { ...e.data, syncedAt: performance.now() } as LiveState['clock']
    if (!NOISY.has(e.type)) {
      stream = stream ?? [...cur.stream]
      stream.push(e)
    }
    const p = PULSE[e.type]
    if (p) {
      pulses = pulses ?? [...cur.live.pulses]
      pulses.push({ from: p[0], to: p[1], at: performance.now() })
    }
  }
  for (const [k, v] of touched) (draft as Rec)[k] = v
  if (stream) draft.stream = stream.length > STREAM_MAX ? stream.slice(stream.length - STREAM_MAX) : stream
  if (pulses) draft.live = { ...(draft.live ?? cur.live), pulses: pulses.filter((x) => performance.now() - x.at < 3000) }
  draft.lastMsgAt = performance.now()
  useLive.setState(draft)
}
