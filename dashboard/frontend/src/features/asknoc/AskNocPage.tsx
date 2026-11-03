import clsx from 'clsx'
import { Plus, RefreshCw } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import type { Rec } from '@/core/types'
import { Button, PageHeader, Panel } from '@/ui'
import { ask, CASE_RE, newKey } from './api'
import manifest from './manifest'
import { Activity, CaseFacts, Composer, HealthBanner, LabelChip, Thread } from './parts'
import type { Ctx } from './states'

// Polling, all against this laptop's backend (which alone talks to the kit, one poller per case):
//   health on open and every 60 s; the case list every 5 s; the open case every 2 s while noc_lead works,
//   else every 15 s; a just-sent start every 1 s until it has a case id; activity every 10 s while working.
// Every timer is cleared when the page closes. Leaving the page stops nothing on the kit.

const HEALTH_MS = 60_000
const LIST_MS = 5_000

function usePoll(fn: (() => Promise<number | null>) | null, deps: unknown[]) {
  useEffect(() => {
    if (!fn) return
    let alive = true
    let timer: ReturnType<typeof setTimeout> | undefined
    const run = async () => {
      let next: number | null = null
      try {
        next = await fn()
      } catch {
        next = 5000
      }
      if (alive && next != null) timer = setTimeout(run, next)
    }
    void run()
    return () => { alive = false; if (timer) clearTimeout(timer) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
}

export default function AskNocPage() {
  const [params, setParams] = useSearchParams()
  const q = params.get('case')
  const caseId = q && CASE_RE.test(q) ? q : null
  const select = useCallback((id: string | null) => setParams(id ? { case: id } : {}, { replace: true }), [setParams])

  const [health, setHealth] = useState<Rec | null>(null)
  const [list, setList] = useState<Rec | null>(null)
  const [viewRaw, setView] = useState<Rec | null>(null)
  const [actRaw, setAct] = useState<{ id: string; data: Rec } | null>(null)
  const [pending, setPending] = useState<Rec | null>(null)
  const [viewErr, setViewErr] = useState<string | null>(null)
  const [bump, setBump] = useState(0)
  const [refreshing, setRefreshing] = useState(false)

  const loadHealth = useCallback(async () => {
    try {
      setHealth(await ask.health())
    } catch (e) {
      setHealth({ mode: 'down', live_possible: false, checks: [{ id: 'backend', state: 'fail', detail: String((e as Error).message) }] })
    }
  }, [])
  usePoll(async () => { await loadHealth(); return HEALTH_MS }, [loadHealth])

  const local = !!health && health.mode !== 'loopback_only'

  usePoll(local ? async () => { setList(await ask.cases()); return LIST_MS } : null, [local, bump])

  usePoll(local && caseId ? async () => {
    try {
      const v = await ask.caseView(caseId)
      setView({ ...v, _for: caseId })
      setViewErr(null)
      return (v.turns ?? []).some((t: Rec) => !t.terminal) ? 2000 : 15000
    } catch (e) {
      setViewErr(String((e as Error).message))
      return 10000
    }
  } : null, [local, caseId, bump])

  // a view or activity fetched for another case is never shown for this one
  const view = viewRaw && viewRaw._for === caseId ? viewRaw : null
  const act = actRaw && actRaw.id === caseId ? actRaw.data : null
  const working = !!view && (view.turns ?? []).some((t: Rec) => !t.terminal)
  usePoll(local && caseId ? async () => { setAct({ id: caseId, data: await ask.activity(caseId) }); return working ? 10000 : 30000 } : null, [local, caseId, working, bump])

  usePoll(local && pending && !pending.terminal && !pending.case_id ? async () => {
    const t = await ask.turn(pending.turn_id)
    setPending(t)
    if (t.case_id && t.ui_state === 'running') {
      setPending(null)
      select(t.case_id)
      return null
    }
    return t.terminal ? null : 1000
  } : null, [local, pending?.turn_id, pending?.terminal])

  const ctx: Ctx = useMemo(() => ({ measured: health?.measured, timing: health?.timing, fallback: health?.fallback }), [health])
  const suite = ((health?.checks ?? []) as Rec[]).find((c) => c.id === 'suite_run') ?? null

  const send = async (text: string) => {
    const key = newKey()
    const once = () => (caseId ? ask.followUp(caseId, text, key) : ask.start(text, key))
    let out: Rec
    try {
      out = await once()
    } catch (e) {
      if (e instanceof TypeError) out = await once() // network blip: same key, so the backend never sends twice
      else throw e
    }
    if (caseId) setBump((b) => b + 1)
    else setPending(out)
  }

  const stop = async (turnId: string) => {
    try {
      const t = await ask.stop(turnId)
      if (pending?.turn_id === turnId) setPending(t)
    } finally {
      setBump((b) => b + 1)
    }
  }

  const refresh = async () => {
    setRefreshing(true)
    await loadHealth()
    setBump((b) => b + 1)
    setRefreshing(false)
  }

  const turns: Rec[] = caseId ? (view?.turns ?? []) : (pending ? [pending] : [])
  const blockedWhy = !health ? 'Checking the kit…'
    : !health.live_possible ? 'Sending is off until the kit can be used (see above).'
      : caseId && view && !view.can_continue ? view.cannot_continue_reason
        : !caseId && pending && !pending.terminal ? 'Waiting for the kit to open the case…' : null
  const cases = (list?.cases ?? []) as Rec[]
  const label = health?.mode === 'offline' ? 'OFFLINE' : health?.real_gateway === false ? 'TEST' : health?.live_possible ? 'LIVE' : 'NOT CONNECTED'

  if (health?.mode === 'loopback_only') {
    return (
      <div className="min-w-0" data-slice="asknoc">
        <PageHeader title={manifest.title} tracks={manifest.tracks} description={manifest.description} />
        <HealthBanner health={health} />
      </div>
    )
  }

  return (
    <div className="min-w-0" data-slice="asknoc">
      <PageHeader title={manifest.title} tracks={manifest.tracks} description={manifest.description}
        actions={<>
          <LabelChip label={label} title={label === 'LIVE' ? 'Replies come from the real checkpoint kit.' : 'No calls are made.'} />
          {health && <span className="text-cap text-ink-3">kit: <span className="mono text-ink-1">{health.mode}</span></span>}
          <Button icon={<RefreshCw size={14} className={clsx(refreshing && 'animate-spin')} />} disabled={refreshing} onClick={refresh}>Refresh</Button>
        </>} />
      <HealthBanner health={health} />
      <div className="grid min-w-0 gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <Panel id="asknoc-thread" plane="operate" title="Conversation"
          subtitle={caseId ? <span className="mono">{caseId}</span> : 'new case'}
          actions={<Button size="sm" icon={<Plus size={13} />} onClick={() => { setPending(null); select(null) }}>New case</Button>}>
          <div className="p-4">
            {cases.length > 0 && (
              <select aria-label="Open a case" value={caseId ?? ''} onChange={(e) => { setPending(null); select(e.target.value || null) }}
                className="mb-3 h-8 w-full rounded-md border border-line bg-s2 px-2 text-cap text-ink-1">
                <option value="">New case</option>
                {cases.map((c) => <option key={c.case_id} value={c.case_id}>{c.case_id} · {c.label} · {c.ui_state} · {c.first_text}</option>)}
              </select>
            )}
            {viewErr && <div role="alert" className="mb-2 text-cap text-critical-ink">{viewErr}</div>}
            <Thread turns={turns} ctx={ctx} onStop={stop} />
            <Composer onSend={send} disabled={!!blockedWhy} why={blockedWhy}
              placeholder={caseId ? 'Follow up with noc_lead…' : 'Describe an incident or ask a question, e.g. “Cell 2 users report poor coverage at the cell edge. Find the cause.”'} />
          </div>
        </Panel>
        <div className="flex min-w-0 flex-col gap-4">
          <Panel id="asknoc-facts" plane="operate" title="Case">
            <div className="p-4">
              {caseId && view ? <CaseFacts view={view} suite={suite} activity={act} />
                : <div className="text-cap text-ink-3">{caseId ? 'Loading…' : 'A case id appears here once the kit opens the case. The kit uses it as the trace id.'}
                  {suite?.state === 'warn' && <div className="mt-2 rounded-md border border-warn/60 bg-warn/10 px-3 py-2 text-ink-1">{suite.detail}</div>}</div>}
            </div>
          </Panel>
          <Panel id="asknoc-activity" plane="operate" title="What the swarm did" subtitle="governed calls, from the kit's audit log">
            <div className="p-4">{caseId ? <Activity data={act} /> : <div className="text-cap text-ink-3">Open a case to see its calls.</div>}</div>
          </Panel>
        </div>
      </div>
    </div>
  )
}
