import clsx from 'clsx'
import { ExternalLink, Play, ShieldCheck, Zap } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import type { Rec } from '@/core/types'
import { Button, PageHeader, Panel, Segmented, Steps } from '@/ui'
import QRCode from 'qrcode'
import { accessToken, withToken } from '@/core/access'
import { useLive } from '@/core/store'
import { DecideForm } from '../kitapprovals/DecideForm'
import { kit, PIN_COMMAND, type Detail, type Status as KitStatus } from '../kitapprovals/api'
import { CELL_POS, hexPath } from '../swarm/hex'
import manifest from './manifest'

// Live Fix (/api/livefix): one fault, end to end. LIVE = the real kit (Gateway, Cedar, agents, approvals table) on
// the team's in-process Maveric twin; RECORDED = the judged swarm run's records, replayed. The page never approves:
// the decision is the person's, through the Approvals sheet (PIN) or the kit's CLI.

type Mode = 'live' | 'recorded'
const POLL_MS = 1500

async function lf<T = Rec>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`/api/livefix${path}`, body === undefined ? { cache: 'no-store' } : {
    method: 'POST', cache: 'no-store',
    headers: { 'Content-Type': 'application/json', 'X-Spectrum-Client': 'livefix' }, body: JSON.stringify(body),
  })
  const data = await res.json().catch(() => null)
  if (!res.ok) throw new Error(data?.detail?.message ?? `The backend answered ${res.status}.`)
  return data as T
}

const TONE: Record<string, string> = {
  green: 'color-mix(in oklab, var(--good) 34%, var(--surface-2))',
  amber: 'color-mix(in oklab, var(--warn) 40%, var(--surface-2))',
  red: 'color-mix(in oklab, var(--critical) 45%, var(--surface-2))',
  unknown: 'var(--surface-2)',
}
const pct = (v: unknown) => (typeof v === 'number' ? `${(v * 100).toFixed(0)}%` : '—')
const num = (v: unknown, d = 3) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : '—')

function Cells({ cells, fault, size = 340 }: { cells: Rec[]; fault: number[]; size?: number }) {
  const R = size / 7.2
  const ox = R * 1.3, oy = R * 2.0
  const by = Object.fromEntries((cells ?? []).map((c) => [c.cell, c]))
  return (
    <svg viewBox={`0 0 ${size} ${size * 0.95}`} className="h-auto w-full max-w-[420px]" role="img"
      aria-label="The 4 twin cells, coloured by how their users are served against the healthy network">
      {CELL_POS.map(([dx, dy], cell) => {
        const cx = ox + dx * R * 1.12, cy = oy + dy * R * 1.12
        const c = by[cell] ?? { status: 'unknown' }
        const f = fault.includes(cell)
        return (
          <g key={cell} data-cell={cell} data-status={c.status}>
            <path d={hexPath(cx, cy, R)} style={{ fill: TONE[c.status] ?? TONE.unknown, stroke: f ? 'var(--critical)' : 'var(--hairline)', transition: 'fill 600ms' }}
              strokeWidth={f ? 2.5 : 1.2} strokeDasharray={f ? '5 3' : undefined} />
            <text x={cx} y={cy - R * 0.45} textAnchor="middle" style={{ fill: 'var(--ink-1)', fontSize: 12, fontWeight: 600 }}>cell {cell}</text>
            <text x={cx} y={cy + 2} textAnchor="middle" className="tnum" style={{ fill: 'var(--ink-1)', fontSize: 11 }}>
              {typeof c.coverage_share === 'number' ? `covered ${pct(c.coverage_share)}` : c.status}
            </text>
            {typeof c.sla_share === 'number' && <text x={cx} y={cy + 16} textAnchor="middle" className="tnum" style={{ fill: 'var(--ink-1)', fontSize: 11 }}>SLA {pct(c.sla_share)}</text>}
          </g>
        )
      })}
    </svg>
  )
}

const ORDER = ['healthy', 'faulted', 'diagnosing', 'awaiting_approval', 'applying', 'graded']
const LABEL: Record<string, string> = {
  healthy: 'Healthy', faulted: 'Fault injected', diagnosing: 'Swarm diagnoses', awaiting_approval: 'Cedar holds: a person decides',
  applying: 'Applied on the twin', graded: 'Graded vs healthy',
}
function steps(run: Rec | null) {
  const st: string = run?.state ?? 'idle'
  const at = st === 'preparing' ? -1 : ['recovered', 'not_recovered'].includes(st) ? 5 : ORDER.indexOf(st)
  const failed = ['rejected', 'no_proposal', 'failed', 'timed_out'].includes(st)
  const reached = new Set<string>((run?.stages ?? []).map((s: Rec) => String(s.stage)))
  const secs = Object.fromEntries((run?.stages ?? []).map((s: Rec) => [s.stage, s.seconds]))
  return ORDER.map((k, i) => {
    let state: 'done' | 'active' | 'pending' | 'failed' = 'pending'
    if (failed) state = reached.has(k) ? (k === [...reached].filter((x) => ORDER.includes(x)).pop() ? 'failed' : 'done') : 'pending'
    else if (k === 'graded') state = at === 5 ? (st === 'recovered' ? 'done' : 'failed') : 'pending'
    else if (i < at) state = 'done'
    else if (i === at) state = 'active'
    return { label: LABEL[k], state, detail: secs[k] !== undefined ? `${secs[k]} s` : undefined }
  })
}

function Kpis({ h }: { h: Rec }) {
  const rows: [string, string, number][] = [['coverage_ratio', 'Coverage', 3], ['sla_ratio', 'SLA', 3], ['sinr_p10_db', 'p10 SINR (dB)', 2]]
  const now = h?.final ?? h?.faulted
  const ref = h?.final ? h?.healthy_final : h?.healthy
  return (
    <table className="w-full text-cap">
      <thead><tr className="text-left text-micro text-ink-3"><th className="font-normal">KPI</th><th className="font-normal">healthy</th><th className="font-normal">{h?.final ? 'after the fix' : 'now'}</th></tr></thead>
      <tbody>
        {rows.map(([k, label, d]) => {
          const bad = now && ref && typeof now[k] === 'number' && typeof ref[k] === 'number' && now[k] < ref[k] - (k === 'sinr_p10_db' ? 0.5 : 0.02)
          return (
            <tr key={k} className="border-t border-line">
              <td className="py-1 text-ink-2">{label}</td>
              <td className="tnum text-ink-1">{num(ref?.[k], d)}</td>
              <td className="tnum font-medium" style={{ color: now ? (bad ? 'var(--critical-ink)' : 'var(--good-ink, var(--ink-1))') : 'var(--ink-3)' }}>{num(now?.[k], d)}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

function parseChange(text: unknown): { cell: number; dp: number; dt: number }[] {
  if (typeof text !== 'string') return []
  return text.split('|').map((p) => /cell=(\d+);d_power_db=([+-][\d.]+);d_tilt_deg=([+-][\d.]+)/.exec(p.trim()))
    .filter((m): m is RegExpExecArray => !!m).map((m) => ({ cell: Number(m[1]), dp: Number(m[2]), dt: Number(m[3]) }))
}
const sgn = (v: number) => `${v > 0 ? '+' : ''}${v.toFixed(1)}`

function PhoneQr({ approvalId }: { approvalId: string }) {
  const token: string | null = useLive((s) => s.config?.access_token) ?? accessToken()
  const [ph, setPh] = useState<Rec | null>(null)
  const [qr, setQr] = useState('')
  useEffect(() => { lf('/phone').then(setPh).catch(() => setPh(null)) }, [])
  const url = ph?.ready ? withToken(`http://${ph.lan_ip}:${ph.port}/m/kit?id=${encodeURIComponent(approvalId)}`, token) : ''
  useEffect(() => { if (url) QRCode.toDataURL(url, { margin: 1, width: 132 }).then(setQr).catch(() => setQr('')) }, [url])
  if (!ph) return null
  return (
    <div className="flex items-center gap-3 rounded-md border border-line bg-s1 p-2" data-testid="livefix-phone">
      {ph.ready && qr ? <img src={qr} alt="QR code: approve this change on a paired phone" width={92} height={92} className="rounded" />
        : <div className="grid size-[92px] place-items-center rounded border border-dashed border-line text-center text-micro text-ink-3">no phone link</div>}
      <div className="min-w-0 max-w-[220px] text-micro text-ink-2">
        <div className="text-cap font-medium text-ink-1">Approve on a phone</div>
        {ph.ready ? 'Scan on the same Wi-Fi: the same approval, PIN and audit record as here.' : ph.note}
      </div>
    </div>
  )
}

/** Inline approval: the change per cell, the agent's reason, and Approve / Deny with the approver PIN. The decision
 *  goes through /api/kitapprovals (the Approvals page's code path); this page never decides on its own. */
function ApprovalCard({ run, act }: { run: Rec; act: Rec | null }) {
  const a = run.approval
  const [row, setRow] = useState<Detail | null>(null)
  const [status, setStatus] = useState<KitStatus | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  const id: string | undefined = a?.approval_id
  const live = run.mode === 'live'
  useEffect(() => {
    if (!id || !live) return
    let stop = false
    const load = async () => {
      const [d, s] = await Promise.all([kit.detail(id), kit.status()])
      if (stop) return
      if (d.ok) setRow(d.data); else setMsg(d.fail.message)
      if (s.ok) setStatus(s.data)
    }
    load()
    const t = setInterval(() => { if (!document.hidden) load() }, 3000) // a phone or CLI decision shows up here
    return () => { stop = true; clearInterval(t) }
  }, [id, live])
  if (!a) return null
  const before: Rec = Object.fromEntries((run.cells_at_proposal ?? run.cells ?? []).map((c: Rec) => [c.cell, c]))
  const cells = parseChange(a.change_text)
  const awsTime = (act?.events ?? []).find((e: Rec) => e.kind === 'approval' && e.approval_id === id)?.time
  const tier: string | undefined = run.confidence?.tier
  const decided = live ? !!row && row.status !== 'PENDING' : run.state !== 'awaiting_approval'
  if (decided) {
    const st = row?.status ?? a.status
    return (
      <div className="rounded-md border border-line bg-s1 px-3 py-2 text-cap" data-testid="livefix-approval" data-state="decided">
        <ShieldCheck size={14} className="mr-1 inline text-ink-2" aria-hidden />
        <span className="font-medium text-ink-1">{String(st).toLowerCase()}</span>{' '}
        <span className="font-mono text-micro text-ink-3">{id}</span>
        {row?.decided_by && <> by <span className="text-ink-1">{row.decided_by}</span></>}
        {row?.decided_at && <> at {new Date(row.decided_at * 1000).toLocaleTimeString()}</>}
        {awsTime && <> · CloudWatch {String(awsTime).slice(11, 19)}Z</>}
        {row?.decision_reason && <span className="text-ink-3"> · “{row.decision_reason}”</span>}
        <span className="text-ink-3"> · {a.change_text}</span>
      </div>
    )
  }
  return (
    <div className="rounded-[10px] border-2 p-3" style={{ borderColor: 'color-mix(in oklab, var(--warn) 65%, transparent)' }} data-testid="livefix-approval" data-state="pending">
      <div className="flex flex-wrap items-center gap-2 text-cap">
        <ShieldCheck size={16} className="text-ink-2" aria-hidden />
        <span className="font-medium text-ink-1">Cedar held the change: a person decides</span>
        <span className="font-mono text-micro text-ink-3">{id}</span>
        {tier && <span className="rounded border border-line px-1.5 text-micro text-ink-2" title="confidence tier">tier {tier}</span>}
      </div>
      <table className="mt-2 w-full text-cap">
        <thead><tr className="text-left text-micro text-ink-3"><th className="font-normal">cell</th><th className="font-normal">power (dBm)</th><th className="font-normal">tilt (°)</th></tr></thead>
        <tbody>
          {cells.map((c) => {
            const b = before[c.cell] ?? {}
            const p = typeof b.power_dbm === 'number' ? b.power_dbm : null
            const t = typeof b.tilt_deg === 'number' ? b.tilt_deg : null
            return (
              <tr key={c.cell} className="border-t border-line">
                <td className="py-1 text-ink-2">cell {c.cell}</td>
                <td className="tnum text-ink-1">{p !== null ? `${p.toFixed(1)} → ${(p + c.dp).toFixed(1)}` : '—'} <span className="text-micro text-ink-3">({sgn(c.dp)})</span></td>
                <td className="tnum text-ink-1">{t !== null ? `${t.toFixed(1)} → ${(t + c.dt).toFixed(1)}` : '—'} <span className="text-micro text-ink-3">({sgn(c.dt)})</span></td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="mt-1 text-micro text-ink-3">Before = the twin now; after = before plus the requested change, before the SDK safety filter.</p>
      {a.reason && <p className="mt-1 text-cap text-ink-2">Agent's reason: {a.reason}</p>}
      {!live && <p className="mt-2 text-cap text-ink-3">RECORDED: in the judged run nobody approved this change. Press Continue (recorded) to see the scored outcome.</p>}
      {live && (
        <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-[1fr_auto]">
          <div>
            {status && !status.pin.set && <p className="mb-2 text-cap" style={{ color: 'var(--warn-ink)' }}>No approver PIN is set on this laptop. In a terminal: <code className="font-mono">{PIN_COMMAND}</code></p>}
            {status?.approver && <p className="mb-1 text-micro text-ink-3">Recorded as {status.approver.pseudonym}, audited in CloudWatch.</p>}
            {tier === 'two-person' && <p className="mb-2 text-cap" style={{ color: 'var(--warn-ink)' }}>Two-person tier: a second approver must also decide.</p>}
            {row && <DecideForm approvalId={row.approval_id} changeSha256={row.change_sha256} compact onDone={() => { kit.detail(row.approval_id).then((d) => { if (d.ok) setRow(d.data) }) }} />}
            {msg && <p role="alert" className="mt-1 text-cap" style={{ color: 'var(--critical-ink)' }}>{msg}</p>}
            <p className="mt-2 text-micro text-ink-3">Also: <Link to={a.approve_link} className="underline decoration-dotted">the Approvals page</Link> or <code className="font-mono">checkpoint_kit.py approve {id} "why"</code></p>
          </div>
          <PhoneQr approvalId={id!} />
        </div>
      )}
    </div>
  )
}

export default function LiveFixPage() {
  const [mode, setMode] = useState<Mode>('live')
  const [inc, setInc] = useState<Rec | null>(null)
  const [pick, setPick] = useState<string>('')
  const [run, setRun] = useState<Rec | null>(null)
  const [act, setAct] = useState<Rec | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => { lf('/incidents').then(setInc).catch((e) => setErr(String(e.message ?? e))) }, [])
  useEffect(() => {
    lf<Rec>('/runs').then((r) => { const last = (r.runs ?? []).at(-1); if (last && !last.end) setRun(last) }).catch(() => {})
  }, [])
  const list: Rec[] = (mode === 'live' ? inc?.live : inc?.recorded) ?? []
  const chosen = pick && list.some((i) => i.incident_id === pick) ? pick : list[0]?.incident_id ?? ''

  const poll = useCallback(async (id: string) => {
    try { setRun(await lf(`/runs/${encodeURIComponent(id)}`)) } catch (e) { setErr(String((e as Error).message)) }
  }, [])
  useEffect(() => {
    if (!run?.run_id || run.end) return
    const t = setInterval(() => { if (!document.hidden) poll(run.run_id) }, POLL_MS)
    return () => clearInterval(t)
  }, [run?.run_id, run?.end, poll])
  useEffect(() => {
    if (!run?.run_id || run.mode !== 'live' || !run.cases?.faulted) return
    const get = () => lf(`/runs/${encodeURIComponent(run.run_id)}/activity`).then(setAct).catch(() => {})
    get()
    if (run.end) return
    const t = setInterval(() => { if (!document.hidden) get() }, 6000)
    return () => clearInterval(t)
  }, [run?.run_id, run?.mode, run?.cases?.faulted, run?.end])

  const act0 = async (fn: () => Promise<Rec>) => {
    setBusy(true); setErr(null)
    try { setRun(await fn()) } catch (e) { setErr(String((e as Error).message)) } finally { setBusy(false) }
  }
  const start = () => { setAct(null); act0(() => lf('/runs', { incident_id: chosen, mode })) }
  const inject = () => run && act0(() => lf(`/runs/${encodeURIComponent(run.run_id)}/inject`, {}))
  const cont = () => run && act0(() => lf(`/runs/${encodeURIComponent(run.run_id)}/continue`, {}))
  const running = run && !run.end
  const label = run?.label ?? (mode === 'live' ? 'LIVE' : 'RECORDED')
  const incident = run?.incident ?? list.find((i) => i.incident_id === chosen)

  return (
    <div className="min-w-0" data-slice="livefix">
      <PageHeader title={manifest.title} description={manifest.description} tracks={manifest.tracks}
        actions={<span data-testid="livefix-mode" className="rounded border px-2 py-0.5 text-cap font-semibold"
          style={{ color: label === 'LIVE' ? 'var(--good-ink, var(--ink-1))' : 'var(--warn-ink)', borderColor: 'currentColor' }}
          title={label === 'LIVE' ? 'Real calls to the kit and the twin' : 'A replay of recorded runs: no AWS call'}>{label}</span>} />
      {err && <div role="alert" className="mb-3 rounded-md border border-critical/60 bg-critical/10 p-3 text-cap text-ink-1">{err}</div>}

      <div className="mb-4 flex flex-wrap items-center gap-2 rounded-[10px] border border-line bg-s1 px-3 py-2">
        <Segmented<Mode> ariaLabel="Mode" value={mode} onChange={(m) => { if (!running) { setMode(m); setPick('') } }}
          options={[{ value: 'live', label: 'Live', title: inc && !inc.live_ready ? 'Live needs the kit and an AWS session on this laptop' : 'Real calls' }, { value: 'recorded', label: 'Recorded', title: 'Fallback: replay the judged run' }]} />
        <select aria-label="Incident" value={chosen} disabled={!!running} onChange={(e) => setPick(e.target.value)}
          className="h-8 max-w-full min-w-0 rounded-md border border-line bg-s2 px-2 text-cap text-ink-1">
          {list.map((i) => <option key={i.incident_id} value={i.incident_id}>{i.fault} · {i.preset}{i.success === false ? ' (not recovered in the judged run)' : ''}</option>)}
        </select>
        <Button icon={<Play size={14} />} disabled={busy || !!running || !chosen || (mode === 'live' && inc?.live_ready === false)} onClick={start}>Load network</Button>
        <Button variant="primary" icon={<Zap size={14} />} disabled={busy || run?.state !== 'healthy'} onClick={inject}>Inject fault</Button>
        {run?.mode === 'recorded' && run.state === 'awaiting_approval' && <Button onClick={cont}>Continue (recorded)</Button>}
        {mode === 'live' && inc?.live_ready === false && <span className="text-micro" style={{ color: 'var(--warn-ink)' }}>Live needs the kit folder and an AWS session on this laptop; Recorded still works.</span>}
      </div>

      {run && <div className="mb-3 rounded-[10px] border border-line bg-s1 px-3 py-3"><Steps steps={steps(run)} /></div>}
      {run?.approval && <div className="mb-4"><ApprovalCard run={run} act={act} /></div>}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <Panel title="The network" subtitle={run?.twin ?? 'Load a network to start'} className="lg:col-span-5">
          <Cells cells={run?.cells ?? []} fault={run && run.state !== 'healthy' && run.state !== 'preparing' ? (incident?.fault_cells ?? []) : []} />
          {incident && <p className="mt-1 text-micro text-ink-3">For the audience: the fault is {incident.fault} (dashed outline once injected). The agents are given symptoms only.</p>}
          {run?.headline && <div className="mt-3"><Kpis h={run.headline} /></div>}
          {run?.grade && (
            <p className="mt-2 text-cap" data-testid="livefix-grade">
              {run.grade.success ? 'Recovered: every graded KPI is within tolerance of the healthy twin' : 'Not recovered: some graded KPI is outside tolerance'}
              {typeof run.grade.recovery === 'number' && <> · recovery {num(run.grade.recovery, 2)}</>}
            </p>
          )}
        </Panel>

        <div className="space-y-4 lg:col-span-7">
          <Panel title="What happened" subtitle={run ? `run ${run.run_id}` : undefined}
            actions={run && <Link to={run.evidence_link} className="inline-flex items-center gap-1 text-cap text-ink-2 underline decoration-dotted"><ExternalLink size={13} />Open evidence</Link>}>
            {!run && <p className="text-cap text-ink-3">Pick an incident, load the healthy network, then inject the fault.</p>}
            <ol className="space-y-1 text-cap">
              {(run?.events ?? []).map((e: Rec, i: number) => (
                <li key={i} className={clsx('flex gap-2', e.stage === 'failed' && 'text-critical-ink')}>
                  <span className="tnum shrink-0 text-micro text-ink-3">{String(e.t).slice(11, 19)}</span><span className="text-ink-1">{e.text}</span>
                </li>
              ))}
            </ol>
            {run?.problem && <p role="alert" className="mt-2 text-cap" style={{ color: 'var(--critical-ink)' }}>{run.problem}</p>}
            {run?.reply && <details className="mt-2"><summary className="cursor-pointer text-cap text-ink-2">noc_lead's last reply</summary><pre className="mt-1 text-micro whitespace-pre-wrap text-ink-2">{run.reply}</pre></details>}
            {run?.evidence && Object.keys(run.evidence).some((k) => k.endsWith('_error')) && (
              <p className="mt-2 text-micro" style={{ color: 'var(--warn-ink)' }}>Evidence: {Object.entries(run.evidence).filter(([k]) => k.endsWith('_error')).map(([k, v]) => `${k} ${v}`).join('; ')}</p>
            )}
          </Panel>
          {run?.mode === 'live' && (
            <Panel title="The agents' governed calls" subtitle={act?.source ?? 'the kit audit log in CloudWatch'}>
              {act?.lag_note && <p className="mb-1 text-micro text-ink-3">{act.lag_note}</p>}
              {!act?.events?.length && <p className="text-cap text-ink-3">{act?.message ?? 'No calls logged yet.'}</p>}
              <ol className="max-h-[300px] space-y-0.5 overflow-y-auto text-cap">
                {(act?.events ?? []).map((e: Rec, i: number) => (
                  <li key={i} className="flex flex-wrap gap-x-2">
                    <span className="tnum text-micro text-ink-3">{String(e.time).slice(11, 19)}</span>
                    <span className="text-ink-2">{e.agent ?? e.by ?? e.kind}</span>
                    <span className="font-mono text-micro text-ink-1">{e.tool ?? e.kind}{e.to ? ` → ${e.to}` : ''}</span>
                    {e.decision && <span className="text-micro font-semibold" style={{ color: e.decision === 'DENY' ? 'var(--critical-ink)' : 'var(--ink-2)' }}>{e.decision}{e.rule ? ` (${e.rule})` : ''}{e.approval ? ` · ${e.approval}` : ''}</span>}
                  </li>
                ))}
              </ol>
            </Panel>
          )}
          <p className="text-micro text-ink-3">Agents: {run?.models ?? '—'}. Approvals work only on this laptop (a person, the approver PIN, an AWS session); the page itself never decides.</p>
        </div>
      </div>
    </div>
  )
}
