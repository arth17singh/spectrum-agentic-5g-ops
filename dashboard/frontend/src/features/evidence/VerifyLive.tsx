import { CircleCheck, CircleDashed, CircleHelp, CircleX, Loader2, ShieldCheck } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { accessToken } from '@/core/access'
import type { Rec } from '@/core/types'
import { Button } from '@/ui'

// "Verify live on AWS": POST /api/evidence/verify/{run_id} runs tools/check_unit.py read-only against S3,
// CloudWatch Logs and SageMaker with the server's own AWS environment, writes the result where the sync
// caches checks (so every screen updates), and returns parts a-e. Laptop only; one check at a time.
// Outcomes are shown as they are: CANNOT CHECK stays CANNOT CHECK, with its reason.

const PART: Record<string, { Icon: typeof CircleCheck; color: string }> = {
  PASS: { Icon: CircleCheck, color: 'var(--good-ink)' },
  FAIL: { Icon: CircleX, color: 'var(--critical-ink)' },
  'CANNOT CHECK': { Icon: CircleHelp, color: 'var(--warn-ink)' },
  SKIP: { Icon: CircleDashed, color: 'var(--ink-3)' },
}

/** Where an AWS time came from, in words. */
function sourceLabel(src?: string | null): string {
  const s = String(src ?? '')
  if (/team13-evidence|attest/i.test(s)) return 'CloudWatch job attestation'
  if (/createdAt/i.test(s)) return 'policy createdAt'
  if (/^s3:LastModified/i.test(s)) return 'S3 LastModified'
  if (/^sagemaker:/i.test(s)) return 'SageMaker job window'
  if (/^cloudwatch:|^logs?:/i.test(s)) return 'CloudWatch log event'
  return s.split(' ')[0] || 'unknown source'
}

const utc = (iso?: string | null) => (!iso ? '—' : `${iso.slice(0, 10)} ${iso.slice(11, 19)}Z`)

async function call(runId: string, method: 'GET' | 'POST'): Promise<{ ok: boolean; status: number; body: Rec }> {
  const headers: Record<string, string> = { 'X-Spectrum-Client': 'verify-live' }
  const t = accessToken()
  if (t) headers['X-Spectrum-Token'] = t
  let res: Response
  try {
    res = await fetch(`/api/evidence/verify/${encodeURIComponent(runId)}`, { method, headers })
  } catch {
    return { ok: false, status: 0, body: { message: 'SPECTRUM backend not reachable' } }
  }
  let body: Rec = {}
  try { body = await res.json() } catch { body = { message: `HTTP ${res.status}` } }
  return { ok: res.ok, status: res.status, body }
}

/** "Verified" only when every part passed. A failed part and a part that could not be checked are said as they are. */
function headline(res: Rec, retained: boolean): string {
  const parts: Rec[] = res.parts ?? []
  const ids = (state: string) => parts.filter((p) => p.state === state).map((p) => String(p.id))
  const list = (xs: string[]) => (xs.length ? xs.join(xs.length > 2 ? ', ' : ' and ') : '')
  const at = `at ${utc(res.checked_at)}`
  const how = retained ? '' : ' live against AWS'
  if (res.exit_code === 0) return `Verified${how} ${at} · all ${parts.length || 5} parts passed`
  if (res.exit_code === 1) {
    const f = ids('FAIL')
    return `Checked${how} ${at} · ${f.length > 1 ? 'parts' : 'a part'} ${f.length ? `${list(f)} ` : ''}FAILED`
  }
  if (res.exit_code === 2) {
    const c = ids('CANNOT CHECK')
    return `Checked${how} ${at} · nothing failed; ${c.length ? `part${c.length > 1 ? 's' : ''} ${list(c)}` : 'a part'} could not be checked`
  }
  return `Checked${how} ${at} · the check did not complete`
}

export function VerifyLive({ runId, compact = false }: { runId: string; compact?: boolean }) {
  const [phase, setPhase] = useState<'idle' | 'running' | 'done' | 'error'>('idle')
  const [t0, setT0] = useState(0)
  const [now, setNow] = useState(0)
  const [res, setRes] = useState<Rec | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  const [retained, setRetained] = useState<string | null>(null)
  const [open, setOpen] = useState(!compact)
  const timer = useRef<number | null>(null)

  useEffect(() => () => { if (timer.current) window.clearInterval(timer.current) }, [])

  const run = async () => {
    setPhase('running'); setMsg(null); setRetained(null); setOpen(true)
    const start = Date.now()
    setT0(start); setNow(start)
    timer.current = window.setInterval(() => setNow(Date.now()), 500)
    const r = await call(runId, 'POST')
    if (timer.current) window.clearInterval(timer.current)
    // never shown as newer than it is: when the live check could not complete, the cached result stays, said so
    const cached: Rec | undefined = r.body.cache === 'retained' ? r.body.cached : undefined
    if (cached) {
      const why = r.ok ? `a part could not be checked live (${r.body.verdict})` : (r.body.message ?? `HTTP ${r.status}`)
      setRetained(`Live check could not complete: ${why}. Showing the result verified at ${utc(cached.checked_at)}.`)
      setRes(cached); setPhase('done')
    } else if (r.ok) { setRes(r.body); setPhase('done') } else { setMsg(r.body.message ?? `HTTP ${r.status}`); setPhase('error') }
  }

  const elapsed = Math.max(0, Math.round((now - t0) / 1000))
  return (
    <div className="min-w-0" data-testid="verify-live" data-run={runId}>
      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" icon={phase === 'running' ? <Loader2 size={13} className="animate-spin" /> : <ShieldCheck size={13} />}
          disabled={phase === 'running'} onClick={run} title="Run check_unit now, read-only, against S3, CloudWatch and SageMaker">
          Verify live on AWS
        </Button>
        {phase === 'running' && <span className="tnum text-micro text-ink-2">Checking S3, CloudWatch and SageMaker… {elapsed} s</span>}
        {phase === 'error' && <span className="text-micro text-critical-ink">{msg}</span>}
        {phase === 'done' && compact && (
          <button type="button" className="text-micro text-ink-3 underline decoration-dotted" onClick={() => setOpen(!open)}>{open ? 'hide' : 'show'} result</button>
        )}
      </div>
      {phase === 'done' && res && open && (
        <div className="mt-2 min-w-0 rounded-md border border-line bg-s2/50 p-2.5 text-micro [overflow-wrap:anywhere]" data-testid="verify-live-result">
          {retained && <div className="mb-1 text-warn-ink" data-testid="verify-live-retained">{retained}</div>}
          <div className="mb-1 font-medium text-ink-1" data-testid="verify-live-header">{headline(res, !!retained)}{res.elapsed_s != null ? ` · ${res.elapsed_s} s` : ''}</div>
          {res.attest && <div className="mb-1 text-ink-3">Job-log attestation cache: {res.attest}</div>}
          {res.error && <div className="text-critical-ink">{res.error}</div>}
          <ul className="space-y-1">
            {(res.parts ?? []).map((p: Rec) => {
              const s = PART[p.state] ?? PART.SKIP
              return (
                <li key={p.id}>
                  <div className="flex items-center gap-1.5 text-ink-1"><s.Icon size={12} style={{ color: s.color }} aria-hidden /><b>[{p.id}]</b> {p.title} <span style={{ color: s.color }}>{p.state}</span></div>
                  {(p.lines ?? []).slice(0, p.state === 'PASS' ? 2 : 6).map((l: string, i: number) => <div key={i} className="mono pl-5 text-[11px] text-ink-3">{l}</div>)}
                </li>
              )
            })}
          </ul>
          {(res.derived_aws_times ?? []).length > 0 && (
            <div className="mt-2">
              <div className="text-ink-2">AWS times derived for records written without one ({res.n_derived_aws_times}):</div>
              {Object.entries((res.derived_aws_times as Rec[]).reduce((m: Record<string, number>, t) => { const k = sourceLabel(t.source); m[k] = (m[k] ?? 0) + 1; return m }, {}))
                .map(([k, n]) => <div key={k} className="pl-5 text-ink-3">{n} from {k}</div>)}
              <div className="mono pl-5 text-[11px] text-ink-3">e.g. {res.derived_aws_times[0].record}: {utc(res.derived_aws_times[0].aws_time)} ({sourceLabel(res.derived_aws_times[0].source)})</div>
            </div>
          )}
          {Array.isArray(res.window) && <div className="mt-1 text-ink-3">AWS window {utc(res.window[0])} → {utc(res.window[1])}{(res.window_sources ?? []).length ? ` (${(res.window_sources as unknown[][]).map((w) => sourceLabel(String(w[0]))).join(', ')})` : ''}</div>}
        </div>
      )}
    </div>
  )
}
