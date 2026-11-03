// Client for /api/kitapprovals (laptop only). Errors come back as {error, message}; the caller shows them.

export type Mode = 'ok' | 'offline' | 'aws_session_expired' | 'no_access' | 'aws_unreachable' | 'aws_error' | 'laptop_only'

export interface Status {
  mode: Mode
  label: string
  message?: string
  approver: { pseudonym: string; aws_session: string } | null
  pin: { set: boolean; locked_for_s: number }
  audit_outbox: number
  note: string
}

export interface Row {
  approval_id: string
  status: 'PENDING' | 'APPROVED' | 'REJECTED' | 'APPLIED' | string
  agent: string | null
  case_id: string | null
  item_id: string | null
  change_text: string | null
  canonical_text: string | null
  canonical: boolean
  cells: { cell: number; d_power_db: number; d_tilt_deg: number }[] | null
  change_sha256: string
  created_at: number | null
  age_s: number | null
  decided_by: string | null
  decided_at: number | null
  decision_reason: string | null
  applied_at: number | null
  /** the case belongs to an evaluation run (t3/evals/live_swarm.py): recorded, scored offline, never approved */
  proposal_only: boolean
}

export interface Pending { mode: Mode; label: string; rows: Row[]; read_at: string; message?: string }

export interface Detail extends Row {
  label: string
  note: string
  agent_reason: string | null
  case: { run_id: string | null; step_now: number; twin_kind: string | null } | null
  case_note: string | null
  prediction: { predicted_delta: Record<string, number> | null; step: number; stale: boolean; predictor: { model_id: string; version: string }; source: string } | null
  prediction_note: string | null
  trust: { trust: number | null; gate_status: string; n: number | null; source: string } | null
  warnings: string[]
  links: { ask_noc: string; trace: string; twin_view: string }
}

export interface DecideOk { approval_id: string; status: string; decided_at: number; by: string; audit: 'written' | 'not_audited'; note: string | null; audit_message: string | null }
export interface Failure { error: string; message: string; status?: string; by?: string; decided_at?: number }

export type Result<T> = { ok: true; data: T } | { ok: false; code: number; fail: Failure }

async function call<T>(path: string, init?: RequestInit): Promise<Result<T>> {
  let res: Response
  try {
    res = await fetch(path, { cache: 'no-store', ...init })
  } catch (e) {
    return { ok: false, code: 0, fail: { error: 'network', message: `The console backend did not answer (${String(e)}).` } }
  }
  let body: unknown = null
  try { body = await res.json() } catch { /* not JSON */ }
  if (res.ok) return { ok: true, data: body as T }
  const f = (body && typeof body === 'object' && 'error' in body) ? body as Failure
    : { error: `http_${res.status}`, message: `The backend answered ${res.status}.` }
  return { ok: false, code: res.status, fail: f }
}

export const kit = {
  status: () => call<Status>('/api/kitapprovals/status'),
  pending: () => call<Pending>('/api/kitapprovals/pending'),
  detail: (id: string) => call<Detail>(`/api/kitapprovals/${encodeURIComponent(id)}`),
  decide: (id: string, body: { decision: 'approve' | 'reject'; reason: string; change_sha256: string; pin: string }) =>
    call<DecideOk>(`/api/kitapprovals/${encodeURIComponent(id)}/decide`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Spectrum-Client': 'kitapprovals' },
      body: JSON.stringify(body),
    }),
}

export const clock = (epoch: number | null | undefined) =>
  epoch ? new Date(epoch * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—'

export const age = (s: number | null | undefined) =>
  s == null ? '—' : s < 90 ? `${s}s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`

export const PIN_COMMAND = 'cd dashboard/backend && uv run python -m spectrum.slices.kitapprovals.pin'
