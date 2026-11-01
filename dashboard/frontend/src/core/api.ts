// Thin REST client. Every command returns JSON; failures surface as toasts via the caller.

export async function get<T = any>(path: string): Promise<T> {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`${path} returned ${res.status}`)
  return res.json()
}

export async function post<T = any>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`${path} returned ${res.status}`)
  return res.json()
}

export const api = {
  snapshot: () => get('/api/snapshot'),
  config: () => get('/api/config'),
  flags: () => get('/api/config/flags'),
  sim: (b: { paused?: boolean; speed?: number }) => post('/api/sim', b),
  fl: (action: 'start' | 'pause' | 'resume' | 'stop' | 'next_round' | 'checkpoint') => post(`/api/fl/${action}`),
  flConfig: (changes: Record<string, unknown>) => post('/api/fl-config', { changes }),
  projectedEps: (sigma: number, rounds?: number) =>
    get<Record<string, number>>(`/api/fl/projected_eps?sigma=${sigma}${rounds ? `&rounds=${rounds}` : ''}`),
  checkpoint: (id: string, action: 'restore' | 'promote') => post(`/api/fl/checkpoint/${id}/${action}`),
  flClient: (op: string, action: 'pause' | 'drop' | 'poison') => post(`/api/fl/client/${op}/${action}`),
  reveal: (on: boolean) => post(`/api/fl/reveal?on=${on}`),
  evalRun: (model?: string) => post('/api/eval/run', { model: model ?? null }),
  verifyAibom: (model: string) => get(`/api/eval/aibom/${model}/verify`),
  release: (action: 'advance' | 'rollback' | 'pause' | 'regression' | 'shadow') => post(`/api/release/${action}`),
  inject: (kind: string, target?: string) => post('/api/twin/inject', { kind, target: target ?? null }),
  swarm: (action: 'freeze' | 'resume' | 'pause') => post(`/api/swarm/${action}`),
  autonomy: (cls: string, level: number, reason: string) => post('/api/autonomy', { cls, level, reason }),
  agent: (id: string, action: 'quarantine' | 'release') => post(`/api/agents/${id}/${action}`),
  decide: (id: string, decision: 'approve' | 'approve_with_changes' | 'reject' | 'escalate',
    body: { by?: string; device?: string; reason?: string; changes?: Record<string, unknown> } = {}) =>
    post(`/api/approvals/${id}/${decision}`, body),
  verifyLedger: () => post('/api/ledger/verify'),
  tamper: (seq?: number) => post(`/api/ledger/tamper${seq ? `?seq=${seq}` : ''}`),
  restoreLedger: () => post('/api/ledger/restore'),
  record: (seq: number) => get(`/api/ledger/${seq}`),
  proof: (seq: number) => get(`/api/ledger/proof/${seq}`),
  key: (keyId: string) => get(`/api/keys/${encodeURIComponent(keyId)}`),
  provenance: () => get('/api/provenance/latest'),
  dossier: (incident: string) => get(`/api/dossier/${incident}`),
  beat: (n: number) => post(`/api/director/beat/${n}`),
  reset: () => post('/api/director/reset'),
  preflight: () => get('/api/director/preflight'),
}
