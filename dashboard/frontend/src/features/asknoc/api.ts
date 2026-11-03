import type { Rec } from '@/core/types'

// Ask the NOC's own fetch wrapper. core/api.ts can't add headers, and the backend guard requires them on every
// write: X-Spectrum-Client (forces a CORS preflight, which the server never answers, so another page can't post)
// and Idempotency-Key (a double click or a network retry never sends twice). The page runs on this laptop only.

export interface Detail { error: string; message: string; reasons?: string[]; [k: string]: unknown }

export class AskError extends Error {
  status: number
  detail: Detail
  constructor(status: number, detail: Detail) {
    super(detail.message)
    this.status = status
    this.detail = detail
  }
}

export function newKey(): string {
  try {
    if (typeof crypto.randomUUID === 'function') return crypto.randomUUID()
  } catch {
    // fall through
  }
  const b = crypto.getRandomValues(new Uint8Array(16))
  b[6] = (b[6] & 0x0f) | 0x40
  b[8] = (b[8] & 0x3f) | 0x80
  const h = Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('')
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`
}

async function call<T = Rec>(method: 'GET' | 'POST', path: string, body?: unknown, key?: string): Promise<T> {
  const headers: Record<string, string> = { 'X-Spectrum-Client': 'asknoc' }
  if (method === 'POST') headers['Content-Type'] = 'application/json'
  if (key) headers['Idempotency-Key'] = key
  const res = await fetch(path, { method, headers, body: method === 'POST' ? JSON.stringify(body ?? {}) : undefined, credentials: 'same-origin' })
  let j: Rec | null = null
  try {
    j = await res.json()
  } catch {
    j = null
  }
  if (!res.ok) {
    const d = (j?.detail && typeof j.detail === 'object' ? j.detail : null) as Detail | null
    throw new AskError(res.status, d ?? { error: `http_${res.status}`, message: `The panel's backend answered HTTP ${res.status}.` })
  }
  return (j ?? {}) as T
}

const enc = encodeURIComponent

export const ask = {
  health: () => call('GET', '/api/asknoc/health'),
  cases: () => call('GET', '/api/asknoc/cases'),
  caseView: (id: string) => call('GET', `/api/asknoc/cases/${enc(id)}`),
  activity: (id: string) => call('GET', `/api/asknoc/cases/${enc(id)}/activity`),
  turn: (id: string) => call('GET', `/api/asknoc/turns/${enc(id)}`),
  start: (text: string, key: string) => call('POST', '/api/asknoc/cases', { text }, key),
  followUp: (caseId: string, text: string, key: string) => call('POST', `/api/asknoc/cases/${enc(caseId)}/turns`, { text }, key),
  stop: (turnId: string) => call('POST', `/api/asknoc/turns/${enc(turnId)}/stop`, {}),
}

export const CASE_RE = /^case-[0-9a-f]{8}$/
