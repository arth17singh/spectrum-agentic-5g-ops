import type { Rec } from '@/core/types'

/** What a check_unit result says in plain words. `parts` is {a: 'PASS', c: 'CANNOT CHECK', ...} or a list of {id, state}. */
export function checkWords(exitCode: number | null | undefined, parts?: Rec | Rec[] | null): string {
  const list: [string, string][] = Array.isArray(parts)
    ? parts.map((p) => [String(p.id), String(p.state)])
    : Object.entries(parts ?? {}).map(([k, v]) => [k, String(v)])
  const ids = (state: string) => list.filter(([, s]) => s === state).map(([k]) => k)
  const named = (xs: string[]) => (xs.length > 1 ? `parts ${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}` : xs.length === 1 ? `part ${xs[0]}` : 'a part')
  if (exitCode === 0) return `all ${list.length || 5} parts passed`
  if (exitCode === 1) return `${named(ids('FAIL'))} FAILED`
  if (exitCode === 2) return `nothing failed; ${named(ids('CANNOT CHECK'))} could not be checked`
  return 'the check did not complete'
}
