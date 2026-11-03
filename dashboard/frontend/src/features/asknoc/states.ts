import type { Rec } from '@/core/types'

// One place for every operator-facing sentence the panel shows about a message's state. Numbers about time are
// either MEASURED (from the real kit, with the case id) or named as a setting that was not verified.

export type Tone = 'ok' | 'info' | 'warn' | 'critical' | 'neutral'
export interface Copy { tone: Tone; title: string; body: string }

export interface Ctx { measured?: Rec; timing?: Rec; fallback?: string }

const measuredLine = (c: Ctx) => {
  const m = c.measured
  return m ? `Measured on ${String(m.started ?? '').slice(0, 10)}: a ${m.kind} took about ${Math.round(Number(m.seconds) / 60)} min (${m.case_id}). Cases that propose a change ask three agents and take longer (not measured yet).` : ''
}

const limitLine = (c: Ctx) => {
  const s = c.timing?.lead_timeout_s
  return s ? `noc_lead is set to stop at ${s} s (an Agent Builder setting, not verified).` : ''
}

const CAUSE: Record<string, string> = {
  stopped_near_limit: 'It stopped near its configured time limit. This cause is inferred from the elapsed time; the kit doesn’t say why a task failed.',
  not_reported: 'The kit doesn’t report why a task failed. Check the activity beside this.',
  kit_lost_case: 'The kit has no record of this case any more.',
}

export function copyFor(turn: Rec, c: Ctx): Copy {
  const s = String(turn.ui_state ?? '')
  const elapsed = turn.elapsed_s != null ? ` (${Math.round(turn.elapsed_s)} s)` : ''
  const fallback = c.fallback ? ` ${c.fallback}` : ''
  switch (s) {
    case 'queued':
    case 'sending':
      return { tone: 'neutral', title: 'Sending to noc_lead…', body: 'Through the checkpoint’s front door, as the team’s app identity.' }
    case 'running': {
      if (turn.status_unavailable) return { tone: 'warn', title: 'Can’t read the case status right now', body: 'The kit didn’t answer the last few status checks. The panel keeps trying, with growing gaps, and never resends your message.' }
      const waiting = (turn.waiting_approvals ?? []) as Rec[]
      const extra = waiting.length ? ` A change is already waiting for a person: ${waiting.map((w) => w.approval_id).join(', ')}.` : ''
      if (turn.overdue) return { tone: 'warn', title: `Still working${elapsed}`, body: `This is longer than noc_lead’s configured limit. ${limitLine(c)} The panel keeps checking for up to 15 minutes; you can press Stop following.${extra}` }
      return { tone: 'info', title: `noc_lead is working${elapsed}`, body: `${measuredLine(c)} You can leave this page: the case keeps running and this laptop keeps following it.${extra}` }
    }
    case 'answered':
      return turn.reply?.empty
        ? { tone: 'warn', title: 'Finished with no text', body: `noc_lead finished but the kit returned no text (state ${turn.a2a_state}).` }
        : { tone: 'ok', title: 'Answered', body: '' }
    case 'awaiting_approval':
      return { tone: 'warn', title: 'A change is waiting for a person', body: 'This panel can’t approve it. A person decides it with the kit’s approve or reject command. After the approvals table shows the decision, you may tell noc_lead to go ahead (for example “Approved.”). The checkpoint applies a change only if the table says APPROVED.' }
    case 'needs_input':
      return { tone: 'info', title: 'noc_lead needs more from you', body: 'Answer below.' }
    case 'failed':
      return { tone: 'critical', title: `The case stopped without an answer (${turn.a2a_state || 'no state'})`, body: CAUSE[turn.failed_cause] ?? `The kit returned a state the panel doesn’t know: ${String(turn.failed_cause ?? '').replace('unrecognised_state:', '')}.` }
    case 'lost':
      return { tone: 'critical', title: 'Stopped following after 15 minutes', body: 'The case may still finish on the kit. Its activity is beside this.' }
    case 'abandoned':
      return { tone: 'neutral', title: 'Stopped following', body: 'Only this laptop stopped. The kit has no cancel, so noc_lead may still be working; its activity stays visible.' }
    case 'cancelled_by_restart':
      return { tone: 'neutral', title: 'Not sent', body: 'The panel restarted before this message left the laptop. Nothing reached the kit; send it again if you still want to.' }
    case 'refused_policy':
      return { tone: 'critical', title: 'The checkpoint refused this message', body: `Cedar denied it: ${turn.deny_rule ? `rule ${turn.deny_rule}` : 'no policy allowed it (for the app this usually means the prompt-attack guardrail scored the text)'}. The panel doesn’t rephrase or retry. If it was a false alarm, reword it yourself.` }
    case 'refused_auth':
      return { tone: 'critical', title: 'The Gateway refused the app’s token', body: `${turn.category ?? ''}. The lead must re-run the kit’s deploy, which issues new tokens.` }
    case 'token_expired':
      return { tone: 'critical', title: 'The app’s token has expired', body: 'Nothing was sent. The lead must re-run the kit’s deploy to issue new tokens.' }
    case 'kit_unreachable':
      return { tone: 'critical', title: 'Can’t reach the kit', body: `The message never reached AWS (${turn.category ?? 'network'}).${fallback}` }
    case 'kit_error':
      return { tone: 'critical', title: 'The kit returned an error', body: String(turn.detail ?? '') }
    case 'outcome_unknown':
      return { tone: 'warn', title: 'Not sure the message arrived', body: 'The send timed out or failed after it left this laptop. The panel is checking the kit’s audit log. It won’t send it twice.' }
    case 'unconfirmed':
      return { tone: 'warn', title: 'Couldn’t confirm whether it arrived', body: `This laptop has no audit-log access to check (${turn.detail ?? 'no AWS reads'}). Don’t resend. The lead can search the audit log for ${turn.tag}.` }
    case 'not_received':
      return { tone: 'critical', title: 'The kit didn’t receive it', body: `The kit’s audit log has no record of this message after 2 minutes. It is safe to send again.${fallback}` }
    case 'lead_unconfirmed':
      return { tone: 'warn', title: 'The kit couldn’t confirm noc_lead received it', body: `${turn.case_id ? `The kit opened ${turn.case_id}, but` : 'The kit answered, but'} it couldn’t confirm noc_lead got the message. noc_lead may be working: watch the activity. Don’t resend for 5 minutes.` }
    default:
      return { tone: 'neutral', title: s || 'Unknown state', body: '' }
  }
}

export const TONE_CLASS: Record<Tone, string> = {
  ok: 'border-good/50 bg-good/10',
  info: 'border-line bg-s2',
  warn: 'border-warn/60 bg-warn/10',
  critical: 'border-critical/60 bg-critical/10',
  neutral: 'border-line bg-s2',
}

export const KIND_LABEL: Record<string, string> = {
  case_opened: 'case opened',
  ask_agent: 'asked an agent',
  tool_call: 'tool call',
  apply_call: 'apply_change',
  front_door: 'front door',
  approval: 'a person decided',
  tool_error: 'tool error',
  model_control: 'model control',
}
