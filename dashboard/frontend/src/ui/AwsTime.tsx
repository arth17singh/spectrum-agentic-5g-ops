// A time shown as "AWS time" only if AWS produced it, with where it came from; the clock of our own runner (or of
// a Lambda) is shown apart, as "runner clock", and never in place of it. The fields come from the backend
// (`aws_time`, `aws_time_source`, `runner_time`, see backend/spectrum/slices/_shared/awstime.py).

type AwsSource = 'cloudwatch' | 's3_stored' | 'policy_createdAt' | 'gateway_date' | 'none'

const AWS_SOURCE_LABEL: Record<AwsSource, string> = {
  cloudwatch: 'CloudWatch',
  s3_stored: 'S3 stored',
  policy_createdAt: 'policy createdAt',
  gateway_date: 'gateway Date',
  none: 'none',
}

const SOURCE_HINT: Record<AwsSource, string> = {
  cloudwatch: 'The time CloudWatch logged the event.',
  s3_stored: 'The time S3 stored this record (S3 LastModified). It is when the record was stored, not when the event happened.',
  policy_createdAt: 'The createdAt AgentCore gave the policy.',
  gateway_date: 'The Date header of the AgentCore Gateway response.',
  none: 'AWS produced no time for this record.',
}

/** UTC wall clock from an ISO string: HH:MM:SS(.mmm)Z, with the date when asked. */
const clock = (iso?: string | null, withDate = false) => {
  const m = iso?.match(/^(\d{4}-\d\d-\d\d)T(\d\d:\d\d:\d\d(?:\.\d+)?)/)
  return m ? `${withDate ? `${m[1]} ` : ''}${m[2]}Z` : '—'
}

type Props = { aws?: string | null; source?: string | null; runner?: string | null; withDate?: boolean; compact?: boolean }

export function AwsTime({ aws, source, runner, withDate = false, compact = false }: Props) {
  const s = (source && source in AWS_SOURCE_LABEL ? source : 'none') as AwsSource
  const has = !!aws && s !== 'none'
  return (
    <span className="inline-flex min-w-0 flex-wrap items-baseline gap-x-1.5" data-aws-source={s}>
      {has ? (
        <>
          <span className="tnum mono text-ink-2" title={`AWS time ${aws}. ${SOURCE_HINT[s]}`}>{clock(aws, withDate)}</span>
          <span className="text-micro text-ink-3" title={SOURCE_HINT[s]}>AWS · {AWS_SOURCE_LABEL[s]}</span>
        </>
      ) : (
        <span className="text-micro text-ink-3" title={SOURCE_HINT.none}>no AWS time</span>
      )}
      {runner && (
        <span className="text-micro text-ink-3" title={`Runner clock ${runner}: the clock of our own code, not AWS.`}>
          {compact ? '' : 'runner clock '}<span className="tnum mono">{clock(runner, withDate && !has)}</span>{compact ? ' runner' : ''}
        </span>
      )}
    </span>
  )
}

/** "AWS time on k of n (CloudWatch a, S3 stored b, ...)" from a backend `aws_time_summary`. */
export function AwsTimeSummary({ s, className }: { s?: { n: number; k: number; text: string } | null; className?: string }) {
  if (!s || !s.n) return null
  return (
    <span className={className ?? 'text-micro text-ink-3'} data-aws-summary={`${s.k}/${s.n}`}
      title="Each record shows the time AWS produced for it and the source. A runner or Lambda clock is not counted.">
      {s.text}
    </span>
  )
}
