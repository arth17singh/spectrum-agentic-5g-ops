import clsx from 'clsx'
import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { useRef, useState } from 'react'

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger'

export function Button({ variant = 'secondary', size = 'md', icon, className, children, ...rest }:
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: 'sm' | 'md'; icon?: ReactNode }) {
  return (
    <button
      type="button"
      {...rest}
      className={clsx(
        'inline-flex shrink-0 items-center justify-center gap-1.5 rounded-md font-medium whitespace-nowrap transition-colors',
        'disabled:cursor-not-allowed disabled:opacity-40',
        size === 'md' ? 'h-8 px-3 text-body' : 'h-7 px-2.5 text-cap',
        variant === 'primary' && 'bg-ink-1 text-page hover:bg-ink-2',
        variant === 'secondary' && 'border border-line bg-s2 text-ink-1 hover:bg-s3',
        variant === 'ghost' && 'text-ink-2 hover:bg-s2 hover:text-ink-1',
        variant === 'danger' && 'border border-critical/50 text-critical-ink hover:bg-critical/15',
        className,
      )}
    >
      {icon}
      {children}
    </button>
  )
}

export function IconButton({ label, className, children, ...rest }:
  ButtonHTMLAttributes<HTMLButtonElement> & { label: string }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      {...rest}
      className={clsx('grid size-8 shrink-0 place-items-center rounded-md text-ink-2 transition-colors hover:bg-s2 hover:text-ink-1',
        'disabled:opacity-40', className)}
    >
      {children}
    </button>
  )
}

/** Press-and-hold button for irreversible or high-impact actions (kill switch, reset). */
export function HoldButton({ onConfirm, ms = 1200, label, children, className, danger = true }: {
  onConfirm: () => void; ms?: number; label: string; children: ReactNode; className?: string; danger?: boolean
}) {
  const [p, setP] = useState(0)
  const raf = useRef<number | null>(null)
  const start = useRef(0)
  const begin = () => {
    start.current = performance.now()
    const step = () => {
      const k = Math.min(1, (performance.now() - start.current) / ms)
      setP(k)
      if (k >= 1) {
        onConfirm()
        setP(0)
        return
      }
      raf.current = requestAnimationFrame(step)
    }
    raf.current = requestAnimationFrame(step)
  }
  const cancel = () => {
    if (raf.current) cancelAnimationFrame(raf.current)
    setP(0)
  }
  return (
    <button
      type="button"
      aria-label={label}
      title={`${label} (hold)`}
      onPointerDown={begin}
      onPointerUp={cancel}
      onPointerLeave={cancel}
      onKeyDown={(e) => { if (e.key === 'Enter' && !e.repeat) begin() }}
      onKeyUp={(e) => { if (e.key === 'Enter') cancel() }}
      className={clsx('relative inline-flex select-none items-center gap-1.5 overflow-hidden rounded-md border px-2.5 h-8 text-cap font-medium',
        danger ? 'border-critical/50 text-critical-ink' : 'border-line text-ink-1', className)}
    >
      <span className="absolute inset-y-0 left-0 bg-critical/25" style={{ width: `${p * 100}%` }} aria-hidden />
      <span className="relative flex items-center gap-1.5">{children}</span>
    </button>
  )
}

export function Segmented<T extends string | number>({ value, options, onChange, size = 'md', ariaLabel }: {
  value: T; options: { value: T; label: ReactNode; title?: string }[]; onChange: (v: T) => void; size?: 'sm' | 'md'; ariaLabel?: string
}) {
  return (
    <div role="radiogroup" aria-label={ariaLabel} className="inline-flex rounded-md border border-line bg-s2 p-0.5">
      {options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          title={o.title}
          onClick={() => onChange(o.value)}
          className={clsx('rounded-[5px] font-medium transition-colors', size === 'md' ? 'h-7 px-2.5 text-cap' : 'h-6 px-2 text-micro',
            o.value === value ? 'bg-s1 text-ink-1 shadow-[0_0_0_1px_var(--hairline)]' : 'text-ink-3 hover:text-ink-1')}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Toggle({ checked, onChange, label, disabled }: {
  checked: boolean; onChange: (v: boolean) => void; label: string; disabled?: boolean
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={clsx('relative h-[18px] w-8 shrink-0 rounded-full transition-colors disabled:opacity-40',
        checked ? 'bg-release' : 'bg-s3')}
    >
      <span className={clsx('absolute top-[2px] size-[14px] rounded-full bg-white transition-[left]', checked ? 'left-[16px]' : 'left-[2px]')} />
    </button>
  )
}

export function Slider({ label, value, min, max, step, onChange, format, unit, log }: {
  label: string; value: number; min: number; max: number; step?: number; onChange: (v: number) => void
  format?: (v: number) => string; unit?: string; log?: boolean
}) {
  const toPos = (v: number) => (log ? Math.log10(v) : v)
  const fromPos = (p: number) => (log ? 10 ** p : p)
  return (
    <label className="block">
      <div className="flex items-baseline justify-between">
        <span className="text-cap text-ink-2">{label}</span>
        <span className="tnum text-cap font-medium text-ink-1">{format ? format(value) : value}{unit ? ` ${unit}` : ''}</span>
      </div>
      <input
        type="range"
        min={toPos(min)}
        max={toPos(max)}
        step={log ? 0.01 : step}
        value={toPos(value)}
        onChange={(e) => onChange(fromPos(Number(e.target.value)))}
        className="mt-1.5 w-full accent-[var(--release)]"
      />
    </label>
  )
}

export function Tabs<T extends string>({ value, tabs, onChange }: {
  value: T; tabs: { value: T; label: ReactNode; count?: number }[]; onChange: (v: T) => void
}) {
  return (
    <div role="tablist" className="flex gap-4 border-b border-line">
      {tabs.map((t) => (
        <button
          key={t.value}
          role="tab"
          type="button"
          aria-selected={t.value === value}
          onClick={() => onChange(t.value)}
          className={clsx('-mb-px flex items-center gap-1.5 border-b-2 pb-2 text-body font-medium transition-colors',
            t.value === value ? 'border-ink-1 text-ink-1' : 'border-transparent text-ink-3 hover:text-ink-1')}
        >
          {t.label}
          {t.count != null && <span className="tnum rounded bg-s3 px-1 text-micro text-ink-2">{t.count}</span>}
        </button>
      ))}
    </div>
  )
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="inline-grid min-w-[18px] place-items-center rounded border border-line bg-s2 px-1 font-sans text-micro text-ink-2">
      {children}
    </kbd>
  )
}
