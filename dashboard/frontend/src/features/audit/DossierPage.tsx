import QRCode from 'qrcode'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { useParams } from 'react-router-dom'
import { api } from '@/core/api'
import { duration, shortHash, usd } from '@/core/format'
import type { Rec } from '@/core/types'

// Printable incident dossier. Always rendered light (it is a document), with print CSS; Cmd+P
// saves a PDF. Generating it writes the dossier digest to the ledger.

export default function DossierPage() {
  const { incident = 'INC-2040' } = useParams()
  const [d, setD] = useState<Rec | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [qr, setQr] = useState('')
  const once = useRef(false)
  useEffect(() => {
    if (once.current) return
    once.current = true
    api.dossier(incident).then((x) => (x.error ? setErr(x.error) : setD(x))).catch((e) => setErr(String(e)))
  }, [incident])
  useEffect(() => {
    if (d?.digest) QRCode.toDataURL(`spectrum:dossier:${d.digest}`, { margin: 0, width: 132, color: { dark: '#0e1319', light: '#ffffff' } }).then(setQr)
  }, [d?.digest])
  useEffect(() => { document.title = `Dossier ${incident}` }, [incident])

  return (
    <div className="dossier h-full overflow-auto bg-[#e9ecef] text-[#0e1319]">
      <style>{`
        .dossier { color-scheme: light; }
        .dossier .sheet { font-size: 12px; line-height: 18px; }
        .dossier h2 { font-size: 14px; font-weight: 600; margin: 22px 0 8px; padding-bottom: 4px; border-bottom: 1px solid #c3cad3; }
        .dossier table { width: 100%; border-collapse: collapse; }
        .dossier th { text-align: left; font-weight: 500; color: #46505c; border-bottom: 1px solid #c3cad3; padding: 4px 8px 4px 0; }
        .dossier td { border-bottom: 1px solid #e5e9ee; padding: 3px 8px 3px 0; vertical-align: top; }
        .dossier .m { font-family: var(--font-mono); font-size: 11px; }
        @media print {
          @page { size: A4; margin: 14mm; }
          html, body, #root { height: auto !important; overflow: visible !important; background: white !important; }
          .dossier { height: auto !important; overflow: visible !important; background: white !important; }
          .dossier .sheet { box-shadow: none !important; margin: 0 !important; padding: 0 !important; width: auto !important; }
          .noprint { display: none !important; }
          .dossier tr { break-inside: avoid; }
        }
      `}</style>
      <div className="noprint sticky top-0 z-10 flex items-center gap-3 border-b border-[#c3cad3] bg-white px-6 py-2 text-[13px]">
        <span className="font-semibold">SPECTRUM incident dossier</span>
        <span className="text-[#76808c]">{d ? `ledger record #${d.ledger_seq?.toLocaleString()}` : 'generating…'}</span>
        <button type="button" onClick={() => window.print()} className="ml-auto rounded-md bg-[#0e1319] px-3 py-1.5 text-[13px] font-medium text-white">Print or save as PDF</button>
      </div>
      {err && <div className="p-8">Could not generate the dossier: {err}</div>}
      {d && (
        <article className="sheet mx-auto my-8 w-[820px] bg-white px-14 py-12 shadow-[0_4px_24px_rgba(0,0,0,0.12)]">
          <header className="flex items-start gap-6 border-b-2 border-[#0e1319] pb-5">
            <div className="flex-1">
              <div className="text-[11px] text-[#76808c]">Incident dossier, generated {d.generated_at} by {d.generated_by}</div>
              <h1 className="mt-1 text-[24px] leading-8 font-semibold">{d.incident.id}: {d.incident.title}</h1>
              <div className="mt-2 flex flex-wrap gap-x-6 gap-y-1">
                <Meta k="Severity" v={d.incident.severity} />
                <Meta k="Status" v={d.incident.status} />
                <Meta k="Started" v={d.incident.started_at} />
                <Meta k="Time to restore" v={d.incident.mttr_s != null ? duration(d.incident.mttr_s) : 'open'} />
                <Meta k="Agent compute cost" v={d.incident.cost_usd != null ? usd(d.incident.cost_usd) : '—'} />
              </div>
            </div>
            {qr && (
              <div className="text-center">
                <img src={qr} width={96} height={96} alt="Dossier digest QR code" />
                <div className="m mt-1 text-[#76808c]">{shortHash(d.digest, 6, 4)}</div>
              </div>
            )}
          </header>

          <h2>Verification</h2>
          <table>
            <tbody>
              <Row k="Ledger range" v={<span className="m">#{d.ledger_range[0].toLocaleString()} to #{d.ledger_range[1].toLocaleString()} ({d.timeline.length} records listed)</span>} />
              <Row k="Chain verification at generation" v={d.verification.ok ? `Intact: ${d.verification.total.toLocaleString()} records, hash links and Ed25519 signatures valid` : `BROKEN at #${d.verification.checked.toLocaleString()} of ${d.verification.total.toLocaleString()}`} />
              <Row k="Dossier digest (SHA-256)" v={<span className="m break-all">{d.digest}</span>} />
              <Row k="Dossier signature" v={<span className="m break-all">{d.signature}</span>} />
              <Row k="Recorded in ledger" v={<span className="m">#{d.ledger_seq?.toLocaleString()} (kms:spectrum/auditor)</span>} />
            </tbody>
          </table>

          {d.anchors?.length > 0 && (
            <>
              <h2>Merkle anchors covering this incident</h2>
              <table>
                <thead><tr><th>Anchor</th><th>Records</th><th>Merkle root</th><th>Object (S3 Object Lock)</th><th>Retain until</th></tr></thead>
                <tbody>
                  {d.anchors.map((a: Rec) => (
                    <tr key={a.n}><td className="m">#{a.n}</td><td className="m">{a.range[0]}–{a.range[1]}</td><td className="m">{shortHash(a.root, 10, 6)}</td><td className="m">{a.s3_uri}</td><td>{a.retain_until} ({a.object_lock})</td></tr>
                  ))}
                </tbody>
              </table>
            </>
          )}

          {d.plan && (
            <>
              <h2>Plan executed</h2>
              <table>
                <thead><tr><th>Step</th><th>Tool</th><th>Target</th><th>Agent</th><th>Policy</th><th>Outcome</th></tr></thead>
                <tbody>
                  {d.plan.steps.map((s: Rec) => (
                    <tr key={s.n}><td>{s.n}</td><td className="m">{s.tool}</td><td>{s.target}</td><td className="m">{s.agent}</td><td className="m">{s.policy}</td><td>{s.verdict} ({s.state})</td></tr>
                  ))}
                </tbody>
              </table>
            </>
          )}

          {d.preflight?.before && (
            <>
              <h2>Pre-flight prediction (network twin)</h2>
              <table>
                <thead><tr><th>KPI</th><th>Before</th><th>Predicted after</th></tr></thead>
                <tbody>
                  {['congestion_pct', 'dropped_pct', 'throughput_gbps', 'energy_kw', 'health_pct'].map((k) => (
                    <tr key={k}><td>{k.replace(/_/g, ' ')}</td><td className="m">{d.preflight.before[k]}</td><td className="m">{d.preflight.after[k]}</td></tr>
                  ))}
                  <tr><td>blast radius</td><td colSpan={2} className="m">{d.preflight.blast_radius_pct}% of cells ({d.preflight.cells_touched} cells)</td></tr>
                </tbody>
              </table>
            </>
          )}

          {d.negotiation && (
            <>
              <h2>Agent negotiation and dissent</h2>
              <table>
                <tbody>
                  {d.negotiation.rounds.map((r: Rec, i: number) => (
                    <tr key={i}><td className="m w-[90px]">{r.ts.slice(11, 19)}</td><td className="m w-[100px]">{r.agent}</td><td>{r.kind}: {r.text}</td></tr>
                  ))}
                </tbody>
              </table>
              {d.negotiation.dissent?.map((x: Rec, i: number) => <p key={i} className="mt-2"><b>Recorded dissent ({x.agent}):</b> {x.text}</p>)}
            </>
          )}

          <h2>Human decisions</h2>
          {d.approvals?.length ? (
            <table>
              <thead><tr><th>Approval</th><th>Decision</th><th>Signed by</th><th>Device</th><th>Signature</th><th>Ledger</th></tr></thead>
              <tbody>
                {d.approvals.flatMap((a: Rec) => (a.approvals.length ? a.approvals : [{ name: '—', device: '—', sig: '', ledger_seq: null }]).map((x: Rec, i: number) => (
                  <tr key={`${a.id}-${i}`}><td>{a.id}: {a.title}</td><td>{a.status}</td><td>{x.name}</td><td>{x.device}</td><td className="m">{shortHash(x.sig, 8, 6)}</td><td className="m">{x.ledger_seq ? `#${x.ledger_seq}` : ''}</td></tr>
                )))}
              </tbody>
            </table>
          ) : <p>No human approvals were required for this incident.</p>}

          <h2>Policies applied</h2>
          <table>
            <thead><tr><th>Policy</th><th>Name</th><th>Version</th><th>Hash</th></tr></thead>
            <tbody>{d.policies.map((p: Rec) => <tr key={p.id}><td className="m">{p.id}</td><td>{p.name}</td><td>{p.version}</td><td className="m">{shortHash(p.hash, 10, 6)}</td></tr>)}</tbody>
          </table>

          <h2>Models</h2>
          <p>Site agents: <span className="m">{d.models.site_agents.join(', ')}</span>. Coordinator: <span className="m">{d.models.coordinator}</span>.</p>

          <h2>Timeline</h2>
          <table>
            <thead><tr><th>#</th><th>Time</th><th>Event</th><th>Actor</th><th>Record hash</th><th>Signature</th></tr></thead>
            <tbody>
              {d.timeline.map((t: Rec) => (
                <tr key={t.seq}><td className="m">{t.seq}</td><td className="m">{t.ts.slice(11, 23)}</td><td>{t.summary}</td><td className="m">{t.actor}</td><td className="m">{shortHash(t.hash, 8, 4)}</td><td className="m">{shortHash(t.sig, 6, 4)}</td></tr>
              ))}
            </tbody>
          </table>

          <footer className="mt-8 border-t border-[#c3cad3] pt-3 text-[11px] text-[#76808c]">
            To verify independently: recompute SHA-256(canonical(record) ‖ prev_hash) for each listed record, check each Ed25519 signature against the published key,
            and rebuild the Merkle roots against the S3 Object Lock anchors above. The QR code encodes this dossier’s digest.
          </footer>
        </article>
      )}
    </div>
  )
}

const Meta = ({ k, v }: { k: string; v: ReactNode }) => (
  <span className="text-[12px]"><span className="text-[#76808c]">{k}</span> <span className="font-medium">{v}</span></span>
)
const Row = ({ k, v }: { k: string; v: ReactNode }) => (
  <tr><td className="w-[220px] text-[#46505c]">{k}</td><td>{v}</td></tr>
)
