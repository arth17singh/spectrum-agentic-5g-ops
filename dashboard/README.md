# SPECTRUM console

The operator-and-judge console for SPECTRUM, our Breaking Barriers @ FYUZ 2026 entry. It shows the full NLM lifecycle in one place:

- **Learn.** Five operators fine-tune one network model together with DP-SGD, secure aggregation, an attested enclave aggregator, and an optional homomorphic-encryption path.
- **Verify.** Every model is benchmarked on price, performance, accuracy, and safety, gets a signed AI-BOM, and has to pass a deploy gate.
- **Release.** Models ship to several vendors' targets from one signed manifest: shadow first, then canary, then full rollout.
- **Operate.** A governed agent swarm runs a network twin. Agents use signed agent-to-agent messages, every action goes through a Cedar policy gateway, and changes that are too big wait for a human approval with an SLA and escalation.
- **Prove.** Everything is written to a hash-chained, signed ledger anchored to S3 Object Lock, with a provenance graph and an incident dossier on top.

Design docs:
- [DASHBOARD_SPEC.md](DASHBOARD_SPEC.md): UI/UX plan and feature spec.
- [SPECTRUM.md](SPECTRUM.md): system design and run-of-show.
- [IDEAS.md](IDEAS.md): idea book.

## Run it

Prerequisites: Node 20+, Python 3.12+, [uv](https://docs.astral.sh/uv/).

```bash
make install     # backend (uv) + frontend (npm)
make dev         # API on :8787 (mock engine, hot reload) + UI on :5173
open http://localhost:5173
```

- **Phones.** Run `make demo`. Phone approvals open `http://<laptop-LAN-IP>:8787/m/approve?t=<access token>`, and the Demo Director's Preflight tab shows a QR code for it. Every device except the laptop needs that token. It changes on every restart unless `SPECTRUM_TOKEN` pins it. `make dev` only listens on the laptop.
- **Second laptop.** A presenter laptop can drive the projector from `/presenter`. Both screens follow the server-side beat. Open the phone link on it once first, for the token.
- **Demo day.** `make demo` builds the UI and serves API + UI from one process on `:8787`, with no Vite needed.
- **Tests.** `make test` runs the backend tests, the cross-language canonical-hash check, and the TypeScript typecheck.

## Giving the demo

1. Open the Demo Director (`D`), go to the **Preflight** tab, and check that every row is green.
2. Hold **Reset**, or hold `R`. This rebuilds the deterministic pre-warmed state: round 12 done, about 3,400 signed ledger records.
3. Press `→`, `Space`, or a clicker's PageDown for each of the 14 beats. Each beat does three things:
   - navigates to the right screen;
   - fires the backend scenario (start a round, poison an operator, run the gate, inject the incident, attack, tamper, and so on);
   - spotlights the panels to talk about.
4. Approve the human decision from the console or a phone. The four-eyes item needs two different people.

| Key | Action |
|---|---|
| `→` `Space` `PageDown` | Next beat |
| `←` `PageUp` | Previous beat |
| `1`–`9` | Jump to that beat |
| `D` | Show or hide the Director |
| `N` | Presenter notes |
| `S` | Spotlight |
| `T` | Ops / Projector theme. Use Projector on washed-out projectors. |
| `P` | Pause the simulation |
| `L` | Live stream drawer |
| `⌘K` | Command palette |
| `g` + letter | Go to a screen |
| `?` | All shortcuts |

## What is real and what is simulated

The UI is always honest about this. Every panel has a source chip showing the AWS service it maps to, and the mode badge reads `MOCK` or `AWS`.

**Real, even in mock mode:**

- **Ledger.** SHA-256 hash chain over canonical JSON, Ed25519 signatures, and Merkle anchors every 32 records.
  - Tampering breaks verification at exactly the edited record, and inclusion proofs fail against the anchor.
  - The browser recomputes hashes with WebCrypto. `frontend/scripts/check-canonical.mjs` proves JS and Python hash identically.
- **Secure aggregation.** Pairwise additive masks over fixed-point integers mod 2^32. The masks cancel with zero error.
- **Agent-to-agent messages.** Signed with per-agent keys. An authority mismatch, meaning an agent claiming to speak for someone else, is detected.
- **Policy.** Enforced outside the model by a deterministic evaluator that mirrors the Cedar policies shown. This covers delegation-chain checks, quarantine, and identity revocation.

**Simulated, labelled as such:**

- Training curves and the privacy accountant. The accountant is an approximation that composes correctly when σ changes.
- The network twin's KPIs.
- Model benchmark scores. These are illustrative, not real results for the named products.
- Token prices. These are modelled list-price assumptions.

## Architecture

```
frontend/  React 19 + Vite + TypeScript + Tailwind v4, zustand store, d3 scales, hand-built SVG/canvas charts
  src/core/       store (one slice per engine), WebSocket sync, REST client, feature flags, formatting, canonical JSON
  src/ui/         design-system primitives (Panel, StatusPill, Hash, SignatureBadge, CodeBlock, …)
  src/charts/     LineChart, Scatter, Heatmap, Gantt, Ridgeline, Sankey, … (dataviz rules: 2px lines, hairline grid,
                  legends, crosshair tooltips, table views)
  src/app/        shell: top bar, Lifecycle Ribbon, rail, live stream, inspector, palette, Demo Director, hotkeys
  src/features/   one folder per screen + manifest.ts; src/features/index.ts is the registry
backend/   FastAPI + asyncio simulation loop (10 Hz), one WebSocket (batched 100 ms), REST commands
  spectrum/engines/  federation, proving, release, twin, swarm, trust, ledger, oversight, observability
  spectrum/adapters/ mock (default) and aws (boto3 seams: DynamoDB, S3 Object Lock, KMS, Verified Permissions,
                     Bedrock Guardrails, Step Functions, CloudWatch, AppConfig)
```

### Feature gating

- Every screen, panel, and demo-only control sits behind a flag (`src/core/flags.tsx`).
- Flags resolve in this order, with later sources winning:
  1. Code default
  2. Remote flags from `/api/config/flags`, served by AWS AppConfig in AWS mode
  3. This browser's overrides
  4. The URL, e.g. `?ff=fl.he:off,demo.tamper:on`
- Settings → Flags shows which source won for each flag and offers presets (Demo, Full, Minimal, Ops).
- To remove a screen, delete its folder and its one line in `src/features/index.ts`.

### AWS mapping

| Seam | Mock | AWS |
|---|---|---|
| Federated clients | simulated | ECS tasks running Flower and Opacus, one IAM role and S3 bucket per operator |
| Aggregator | simulated | Nitro Enclave, mask seeds released by KMS on PCR0 attestation |
| Verification pipeline | simulated stages | Step Functions, SageMaker Processing, Bedrock evaluation, KMS-signed AI-BOM |
| Deployment | simulated | SageMaker endpoints, Bedrock Custom Model Import, ECS edge services, CDK |
| Agents | scripted + stochastic | Strands Agents on Bedrock AgentCore (Runtime, Gateway as MCP, Identity) |
| Policy | Cedar mirror | Amazon Verified Permissions `IsAuthorized` |
| Guardrails | pattern filter | Bedrock `ApplyGuardrail` |
| Human approvals | in-memory queue | Step Functions task tokens, SNS / Incident Manager, Cognito |
| Ledger / anchors / signing | in-memory, real crypto | DynamoDB, S3 Object Lock (COMPLIANCE), KMS |
| Telemetry / metrics / traces | simulated | Kinesis, CloudWatch, X-Ray via ADOT |

**Before relying on these in an event account, check region availability for:**
- AgentCore and its agent-to-agent (A2A) support
- AgentCore Policy
- The architectures that Bedrock Custom Model Import accepts
