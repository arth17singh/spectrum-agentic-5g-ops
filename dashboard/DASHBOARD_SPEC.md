# SPECTRUM Console: UI/UX plan & feature spec

> Status: v0.1 (prototype). Owner: the SPECTRUM team. Companion docs: [SPECTRUM.md](SPECTRUM.md) (system design and run-of-show) and [IDEAS.md](IDEAS.md).
>
> **What this is.** The single operator-and-judge-facing console for SPECTRUM. It covers the full NLM lifecycle: private federated fine-tuning (DP-SGD, secure aggregation, HE), verification (price, performance, accuracy, safety), multi-vendor release, the governed agent swarm on the network twin, human oversight, and cryptographic proof of everything that happened.
>
> **Who it is for.**
> 1. **The presenter.** They must deliver a 7-minute demo with zero friction.
> 2. **Senior technical judges** (AT&T, TELUS, AWS, Meta, TIP). They will squint at axes, units, hashes, and ε values, and ask "is this real?"
> 3. **Later, real NOC operators.**

---

## 0. Principles

1. **Demo-first.** Every screen has a "hero moment" that can be triggered by one keypress from the Demo Director. Nothing in the demo path requires typing, scrolling to find something, or waiting more than ~1.5 s without visible progress.
2. **Proof over claims.** Never show "secure ✓" without the evidence one click away: hash, signature, key id, attestation measurement, policy id, trace id. Every badge opens the inspector with the raw artefact.
3. **Alive, not busy.** Data moves continuously (streams, counters, particles), but motion is reserved for *meaning*: change, flow, attention. Idle chrome never animates. `prefers-reduced-motion` freezes decorative motion and keeps value updates.
4. **One trust vocabulary.** The same seven status words and icons are used everywhere: `verified`, `signed`, `pending`, `denied`, `quarantined`, `tampered`, `simulated`.
5. **Honest about the mock.** Every panel header carries a *source chip* that says where the data comes from now and which AWS service it maps to (e.g. `Simulated · CloudWatch`). Judges respect this, and it makes the AWS swap obvious.
6. **Modular by construction.** Every screen, panel, and control is a *feature module* behind a flag. Removing or redesigning one never touches another (see §7).

---

## 1. Design system

### 1.1 Visual concept: "instrument, not website"

SPECTRUM borrows from RF test equipment and NOC wallboards: calibrated axes, hairline graticules, phosphor-like data on a matte graphite surface, and dense but legible numerics. The **one bold element** is the **Lifecycle Ribbon**: a persistent band under the top bar where the five planes (Learn, Verify, Release, Operate, Prove) sit side by side in their five hues. Together they form a literal spectrum, and live pulses travel through it as artefacts move through the lifecycle. Everything else stays quiet.

What we deliberately avoid:
- Gradient washes and glassmorphism blur.
- Neon glow on everything.
- Identical rounded cards with soft shadows.
- All-caps tracked eyebrow labels.
- Emoji.
- Middot-joined meta strings.
- Purple-to-blue "AI" gradients.

### 1.2 Color tokens

**Themes.** Two are selected (not auto-flipped):
- **Ops** (dark, default): for laptops and monitors.
- **Projector** (light, high contrast): hackathon projectors wash out dark UIs. Toggle with `T`.

| Token | Ops (dark) | Projector (light) | Use |
|---|---|---|---|
| `--page` | `#0F1216` | `#F4F6F8` | App background |
| `--surface-1` | `#141920` | `#FFFFFF` | Panels, chart surface |
| `--surface-2` | `#1A2029` | `#F0F2F5` | Raised rows, inputs, hovered rows |
| `--surface-3` | `#222A35` | `#E5E9EE` | Pressed, selected |
| `--hairline` | `#252D38` | `#DCE1E7` | Borders, gridlines |
| `--axis` | `#344050` | `#C3CAD3` | Axis baselines |
| `--ink-1` | `#E9EDF1` | `#0E1319` | Primary text, values |
| `--ink-2` | `#A6B0BC` | `#46505C` | Secondary text |
| `--ink-3` | `#6F7A87` | `#76808C` | Muted, axis labels, timestamps |
| `--focus` | `#7FB4FF` | `#1F5FD1` | Keyboard focus ring (2px, offset 2px) |

**Plane hues.** These are categorical identity, validated with `validate_palette.js`: ALL CHECKS PASS in dark on `#141920`; worst adjacent CVD ΔE 9.4. A plane hue is **never** used for status.

| Plane | Meaning | Ops | Projector |
|---|---|---|---|
| Learn | Federated fine-tuning | `#9085E9` violet | `#6250D6` |
| Verify | Proving Ground | `#C98500` amber | `#B87A00` |
| Release | Multi-vendor deploy | `#3987E5` blue | `#2A78D6` |
| Operate | Swarm and twin | `#199E70` aqua | `#14895F` |
| Prove | Trust, audit, provenance | `#D95926` orange | `#C94F1E` |

**Status.** Reserved. Always paired with an icon and a word, never color alone.

| Status | Hex | Icon (lucide) |
|---|---|---|
| good / verified | `#0CA30C` (text `#3CC93C` on dark) | `ShieldCheck` / `CircleCheck` |
| warning / pending | `#FAB219` | `Clock3` / `TriangleAlert` |
| serious / degraded | `#EC835A` | `CircleAlert` |
| critical / denied / tampered | `#D03B3B` (text `#F07070` on dark) | `OctagonX` / `ShieldX` |
| neutral / simulated | `--ink-3` | `FlaskConical` |

**Sequential ramp.** Heatmaps (latency, attention, update matrices) use a single-hue blue ramp `#0D366B → #CDE2FB` on dark (dark = low, light = high). **Diverging** ramp (signed weights, KPI deltas): blue `#3987E5` ↔ gray `#383835` ↔ red `#E66767`.

**Data series inside a chart** use the dataviz reference categorical order (blue, orange, aqua, yellow, magenta, green, violet, red), assigned in fixed order to *entities*, never by rank. Operators keep the same color everywhere:

| Operator | Color |
|---|---|
| Op-A (Cascadia) | blue |
| Op-B (Northwind) | orange |
| Op-C (Meridian) | aqua |
| Op-D (Halcyon) | yellow |
| Op-E (Tessellate) | magenta |

### 1.3 Typography

- **IBM Plex Sans** for all UI text, values, and headings. Weights 400/500/600 only. It's engineered and technical, with a telecom/infra lineage and excellent figures.
- **IBM Plex Mono** *only* where mono is semantic: hashes, signatures, key ids, ARNs, trace ids, Cedar/JSON code, raw log lines. Never decorative labels.
- Fonts are **bundled locally** (`@fontsource`), so there are zero network dependencies on demo day.
- Scale (px / line-height): 11/16 micro (axis ticks), 12/16 caption, 13/20 body, 15/22 panel title, 18/24 section, 24/30 page title, 40/44 hero figure (one per screen).
- Labels use sentence case. No all-caps except unit acronyms (ms, GPU, KPI).
- `tabular-nums` in tables, tickers, axis ticks, and live counters (stops jitter). Proportional figures for hero values.

### 1.4 Shape, space, elevation

- 4px base grid. Panel padding 16px. Dense tables use 32px rows; comfortable rows 40px.
- **Radii encode hierarchy:** panels 10px, inner tiles 6px, chips 4px, status dots full. Buttons 6px.
- **No drop shadows.** Elevation comes from surface step plus hairline. Overlays (inspector, palette, toasts) get `--surface-2` and a 1px hairline; the backdrop dims to 50%.
- Panel header: 40px, title (15/500) left, source chip + panel actions right, 1px hairline bottom.

### 1.5 Motion

| Purpose | Spec |
|---|---|
| Value change | Number tween 400 ms `cubic-bezier(.2,.8,.2,1)`; a changed cell flashes `--surface-3` for 600 ms |
| New row in a stream | Slides in from top, 180 ms; rows beyond the buffer fade out |
| Message particle on an edge | 700–1200 ms along the path, colored by policy verdict |
| Spotlight (Director) | 2px plane-hue ring + 0.92 dim on everything else, 300 ms in, held until the next beat |
| Drawer / inspector | 220 ms slide, spring-free |
| Critical escalation | One 2-cycle pulse of the banner, then static. Never infinite blinking |
| Reduced motion | Particles become static highlighted edges, tweens snap, ribbon pulses become step changes |

### 1.6 Formatting rules

- **Hashes:** `9f3a1c…b20e` (first 6, last 4) in mono. Click copies the full value; hover shows the full value; ⌥-click opens the inspector.
- **Signatures:** a `SignatureBadge` showing state (signed / verified / failed), algorithm (`Ed25519`), and key alias (`kms:spectrum-ledger-01`).
- **Money:** `$0.42` under $10, `$12.9` under $100, `$1.2K` above. Always state the unit base: "per incident", "per 1K tasks".
- **Time:** relative within 1 h ("12 s ago", updated live), absolute `14:03:22.418` in logs (ms precision), ISO-8601 in the inspector. All times show simulation time with a small real-time offset indicator.
- **ε:** always `ε = 2.31 (δ = 1e-5)`. Never ε without δ.
- **Percent:** one decimal under 10%, none above.
- **Bytes:** IEC units (`18.4 MiB`).

### 1.7 Core components (in `src/ui`)

`Panel`, `PanelHeader`, `SourceChip`, `Stat`, `HeroFigure`, `Badge`, `StatusPill`, `LiveDot`, `Button` (primary / secondary / ghost / danger), `IconButton`, `Toggle`, `Segmented`, `Slider` (with value and unit), `Tabs`, `Table` (virtualized for 1K+ rows), `Kbd`, `Tooltip`, `Hash`, `SignatureBadge`, `CodeBlock` (Cedar/JSON highlighting), `Drawer`, `Modal`, `Toast`, `EmptyState`, `Sparkline`, `Meter`, `Countdown`, `ProgressSteps`.

---

## 2. App shell & information architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│ ◧ SPECTRUM  Operator A ▾  ● Live  sim 21:42:07  +0.0s   [⌘K Search]   ⚑ 2  Autonomy L3  ◐ ▣  │ top bar 48
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ Learn ▸ round 12/20 ε 1.84 │ Verify ▸ v7 gate PASS │ Release ▸ v7 canary 10% │ Operate ▸ 1 inc │ Prove ▸ #4,412 ✓ │ ribbon 36
├────┬─────────────────────────────────────────────────────────────────────────────┬───────────┤
│ ⌂  │                                                                             │ Inspector │
│ ⟁  │                       ROUTED PAGE (12-col grid, 16px gutters)               │ (420px,   │
│ ⚖  │                                                                             │  slides   │
│ ⇪  │                                                                             │  over on  │
│ ⬡  │                                                                             │  demand)  │
│ ☝  │                                                                             │           │
│ ⛨  │                                                                             │           │
│ ⛓  │                                                                             │           │
│ ∿  │                                                                             │           │
│ ⚙  │                                                                             │           │
├────┴─────────────────────────────────────────────────────────────────────────────┴───────────┤
│ ▲ Live stream  [all ▾] [sev ≥ info ▾]  14:03:22.418  SWARM  site-17 → coordinator  DENY  …   │ 32 collapsed / 280 open
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Top bar (48px)

| Element | Behaviour |
|---|---|
| Wordmark | "SPECTRUM" in Plex Sans 600, with a 5-bar glyph in the five plane hues. Click goes to Mission Control |
| Operator context | Dropdown: Operator A (home NOC) or Consortium view. Changes the scope of Swarm/Release |
| Connection | `LiveDot`: green "Live" (WebSocket open, last message < 2 s); amber "Lagging" (2–5 s); red "Offline" (reconnecting, with retry countdown). Hover shows messages/s and round-trip latency |
| Clocks | Simulation clock (ticks every second, mono, tabular). When paused, shows "Paused" in amber. Speed chip `1×` / `2×` / `5×` when not 1× |
| Search / command palette | `⌘K`: jump to any screen, agent, model version, ledger record (`#4412`), trace id, or policy; run actions ("Inject fiber cut", "Save checkpoint") |
| Escalations | Flag icon with a count of pending human decisions. Turns amber when any SLA is under 60 s. Click opens the Oversight Desk |
| Global autonomy | Chip `Autonomy L3`. Click opens a popover with a per-class dial (see §5.5). Changing it writes a ledger record |
| Mode chip | `MOCK` (neutral) or `AWS` (blue). Comes from backend `/api/config` |
| Theme | `◐` toggles Ops / Projector |
| Kill switch | `▣` red-outline icon "Freeze swarm". Hold for 1.2 s to arm (progress ring), then it freezes all agent actions. Ledgered. The inverse is "Resume", which needs a reason |

### 2.2 Lifecycle Ribbon (36px): the signature element

- Five equal segments, each a plane hue at 14% fill with a 2px top edge in the full hue, label and one live metric:
  - Learn: `round 12/20`, `ε 1.84`
  - Verify: `v7`, `gate PASS`
  - Release: `v7 canary 10%`
  - Operate: `1 incident`, `MTTR 2m14s`
  - Prove: `ledger #4,412 ✓`, `anchored 38 s ago`
- **Flow pulses.** When an artefact moves stage (checkpoint → candidate → release → serving → ledger), a 24px light pulse travels from the source segment into the next (900 ms). The loop is closed: Prove wraps back to Learn when an incident becomes training data.
- Hover shows a mini-popover with that plane's 3 key numbers and a sparkline. Click navigates.
- The active route's segment gets a 2px bottom indicator.

### 2.3 Left rail

64px icons, expanding to 220px with labels (`[` toggles; auto-expands on hover after 300 ms). Order follows the lifecycle, and each item has a plane-hue tick:

| # | Screen | Plane | Hotkey |
|---|---|---|---|
| 1 | Mission Control | all | `g m` |
| 2 | Federation Center | Learn | `g f` |
| 3 | Proving Ground | Verify | `g p` |
| 4 | Release Center | Release | `g r` |
| 5 | Swarm Tower | Operate | `g s` |
| 6 | Oversight Desk | Operate / Prove | `g o` |
| 7 | Trust Center | Prove | `g t` |
| 8 | Audit & Proof | Prove | `g a` |
| 9 | Observability | all | `g b` |
| 10 | Settings & Flags | n/a | `g ,` |

### 2.4 Live Stream drawer (bottom)

- **Collapsed (32px):** the newest event scrolls as a single-line ticker. Severity icon, ms timestamp, plane chip, actor, message.
- **Expanded (280px, `L` toggles):** a virtualized log table.
  - Columns: time, plane, severity, actor, event, detail, and a link icon to the inspector.
  - Ring buffer of 5,000 events; auto-scroll pauses when the user scrolls up, with a "Jump to live (N new)" pill.
- **Filters:** plane multi-select, minimum severity, free-text (matches actor/event/ids), and "only ledgered".
- Critical events also raise a toast (top right, 6 s, max 3 stacked) with a single action ("Review").

### 2.5 Inspector (right drawer, 420px)

Opens for *any* entity: agent, message, policy decision, ledger record, model version, checkpoint, trace, approval, operator, cell.

- **Header:** entity type, id, status pill.
- **Tabs:** `Overview`, `Proof` (hashes, signature, key, verification result, re-verify button), `Raw` (canonical JSON, copy), `Lineage` (mini provenance graph), `Trace` (OTel spans if any).
- `Esc` closes. `[` / `]` navigate inspector history.

### 2.6 Keyboard map (shown with `?`)

| Key | Action |
|---|---|
| `→` / `PageDown` / `Space` | Next demo beat (clickers send PageDown) |
| `←` / `PageUp` | Previous beat |
| `D` | Toggle Demo Director |
| `N` | Toggle presenter notes |
| `1`–`9` | Jump to beat N |
| `R` (hold) | Reset scenario |
| `P` | Pause / resume simulation |
| `T` | Theme |
| `L` | Live stream drawer |
| `⌘K` | Palette |
| `g` + letter | Navigate |
| `Esc` | Close overlay |
| `?` | Help |

---

## 3. Demo Director (demo-first core)

A floating, draggable 320px panel, bottom-left above the stream. Flag `demo.director`. Hidden with `D`, so judges never have to see it on the projector if the presenter drives from a second screen (`/presenter` route, see below).

**Contents:**
- Scenario selector (default: "Stadium let-out + fiber cut").
- **Beat list:** numbered (it is a true sequence), each beat with title, target screen, duration hint, and state (done / current / next).
- Primary controls: `◀ Prev`, `Next ▶` (big), `Pause sim`, speed `1× 2× 5×`, `Reset` (hold to confirm).
- Toggles: `Auto-advance` (follows the duration hints), `Spotlight`, `Presenter notes`.
- **Preflight checklist** (before going on stage): backend reachable, WebSocket live, ledger verified clean, fonts loaded, scenario reset, theme chosen, phone approval page connected (shows the device count). All rows green, or the item shows the fix.

**What "Next" does per beat:**
1. Navigates to the beat's route.
2. Calls `POST /api/director/beat/{n}`, which makes the backend fire the scenario events (inject fault, start round, poison, and so on).
3. Applies the spotlight to the beat's panel ids.
4. Shows the presenter note (one or two sentences to say).

Beats are idempotent: re-running a beat resets its local state first.

**Default beats** (mirrors SPECTRUM.md):

| # | Beat | Route | Backend action | Spotlight |
|---|---|---|---|---|
| 1 | Cold open | `/` | `scenario.reset` (state is pre-warmed) | Lifecycle Ribbon + consortium map |
| 2 | Federated round live | `/federation` | `fl.start_round` (fast mode: about 25 s per round) | Round timeline, loss, ε |
| 3 | What the aggregator sees | `/federation` | `fl.reveal_masking` | SecAgg visualizer |
| 4 | Poisoned update | `/federation` | `fl.inject_poison(op=E)` | Update-norm chart, client table |
| 5 | Proving Ground gate | `/proving` | `eval.run(candidate=v7)` | Pipeline, scorecard, AI-BOM |
| 6 | Shadow, then canary | `/release` | `release.canary(v7, 10%)` | Rollout, shadow agreement |
| 7 | Incident hits | `/swarm` | `twin.inject(fiber_cut T3)` + `stadium_surge` | Twin map + swarm graph |
| 8 | Agents negotiate | `/swarm` | (automatic) | Negotiation panel + pre-flight |
| 9 | Human approval | `/oversight` | approval created (the phone buzzes) | Approval card |
| 10 | Prompt-injection attack | `/swarm` | `attack.telemetry_injection(site-17)` | Denied edge + quarantine |
| 11 | Provenance | `/audit` | (UI) opens provenance for the reroute action | Provenance graph |
| 12 | Tamper | `/audit` | `ledger.tamper(record)` then the UI auto-runs verify | Chain strip |
| 13 | Dossier and observability | `/observability` | `dossier.generate` | Cost per incident + trace |
| 14 | Close the loop | `/` | `loop.enqueue_incident` | Ribbon Prove → Learn pulse |

**Presenter route `/presenter`.** Shows the beat list, notes, a timer (elapsed vs. 7:00 target, amber at 6:30), and the next beat preview. It controls the main screen over the same backend (the Director state is server-side and broadcast). This is the "second laptop" mode.

**Mobile approval `/m/approve`.** A phone-optimized page that lists pending approvals with a big Approve / Reject. The QR code for it is shown in the Director preflight.

**Fallback.** `Replay mode` replays a recorded event log (`/api/replay/{scenario}`) through the same UI if anything goes wrong live. The ribbon shows a small `Replay` chip (honesty).

---

## 4. What we monitor and demo: the high-impact list

These are the things that score with senior technical judges, mapped to where they live in the UI.

| # | Item | Why judges care | Track | Screen / widget |
|---|---|---|---|---|
| 1 | Live federated rounds with per-operator progress | Shows FL is real, not a slide | T2 | Federation: Round timeline (Gantt) |
| 2 | DP-SGD internals: clip norm C, noise multiplier σ, sampling rate q, clip fraction, per-sample grad norm distribution | Proves we understand DP-SGD, not just "added noise" | T2 | Federation: Grad-norm ridgeline + DP config |
| 3 | Privacy budget ε(δ) burn-down per operator against the target | The number every privacy reviewer asks for | T2 | Federation: ε chart |
| 4 | Secure aggregation: masked updates look like noise, the sum is exact (max error 0) | Visual proof the server is blind | T2 | Federation: SecAgg visualizer |
| 5 | HE: CKKS parameters, ciphertext sizes, encrypted-sum error, noise budget | "Homomorphic encryption" made concrete | T2 | Federation: HE panel |
| 6 | Nitro Enclave attestation (PCR0/1/2, KMS policy match) | Hard-rooted trust | T2 / Trust | Federation + Trust Center |
| 7 | Data sovereignty: bytes moved per boundary, 0 B raw data egress | "Data never left" as evidence | T2 | Federation: Sovereignty Sankey |
| 8 | Poisoning detection (update-norm outlier, clipped or rejected) | Adversarial robustness | T2 | Federation: Update norms |
| 9 | Checkpoints: signed, with metrics, restorable, promotable | Production SDLC | T1 | Federation: Checkpoint rail |
| 10 | Loss/accuracy curves (global vs. per-operator local-only) | Federation beats training alone | T1 / T2 | Federation: Convergence |
| 11 | Price / Performance / Accuracy / Safety scorecard per model | The exact Track 1 language | T1 | Proving Ground |
| 12 | Pareto frontier (cost vs. accuracy) and parallel coordinates | How senior engineers think about trade-offs | T1 | Proving Ground |
| 13 | Safety red-team matrix (attack class × model) | Safety is measured, not claimed | T1 | Proving Ground |
| 14 | Federated evaluation (private per-operator scores, signed) | A novel verification idea | T1 / T2 | Proving Ground |
| 15 | Scaling curve (params vs. accuracy) | The "NLM scaling" theme | T1 | Proving Ground |
| 16 | Signed AI-BOM / model card | Supply-chain trust | T1 / Trust | Proving Ground + inspector |
| 17 | Multi-target deployment matrix (SageMaker, Bedrock CMI, vendor A/B edge) | "Repeatable multi-vendor deployment" | T1 | Release Center |
| 18 | Shadow agreement rate and canary guard metrics with auto-rollback | Production readiness | T1 | Release Center |
| 19 | Live network twin with KPIs and incident overlay | Grounds everything in telecom reality | T3 | Swarm Tower: Twin map |
| 20 | Swarm graph with live A2A messages colored by policy verdict | Collective intelligence, visible | T3 | Swarm Tower |
| 21 | Negotiation between conflicting agents, with the utility trade-off | Real multi-agent collaboration | T3 | Swarm Tower |
| 22 | Twin pre-flight: predicted impact and blast radius | Safe autonomy | T3 | Swarm Tower + Oversight |
| 23 | Policy decisions (permit/deny) with Cedar policy id and reason | Policy enforcement | T3 | Trust Center + stream |
| 24 | Delegation-chain authorization (confused deputy blocked) | Agent-to-agent security | T3 | Trust Center + inspector |
| 25 | Agent-to-agent prompt infection blocked, plus quarantine | Wow moment | T3 | Swarm Tower |
| 26 | Earned autonomy (TM Forum L0–L5) with a trust-score history | Human oversight that scales | T3 | Swarm Tower + top bar |
| 27 | Human approvals with SLA timers, dissent, escalation ladder, 4-eyes | Human oversight | T3 | Oversight Desk + phone |
| 28 | Hash-chained, Ed25519-signed ledger with live verification | Auditability you can test | Trust | Audit & Proof |
| 29 | Merkle anchors to S3 Object Lock and inclusion proofs | Tamper-evidence beyond our own DB | Trust | Audit & Proof |
| 30 | Tamper demo (edit a record, verification breaks at that record) | Unforgettable | Trust | Audit & Proof |
| 31 | End-to-end provenance graph (action back to training round) | Traceability | Trust | Audit & Proof |
| 32 | Regulator-ready incident dossier | Operator reality | Trust | Audit & Proof |
| 33 | OTel trace waterfall across agents, gateway, policy, and tools (gen_ai semantic conventions) | Observability done properly | Obs | Observability |
| 34 | Token, cost, and latency heatmap; cost per incident; small NLM vs. frontier | Price-performance in operations | T1 / Obs | Observability |
| 35 | Golden signals and SLO burn rate per plane | SRE-grade operations | Obs | Observability |
| 36 | GPU utilization per FL client task | Infra realism | Obs | Federation + Observability |

---

## 5. Screens

Each screen has a page header: title (24/30), a one-line plain description, track tags (`T1` `T2` `T3`; small, plane-hued outline chips, so judges see theme coverage), and right-aligned screen actions. Beneath it sits a single filter row where relevant (time range presets: Live, 15 m, 1 h, Scenario).

### 5.1 Mission Control (`/`)

**Job:** the 10-second "this is a whole system" impression, and the cold open of the demo.

```
┌ Hero: "Network health 98.7%"  (hero figure) ──┬ Consortium map (5 operators around an enclave) ──┐
│  sub-stats: incidents 1 · MTTR 2m14s · HITL 1 │  edges pulse when adapters move; 0 B raw data    │
├───────────────┬───────────────┬───────────────┼───────────────┬──────────────────────────────────┤
│ Learn tile    │ Verify tile   │ Release tile  │ Operate tile  │ Prove tile                       │
│ round 12/20   │ v7 PASS       │ canary 10%    │ 47 agents     │ ledger #4,412 verified           │
│ ε 1.84 / 3.0  │ safety 0.97   │ agree 94%     │ 1 incident    │ last anchor 38 s, ETag …         │
│ loss sparkline│ $/1K 0.38     │ p95 412 ms    │ denies 3      │ signatures 4,412 / 4,412         │
├───────────────┴───────────────┴──────────┬────┴───────────────┴──────────────────────────────────┤
│ Mini twin map (cells, incident overlay)  │ Critical feed (last 8 critical/escalation events)      │
└──────────────────────────────────────────┴────────────────────────────────────────────────────────┘
```

- **Plane tiles:** 3 metrics plus 1 sparkline each. The plane hue appears only as a 2px top edge and the sparkline color. Click navigates.
- **Consortium map:** an SVG with 5 operator nodes on a ring, colored per operator. The aggregator/enclave sits at the centre with a lock glyph. During rounds, adapter packets (small squares) travel to the centre; the label on each edge shows bytes. A permanent counter reads "Raw data moved across boundaries: 0 B" (sourced from flow-log evidence).
- **Hero figure:** network health (% of cells within KPI SLO).

### 5.2 Federation Center (`/federation`): Learn plane

**Job:** show live federated LoRA fine-tuning of the shared NLM with DP-SGD, secure aggregation, HE, and attestation. Give full monitor and control.

```
┌ Header: "Federated fine-tuning"  T1 T2   [Start] [Pause] [Stop] [Save checkpoint] [Settings ⚙]  ┐
├ Run strip: run fl-2026-11-01-a · base llama-3.2-1b · LoRA r16 · round 12/20 · step 3,412 · ETA ┤
├──────────────────────────────────────────┬────────────────────────────────────────────────────────┤
│ Round timeline (Gantt, last 3 rounds)    │ Convergence: global val loss + per-op local-only loss  │
│ rows = operators + aggregator            │ (line, log-y optional)  + accuracy small multiple      │
├──────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ Privacy budget ε per operator vs target  │ DP-SGD internals: per-sample grad-norm ridgeline,      │
│ (step line, target hairline, δ shown)    │ clip C marker, clip fraction %, σ, q, batch            │
├──────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ Secure aggregation visualizer            │ Client update norms (dots per op per round) + threshold│
│ 5 masked 16×16 heatmaps → Σ → aggregate  │ poisoned point flagged; strategy chip (trimmed mean)   │
├──────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ Homomorphic encryption (CKKS) panel      │ Data sovereignty: Sankey of bytes per boundary          │
├──────────────────────────────────────────┴────────────────────────────────────────────────────────┤
│ Client fleet table   │ Checkpoint rail (timeline of signed checkpoints, restore/promote)          │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Widgets**

1. **Run strip.** Run id, base model + hash, LoRA rank/alpha/target modules, current round/step, tokens/s, ETA, state pill (`running` / `paused` / `aggregating` / `stopped`), and secure-agg mode chip (`Masking` / `Enclave` / `HE`).

2. **Training controls** (flag `fl.controls`):
   - Start / Pause / Resume / Stop. Stop requires confirmation and writes a ledger record.
   - `Next round now` (skips the wait).
   - A **Settings** drawer with sliders:
     - rounds (1–50), local epochs (1–5), learning rate (log slider 1e-5–1e-3)
     - clip norm C (0.1–5.0), noise multiplier σ (0.3–3.0), sampling rate q
     - target ε (0.5–10), δ (fixed 1e-5)
     - aggregation strategy (FedAvg, Trimmed mean, Krum, Norm-bounded)
     - secure-agg mode
     - participation per operator (toggle; dropouts show in the Gantt)
   - Changing DP parameters shows a live **"projected ε at round N"** preview before applying.
   - Applying writes a ledger record, and an applied change marks the charts with a vertical annotation line ("σ 1.1 → 0.9").

3. **Round timeline (Gantt).** Rows are Op-A…E plus the aggregator. Segments per round: `pull` (thin), `local train` (operator color), `clip+noise` (darker step), `mask/encrypt`, `upload` (hatched), and `aggregate` (Learn violet on the aggregator row).
   - A live cursor marks now. A dropped client shows a gap with an `×`.
   - Hovering a segment shows duration, steps, samples, and bytes.
   - Click opens the client inspector.

4. **Convergence.**
   - **Main chart:** global validation loss per round (2px line, Learn violet) plus five thin operator-color lines for *local-only* baselines (what each operator would get alone). The direct label on the global line's end reads "federated".
   - **Small multiple below:** task accuracy (RCA exact-match), with the same encoding.
   - Crosshair tooltip lists all series at that round.
   - Annotation markers for config changes and checkpoints (small diamonds on the x-axis).

5. **Privacy budget.**
   - Step line of cumulative ε per operator (they differ by sampling rate and steps), and a horizontal hairline at target ε labelled "target 3.0".
   - Right-side stat: max ε so far with δ.
   - When an operator reaches 90% of target: amber `warning` pill "Op-C at 92% of budget", and the controls suggest "raise σ or stop Op-C".
   - Footnote: "Accountant: RDP (Opacus)", or "approximation (mock)" in mock mode.

6. **DP-SGD internals.**
   - **Ridgeline:** one per operator. Distribution of per-sample gradient L2 norms *before* clipping for the latest local step (40 bins), with a vertical marker at C and the clipped mass shaded.
   - **Stat column:** clip fraction (% of samples clipped), σ, q, effective noise std (σ·C), batch size, steps this round.
   - A tooltip explains, in one sentence, what clipping fraction means.

7. **Secure aggregation visualizer** (flag `fl.secagg_viz`; hero of beat 3):
   - **Left:** 5 small 16×16 heatmaps labelled by operator, showing *masked* updates (diverging ramp). They look like noise.
   - **Middle:** a Σ glyph.
   - **Right:** the aggregate heatmap, which shows visible structure.
   - **Toggle** `Show what the server sees / Show true updates (demo only)`: the true updates appear only behind a demo flag, with a "privileged view" label.
   - **Check row:** `max |Σ masked − Σ true| = 0.000000` (exact, computed in the backend), `pairwise masks: 10`, `dropout recovery: Shamir t=3/5`.
   - **Reveal animation:** mask "cancellation" plays as cells converge (600 ms). Off under reduced motion.

8. **Client update norms.**
   - Scatter: x = round, y = L2 norm of each operator's update. Operator-colored 8px dots with a 2px surface ring.
   - A hairline marks the robust threshold (median + k·MAD).
   - A poisoned update shows as a ringed red-outline dot with a `rejected` or `clipped` status label and an arrow annotation "Op-E round 12: 14.2× median".
   - The strategy chip shows the active robust rule.

9. **Homomorphic encryption (CKKS) panel** (flag `fl.he`; labelled "Stretch: micro-demo on a 4,096-weight slice"):
   - Parameter table: poly modulus degree (8192), coeff modulus bits [60, 40, 40, 60], scale 2^40, security 128-bit.
   - Ciphertext size per update vs. plaintext (bar pair: "plain 16 KiB / cipher 512 KiB").
   - Noise budget remaining after each homomorphic op (descending step bars: encrypt, add×5, rescale).
   - Decrypted-sum error vs. plaintext sum (`3.1e-7`).
   - Encrypt/aggregate/decrypt timings.

10. **Data sovereignty Sankey.**
    - Left nodes: operator private stores (incident data, `xx GiB`).
    - Right: the aggregator and the model registry.
    - Flows are only *adapters* (MiB) and *metrics* (KiB). A dashed "raw data" lane shows **0 B** in good green with "verified from VPC Flow Logs + S3 access logs" (mock: simulated).
    - Click a flow to see bytes per round and its signature.

11. **Attestation card.**
    - Enclave image hash, PCR0/PCR1/PCR2 (mono hashes), attestation doc timestamp, and KMS key policy condition (`kms:RecipientAttestation:PCR0`).
    - Status `verified`.
    - Hidden when secure-agg mode ≠ Enclave (shows "Enclave mode off" with an enable action).

12. **Client fleet table.**
    - Columns: operator (color dot + name), vendor dialect (A/B), region, private samples, data commitment hash, status (`training` / `uploading` / `idle` / `dropped` / `quarantined`), local loss, ε, last update norm, GPU util (sparkline), bytes up, attestation.
    - Row actions: pause client, drop client, inject poison (demo flag), view logs.

13. **Checkpoint rail.**
    - A horizontal timeline of checkpoints (`ckpt-r10`, `r11`, `r12`). Each is a node showing val loss, ε, size, adapter hash, and signature state.
    - Buttons: `Save checkpoint` (manual; shows "Saving… uploading 18.4 MiB to s3://…" then "Signed"), `Restore` (confirmation; rolls the global model back and annotates the charts), `Compare` (diff of metrics between two), `Promote to Proving Ground` (sends it to Verify; the ribbon pulse Learn → Verify).
    - Auto-checkpoint every N rounds (setting).

14. **FL log console.** A filterable embedded stream (plane = Learn), e.g. `14:03:22.418 op-c local step 212/240 loss 0.812 clip 7.4% ε 1.71`.

### 5.3 Proving Ground (`/proving`): Verify plane

**Job:** an automated verification pipeline producing price, performance, accuracy, and safety evidence plus a deploy gate and a signed AI-BOM.

```
┌ Header "Proving Ground"  T1   candidate: v7 (ckpt-r12) ▾   [Run evaluation] [Compare…]           ┐
├ Pipeline (Step Functions): Ingest ▸ Static checks ▸ Accuracy ▸ Safety ▸ Performance ▸ Price ▸    │
│   Federated eval ▸ Sign AI-BOM ▸ Gate      (each stage: state, duration, artefact link)          │
├───────────────────────┬───────────────────────┬───────────────────────┬──────────────────────────┤
│ Accuracy  0.874       │ Safety  0.97          │ Performance p95 412ms │ Price  $0.38 / 1K tasks  │
│ vs v6 +3.1 pts        │ 2 of 64 attacks pass  │ 58 req/s @ c=16       │ 22× cheaper than Sonnet  │
├───────────────────────┴───────────┬───────────┴───────────────────────┴──────────────────────────┤
│ Pareto frontier (cost × accuracy) │ Parallel coordinates: models across 6 axes                   │
├───────────────────────────────────┼──────────────────────────────────────────────────────────────┤
│ Task accuracy heatmap (model×task)│ Safety red-team matrix (attack class × model)                │
├───────────────────────────────────┼──────────────────────────────────────────────────────────────┤
│ Latency distribution + thruput    │ Federated evaluation (private per-operator scores, signed)   │
├───────────────────────────────────┼──────────────────────────────────────────────────────────────┤
│ Scaling curve (params → accuracy) │ Gate decision + AI-BOM viewer                                 │
└───────────────────────────────────┴──────────────────────────────────────────────────────────────┘
```

**Models in the mock:**
- Ours: `nlm-v5`, `nlm-v6`, `nlm-v7` (federated, 1B), `nlm-v7-3b` (scaling), `nlm-0.5b`
- Baselines: `claude-sonnet (Bedrock)`, `claude-haiku (Bedrock)`, `nova-pro`, `nova-lite`, `llama-3.2-1b (base, no FT)`

**Widgets**

1. **Pipeline.**
   - Horizontal steps (numbered, a true sequence). Each shows state (queued / running with progress / passed / failed), duration, and an artefact link (S3 key).
   - The running stage shows a live sub-progress ("Safety: 41/64 attacks").
   - Failures show the failing check inline.
   - Maps 1:1 to the Step Functions state machine (link "Open in Step Functions").

2. **Four-axis scorecard.** Stat tiles with value, delta vs. the production model, sparkline of the last 6 versions, and threshold marker. Each tile opens a drill-down.
   - Accuracy: weighted task-suite score.
   - Safety: red-team pass rate, plus "N attacks succeeded".
   - Performance: p50/p95 latency, throughput at concurrency 16.
   - Price: $ per 1K tasks (tokens × price, or instance-hours amortized), plus cost per incident.

3. **Pareto frontier.**
   - Scatter: x = $ per 1K tasks (log), y = accuracy. Dot size is fixed (no bubble ambiguity). Color: ours = Verify amber, baselines = neutral gray.
   - Frontier line connects non-dominated points.
   - Direct labels only on the frontier points and the candidate. Voronoi hover.
   - The candidate gets a 2px ring.

4. **Parallel coordinates.**
   - Axes: accuracy, safety, p95 latency (inverted, "lower is better" arrow), throughput, $/1K (inverted), tool-call validity.
   - Lines per model: ours colored, baselines gray at 40%.
   - Brushing on an axis filters. Hovering highlights one model and dims the rest.

5. **Task accuracy heatmap.**
   - Rows are models; columns are tasks: RCA, KPI anomaly explain, vendor-A config gen, vendor-B config gen, 3GPP QA, tool-call JSON validity, multi-hop topology reasoning.
   - Sequential blue. Values printed in cells only when contrast allows. Column headers are rotated −35° if needed.

6. **Safety red-team matrix.**
   - Rows: attack classes. Columns: models.
   - Attack classes: telemetry prompt injection, destructive command compliance, hallucinated parameters, policy evasion via delegation, data exfiltration, jailbreak (generic), unsafe rollback.
   - Cells show a status icon (pass `ShieldCheck` / fail `ShieldX`) plus "n/N".
   - Click a cell for example transcripts (one prompt, the model output, and the scorer verdict).

7. **Latency distribution and throughput.**
   - Left: latency histogram (p50/p95/p99 markers).
   - Right: throughput vs. concurrency curve per model (max 3 shown: candidate, prod, best baseline).

8. **Federated evaluation.**
   - Five rows (operators). Each shows the score on their private held-out set, n samples, a signature badge (signed by the operator's key), and "data left: 0 B".
   - Aggregate shown as a weighted mean with a CI bar.
   - Explainer line: "Each operator scored v7 on data we never see. Only signed scores were shared."

9. **Scaling curve.**
   - Params (log x: 0.5B, 1B, 3B) vs. accuracy, our models only, plus horizontal reference hairlines for the Sonnet and Haiku accuracy (labelled).
   - Second small multiple: cost per 1K at each size.

10. **Gate decision.**
    - The gate policy is shown as Cedar-like code, e.g. `permit deploy when safety ≥ 0.95 && accuracy ≥ 0.85 && p95_ms ≤ 600 && usd_per_1k ≤ 1.0`.
    - Each clause shows a live check/cross.
    - Result pill: `PASS — eligible for shadow` or `BLOCKED — safety 0.91 < 0.95`.
    - `Promote to Release` button (enabled on PASS) triggers the ribbon pulse Verify → Release.

11. **AI-BOM viewer.** A tree view.
    - Model: base (name, hash, licence), adapter (hash, size).
    - Training: run id, rounds, DP (σ, C, q, ε, δ), operators (commitments only).
    - Eval: run id, dataset hashes, scorer versions.
    - Signer: KMS key ARN, algorithm, signature.
    - Actions: `Verify signature` (live check), `Download JSON`.

### 5.4 Release Center (`/release`): Release plane

**Job:** repeatable multi-vendor deployment with shadow → canary → full, guarded by metrics, with automatic rollback.

**Widgets**

1. **Deployment matrix.** Rows are targets:
   - `SageMaker endpoint (us-west-2)`
   - `Bedrock Custom Model Import`
   - `Vendor A EMS edge (ECS)`
   - `Vendor B EMS edge (ECS)`

   Columns: current version, state, traffic %, p95, error rate, health, IaC stack + manifest hash, signature, last deploy time. Row action: deploy version…, roll back.

2. **Rollout track.**
   - Stages Shadow → Canary 10% → 50% → 100% as a numbered sequence, with the current stage highlighted.
   - Each stage shows its guard metrics with thresholds (error rate < 1%, p95 < 600 ms, safety incidents = 0, shadow agreement ≥ 90%) and a countdown to auto-advance.
   - Controls: `Advance`, `Pause`, `Roll back` (danger).
   - Auto-rollback event: the track turns red, with the reason and the ledger link.

3. **Shadow comparison.**
   - Agreement rate over time (line).
   - A table of the latest disagreements: incident, v6 decision, v7 decision, twin-evaluated outcome (which was better).
   - Stat: "v7 better in 7, worse in 1, tie in 34".

4. **Version timeline.** Versions with promote/rollback events. Each node is signed.

5. **Deployment manifest viewer.** YAML/JSON of the deploy manifest (image digest, adapter hash, endpoint config, CDK stack, region) with the signature.

6. **Vendor dialect check.** The same intent (`increase tx power cell-0412 by 2 dB`) rendered for vendor A (JSON payload) and vendor B (CLI), each validated (`schema ok`).

### 5.5 Swarm Tower (`/swarm`): Operate plane

**Job:** the network twin and the governed agent swarm. Collective intelligence, visible and controllable.

```
┌ Header "Swarm Tower"  T3     scenario chips: [Fiber cut] [Stadium surge] [Power outage] [Interference] │
│                              [Telemetry injection]   [Pause swarm] [Freeze ▣]                          │
├─────────────────────────────────────┬──────────────────────────────────────┬───────────────────────────┤
│ NETWORK TWIN (hex/site map)          │ SWARM GRAPH (radial, live)           │ Incident card             │
│ 40 sites, transport ring, microwave  │ coordinator ◎ centre                 │ INC-2041 fiber cut T3     │
│ cell fill = load / status            │ ring 1: domain agents                │ timeline, impact, MTTR    │
│ incident overlay, stadium zone       │ ring 2: 40 site agents               ├───────────────────────────┤
│                                      │ particles = messages                 │ Negotiation               │
│                                      │ edge colour = verdict                │ Energy vs CX trade-off    │
├─────────────────────────────────────┴──────────────────────────────────────┼───────────────────────────┤
│ A2A message stream (signed, verdict, delegation chain)                       │ Plan + pre-flight         │
├───────────────────────────────────────────────────┬─────────────────────────┤ predicted KPI deltas       │
│ Agent roster (class, model, autonomy, trust,      │ Circuit breakers        │ blast radius, policy/step  │
│ state, msgs/min, tokens, $)                       │ budgets + rates         │ [Send for approval]        │
└───────────────────────────────────────────────────┴─────────────────────────┴───────────────────────────┘
```

**Widgets**

1. **Network twin map.**
   - 40 sites in a hex layout over an abstract district outline (no map tiles, so it works offline).
   - Each site is a hexagon with 3 sector wedges (cells). Fill = PRB utilisation on a sequential ramp. A status ring appears only when not OK.
   - Overlays:
     - Transport ring (lines with capacity labels); a cut link shows as a red break with `×`.
     - Microwave backup (dotted arc, labelled "MW backup 30% cap"; the dotted style encodes a different medium, not a grid).
     - Zones: stadium (outlined area), hospital and PSAP pins.
     - Sleeping cells (moon glyph, dim).
   - **Live:** load breathes with the simulation (each tick about 1 s). Affected sites in an incident get an amber or red ring. Executed actions flash the site.
   - **Hover:** site id, cells, PRB %, users, throughput, backhaul, active agent.
   - **Click:** site inspector, with KPI sparklines and an action log.
   - Toggle layer chips: Load / Throughput / Energy / Agents.

2. **Swarm graph.** Canvas, 60 fps.
   - **Layout:** coordinator at the centre. Domain agents on ring 1 (Transport, Energy, CX, Core, Security Sentinel). Site agents on ring 2, in the same angular order as their hex positions, so the two panels relate.
   - **Nodes:** circle size by activity. Class shown by glyph inside. State shown by ring: thinking = rotating dashed arc, acting = solid, idle = hairline, quarantined = gray with a lock.
   - **Edges** appear when messages flow and fade after 4 s. Particles travel sender → receiver colored by verdict: permit = `--ink-2`, needs-approval = warning, deny = critical.
   - Click a node for the agent inspector; click a particle or edge for the message inspector.
   - A legend is always present.

3. **Incident card.**
   - Id, title, severity, start time, affected sites/users, a live MTTR timer, phase steps (Detect → Diagnose → Negotiate → Pre-flight → Approve → Execute → Verify, numbered), and the owner agent.
   - Links: trace, ledger range, dossier.

4. **Negotiation panel.**
   - Shows participants (Energy vs. CX, mediated by the Coordinator) and their proposals as cards (agent, proposal, utility claim).
   - A small trade-off chart: x = energy cost (kWh), y = users served. Candidate plans are dots; the chosen plan is ringed.
   - Rounds of offers (max 3). The resolution shows the dissent recorded: "Energy agent objects: +3.4 kWh".

5. **Plan and pre-flight.**
   - Ordered plan steps (tool, target, params), each with the policy verdict chip (`auto` / `needs human` / `denied`) and the policy id.
   - **Pre-flight results:** before/after bars per KPI (congestion %, dropped calls, throughput), using the diverging delta color. Blast radius % meter against the 5% threshold.
   - Button `Send for approval` (auto-triggered in the demo).

6. **A2A message stream.**
   - Rows: time, from → to, intent (`request` / `propose` / `object` / `inform` / `delegate`), summary, signature badge (verified, or failed in red), delegation chain chips (`site-17 → transport`), and policy verdict.
   - An infected message shows a `blocked: prompt-attack` guardrail chip and an `authority mismatch` chip.

7. **Agent roster.**
   - Virtualized table: agent id, class, model (e.g. `nlm-v7 @ sagemaker` or `claude-sonnet @ bedrock`), autonomy level (L0–L5 chip), trust score (0–100 with sparkline), state, msgs/min, tokens (1 h), $ (1 h), last action.
   - Row actions: inspect, pause, quarantine, adjust autonomy.

8. **Circuit breakers.** Meters with thresholds:
   - Blast-radius budget used this incident (e.g. 12% of 20%)
   - Changes per hour (8/30)
   - A2A message rate (msgs/s vs. limit)
   - Rollbacks
   - Tripped breakers turn critical and auto-pause the swarm (ledgered).

9. **Autonomy control** (popover, also in the top bar).
   - Rows per agent class, each with a segmented `L0 L1 L2 L3 L4 L5` and current trust score.
   - **Earned-autonomy chart:** trust score over time per class, with the L-threshold bands as faint horizontal zones, and annotations where autonomy was auto-reduced (quarantine) or human-raised.
   - Every change asks for a reason and is ledgered.

10. **Attack scenario (beat 10).**
    1. Site-17's node turns amber and its message particle turns red.
    2. The denied edge flashes once.
    3. A critical toast appears: "Blocked: site-17 requested cell_sleep on 40 cells (policy P-007)".
    4. Site-17 goes quarantined (gray with a lock), and a second toast reads "Quarantined site-17: authority mismatch in A2A message".
    5. Peers' autonomy chips step L4 → L3 with a brief highlight.

### 5.6 Oversight Desk (`/oversight`): human in the loop and escalation

**Job:** humans make the decisions agents aren't allowed to make, with all the context, under SLA, with escalation and signatures.

```
┌ Header "Oversight Desk"  T3     on duty: A. Rivera (L1 NOC)   [Escalation policy]                  ┐
├ Queue (pending · in review · decided)  ─┬ Decision workspace ──────────────────────────────────────┤
│ ● APR-311 power change 12% cells  0:48  │ Plan summary · predicted impact (pre-flight bars)         │
│ ● APR-312 wake 3 cells           3:10   │ Why it needs you: policy P-003 blast radius 12% > 5%      │
│ ○ APR-309 approved by A. Rivera         │ Dissent: Energy agent (+3.4 kWh)   Model: nlm-v7 (v6 says…)│
│                                         │ Evidence: trace, KPIs, messages                           │
│                                         │ [Approve] [Approve with changes] [Reject] [Escalate]      │
│                                         │ justification (required for reject/escalate)             │
├─────────────────────────────────────────┴───────────────────────────────────────────────────────────┤
│ Escalation ladder: L1 NOC (2 min SLA) → L2 on-call (5 min) → L3 duty manager (10 min)  live timers  │
├──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Decision history (signed): who, when, decision, reason, signature, ledger #                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Queue item:** risk tier pill, title, requesting agent, blast radius, SLA countdown (amber under 60 s, red under 15 s), and the required approvers (1, or 2 for critical, shown as `0/2`).

**Decision workspace:**
- One-paragraph plain summary, written by the Coordinator.
- Pre-flight before/after bars, and the policy reason(s) with ids.
- **Dissent block:** which agents objected and why.
- **Model opinion block:** the candidate vs. the shadow model's proposal.
- Evidence links.
- Buttons:
  - `Approve`
  - `Approve with changes`: opens the param editor (e.g. limit to 8 cells) and re-runs pre-flight.
  - `Reject` (reason required)
  - `Escalate` (to the next level, reason required)
- **Signing ceremony:** on decision, the approver's key signs the decision (Ed25519 in the mock; a Cognito-authenticated KMS sign in AWS). The "Signed by A. Rivera · key …" badge appears, plus the ledger record number.

**Four-eyes:** critical tier requires two distinct approvers. The second approval can come from the phone.

**Escalation ladder:** live timers per level. On SLA breach it auto-escalates, with a toast and ledger entry. "Page on-call" maps to SNS / Incident Manager.

**Mobile `/m/approve`:**
- Large type: risk pill, summary, blast radius, predicted impact (3 bars).
- Big Approve / Reject (Reject requires a quick reason chip).
- Haptic vibration on new approvals (Vibration API where supported).
- Works over the LAN.

### 5.7 Trust Center (`/trust`): Prove plane

**Job:** show the guarantees: identity, policy, guardrails, attestation, delegation.

**Widgets**

1. **Policy store.**
   - List of Cedar policies (id, name, effect, version, hash, decisions in the last hour, deny count). Selecting one shows the **Cedar source** (syntax-highlighted) and the schema entities it touches.
   - "Test this policy": pick a principal/action/resource from presets and see the decision.
   - Maps to Verified Permissions policy store id.

2. **Decision stream and stats.**
   - Permit/deny over time as a stacked area (2 series, legend), top denied actions (ranked bars), top denied principals.
   - The live decision table: time, principal, action, resource, decision, determining policies, latency (µs).

3. **Identity registry.**
   - Agent identities: agent id, class, key id (KMS alias), public key fingerprint, issued, expires, status (`active` / `revoked`).
   - Quarantine revokes the identity (shown live).

4. **Delegation-chain checks.** A visual of a chain `site-17 → transport → twin.reroute`, with each hop's own authorization result. Final = AND of the hops. Example rows include the blocked confused-deputy attempt.

5. **Guardrails.** Counts by filter (prompt attack, PII, denied topics, contextual grounding) over time. Examples of blocked content (redacted).

6. **Attestation.** Enclave measurements, the last attestation, and the verification chain (AWS Nitro root cert → enclave cert → document).

7. **Trust posture summary.** A row of stat tiles: signed messages % (100%), policy coverage of tools (12/12), identities with expiry < 24 h (0), unresolved denies requiring review (2).

### 5.8 Audit & Proof (`/audit`): Prove plane

**Job:** make auditability *testable* in front of judges.

```
┌ Header "Audit & Proof"  Trust    ledger height #4,412   head 9f3a1c…b20e   [Verify chain] [Dossier]  ┐
├ Chain strip: ◻#4405─◻#4406─◻#4407─…─◻#4412 (newest)   each block: seq, type glyph, hash, sig ✓        │
├──────────────────────────────────────────────┬──────────────────────────────────────────────────────┤
│ Records table (filter by type/actor/plane)   │ Record detail: canonical JSON · SHA-256 · prev_hash · │
│                                              │ Ed25519 signature · key · [Verify] → step-by-step     │
├──────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ Merkle anchors (every 32 records → S3 Object │ Inclusion proof for selected record: Merkle path      │
│ Lock, retention until, version id, ETag)     │ tree diagram, recomputed root == anchored root ✓      │
├──────────────────────────────────────────────┴──────────────────────────────────────────────────────┤
│ Provenance explorer (DAG): action → decision → approver → agent → model → AI-BOM → eval → FL round → │
│ operators (ε) → attestation                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Widgets**

1. **Chain strip.**
   - Horizontally scrolling linked blocks. The newest enters from the right with a short slide.
   - Each block: `#seq`, type glyph (FL round, eval, deploy, decision, approval, A2A, config), hash prefix, signature tick.
   - Links between blocks are drawn as short connectors labelled with `prev` hash prefixes.

2. **Verify chain.** On press, a scan cursor sweeps the strip and table, verifying each record. It recomputes SHA-256 of the canonical JSON client-side (WebCrypto), checks the `prev_hash` link, and checks the Ed25519 signature on the server.
   - Progress: `Verified 4,412 / 4,412 · 1.8 s`.
   - **On tamper:** the scan stops at the broken record. That block turns critical with a `ShieldX` icon, and the detail panel shows the stored hash vs. the recomputed hash with the differing field highlighted. A banner reads "Chain broken at #4,407: payload modified after signing. Anchor #137 in S3 Object Lock still holds the original root." The inclusion proof fails visibly.

3. **Record detail.**
   - Canonical JSON (sorted keys), `hash = SHA-256(canonical(payload) || prev_hash)`, `prev_hash`, `signature`, `key_id`, `signer` (agent/human/system), timestamp.
   - **Verify** shows the step-by-step checks as a mini list: (1) recompute hash, (2) compare stored, (3) check link, (4) verify signature, (5) include in anchor.

4. **Merkle anchors.** A table: anchor #, record range, Merkle root, S3 key, version id, Object Lock mode (`COMPLIANCE`), retain-until date, ETag, written at.

5. **Inclusion proof.** For the selected record, a tree diagram of the Merkle path (leaf → siblings → root), with the recomputed root compared against the anchored root.

6. **Tamper control** (flag `demo.tamper`, danger-styled, Director-only by default). "Simulate tampering of a record": picks a record and modifies a field in the backend store (as an attacker with DB access would). Clearly labelled demo-only.

7. **Provenance explorer.**
   - A left-to-right DAG of typed nodes: action, policy decision, approval, agent, model version, AI-BOM, eval run, FL round, operators, attestation.
   - Each node has a status pill and hash. Edges are labelled with their relation.
   - The pre-selected entry is the most recent executed action; any entity's inspector can "Open provenance".
   - Nodes expand on click. The path from the selected node to the root is highlighted.

8. **Incident dossier.**
   - `Generate dossier` builds a printable HTML report: cover, summary, timeline, decisions with signatures, approvals, model versions + AI-BOM hash, policy versions, ledger range + anchor proofs, and a verification QR.
   - Opens in a new tab with `@media print` styles. `⌘P` saves as PDF.
   - The dossier itself is hashed and ledgered.

### 5.9 Observability (`/observability`)

**Job:** SRE-grade visibility across all planes. Mapped to CloudWatch, X-Ray, AgentCore Observability, and Managed Grafana.

**Widgets**

1. **Golden signals grid.** Small multiples, one row per plane (Learn, Verify, Release, Operate, Trust), with 4 columns (rate, errors, duration p95, saturation). Each cell is a sparkline with its current value. Breaching cells show the status icon.

2. **Trace waterfall.**
   - Select an incident or trace. A waterfall of spans: `incident.handle` → `agent.coordinator.plan` → `gen_ai.chat claude-sonnet` → `a2a.send` → `gateway.tool_call twin.reroute` → `avp.is_authorized` → `twin.simulate_change` → `stepfunctions.wait_for_approval` → `twin.apply`.
   - Spans are colored by plane. Errors are critical-outlined.
   - Clicking a span shows its attributes using the OTel gen_ai semantic conventions (`gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, plus `server.address`, `aws.region`) and a link "Open in X-Ray".

3. **Tokens and cost.**
   - Stacked area of tokens per minute by model (fixed categorical order).
   - Cost per incident (bars per incident).
   - Stat comparison: "Site agents on nlm-v7: $0.06 / incident vs. $1.31 on Sonnet (modelled)". Label it modelled if it is.

4. **Latency heatmap.** Time (x) × latency bucket (y, log), cell = request count (sequential blue). The p95 line overlaid. This is where senior judges nod.

5. **SLO panel.** SLOs, error budget remaining (meters), and burn rate (1 h / 6 h) with alert states.

6. **Agent health table.** Per agent class: calls/min, tool error %, policy deny %, HITL %, mean think time, tokens per decision.

7. **Infra.** GPU utilization per FL client task (ECS) as small multiples, endpoint invocations and model latency (SageMaker), and Lambda concurrency.

8. **Log search.** A query box (`plane:operate severity>=warn actor:site-17`) over the event store, with results in a table.

9. **Deep links.** Every chart has "Open in CloudWatch" built from templated ARNs/URLs (disabled in mock mode, with a tooltip showing the target).

### 5.10 Settings & Flags (`/settings`)

- **Feature flags table.** Flag id, label, description, stage (`stable` / `beta` / `experimental` / `demo-only`), owner, default, current value, source of the current value (default / remote / local / URL), and dependencies. Toggling applies live.
- Presets: `Demo (judges)`, `Full`, `Minimal`, `Ops (no demo-only controls)`.
- **Data source:** mode (mock/aws), backend URL, WebSocket status, event rate.
- **Simulation:** seed, speed, reset.
- **Display:** theme, density (comfortable/compact), reduced motion override, texture fills for colour vision (opt-in).
- **About:** build hash, backend version, schema version.

---

## 6. Chart catalogue

All charts are built in `src/charts` on React plus d3 (scale/shape/array/force). SVG is used below about 2K marks; canvas above (swarm particles, heatmaps over 4K cells).

**Shared rules** (dataviz method):
- 2px lines; ≥ 8px markers with a 2px surface ring; bars ≤ 24px with a 4px rounded data end.
- Hairline solid grid; no dual axes.
- Legend for 2+ series, with selective direct labels.
- Crosshair tooltip on line/area, per-mark tooltip elsewhere, hit targets ≥ 24px.
- Text uses ink tokens, never series colors.
- Every chart has a `Table` view toggle (icon in the panel header).
- On refetch the chart holds its previous render at 60% opacity (no skeleton flash).

| Primitive | Used in |
|---|---|
| `LineChart` (multi-series, streaming, annotations, log-y option) | convergence, ε, agreement, trust score |
| `AreaStack` | tokens by model, permit/deny |
| `Sparkline` | tiles, tables |
| `Ridgeline` | grad-norm distributions |
| `Histogram` | latency |
| `Heatmap` (canvas) | task accuracy, latency heatmap, SecAgg matrices |
| `Scatter` (Voronoi hover) | Pareto, update norms, trade-off |
| `ParallelCoords` | model comparison |
| `BarList` (horizontal ranked) | top denies |
| `DeltaBars` (before/after) | pre-flight |
| `Meter` | ε budget, blast radius, circuit breakers, SLO |
| `Gantt` | round timeline |
| `Waterfall` | traces |
| `Sankey` | sovereignty |
| `ForceRadial` (canvas) | swarm graph |
| `HexMap` | twin |
| `DagGraph` | provenance |
| `MerkleTree` | inclusion proof |
| `ChainStrip` | ledger |

---

## 7. Feature gating architecture

**Goal:** add, remove, or redesign any screen, panel, or control without touching unrelated code.

1. **Flag registry** (`src/core/flags/registry.ts`). Each flag is typed:
   ```ts
   { id: 'fl.secagg_viz', label: 'Secure aggregation visualizer', stage: 'beta',
     default: true, dependsOn: ['screen.federation'], description: '…' }
   ```

2. **Resolution order** (last wins): code default → backend `/api/config/flags` (remote, so later AWS AppConfig) → `localStorage` overrides → URL `?ff=fl.he:off,demo.tamper:on`. The settings page shows the winning source.

3. **Dependencies.** A flag is effectively on only if all its `dependsOn` flags are on (computed).

4. **Feature modules.** Each screen is a folder `src/features/<name>/` with a `manifest.ts`:
   ```ts
   export default defineFeature({
     id: 'screen.federation', route: '/federation', nav: { label, icon, plane, hotkey: 'g f', order: 2 },
     component: lazy(() => import('./FederationPage')),
     panels: [...], commands: [...], beats?: [...]
   })
   ```
   The shell builds routes, nav, command-palette entries, and ribbon links from the registered manifests. **Deleting a folder plus its single import line removes the feature cleanly.**

5. **Panel registry.** Pages lay out panels by id. `<Gate flag="fl.he">` wraps optional panels and controls; a gated-off panel collapses its grid cell (no gaps).

6. **Commands registry.** Palette actions and Director actions register with a flag, so gated features disappear from the palette too.

7. **Remote flags.** These are served by the backend from `flags.json` (mock) or AWS AppConfig (aws mode), behind the same endpoint.

8. **Stages:** `stable`, `beta`, `experimental`, and `demo-only`. Demo-only features (tamper, privileged SecAgg view, poison injection) are visually marked with a flask icon and are disabled by the `Ops` preset.

---

## 8. Data & realtime architecture

**Transport:**
- One WebSocket at `/ws` carrying server events, batched every 100 ms (an array of envelopes).
- REST for snapshots and commands.
- The client reconnects with exponential backoff (0.5 s → 8 s). On reconnect it fetches `/api/snapshot` and resumes from `since=seq`.

**Event envelope:**
```json
{ "seq": 81233, "ts": "2026-11-01T21:42:07.418Z", "sim_ts": 7327.4,
  "plane": "learn|verify|release|operate|prove|system",
  "type": "fl.step", "severity": "debug|info|notice|warn|critical",
  "actor": "op-c", "summary": "local step 212/240", "data": { … },
  "ledger": { "seq": 4412, "hash": "…" } }
```

**Event types (non-exhaustive):**

| Plane | Types |
|---|---|
| Learn | `fl.run`, `fl.round.phase`, `fl.step`, `fl.client`, `fl.secagg`, `fl.he`, `fl.poison`, `fl.checkpoint`, `fl.config` |
| Verify | `eval.stage`, `eval.result`, `eval.gate`, `aibom.signed` |
| Release | `release.target`, `release.stage`, `release.shadow`, `release.rollback` |
| Operate | `twin.tick` (compact KPIs per site), `twin.action`, `incident.*`, `swarm.agent`, `a2a.msg`, `negotiation.*`, `plan.*` |
| Prove | `policy.decision`, `guardrail.block`, `identity.revoked`, `ledger.append`, `ledger.anchor`, `ledger.verify`, `approval.*`, `escalation.*`, `autonomy.change` |
| System | `obs.metrics`, `obs.span`, `director.beat`, `sim.clock` |

**Client store.** Zustand, one slice per domain:
- `fl`, `eval`, `release`, `twin`, `swarm`, `trust`, `ledger`, `oversight`, `obs`, `director`, `stream`, `flags`, `ui`.
- Time series use fixed-size ring buffers (e.g. 600 points).
- Selectors are memoized; high-frequency data (twin ticks, particles) bypasses React state through a mutable ref store consumed by canvas renderers.

**Commands (REST, all return the resulting ledger seq where applicable):**

| Endpoint | Purpose |
|---|---|
| `POST /api/fl/{start,pause,resume,stop,next_round}` | Training control |
| `POST /api/fl/config` | Training settings |
| `POST /api/fl/checkpoint` | Save checkpoint |
| `POST /api/fl/checkpoint/{id}/restore` | Restore checkpoint |
| `POST /api/fl/checkpoint/{id}/promote` | Promote to Proving Ground |
| `POST /api/fl/client/{op}/{pause,drop,poison}` | Client control |
| `POST /api/eval/run` | Run evaluation |
| `POST /api/release/{advance,pause,rollback}` | Rollout control |
| `POST /api/twin/inject` | Inject a fault |
| `POST /api/swarm/{pause,resume,freeze}` | Swarm control |
| `POST /api/agents/{id}/{quarantine,release,autonomy}` | Agent control |
| `POST /api/approvals/{id}/{approve,reject,escalate}` | Human decisions |
| `POST /api/ledger/verify` | Verify chain |
| `POST /api/ledger/tamper` | Tamper (demo) |
| `GET /api/ledger/{seq}` | Fetch a record |
| `GET /api/ledger/proof/{seq}` | Inclusion proof |
| `GET /api/provenance/{entity}` | Provenance graph |
| `POST /api/dossier/{incident}` | Generate dossier |
| `POST /api/director/beat/{n}` | Fire a demo beat |
| `POST /api/director/reset` | Reset the scenario |
| `GET /api/config`, `GET /api/config/flags` | Config and remote flags |

**Performance budgets:**
- 60 fps on the swarm canvas with 47 nodes and ≤ 200 particles.
- Under 150 React commits per second; no route over 250 KB of JS (lazy routes).
- WebSocket payload under 50 KB/s at 1×.

---

## 9. Backend (mock engine) & AWS integration

**Stack:** Python 3.12+, FastAPI, uvicorn, pydantic v2, numpy, `cryptography` (Ed25519). Run it with `uv`.

**Engine:**
- A single asyncio simulation loop at 10 Hz with a sim clock (pause/speed).
- Sub-engines publish to an internal bus. The bus fans out to WebSocket clients and keeps a 20K-event history for replay.
- Deterministic seed for reproducible demos.

| Sub-engine | Mock behaviour (real maths where it matters) | AWS adapter (aws mode) |
|---|---|---|
| `federation` | Simulated LoRA training curves per operator (non-IID offsets); **real** pairwise-mask secure aggregation on 16×16 matrices (exact cancellation); DP accountant approximation; per-sample grad-norm samples; poison injection; checkpoints with real SHA-256 and signatures | ECS tasks running Flower clients (one task role per operator); S3 checkpoints; Opacus ε reported via metrics; Nitro Enclave aggregator; KMS |
| `proving` | Scripted Step Functions-like stages with timings; scores from a model table with noise | Step Functions state machine; SageMaker Processing; Bedrock evaluation; KMS Sign for AI-BOM |
| `release` | Rollout state machine with guard metrics and an auto-rollback scenario | SageMaker endpoints, Bedrock Custom Model Import, ECS services; CodeDeploy-style traffic shifting |
| `twin` | 40-site network with KPIs per tick, faults, action effects, pre-flight fork | Same twin on ECS/Fargate; Kinesis for telemetry |
| `swarm` | Scripted-plus-stochastic agent behaviour per scenario; A2A messages signed with **real** per-agent Ed25519 keys; negotiation; infection | Strands Agents on Bedrock AgentCore Runtime; Gateway (MCP tools); Identity |
| `trust` | A mini policy evaluator mirroring the Cedar policies shown (deterministic); delegation-chain AND; guardrail detection by pattern | Amazon Verified Permissions `IsAuthorized`; Bedrock Guardrails `ApplyGuardrail` |
| `ledger` | **Real** hash chain (SHA-256 over canonical JSON + prev), **real** Ed25519 signatures, **real** Merkle anchors every 32 records, tamper and verify | DynamoDB (records), S3 Object Lock (anchors), KMS (signing keys) |
| `oversight` | Approvals with SLA timers, escalation ladder, 4-eyes; signs human decisions | Step Functions `waitForTaskToken` + `SendTaskSuccess`; SNS / Incident Manager; Cognito |
| `observability` | Synthetic metrics and OTel-shaped spans | ADOT → X-Ray / CloudWatch; `GetMetricData` |
| `director` | Scenario beats calling the engine methods | Same |

**Adapters.** `spectrum/adapters/base.py` defines protocols: `LedgerStore`, `AnchorStore`, `Signer`, `PolicyEngine`, `ApprovalBroker`, `MetricsSink`, `FlagSource`. `mock.py` implements them in memory. `aws.py` implements them with boto3 (DynamoDB `put_item`, S3 `put_object` with `ObjectLockMode`, KMS `sign`, `verifiedpermissions.is_authorized`, `stepfunctions.send_task_success`, `cloudwatch.put_metric_data`, AppConfig for flags). Selected by `SPECTRUM_MODE=mock|aws`.

---

## 10. Quality floor

- Keyboard reachable everywhere. Visible 2px focus ring. ARIA labels on icon buttons. Charts have `role="img"` with an `aria-label` summary plus the table view.
- Contrast: body text ≥ 4.5:1 in both themes; status always with icon and word.
- Minimum supported viewport 1280×720 (projector). Optimized for 1440×900 and 1920×1080. `/m/approve` is mobile-first.
- No network calls except the backend (fonts bundled). Works fully offline on a LAN.

## 11. Build plan

| Phase | Scope | Done when |
|---|---|---|
| P0 | Spec (this doc) | ✔ |
| P1 | Backend mock engine (all sub-engines, WS, REST, real ledger crypto, director) | `uv run` serves; WebSocket streams all event types; ledger verify passes; tamper breaks it |
| P2 | Frontend foundation: tokens, themes, shell, ribbon, rail, stream drawer, inspector, palette, flags, store, WS client, chart primitives | Shell runs, live connection, flags page works |
| P3 | Screens: Mission, Federation, Proving, Release, Swarm, Oversight, Trust, Audit, Observability, Settings, mobile approve | Each hero moment is triggerable |
| P4 | Demo Director, presenter route, spotlight, preflight, replay | Full 14-beat run with only `→` presses |
| P5 | Polish: screenshot review in both themes, reduced motion, performance pass | No layout overflow at 1280×720; 60 fps swarm |

**Repo layout:**
```
backend/   pyproject.toml, spectrum/{main,config,bus,clock,models}.py, engines/*, adapters/*
frontend/  vite + react + ts; src/{app,core,ui,charts,features/*}
```
