# SPECTRUM dashboard: setup and run guide

This guide gets the SPECTRUM dashboard prototype (`dashboard prototype 1.zip`) running on your laptop and walks you through the demo.

The dashboard runs fully offline in **mock mode**. You don't need an AWS account, API keys or internet once it's installed.

---

## 1. Install the prerequisites (one time)

| Tool | Version | How to install |
|---|---|---|
| **Node.js** | 22 LTS (minimum 20.19) | https://nodejs.org. Download the LTS installer. |
| **uv** (Python manager) | any recent version | macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh \| sh`<br>Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` |
| **make** | any | Already on macOS and Linux. On Windows, skip it and use the commands in section 3b. |

You **don't** need to install Python yourself, because uv downloads the right version (3.12+) automatically.

Check the installs in a **new** terminal window:

```bash
node -v     # should print v22.x (or at least v20.19)
uv --version
```

---

## 2. Unzip the project

1. Unzip `dashboard prototype 1.zip`.
2. Open a terminal **inside the folder that contains `Makefile`, `backend/` and `frontend/`**.

   Depending on how your unzip tool works, this may be one folder deeper. The folder name has spaces, so wrap it in quotes:

```bash
cd "dashboard prototype 1"
ls          # you should see: Makefile  README.md  backend  frontend  ...
```

---

## 3a. Run it on macOS / Linux

```bash
make install    # first time only: installs backend + frontend dependencies (takes 1-3 minutes)
make dev        # starts the backend (port 8787) and the dashboard (port 5173)
```

Then open **http://localhost:5173** in Chrome.

To stop it, press `Ctrl + C` in the terminal.

### Demo-day mode (recommended on stage)

```bash
make demo
```

This builds the dashboard and serves everything from **one process** on **http://localhost:8787**. It's more stable for a live presentation because no dev server is involved.

---

## 3b. Run it on Windows (no `make`)

Open **two** terminals in the project folder.

**Terminal 1: backend**

```powershell
cd backend
uv sync
$env:PYTHONPATH="."          # in cmd.exe use:  set PYTHONPATH=.
uv run uvicorn spectrum.main:app --host 0.0.0.0 --port 8787
```

**Terminal 2: dashboard**

```powershell
cd frontend
npm install
npx vite --host --port 5173
```

Then open **http://localhost:5173**.

---

## 4. Open the dashboard and check it's healthy

1. Open http://localhost:5173 (or http://localhost:8787 in demo-day mode).
2. Press **`D`** to open the **Demo Director** (the floating panel).
3. Click the **Preflight** tab. Every row should have a green tick:
   - Backend reachable
   - WebSocket live
   - Ledger verified clean
   - Scenario reset (beat 0)
   - Fonts loaded
4. If the "Scenario reset" row isn't green, **hold the Reset button** (or hold `R`). This puts the system back into the clean pre-demo state:
   - federated training round 12 complete;
   - about 3,400 signed ledger records.

The top bar shows a `MOCK` badge. Every panel has a small source chip showing which AWS service it maps to.

---

## 5. Give the demo

The demo is scripted as **14 beats**. Each press of `→` does three things:

- moves to the right screen;
- triggers the backend scenario, for example starting a training round, poisoning an operator, injecting an incident, running an attack or tampering with the ledger;
- spotlights the panels to talk about.

| Key | Action |
|---|---|
| `→` / `Space` / `PageDown` | Next beat (clickers work) |
| `←` / `PageUp` | Previous beat |
| `1`-`9` | Jump to a beat |
| `D` | Show or hide the Demo Director |
| `N` | Show presenter notes (what to say for each beat) |
| `S` | Spotlight on or off |
| `T` | Switch theme: Ops (dark) / Projector (light). **Use Projector on washed-out projectors.** |
| `P` | Pause or resume the simulation |
| `L` | Live event stream drawer |
| `⌘K` / `Ctrl+K` | Command palette (search any screen or action) |
| `g` + letter | Go to a screen |
| `?` | Show all shortcuts |

When a beat asks for **human approval**, approve it in the dashboard or from a phone (see below). The critical "four-eyes" approval needs **two different people**.

To run the demo again, **hold Reset** in the Demo Director.

### Screens

| URL | Screen |
|---|---|
| `/` | Mission Control: the whole lifecycle at a glance |
| `/federation` | Federation Center: federated fine-tuning with DP, secure aggregation, enclave, checkpoints |
| `/proving` | Proving Ground: price / performance / accuracy / safety, AI-BOM, deploy gate |
| `/release` | Release: shadow, canary and full rollout across vendors |
| `/swarm` | Swarm Tower: agent swarm, signed agent-to-agent messages, network twin |
| `/oversight` | Human oversight: approvals, SLA, escalation |
| `/trust` | Trust Center: policies, guardrails, attestation |
| `/audit` | Audit: signed ledger, tamper detection, Merkle proofs, provenance |
| `/observability` | Observability: traces, cost, latency, SLOs |
| `/settings` | Settings and feature flags |
| `/presenter` | Presenter view, for driving the projector from a second laptop |
| `/m/approve` | Mobile approval page, for phones |

---

## 6. Phone approvals (optional, great for the demo)

1. Connect the phone to the **same Wi-Fi** as the laptop. A phone hotspot is the most reliable option at a venue.
2. In the Demo Director, go to **Preflight** and scan the **QR code** with the phone.
3. Approvals sent from the phone appear live on the dashboard and are signed into the ledger.

If the phone can't connect, allow incoming connections for **Node** and **Python** when the firewall asks. On macOS, check System Settings → Network → Firewall.

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `make: command not found` (Windows) | Use section 3b. |
| `uv: command not found` right after installing | Close the terminal and open a new one. |
| Vite error mentioning Node version | Install Node 22 LTS. Node 18 and old Node 20 versions don't work. |
| Dashboard loads but shows "Offline" | The backend isn't running. Start it (`make api`, or Terminal 1 in 3b) and check that http://localhost:8787/api/director/preflight responds. |
| `address already in use` on port 8787 or 5173 | Something is already running there. Close the old terminal, or on macOS/Linux run `lsof -ti:8787 \| xargs kill`. |
| Ledger shows "broken" | That's expected after the tamper beat (beat 12). Click "Restore originals" on the Audit screen, or hold Reset. |
| Demo got into a weird state | Hold **Reset** in the Demo Director. It's deterministic, so you get the same starting point every time. |

---

## 8. Further reading (in this folder)

- `README.md`: architecture, what is real and what is simulated, and the AWS service mapping.
- `DASHBOARD_SPEC.md`: full UI/UX and feature spec.
- `SPECTRUM.md`: system design and the 7-minute run-of-show.
- `IDEAS.md`: the original idea book.
