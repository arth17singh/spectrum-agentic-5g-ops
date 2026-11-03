# SPECTRUM console: one-command dev and demo.
.PHONY: dev api web install test check build demo evidence-rebuild evidence-sync demo-real gate

install:            ## install backend + frontend dependencies
	cd backend && uv sync --group dev
	cd frontend && npm install

api:                ## backend only (mock engine, hot reload) on 127.0.0.1:8787, this laptop only
	cd backend && PYTHONPATH=. uv run uvicorn spectrum.main:app --host 127.0.0.1 --port 8787 --reload --reload-dir spectrum

web:                ## frontend only (Vite) on localhost:5173; its proxy looks local to the API, so keep it off the LAN
	cd frontend && npx vite --port 5173

dev:                ## backend + frontend together
	@trap 'kill 0' INT TERM; $(MAKE) api & $(MAKE) web & wait

test:               ## backend tests + cross-language canonical hash check + typecheck
	cd backend && uv run pytest -q
	node frontend/scripts/check-canonical.mjs
	cd frontend && ./node_modules/.bin/tsc -p tsconfig.app.json --noEmit

build:              ## production frontend bundle (served by the backend on :8787)
	cd frontend && npx vite build

demo: build         ## demo-day mode: a single process serves API + built UI on :8787; phones need the access token
	cd backend && PYTHONPATH=. uv run uvicorn spectrum.main:app --host 0.0.0.0 --port 8787

evidence-rebuild:   ## offline: re-verify evidence/ (chain, signatures, file hashes) and reprint per-run results
	cd backend && uv run python -m spectrum.evidence rebuild $(if $(DIR),--dir $(DIR),)

EVIDENCE ?= $(abspath .evidence)
HOST ?= 0.0.0.0
PORT ?= 8787
AWS_PROFILE ?= fyuz

evidence-sync:      ## mirror S3 evidence/ into $(EVIDENCE) (read-only; new/changed keys only; checks sha256 vs the job logs' attestations). CHECK=1 caches check_unit per run; HANDIN=<git ref> copies judging.json + superseded.json
	cd backend && AWS_PROFILE=$(AWS_PROFILE) uv run python ../../tools/evidence_sync.py --dest $(EVIDENCE) --jobs --attest $(if $(CHECK),--check,) $(foreach r,$(RUN),--run $(r)) $(if $(HANDIN),--handin $(HANDIN),)

demo-real: build    ## like demo, but every screen reads the mirrored real evidence ($(EVIDENCE)) on $(HOST):$(PORT) (HOST=/PORT= to override); run evidence-sync first. Starts in ~5 s; the evidence ledger then ingests new files in the background (~1 s per 1,000; a fresh 65k mirror ~1 min, kept in $(EVIDENCE)/ledger.jsonl), so start it once before the demo to warm it
	cd backend && SPECTRUM_EVIDENCE_DIR=$(EVIDENCE) SPECTRUM_RECORDS_ROOT=$(EVIDENCE) PYTHONPATH=. uv run uvicorn spectrum.main:app --host $(HOST) --port $(PORT)

gate:               ## merge gate (worktrees only, never the main checkout): tsc, oxlint, registry, build, canonical, pytest, route smoke on :8790
	../scripts/gate.sh
