# SPECTRUM console: one-command dev and demo.
.PHONY: dev api web install test check build demo

install:            ## install backend + frontend dependencies
	cd backend && uv sync --group dev
	cd frontend && npm install

api:                ## backend only (mock engine, hot reload) on :8787
	cd backend && PYTHONPATH=. uv run uvicorn spectrum.main:app --host 0.0.0.0 --port 8787 --reload --reload-dir spectrum

web:                ## frontend only (Vite, LAN-reachable for phone approvals) on :5173
	cd frontend && npx vite --host --port 5173

dev:                ## backend + frontend together
	@trap 'kill 0' INT TERM; $(MAKE) api & $(MAKE) web & wait

test:               ## backend tests + cross-language canonical hash check + typecheck
	cd backend && uv run pytest -q
	node frontend/scripts/check-canonical.mjs
	cd frontend && ./node_modules/.bin/tsc -p tsconfig.app.json --noEmit

build:              ## production frontend bundle (served by the backend on :8787)
	cd frontend && npx vite build

demo: build         ## demo-day mode: a single process serves API + built UI on :8787
	cd backend && PYTHONPATH=. uv run uvicorn spectrum.main:app --host 0.0.0.0 --port 8787
