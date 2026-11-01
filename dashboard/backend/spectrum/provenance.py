"""Provenance graph and incident dossier, assembled from the ledger and engine state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core import canonical, sha256_hex

if TYPE_CHECKING:
    from .engine import Engine


READ_ONLY = {"get_kpis", "get_alarms", "get_topology", "trace_path", "simulate_change", "open_ticket",
             "dispatch_field_tech"}


def _find(records: list[dict], pred) -> dict | None:
    for rec in reversed(records):
        if pred(rec):
            return rec
    return None


def build(e: "Engine") -> dict:
    recs = e.ledger.records
    action = _find(recs, lambda r: r["type"] == "tool.call" and r["payload"].get("human_approved") is True
                   and r["payload"].get("decision") == "permit") or \
        _find(recs, lambda r: r["type"] == "tool.call" and r["payload"].get("tool") == "reroute") or \
        _find(recs, lambda r: r["type"] == "tool.call" and r["payload"].get("decision") == "permit"
              and r["payload"].get("tool") not in READ_ONLY)
    nodes: list[dict] = []
    edges: list[dict] = []

    def node(nid, type, label, status="verified", **detail):
        nodes.append({"id": nid, "type": type, "label": label, "status": status, **detail})
        return nid

    def edge(a, b, rel):
        edges.append({"from": a, "to": b, "rel": rel})

    p = action["payload"]
    a_id = node(f"rec-{action['seq']}", "action", f"{p['tool']} {p.get('resource', '')}", ledger_seq=action["seq"],
                hash=action["hash"], actor=action["actor"], ts=action["ts"])
    d_id = node("decision", "policy", f"{p['decision'].upper()} via {', '.join(p['policies'])}",
                policies=p["policies"], detail={"blast_radius_pct": p.get("blast_radius_pct"),
                                                "human_approved": p.get("human_approved")})
    edge(a_id, d_id, "authorised by")

    appr = _find(recs, lambda r: r["type"] == "approval.decision" and r["seq"] < action["seq"])
    if appr and p.get("human_approved"):
        ap = appr["payload"]
        h_id = node(f"rec-{appr['seq']}", "approval", f"Approved by {ap['by']} ({ap['device']})",
                    ledger_seq=appr["seq"], hash=appr["hash"], key_id=appr["key_id"])
        edge(d_id, h_id, "human approval")

    agent = e.swarm.agents.get(action["actor"])
    ag_id = node(f"agent-{action['actor']}", "agent", action["actor"],
                 status="quarantined" if agent and agent["status"] == "quarantined" else "verified",
                 key_id=action["key_id"], fingerprint=e.keys.fingerprint(action["key_id"]),
                 model=agent["model"] if agent else None)
    edge(a_id, ag_id, "executed by")

    if agent and agent["model"].startswith("nlm"):
        # the acting agent runs our federated model itself
        proposer = None
        model_id = agent["model"]
    else:
        # plans executed by the coordinator are proposed by site agents running the NLM; prefer the canary cohort
        proposer = next((a for a in e.swarm.agents.values() if a["class"] == "site" and a["model"] != "nlm-v6"), None) \
            or e.swarm.agents.get("site-21")
        model_id = proposer["model"] if proposer else "nlm-v6"
    if proposer:
        pr_id = node(f"agent-{proposer['id']}", "agent",
                     f"{proposer['id']} ({'canary cohort' if model_id != 'nlm-v6' else 'proposer'})", model=model_id,
                     key_id=proposer["key_id"])
        edge(ag_id, pr_id, "plan proposed with")
        src = pr_id
    else:
        src = ag_id
    m = e.proving.models.get(model_id, {})
    m_id = node(f"model-{model_id}", "model", model_id, accuracy=m.get("accuracy"), safety=m.get("safety"),
                p95_ms=m.get("p95_ms"), usd_per_1k=m.get("usd_per_1k"))
    edge(src, m_id, "runs")

    bom = e.proving.aiboms.get(model_id)
    if bom:
        b_id = node(f"aibom-{model_id}", "aibom", f"AI-BOM {bom['digest'][:8]}", hash=bom["digest"],
                    signature=bom["signature"], key=bom["signer"]["key"])
        edge(m_id, b_id, "described by")
        gate = _find(recs, lambda r: r["type"] == "eval.gate" and r["payload"].get("model") == model_id)
        if gate:
            g_id = node(f"rec-{gate['seq']}", "eval", f"Gate {'PASS' if gate['payload']['passed'] else 'BLOCKED'} "
                        f"({gate['payload']['run']})", ledger_seq=gate["seq"], hash=gate["hash"])
            edge(b_id, g_id, "verified by")
        ck_id = bom["training"]["checkpoint"]
        ck = next((c for c in e.fl.checkpoints if c["id"] == ck_id), None)
        if not ck:
            # trained in an earlier federated run: show exactly what the signed AI-BOM attests to
            tr = bom["training"]
            c_id = node(f"ckpt-{ck_id}", "checkpoint", f"{ck_id} (prior run)", hash=bom["model"]["adapter_hash"])
            edge(b_id, c_id, "trained as")
            r_id = node(f"run-{tr['run']}", "fl_round", f"Prior federated run, {tr['rounds']} rounds", sigma=tr["dp"]["noise_multiplier"],
                        decisions={o["id"]: "accepted" for o in tr["operators"]}, detail={"eps_max": tr["dp"]["eps_max"]})
            edge(c_id, r_id, "aggregated in")
            for o in tr["operators"]:
                o_id = node(f"op-{o['id']}", "operator", o["id"], eps=tr["dp"]["eps_max"], delta="1e-05",
                            commitment=o["data_commitment"], hash=o["data_commitment"])
                edge(r_id, o_id, "contributed (DP)")
            at = e.trust.attestation
            t_id = node("attestation", "attestation", "Nitro Enclave attestation", hash=at["pcr0"], kms_key=at["kms_key"])
            edge(r_id, t_id, "aggregated inside")
        if ck:
            c_id = node(f"ckpt-{ck_id}", "checkpoint", ck_id, hash=ck["adapter_hash"], ledger_seq=ck["ledger_seq"])
            edge(b_id, c_id, "trained as")
            rnd = _find(recs, lambda r: r["type"] == "fl.round" and r["payload"].get("round") == ck["round"])
            if rnd:
                rp = rnd["payload"]
                r_id = node(f"rec-{rnd['seq']}", "fl_round", f"Federated round {rp['round']}",
                            ledger_seq=rnd["seq"], hash=rnd["hash"], sigma=rp["sigma"], decisions=rp["decisions"])
                edge(c_id, r_id, "aggregated in")
                for op, eps in rp["eps"].items():
                    o_id = node(f"op-{op}", "operator", op, eps=eps, delta="1e-05",
                                commitment=rp["commitments"].get(op), status="verified" if
                                rp["decisions"].get(op, "accepted") == "accepted" else "rejected")
                    edge(r_id, o_id, "contributed (DP)")
                at = e.trust.attestation
                t_id = node("attestation", "attestation", "Nitro Enclave attestation", hash=at["pcr0"],
                            kms_key=at["kms_key"])
                edge(r_id, t_id, "aggregated inside")
    return {"root": a_id, "nodes": nodes, "edges": edges, "generated_at": e.clock.iso()}


def dossier(e: "Engine", incident_id: str) -> dict:
    inc = next((i for i in e.swarm.incidents if i["id"] == incident_id), None) or e.swarm.view_incident() \
        or (e.swarm.incidents[-1] if e.swarm.incidents else None)
    if not inc:
        return {"error": "no incident"}
    lo = inc.get("ledger_from", max(1, len(e.ledger.records) - 60))
    hi = inc.get("ledger_to", len(e.ledger.records))
    recs = e.ledger.records[lo - 1:hi]
    timeline = [{"seq": r["seq"], "ts": r["ts"], "type": r["type"], "actor": r["actor"],
                 "summary": _summ(r), "hash": r["hash"], "sig": r["sig"], "key_id": r["key_id"]}
                for r in recs if r["type"] not in ("a2a.message",) or r["payload"].get("intent") != "inform"]
    approvals = [i for i in e.oversight.items.values() if i.get("incident") == inc["id"]]
    used = sorted({pid for r in recs if r["type"] == "tool.call" for pid in r["payload"].get("policies", [])})
    pol = [{"id": p["id"], "name": p["name"], "version": p["version"], "hash": p["hash"]}
           for p in e.trust.policies if p["id"] in used]
    verify = e.ledger.verify(emit=False)
    anchors = [a for a in e.ledger.anchors if a["range"][1] >= lo and a["range"][0] <= hi]
    body = {
        "incident": {k: v for k, v in inc.items() if k in ("id", "title", "severity", "status", "started_at",
                                                           "mttr_s", "cost_usd", "affected_sites", "affected_users")},
        "plan": inc.get("plan"), "preflight": inc.get("preflight"), "negotiation": inc.get("negotiation"),
        "approvals": [{"id": a["id"], "title": a["title"], "status": a["status"], "approvals": a["approvals"],
                       "history": a["history"]} for a in approvals],
        "policies": pol, "timeline": timeline, "ledger_range": [lo, hi], "anchors": anchors,
        "verification": {"ok": verify["ok"], "checked": verify["checked"], "total": verify["total"]},
        "models": {"site_agents": sorted({a["model"] for a in e.swarm.agents.values() if a["class"] == "site"}),
                   "coordinator": e.swarm.agents["coordinator"]["model"]},
        "generated_at": e.clock.iso(), "generated_by": "SPECTRUM console",
    }
    digest = sha256_hex(canonical(body))
    rec = e.ledger.append("dossier.generated", "auditor", "prove", {"incident": inc["id"], "digest": digest},
                          key_id="kms:spectrum/auditor")
    return {**body, "digest": digest, "signature": rec["sig"], "ledger_seq": rec["seq"]}


def _summ(r: dict) -> str:
    p = r["payload"]
    t = r["type"]
    if t == "tool.call":
        return f"{p['tool']} {p.get('resource', '')}: {p['decision']}"
    if t == "a2a.message":
        return f"{p['from']} → {p['to']} ({p['intent']}): {p['verdict']}"
    if t.startswith("approval"):
        return f"{p.get('approval')} {p.get('decision', 'requested')} {p.get('by', '')}".strip()
    return t
