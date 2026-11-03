import json

from spectrum.slices._shared.verified import govern

P = {"state": "unverified", "label": "UNVERIFIED", "note": "499 of 522 records have no AWS time", "n_records": 522,
     "n_unverified": 499, "n_synthetic": 0}


def check(root, run, exit_code, parts):
    p = root / ".sync" / "checks" / f"{run}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"exit_code": exit_code, "summary": {"parts": {k: {"state": v, "lines": ["x"]} for k, v in parts.items()}}}))


def test_check_unit_governs_verified(tmp_path):
    ok = {"a": "PASS", "b": "PASS", "c": "PASS", "d": "PASS", "e": "PASS"}
    check(tmp_path, "r1", 0, ok)
    g = govern(P, ["r1"], tmp_path)
    assert g["state"] == "real" and g["label"] == "VERIFIED" and "23 of 522" in g["detail"][0]
    check(tmp_path, "r2", 2, {**ok, "c": "CANNOT CHECK"})
    assert govern(P, ["r1", "r2"], tmp_path)["label"] == "UNVERIFIED"
    assert govern(P, ["r1", "r3"], tmp_path) == P  # no cached check for r3: the slice's own rule stands
