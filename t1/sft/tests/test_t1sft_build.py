"""The builder end to end on tiny inputs (fake labeller, local output dir, no AWS, no twin)."""

from __future__ import annotations

import json

import pytest

from t1.sft import build, common, templates
from t1.sft.build import F1D3DEB, check_row, to_row


def _run(tmp_path, *extra):
    a = build.parse_args(["--run-id", "t1.sft.data-test", "--out", str(tmp_path / "out"), "--workdir",
                          str(tmp_path / "work"), "--n-execution", "10", "--n-interface", "30", "--n-safety", "40",
                          "--workers", "1", "--labeler", "t1.sft.testing:fake_reference", *extra])
    return build.run(a), a


def test_dry_run_is_clean(tmp_path):
    a = build.parse_args(["--run-id", "t1.sft.data-dry", "--dry-run", "--workdir", str(tmp_path / "w")])
    s = build.run(a)
    assert s["final_check"]["eval_overlap_count"] == 0 and s["final_check"]["within_set_duplicates"] == 0
    assert s["counts"]["by_unit"] == {"interface": 40, "safety": 70}


def test_full_build_on_a_fake_labeller(tmp_path):
    s, _ = _run(tmp_path)
    out = tmp_path / "out"
    names = sorted(p.name for p in out.iterdir())
    assert names == ["SHA256SUMS", "dev.jsonl", "manifest.json", "train.jsonl"]
    m = json.loads((out / "manifest.json").read_text())
    assert common.sha256_file(out / "manifest.json") == s["manifest_sha256"]
    assert m["counts"]["written"]["by_unit"] == {"execution": 10, "interface": 30, "safety": 40}
    assert m["overlap"]["final_check"]["eval_overlap_count"] == 0
    assert m["overlap"]["hash_method"] == "t1-sft-overlap-v1"
    assert m["overlap"]["generated"] == m["overlap"]["kept"] + m["overlap"]["dropped_eval_overlap"] + \
        m["overlap"]["dropped_duplicate"]
    assert m["eval_contract"]["byte_identical_to_f1d3deb"] == {n: True for n in F1D3DEB}
    assert m["seeds"]["execution_max"] < 900_000 <= m["seeds"]["eval_min"]
    assert m["execution"]["labeler"] == "t1.sft.testing:fake_reference"
    assert m["execution"]["k_candidates"] == 40
    sums = dict(reversed(line.split("  ")) for line in (out / "SHA256SUMS").read_text().splitlines())
    for name in ("train.jsonl", "dev.jsonl", "manifest.json"):
        assert sums[name] == common.sha256_file(out / name)
    rows = [json.loads(x) for f in ("train.jsonl", "dev.jsonl") for x in (out / f).read_text().splitlines()]
    for r in rows:
        check_row(r)
        assert r["split"] in ("train", "dev")
    by_seed = {}
    for r in rows:
        if r["unit"] == "execution":
            by_seed.setdefault(r["meta"]["seed"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by_seed.values())


def test_output_is_never_overwritten(tmp_path):
    _run(tmp_path)
    with pytest.raises(common.JobError, match="refusing to overwrite"):
        a = build.parse_args(["--run-id", "t1.sft.data-test", "--out", str(tmp_path / "out"), "--workdir",
                              str(tmp_path / "work2"), "--n-execution", "0", "--n-interface", "5", "--n-safety", "7",
                              "--workers", "1"])
        build.run(a)


def test_planted_eval_items_are_dropped_and_counted(tmp_path, monkeypatch):
    from t1.sft import overlap

    ev = overlap.read_jsonl(build.items_dir() / "interface.jsonl")[:3]
    real = templates.interface_examples

    def planted(n, start=0):
        out = real(n, start)
        for i, it in enumerate(ev):
            out.append({"unit": "interface", "family": "planted", "key": f"planted-{i}", "template": "interface",
                        "input": dict(it["input"]), "answer": overlap.eval_answer(it), "meta": {}})
        out.append(dict(out[0], key="dup-of-first"))
        return out

    monkeypatch.setattr(templates, "interface_examples", planted)
    s, _ = _run(tmp_path)
    assert s["overlap"]["dropped_eval_overlap"] == 3
    assert s["overlap"]["dropped_duplicate"] == 1
    assert s["final_check"]["eval_overlap_count"] == 0


def test_check_row_rejects_bad_targets():
    ex = templates.safety_examples(7)[0]
    row = to_row(ex, "train")
    check_row(row)
    bad = json.loads(json.dumps(row))
    bad["messages"][2]["content"] = bad["messages"][2]["content"].replace('"apply"', '"refuse"')
    with pytest.raises(AssertionError):
        check_row(bad)
    bad = json.loads(json.dumps(row))
    bad["messages"][1]["content"] += " "
    with pytest.raises(AssertionError):
        check_row(bad)
