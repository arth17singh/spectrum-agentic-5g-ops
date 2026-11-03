"""check_unit reads per-record S3 objects on a thread pool. These tests pin that it changes timing only:
many records, small list pages, a slow store, and failing reads give the same report and the same error
as reading one object at a time. Offline (FakeAws); nothing here can reach AWS."""

import io
import json
import threading
import time
from datetime import timedelta

import pytest

import check_unit
import unitfix
from evfake import FakeAws, FakeClientError

RID = unitfix.RUN_ID
N_EXTRA = 400


class SequentialEvidence(check_unit.S3Evidence):
    """The old behaviour: one GET after the other."""

    def get_many(self, rels, workers=1):
        return check_unit.fetch_parallel(self.get, rels, 1)


def many(tmp_path, n=N_EXTRA, page_size=37):
    """The per-record-object fixture with n more item objects (and small list pages, so listing paginates)."""
    fx = unitfix.build(tmp_path, "R-objects", unitfix.BASE_SHA, objects=True)
    aws, uri = FakeAws.from_fixture(fx)
    aws.s3.page_size = page_size
    base = json.loads(next(v["body"] for (b, k), v in aws.s3.objects.items() if "/item/" in k))
    for i in range(n):
        rec = {**base, "item_id": f"extra-{i:05d}", "created_at": f"2026-11-02T18:2{i % 10}:00.{i % 1000:03d}Z"}
        body = (json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n").encode()
        aws.s3.put(unitfix.BUCKET, f"{unitfix.PREFIX}records/{RID}/item/x{i:05d}.json", body,
                   "2026-11-02T18:2%d:%02dZ" % (i % 10, i % 60))
    return fx, aws, uri


def run(aws, uri, ev_cls, fx):
    ev = ev_cls(aws.client("s3"), uri)
    ck = check_unit.Checker(RID, ev, aws, fx / "ledger", poll_s=0.0, now=aws.now())
    code = ck.run()
    out = io.StringIO()
    ck.report(out)
    return code, out.getvalue(), json.dumps(ck.as_json(), sort_keys=True), ck


def test_many_records_same_report_as_one_by_one(tmp_path):
    fx, aws, uri = many(tmp_path)
    seq = run(aws, uri, SequentialEvidence, fx)
    par = run(aws, uri, check_unit.S3Evidence, fx)
    assert len(par[3].records["item"]) == N_EXTRA + len(unitfix.ITEMS)  # every object was read and listed
    assert par[:3] == seq[:3]
    assert par[3].record_bytes == seq[3].record_bytes
    assert list(par[3].record_bytes) == sorted(par[3].record_bytes)
    assert par[3].records == seq[3].records and par[3].record_src == seq[3].record_src
    # the same records, in the same order, came out of the sort that merges per-object records
    assert [r["item_id"] for r in par[3].records["item"]] == [r["item_id"] for r in seq[3].records["item"]]


def test_reads_really_overlap_and_stay_bounded(tmp_path):
    fx, aws, uri = many(tmp_path, n=120)
    real, lock, state = aws.s3.get_object, threading.Lock(), {"now": 0, "max": 0}

    def slow(**kw):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.02)
        try:
            return real(**kw)
        finally:
            with lock:
                state["now"] -= 1

    aws.s3.get_object = slow
    run(aws, uri, check_unit.S3Evidence, fx)
    assert 2 <= state["max"] <= check_unit.FETCH_WORKERS


def test_parallel_is_much_faster_on_a_slow_store(tmp_path):
    fx, aws, uri = many(tmp_path, n=160)
    real = aws.s3.get_object

    def slow(**kw):
        time.sleep(0.01)
        return real(**kw)

    aws.s3.get_object = slow
    t0 = time.monotonic()
    run(aws, uri, SequentialEvidence, fx)
    t_seq = time.monotonic() - t0
    t0 = time.monotonic()
    run(aws, uri, check_unit.S3Evidence, fx)
    t_par = time.monotonic() - t0
    assert t_par < t_seq / 3, (t_seq, t_par)


def test_first_failing_read_in_key_order_is_the_error_raised(tmp_path):
    fx, aws, uri = many(tmp_path, n=60)
    real = aws.s3.get_object
    bad = {f"{unitfix.PREFIX}records/{RID}/item/x00050.json", f"{unitfix.PREFIX}records/{RID}/item/x00007.json"}

    def flaky(Bucket, Key, **kw):
        if Key in bad:
            time.sleep(0.05 if Key.endswith("x00007.json") else 0)  # the later key fails first in time
            raise FakeClientError("AccessDenied", f"denied {Key}", "GetObject", 403)
        return real(Bucket=Bucket, Key=Key, **kw)

    aws.s3.get_object = flaky
    msgs = []
    for cls in (SequentialEvidence, check_unit.S3Evidence):
        with pytest.raises(FakeClientError) as ei:
            run(aws, uri, cls, fx)
        msgs.append(str(ei.value))
    assert msgs[0] == msgs[1] and "x00007.json" in msgs[0]


def test_missing_object_reads_as_empty_like_before(tmp_path):
    fx, aws, uri = many(tmp_path, n=20)
    del aws.s3.objects[(unitfix.BUCKET, f"{unitfix.PREFIX}records/{RID}/item/x00003.json")]  # listed -> gone
    ev = check_unit.S3Evidence(aws.client("s3"), uri)
    got = ev.get_many([f"records/{RID}/item/x00003.json", f"records/{RID}/item/x00004.json"])
    assert got[f"records/{RID}/item/x00003.json"] is None and got[f"records/{RID}/item/x00004.json"]


def test_fetch_parallel_edge_cases():
    assert check_unit.fetch_parallel(str.upper, []) == {}
    assert check_unit.fetch_parallel(str.upper, ["a"]) == {"a": "A"}
    out = check_unit.fetch_parallel(lambda k: 1 // (k - 2), [1, 2, 3], 4)
    assert out[1] == -1 and out[3] == 1
    with pytest.raises(ZeroDivisionError):
        out[2]
