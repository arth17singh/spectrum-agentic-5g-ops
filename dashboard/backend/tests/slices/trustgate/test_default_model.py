from spectrum.slices.trustgate.logic import _live_since, default_model


def test_opens_on_newest_version_with_scored_changes():
    fed = {"model_id": "t2-fed", "version": "v1", "n_scored": 0, "last_scored_at": None}
    central = {"model_id": "t2-central", "version": "v0", "n_scored": 14, "last_scored_at": "2026-11-03T05:31:52Z"}
    old = {"model_id": "t2-old", "version": "v0", "n_scored": 3, "last_scored_at": "2026-11-03T04:00:00Z"}
    assert default_model([fed, old, central]) is central
    assert default_model([fed]) is fed
    assert default_model([]) is None


def test_live_since_prefers_the_model_record():
    assert _live_since({"aws_time": "2026-11-03T05:15:43Z"}, [], []) == ("2026-11-03T05:15:43Z", "model record AWS time")
    assert _live_since(None, [{"time": "2026-11-03T05:20:00Z"}], [{"time": "2026-11-03T05:10:00Z"}]) == (
        "2026-11-03T05:10:00Z", "first change or control record")
    assert _live_since(None, [], []) == (None, None)
