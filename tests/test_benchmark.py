"""Offline checks for fairness, validation, aggregation, and saved evidence."""
import json
import sys

import pytest
import requests

import compare_models as benchmark
from make_dataset import build
from review_task import QUESTIONS, normalize_jev, state_of, validate_prediction


def prediction(review):
    result = dict(review["expected"])
    result["rating"] = result["rating"][0]
    return result


def payload(model, review):
    pred = prediction(review)
    usage = {"cost": .001}
    if model == "luna":
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(pred)}}], "usage": usage}
    return {"answers": {"topic": {"choice": pred["topic"]}, "sentiment": {"choice": pred["sentiment"]},
                        "rating": {"score": pred["rating"] - 1},
                        "needs_reply": {"noul": int(pred["needs_reply"])}, "defect": {"noul": int(pred["defect"])}}, "usage": usage}


def row(model, review, pred, phase="measured", repeat=1):
    return {"model": model, "review_id": review["id"], "phase": phase,
            "repeat": repeat, "prediction": pred, "seconds": 1, "raw": {"usage": {"cost": .001}}}


def test_dataset_is_reproducible_and_explicit():
    reviews = benchmark.load_reviews(benchmark.DATA_PATH)
    assert len(reviews) == 100
    assert len({r["text"] for r in reviews}) == 100
    assert len({r["case_id"] for r in reviews}) == 50
    assert sorted(reviews, key=lambda r: r["id"]) == sorted(build(), key=lambda r: r["id"])
    assert any(r["expected"]["sentiment"] == "negative" and not r["expected"]["needs_reply"] for r in reviews)
    assert any(r["expected"]["sentiment"] == "positive" and r["expected"]["needs_reply"] for r in reviews)
    assert any(r["expected"]["defect"] and not r["expected"]["needs_reply"] for r in reviews)


@pytest.mark.parametrize("field,value", [("defect", "false"), ("needs_reply", 0), ("rating", True),
                                         ("rating", 2.5), ("rating", 6), ("topic", "invalid")])
def test_invalid_predictions_are_rejected(field, value):
    pred = prediction(build()[0])
    pred[field] = value
    with pytest.raises(ValueError):
        validate_prediction(pred)


def test_missing_fields_are_rejected():
    with pytest.raises(ValueError):
        validate_prediction({})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 5, True])
def test_invalid_jev_scores_are_rejected(value):
    answers = payload("jev", build()[0])["answers"]
    answers["rating"]["score"] = value
    with pytest.raises(ValueError):
        normalize_jev(answers)


def test_shared_rubric_and_no_label_leakage():
    assert json.dumps(QUESTIONS, ensure_ascii=False, sort_keys=True) in benchmark.SYSTEM_PROMPT
    assert set(state_of(build()[0])) == {"text"}


def test_paired_schedule_balances_order_and_counts_every_repeat():
    reviews = build()[:10]
    jobs = list(benchmark.schedule(reviews, 3, 42))
    assert jobs == list(benchmark.schedule(reviews, 3, 42))
    assert len(jobs) == 60
    for i in range(0, len(jobs), 2):
        assert jobs[i][1:] == jobs[i + 1][1:]
        assert jobs[i][0] != jobs[i + 1][0]
    for repeat in (1, 2, 3):
        first = [jobs[i][0] for i in range(0, len(jobs), 2) if jobs[i][2] == repeat]
        assert first.count("jev") == first.count("luna")


def test_every_repeat_and_failure_counts():
    review = build()[0]
    rows = [row("jev", review, prediction(review)), row("jev", review, None, repeat=2)]
    report = benchmark.summarize(rows, [review])["jev"]
    assert report["micro_accuracy"] == .5
    assert report["exact_match"] == .5
    assert report["errors"] == 1
    assert report["latency_all_attempts"]["n"] == 2
    assert report["latency_successes"]["n"] == 1


def test_odd_schedule_alternates_the_extra_first_position():
    jobs = list(benchmark.schedule(build()[:1], 4, 42))
    assert [jobs[i][0] for i in range(0, len(jobs), 2)] == ["jev", "luna", "jev", "luna"]


def test_truncated_completion_is_rejected():
    raw = payload("luna", build()[0])
    raw["choices"][0]["finish_reason"] = "length"
    with pytest.raises(ValueError, match="Incomplete"):
        benchmark.decode("luna", raw)


def test_warmup_cost_included_but_not_scored():
    review = build()[0]
    rows = [row("jev", review, None, "warmup"), row("jev", review, prediction(review))]
    report = benchmark.summarize(rows, [review])["jev"]
    assert report["micro_accuracy"] == 1
    assert report["reported_cost_usd"] == .002
    assert report["cost_complete"]
    rows[0]["raw"] = None
    report = benchmark.summarize(rows, [review])["jev"]
    assert report["reported_cost_usd"] == .001
    assert not report["cost_complete"]


def test_all_failure_and_empty_statistics():
    review = build()[0]
    report = benchmark.summarize([row("luna", review, None)], [review])["luna"]
    assert report["micro_accuracy"] == 0
    assert report["latency_successes"] is None
    assert benchmark.stats([]) is None
    assert benchmark.stats(list(range(1, 101)))["p95"] == 95


def test_failed_validation_preserves_raw_cost():
    review = build()[0]
    raw = {"choices": [], "usage": {"cost": .25}}
    result = benchmark.attempt("luna", review, 1, "measured", lambda r: raw)
    assert result["prediction"] is None
    assert result["raw"] == raw
    assert result["error"]["type"] == "IndexError"


def test_http_errors_keep_status_without_secret_text():
    def fail(review):
        response = requests.Response()
        response.status_code = 401
        raise requests.HTTPError("secret-text", response=response)
    result = benchmark.attempt("jev", build()[0], 1, "measured", fail)
    assert result["error"] == {"type": "HTTPError", "http_status": 401}
    assert "secret-text" not in json.dumps(result)


def test_luna_request_contract(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {}
    def post(url, **kwargs):
        body = kwargs["json"]
        assert body["reasoning"] == {"effort": "none"}
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"]
        assert body["messages"][0]["content"] == benchmark.SYSTEM_PROMPT
        assert set(json.loads(body["messages"][1]["content"])) == {"text"}
        return Response()
    monkeypatch.setattr(benchmark.requests, "post", post)
    benchmark.call_gpt(build()[0], "test-model", "none", "test-key")


def test_duplicate_ids_rejected(tmp_path):
    review = build()[0]
    path = tmp_path / "reviews.json"
    path.write_text(json.dumps([review, review]))
    with pytest.raises(ValueError, match="Duplicate"):
        benchmark.load_reviews(path)


def test_end_to_end_saved_run_with_fake_apis(tmp_path, monkeypatch):
    reviews = benchmark.load_reviews(benchmark.DATA_PATH, 2)
    by_text = {r["text"]: r for r in reviews}
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    monkeypatch.setattr(benchmark, "evaluate", lambda state, *a, **k: payload("jev", by_text[state["text"]]))
    monkeypatch.setattr(benchmark, "call_gpt", lambda review, *a, **k: payload("luna", review))
    monkeypatch.setattr(sys, "argv", ["compare_models.py", "--limit", "2", "--repeats", "2", "--output", str(tmp_path)])
    benchmark.main()
    directory = next(tmp_path.iterdir())
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["completed_attempts"] == 10
    assert "secret-test-key" not in (directory / "manifest.json").read_text()
    rows = [json.loads(line) for line in (directory / "attempts.jsonl").read_text().splitlines()]
    assert len(rows) == 10
    summary = json.loads((directory / "summary.json").read_text())
    for report in summary.values():
        assert report["micro_accuracy"] == 1
        assert report["attempts"] == 4
        assert report["cost_known_attempts"] == 5


def test_interruption_preserves_partial_run(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(benchmark, "evaluate", interrupt)
    monkeypatch.setattr(sys, "argv", ["compare_models.py", "--limit", "1", "--output", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 130
    directory = next(tmp_path.iterdir())
    assert json.loads((directory / "manifest.json").read_text())["status"] == "incomplete"
    assert (directory / "summary.json").exists()
