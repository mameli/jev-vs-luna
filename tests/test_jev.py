"""Integration tests for the Jev model (`~typesafe/jev-latest`) on OpenRouter.

Integration tests require --live and a valid OPENROUTER_API_KEY in .env.
"""

import os

import pytest
from dotenv import load_dotenv

from jev_client import JevError, evaluate

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

needs_key = pytest.mark.skipif(
    not API_KEY,
    reason="OPENROUTER_API_KEY is not set",
)

SUPPORT_TICKET = "Help! My payouts have been failing for 3 days. I'm losing money."

QUESTIONS = {
    "is_urgent": {
        "type": "noul",
        "instructions": "Does this message convey urgency?",
        "criteria": {
            "true": "Explicitly time-sensitive",
            "false": "No urgency expressed",
        },
    },
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "Payments, invoicing, refunds",
            "technical": "Bugs, outages, integrations",
            "sales": "Pricing, upgrades, new accounts",
        },
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"],
    },
}


@pytest.fixture(scope="module")
def result():
    return evaluate(state=SUPPORT_TICKET, questions=QUESTIONS)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(JevError):
        evaluate(
            state="hello",
            questions={
                "is_greeting": {"type": "noul", "instructions": "Is this a greeting?"}
            },
        )


@needs_key
@pytest.mark.live
def test_response_shape(result):
    assert result["model"]
    assert set(result["answers"]) == set(QUESTIONS)
    usage = result["usage"]
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] >= 0


@needs_key
@pytest.mark.live
def test_noul_answer(result):
    answer = result["answers"]["is_urgent"]
    assert answer["type"] == "noul"
    assert 0.0 <= answer["noul"] <= 1.0
    assert answer["noul"] > 0.5


@needs_key
@pytest.mark.live
def test_choice_answer(result):
    answer = result["answers"]["department"]
    assert answer["type"] == "choice"
    assert answer["choice"] in {"billing", "technical", "sales"}
    assert set(answer["probabilities"]) == {"billing", "technical", "sales"}
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert 0.0 <= answer["confidence"] <= 1.0


@needs_key
@pytest.mark.live
def test_score_answer(result):
    answer = result["answers"]["frustration"]
    assert answer["type"] == "score"
    assert 0.0 <= answer["score"] <= 2.0
    assert set(answer["probabilities"]) == {"0", "1", "2"}
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert 0.0 <= answer["confidence"] <= 1.0
