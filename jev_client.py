"""Minimal client for the TypeSafe Jev model served through OpenRouter.

Jev does not generate text. It evaluates a `state` against a map of typed
questions (noul / choice / score) and returns calibrated answers.

Endpoint: POST https://openrouter.ai/api/alpha/decisions
Model:    ~typesafe/jev-latest
"""

import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "~typesafe/jev-latest"


class JevError(RuntimeError):
    """Missing credentials or unexpected payload; HTTP errors use requests.HTTPError."""


def evaluate(
    state: str | dict | list,
    questions: dict[str, dict[str, Any]],
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """Evaluate `state` against `questions` and return the raw API payload."""
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise JevError(
            "OPENROUTER_API_KEY is not set: configure it in .env"
        )

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={"model": model, "state": state, "questions": questions},
        timeout=timeout,
    )

    response.raise_for_status()

    payload = response.json()
    if "answers" not in payload:
        raise JevError("Unexpected OpenRouter response: missing answers")
    return payload
