"""Classify a small selection of English reviews using the shared rubric."""
import json
from jev_client import evaluate
from make_dataset import build
from review_task import QUESTIONS, normalize_jev, state_of


def main():
    for review in build()[:4]:
        payload = evaluate(state_of(review), QUESTIONS)
        print(f"\n[{review['id']}] {review['text']}")
        print(json.dumps(normalize_jev(payload["answers"]), indent=2))


if __name__ == "__main__":
    main()
