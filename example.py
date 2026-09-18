"""Runnable example: route a support ticket with Jev."""

from jev_client import evaluate

result = evaluate(
    state="Help! My payouts have been failing for 3 days.",
    questions={
        "is_urgent": {
            "type": "noul",
            "instructions": "Does this message convey urgency?",
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
    },
)

print("model:", result["model"])
print("urgent:", result["answers"]["is_urgent"]["noul"])
print("department:", result["answers"]["department"]["choice"])
