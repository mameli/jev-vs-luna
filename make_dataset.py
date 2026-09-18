"""Generate English synthetic diagnostic cases; these are not real-world ground truth."""
import json
import random
from pathlib import Path

SEED = 42
# Every row explicitly defines topic, sentiment, needs_reply, defect, and text.
# Two contextual variants per case are correlated, not independent samples.
CASES = [
    ("product", "positive", False, False, "The blender works perfectly and feels solid."),
    ("product", "positive", False, False, "The headphones sound fantastic and fit comfortably."),
    ("product", "positive", False, False, "Easy to assemble and very sturdy. Great desk."),
    ("product", "neutral", False, False, "It works as described, but nothing about it stands out."),
    ("product", "neutral", False, False, "The lamp is neither impressive nor disappointing."),
    ("product", "negative", True, True, "The blender will not turn on. Please arrange a replacement."),
    ("product", "negative", True, True, "The battery dies after five minutes. I still need a fix."),
    ("product", "negative", True, True, "The switch is jammed and the unit cannot be used. Please help."),
    ("product", "negative", True, True, "The chair leg snapped during normal use. I need a replacement."),
    ("product", "negative", False, True, "It never switched on. The refund is complete; no further help needed."),
    ("product", "negative", False, False, "I dislike the rough fabric. I am keeping it and need no assistance."),
    ("product", "negative", True, False, "The wrong size was sent. Please exchange it; the item itself works."),
    ("product", "positive", True, False, "Great camera, everything works. Could you explain how to enable the timer?"),
    ("product", "neutral", True, False, "The device works adequately. Where can I download the manual?"),
    ("product", "negative", True, True, "Wonderful: a brand-new kettle that leaks everywhere. Please replace it."),
    ("product", "positive", False, False, "I feared it would be faulty, but it works flawlessly."),
    ("product", "negative", True, True, "Delivery was quick, but the laptop screen is dead. Please repair it."),
    ("product", "positive", False, False, "Shipping was late, but the outstanding product makes me very happy overall."),
    ("product", "negative", True, True, "The lining came unglued immediately. Please replace this defective bag."),
    ("product", "neutral", False, False, "The product is average overall: useful features, awkward controls."),
    ("shipping", "positive", False, False, "Delivery arrived a day early with careful packaging."),
    ("shipping", "positive", False, False, "The courier followed my delivery instructions perfectly."),
    ("shipping", "neutral", False, False, "Delivery was routine, neither quick nor slow."),
    ("shipping", "negative", True, False, "My parcel is a week overdue and still missing. Please locate it."),
    ("shipping", "negative", False, False, "Delivery was very late, but it arrived intact. No follow-up needed."),
    ("shipping", "negative", False, False, "The box was crushed, but the product works. My shipping complaint is already resolved."),
    ("shipping", "neutral", True, False, "No opinion on delivery yet. Could you send me the tracking number?"),
    ("shipping", "positive", True, False, "Excellent delivery service. Can you confirm the delivery date for my second parcel?"),
    ("shipping", "negative", True, False, "Tracking says delivered, but I received nothing. Please investigate."),
    ("shipping", "negative", True, False, "Lovely, another week without my parcel. Can someone find it?"),
    ("support", "positive", False, False, "Support answered my billing question immediately and resolved it."),
    ("support", "positive", False, True, "Support replaced my broken kettle quickly. The replacement works; everything is resolved."),
    ("support", "negative", True, False, "Support has ignored my refund request for two weeks."),
    ("support", "negative", True, False, "Three agents transferred me without resolving my billing problem."),
    ("support", "neutral", False, False, "Support gave a standard answer that settled my question. Average service."),
    ("support", "neutral", True, False, "The support reply was neither good nor bad, but I still need clarification on returns."),
    ("support", "negative", False, False, "The agent was rude, although the refund is now complete. No further action needed."),
    ("support", "positive", True, False, "The support team was helpful. Could they also send my invoice copy?"),
    ("price", "positive", False, False, "Excellent value for money at this price."),
    ("price", "positive", False, False, "The discount made this a bargain."),
    ("price", "neutral", False, False, "The price is average for this market, neither cheap nor expensive."),
    ("price", "negative", False, False, "Far too expensive for what it offers. I am not requesting any action."),
    ("price", "negative", True, False, "The advertised discount was missing at checkout. Please refund the difference."),
    ("price", "neutral", True, False, "The current price seems average. Does it include tax?"),
    ("price", "positive", True, False, "Great value. Is the same discount available on a second order?"),
    ("other", "positive", False, False, "A very pleasant shopping experience overall."),
    ("other", "neutral", False, False, "An ordinary shopping experience, nothing noteworthy."),
    ("other", "negative", False, False, "A disappointing overall experience. I do not want follow-up contact."),
    ("other", "neutral", True, False, "I have no overall opinion yet. Where can I change my account language?"),
    ("other", "negative", True, False, "I cannot sign into my account and need help. This is frustrating."),
]
RATINGS = {"positive": [4, 5], "neutral": [3, 3], "negative": [1, 2]}


def build():
    reviews = []
    for index, (topic, sentiment, reply, defect, text) in enumerate(CASES, 1):
        for variant, prefix in enumerate(("", "For context, I placed this order online last month. "), 1):
            reviews.append({"id": f"rev{len(reviews)+1:03d}", "case_id": f"case{index:03d}",
                            "variant": variant, "text": prefix + text,
                            "expected": dict(topic=topic, sentiment=sentiment,
                                             rating=RATINGS[sentiment], needs_reply=reply, defect=defect)})
    random.Random(SEED).shuffle(reviews)
    return reviews


def main():
    path = Path(__file__).parent / "data" / "reviews_100.json"
    path.parent.mkdir(exist_ok=True)
    reviews = build()
    path.write_text(json.dumps(reviews, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(reviews)} reviews to {path}")


if __name__ == "__main__":
    main()
