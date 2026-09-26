"""Check the ScaleDown endpoints this package uses against the live API.

Not part of the test suite. Needs a real key:

    SCALEDOWN_API_KEY=... poetry run python scripts/live_check.py

Summarization is in private preview, so it may fail on keys without access.
"""

import sys

from langchain_scaledown import ScaledownAPIError, ScaledownClient

client = ScaledownClient()
failures = 0


def check(name, fn):
    global failures
    try:
        print(f"[ok]   {name}: {fn()}")
    except (ScaledownAPIError, KeyError, AssertionError) as e:
        failures += 1
        print(f"[FAIL] {name}: {e!r}")


def extract():
    response = client.extract(
        "Order A-1042 failed with ERR_PAYMENT_DECLINED. Third time, I'm fed up. "
        "Reach me at jane@example.com.",
        {
            "order_id": "Order number the customer mentions",
            "error_message": "Exact error message or code the customer saw",
            "customer_email": "Customer's email address",
            "issue_type": {"labels": ["billing", "technical", "account"]},
            "sentiment": {"labels": ["frustrated", "neutral", "satisfied"]},
        },
        context_chars=0,
    )
    assert isinstance(response.get("entities"), list), response
    return {e["type"]: e["text"] for e in response["entities"]}


def compress():
    response = client.compress(
        "ScaleDown is a context engineering platform. " * 50,
        "What is ScaleDown?",
    )
    assert response.get("successful") is True, response
    assert isinstance(response.get("compressed_prompt"), str), response
    return (
        f"{response.get('original_prompt_tokens')} -> "
        f"{response.get('compressed_prompt_tokens')} tokens"
    )


def summarize():
    return client.summarize("The customer asked about a refund. " * 20)[:80]


check("extract", extract)
check("compress", compress)
check("summarize (private preview)", summarize)
sys.exit(1 if failures else 0)
