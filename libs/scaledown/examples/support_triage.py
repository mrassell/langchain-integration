"""Support-ticket triage with ScaledownExtractionMiddleware.

ScaleDown reads each customer message and fills in a structured ticket (order ID,
error, email, issue type, sentiment). The agent only talks to the customer and
decides when to file; the ticket fields come straight from what the customer
said, never retyped by the LLM.

    pip install langchain-scaledown langchain-anthropic
    export SCALEDOWN_API_KEY=...  ANTHROPIC_API_KEY=...
    python examples/support_triage.py

Any LangChain chat model works, e.g. `--model openai:gpt-4.1`
(with `pip install langchain-openai` and OPENAI_API_KEY).

Voicemail mode transcribes a recording locally with Whisper, then files it:

    pip install faster-whisper "av<19"  # PyAV 19 dropped an option it uses
    python examples/support_triage.py --audio voicemail.aiff
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_scaledown import (
    ScaledownAPIError,
    ScaledownClient,
    ScaledownExtractionMiddleware,
)

TICKET_SCHEMA: dict[str, Any] = {
    "order_id": "Order number the customer mentions",
    "error_message": "Exact error message or code the customer saw",
    "steps_tried": "What the customer says they already tried",
    "customer_email": "Customer's email address",
    "issue_type": {
        "labels": [
            {
                "name": "billing",
                "rubric": "Is this about a charge, refund, or payment?",
            },
            {
                "name": "technical",
                "rubric": "Is this about a bug, crash, or something not working?",
            },
            {
                "name": "account",
                "rubric": "Is this about login, password, or account settings?",
            },
            {
                "name": "shipping",
                "rubric": "Is this about delivery, tracking, or a missing package?",
            },
        ]
    },
    "sentiment": {"labels": ["frustrated", "neutral", "satisfied"]},
}

TEAMS = {
    "billing": "Billing",
    "technical": "Tech Support",
    "account": "Accounts",
    "shipping": "Fulfillment",
}

SYSTEM_PROMPT = """You are a customer support agent for an online store.
To file a ticket you need the customer's order number and email address.
If either is missing from the extracted fields, ask the customer for it.
Once you have both, call file_ticket with a one-sentence summary, then tell the
customer their ticket number and which team has it. Keep replies short.
Reply in plain text, no Markdown."""

VOICEMAIL_PROMPT = """You are a customer support agent for an online store.
The message is a transcribed voicemail, so the customer can't answer questions.
Call file_ticket with a one-sentence summary, noting anything missing (order
number, email). Then write a short callback note for the team.
Reply in plain text, no Markdown."""

TICKETS: list[dict[str, Any]] = []


def route(fields: dict[str, Any]) -> tuple[str, str]:
    """Pick team and priority from the classification fields only."""
    team = TEAMS.get(fields.get("issue_type") or "", "General Support")
    priority = "high" if fields.get("sentiment") == "frustrated" else "normal"
    return team, priority


@tool
def file_ticket(summary: str, runtime: ToolRuntime) -> str:
    """File a support ticket. Order ID, email, and the rest are filled in for you."""
    fields = runtime.state.get("scaledown_extraction", {}).get("fields", {})
    team, priority = route(fields)
    ticket = {
        "id": f"T-{1000 + len(TICKETS) + 1}",
        "summary": summary,
        "team": team,
        "priority": priority,
        **fields,
    }
    TICKETS.append(ticket)
    return f"Filed {ticket['id']} with {team} ({priority} priority)."


def build_agent(model: BaseChatModel | str, system_prompt: str = SYSTEM_PROMPT) -> Any:
    return create_agent(
        model=model,
        tools=[file_ticket],
        system_prompt=system_prompt,
        middleware=[
            ScaledownExtractionMiddleware(
                TICKET_SCHEMA, context_chars=0, inject_into_prompt=True
            )
        ],
        checkpointer=InMemorySaver(),
    )


BATCH = [
    "Hi, order A-1042 got charged twice on my card. Can you refund one? "
    "priya.n@example.com",
    "The app crashes with ERR_SYNC_TIMEOUT every time I open my cart. I already "
    "reinstalled it twice. This is ridiculous. Order B-2207, marco@example.com",
    "I can't log in, it keeps saying INVALID_2FA_CODE even though I just reset "
    "my password. Order C-3310, sam.lee@example.com",
    "Package for order D-4481 says delivered but I can't find it. No stress, "
    "probably a neighbor grabbed it. Thanks so much for checking! ana.r@example.com",
    "Is order E-5092 shipping this week? Just planning ahead. jo@example.com",
]

CHAT = [
    "My checkout keeps failing with ERR_PAYMENT_DECLINED on order A-1042. "
    "I've tried two different cards. Third time I'm writing in, I'm fed up.",
    "It's jane@example.com",
]


def fields_of(result: dict[str, Any]) -> dict[str, Any]:
    return dict(result.get("scaledown_extraction", {}).get("fields", {}))


def cell(value: Any, width: int) -> str:
    text = "-" if value in (None, "") else str(value)
    return (text[: width - 1] + "…" if len(text) > width else text).ljust(width)


def run_batch(agent: Any) -> None:
    print("=== Part 1: batch triage ===\n")
    cols = [
        ("#", 3),
        ("order_id", 9),
        ("error_message", 18),
        ("issue_type", 10),
        ("sentiment", 10),
        ("team", 14),
        ("priority", 8),
    ]
    print("  ".join(cell(name, w) for name, w in cols))
    print("  ".join("-" * w for _, w in cols))
    for i, message in enumerate(BATCH, 1):
        filed_before = len(TICKETS)
        result = agent.invoke(
            {"messages": [HumanMessage(message)]},
            {"configurable": {"thread_id": f"batch-{i}"}},
        )
        fields = fields_of(result)
        if len(TICKETS) > filed_before:
            team, priority = TICKETS[-1]["team"], TICKETS[-1]["priority"]
        else:
            team, priority = "(not filed)", ""
        row = [
            i,
            fields.get("order_id"),
            fields.get("error_message"),
            fields.get("issue_type"),
            fields.get("sentiment"),
            team,
            priority,
        ]
        print("  ".join(cell(v, w) for v, (_, w) in zip(row, cols)))
    print()


def run_chat(agent: Any) -> None:
    print("=== Part 2: one conversation, record fills in turn by turn ===")
    config = {"configurable": {"thread_id": "chat"}}
    previous: dict[str, Any] = {}
    for turn, message in enumerate(CHAT, 1):
        filed_before = len(TICKETS)
        result = agent.invoke({"messages": [HumanMessage(message)]}, config)
        fields = fields_of(result)
        print(f"\n--- Turn {turn} ---")
        print(f"Customer: {message}")
        print("ScaleDown extracted:")
        for key, value in fields.items():
            new = value is not None and previous.get(key) != value
            print(f"  {'+' if new else ' '} {key}: {value}")
        reply = result["messages"][-1]
        if isinstance(reply, AIMessage):
            print(f"Agent: {reply.text}")
        for ticket in TICKETS[filed_before:]:
            print_ticket(ticket)
        previous = fields
    print()


def print_ticket(ticket: dict[str, Any]) -> None:
    print(
        f"Ticket filed: {ticket['id']} -> {ticket['team']} "
        f"({ticket['priority']} priority): {ticket['summary']}"
    )


def transcribe(path: str) -> str:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        sys.exit("Setup error: voicemail needs `pip install faster-whisper 'av<19'`")
    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(path)
    return " ".join(segment.text.strip() for segment in segments)


def run_voicemail(agent: Any, transcript: str) -> None:
    print("=== Voicemail -> ticket ===\n")
    print(f"Transcript: {transcript}\n")
    filed_before = len(TICKETS)
    result = agent.invoke(
        {"messages": [HumanMessage(transcript)]},
        {"configurable": {"thread_id": "voicemail"}},
    )
    print("ScaleDown extracted:")
    for key, value in fields_of(result).items():
        print(f"    {key}: {value}")
    reply = result["messages"][-1]
    if isinstance(reply, AIMessage):
        print(f"\nAgent: {reply.text}\n")
    for ticket in TICKETS[filed_before:]:
        print_ticket(ticket)
    print()


def check_scaledown() -> None:
    """Fail fast on a bad key instead of running with extraction silently off."""
    try:
        ScaledownClient().extract(
            "Order A-1042", {"order_id": "Order number"}, context_chars=0
        )
    except ScaledownAPIError as e:
        sys.exit(
            "ScaleDown rejected SCALEDOWN_API_KEY. New keys can take a few "
            f"minutes to activate. ({e})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        default="anthropic:claude-sonnet-5",
        help="Any init_chat_model string (default: %(default)s)",
    )
    parser.add_argument(
        "--audio",
        metavar="PATH",
        help="Transcribe this voicemail and file it as a ticket (needs faster-whisper)",
    )
    args = parser.parse_args()
    if args.audio and not Path(args.audio).is_file():
        sys.exit(f"No such audio file: {args.audio}")
    try:
        check_scaledown()
        model = init_chat_model(args.model)
    except (ImportError, ValueError) as e:
        sys.exit(f"Setup error: {e}")
    if args.audio:
        run_voicemail(build_agent(model, VOICEMAIL_PROMPT), transcribe(args.audio))
        return
    agent = build_agent(model)
    run_batch(agent)
    run_chat(agent)


if __name__ == "__main__":
    main()
