"""Smoke test: run ScaleDown middleware inside a real `create_agent()` agent.

Not part of the test suite. Uses FakeListChatModel and mocks the ScaleDown
client methods, so no API key or network is needed:

    poetry run python scripts/smoke_test.py
"""

import asyncio
import os
from unittest.mock import patch

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from langchain_scaledown import (
    ScaledownClient,
    ScaledownCompressionMiddleware,
    ScaledownSummarizationMiddleware,
)

os.environ.setdefault("SCALEDOWN_API_KEY", "dummy-key-for-smoke-test")


class Spy(AgentMiddleware):
    """Innermost middleware: records the request the model actually receives."""

    def __init__(self) -> None:
        super().__init__()
        self.seen = []

    def wrap_model_call(self, request, handler):
        self.seen.append(request)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        self.seen.append(request)
        return await handler(request)


def run_summarization(use_async: bool) -> None:
    spy = Spy()
    agent = create_agent(
        model=FakeListChatModel(responses=["final answer"]),
        middleware=[
            ScaledownSummarizationMiddleware(
                trigger=("messages", 5), keep=("messages", 2)
            ),
            spy,
        ],
    )
    history = []
    for i in range(3):
        history += [HumanMessage(f"question {i}"), AIMessage(f"answer {i}")]
    history.append(HumanMessage("latest question"))

    with patch.object(
        ScaledownClient, "summarize", return_value="MOCK SUMMARY"
    ) as summarize:
        inputs = {"messages": history}
        result = (
            asyncio.run(agent.ainvoke(inputs)) if use_async else agent.invoke(inputs)
        )

    sent = spy.seen[-1].messages
    print(f"[summarization {'async' if use_async else 'sync'}]")
    print(f"  sd_summarize called: {summarize.call_count}x")
    print(f"  model saw {len(sent)} messages (state had {len(history)}):")
    for m in sent:
        print(f"    {type(m).__name__}: {m.text[:60]!r}")
    print(f"  final reply: {result['messages'][-1].text!r}")
    assert summarize.call_count == 1
    assert len(sent) == 3 and "MOCK SUMMARY" in sent[0].text
    assert result["messages"][-1].text == "final answer"


def run_compression(use_async: bool) -> None:
    spy = Spy()
    big_context = "The launch code is 7-4-1. " + "Irrelevant filler. " * 300
    agent = create_agent(
        model=FakeListChatModel(responses=["the code is 7-4-1"]),
        system_prompt=big_context,
        middleware=[ScaledownCompressionMiddleware(min_context_chars=2000), spy],
    )
    with patch.object(
        ScaledownClient,
        "compress",
        return_value={"successful": True, "compressed_prompt": "launch code: 7-4-1"},
    ) as compress:
        inputs = {"messages": [HumanMessage("What is the launch code?")]}
        result = (
            asyncio.run(agent.ainvoke(inputs)) if use_async else agent.invoke(inputs)
        )

    system = spy.seen[-1].system_message
    print(f"[compression {'async' if use_async else 'sync'}]")
    print(f"  sd_compress called: {compress.call_count}x")
    print(f"  compress prompt arg: {compress.call_args.args[1]!r}")
    print(f"  system prompt: {len(big_context)} chars -> {len(system.text)} chars")
    print(f"  model saw system message: {system.text!r}")
    print(f"  final reply: {result['messages'][-1].text!r}")
    assert compress.call_count == 1
    assert compress.call_args.args == (big_context, "What is the launch code?")
    assert system.text == "launch code: 7-4-1"


if __name__ == "__main__":
    for use_async in (False, True):
        run_summarization(use_async)
        run_compression(use_async)
    print("\nSMOKE TEST PASSED")
