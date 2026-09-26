"""Unit tests for ScaleDown middleware.

`langchain_tests` has no standard test base for middleware, so these are
hand-rolled against a fake `ModelRequest`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_scaledown import (
    ScaledownAPIError,
    ScaledownClient,
    ScaledownCompressionMiddleware,
    ScaledownSummarizationMiddleware,
)


def make_request(
    messages: list[AnyMessage], system_message: SystemMessage | None = None
) -> MagicMock:
    """Minimal fake ModelRequest whose .override() returns a new fake."""
    request = MagicMock(name="ModelRequest")
    request.messages = messages
    request.system_message = system_message
    request.system_prompt = system_message.text if system_message else None

    def override(**kw: Any) -> MagicMock:
        return make_request(
            kw.get("messages", request.messages),
            kw.get("system_message", request.system_message),
        )

    request.override = MagicMock(side_effect=override)
    return request


@pytest.fixture
def client() -> ScaledownClient:
    return ScaledownClient(api_key="test-key")


def conversation(n: int) -> list[AnyMessage]:
    msgs: list[AnyMessage] = []
    for i in range(n):
        cls = HumanMessage if i % 2 == 0 else AIMessage
        msgs.append(cls(content=f"message {i}"))
    return msgs


# --------------------------------------------------------------------------
# ScaledownSummarizationMiddleware
# --------------------------------------------------------------------------


class TestSummarization:
    def test_below_trigger_passes_through(self, client: ScaledownClient) -> None:
        mw = ScaledownSummarizationMiddleware(
            trigger=("messages", 10), keep=("messages", 2), client=client
        )
        request = make_request(conversation(5))
        handler = MagicMock(return_value="response")
        with patch.object(ScaledownClient, "summarize") as summarize:
            result = mw.wrap_model_call(request, handler)
        summarize.assert_not_called()
        request.override.assert_not_called()
        handler.assert_called_once_with(request)
        assert result == "response"

    def test_message_trigger_summarizes(self, client: ScaledownClient) -> None:
        mw = ScaledownSummarizationMiddleware(
            trigger=("messages", 6),
            keep=("messages", 2),
            instructions="be brief",
            max_tokens=100,
            client=client,
        )
        messages = conversation(6)
        request = make_request(messages)
        handler = MagicMock(return_value="response")
        with patch.object(
            ScaledownClient, "summarize", return_value="the summary"
        ) as summarize:
            mw.wrap_model_call(request, handler)

        summarize.assert_called_once()
        transcript = summarize.call_args.args[0]
        assert "message 0" in transcript and "message 3" in transcript
        assert "message 4" not in transcript
        assert summarize.call_args.kwargs == {
            "instructions": "be brief",
            "max_tokens": 100,
        }

        request.override.assert_called_once()
        new_messages = request.override.call_args.kwargs["messages"]
        assert len(new_messages) == 3
        assert isinstance(new_messages[0], SystemMessage)
        assert "the summary" in new_messages[0].text
        # Kept messages are the same objects, untouched.
        assert new_messages[1] is messages[4]
        assert new_messages[2] is messages[5]

        sent = handler.call_args.args[0]
        assert sent is not request
        assert sent.messages == new_messages
        # Original request not mutated.
        assert request.messages is messages

    def test_token_trigger_uses_custom_counter(self, client: ScaledownClient) -> None:
        counter = MagicMock(return_value=10_000)
        mw = ScaledownSummarizationMiddleware(
            trigger=("tokens", 500),
            keep=("messages", 1),
            token_counter=counter,
            client=client,
        )
        request = make_request(conversation(3))
        handler = MagicMock()
        with patch.object(ScaledownClient, "summarize", return_value="s") as summ:
            mw.wrap_model_call(request, handler)
        counter.assert_called_once_with(request.messages)
        summ.assert_called_once()
        assert len(handler.call_args.args[0].messages) == 2

    def test_default_token_counter_below_threshold(
        self, client: ScaledownClient
    ) -> None:
        mw = ScaledownSummarizationMiddleware(
            trigger=("tokens", 10_000), keep=("messages", 1), client=client
        )
        request = make_request(conversation(4))
        handler = MagicMock()
        with patch.object(ScaledownClient, "summarize") as summarize:
            mw.wrap_model_call(request, handler)
        summarize.assert_not_called()
        handler.assert_called_once_with(request)

    def test_does_not_orphan_tool_messages(self, client: ScaledownClient) -> None:
        mw = ScaledownSummarizationMiddleware(
            trigger=("messages", 3), keep=("messages", 1), client=client
        )
        ai = AIMessage(
            content="", tool_calls=[{"name": "t", "args": {}, "id": "call_1"}]
        )
        tool = ToolMessage(content="result", tool_call_id="call_1")
        request = make_request([HumanMessage(content="q"), ai, tool])
        handler = MagicMock()
        with patch.object(ScaledownClient, "summarize", return_value="s"):
            mw.wrap_model_call(request, handler)
        new_messages = handler.call_args.args[0].messages
        assert new_messages[1:] == [ai, tool]

    def test_api_error_fails_open(self, client: ScaledownClient) -> None:
        mw = ScaledownSummarizationMiddleware(
            trigger=("messages", 2), keep=("messages", 1), client=client
        )
        request = make_request(conversation(4))
        handler = MagicMock(return_value="response")
        with patch.object(
            ScaledownClient, "summarize", side_effect=ScaledownAPIError("boom")
        ):
            result = mw.wrap_model_call(request, handler)
        request.override.assert_not_called()
        handler.assert_called_once_with(request)
        assert result == "response"

    async def test_async_summarizes(self, client: ScaledownClient) -> None:
        mw = ScaledownSummarizationMiddleware(
            trigger=("messages", 4), keep=("messages", 1), client=client
        )
        request = make_request(conversation(4))

        async def handler(req: Any) -> Any:
            return req

        with patch.object(ScaledownClient, "summarize", return_value="s") as summ:
            sent = await mw.awrap_model_call(request, handler)
        summ.assert_called_once()
        assert len(sent.messages) == 2

    def test_invalid_config(self, client: ScaledownClient) -> None:
        with pytest.raises(ValueError):
            ScaledownSummarizationMiddleware(trigger=("fraction", 1), client=client)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            ScaledownSummarizationMiddleware(keep=("tokens", 5), client=client)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# ScaledownCompressionMiddleware
# --------------------------------------------------------------------------

BIG_CONTEXT = "Relevant fact: the answer is 42. " + ("filler text. " * 300)


class TestCompression:
    def test_short_context_passes_through(self, client: ScaledownClient) -> None:
        mw = ScaledownCompressionMiddleware(min_context_chars=2000, client=client)
        request = make_request(
            [HumanMessage(content="what is the answer?")],
            SystemMessage(content="short context"),
        )
        handler = MagicMock(return_value="response")
        with patch.object(ScaledownClient, "compress") as compress:
            result = mw.wrap_model_call(request, handler)
        compress.assert_not_called()
        request.override.assert_not_called()
        handler.assert_called_once_with(request)
        assert result == "response"

    def test_missing_system_or_human_passes_through(
        self, client: ScaledownClient
    ) -> None:
        mw = ScaledownCompressionMiddleware(min_context_chars=10, client=client)
        handler = MagicMock()
        no_system = make_request([HumanMessage(content="q")])
        no_human = make_request(
            [AIMessage(content="hi")], SystemMessage(content=BIG_CONTEXT)
        )
        with patch.object(ScaledownClient, "compress") as compress:
            mw.wrap_model_call(no_system, handler)
            mw.wrap_model_call(no_human, handler)
        compress.assert_not_called()
        assert [c.args[0] for c in handler.call_args_list] == [no_system, no_human]

    def test_large_context_compresses(self, client: ScaledownClient) -> None:
        mw = ScaledownCompressionMiddleware(
            min_context_chars=2000, rate=0.5, client=client
        )
        request = make_request(
            [
                HumanMessage(content="old question"),
                AIMessage(content="old answer"),
                HumanMessage(content="what is the answer?"),
            ],
            SystemMessage(content=BIG_CONTEXT),
        )
        handler = MagicMock()
        with patch.object(
            ScaledownClient,
            "compress",
            return_value={"successful": True, "compressed_prompt": "answer is 42"},
        ) as compress:
            mw.wrap_model_call(request, handler)

        compress.assert_called_once_with(BIG_CONTEXT, "what is the answer?", rate=0.5)
        request.override.assert_called_once()
        new_system = request.override.call_args.kwargs["system_message"]
        assert isinstance(new_system, SystemMessage)
        assert new_system.text == "answer is 42"
        sent = handler.call_args.args[0]
        assert sent is not request
        assert sent.system_message is new_system
        assert sent.messages == request.messages
        assert request.system_message.text == BIG_CONTEXT

    def test_unsuccessful_response_passes_through(
        self, client: ScaledownClient
    ) -> None:
        mw = ScaledownCompressionMiddleware(client=client)
        request = make_request(
            [HumanMessage(content="q")], SystemMessage(content=BIG_CONTEXT)
        )
        handler = MagicMock()
        with patch.object(
            ScaledownClient,
            "compress",
            return_value={"successful": False, "compressed_prompt": "garbage"},
        ) as compress:
            mw.wrap_model_call(request, handler)
        compress.assert_called_once()
        request.override.assert_not_called()
        handler.assert_called_once_with(request)

    def test_custom_context_extractor(self, client: ScaledownClient) -> None:
        extractor = MagicMock(return_value=("x" * 50, "custom prompt"))
        mw = ScaledownCompressionMiddleware(
            min_context_chars=10, context_extractor=extractor, client=client
        )
        request = make_request([HumanMessage(content="q")])
        handler = MagicMock()
        with patch.object(
            ScaledownClient,
            "compress",
            return_value={"successful": True, "compressed_prompt": "c"},
        ) as compress:
            mw.wrap_model_call(request, handler)
        extractor.assert_called_once_with(request)
        compress.assert_called_once_with("x" * 50, "custom prompt", rate="auto")
        assert handler.call_args.args[0].system_message.text == "c"

    def test_api_error_fails_open(self, client: ScaledownClient) -> None:
        mw = ScaledownCompressionMiddleware(client=client)
        request = make_request(
            [HumanMessage(content="q")], SystemMessage(content=BIG_CONTEXT)
        )
        handler = MagicMock(return_value="response")
        with patch.object(
            ScaledownClient, "compress", side_effect=ScaledownAPIError("boom")
        ):
            result = mw.wrap_model_call(request, handler)
        request.override.assert_not_called()
        handler.assert_called_once_with(request)
        assert result == "response"

    async def test_async_api_error_fails_open(self, client: ScaledownClient) -> None:
        mw = ScaledownCompressionMiddleware(client=client)
        request = make_request(
            [HumanMessage(content="q")], SystemMessage(content=BIG_CONTEXT)
        )

        async def handler(req: Any) -> Any:
            return req

        with patch.object(
            ScaledownClient, "compress", side_effect=ScaledownAPIError("boom")
        ):
            sent = await mw.awrap_model_call(request, handler)
        assert sent is request


# --------------------------------------------------------------------------
# ScaledownClient
# --------------------------------------------------------------------------


class TestClient:
    def test_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SCALEDOWN_API_KEY", raising=False)
        with pytest.raises(ValueError, match="scaledown.ai/dashboard"):
            ScaledownClient()

    def test_env_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCALEDOWN_API_KEY", "env-key")
        monkeypatch.setenv("SCALEDOWN_BASE_URL", "https://example.test/")
        c = ScaledownClient()
        assert c.api_key == "env-key"
        assert c.base_url == "https://example.test"

    def test_summarize_request_shape(self, client: ScaledownClient) -> None:
        with patch("langchain_scaledown._client.requests.post") as post:
            post.return_value.json.return_value = {"summary": "short"}
            assert client.summarize("long text", max_tokens=10) == "short"
        url = post.call_args.args[0]
        kw = post.call_args.kwargs
        assert url == "https://api.scaledown.xyz/summarization/abstractive"
        assert kw["headers"]["x-api-key"] == "test-key"
        assert "Authorization" not in kw["headers"]
        assert kw["json"] == {"text": "long text", "max_tokens": 10}

    def test_compress_request_shape(self, client: ScaledownClient) -> None:
        with patch("langchain_scaledown._client.requests.post") as post:
            post.return_value.json.return_value = {"successful": True}
            client.compress("ctx", "prompt")
        assert post.call_args.args[0] == "https://api.scaledown.xyz/compress/raw/"
        assert post.call_args.kwargs["json"] == {
            "context": "ctx",
            "prompt": "prompt",
            "scaledown": {"rate": "auto"},
        }

    def test_request_failure_wrapped(self, client: ScaledownClient) -> None:
        import requests

        with patch(
            "langchain_scaledown._client.requests.post",
            side_effect=requests.ConnectionError("down"),
        ):
            with pytest.raises(ScaledownAPIError):
                client.compress("ctx", "prompt")
