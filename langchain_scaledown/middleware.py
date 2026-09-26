"""ScaleDown middleware for LangChain agents."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, Literal

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    get_buffer_string,
)
from langchain_core.messages.utils import count_tokens_approximately

from langchain_scaledown._client import ScaledownAPIError, ScaledownClient

logger = logging.getLogger(__name__)

TokenCounter = Callable[[Iterable[AnyMessage]], int]
ContextExtractor = Callable[[ModelRequest], "tuple[str, str] | None"]
Trigger = tuple[Literal["tokens", "messages"], int]
Keep = tuple[Literal["messages"], int]

_SUMMARY_PREFIX = "Summary of the earlier conversation:\n\n"


def _resolve_client(
    client: ScaledownClient | None, api_key: str | None, base_url: str | None
) -> ScaledownClient:
    if client is not None:
        return client
    return ScaledownClient(api_key=api_key, base_url=base_url)


class ScaledownSummarizationMiddleware(AgentMiddleware):
    """Summarize older conversation history with ScaleDown's `sd_summarize`.

    Interface-compatible with
    `langchain.agents.middleware.SummarizationMiddleware`: once the conversation
    crosses `trigger`, every message except the most recent `keep` messages is
    sent to ScaleDown's abstractive summarization endpoint and replaced with a
    single `SystemMessage` holding the summary. The kept messages are passed
    through untouched.

    The rewrite is applied to the model request only (via
    `ModelRequest.override`); the agent's stored state keeps the full history.

    If the ScaleDown call fails, the middleware fails open and the model is
    called with the original, unsummarized request.

    Note:
        `sd_summarize` is in private preview. Contact ScaleDown to enable it on
        your account.

    Args:
        trigger: When to summarize: `("tokens", N)` or `("messages", N)`.
        keep: How many recent messages to keep verbatim: `("messages", N)`.
        token_counter: Callable counting tokens in a list of messages. Defaults
            to LangChain's character-based `count_tokens_approximately`.
        instructions: Optional instructions forwarded to ScaleDown.
        max_tokens: Maximum length of the generated summary.
        client: Preconfigured `ScaledownClient`. If omitted, one is created from
            `api_key` / `base_url` (or the `SCALEDOWN_API_KEY` /
            `SCALEDOWN_BASE_URL` environment variables).

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain_scaledown import ScaledownSummarizationMiddleware

        agent = create_agent(
            "openai:gpt-4.1",
            tools=[...],
            middleware=[
                ScaledownSummarizationMiddleware(
                    trigger=("tokens", 4000),
                    keep=("messages", 20),
                )
            ],
        )
        ```
    """

    def __init__(
        self,
        *,
        trigger: Trigger = ("tokens", 4000),
        keep: Keep = ("messages", 20),
        token_counter: TokenCounter = count_tokens_approximately,
        instructions: str | None = None,
        max_tokens: int = 2048,
        client: ScaledownClient | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__()
        kind, value = trigger
        if kind not in ("tokens", "messages") or value <= 0:
            raise ValueError(
                f"trigger must be ('tokens', N) or ('messages', N) with N > 0, "
                f"got {trigger!r}"
            )
        keep_kind, keep_value = keep
        if keep_kind != "messages" or keep_value < 0:
            raise ValueError(f"keep must be ('messages', N) with N >= 0, got {keep!r}")
        self.trigger = trigger
        self.keep = keep
        self.token_counter = token_counter
        self.instructions = instructions
        self.max_tokens = max_tokens
        self.client = _resolve_client(client, api_key, base_url)

    def _should_summarize(self, messages: list[AnyMessage]) -> bool:
        kind, threshold = self.trigger
        if kind == "messages":
            return len(messages) >= threshold
        return self.token_counter(messages) >= threshold

    def _split(
        self, messages: list[AnyMessage]
    ) -> tuple[list[AnyMessage], list[AnyMessage]] | None:
        """Split into (to_summarize, to_keep), or None if nothing to summarize."""
        cutoff = max(len(messages) - self.keep[1], 0)
        # Don't let the kept tail start with tool results whose requesting
        # AIMessage was summarized away; pull the cutoff back to include it.
        while 0 < cutoff < len(messages) and isinstance(messages[cutoff], ToolMessage):
            cutoff -= 1
        if cutoff == 0:
            return None
        return messages[:cutoff], messages[cutoff:]

    def _apply(
        self, request: ModelRequest, summary: str, n_summarized: int
    ) -> ModelRequest:
        summary_message = SystemMessage(content=_SUMMARY_PREFIX + summary)
        kept = request.messages[n_summarized:]
        return request.override(messages=[summary_message, *kept])

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Summarize older history before the model call, if triggered."""
        if not self._should_summarize(request.messages):
            return handler(request)
        split = self._split(request.messages)
        if split is None:
            return handler(request)
        to_summarize, _ = split
        try:
            summary = self.client.summarize(
                get_buffer_string(to_summarize),
                instructions=self.instructions,
                max_tokens=self.max_tokens,
            )
        except ScaledownAPIError:
            logger.warning(
                "ScaleDown summarization failed; passing request through.",
                exc_info=True,
            )
            return handler(request)
        return handler(self._apply(request, summary, len(to_summarize)))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version of `wrap_model_call`."""
        if not self._should_summarize(request.messages):
            return await handler(request)
        split = self._split(request.messages)
        if split is None:
            return await handler(request)
        to_summarize, _ = split
        try:
            summary = await asyncio.to_thread(
                self.client.summarize,
                get_buffer_string(to_summarize),
                instructions=self.instructions,
                max_tokens=self.max_tokens,
            )
        except ScaledownAPIError:
            logger.warning(
                "ScaleDown summarization failed; passing request through.",
                exc_info=True,
            )
            return await handler(request)
        return await handler(self._apply(request, summary, len(to_summarize)))


def default_context_extractor(request: ModelRequest) -> tuple[str, str] | None:
    """Use the system message as context and the last human message as prompt."""
    if request.system_message is None:
        return None
    context = request.system_message.text
    prompt = next(
        (m.text for m in reversed(request.messages) if isinstance(m, HumanMessage)),
        None,
    )
    if not context or not prompt:
        return None
    return context, prompt


class ScaledownCompressionMiddleware(AgentMiddleware):
    """Compress large retrieved context with ScaleDown's `sd_compress`.

    This is the real differentiator of this package: LangChain has no built-in
    middleware that does query-aware prompt compression. Before each model
    call, the context (by default, the system message) is compressed relative
    to the user's question, and the system message is replaced with the
    compressed version via `ModelRequest.override(system_message=...)`.

    Warning:
        Only use this for needle-in-a-haystack, RAG-style workloads, where a
        large block of retrieved documents is stuffed into the system prompt
        and the model needs a few relevant facts from it. Do not use it for
        short prompts or creative generation: compression can drop
        instructions, tone, or detail the model needs.

    Compression is skipped (the request passes through unchanged) when:

    - the context extractor returns `None`,
    - the context is not longer than `min_context_chars`,
    - ScaleDown reports `successful: False`, or
    - the ScaleDown call raises `ScaledownAPIError` (fail open).

    Args:
        min_context_chars: Only compress contexts longer than this.
        rate: Compression rate passed to ScaleDown (`"auto"` by default).
        context_extractor: `(ModelRequest) -> (context, prompt) | None`. The
            compressed context always replaces `request.system_message`, so a
            custom extractor should return the system message's content (or
            the part of it you want compressed) as `context`. Defaults to the
            system message as context and the last `HumanMessage` as prompt.
        client: Preconfigured `ScaledownClient`. If omitted, one is created from
            `api_key` / `base_url` (or the `SCALEDOWN_API_KEY` /
            `SCALEDOWN_BASE_URL` environment variables).

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain_scaledown import ScaledownCompressionMiddleware

        agent = create_agent(
            "openai:gpt-4.1",
            system_prompt=f"Answer using these documents:\\n\\n{retrieved_docs}",
            middleware=[ScaledownCompressionMiddleware(min_context_chars=2000)],
        )
        ```
    """

    def __init__(
        self,
        *,
        min_context_chars: int = 2000,
        rate: Any = "auto",
        context_extractor: ContextExtractor | None = None,
        client: ScaledownClient | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__()
        self.min_context_chars = min_context_chars
        self.rate = rate
        self.context_extractor = context_extractor or default_context_extractor
        self.client = _resolve_client(client, api_key, base_url)

    def _extract(self, request: ModelRequest) -> tuple[str, str] | None:
        extracted = self.context_extractor(request)
        if extracted is None:
            return None
        context, prompt = extracted
        if not context or not prompt or len(context) <= self.min_context_chars:
            return None
        return context, prompt

    def _apply(self, request: ModelRequest, response: dict[str, Any]) -> ModelRequest:
        compressed = response.get("compressed_prompt")
        if not response.get("successful") or not isinstance(compressed, str):
            logger.warning(
                "ScaleDown compression unsuccessful; passing request through."
            )
            return request
        return request.override(system_message=SystemMessage(content=compressed))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Compress the context before the model call, if large enough."""
        extracted = self._extract(request)
        if extracted is None:
            return handler(request)
        context, prompt = extracted
        try:
            response = self.client.compress(context, prompt, rate=self.rate)
        except ScaledownAPIError:
            logger.warning(
                "ScaleDown compression failed; passing request through.",
                exc_info=True,
            )
            return handler(request)
        return handler(self._apply(request, response))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version of `wrap_model_call`."""
        extracted = self._extract(request)
        if extracted is None:
            return await handler(request)
        context, prompt = extracted
        try:
            response = await asyncio.to_thread(
                self.client.compress, context, prompt, rate=self.rate
            )
        except ScaledownAPIError:
            logger.warning(
                "ScaleDown compression failed; passing request through.",
                exc_info=True,
            )
            return await handler(request)
        return await handler(self._apply(request, response))
