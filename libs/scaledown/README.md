# langchain-scaledown

ScaleDown middleware for LangChain agents. `langchain-scaledown` plugs
[ScaleDown](https://scaledown.ai)'s task-specific models into
`langchain.agents.create_agent()` as agent middleware: one class summarizes long
conversation history, one compresses large retrieved context relative to the
user's question, and one extracts structured fields (like a support ticket)
from the conversation as it happens. All three fail open: if ScaleDown is
unavailable, the agent runs as if the middleware weren't there.

## Installation

```bash
pip install -U langchain-scaledown
```

Set your API key (get one at <https://scaledown.ai/dashboard>):

```bash
export SCALEDOWN_API_KEY="your-api-key"
# optional, defaults to https://api.scaledown.xyz
export SCALEDOWN_BASE_URL="https://api.scaledown.xyz"
```

Requires Python 3.10+ and `langchain` 1.3+. The middleware works with
`create_agent()` and with [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)
(`create_deep_agent(middleware=[...])`).

## `ScaledownSummarizationMiddleware`

A drop-in alternative to LangChain's built-in
[`SummarizationMiddleware`](https://docs.langchain.com/oss/python/langchain/middleware/built-in#summarization):
same `trigger` / `keep` options, same behavior, but the summary comes from
ScaleDown instead of a second LLM call, so there's no summarization model to
configure or pay for.

Once the conversation crosses `trigger`, every message except the most recent
ones selected by `keep` is summarized by ScaleDown and replaced in the agent's
state by a single summary message. Each stretch of history is summarized once,
and the conversation continues from the summary. Kept messages are untouched,
and tool results are never separated from the call that requested them.

> **Note:** `sd_summarize` is in private preview. Contact ScaleDown to enable it
> on your account.

```python
from langchain.agents import create_agent
from langchain_scaledown import ScaledownSummarizationMiddleware

agent = create_agent(
    "openai:gpt-4.1",
    tools=[...],
    middleware=[
        ScaledownSummarizationMiddleware(
            trigger=("tokens", 4000),   # or ("messages", 50), or a list of both
            keep=("messages", 20),      # or ("tokens", 2000)
        )
    ],
)
```

The default token counter is LangChain's character-based
`count_tokens_approximately`; pass `token_counter=` to use your own. You can also
pass `instructions=` and `max_tokens=` through to ScaleDown.

Fractional sizes (`("fraction", 0.5)`) aren't supported, because they need a
summarization model's context window. Use a token count instead.

## `ScaledownCompressionMiddleware`

Query-aware compression of large context. LangChain has no built-in equivalent.
Before each model call, the system message is compressed relative to the last
user message, and the model sees the compressed version.

> **Use it for needle-in-a-haystack, RAG-style workloads**: a large block of
> retrieved documents in the system prompt where the model needs a few relevant
> facts. **Don't use it for short prompts or creative generation**, where
> compression can drop instructions, tone, or detail the model needs.

```python
from langchain.agents import create_agent
from langchain_scaledown import ScaledownCompressionMiddleware

agent = create_agent(
    "openai:gpt-4.1",
    system_prompt=f"Answer using these documents:\n\n{retrieved_docs}",
    middleware=[ScaledownCompressionMiddleware(min_context_chars=2000)],
)

agent.invoke({"messages": [{"role": "user", "content": "What is the refund window?"}]})
```

Compression is skipped (request passed through unchanged) when the context is
at most `min_context_chars` long (default 2000), when there's no system message
or user message, when ScaleDown reports `successful: false`, or when the API call
fails.

> **With Deep Agents:** a deep agent's system message also holds the harness's
> own instructions (planning, file tools), and compression would condense those
> too. Use compression there only with care, or keep documents out of the system
> prompt.

To choose what gets compressed, pass a `context_extractor` returning
`(context, prompt)` or `None` to skip. The compressed context always replaces
the system message:

```python
def extract(request):
    if request.system_message is None:
        return None
    return request.system_message.text, request.messages[-1].text

ScaledownCompressionMiddleware(context_extractor=extract, rate="auto")
```

## `ScaledownExtractionMiddleware`

Turns a conversation into a structured record as it happens. Once per agent
invocation, before the model runs, the user's messages are sent to ScaleDown's
`/extract` endpoint with your schema. The result lands in agent state, so your
tools can read it mid-run and you get it back from `agent.invoke()`.

Two kinds of fields go in one schema, and ScaleDown handles both in a single call:

- **Extractive fields** (a description string): the value is copied verbatim
  from what the user said, such as an order number, an error message, or an email.
- **Classification keys** (an object with `labels`): ScaleDown routes these to
  its classification model, which picks one of your labels. Use them for
  decisions such as sentiment or issue type, where there's nothing to copy.

By default only the user's messages are extracted from, never the agent's own
replies, so every extractive value is something the customer actually said.

```python
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_scaledown import ScaledownExtractionMiddleware

ticket = ScaledownExtractionMiddleware(
    entities={
        # Extractive: quoted from the customer's messages
        "order_id": "Order number the customer mentions",
        "error_message": "Exact error message or code the customer saw",
        "steps_tried": "What the customer says they already tried",
        "customer_email": "Customer's email address",
        # Classification keys: routed to ScaleDown's classify model
        "issue_type": {
            "labels": [
                {"name": "billing", "rubric": "Is this about a charge, refund, or payment?"},
                {"name": "technical", "rubric": "Is this about a bug or something not working?"},
                {"name": "account", "rubric": "Is this about login, password, or account settings?"},
            ]
        },
        "sentiment": {"labels": ["frustrated", "neutral", "satisfied"]},
    },
    context_chars=0,          # skip evidence context to keep state small
    inject_into_prompt=True,  # let the agent see the fields (see note below)
)


@tool
def file_ticket(runtime: ToolRuntime) -> str:
    """File a support ticket with the details gathered so far."""
    # .get(): the key is absent if extraction hasn't succeeded yet
    fields = runtime.state.get("scaledown_extraction", {}).get("fields", {})
    return crm.create_ticket(**fields)  # your ticketing system


agent = create_agent("openai:gpt-4.1", tools=[file_ticket], middleware=[ticket])

result = agent.invoke({"messages": [{"role": "user", "content": "Order A-1042 failed ..."}]})
result["scaledown_extraction"]["fields"]
# {"order_id": "A-1042", "error_message": "ERR_PAYMENT_DECLINED", "steps_tried": None,
#  "customer_email": None, "issue_type": "billing", "sentiment": "frustrated"}
```

`fields` has one entry per key in your schema: the highest-confidence value, or
`None` if nothing was found. `entities` holds ScaleDown's raw matches with
confidence scores. Nested and array schemas work too; their values come from
ScaleDown's `structured_result`.

With a checkpointer, extraction re-runs every turn over the whole conversation,
so the record fills in as the customer adds detail (turn 1 has the order number;
turn 2 adds their email).

> **About `inject_into_prompt`** (off by default): it appends the extracted fields
> to the system message so the agent can act on them, for example escalating a
> frustrated customer. Extractive values are customer text, so enabling this puts
> customer text into the system prompt. Only turn it on when that's acceptable,
> and have the agent branch on classification keys, whose values can only be
> one of your labels.

Pass `text_builder=` to extract from something other than the user's messages.

## Development

The package lives in `libs/scaledown` and uses [uv](https://docs.astral.sh/uv/):

```bash
cd libs/scaledown
uv sync --all-groups
make test                                 # unit tests (no network)
make lint                                 # ruff + mypy
SCALEDOWN_API_KEY=... make integration_tests   # tests against the live ScaleDown API
uv run python scripts/smoke_test.py       # runs every middleware in a real agent (mocked API)
```
