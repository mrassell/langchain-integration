# langchain-scaledown

ScaleDown middleware for LangChain agents. `langchain-scaledown` plugs
[ScaleDown](https://scaledown.ai)'s context-reduction API into
`langchain.agents.create_agent()` as agent middleware: one class summarizes long
conversation history, the other compresses large retrieved context relative to
the user's question. Both fail open: if ScaleDown is unavailable, the model is
called with the original request.

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

Requires Python 3.10+ and `langchain` 1.x.

## `ScaledownSummarizationMiddleware`

A drop-in alternative to LangChain's built-in `SummarizationMiddleware`, with
the same `trigger` / `keep` interface. Once the conversation crosses `trigger`,
every message except the most recent `keep` is summarized by ScaleDown and
replaced with a single `SystemMessage` holding the summary. The kept messages
are passed through untouched.

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
            trigger=("tokens", 4000),   # or ("messages", 50)
            keep=("messages", 20),
        )
    ],
)
```

The default token counter is LangChain's character-based
`count_tokens_approximately`; pass `token_counter=` to use your own. You can also
pass `instructions=` and `max_tokens=` through to ScaleDown.

The summary is applied to the request sent to the model; the agent's stored
state keeps the full history.

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

## Development

```bash
poetry install --with test,lint
make test                             # unit tests
make lint
poetry run python scripts/smoke_test.py   # runs both middlewares in a real agent (mocked API)
```
