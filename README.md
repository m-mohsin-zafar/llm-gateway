# llm-gateway

Private, authenticated access to the local Ollama model at `llm.ultracodes.io`.

## Client contract

Use `Authorization: Bearer YOUR_API_KEY` (recommended) or the legacy `X-API-Key` header. Keys are created and rotated at `/admin` and have scopes: `openai`, `ollama:inference`, and `ollama:read`.

| Surface | Base URL | Model |
|---|---|---|
| OpenAI-compatible | `https://llm.ultracodes.io/v1` | `default` |
| Native Ollama | `https://llm.ultracodes.io` | `default` or `qwen3:4b` |

OpenAI routes: `/chat/completions`, `/completions`, `/responses` (stateless only), `/embeddings`, and `/models`. Native routes: `/api/chat`, `/api/generate`, `/api/embed`, `/api/tags`, `/api/show`, `/api/ps`, and `/api/version`. Model management paths are intentionally unavailable.

```bash
curl https://llm.ultracodes.io/v1/chat/completions \
  -H 'Authorization: Bearer YOUR_API_KEY' -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello"}],"max_tokens":256}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="https://llm.ultracodes.io/v1", api_key="YOUR_API_KEY")
print(client.chat.completions.create(model="default", messages=[{"role":"user","content":"Hello"}]).choices[0].message.content)
```

For native thinking control use `{"think": false}`. Native `options` are forwarded, while `num_ctx` and `num_predict` are capped at the configured limits. OpenAI `max_tokens`, `max_completion_tokens`, and Responses `max_output_tokens` are capped at `MAX_OUTPUT_TOKENS`.

Streaming is supported: use `stream: true`; OpenAI returns SSE and native Ollama returns NDJSON. Tools, structured outputs, logprobs, image content, and documented forward-compatible fields are passed through when Ollama/model support them. Pydantic AI, LangChain, and LangGraph use their normal OpenAI-compatible configuration with the OpenAI base URL above.

## Operations

Default limits: 180-second upstream/proxy timeout, one active inference request plus `OLLAMA_MAX_QUEUE` queued requests. A full queue returns 429 with `Retry-After`; unavailable Ollama returns 503; timeout returns 504. Every response has `X-Request-ID`. Usage records metadata and tokens only—never prompts, messages, generated text, tools, or embeddings.

Install test tools with `pip install -r requirements.txt -r requirements-dev.txt`. Run unit tests with `pytest -q`. Live tests are opt-in:

```bash
RUN_LIVE_LLM_TESTS=1 LLM_GATEWAY_BASE_URL=https://llm.ultracodes.io \
LLM_GATEWAY_API_KEY=YOUR_API_KEY pytest -q tests/live
```

Deployment uses `docker compose up -d --build`. Back up `.env` and the PostgreSQL schema first; rollback with the prior image/commit then `docker compose up -d`. Nginx must disable buffering for streaming and use timeouts at least as long as `REQUEST_TIMEOUT_SECONDS`.
