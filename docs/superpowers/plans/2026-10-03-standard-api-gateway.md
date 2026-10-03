# Standard API Gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the custom single-endpoint gateway with a documented, scoped FastAPI gateway that preserves Ollama-native protocols and Ollama's OpenAI-compatible protocols for standard client libraries.

**Architecture:** Split the FastAPI application into configuration, persistence, authentication, upstream transport, protocol routers, and a server-rendered admin portal. Public protocol handlers validate credentials and resource policy, then proxy buffered or streaming traffic to the matching localhost Ollama endpoint while recording content-free usage events.

**Tech Stack:** Python 3.12, FastAPI 0.128, Pydantic 2, HTTPX 0.28, asyncpg 0.30, Uvicorn, PostgreSQL 16, Jinja2, pytest, Docker Compose, Ollama 0.35.

**Spec:** `docs/superpowers/specs/2026-10-03-standard-api-gateway-design.md`

## Global Constraints

- Preserve `/v1/chat/completions`, `/v1/models`, `/health`, `/admin`, and existing `X-API-Key` authentication during migration.
- Prefer `Authorization: Bearer`; reject conflicting Bearer and `X-API-Key` values.
- Expose only the native endpoints listed in the spec; never implement a wildcard `/api/{path}` proxy.
- Never persist request content, output content, tool payloads, images, or embeddings.
- Keep `default` mapped to `DEFAULT_MODEL`; reject unapproved model identifiers.
- Enforce `MAX_CONTEXT_TOKENS`, `MAX_OUTPUT_TOKENS`, single active inference, and a bounded queue.
- Preserve upstream SSE and NDJSON wire formats and cancel upstream work after client disconnects.
- Keep secrets in `.env`; never commit or log keys, database credentials, or admin credentials.
- Keep Ollama bound to localhost and retain the existing Nginx/Cloudflare public path.
- Use additive PostgreSQL migrations that preserve the bootstrap key and all existing usage events.

## Review Focus

- A request containing different Bearer and `X-API-Key` credentials must return `401`, never choose one silently; pinned in Task 2.
- A valid key with only `ollama:read` must not invoke inference through aliases, alternate verbs, or OpenAI routes; pinned in Tasks 4 and 5.
- A stream disconnected before its usage trailer must release its inference slot and record a failed/incomplete event; pinned in Task 3.
- Client-supplied output/context aliases (`num_ctx`, `num_predict`, `max_tokens`, `max_completion_tokens`, `max_output_tokens`) must not bypass server ceilings; pinned in Tasks 4 and 5.
- Admin HTML and integration snippets must never contain a previously generated API secret; pinned in Task 6.

---

### Task 1: Application foundation, settings, and test harness

**Files:**
- Create: `app/config.py`
- Create: `app/dependencies.py`
- Create: `tests/conftest.py`
- Create: `tests/test_system.py`
- Modify: `app/main.py`
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`

**Interfaces:**
- Produces: `Settings.from_env() -> Settings`; `GatewayServices`; `create_app(settings: Settings, services: GatewayServices | None = None) -> FastAPI`.
- Produces: test fixtures `settings`, `fake_database`, `fake_upstream`, `app`, and `client` used by all later tasks.

- [ ] **Step 1: Add failing system and OpenAPI tests**

Add tests asserting `GET /health` returns `{"status": "ok"}`, `/docs`, `/redoc`, and `/openapi.json` are enabled, and `/ready` reports separate database and Ollama status. Add a settings test asserting invalid/non-positive limits fail at startup.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest -q tests/test_system.py`

Expected: FAIL because `create_app`, validated settings, readiness, and documentation are absent.

- [ ] **Step 3: Implement the application factory and validated settings**

Implement `Settings`, `GatewayServices`, and `create_app`. Keep module-level `app = create_app(Settings.from_env())` for Uvicorn. Enable Swagger, ReDoc, and OpenAPI; add tagged route metadata. Add Jinja2 to runtime requirements and `pytest`, `pytest-asyncio`, `asgi-lifespan`, and `beautifulsoup4` to `requirements-dev.txt`.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest -q tests/test_system.py && pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/main.py app/config.py app/dependencies.py requirements.txt requirements-dev.txt tests
git commit -m "refactor: establish gateway application foundation"
```

### Task 2: PostgreSQL migrations and scoped API-key lifecycle

**Files:**
- Create: `app/database.py`
- Create: `app/auth.py`
- Create: `app/models/admin.py`
- Create: `tests/test_auth.py`
- Create: `tests/test_key_repository.py`
- Modify: `app/dependencies.py`
- Modify: `app/main.py`

**Interfaces:**
- Consumes: `Settings`, `GatewayServices`, and app fixtures from Task 1.
- Produces: `ApiKeyPrincipal`; `parse_api_key(authorization: str | None, x_api_key: str | None) -> str`; `require_scope(scope: str)`; `ApiKeyRepository.create/list/rotate/set_enabled/revoke/authenticate`; `UsageRepository.record`.

- [ ] **Step 1: Add failing authentication and repository tests**

Cover Bearer and legacy headers, matching dual headers, conflicting dual headers, malformed tokens, unknown keys, disabled/revoked/expired keys, and missing scopes. Assert OpenAPI declares HTTP Bearer and `X-API-Key` schemes. Repository tests must assert a new `llmgw_<public-id>_<secret>` is returned once, only its scrypt hash is stored, lookup uses the public ID, rotation creates a distinct key, revocation is immediate, and the existing bootstrap row and usage history survive migration.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest -q tests/test_auth.py tests/test_key_repository.py`

Expected: FAIL because scoped key models, migration, and authentication dependencies do not exist.

- [ ] **Step 3: Implement additive migrations and key services**

Add key public ID, scopes, expiry, revoked timestamp, and rotation metadata without replacing existing rows. Extend usage events with protocol and endpoint columns using safe defaults. Preserve legacy hash verification as a migration fallback; all new keys use indexed public-ID lookup. Implement constant-time secret comparison and dependencies that return `ApiKeyPrincipal`.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest -q tests/test_auth.py tests/test_key_repository.py && pytest -q`

Expected: all tests pass and no secret appears in captured logs.

- [ ] **Step 5: Commit**

```bash
git add app/auth.py app/database.py app/dependencies.py app/main.py app/models/admin.py tests/test_auth.py tests/test_key_repository.py
git commit -m "feat: add scoped API key lifecycle"
```

### Task 3: Upstream transport, inference admission, and protocol errors

**Files:**
- Create: `app/upstream.py`
- Create: `app/errors.py`
- Create: `tests/test_upstream.py`
- Create: `tests/test_errors.py`
- Modify: `app/dependencies.py`
- Modify: `app/main.py`

**Interfaces:**
- Consumes: `Settings`, `GatewayServices`, and `UsageRepository.record`.
- Produces: `UpstreamClient.request_json(method, path, payload=None) -> UpstreamResponse`; `UpstreamClient.stream(method, path, payload) -> AsyncIterator[bytes]`; `InferenceAdmission.acquire()` async context manager; `openai_error(...)`; `ollama_error(...)`.

- [ ] **Step 1: Add failing buffered, streaming, and admission tests**

Use `httpx.MockTransport` to prove headers and internal URLs are not leaked, safe upstream statuses are preserved, connect errors map to 503, invalid upstream payloads to 502, and timeouts to 504. Assert SSE and NDJSON bytes remain byte-for-byte unchanged, disconnect closes the upstream response and releases admission, queue overflow returns 429 with `Retry-After`, and incomplete streams record a failed event.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest -q tests/test_upstream.py tests/test_errors.py`

Expected: FAIL because transport, admission, cancellation, and protocol error helpers do not exist.

- [ ] **Step 3: Implement transport and bounded inference admission**

Use one lifespan-managed `httpx.AsyncClient`. Implement explicit-path buffered and streaming calls, an admission controller with one active request plus `OLLAMA_MAX_QUEUE` waiters, protocol-aware exception handlers, `X-Request-ID`, and usage finalization that never blocks response forwarding.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest -q tests/test_upstream.py tests/test_errors.py && pytest -q`

Expected: all tests pass with no leaked async resources.

- [ ] **Step 5: Commit**

```bash
git add app/upstream.py app/errors.py app/dependencies.py app/main.py tests/test_upstream.py tests/test_errors.py
git commit -m "feat: add resilient Ollama transport"
```

### Task 4: Read-only and inference-only native Ollama API

**Files:**
- Create: `app/models/native.py`
- Create: `app/routers/ollama.py`
- Create: `tests/test_ollama_api.py`
- Modify: `app/main.py`

**Interfaces:**
- Consumes: `require_scope`, `UpstreamClient`, `InferenceAdmission`, `Settings`, and `UsageRepository`.
- Produces: FastAPI router exposing only `/api/chat`, `/api/generate`, `/api/embed`, `/api/tags`, `/api/show`, `/api/ps`, and `/api/version`.

- [ ] **Step 1: Add failing native route tests**

Parametrize every route for method, required scope, and upstream path. Test buffered and NDJSON chat/generate responses; documented top-level fields and arbitrary documented `options`; tools, `format`, `think`, `keep_alive`, and logprobs passthrough; default-model mapping; rejection of unapproved models; and clamps/rejections for `num_ctx` and `num_predict`. Assert `/api/pull`, `/api/create`, `/api/copy`, `/api/delete`, `/api/push`, and arbitrary `/api/*` paths return 404 without calling upstream. Include the review-focus test proving `ollama:read` cannot invoke inference.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest -q tests/test_ollama_api.py`

Expected: FAIL because native models and routes do not exist.

- [ ] **Step 3: Implement explicit native schemas and routes**

Define typed top-level request fields with `extra="allow"` for forward-compatible documented additions and a typed/free-form `options` map. Register each route explicitly, use native error bodies, preserve content types and stream bytes, and record native usage without content.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest -q tests/test_ollama_api.py && pytest -q`

Expected: all tests pass; blocked paths produce zero upstream calls.

- [ ] **Step 5: Commit**

```bash
git add app/models/native.py app/routers/ollama.py app/main.py tests/test_ollama_api.py
git commit -m "feat: expose scoped native Ollama API"
```

### Task 5: OpenAI-compatible endpoints and SDK semantics

**Files:**
- Create: `app/models/openai.py`
- Create: `app/routers/openai.py`
- Create: `tests/test_openai_api.py`
- Modify: `app/main.py`

**Interfaces:**
- Consumes: `require_scope("openai")`, `UpstreamClient`, `InferenceAdmission`, `Settings`, and `UsageRepository`.
- Produces: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/models`, and `/v1/models/{model}`.

- [ ] **Step 1: Add failing OpenAI contract tests**

Cover model listing/retrieval, chat, legacy completions, stateless Responses, embeddings, SSE passthrough and `[DONE]`, `stream_options.include_usage`, tools and tool-choice payloads, structured response formats, image content, logprobs, reasoning settings, and default-model mapping. Prove `max_tokens` and `max_completion_tokens` converge on the same output ceiling, `max_output_tokens` is bounded for Responses, stateful Responses fields are rejected, and non-`openai` scopes cannot call any `/v1` route. Assert all local validation/auth/upstream failures have OpenAI-style `error` objects.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest -q tests/test_openai_api.py`

Expected: FAIL because the standards-oriented routes and schemas do not exist.

- [ ] **Step 3: Implement the OpenAI protocol router**

Delegate supported payloads to Ollama's matching `/v1` endpoints. Allow documented forward fields, reject explicitly unsupported stateful features, normalize only model aliases and output-token compatibility, preserve SSE bytes, and derive usage from buffered responses or streaming usage trailers.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest -q tests/test_openai_api.py && pytest -q`

Expected: all tests pass and the old custom response construction is removed.

- [ ] **Step 5: Commit**

```bash
git add app/models/openai.py app/routers/openai.py app/main.py tests/test_openai_api.py
git commit -m "feat: add OpenAI compatible API surface"
```

### Task 6: Admin key management and maintained integration portal

**Files:**
- Create: `app/routers/admin.py`
- Create: `app/templates/admin.html`
- Create: `tests/test_admin.py`
- Modify: `app/main.py`
- Modify: `app/database.py`

**Interfaces:**
- Consumes: `ApiKeyRepository`, `UsageRepository`, `Settings`, readiness services, and existing HTTP Basic credentials.
- Produces: dashboard tabs; `POST /admin/api-keys`; `POST /admin/api-keys/{public_id}/rotate`; `PATCH /admin/api-keys/{public_id}`; `POST /admin/api-keys/{public_id}/revoke`.

- [ ] **Step 1: Add failing admin security and content tests**

Assert Basic auth is required, mutating requests require the same-origin custom CSRF header, create returns a secret exactly once, list/HTML never return stored secrets, and rotation/revoke/enable/disable update immediately. Parse HTML to assert Overview, API Keys, Integration Guide, and Reference tabs; links to docs; and copyable snippets for curl, OpenAI Python/JavaScript, Pydantic AI, LangChain, LangGraph, Ollama Python, streaming, tools, structured output, and `think: false`. Seed a recognizable old secret and prove it never appears in HTML or JSON list responses.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest -q tests/test_admin.py`

Expected: FAIL because key-management routes, CSRF guard, template, and integration content do not exist.

- [ ] **Step 3: Implement the server-rendered Tailwind portal**

Move inline HTML to Jinja2, retain the current visual language and refresh behavior, add tab navigation and key forms, and serve same-origin JavaScript only for interactions. Render placeholders rather than live keys in snippets. Keep admin mutations out of the public OpenAPI tags or mark them clearly as administrative.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest -q tests/test_admin.py && pytest -q`

Expected: all tests pass and no seeded secret appears in captured output.

- [ ] **Step 5: Commit**

```bash
git add app/routers/admin.py app/templates/admin.html app/main.py app/database.py tests/test_admin.py
git commit -m "feat: add API key administration portal"
```

### Task 7: Framework compatibility, documentation, and deployment verification

**Files:**
- Create: `tests/live/test_openai_sdk.py`
- Create: `tests/live/test_pydantic_ai.py`
- Create: `tests/live/test_langchain.py`
- Create: `tests/live/test_langgraph.py`
- Create: `tests/live/test_ollama_client.py`
- Create: `tests/live/README.md`
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `compose.yml`
- Modify: `deployment/llm.ultracodes.io.conf`

**Interfaces:**
- Consumes: the complete public contract from Tasks 1-6.
- Produces: opt-in `RUN_LIVE_LLM_TESTS=1` compatibility suite and operator/client documentation.

- [ ] **Step 1: Add live compatibility tests and prove they skip safely by default**

Implement low-token tests for OpenAI sync/async chat and streaming; Pydantic AI typed output and one tool call; LangChain invoke/stream/tool binding; a LangGraph message/tool round trip; and Ollama native chat/options/NDJSON streaming. Tests read only `LLM_GATEWAY_BASE_URL` and `LLM_GATEWAY_API_KEY`, use `model="default"`, and skip unless `RUN_LIVE_LLM_TESTS=1`.

Run: `pytest -q tests/live`

Expected: all tests SKIPPED when the opt-in variable is absent.

- [ ] **Step 2: Update versioned usage and operations documentation**

Document both base URLs, authentication headers, scopes, key rotation, every endpoint, SDK/framework examples, streaming, tools, structured output, thinking controls, limits, errors, timeouts, concurrency, live-test setup, deployment, and rollback. Add queue configuration to `.env.example`, test dependency installation instructions, proxy buffering-off rules for streaming, and Nginx timeout values consistent with the gateway.

- [ ] **Step 3: Run static and normal-suite verification**

Run: `python -m compileall -q app tests && pytest -q && docker compose config -q && git diff --check`

Expected: exit 0; normal tests pass; live tests skip; Compose and whitespace checks pass.

- [ ] **Step 4: Rebuild and run public smoke tests**

Back up the database schema and `.env`, then run `docker compose up -d --build`. Verify `/health`, `/ready`, `/docs`, `/openapi.json`, `/admin`, one buffered and one streaming OpenAI call, one buffered and one streaming native call, blocked model management, and legacy `X-API-Key` authentication through `https://llm.ultracodes.io`.

Expected: all allowed checks succeed, blocked paths return 404, and no request content is present in PostgreSQL.

- [ ] **Step 5: Run the opt-in framework suite against production**

Run with secrets supplied only through environment variables: `RUN_LIVE_LLM_TESTS=1 pytest -q tests/live`.

Expected: all OpenAI SDK, Pydantic AI, LangChain, LangGraph, and Ollama client tests pass.

- [ ] **Step 6: Commit**

```bash
git add README.md .env.example compose.yml deployment/llm.ultracodes.io.conf tests/live
git commit -m "docs: add client compatibility and deployment guide"
```
