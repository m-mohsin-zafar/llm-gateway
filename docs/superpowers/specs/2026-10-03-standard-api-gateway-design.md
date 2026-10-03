# Standard API Gateway Design

**Date:** 2026-10-03

## Purpose

Turn the existing LLM gateway into a documented, authenticated FastAPI service with two public protocol surfaces:

- an Ollama-native, read/inference-only API; and
- an OpenAI-compatible API suitable for standard SDKs and agent frameworks.

The gateway must remain usable on the existing CPU-only VPS, preserve the current PostgreSQL-backed usage accounting and admin dashboard, and prevent public callers from managing models or overriding server resource ceilings.

## Success Criteria

- FastAPI serves `/docs`, `/redoc`, and `/openapi.json` with useful schemas and authentication controls.
- Standard OpenAI clients can connect with only a base URL, API key, and model name.
- Pydantic AI, LangChain, and LangGraph can perform normal chat, streaming, structured-output, and tool-call workflows supported by Ollama and the selected model.
- Native Ollama clients can use documented read and inference endpoints, including native options and NDJSON streaming.
- API keys can be created, scoped, rotated, expired, disabled, and revoked from the admin application.
- The admin application and repository contain maintained, copyable integration documentation.
- Request and response content is never persisted.

## Selected Architecture

FastAPI is an authenticated protocol gateway in front of the locally bound Ollama server. It owns authentication, authorization, model aliases, resource policy, errors, documentation, and usage accounting. It delegates protocol behavior to Ollama's corresponding native or OpenAI-compatible endpoint and preserves upstream buffered and streaming response formats.

This avoids maintaining a custom reimplementation of either protocol. Ollama's OpenAI API is explicitly a supported subset, so compatibility claims are limited to Ollama's documented features and the selected model's capabilities.

## Public API

### OpenAI-compatible endpoints

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/responses`
- `POST /v1/embeddings`
- `GET /v1/models`
- `GET /v1/models/{model}`

The surface supports the fields Ollama documents for each endpoint, including streaming, tools, structured output, log probabilities, and reasoning controls where supported. The gateway accepts both `max_tokens` and `max_completion_tokens` and performs a compatibility translation only when the installed Ollama endpoint requires it.

OpenAI streaming uses server-sent events and terminates according to the upstream protocol, including `[DONE]` where applicable. Tool-call IDs, arguments, response fields, and stream chunks are not rewritten except for configured model aliases.

The Responses API is stateless. Stateful OpenAI features such as stored responses, conversations, and `previous_response_id` are outside scope.

### Native Ollama endpoints

- `POST /api/chat`
- `POST /api/generate`
- `POST /api/embed`
- `GET /api/tags`
- `POST /api/show`
- `GET /api/ps`
- `GET /api/version`

Native endpoints preserve Ollama request and response formats, including NDJSON streaming. Inference requests support documented top-level fields such as `think`, `tools`, `format`, `keep_alive`, `logprobs`, and `top_logprobs`, plus Ollama's documented runtime `options` object.

### Blocked native operations

The gateway does not route model or blob management operations, including pull, push, create, copy, delete, and blob upload/download routes. Unknown and blocked paths are never forwarded to Ollama.

### Service and documentation endpoints

- `GET /health` reports process health without authentication.
- `GET /ready` verifies PostgreSQL and Ollama connectivity.
- `GET /docs` serves Swagger UI.
- `GET /redoc` serves ReDoc.
- `GET /openapi.json` serves the OpenAPI 3.1 schema.
- `GET /admin` serves the protected administration application.

## Authentication and API-Key Lifecycle

The preferred client authentication format is:

```http
Authorization: Bearer llmgw_<public-id>_<secret>
```

`X-API-Key` remains supported for existing clients. Supplying conflicting values in both headers is rejected.

New keys are opaque, high-entropy values. The public ID selects a database row without scanning every key; only a memory-hard hash of the secret is stored. The complete secret is returned once at creation and cannot be retrieved later.

Each key has:

- a unique name and public ID;
- a secret hash;
- one or more scopes;
- created, last-used, and optional expiry timestamps; and
- enabled and revoked state.

Scopes are:

- `openai` for `/v1` endpoints;
- `ollama:inference` for `/api/chat`, `/api/generate`, and `/api/embed`; and
- `ollama:read` for `/api/tags`, `/api/show`, `/api/ps`, and `/api/version`.

Rotation creates a new key and permits an intentional overlap period. The administrator updates consumers and then revokes the old key. Revocation, disablement, and expiry take effect on the next request. Service API keys do not use refresh tokens.

The existing bootstrap key remains usable during migration and can be revoked afterward. The migration must preserve its current hash and usage history.

## Model and Resource Policy

`default` maps to `DEFAULT_MODEL`. OpenAI and native inference calls may use the alias. Access to real downloaded model names is permitted only by explicit server policy; arbitrary cloud or unavailable model identifiers are rejected.

The gateway enforces configured context and output ceilings regardless of client input. Native options remain available, but `num_ctx` and output-token fields are clamped or rejected when they exceed server limits. OpenAI output-token fields are handled consistently with the same ceiling.

The gateway limits admitted inference work to match the single-parallel Ollama deployment. When the bounded queue is full, it returns `429` and `Retry-After`. Client disconnects cancel upstream streams where possible. Read-only metadata requests do not consume inference capacity.

## Client Compatibility

The canonical OpenAI configuration is:

```text
base_url = https://llm.ultracodes.io/v1
api_key  = llmgw_...
model    = default
```

Compatibility is verified for:

- OpenAI Python SDK: synchronous and asynchronous calls, Chat Completions, stateless Responses, streaming, tools, and structured output;
- Pydantic AI: `OpenAIChatModel`, stateless `OpenAIResponsesModel`, typed output, tools, and streaming;
- LangChain: `ChatOpenAI.invoke`, `ainvoke`, `stream`, `astream`, `bind_tools`, and structured output;
- LangGraph: model calls inside graph nodes, message streaming, and tool-call round trips;
- Ollama Python client: native chat, options, and streaming; and
- direct HTTP clients using either supported authentication header.

The gateway will not implement framework-specific endpoints or adapters. Frameworks use the standard OpenAI client contract.

## Internal Components

The application is split by responsibility:

```text
app/
├── main.py
├── config.py
├── auth.py
├── database.py
├── upstream.py
├── models/
│   ├── native.py
│   ├── openai.py
│   └── admin.py
├── routers/
│   ├── system.py
│   ├── openai.py
│   ├── ollama.py
│   └── admin.py
└── templates/
    └── admin.html
```

- `main.py` constructs the FastAPI application and owns its lifespan.
- `config.py` validates environment configuration and resource limits.
- `auth.py` parses credentials, verifies keys, and enforces scopes.
- `database.py` owns schema migration, key persistence, and usage events.
- `upstream.py` owns buffered and streaming HTTP transport to Ollama.
- `models/` contains public request, response, and admin schemas.
- `routers/` exposes protocol-specific routes.
- `templates/admin.html` is the server-rendered Tailwind administration interface.

## Request Flow and Usage Accounting

Each protected request passes through:

1. credential parsing and key lookup;
2. enabled, revoked, and expiry validation;
3. endpoint scope enforcement;
4. model alias and resource-policy enforcement;
5. bounded inference admission when applicable;
6. buffered or streaming forwarding to the corresponding Ollama endpoint; and
7. usage and latency recording.

Usage events record API key ID, protocol, endpoint, model alias, upstream model, prompt/input tokens, completion/output tokens, duration, status code, and timestamp. They do not record messages, prompts, generated text, tool arguments, embeddings, or images.

For streaming responses, the gateway observes chunks while forwarding them and records final usage when supplied by Ollama. A disconnected or failed stream records the status and available counters without delaying delivery.

## Error Contract

- Missing, invalid, expired, disabled, or revoked credentials return `401`.
- Valid keys without the required scope return `403`.
- Invalid request bodies and unsupported fields return `422` or the upstream protocol's documented `400` response.
- Queue saturation returns `429` with `Retry-After`.
- Ollama connection failures return `503`; invalid upstream responses return `502`.
- Timeouts return `504`.

OpenAI routes return errors under an OpenAI-style `error` object. Native routes return Ollama-style error bodies. Upstream status codes and safe error details are preserved where doing so does not expose internal addresses or secrets. Every response includes a gateway request ID.

## Admin Application

The existing Tailwind dashboard becomes a tabbed administration portal:

### Overview

- gateway, database, and Ollama status;
- current default model and configured limits;
- request, token, error, and latency metrics; and
- links to Swagger, ReDoc, and OpenAPI JSON.

### API keys

- create a key with name, scopes, and optional expiry;
- display its secret exactly once;
- list metadata without secrets;
- rotate, enable, disable, and revoke keys; and
- show last use and aggregate usage.

Mutating admin requests require HTTP Basic authentication plus a same-origin custom-header CSRF guard. CORS is not enabled. Administrative credentials and generated secrets are never logged.

### Integration guide

The portal contains maintained, copyable examples for curl, OpenAI Python and JavaScript SDKs, Pydantic AI, LangChain, LangGraph, the Ollama Python client, native HTTP, streaming, tool calls, structured output, and thinking control. Examples use placeholders and never insert an existing secret.

### Reference and operations

The portal documents endpoints, authentication, scopes, model aliases, configured limits, timeouts, concurrency behavior, errors, key rotation, and the supported OpenAI subset. It links to relevant upstream documentation.

The repository README carries the same core integration examples so documentation changes are versioned with the API.

## Testing Strategy

Fast unit and application tests use a fake upstream transport and isolated database fixtures. They cover:

- Bearer and `X-API-Key` parsing, conflict handling, and authentication failures;
- key creation, one-time display, expiry, rotation, disablement, and revocation;
- scope enforcement for every public route;
- bootstrap-key migration and historical usage preservation;
- model aliasing and resource ceilings;
- OpenAI and native buffered responses;
- SSE and NDJSON streaming, cancellation, and final usage capture;
- tool-call and structured-output passthrough;
- OpenAI and native error formats;
- queue saturation and timeout behavior;
- dashboard integration documentation and secret redaction; and
- OpenAPI route schemas and Bearer security definitions.

An opt-in live compatibility suite runs against the deployed Qwen model. It exercises OpenAI SDK, Pydantic AI, LangChain, LangGraph, and Ollama clients. Live tests use short prompts and low output limits so they remain practical on the CPU-only host. Deployment is complete only after the normal suite and live compatibility suite pass.

## Deployment and Migration

The gateway remains containerized with Docker Compose and uses host networking to reach the localhost-only Ollama and shared PostgreSQL services. Nginx and Cloudflare continue to terminate and route public HTTPS traffic.

Deployment order:

1. back up the gateway database schema and current environment file;
2. apply additive database migrations;
3. build and start the new container;
4. verify health, readiness, docs, admin login, and existing-key compatibility;
5. run protocol and framework live checks through the public hostname; and
6. create scoped replacement keys and migrate consumers before optionally revoking the bootstrap key.

The existing `/v1/chat/completions`, `/v1/models`, `/health`, and `/admin` paths remain available. Existing `X-API-Key` clients continue to authenticate during migration.

## Explicit Non-Goals

- Exposing model-management or blob routes.
- Implementing OpenAI features that Ollama does not support.
- Stateful OpenAI Responses, conversations, or stored completions.
- OAuth user login or refresh-token flows.
- Persisting prompts, responses, tool payloads, images, or embeddings.
- Increasing Ollama parallelism as part of this work.
