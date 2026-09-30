# llm-gateway

An API-key protected, OpenAI-compatible gateway for a private Ollama deployment.

## API

`POST /v1/chat/completions` uses `X-API-Key` and currently exposes `default`, mapped to `qwen3:4b`.

Generation controls:

```json
{
  "model": "default",
  "messages": [{"role": "user", "content": "Hello"}],
  "think": false,
  "temperature": 0.2,
  "max_tokens": 512,
  "options": {"top_p": 0.9, "seed": 7}
}
```

`think` defaults to `false`. The optional `options` object supports `top_k`, `top_p`, `min_p`, `typical_p`, `repeat_last_n`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`, `seed`, and `stop`. Context size and output limits remain gateway-controlled; use `max_tokens` rather than `options.num_predict`.

`GET /admin` requires HTTP Basic authentication and shows API-key usage for the last 24 hours.

## Deployment

1. Run `sudo ./scripts/provision.sh` once to create an isolated PostgreSQL database and `/opt/llm-gateway/.env`.
2. Run `sudo docker compose up -d --build`.
3. Install the Nginx virtual host and obtain the TLS certificate.

Secrets are held only in `.env`, which is ignored by Git.
