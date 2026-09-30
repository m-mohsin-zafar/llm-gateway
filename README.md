# llm-gateway

An API-key protected, OpenAI-compatible gateway for a private Ollama deployment.

## API

`POST /v1/chat/completions` uses `X-API-Key` and currently exposes `default`, mapped to `qwen3:4b`.

`GET /admin` requires HTTP Basic authentication and shows API-key usage for the last 24 hours.

## Deployment

1. Run `sudo ./scripts/provision.sh` once to create an isolated PostgreSQL database and `/opt/llm-gateway/.env`.
2. Run `sudo docker compose up -d --build`.
3. Install the Nginx virtual host and obtain the TLS certificate.

Secrets are held only in `.env`, which is ignored by Git.
