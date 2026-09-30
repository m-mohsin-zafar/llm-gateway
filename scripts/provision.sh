#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="$project_dir/.env"
db_container="shared-postgres"
db_user="llm_gateway"
db_name="llm_gateway"

if [ -e "$env_file" ]; then
  echo "Refusing to replace existing $env_file" >&2
  exit 1
fi

admin_user="$(docker inspect "$db_container" --format '{{range .Config.Env}}{{println .}}{{end}}' | awk -F= '$1 == "POSTGRES_USER" {print $2}')"
admin_password="$(docker inspect "$db_container" --format "{{range .Config.Env}}{{println .}}{{end}}" | sed -n "s/^POSTGRES_PASSWORD=//p")"
db_password="$(openssl rand -hex 32)"
api_key="llmgw_$(openssl rand -hex 32)"
admin_password_web="$(openssl rand -base64 32 | tr -d '\n')"

PGPASSWORD="$admin_password" docker exec -e PGPASSWORD="$admin_password" "$db_container" psql -v ON_ERROR_STOP=1 -U "$admin_user" -d postgres \
  -v db_password="$db_password" -v db_user="$db_user" -v db_name="$db_name" <<'SQL'
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'db_user', :'db_password') WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'db_user') \gexec
SELECT format('CREATE DATABASE %I OWNER %I', :'db_name', :'db_user') WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = :'db_name') \gexec
SQL

umask 077
cat > "$env_file" <<EOF
DATABASE_URL=postgresql://$db_user:$db_password@127.0.0.1:5433/$db_name
OLLAMA_URL=http://127.0.0.1:11434
DEFAULT_MODEL=qwen3:4b
BOOTSTRAP_API_KEY=$api_key
ADMIN_USERNAME=admin
ADMIN_PASSWORD=$admin_password_web
REQUEST_TIMEOUT_SECONDS=180
MAX_CONTEXT_TOKENS=8192
MAX_OUTPUT_TOKENS=1024
EOF
echo "Created $env_file and the isolated $db_name database. Secrets were not printed."
