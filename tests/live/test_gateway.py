import os

import httpx
import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_LLM_TESTS") != "1",
    reason="set RUN_LIVE_LLM_TESTS=1 to run against production",
)


def test_openai_and_native_live_smoke():
    base_url = os.environ["LLM_GATEWAY_BASE_URL"].rstrip("/")
    headers = {"Authorization": f"Bearer {os.environ['LLM_GATEWAY_API_KEY']}"}
    with httpx.Client(timeout=60) as client:
        models = client.get(f"{base_url}/v1/models", headers=headers)
        chat = client.post(f"{base_url}/v1/chat/completions", headers=headers, json={"model":"default","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":16})
        native = client.post(f"{base_url}/api/generate", headers=headers, json={"model":"default","prompt":"Reply with OK.","stream":False,"think":False,"options":{"num_predict":16}})
    assert models.status_code == chat.status_code == native.status_code == 200
