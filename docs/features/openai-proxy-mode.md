# OpenAI Proxy Mode

Set `CCPROXY_PROFILE=openai` to route Anthropic `POST /v1/messages` requests through an OpenAI-compatible backend.

## How it works

1. The proxy receives an Anthropic Messages request.
2. `services/translator.py` maps it to an OpenAI `/chat/completions` request (`to_openai_request`).
3. The request is forwarded to `OPENAI_BASE_URL/chat/completions`.
4. The OpenAI response is mapped back to an Anthropic `MessagesResponse` (`from_openai_response`).
5. The client receives a standard Anthropic response — no client changes needed.

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `CCPROXY_PROFILE` | yes | `anthropic` | Set to `openai` to enable this mode |
| `OPENAI_BASE_URL` | yes | — | Base URL of the OpenAI-compatible API |
| `OPENAI_API_KEY` | yes | — | API key for the OpenAI-compatible API |
| `OPENAI_MODEL` | no | `gpt-4o` | Model name sent in every upstream request |

## M1 limitations

- **No streaming:** Always issues a blocking `POST /chat/completions` regardless of client `stream` flag. SSE streaming is planned for M2.
- **Token counting:** `POST /v1/messages/count_tokens` returns a heuristic estimate (`total_chars / 4`), not an exact count.
