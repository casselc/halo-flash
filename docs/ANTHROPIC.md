# Anthropic Messages API (stream G2, 2026-10-07; working files `/data/halo-work/streams/G2/`)

`halo_serve` answers `POST /v1/messages` and `POST /v1/messages/count_tokens` in the Anthropic wire format
(API version 2023-06-01 as the `anthropic` 1.12 Python SDK and Claude Code 2.1.293 use it), so Claude Code and the
Anthropic SDKs run against the local Qwen3.8-Flash-Next. Gap G2 of the agent-platform report.

```
halo_serve --model <shard-00001.gguf> --max-context 65536 --api-key KEY [--anthropic-alias PATTERN=TARGET ...]
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=KEY claude -p "..." --model claude-sonnet-5-5
```

Code: `src/server/anthropic_request.*` (body -> template messages), `anthropic_response.*` (blocks, events,
usage), `anthropic_service.*` (routes, streaming, departures). Requests render through the same chat template,
tool-call parser, StopMatcher, session cache and observer as the OpenAI routes.

## Coverage

**Supported**

| Feature | Behaviour |
|---|---|
| `model` | the served id, or any name the alias table maps (default `claude-*`); the response's `model` is the served id (as the API reports a resolved alias); unknown names are a 404 `not_found_error` |
| `system` | string or text blocks, joined with blank lines |
| `messages` user content | string or blocks: `text`, `tool_result` (string or text blocks; `is_error` prefixes `[error] `) |
| `messages` assistant content | `text`, `tool_use` (input -> template tool-call arguments), `thinking` (-> `reasoning_content`) |
| `tools` | custom tools (`name`, `description`, `input_schema`; `cache_control`, `strict`, `defer_loading`, `eager_input_streaming`, `type: "custom"` accepted) |
| `tool_choice` | `auto`; `none` (tools stay in the prompt; a call the model writes anyway is returned as text); `any` / `tool` (the prompt opens `<tool_call>\n<function=[NAME>\n]`, thinking off, as the API requires); `disable_parallel_tool_use` (generation stops after the first call) |
| `max_tokens` (required), `temperature` (0..1), `top_p`, `top_k`, `stop_sequences`, `stream` | as specified; unset sampling keeps the server's defaults (the model's recommended 0.6 / 20 / 0.95), not the API's temperature 1 |
| `thinking` | `adaptive` / `enabled` -> thinking on; `disabled` / `between_tools` -> off; unset -> the template's default (on) |
| `output_config.effort` | low / medium / high / xhigh -> the template's reasoning effort; `max` -> xhigh |
| `metadata.user_id` | recorded as the observer's `user` |
| response | `id` (`msg_...`), `content` blocks `thinking` (with `signature`), `text`, `tool_use` (`toolu_...`, `input` an object); `stop_reason` `end_turn`, `max_tokens`, `stop_sequence` (+ `stop_sequence`), `tool_use`, `model_context_window_exceeded`; `request-id` header |
| streaming | `message_start` (sent at once), `content_block_start` / `_delta` (`text_delta`, `thinking_delta`, `signature_delta`, `input_json_delta`) / `_stop`, `message_delta` (cumulative usage), `message_stop`; `ping` every `--sse-keepalive` seconds without a write (queued, prefilling, thinking); a failure after the start is an `error` event that ends the stream |
| usage | `cache_read_input_tokens` = prompt tokens whose state the session cache reused; `cache_creation_input_tokens` = prompt tokens prefilled now that a retained state holds afterwards; `input_tokens` = the rest (the three add up to the prompt) |
| `count_tokens` | the rendered prompt's token count (equals input + cache read + cache creation of the same request; checked by the SDK test) |
| auth | `x-api-key: KEY` or `Authorization: Bearer KEY` (the `--api-key` store); a 401 on these routes is `authentication_error` in the API's shape |
| errors | `{"type": "error", "error": {"type", "message"}}`: 400 `invalid_request_error` (message names the field, `messages.2.content.0.type: ...`), 401, 404 `not_found_error`, 500 `api_error`; 503 for a cancelled non-streaming request (nobody reads it) |
| departures | a client that leaves (or stalls past `--send-timeout`) cancels its generation at the next chunk or token, recorded `cancelled`, as on the OpenAI routes |

**Approximated**

| Feature | What happens |
|---|---|
| `thinking.display` | `summarized` returns the full reasoning (there is no summarizer); `omitted` and `updates` return empty thinking text. Either way the block's `signature` is `halo1.` + base64 of the exact reasoning, and a block sent back with one of our signatures renders its exact text, so the next turn's prompt matches what was generated (a session-cache prefix hit). Signatures are not cryptographic: a client that edits one edits its own prompt. Blocks with other signatures (from Anthropic) render their `thinking` text |
| `thinking.budget_tokens` | validated, not enforced (listed in `x-halo-ignored`) |
| `redacted_thinking` | dropped (opaque) |
| mid-conversation `system` messages (Claude Code sends them) | the template allows system messages only at the start: the text joins the user turn before it, or becomes a user turn after tool results. The model sees it as user text |
| Claude Code's `x-anthropic-billing-header: ...` system block | dropped: billing metadata that changes every session |
| `cache_control` (blocks, tools, top level) | accepted and counted; does not steer the cache, which reuses the longest retained prefix of every prompt by itself (exact mode: checkpoints at multiples of 64 tokens where each prompt's generation starts, docs/SESSIONS.md 8). A prompt shorter than 64 tokens never has a cache read |
| `service_tier`, `safeguards`, `context_management`, `inference_geo`, `diagnostics`, unknown top-level fields | accepted, not acted on, named in the `x-halo-ignored` response header. `context_management` edits are not applied (the full history is rendered) |
| `anthropic-version`, `anthropic-beta` headers | not checked |

**Refused** (400, never silently ignored)

images and documents (also inside `tool_result`), `search_result` and other non-text blocks; Anthropic-defined
tools (`web_search_*`, `bash_*`, `text_editor_*`, `computer_*`, ...); `output_config.format` / `output_format`
(structured outputs); `mcp_servers`; `container`; citations; assistant prefill (a conversation ending in an
assistant turn); thinking explicitly on with a forced `tool_choice`. Message Batches and Files are not served
(404); `GET /v1/models` answers in the OpenAI shape.

## What Claude Code sends (2.1.293, captured with `capture/capture_server.py`)

`POST /v1/messages?beta=true`, `x-api-key`, `anthropic-version: 2023-06-01`, `anthropic-beta:
claude-code-20250219, interleaved-thinking-2025-05-14, context-management-2025-06-27, prompt-caching-scope-2026-01-05,
mid-conversation-system-2026-04-07, effort-2025-11-24, ...`, `stream: true`, `max_tokens: 128000`,
`thinking: {adaptive, display: omitted | updates}`, `output_config: {effort: medium}`, `context_management:
{edits: [clear_thinking_20251015, keep: all]}`, `safeguards: [dangerous_tool_use ...]` (auto permission mode only),
`metadata.user_id`, 20-23 custom tools (no Anthropic-defined types), `system` = 3 text blocks (the billing line,
"You are a Claude agent...", the main prompt) with `cache_control` on the last two, a `system` message after
the first user turn, `cache_control` on the last message. With `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`
nothing else; without it, also `HEAD /api/hello` (answered 401 under `--api-key`; harmless). No `count_tokens`
call in `-p` mode.

## Measured (real model, `real_smoke.sh`, `--max-context 65536`, exact prefix cache, MTP off)

- `tests/functional/anthropic_sdk_check.py` with anthropic 1.12.0: 10/10 (create, stream, thinking round trip
  with a cache read of 128 of 195 tokens, tool use whole and streamed with a tool_result continuation reusing
  256 tokens, forced tool_choice, stop_sequences, max_tokens, count_tokens = the request's total, error types).
- Claude Code `claude -p` (2.1.293, `--model claude-sonnet-5-5 --permission-mode acceptEdits`): "create
  hello.txt containing hello" finished in 2 API calls, 31 s (first prompt 15,556 tokens prefilled in 12.5 s; the
  second reused 15,488 and prefilled 481). "Sum numbers.txt with Bash into sum.txt": 3 API calls (a parallel Bash +
  Read, then Bash awk), correct `12`, 40 s; prompts of 20.3K tokens, turns 2 and 3 reused 20,224 and 20,480.
- message_start's provisional count tokenizes the prompt once more: 7 ms for Claude Code's 16.7K-token request.

## Tests and mutants

`tests/server/anthropic_test.cpp` (server_tests): parsing and refusals, alias resolution, signature round trip
(property), golden event sequences, a Hegel property that every stream follows the grammar and accumulates to the
final Message, the service over HTTP (tool-use round trip renders what the template renders for the equivalent
chat messages, stop sequences, limits, cache usage, forced tool choice, disable_parallel_tool_use, auth, errors,
count_tokens, pings, error events, departures). `OpenAiService.ServingModelUnderDeparturesStallsAndRetries`
now sends each step's requests to `/v1/messages` or `/v1/responses`. `FlashNextGenerator.CacheUsageCountsWhatIs
ReusedAndWhatIsRetained` (gpu_tests, tiny model) checks the cache accounting. 14 planted mutants (`anth-*` in
tests/mutants.toml), all killed.

## Not verified

- The TypeScript SDK and other language SDKs (Claude Code uses the TypeScript SDK and works; the others are not run).
- Interactive Claude Code (only `-p`), subagents, compaction, and Claude Code's behaviour on a `max_tokens` or
  context-window stop from this server.
- Prompts longer than 65K tokens through Claude Code (the server's `--max-context` bounds them; the generator caps
  `max_tokens` to what the context leaves and reports `model_context_window_exceeded`).
