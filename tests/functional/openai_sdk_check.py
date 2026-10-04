#!/usr/bin/env python3
"""OpenAI SDK conformance checks against a running halo_serve.

Uses the official openai package's typed clients and stream helpers (which
reassemble responses from events and so reject malformed wire formats):
chat (whole, streamed, tools, usage chunk), legacy completions (whole,
streamed, stop sequences), responses (whole, streamed via
client.responses.stream, function calling with a continuation through
previous_response_id, retrieval), models, errors, and /metrics.

  python3 tests/functional/openai_sdk_check.py --base-url http://127.0.0.1:8080/v1

Exit status 0 when every check passes; each failure is printed.
"""

import argparse
import json
import sys
import traceback
import urllib.request

import openai
from openai import OpenAI

TOOLS_CHAT = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}]
TOOLS_RESPONSES = [{
    "type": "function",
    "name": "get_weather",
    "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}]


def check(name, fn, failures):
    try:
        detail = fn()
        print(f"ok    {name}" + (f"  ({detail})" if detail else ""))
    except Exception as e:  # noqa: BLE001 - report every failure and continue
        failures.append(name)
        print(f"FAIL  {name}: {e}")
        traceback.print_exc(limit=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", default="unused")
    args = ap.parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0, timeout=300)
    failures = []
    no_think = {"reasoning_effort": "none"}

    def models():
        ids = [m.id for m in client.models.list().data]
        assert ids, "no models"
        return ids[0]

    def chat_whole():
        r = client.chat.completions.create(
            model="flash", messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            max_tokens=8, temperature=0, **no_think)
        c = r.choices[0]
        assert c.message.role == "assistant"
        assert c.message.content and "pong" in c.message.content.lower(), c.message.content
        assert c.finish_reason in ("stop", "length")
        assert r.usage.prompt_tokens > 0 and r.usage.completion_tokens > 0
        assert r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens
        return repr(c.message.content)

    def chat_stream():
        text, finish, usage, roles = "", None, None, 0
        with client.chat.completions.create(
                model="flash", messages=[{"role": "user", "content": "Count from 1 to 5, comma separated."}],
                max_tokens=24, temperature=0, stream=True, stream_options={"include_usage": True},
                **no_think) as stream:
            for chunk in stream:
                if chunk.usage:
                    usage = chunk.usage
                for c in chunk.choices:
                    roles += c.delta.role == "assistant"
                    text += c.delta.content or ""
                    finish = c.finish_reason or finish
        assert roles == 1, f"role sent {roles} times"
        assert "1" in text and "5" in text, text
        assert finish in ("stop", "length")
        assert usage and usage.completion_tokens > 0
        return repr(text)

    def chat_tools():
        r = client.chat.completions.create(
            model="flash", messages=[{"role": "user", "content": "What is the weather in Paris right now?"}],
            tools=TOOLS_CHAT, max_tokens=400, temperature=0, reasoning_effort="low")
        c = r.choices[0]
        assert c.finish_reason == "tool_calls", c.finish_reason
        call = c.message.tool_calls[0]
        assert call.type == "function" and call.function.name == "get_weather"
        assert json.loads(call.function.arguments)["city"].lower().startswith("paris")
        # Send the result back; the answer should use it.
        r2 = client.chat.completions.create(
            model="flash", tools=TOOLS_CHAT, max_tokens=200, temperature=0, reasoning_effort="low",
            messages=[{"role": "user", "content": "What is the weather in Paris right now?"},
                      {"role": "assistant", "content": c.message.content or "", "tool_calls": [call.model_dump()]},
                      {"role": "tool", "tool_call_id": call.id, "content": '{"temp_c": 18, "sky": "cloudy"}'}])
        answer = r2.choices[0].message.content or ""
        assert "18" in answer, answer
        return f"call {call.function.arguments}, answer {answer[:60]!r}"

    def completions():
        r = client.completions.create(model="flash", prompt="The capital of France is", max_tokens=6, temperature=0)
        assert "Paris" in r.choices[0].text, r.choices[0].text
        s = client.completions.create(model="flash", prompt="1, 2, 3,", max_tokens=12, temperature=0,
                                      stop=[", 6"], stream=True)
        text = "".join(c.choices[0].text for c in s if c.choices)
        assert ", 6" not in text and "5" in text, text
        return repr(r.choices[0].text) + " / " + repr(text)

    def responses_whole():
        r = client.responses.create(model="flash", input="Reply with the single word: pong",
                                    reasoning={"effort": "none"}, max_output_tokens=8, temperature=0)
        assert r.status == "completed", r.status
        assert "pong" in r.output_text.lower(), r.output_text
        got = client.responses.retrieve(r.id)
        assert got.id == r.id and got.output_text == r.output_text
        return repr(r.output_text)

    def responses_stream():
        with client.responses.stream(model="flash", input="Count from 1 to 3, comma separated.",
                                     reasoning={"effort": "none"}, max_output_tokens=16, temperature=0) as stream:
            deltas = "".join(e.delta for e in stream if e.type == "response.output_text.delta")
            final = stream.get_final_response()
        assert final.output_text == deltas, (final.output_text, deltas)
        assert "1" in deltas and "3" in deltas
        return repr(deltas)

    def responses_tools():
        r = client.responses.create(model="flash", input="What is the weather in Paris right now?",
                                    tools=TOOLS_RESPONSES, reasoning={"effort": "low"}, max_output_tokens=400,
                                    temperature=0)
        calls = [o for o in r.output if o.type == "function_call"]
        assert calls, [o.type for o in r.output]
        call = calls[0]
        assert call.name == "get_weather" and json.loads(call.arguments)["city"].lower().startswith("paris")
        r2 = client.responses.create(model="flash", previous_response_id=r.id, tools=TOOLS_RESPONSES,
                                     reasoning={"effort": "low"}, max_output_tokens=200, temperature=0,
                                     input=[{"type": "function_call_output", "call_id": call.call_id,
                                             "output": '{"temp_c": 18, "sky": "cloudy"}'}])
        assert "18" in r2.output_text, r2.output_text
        cached = r2.usage.input_tokens_details.cached_tokens
        assert cached > 0, "the continuation reused no engine state"
        return f"cached {cached} of {r2.usage.input_tokens}, answer {r2.output_text[:60]!r}"

    def errors():
        try:
            client.chat.completions.create(model="flash", messages=[], max_tokens=4)
        except openai.BadRequestError as e:
            assert e.body and e.body.get("param") == "messages", e.body
        else:
            raise AssertionError("empty messages accepted")
        try:
            client.responses.retrieve("resp_does_not_exist")
        except openai.NotFoundError:
            pass
        else:
            raise AssertionError("unknown response found")

    def metrics():
        base = args.base_url.rsplit("/v1", 1)[0]
        text = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
        assert "vllm:generation_tokens_total" in text and "vllm:e2e_request_latency_seconds_count" in text

    for name, fn in [("models", models), ("chat", chat_whole), ("chat stream", chat_stream),
                     ("chat tools round trip", chat_tools), ("completions + stop", completions),
                     ("responses", responses_whole), ("responses stream helper", responses_stream),
                     ("responses tools + previous_response_id", responses_tools), ("errors", errors),
                     ("metrics", metrics)]:
        check(name, fn, failures)
    print(f"\n{len(failures)} failed" if failures else "\nall passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
