#!/usr/bin/env python3
"""Anthropic SDK conformance checks against a running halo_serve (docs/ANTHROPIC.md).

Uses the official anthropic package's typed client and stream helpers (which accumulate the Message from the
events and so reject malformed streams): create, stream (text_stream and the final message), thinking blocks
sent back on the next turn, tool use (whole and streamed) with a tool_result continuation, forced tool_choice,
stop sequences, max_tokens, count_tokens against the usage of the same request, cache usage on a continued
conversation, and the error types.

  python3 tests/functional/anthropic_sdk_check.py --base-url http://127.0.0.1:8080 [--api-key KEY]

The base URL has no /v1 (the SDK adds it). Exit status 0 when every check passes; each failure is printed.
"""

import argparse
import sys
import traceback

import anthropic
from anthropic import Anthropic

MODEL = "claude-sonnet-5-5"  # resolved by halo_serve's alias table to the served model
TOOLS = [{
    "name": "get_weather",
    "description": "Current weather for a city",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}]
NO_THINK = {"type": "disabled"}


def check(name, fn, failures):
    try:
        detail = fn()
        print(f"ok    {name}" + (f"  ({detail})" if detail else ""))
    except Exception as e:  # noqa: BLE001 - report every failure and continue
        failures.append(name)
        print(f"FAIL  {name}: {e}")
        traceback.print_exc(limit=2)


def total_input(u):
    return u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", default="unused")
    args = ap.parse_args()
    client = Anthropic(base_url=args.base_url, api_key=args.api_key, max_retries=0, timeout=600)
    failures = []

    def create():
        m = client.messages.create(model=MODEL, max_tokens=64, thinking=NO_THINK,
                                   messages=[{"role": "user", "content": "Reply with the single word: pong"}])
        assert m.type == "message" and m.role == "assistant", m
        assert m.stop_reason == "end_turn", m.stop_reason
        assert m.content and m.content[0].type == "text", m.content
        assert "pong" in m.content[0].text.lower(), m.content[0].text
        u = m.usage
        assert u.input_tokens >= 0 and u.output_tokens > 0, u
        assert u.cache_read_input_tokens is not None and u.cache_creation_input_tokens is not None, u
        return f"model {m.model}, {m.content[0].text.strip()!r}, usage {u.input_tokens}/{u.cache_read_input_tokens}/" \
               f"{u.cache_creation_input_tokens}/{u.output_tokens}"

    def stream_text():
        with client.messages.stream(model=MODEL, max_tokens=64, thinking=NO_THINK,
                                    messages=[{"role": "user", "content": "Count from 1 to 5, comma separated."}]) as s:
            text = "".join(s.text_stream)
            final = s.get_final_message()
        assert final.content[-1].type == "text" and final.content[-1].text == text, (final.content, text)
        assert "3" in text, text
        assert final.stop_reason == "end_turn", final.stop_reason
        assert final.usage.output_tokens > 0
        return repr(text.strip())

    # The session cache keeps checkpoints at multiples of 64 prompt tokens (exact mode): a conversation shorter
    # than that has nothing to reuse, so this one starts with a longer system prompt.
    system = "You are a careful arithmetic assistant. " * 12

    def thinking_round_trip():
        msgs = [{"role": "user", "content": "What is 17 * 3? Answer with the number."}]
        with client.messages.stream(model=MODEL, max_tokens=2048, thinking={"type": "adaptive"}, system=system,
                                    output_config={"effort": "low"}, messages=msgs) as s:
            first = s.get_final_message()
        kinds = [b.type for b in first.content]
        assert kinds[0] == "thinking", kinds
        assert first.content[0].signature, "no signature"
        assert "51" in first.content[-1].text, first.content
        # Send the thinking block back (display omitted on the second turn: its text comes from the signature).
        msgs += [{"role": "assistant", "content": first.content},
                 {"role": "user", "content": "And 51 + 1? Answer with the number."}]
        second = client.messages.create(model=MODEL, max_tokens=2048, messages=msgs, system=system,
                                        thinking={"type": "adaptive", "display": "omitted"},
                                        output_config={"effort": "low"})
        thinking = [b for b in second.content if b.type == "thinking"]
        assert thinking and thinking[0].thinking == "" and thinking[0].signature, second.content
        assert "52" in second.content[-1].text, second.content
        u = second.usage
        assert u.cache_read_input_tokens > 0, f"the continued conversation reused nothing: {u}"
        return f"turn 2 reused {u.cache_read_input_tokens} of {total_input(u)} prompt tokens"

    def tool_use():
        msgs = [{"role": "user", "content": "What is the weather in Paris? Use the get_weather tool."}]
        m = client.messages.create(model=MODEL, max_tokens=512, thinking=NO_THINK, tools=TOOLS, messages=msgs)
        assert m.stop_reason == "tool_use", (m.stop_reason, m.content)
        use = [b for b in m.content if b.type == "tool_use"][0]
        assert use.name == "get_weather" and isinstance(use.input, dict) and "paris" in use.input["city"].lower(), use
        assert use.id.startswith("toolu_"), use.id
        msgs += [{"role": "assistant", "content": m.content},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use.id, "content": "18C, sunny"}]}]
        m2 = client.messages.create(model=MODEL, max_tokens=512, thinking=NO_THINK, tools=TOOLS, messages=msgs)
        assert m2.stop_reason == "end_turn", m2.stop_reason
        text = "".join(b.text for b in m2.content if b.type == "text")
        assert "18" in text, text
        assert m2.usage.cache_read_input_tokens > 0, m2.usage
        return f"{use.input}, then {text.strip()[:60]!r}; reused {m2.usage.cache_read_input_tokens}"

    def tool_use_streamed():
        with client.messages.stream(model=MODEL, max_tokens=512, thinking=NO_THINK, tools=TOOLS,
                                    messages=[{"role": "user", "content": "Weather in Oslo? Use get_weather."}]) as s:
            saw_json = False
            for event in s:
                if event.type == "input_json":
                    saw_json = True
            final = s.get_final_message()
        assert saw_json, "no input_json event"
        use = [b for b in final.content if b.type == "tool_use"][0]
        assert "oslo" in use.input["city"].lower(), use.input
        return str(use.input)

    def forced_tool():
        m = client.messages.create(model=MODEL, max_tokens=256, tools=TOOLS,
                                   tool_choice={"type": "tool", "name": "get_weather"},
                                   messages=[{"role": "user", "content": "I am going to Rome tomorrow."}])
        use = [b for b in m.content if b.type == "tool_use"]
        assert use and use[0].name == "get_weather", m.content
        return str(use[0].input)

    def stop_sequence():
        m = client.messages.create(model=MODEL, max_tokens=128, thinking=NO_THINK, stop_sequences=["7"],
                                   messages=[{"role": "user", "content": "Count from 1 to 10, comma separated."}])
        assert m.stop_reason == "stop_sequence" and m.stop_sequence == "7", (m.stop_reason, m.stop_sequence)
        assert "7" not in m.content[0].text and "6" in m.content[0].text, m.content[0].text
        return repr(m.content[0].text)

    def max_tokens():
        m = client.messages.create(model=MODEL, max_tokens=5, thinking=NO_THINK,
                                   messages=[{"role": "user", "content": "Write a long story about a lighthouse."}])
        assert m.stop_reason == "max_tokens" and m.usage.output_tokens == 5, (m.stop_reason, m.usage)
        return None

    def count_tokens():
        req = dict(model=MODEL, system="You are terse.", tools=TOOLS, thinking=NO_THINK,
                   messages=[{"role": "user", "content": "Hello there"}])
        n = client.messages.count_tokens(**req).input_tokens
        m = client.messages.create(max_tokens=8, **req)
        assert n == total_input(m.usage), (n, m.usage)
        return f"{n} tokens = the request's input + cache read + cache creation"

    def errors():
        try:
            client.messages.create(model=MODEL, max_tokens=8, messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}]}])
            raise AssertionError("an image was accepted")
        except anthropic.BadRequestError as e:
            assert e.body["error"]["type"] == "invalid_request_error", e.body
        try:
            client.messages.create(model="gpt-unknown", max_tokens=8, messages=[{"role": "user", "content": "x"}])
            raise AssertionError("an unknown model was accepted")
        except anthropic.NotFoundError:
            pass
        if args.api_key != "unused":
            bad = Anthropic(base_url=args.base_url, api_key="wrong", max_retries=0)
            try:
                bad.messages.create(model=MODEL, max_tokens=8, messages=[{"role": "user", "content": "x"}])
                raise AssertionError("a wrong key was accepted")
            except anthropic.AuthenticationError:
                pass
        return None

    for name, fn in [("messages.create", create), ("messages.stream text", stream_text),
                     ("thinking blocks round trip + cache read", thinking_round_trip),
                     ("tool use + tool_result", tool_use), ("tool use streamed", tool_use_streamed),
                     ("forced tool_choice", forced_tool), ("stop_sequences", stop_sequence),
                     ("max_tokens", max_tokens), ("count_tokens", count_tokens), ("errors", errors)]:
        check(name, fn, failures)
    print(f"\n{len(failures)} failed" if failures else "\nall passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
