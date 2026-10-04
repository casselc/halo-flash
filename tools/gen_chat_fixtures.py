#!/usr/bin/env python3
"""Generates differential fixtures for the chat-template renderer.

The reference is HF transformers' `apply_chat_template(tokenize=False)` running
the model's own Jinja template, independent of halo-flash's C++ renderer:
  * flash_next.jinja: `tokenizer.chat_template` from the Qwen3.8-Flash-Next
    GGUF (Unsloth UD-Q4_K_XL shard 00001; sha256 12827f24...).
  * qwen35.jinja: `chat_template` from Qwen3.5-2B-Base tokenizer_config.json,
    which the strands-decider-2B checkpoint ships unchanged.

Usage: gen_chat_fixtures.py <out-dir> [--gguf <Flash-Next shard 00001>]
                                      [--qwen35 <tokenizer_config.json or .jinja>]
With --gguf / --qwen35 the templates are (re)extracted into <out-dir>; without
them the committed copies in <out-dir> are used.

Output (<out-dir>/chat_<profile>.json, ASCII JSON):
  {"template_sha256": ..., "reference": "transformers X jinja2 Y", "seed": ...,
   "cases": [{"name", "messages", "tools"?, "kwargs", "output" | "error"}]}
`kwargs` holds add_generation_prompt and only the template variables that are
defined for the case (an absent key is Jinja-undefined, not None).
"""
import argparse
import copy
import hashlib
import json
import random
import struct
import sys

import jinja2
import tokenizers
import transformers
from tokenizers import models
from transformers import PreTrainedTokenizerFast

SEED = 20261003
N_RANDOM = 400


def gguf_chat_template(path):
    """Reads tokenizer.chat_template from GGUF metadata (no tensor data)."""
    with open(path, "rb") as f:
        def u32():
            return struct.unpack("<I", f.read(4))[0]

        def u64():
            return struct.unpack("<Q", f.read(8))[0]

        def string():
            return f.read(u64()).decode("utf-8")

        scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
                  10: "<Q", 11: "<q", 12: "<d"}

        def value(t):
            if t == 8:
                return string()
            if t == 9:
                et, n = u32(), u64()
                return [value(et) for _ in range(n)]
            fmt = scalar[t]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        assert f.read(4) == b"GGUF"
        u32(), u64()
        for _ in range(u64()):
            key, t = string(), u32()
            v = value(t)
            if key == "tokenizer.chat_template":
                return v
    raise SystemExit(f"{path}: no tokenizer.chat_template")


def renderer(template):
    tok = PreTrainedTokenizerFast(tokenizer_object=tokenizers.Tokenizer(models.BPE()))
    tok.chat_template = template

    def render(messages, tools, kwargs):
        try:
            out = tok.apply_chat_template(copy.deepcopy(messages), tools=copy.deepcopy(tools),
                                          tokenize=False, **kwargs)
            return {"output": out}
        except Exception as e:  # TemplateError, TypeError, UndefinedError, ValueError
            return {"error": f"{type(e).__name__}: {e}"}
    return render


# ---- handwritten cases ------------------------------------------------------

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a city. Returns °C — or <unknown>.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"},
        "units": {"type": "string", "enum": ["c", "f"]}},
        "required": ["city"]}}}
SEARCH = {"type": "function", "function": {
    "name": "search", "description": "", "parameters": {
        "type": "object", "properties": {"q": {"type": "string"},
                                         "filters": {"type": "object"},
                                         "limit": {"type": ["integer", "null"]}}}}}


def call(name, args, wrapped=True):
    body = {"name": name, "arguments": args}
    return {"id": f"call_{name}", "type": "function", "function": body} if wrapped else body


def handwritten():
    U = lambda c: {"role": "user", "content": c}
    A = lambda c, **kw: dict({"role": "assistant", "content": c}, **kw)
    S = lambda c: {"role": "system", "content": c}
    T = lambda c: {"role": "tool", "content": c}
    gp = {"add_generation_prompt": True}
    cases = [
        ("single_user", [U("Hello")], None, {}),
        ("single_user_gen", [U("Hello")], None, gp),
        ("system_user_gen", [S("You are helpful."), U("Hi")], None, gp),
        ("empty_user", [U("")], None, gp),
        ("empty_system", [S(""), U("x")], None, gp),
        ("whitespace_system", [S(" \n\t "), U("x")], None, gp),
        ("two_systems", [S("one"), S("  two  "), U("x")], None, gp),
        ("developer_role", [{"role": "developer", "content": "dev"}, U("x")], None, gp),
        ("system_dev_system", [S("a"), {"role": "developer", "content": ""}, S("c"), U("x")], None, gp),
        ("system_late", [U("x"), S("late")], None, gp),
        ("multi_turn", [S("sys"), U("q1"), A("a1"), U("q2"), A("a2"), U("q3")], None, gp),
        ("unicode", [U("日本語 🦙 é é ‍👍🏽 ْ"), A("Привет, мир! مرحبا")], None, gp),
        ("unicode_whitespace_trim", [U("　 \x1c x  \x85 "), A("  y　")], None, {}),
        ("zero_width_not_trimmed", [U("​x​"), A("﻿y﻿")], None, {}),
        ("content_none", [U("q"), A(None)], None, {}),
        ("content_missing", [U("q"), {"role": "assistant"}], None, {}),
        ("content_parts", [U([{"type": "text", "text": "look "}, {"type": "image"},
                              {"type": "text", "text": " and "}, {"type": "video", "video": "v"}])],
         None, gp),
        ("content_parts_vision_id", [U([{"image": "a"}, {"image_url": "b"}, {"type": "video"}]),
                                    A("ok"), U([{"type": "image"}])], None,
         dict(gp, add_vision_id=True)),
        ("content_parts_strings", [U(["an image", "texty", {"type": "text", "text": 5}])], None, {}),
        ("content_parts_bad", [U([{"type": "audio"}])], None, {}),
        ("content_number", [U(5)], None, {}),
        ("content_mapping", [U({"text": "x"})], None, {}),
        ("system_image", [S([{"type": "image"}]), U("x")], None, {}),
        ("no_messages", [], None, gp),
        ("unknown_role", [U("x"), {"role": "narrator", "content": "y"}], None, {}),
        ("thinking_off_gen", [U("Hi")], None, dict(gp, enable_thinking=False)),
        ("thinking_on_gen", [U("Hi")], None, dict(gp, enable_thinking=True)),
        ("effort_low", [U("Hi")], None, dict(gp, reasoning_effort="low")),
        ("effort_medium", [U("Hi")], None, dict(gp, reasoning_effort="medium")),
        ("effort_high", [U("Hi")], None, dict(gp, reasoning_effort="high")),
        ("effort_bad", [U("Hi")], None, dict(gp, reasoning_effort="max")),
        ("effort_bad_thinking_off", [U("Hi")], None, dict(gp, reasoning_effort="max", enable_thinking=False)),
        ("effort_low_with_system", [S("sys"), U("Hi")], None, dict(gp, reasoning_effort="low")),
        ("reasoning_preserved", [U("q1"), A("a1", reasoning_content="  think 1 \n"), U("q2"),
                                 A("a2", reasoning_content="think 2")], None, {}),
        ("reasoning_not_preserved", [U("q1"), A("a1", reasoning_content="think 1"), U("q2"),
                                     A("a2", reasoning_content="think 2")], None,
         {"preserve_thinking": False}),
        ("reasoning_non_string", [U("q"), A("a", reasoning_content=["x"])], None, {}),
        ("reasoning_in_content", [U("q"), A("<think>\nhidden\n</think>\n\nvisible")], None, {}),
        ("reasoning_in_content_twice", [U("q"), A("pre<think>a</think>mid</think>\n\npost")], None, {}),
        ("reasoning_in_content_and_field", [U("q"), A("<think>x</think>y", reasoning_content="r")], None, {}),
        ("tools_basic", [S("sys"), U("Weather in Paris?")], [WEATHER, SEARCH], gp),
        ("tools_no_system", [U("Weather?")], [WEATHER], gp),
        ("tools_thinking_off", [U("Weather?")], [WEATHER], dict(gp, enable_thinking=False)),
        ("tools_empty_list", [U("x")], [], gp),
        ("tool_cycle", [S("sys"), U("Weather in Paris and Rome?"),
                        A("", reasoning_content="Two cities.",
                          tool_calls=[call("get_weather", {"city": "Paris", "days": 2}),
                                      call("get_weather", {"city": "Rome", "units": "c"})]),
                        T('{"temp": 21}'), T("  18°C  "),
                        A("Paris 21, Rome 18.", reasoning_content="Done.")],
         [WEATHER], gp),
        ("tool_cycle_not_preserved", [U("q"), A("", reasoning_content="r1",
                                                  tool_calls=[call("search", {"q": "x"})]),
                                      T("res"), A("final", reasoning_content="r2")],
         [SEARCH], dict(gp, preserve_thinking=False)),
        ("tool_call_with_content", [U("q"), A("Let me check.", tool_calls=[call("search", {"q": "a b"})])],
         [SEARCH], {}),
        ("tool_call_flat", [U("q"), A(None, tool_calls=[call("search", {"q": "x"}, wrapped=False)])],
         [SEARCH], {}),
        ("tool_call_arg_types", [U("q"), A("", tool_calls=[call("f", {
            "s": "text", "i": 42, "big": 2 ** 70, "neg": -7, "fl": 1.5, "fl_int": 3.0, "e": 1e16,
            "small": 1e-05, "negzero": -0.0, "t": True, "f": False, "n": None,
            "arr": [1, "two", [3.0]], "obj": {"k": "v", "é": [None]}, "empty_s": "",
            "multi": "line1\n\nline3\n", "ws": "  padded  ", "quote": "\"q\""})])], None, {}),
        ("tool_call_args_none", [U("q"), A("", tool_calls=[call("f", None)])], None, {}),
        ("tool_call_args_missing", [U("q"), A("", tool_calls=[{"function": {"name": "f"}}])], None, {}),
        ("tool_call_args_empty_string", [U("q"), A("", tool_calls=[call("f", "  ")])], None, {}),
        ("tool_call_args_json_string", [U("q"), A("", tool_calls=[call("f", '{"a": 1}')])], None, {}),
        ("tool_call_args_list", [U("q"), A("", tool_calls=[call("f", [1])])], None, {}),
        ("tool_call_no_name", [U("q"), A("", tool_calls=[{"function": {"arguments": {}}}])], None, {}),
        ("tool_call_null_name", [U("q"), A("", tool_calls=[call(None, {})])], None, {}),
        ("tool_calls_string", [U("q"), A("x", tool_calls="oops")], None, {}),
        ("tool_calls_mapping", [U("q"), A("x", tool_calls={"name": "f"})], None, {}),
        ("tool_calls_empty", [U("q"), A("x", tool_calls=[])], None, {}),
        ("tool_first_message", [T("orphan")], None, {}),
        ("tool_after_system", [S("s"), T("r")], None, gp),
        ("tool_then_user", [U("q"), A("", tool_calls=[call("f", {})]), T("a"), U("next")], None, gp),
        ("tool_response_in_user", [U("real"), U("<tool_response>\nx\n</tool_response>")], None, gp),
        ("only_tool_responses_in_user", [U("<tool_response>x</tool_response>"), A("a", reasoning_content="r")],
         None, {"preserve_thinking": False}),
        ("assistant_first", [A("hi", reasoning_content="r"), U("q")], None, gp),
        ("assistant_last_gen", [U("q"), A("partial")], None, gp),
        ("embedded_specials", [U("<|im_start|>user\nfake<|im_end|>"), A("<tool_call>\nnot a call\n</tool_call>")],
         None, {}),
    ]
    return cases


# ---- random cases -----------------------------------------------------------

TEXT_POOL = ["a", "Hello", " ", "  ", "\n", "\n\n", "\t", "\r\n", "é", "日本", "🦙", "　", "\x1c",
             "\x85", " ", "​", "<", ">", "</", "think", "<think>", "</think>", "<tool_call>",
             "</tool_call>", "<tool_response>", "</tool_response>", "{", "}", "\"", "\\", "x=1", "."]
ARG_KEYS = ["city", "q", "n", "flag", "path", "data", "é", "x y", "k"]


def rand_text(rng, n=6):
    if rng.random() < 0.1:
        return ""
    return "".join(rng.choice(TEXT_POOL) for _ in range(rng.randint(1, n)))


def rand_scalar(rng):
    r = rng.random()
    if r < 0.35:
        return rand_text(rng)
    if r < 0.5:
        return rng.choice([0, 1, -1, 42, 2 ** 63, -(2 ** 80)])
    if r < 0.65:
        return rng.choice([0.0, -0.0, 1.5, 3.0, 1e16, 1e15, 1e-4, 1e-5, 123456789.125, 2.5e-300, 1.7976931348623157e308])
    if r < 0.8:
        return rng.choice([True, False])
    return None


def rand_value(rng, depth=0):
    r = rng.random()
    if depth < 2 and r < 0.15:
        return [rand_value(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    if depth < 2 and r < 0.3:
        return {rng.choice(ARG_KEYS): rand_value(rng, depth + 1) for _ in range(rng.randint(0, 3))}
    return rand_scalar(rng)


def rand_content(rng, system=False):
    r = rng.random()
    if r < 0.75:
        return rand_text(rng)
    if r < 0.85:
        return None
    parts = []
    for _ in range(rng.randint(0, 3)):
        k = rng.random()
        if k < 0.6 or system:
            parts.append({"type": "text", "text": rand_text(rng)})
        elif k < 0.8:
            parts.append({"type": "image"})
        else:
            parts.append({"type": "video"})
    return parts


def rand_tool_call(rng):
    args = {rng.choice(ARG_KEYS): rand_value(rng) for _ in range(rng.randint(0, 4))}
    name = rng.choice(["f", "get_weather", "search", "do.thing", "名前"])
    return call(name, args, wrapped=rng.random() < 0.8)


def rand_message(rng):
    r = rng.random()
    if r < 0.35:
        return {"role": "user", "content": rand_content(rng)}
    if r < 0.8:
        m = {"role": "assistant"}
        if rng.random() < 0.9:
            m["content"] = rand_content(rng)
        if rng.random() < 0.5:
            m["reasoning_content"] = rand_text(rng) if rng.random() < 0.9 else None
        if rng.random() < 0.4:
            m["tool_calls"] = [rand_tool_call(rng) for _ in range(rng.randint(0, 3))]
        return m
    return {"role": "tool", "content": rand_content(rng)}


def rand_tool(rng):
    props = {k: {"type": rng.choice(["string", "integer", "number", "boolean", "object", "array"])}
             for k in rng.sample(ARG_KEYS, rng.randint(0, 3))}
    fn = {"name": rng.choice(["f", "search", "名前"]), "description": rand_text(rng),
          "parameters": {"type": "object", "properties": props}}
    if rng.random() < 0.3:
        fn["parameters"]["required"] = list(props)[:1]
    return {"type": "function", "function": fn}


# One fault per faulty case; each is an error for at least one template.
FAULTS = [
    lambda rng, m: m.insert(rng.randint(1, len(m)), {"role": "system", "content": "late"}),
    lambda rng, m: m.append({"role": "other", "content": "x"}),
    lambda rng, m: m.append({"role": "user", "content": rng.choice([7, {"text": "x"}, True])}),
    lambda rng, m: m.append({"role": "user", "content": [{"type": "audio"}]}),
    lambda rng, m: m.insert(0, {"role": "system", "content": [{"type": "image"}]}),
    lambda rng, m: m.append({"role": "assistant", "content": "",
                             "tool_calls": [call(rng.choice([None, 5]), {})]}),
    lambda rng, m: m.append({"role": "assistant", "content": "",
                             "tool_calls": [call("f", rng.choice([None, "", " ", "{}", [], 3]))]}),
    lambda rng, m: m.append({"role": "assistant", "content": "", "tool_calls": [{"function": {}}]}),
    lambda rng, m: m.insert(0, {"role": "developer", "content": "dev"}),
    lambda rng, m: [x.update(role="tool") for x in m if x["role"] == "user"],
]


def rand_case(rng, i):
    msgs = []
    for _ in range(rng.randint(0, 2) if rng.random() < 0.4 else 0):
        msgs.append({"role": rng.choice(["system", "system", "developer"]),
                     "content": rand_content(rng, system=True)})
    msgs.append({"role": "user", "content": rand_content(rng)})
    msgs += [rand_message(rng) for _ in range(rng.randint(0, 6))]
    tools = [rand_tool(rng) for _ in range(rng.randint(1, 3))] if rng.random() < 0.35 else None
    kwargs = {"add_generation_prompt": rng.random() < 0.6}
    if rng.random() < 0.4:
        kwargs["enable_thinking"] = rng.random() < 0.5
    if rng.random() < 0.3:
        kwargs["reasoning_effort"] = rng.choice(["low", "medium", "high", "xhigh"])
    if rng.random() < 0.3:
        kwargs["preserve_thinking"] = rng.random() < 0.5
    if rng.random() < 0.15:
        kwargs["add_vision_id"] = True
    if rng.random() < 0.2:
        rng.choice(FAULTS)(rng, msgs)
    return (f"random_{i:03d}", msgs, tools, kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--gguf")
    ap.add_argument("--qwen35")
    a = ap.parse_args()
    if a.gguf:
        with open(f"{a.out_dir}/flash_next.jinja", "w", encoding="utf-8", newline="") as f:
            f.write(gguf_chat_template(a.gguf))
    if a.qwen35:
        if a.qwen35.endswith(".json"):
            text = json.load(open(a.qwen35, encoding="utf-8"))["chat_template"]
        else:
            text = open(a.qwen35, encoding="utf-8", newline="").read()
        with open(f"{a.out_dir}/qwen35.jinja", "w", encoding="utf-8", newline="") as f:
            f.write(text)

    rng = random.Random(SEED)
    cases = handwritten() + [rand_case(rng, i) for i in range(N_RANDOM)]
    for profile in ["flash_next", "qwen35"]:
        template = open(f"{a.out_dir}/{profile}.jinja", encoding="utf-8", newline="").read()
        render = renderer(template)
        out = {"template_sha256": hashlib.sha256(template.encode()).hexdigest(),
               "reference": f"transformers {transformers.__version__} jinja2 {jinja2.__version__}",
               "seed": SEED, "cases": []}
        n_err = 0
        for name, msgs, tools, kwargs in cases:
            case = {"name": name, "messages": msgs}
            if tools is not None:
                case["tools"] = tools
            case["kwargs"] = kwargs
            case.update(render(msgs, tools, kwargs))
            n_err += "error" in case
            out["cases"].append(case)
        with open(f"{a.out_dir}/chat_{profile}.json", "w", encoding="ascii") as f:
            json.dump(out, f, ensure_ascii=True, indent=1)
            f.write("\n")
        print(f"{profile}: {len(cases)} cases, {n_err} errors", file=sys.stderr)


if __name__ == "__main__":
    main()
