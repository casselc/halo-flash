#!/usr/bin/env python3
"""CPU-only K2 planning; no runtime routing, timing or exactness claims.

Scratch mirrors AllocChunk in src/engine/prefill_chunk.cpp at 0810614.
The shape below is Config's documented Qwen shape, not metadata from a loaded GGUF.
Device totals exclude KV, weights, mixer permutations and MTP buffers. Host totals
include ck_host_ only, not PLE prefetch vectors, hash rows or allocator overhead.
kAttnSub is 512 (flash_next_gpu.hpp:590). Block scores use their initial stride;
GrowBlockScores growth at longer positions is excluded from these absolute totals.
"""

import argparse
import json
from pathlib import Path
import re


def chunks(tokens, base=2048, long_chunk=8192, context=262144):
    if tokens < 0 or base < 1 or context < 1 or long_chunk < 0:
        raise ValueError("invalid chunk configuration")
    if tokens > context:
        raise ValueError("prefill exceeds available context")
    want = long_chunk if long_chunk > base and tokens > base else base
    cap = min(want, context)
    return [min(cap, tokens - at) for at in range(0, tokens, cap)], cap


def merge_tail(plan, limit=256):
    """Hypothetical final-chunk merge; removes a cancellation poll boundary."""
    if len(plan) < 2 or not 0 < plan[-1] <= limit:
        return list(plan)
    return list(plan[:-2]) + [plan[-2] + plan[-1]]


def scratch(tokens, context=262144):
    if tokens < 1 or context < tokens:
        raise ValueError("invalid scratch capacity")
    t, h, d = tokens, 2560, 10240
    qdim, kvdim, vd = 6144, 512, 6144
    experts, used, ff, hc, low = 512, 10, 640, 4, 320
    max_cols = max(d, h, qdim, vd, ff, low)
    pairs = t * used
    # f(n) is max(n, 1) * sizeof(float), including packed byte/half buffers.
    elements = {
        "res": t*d, "xn": t*d, "lo": t*low, "gate": t*d,
        "mixed": t*h, "inject": t*hc, "block_out": t*h, "embd": t*h,
        "xq": t*max_cols//4+1, "xd": t*max_cols//32+1,
        "qkv": t*10240, "conv": t*10240, "z": t*vd,
        "alpha": t*48, "beta": t*48, "gdn": t*vd,
        "qg": t*2*qdim, "q": t*qdim, "gate_a": t*qdim,
        "k": t*kvdim, "v": t*kvdim, "ctx": t*qdim,
        "iq": t*4*128, "ik": t*128, "router": t*experts,
        "ids": pairs, "wts": pairs, "eg": t*ff, "eu": t*ff,
        "ed": t*h, "sgate": t, "sorted": pairs, "pos": pairs,
        "tiles": 3*(pairs//64+experts+1), "ntiles": 1,
        "hp": pairs*ff, "yp": pairs*h, "x16": t*h//2+1,
        "bcounts": (pairs+1023)//1024*experts, "bbase": experts,
        "pa": t*d, "ipart": t*hc*8,
        "gdnprep": 48*((t+15)//16)*576, "rscale": t*hc,
        "xh": t*max_cols//2+1, "mh": t*h//2+1,
        "lh": t*low//2+1, "ml": t*h//2+1, "pb": t*d,
    }
    sub = min(t, 512)
    visible = min(context, (2048//4+1)*4)
    groups, max_union = (sub+3)//4, 4*(2048//4)+4
    elements.update({
        "bscores": sub*min(context//4+1, 16384),
        "vis": sub*visible, "ascores": sub*24*visible,
        "ublk": groups*max_union, "umask": (groups*max_union+3)//4,
        "ucount": groups, "qsh": sub*4*128//2+1, "qsl": sub*4*128//2+1,
    })
    buffers = {name: max(n, 1)*4 for name, n in elements.items()}
    params_offset = (t*2560*4+63)//64*64
    host = params_offset+t*64
    buffers["host"] = host  # AllocChunk allocates a device copy of ck_host_.
    return {"device_bytes": sum(buffers.values()), "host_bytes": host,
            "buffers": buffers}


def analyze(tokens, base=2048, long_chunk=8192, context=262144, tail_limit=256):
    plan, cap = chunks(tokens, base, long_chunk, context)
    candidate = merge_tail(plan, tail_limit)
    candidate_cap = max([cap] + candidate)
    if candidate_cap > context:
        raise ValueError("candidate exceeds context")
    old, new = scratch(cap, context), scratch(candidate_cap, context)
    return {
        "prefill_tokens": tokens, "chunks": plan, "capacity": cap,
        "candidate_chunks": candidate, "candidate_capacity": candidate_cap,
        "chunk_passes_saved": len(plan)-len(candidate),
        "cancellation_polls_removed": len(plan)-len(candidate),
        "device_bytes": old["device_bytes"], "host_bytes": old["host_bytes"],
        "extra_device_bytes": new["device_bytes"]-old["device_bytes"],
        "extra_host_bytes": new["host_bytes"]-old["host_bytes"],
    }


def log_histogram(paths, base=2048, long_chunk=8192, context=262144, tail_limit=256):
    """Only ckpt=0 records identify a single Prefill call from cache summaries.

    A checkpoint's split positions are absent; its aggregate prefilled token
    count cannot decide whether a tail can be merged within either call, except
    when the entire aggregate is too short to contain any eligible call.
    """
    records = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            match = re.search(r"cache: prompt (\d+) reused (\d+) prefilled (\d+) ckpt ([01])", line)
            if match:
                prompt, reused, prefilled, checkpoint = map(int, match.groups())
                if checkpoint:
                    eligible = False if prefilled <= min(max(base, long_chunk), context) else None
                else:
                    eligible = bool(analyze(prefilled, base, long_chunk, context, tail_limit)["chunk_passes_saved"])
                records.append({"file": Path(path).name, "prompt": prompt,
                                "reused": reused, "prefilled": prefilled,
                                "checkpoint": bool(checkpoint), "eligible_single_call": eligible})
    return {"sample": "preserved benchmark lengths projected onto selected chunk settings; not production traffic",
            "records": records, "total": len(records),
            "checkpoint_records": sum(r["checkpoint"] for r in records),
            "checkpoint_split_unknown": sum(r["eligible_single_call"] is None for r in records),
            "known_eligible": sum(r["eligible_single_call"] is True for r in records),
            "known_ineligible": sum(r["eligible_single_call"] is False for r in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", nargs="+", type=int,
                        default=[2048, 2049, 2304, 4096, 4097, 8192, 8193, 8194, 8448, 16385])
    parser.add_argument("--base", type=int, default=2048)
    parser.add_argument("--long-chunk", type=int, default=8192)
    parser.add_argument("--context", type=int, default=262144)
    parser.add_argument("--tail-limit", type=int, default=256)
    parser.add_argument("--logs", nargs="*", default=[], help="cache-trace logs for a bounded prevalence check")
    parser.add_argument("--output", type=Path, help="archive the CPU result as JSON")
    parser.add_argument("--server", action="store_true",
                        help="interpret lengths as cold request tokens; hold back one for decode; no cache split")
    args = parser.parse_args()
    if args.tail_limit < 0 or any(n < (1 if args.server else 0) for n in args.lengths):
        parser.error("invalid lengths or tail limit")
    if args.server and any(n >= args.context for n in args.lengths):
        parser.error("server rejects requests at or beyond max context")
    try:
        rows = [analyze(n-int(args.server), args.base, args.long_chunk, args.context, args.tail_limit)
                | {"input_tokens": n} for n in args.lengths]
    except ValueError as err:
        parser.error(str(err))
    result = {"baseline_engine": "0810614", "shape": "documented Qwen Config",
                      "mode": "cold-server-no-checkpoint" if args.server else "direct-Prefill",
                      "base": args.base, "long_chunk": args.long_chunk,
                      "context": args.context, "tail_limit": args.tail_limit,
                      "timings_measured": False, "rows": rows}
    if args.logs:
        result["log_histogram"] = log_histogram(args.logs, args.base, args.long_chunk, args.context, args.tail_limit)
    output = json.dumps(result, indent=2)+"\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
