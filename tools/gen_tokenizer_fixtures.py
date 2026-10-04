#!/usr/bin/env python3
"""Generates differential fixtures for the tokenizer tests.

The references are independent of halo-flash's implementation:
  * NFC: HF tokenizers' own NFC normalizer (unicode-normalization-alignments,
    Unicode 9.0), which differs from current unicodedata for newer scripts.
  * Pre-tokenizer pieces: the `regex` module running the exact Split pattern
    from Qwen3.8-Flash-Next's tokenizer.json, applied to the NFC text.
  * Token ids: HF `tokenizers` loading that tokenizer.json,
    encode(text, add_special_tokens=False).

Usage: gen_tokenizer_fixtures.py <tokenizer.json> <out-dir> [llama-qwen35.inp]
       DECIDER_TOKENIZER=<strands-decider tokenizer.json> adds:
  pretok_qwen2.tsv  input_hex  pieces (the decider's Qwen2-style split regex)
  encode_decider.tsv input_hex  ids  char offsets "lo:hi ..." (HF offsets)

Output files (one case per line, fields tab-separated, strings hex-encoded):
  nfc.tsv        input_hex  nfc_hex
  pretok.tsv     input_hex  piece_hex,piece_hex,...   (input is already NFC)
  encode.tsv     input_hex  id id id ...
The first line of each file is a '#' header recording the reference versions.
"""
import json
import os
import random
import sys
import unicodedata

import regex
import tokenizers

SEED = 20261003
HF_NFC = tokenizers.normalizers.NFC()


def hx(s):
    return s.encode("utf-8").hex()


def random_corpus(rng, n):
    # An alphabet chosen to stress every regex branch and class boundary.
    pools = [
        list("abcXYZ"), list("'"), list("sStTrReEvVmMlLdD"), ["ſ"],  # long s folds to s
        list("0123456789"), ["٣", "²", "Ⅷ", "½"],     # Nd, No, Nl, No
        [" ", " ", " ", "\t", "\r", "\n", "\r\n", " ", "　", " ", "\u0085", " "],
        list(".,;:!?-_()[]{}<>/\\\"#$%&*+=@^`|~"),
        ["́", "̈", "̣", "̧", "ि", "ְ"],     # combining marks
        ["é", "Å", "각", "नि"],     # decomposed sequences
        ["é", "Å", "ñ", "ß", "Ω", "Ж", "щ", "ا", "ש", "क", "中", "文", "日本", "한국어", "ไทย"],
        ["🦙", "👍🏽", "‍", "️", "€", "©", "™"],
        ["<", "|", ">", "im_start", "think"],
        # Post-Unicode-9 combining marks and compositions (Unicode 10-16), where
        # HF's Unicode 9 NFC differs from current unicodedata.
        ["\u1ab0", "\u1dfa", "\u0d00", "\u0afa", "\U00011f00", "\U00011383", "\U000113c2\U000113b8",
         "\U00016d63\U00016d67", "\U0001e4ec", "\U00011f41", "a\u1dfa\u0301"],
    ]
    out = []
    for _ in range(n):
        k = rng.randint(1, 40)
        s = "".join(rng.choice(rng.choice(pools)) for _ in range(k))
        out.append(s)
    return out


HANDWRITTEN = [
    "", " ", "  ", "\n", "\n\n", " \n ", "\t\n\t", "a", "Hello world", " Hello world",
    "I'm here, you're there; we'll see. It's DON'T and SHE'LL", "'S 'T 'Re 'VE 'M 'LL 'D",
    "x'ſ", "ok ſ", "3.14159 1,000,000 ١٢٣ ²³ Ⅻ", "   leading", "trailing   ", "mid   dle",
    "a\r\nb\r\n\r\nc", "line1\n  line2\n\tline3", "!!!\n\n???", " ?!.\n", "foo.bar(baz)",
    "def f(x):\n    return x ** 2  # square\n", '{"key": [1, 2.5e-3, "v\\u00e9"]}',
    "é vs é", "Å vs Å", "한국어 각", "नि नि",
    "😀😀 🦙.cpp 👍🏽", "日本語のテキストと English mixed", "Привет, мир!", "مرحبا بالعالم",
    "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHello",
    "<tool_call>\n{\"name\": \"f\"}\n</tool_call>", "<|im_start|><|im_end|><|endoftext|>",
    "<|im_start", "im_end|>", "<<|im_end|>>", "a" * 300, " " * 50 + "x", "\n" * 20,
    "  word", "　full width", "tab\tsep\tvalues", "́́start with marks",
]


def main():
    tok_path, out_dir = sys.argv[1], sys.argv[2]
    llama_inp = sys.argv[3] if len(sys.argv) > 3 else None
    spec = json.load(open(tok_path, encoding="utf-8"))
    pattern = spec["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    assert spec["normalizer"]["type"] == "NFC"
    rx = regex.compile(pattern)
    tk = tokenizers.Tokenizer.from_file(tok_path)
    header = (f"# unicodedata {unicodedata.unidata_version} regex {regex.__version__} "
              f"tokenizers {tokenizers.__version__} seed {SEED}\n")

    rng = random.Random(SEED)
    corpus = list(HANDWRITTEN)
    if llama_inp:
        corpus += open(llama_inp, encoding="utf-8").read().split("\n__ggml_vocab_test__\n")
    corpus += random_corpus(rng, 3000)

    # NFC: every codepoint with a canonical decomposition (as NFD input),
    # plus the corpus.
    nfc_cases = []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        d = unicodedata.normalize("NFD", c)
        if d != c:
            nfc_cases.append(d)
    nfc_cases += corpus
    with open(f"{out_dir}/nfc.tsv", "w") as f:
        f.write(header)
        for s in nfc_cases:
            f.write(f"{hx(s)}\t{hx(HF_NFC.normalize_str(s))}\n")

    with open(f"{out_dir}/pretok.tsv", "w") as f:
        f.write(header)
        for s in corpus:
            n = HF_NFC.normalize_str(s)
            pieces = [m.group() for m in rx.finditer(n)]
            assert "".join(pieces) == n, repr(s)
            f.write(f"{hx(n)}\t{','.join(hx(p) for p in pieces)}\n")

    with open(f"{out_dir}/encode.tsv", "w") as f:
        f.write(header)
        for s in corpus:
            ids = tk.encode(s, add_special_tokens=False).ids
            f.write(f"{hx(s)}\t{' '.join(map(str, ids))}\n")
    dec = os.environ.get("DECIDER_TOKENIZER")
    if dec:
        dspec = json.load(open(dec, encoding="utf-8"))
        drx = regex.compile(dspec["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"])
        dtk = tokenizers.Tokenizer.from_file(dec)
        with open(f"{out_dir}/pretok_qwen2.tsv", "w") as f:
            f.write(header)
            for s in corpus:
                n = HF_NFC.normalize_str(s)
                pieces = [m.group() for m in drx.finditer(n)]
                assert "".join(pieces) == n
                f.write(f"{hx(n)}\t{','.join(hx(p) for p in pieces)}\n")
        with open(f"{out_dir}/encode_decider.tsv", "w") as f:
            f.write(header)
            for s in corpus:
                e = dtk.encode(s, add_special_tokens=False)
                offs = " ".join(f"{a}:{b}" for a, b in e.offsets)
                f.write(f"{hx(s)}\t{' '.join(map(str, e.ids))}\t{offs}\n")
    print(f"nfc {len(nfc_cases)} pretok/encode {len(corpus)}", file=sys.stderr)


if __name__ == "__main__":
    main()
