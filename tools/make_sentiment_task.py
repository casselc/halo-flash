#!/usr/bin/env python3
"""Synthetic review-sentiment task for the stage-1 trainer demonstration
(docs/TRAINING.md): a behavior/format task, answering with a letter code.

  Review: Honestly, the coffee was not delightful.
  Is this review positive or negative? Answer A for positive, B for negative.
  Answer: B

Sentences come from templates over subjects, intensifiers and sentiment
adjectives; a fifth are negated ("was not delightful" is negative). The
held-out set uses only adjectives that never appear in training, so a gain
on it has to come from the base model's own sense of the words, not from
memorizing them. Deterministic for a seed; stdlib only.

Usage: make_sentiment_task.py <out-dir> [--train 1200] [--heldout 300] [--seed 7]
Writes <out-dir>/train.jsonl, heldout.jsonl (LM head rows) and
train_cls.jsonl, heldout_cls.jsonl (classifier rows, label 0 = positive).
"""
import argparse
import json
import os
import random

POSITIVE = """delightful wonderful excellent superb fantastic charming impressive lovely brilliant outstanding
pleasant enjoyable marvelous terrific splendid delicious friendly flawless stunning remarkable amazing great
perfect fabulous exceptional heartwarming refreshing elegant cozy reliable gorgeous uplifting sublime
magnificent satisfying admirable inspiring exquisite spotless gripping""".split()
NEGATIVE = """awful terrible horrible dreadful disappointing mediocre bland rude filthy boring tedious annoying
unpleasant disgusting poor broken overpriced sloppy miserable painful lousy unbearable frustrating dull greasy
stale noisy cramped flimsy pathetic atrocious abysmal shoddy tasteless depressing clumsy unreliable lifeless
grim appalling""".split()
SUBJECTS = ["the food", "the service", "this movie", "the hotel room", "the book", "this phone", "the concert",
            "the coffee", "the staff", "the game", "the tour", "the apartment", "this laptop", "the museum",
            "the flight", "the pizza", "the show", "the class", "the headphones", "the restaurant"]
INTENS = ["", "", "really ", "quite ", "absolutely ", "rather ", "truly "]
TEMPLATES = ["{S} was {i}{a}.", "Honestly, {s} was {i}{a}.", "I thought {s} was {i}{a} overall.",
             "{S} turned out to be {i}{a}.", "Everyone agreed that {s} was {i}{a}.", "We found {s} {i}{a}.",
             "My verdict: {s} was {i}{a}.", "After a week, {s} still felt {i}{a}."]
NEG_TEMPLATES = ["{S} was not {a}.", "Honestly, {s} was not {a} at all.", "I would not call {s} {a}.",
                 "{S} was never {a}."]
QUESTION = "Is this review positive or negative? Answer A for positive, B for negative.\nAnswer:"


def sentence(rng, pos_words, neg_words):
    positive_word = rng.random() < 0.5
    adj = rng.choice(pos_words if positive_word else neg_words)
    s = rng.choice(SUBJECTS)
    if rng.random() < 0.2:
        t = rng.choice(NEG_TEMPLATES)
        text = t.format(S=s[0].upper() + s[1:], s=s, a=adj)
        positive = not positive_word
    else:
        t = rng.choice(TEMPLATES)
        text = t.format(S=s[0].upper() + s[1:], s=s, i=rng.choice(INTENS), a=adj)
        positive = positive_word
    return text, positive


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--train", type=int, default=1200)
    ap.add_argument("--heldout", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pos, neg = POSITIVE[:], NEGATIVE[:]
    rng.shuffle(pos)
    rng.shuffle(neg)
    cut_p, cut_n = len(pos) * 3 // 4, len(neg) * 3 // 4
    splits = {"train": (pos[:cut_p], neg[:cut_n], a.train), "heldout": (pos[cut_p:], neg[cut_n:], a.heldout)}
    os.makedirs(a.out, exist_ok=True)
    meta = {"seed": a.seed, "question": QUESTION}
    for name, (p, n, count) in splits.items():
        lm, cls = [], []
        for _ in range(count):
            text, positive = sentence(rng, p, n)
            prompt = f"Review: {text}\n{QUESTION}"
            lm.append({"prompt": prompt, "completion": " A" if positive else " B"})
            cls.append({"text": prompt, "label": 0 if positive else 1})
        write(os.path.join(a.out, f"{name}.jsonl"), lm)
        write(os.path.join(a.out, f"{name}_cls.jsonl"), cls)
        meta[name] = {"examples": count, "positive_adjectives": p, "negative_adjectives": n,
                      "positive_fraction": sum(r["label"] == 0 for r in cls) / count}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(json.dumps({k: v if not isinstance(v, dict) else v["examples"] for k, v in meta.items()}))


if __name__ == "__main__":
    main()
