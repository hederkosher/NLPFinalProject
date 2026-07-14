"""Build fixed train/val/test splits for the three tasks (seed 42).

Outputs data/{task}_{split}.jsonl with rows {task, input, target, refs},
plus combined data/train_all.jsonl and data/val_all.jsonl, and data/stats.json.

Targets are filtered to fit each task's generation cap (MAX_NEW_TOKENS), so
every reference is fully producible at eval time under the spec's output limits.
"""
import argparse
import json
import random

from datasets import load_dataset
from transformers import AutoTokenizer

from common import BASE_MODEL, DATA, MAX_INPUT_TOKENS, MAX_NEW_TOKENS, SEED

N_TRAIN, N_VAL, N_TEST = 1000, 200, 200

CLS_LABELS = {0: "negative", 1: "positive"}


def dump(rows, name):
    path = DATA / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def take(dataset, n, make_row, seen_keys=None, key_fn=None):
    """First n rows of a shuffled dataset that produce a row and aren't dupes."""
    rows = []
    for ex in dataset:
        row = make_row(ex)
        if row is None:
            continue
        if key_fn is not None:
            k = key_fn(row)
            if k in seen_keys:
                continue
            seen_keys.add(k)
        rows.append(row)
        if len(rows) == n:
            return rows
    raise RuntimeError(f"ran out of data: got {len(rows)}/{n}")


def prep_classification():
    ds = load_dataset("stanfordnlp/sst2")

    def make_row(ex):
        target = CLS_LABELS[ex["label"]]
        return {"task": "classification",
                "input": f"classify sentiment: {ex['sentence'].strip()}",
                "target": target, "refs": [target]}

    seen = set()
    key = lambda r: r["input"]
    pool = take(ds["train"].shuffle(seed=SEED), N_TRAIN + N_VAL, make_row, seen, key)
    test = take(ds["validation"].shuffle(seed=SEED), N_TEST, make_row, seen, key)
    return pool[:N_TRAIN], pool[N_TRAIN:], test


def prep_translation(tokenizer):
    ds = load_dataset("Helsinki-NLP/opus_books", "en-fr")["train"].shuffle(seed=SEED)
    cap = MAX_NEW_TOKENS["translation"]

    def make_row(ex):
        en, fr = ex["translation"]["en"].strip(), ex["translation"]["fr"].strip()
        if not (3 <= len(en.split()) <= 30 and 3 <= len(fr.split()) <= 30):
            return None
        if len(tokenizer(fr).input_ids) > cap:
            return None
        return {"task": "translation",
                "input": f"translate English to French: {en}",
                "target": fr, "refs": [fr]}

    seen = set()
    rows = take(ds, N_TRAIN + N_VAL + N_TEST, make_row, seen, lambda r: r["input"])
    return rows[:N_TRAIN], rows[N_TRAIN:N_TRAIN + N_VAL], rows[N_TRAIN + N_VAL:]


def prep_qa(tokenizer):
    ds = load_dataset("rajpurkar/squad")

    def make_row(ex):
        prompt = f"question: {ex['question'].strip()} context: {ex['context'].strip()}"
        if len(tokenizer(prompt).input_ids) > MAX_INPUT_TOKENS:
            return None
        answers = ex["answers"]["text"]
        if len(tokenizer(answers[0]).input_ids) > MAX_NEW_TOKENS["qa"]:
            return None
        return {"task": "qa", "input": prompt,
                "target": answers[0], "refs": list(dict.fromkeys(answers))}

    seen = set()
    key = lambda r: r["input"]
    pool = take(ds["train"].shuffle(seed=SEED), N_TRAIN + N_VAL, make_row, seen, key)
    test = take(ds["validation"].shuffle(seed=SEED), N_TEST, make_row, seen, key)
    return pool[:N_TRAIN], pool[N_TRAIN:], test


def main():
    # no flags on purpose (rejects e.g. --smoke): this script always builds the
    # full deterministic splits that train/eval --smoke sample from
    argparse.ArgumentParser(description=__doc__).parse_args()
    DATA.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    splits = {}
    for task, prep in [("classification", prep_classification),
                       ("translation", lambda: prep_translation(tokenizer)),
                       ("qa", lambda: prep_qa(tokenizer))]:
        train, val, test = prep()
        splits[task] = {"train": train, "val": val, "test": test}
        for split, rows in splits[task].items():
            dump(rows, f"{task}_{split}")

    rng = random.Random(SEED)
    for split in ("train", "val"):
        combined = [r for task in splits for r in splits[task][split]]
        rng.shuffle(combined)
        dump(combined, f"{split}_all")

    stats = {}
    print(f"\n{'task':<15} {'split':<6} {'n':>5} {'input words avg/max':>20} {'target words avg/max':>21}")
    for task, per_split in splits.items():
        for split, rows in per_split.items():
            in_w = [len(r["input"].split()) for r in rows]
            tg_w = [len(r["target"].split()) for r in rows]
            s = {"n": len(rows),
                 "input_words_avg": round(sum(in_w) / len(in_w), 1), "input_words_max": max(in_w),
                 "target_words_avg": round(sum(tg_w) / len(tg_w), 1), "target_words_max": max(tg_w)}
            stats[f"{task}_{split}"] = s
            print(f"{task:<15} {split:<6} {s['n']:>5} "
                  f"{s['input_words_avg']:>13} / {s['input_words_max']:<4} "
                  f"{s['target_words_avg']:>14} / {s['target_words_max']:<4}")

    with open(DATA / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    for task, per_split in splits.items():
        print(f"\n--- {task} samples ---")
        for r in per_split["train"][:2]:
            print(f"  INPUT : {r['input'][:160]}")
            print(f"  TARGET: {r['target']}")


if __name__ == "__main__":
    main()
