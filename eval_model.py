"""Evaluate a trained checkpoint on the three test sets with all spec metrics.

Usage: python eval_model.py --method full|lora [--smoke]
Writes results/eval_{method}.json and results/preds_{method}_{task}.jsonl.
"""
import argparse
import json
import re
import string
import time
from collections import Counter

import torch
from sacrebleu.metrics import BLEU, CHRF
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common import (DATA, MAX_INPUT_TOKENS, MAX_NEW_TOKENS, RESULTS, TASKS,
                    ckpt_dir, load_jsonl)

BATCH_SIZE = 16
CLS_LABELS = ["negative", "positive"]


def load_model(method, smoke):
    final = ckpt_dir(method, smoke) / "final"
    if method == "lora":
        from peft import AutoPeftModelForSeq2SeqLM
        model = AutoPeftModelForSeq2SeqLM.from_pretrained(str(final))
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(str(final))
    return model, AutoTokenizer.from_pretrained(str(final))


def generate(model, tokenizer, device, inputs, max_new_tokens):
    preds = []
    for i in range(0, len(inputs), BATCH_SIZE):
        enc = tokenizer(inputs[i:i + BATCH_SIZE], return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_INPUT_TOKENS).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 num_beams=1, do_sample=False)
        preds += tokenizer.batch_decode(out, skip_special_tokens=True)
    return preds


# --- classification ---

def cls_metrics(rows, preds):
    norm = [p.strip().lower().strip(string.punctuation + " ") for p in preds]
    norm = [p if p in CLS_LABELS else "invalid" for p in norm]
    gold = [r["target"] for r in rows]
    kw = dict(labels=CLS_LABELS, average="macro", zero_division=0)
    return {
        "accuracy": round(accuracy_score(gold, norm), 4),
        "macro_f1": round(f1_score(gold, norm, **kw), 4),
        "precision_macro": round(precision_score(gold, norm, **kw), 4),
        "recall_macro": round(recall_score(gold, norm, **kw), 4),
        "invalid_rate": round(norm.count("invalid") / len(norm), 4),
    }


# --- translation ---

def mt_metrics(rows, preds):
    refs = [[r["target"] for r in rows]]
    return {
        "bleu": round(BLEU().corpus_score(preds, refs).score, 2),
        "chrf": round(CHRF().corpus_score(preds, refs).score, 2),
    }


# --- QA: standard SQuAD normalization / EM / token F1 ---

def normalize_answer(s):
    s = "".join(ch for ch in s.lower() if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred, ref):
    p, r = normalize_answer(pred).split(), normalize_answer(ref).split()
    overlap = sum((Counter(p) & Counter(r)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)


def qa_metrics(rows, preds):
    em, f1 = [], []
    for r, p in zip(rows, preds):
        em.append(max(float(normalize_answer(p) == normalize_answer(g)) for g in r["refs"]))
        f1.append(max(token_f1(p, g) for g in r["refs"]))
    gold_words = [len(r["target"].split()) for r in rows]
    return {
        "exact_match": round(sum(em) / len(em), 4),
        "token_f1": round(sum(f1) / len(f1), 4),
        "avg_gold_words": round(sum(gold_words) / len(gold_words), 2),
    }


TASK_METRICS = {"classification": cls_metrics, "translation": mt_metrics, "qa": qa_metrics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["full", "lora"], required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    model, tokenizer = load_model(args.method, args.smoke)
    model.to(device).eval()
    # untimed warm-up so MPS kernel compilation doesn't bias the first task's timing
    generate(model, tokenizer, device, ["translate English to French: warm up"] * 4, 8)

    RESULTS.mkdir(exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    per_task, total_sec, total_n = {}, 0.0, 0
    for task in TASKS:
        rows = load_jsonl(DATA / f"{task}_test.jsonl")
        if args.smoke:
            rows = rows[:8]
        inputs = [r["input"] for r in rows]

        t0 = time.perf_counter()
        preds = generate(model, tokenizer, device, inputs, MAX_NEW_TOKENS[task])
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - t0

        out_words = [len(p.split()) for p in preds]
        metrics = {
            "n": len(rows),
            **TASK_METRICS[task](rows, preds),
            "avg_output_words": round(sum(out_words) / len(out_words), 2),
            "inference_sec": round(elapsed, 1),
            "sec_per_100": round(elapsed / len(rows) * 100, 1),
        }
        per_task[task] = metrics
        total_sec += elapsed
        total_n += len(rows)

        with open(RESULTS / f"preds_{args.method}_{task}{suffix}.jsonl", "w") as f:
            for r, p in zip(rows, preds):
                f.write(json.dumps({"input": r["input"], "target": r["target"],
                                    "refs": r["refs"], "pred": p}, ensure_ascii=False) + "\n")
        print(f"{task}: {metrics}")

    report = {
        "method": args.method, "smoke": args.smoke, "device": device.type,
        "generation": {"strategy": "greedy", "batch_size": BATCH_SIZE,
                       "max_new_tokens": MAX_NEW_TOKENS},
        "per_task": per_task,
        "inference_sec_per_100_overall": round(total_sec / total_n * 100, 1),
    }
    with open(RESULTS / f"eval_{args.method}{suffix}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
