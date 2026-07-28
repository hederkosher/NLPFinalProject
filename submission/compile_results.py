"""Merge train/eval outputs of both models into results/results.xlsx
(sheets mirror the spec's tables 1-6) and consolidated results/results.json.

Usage: python compile_results.py [--smoke]
"""
import argparse
import json
import string

import pandas as pd
from sacrebleu.metrics import CHRF

from common import RESULTS, TASKS, load_jsonl
from eval_model import CLS_LABELS, normalize_answer, token_f1

METHODS = ["full", "lora"]
MODEL_NAME = {"full": "Model A (Full FT)", "lora": "Model B (LoRA)"}


def load(kind, method, suffix):
    with open(RESULTS / f"{kind}_{method}{suffix}.json", encoding="utf-8") as f:
        return json.load(f)


def delta_row(a, b, label="B-A"):
    out = {}
    for k, va in a.items():
        vb = b.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            out[k] = round(vb - va, 4)
        else:
            out[k] = ""
    out["Model"] = label
    return out


def cls_ok(pred, gold):
    p = pred.strip().lower().strip(string.punctuation + " ")
    return (p if p in CLS_LABELS else "invalid") == gold


def qa_f1(pred, refs):
    return max(token_f1(pred, g) for g in refs)


def sheet_model_details(train):
    fields = [
        ("Base model", lambda t: t["base_model"]),
        ("Training method", lambda t: "Full fine-tuning" if t["method"] == "full" else "LoRA (r=16, alpha=32, q+v)"),
        ("Trainable parameters", lambda t: f"{t['trainable_params']:,}"),
        ("Trainable %", lambda t: t["trainable_pct"]),
        ("Epochs", lambda t: t["epochs"]),
        ("Learning rate", lambda t: t["learning_rate"]),
        ("Batch size", lambda t: t["batch_size"]),
        ("Max input/output length", lambda t: f"{t['max_input_tokens']} / {t['max_target_tokens']} tokens"),
        ("Train examples per task", lambda t: t["train_samples"] // len(TASKS)),
        ("Training time (min)", lambda t: round(t["train_runtime_sec"] / 60, 1)),
        ("Val loss by epoch", lambda t: ", ".join(map(str, t["val_loss_by_epoch"]))),
        ("Checkpoint size (MB)", lambda t: t["checkpoint_size_mb"]),
        ("Hardware / runtime environment",
         lambda t: f"{t.get('hardware', 'unrecorded')}, device={t['device']}"),
    ]
    return pd.DataFrame(
        {"Field": [name for name, _ in fields]}
        | {MODEL_NAME[m]: [get(train[m]) for _, get in fields] for m in METHODS})


def task_sheet(evals, task, cols):
    rows = []
    for m in METHODS:
        r = {"Model": MODEL_NAME[m]}
        r.update({label: evals[m]["per_task"][task][key] for label, key in cols})
        rows.append(r)
    rows.append(delta_row(rows[0], rows[1]))
    return pd.DataFrame(rows)[["Model"] + [label for label, _ in cols]]


def worst_example(preds, score_fn, fmt):
    scored = sorted(((score_fn(r), r) for r in preds), key=lambda x: x[0])
    return fmt(scored[0][1])


def sheet_scores(train, evals):
    rows = []
    for m in METHODS:
        pt = evals[m]["per_task"]
        cls = round(100 * pt["classification"]["macro_f1"], 2)
        mt = round(pt["translation"]["bleu"], 2)
        qa = round(100 * pt["qa"]["token_f1"], 2)
        rows.append({
            "Model": MODEL_NAME[m],
            "Classification score": cls, "Translation score": mt, "QA score": qa,
            "Overall score": round((cls + mt + qa) / 3, 2),
            "Train time (min)": round(train[m]["train_runtime_sec"] / 60, 1),
            "Inference time / 100 examples (sec)": evals[m]["inference_sec_per_100_overall"],
        })
    rows.append(delta_row(rows[0], rows[1]))
    return pd.DataFrame(rows)


def sheet_qualitative(preds):
    """>=5 rows: one per task + a failure and a success, with A/B outputs."""
    chrf = CHRF()

    def pair(task):  # aligned (row, pred_A, pred_B) triples
        a, b = preds["full"][task], preds["lora"][task]
        return [(ra, ra["pred"], rb["pred"]) for ra, rb in zip(a, b)]

    def correct(task, row, pred):
        if task == "classification":
            return cls_ok(pred, row["target"])
        if task == "qa":
            return qa_f1(pred, row["refs"]) >= 0.99
        return chrf.sentence_score(pred, [row["target"]]).score >= 50

    picks, used = [], set()

    def add(kind, task, row, pa, pb, note):
        picks.append({"Row": kind, "Task": task, "Input": row["input"],
                      "Gold answer": row["target"], "Model A output": pa,
                      "Model B output": pb, "Note": note})
        used.add(row["input"])

    for task in TASKS:  # one representative example per task
        triples = pair(task)
        best = max(triples, key=lambda t: correct(task, t[0], t[1]) + correct(task, t[0], t[2]))
        row, pa, pb = best
        add(task.capitalize(), task, row, pa, pb,
            f"A {'correct' if correct(task, row, pa) else 'wrong'}, "
            f"B {'correct' if correct(task, row, pb) else 'wrong'}")

    # failure: both models wrong (search QA then translation then classification)
    for task in ["qa", "translation", "classification"]:
        fails = [(r, a, b) for r, a, b in pair(task)
                 if r["input"] not in used and not correct(task, r, a) and not correct(task, r, b)]
        if fails:
            row, pa, pb = fails[0]
            add("Failure case", task, row, pa, pb, "both models wrong")
            break

    # success: both models exactly right
    for task in ["qa", "classification", "translation"]:
        wins = [(r, a, b) for r, a, b in pair(task)
                if r["input"] not in used and correct(task, r, a) and correct(task, r, b)]
        if wins:
            row, pa, pb = wins[0]
            add("Success case", task, row, pa, pb, "both models correct")
            break

    return pd.DataFrame(picks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    suffix = "_smoke" if args.smoke else ""

    train = {m: load("train", m, suffix) for m in METHODS}
    evals = {m: load("eval", m, suffix) for m in METHODS}
    preds = {m: {t: load_jsonl(RESULTS / f"preds_{m}_{t}{suffix}.jsonl") for t in TASKS}
             for m in METHODS}

    chrf = CHRF()
    ERR_COL = "Error example (worst case)"
    mt_errors = {MODEL_NAME[m]: worst_example(
        preds[m]["translation"],
        lambda r: chrf.sentence_score(r["pred"], [r["target"]]).score,
        lambda r: f"{r['input'].removeprefix('translate English to French: ')} -> "
                  f"'{r['pred'].strip()}' (gold: '{r['target']}')") for m in METHODS}
    qa_errors = {MODEL_NAME[m]: worst_example(
        preds[m]["qa"],
        lambda r: qa_f1(r["pred"], r["refs"]),
        lambda r: f"{r['input'].split(' context: ')[0].removeprefix('question: ')} -> "
                  f"'{r['pred'].strip()}' (gold: '{r['target']}')") for m in METHODS}

    mt_cols = [("BLEU", "bleu"), ("chrF", "chrf"),
               ("Average output length (words)", "avg_output_words"),
               ("Inference sec / 100", "sec_per_100")]
    qa_cols = [("Exact Match", "exact_match"), ("Token F1", "token_f1"),
               ("Average answer length (words)", "avg_output_words"),
               ("Average gold length (words)", "avg_gold_words"),
               ("Inference sec / 100", "sec_per_100")]
    cls_cols = [("Accuracy", "accuracy"), ("Macro-F1", "macro_f1"),
                ("Precision Macro", "precision_macro"), ("Recall Macro", "recall_macro"),
                ("Invalid output rate", "invalid_rate"),
                ("Inference sec / 100", "sec_per_100")]

    sheets = {
        "1_model_details": sheet_model_details(train),
        "2_classification": task_sheet(evals, "classification", cls_cols),
        "3_translation": task_sheet(evals, "translation", mt_cols),
        "4_qa": task_sheet(evals, "qa", qa_cols),
        "5_scores_and_cost": sheet_scores(train, evals),
        "6_qualitative": sheet_qualitative(preds),
    }
    for name, errors in [("3_translation", mt_errors), ("4_qa", qa_errors)]:
        sheets[name][ERR_COL] = [errors.get(m, "") for m in sheets[name]["Model"]]

    xlsx = RESULTS / f"results{suffix}.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

    consolidated = {
        "formula": "overall = (100*macro_F1 + BLEU + 100*QA_token_F1) / 3",
        "train": train, "eval": evals,
        "scores": sheets["5_scores_and_cost"].to_dict(orient="records"),
    }
    with open(RESULTS / f"results{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2, ensure_ascii=False)

    for name, df in sheets.items():
        print(f"\n=== {name} ===")
        print(df.to_string(index=False, max_colwidth=60))
    print(f"\nWrote {xlsx} and results{suffix}.json")


if __name__ == "__main__":
    main()
