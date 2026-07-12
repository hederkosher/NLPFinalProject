# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## State

Implemented as Python scripts (see README.md for full details). `NLP_Final_Project_2026_B.pdf` is the assignment spec (Hebrew); the sections below summarize it.

## Commands

```bash
source .venv/bin/activate            # Python 3.12 venv, deps pinned in requirements.txt
python data_prep.py                  # rebuild fixed splits (seed 42) into data/
python train.py --method full|lora   # train Model A / Model B -> checkpoints/{method}/final
python eval_model.py --method full|lora   # metrics + predictions -> results/
python compile_results.py            # results/results.xlsx (spec tables 1-6) + results.json
```

`train.py`, `eval_model.py`, and `compile_results.py` take `--smoke` (tiny data, separate `smoke_*`/`*_smoke` output paths) to verify the whole pipeline in under a minute without touching real outputs; smoke runs sample from the real `data/` splits, so `data_prep.py` (which takes no flags and rejects unknown ones) must have run once. Constants shared by all scripts (base model, seed, token limits, paths) live in `common.py`. `eval_model.py` is deliberately not named `evaluate.py` (would shadow imports). Do not re-run real training casually — results feed the user's report; `train_*.json`/`eval_*.json` timings would change.

## The assignment

Train **two** language models and compare them across **three** tasks: classification, translation, QA. The point is the comparison, not raw scores.

Recommended setup (Option A in the spec):
- Same base model for both, so the comparison is fair. Small seq2seq: `t5-small`, `flan-t5-small`, or `mt5-small`.
- **Model A**: full fine-tuning. **Model B**: LoRA / Adapter / PEFT.
- One multi-task model per training method is preferred over three per-task models.

Advanced alternative (Option B): Model A = SFT/instruction-tuning, Model B = an extension (RAG-QA, DPO on preference pairs, small MoE).

## Non-negotiable design constraints

**Unified text-to-text prompt format across both models.** All three tasks go through one model via task prefixes; a differing prompt format between models invalidates the comparison.

```
classify sentiment: This movie was surprisingly good.   -> positive
translate English to French: The child opened the door.  -> L'enfant a ouvert la porte.
question: Where did Mary go? context: Mary went to the kitchen after school. -> kitchen
```

**Fixed splits, shared by both models.** Per task: train 500–1,000 (300–500 acceptable if training is slow — report it), val 100–200, test 100–200. Never evaluate on training examples. Datasets: SST-2 / AG News / Emotion (classification), OPUS Books / Tatoeba / WMT subset (translation, en↔fr or en↔de), SQuAD subset / bAbI QA (QA).

**Hyperparameters** (recommended): 1–3 epochs, batch 4–16, LR 5e-5 for full FT and 5e-4 for LoRA/PEFT, max input 128–256 tokens, max output 5 (classification) / 64 (translation) / 32 (QA). Save a checkpoint per model. Measure train time and inference time.

## Metrics

Per-task, not one metric for all: Accuracy + Macro-F1 (classification), BLEU + chrF (translation), Exact Match + token-level F1 (QA). Also track train time, inference time, trainable parameter count.

Overall score normalizes each task to 0–100:
```
classification_score = 100 * Macro_F1
translation_score    = BLEU              # already 0–100, do not multiply
qa_score             = 100 * QA_F1
overall_score = (classification_score + translation_score + qa_score) / 3
```

## Deliverables

Python script or notebook; README with install + run instructions incl. library versions; results file (CSV/JSON/XLSX) with all metrics; 4–6 page report (PDF/DOCX) with tables 1–6 from the spec (model details, per-task results with a B−A delta row, overall score + compute cost, 5+ qualitative output examples incl. a failure and a success case). The report must analyze, not just tabulate: which method won on which task, whether one model was uniformly better, whether quality gains justified training time and parameter count, what each model failed at, and whether outputs were truncated, over-long, or hallucinated.
