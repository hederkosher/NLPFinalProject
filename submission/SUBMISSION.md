# תוצרי ההגשה - Submission contents

NLP Final Project (Option A): full fine-tuning vs. LoRA on `google/flan-t5-small`,
compared on classification, translation and QA.
Ronen Shershnev (322217175) · Vladislav Pavlyuk (332294891)

| Required deliverable | File(s) in this folder |
|---|---|
| קבצי קוד Python מלאים | `data_prep.py`, `train.py`, `eval_model.py`, `compile_results.py`, `make_report.py`, `common.py` |
| README - הוראות התקנה והרצה, כולל גרסאות ספריות | `README.md` (+ pinned versions in `requirements.txt`) |
| קובץ תוצאות CSV/JSON/XLSX עם כל המדדים | `results/results.xlsx` (sheets = spec tables 1–6), `results/results.json` |
| דוח PDF/DOCX באורך 4–6 עמודים | `final_report.docx` (5 pages, tables 1–6 + analysis sections 5.1–5.5) |
| לפחות 5 דוגמאות פלט איכותניות בטבלה | Table 6 in `final_report.docx`; sheet `6_qualitative` in `results.xlsx` |

## Also included (supporting material)

- `data/` - the fixed splits (seed 42) used identically by both models: 1,000 train / 200 val /
  200 test per task, plus `stats.json` with length statistics.
- `results/train_{full,lora}.json` - training config, wall-clock time, trainable parameter counts,
  val-loss per epoch.
- `results/eval_{full,lora}.json` - all test metrics + inference timings per task.
- `results/preds_{full,lora}_{classification,translation,qa}.jsonl` - every one of the 600 test
  predictions per model (input, gold, prediction), which the report's qualitative analysis cites.

## Not included

- `checkpoints/` - the final checkpoints (Model A ≈ 295.9 MB, Model B adapter ≈ 5.0 MB) were saved
  during training and their sizes are reported in Table 1, but they are not retained in this
  repository (excluded via `.gitignore` for size). They are not part of the required deliverables
  list; re-running `python train.py --method full|lora` regenerates them under `checkpoints/`.
- Smoke-test outputs (`*_smoke.*`) - development artifacts of the pipeline's `--smoke` mode, not
  results of the graded runs.

Run order and environment setup are documented in `README.md`.
