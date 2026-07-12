# Full Fine-tuning vs LoRA on flan-t5-small — Classification, Translation, QA

NLP final project: two models trained from the same base (`google/flan-t5-small`, ~77M params)
and compared on three tasks in one unified text-to-text format.

- **Model A** — full fine-tuning (all parameters updated)
- **Model B** — LoRA (PEFT): r=16, alpha=32, dropout=0.05, adapters on the `q` and `v` attention projections; only the adapters train

Each model is a single **multi-task** model: one training run over a shuffled mix of all three
tasks, distinguished by task prefixes. Both models use the exact same prompts, the same fixed
data splits, the same tokenization limits, and the same evaluation code.

## Unified prompt format

| Task | Input prompt | Target |
|---|---|---|
| Classification (SST-2) | `classify sentiment: {sentence}` | `positive` / `negative` |
| Translation (OPUS Books) | `translate English to French: {english}` | French sentence |
| QA (SQuAD v1.1) | `question: {question} context: {context}` | answer span text |

## Setup

Requires Python 3.12 (tested on macOS, Apple M2, torch MPS backend; CUDA/CPU also work —
device is auto-detected).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key pinned versions: `torch==2.13.0`, `transformers==5.13.1`, `peft==0.19.1`,
`datasets==5.0.0`, `sacrebleu==2.6.0`, `scikit-learn==1.9.0` (full list in `requirements.txt`).

## Run order

```bash
python data_prep.py               # downloads datasets, builds fixed splits into data/
python train.py --method full     # Model A -> checkpoints/full/final
python eval_model.py --method full
python train.py --method lora     # Model B -> checkpoints/lora/final
python eval_model.py --method lora
python compile_results.py         # -> results/results.xlsx + results/results.json
python make_report.py             # -> final_report.docx (full report, tables 1-6 + analysis)
```

`train.py`, `eval_model.py`, and `compile_results.py` also accept `--smoke` (tiny subsets,
1 epoch, separate `smoke_*`/`*_smoke` output paths that never touch real outputs) to verify the
pipeline end-to-end in under a minute. Smoke runs sample from the real `data/` splits, so run
`data_prep.py` once first — it is deterministic (seed 42) and takes ~2 minutes; it takes no
flags and rejects unrecognized ones.

## Data

Fixed splits, seed 42, identical for both models, saved as JSONL under `data/`:

| Task | Source (HF dataset) | Train | Val | Test |
|---|---|---|---|---|
| Classification | `stanfordnlp/sst2` (train split; test from its labeled validation split) | 1,000 | 200 | 200 |
| Translation en→fr | `Helsinki-NLP/opus_books` `en-fr`, pairs filtered to 3–30 words per side | 1,000 | 200 | 200 |
| QA | `rajpurkar/squad` (train split; test from its validation split, multi-reference answers kept) | 1,000 | 200 | 200 |

Documented data choices:
- **QA prompts are filtered to ≤256 T5 tokens** so the gold answer is always inside the visible
  context (the spec's allowed max input is 128–256 tokens; without the filter, truncation would
  silently delete answers).
- **Targets are filtered to fit each task's generation cap** (≤64 T5 tokens for translation,
  ≤32 for QA — the spec's max output lengths). Without this, ~1% of references would be
  impossible to fully generate at eval time and would train on truncated labels.
- Examples are deduplicated by input string within and across splits (OPUS Books repeats short
  lines; this prevents train/test leakage).
- SST-2's published train split contains phrase fragments while its validation split (our test
  source) has full sentences — inherent to the dataset, used as-is.

## Training configuration (identical unless noted)

| Setting | Model A (Full FT) | Model B (LoRA) |
|---|---|---|
| Epochs | 3 | 3 |
| Batch size | 8 | 8 |
| Learning rate | 5e-5 | 5e-4 |
| Max input / target length | 256 / 64 tokens | same |
| Precision | fp32 (T5 is numerically unstable in fp16) | same |
| Seed | 42 | same |
| Checkpointing | eval + save each epoch, best val-loss checkpoint restored at end | same |

## Evaluation

Greedy decoding, batch 16, per-task generation caps per the spec: 5 tokens (classification),
64 (translation), 32 (QA). One untimed warm-up batch precedes timing so GPU kernel compilation
doesn't bias the first task. Metrics:

- Classification: Accuracy, Macro-F1, Macro precision/recall (scikit-learn). Outputs that are
  not exactly a label after lowercase/punctuation strip count as an `invalid` bucket
  (reported as a rate; invalid predictions score as wrong).
- Translation: corpus BLEU and chrF (sacrebleu).
- QA: Exact Match and token-level F1 with the standard SQuAD normalization (lowercase, strip
  punctuation/articles), taking the max over all gold reference answers.
- Cost: training wall-clock time, inference seconds per 100 examples, trainable parameter
  count, checkpoint size on disk.

Normalized overall score (spec formula; BLEU is already on a 0–100 scale):

```
overall = (100 * Macro_F1 + BLEU + 100 * QA_token_F1) / 3
```

## Outputs

- `checkpoints/{full,lora}/final/` — final model / adapter + tokenizer
- `results/train_{method}.json` — training config, times, parameter counts, val-loss curve
- `results/eval_{method}.json` — all metrics per task + timing
- `results/preds_{method}_{task}.jsonl` — every test prediction (input, gold, prediction)
- `results/results.xlsx` — sheets mirroring the spec's tables 1–6 (model details,
  per-task results with B−A delta rows, normalized scores + cost, qualitative examples)
- `results/results.json` — everything above consolidated
- `final_report.docx` — the submission report (generated by `make_report.py` from the files above)
