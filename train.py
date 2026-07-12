"""Multi-task training of flan-t5-small: --method full (fine-tune everything)
or --method lora (PEFT adapter). Same data, prompts, and settings otherwise.

Usage: python train.py --method full|lora [--smoke]
"""
import argparse
import json
import platform

import torch
from datasets import load_dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, set_seed)

from common import (BASE_MODEL, DATA, MAX_INPUT_TOKENS, MAX_TARGET_TOKENS,
                    RESULTS, SEED, ckpt_dir)

LEARNING_RATE = {"full": 5e-5, "lora": 5e-4}
BATCH_SIZE = 8
EPOCHS = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["full", "lora"], required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="32 examples, 1 epoch — pipeline check only")
    args = ap.parse_args()

    set_seed(SEED)  # before get_peft_model: LoRA adapter init draws from the RNG
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    if args.method == "lora":
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32,
            lora_dropout=0.05, target_modules=["q", "v"]))

    ds = load_dataset("json", data_files={"train": str(DATA / "train_all.jsonl"),
                                          "val": str(DATA / "val_all.jsonl")})
    if args.smoke:
        ds["train"] = ds["train"].select(range(32))
        ds["val"] = ds["val"].select(range(16))

    def tokenize(batch):
        enc = tokenizer(batch["input"], max_length=MAX_INPUT_TOKENS, truncation=True)
        enc["labels"] = tokenizer(text_target=batch["target"],
                                  max_length=MAX_TARGET_TOKENS, truncation=True)["input_ids"]
        return enc

    ds = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)

    out = ckpt_dir(args.method, args.smoke)
    epochs = 1 if args.smoke else EPOCHS
    targs = Seq2SeqTrainingArguments(
        output_dir=str(out / "runs"),
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=16,
        learning_rate=LEARNING_RATE[args.method],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=25,
        seed=SEED,
        report_to=[],
        label_names=["labels"],
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs,
        train_dataset=ds["train"], eval_dataset=ds["val"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
    )
    result = trainer.train()

    final = out / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    val_losses = [round(h["eval_loss"], 4)
                  for h in trainer.state.log_history if "eval_loss" in h]
    info = {
        "method": args.method, "base_model": BASE_MODEL, "smoke": args.smoke,
        "epochs": epochs, "learning_rate": LEARNING_RATE[args.method],
        "batch_size": BATCH_SIZE, "max_input_tokens": MAX_INPUT_TOKENS,
        "max_target_tokens": MAX_TARGET_TOKENS, "seed": SEED,
        "device": str(trainer.model.device),
        "hardware": f"{platform.system()} {platform.machine()}, "
                    f"python {platform.python_version()}, torch {torch.__version__}",
        "train_samples": len(ds["train"]), "val_samples": len(ds["val"]),
        "trainable_params": trainable, "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4),
        "train_runtime_sec": round(result.metrics["train_runtime"], 1),
        "final_train_loss": round(result.metrics["train_loss"], 4),
        "val_loss_by_epoch": val_losses,
        "checkpoint": str(final),
        "checkpoint_size_mb": round(sum(
            p.stat().st_size for p in final.rglob("*") if p.is_file()) / 2**20, 1),
    }
    RESULTS.mkdir(exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    with open(RESULTS / f"train_{args.method}{suffix}.json", "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
