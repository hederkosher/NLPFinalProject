"""Shared constants and helpers for the train/eval/compile scripts."""
import json
from pathlib import Path

BASE_MODEL = "google/flan-t5-small"
SEED = 42
TASKS = ["classification", "translation", "qa"]
MAX_INPUT_TOKENS = 256
MAX_TARGET_TOKENS = 64
MAX_NEW_TOKENS = {"classification": 5, "translation": 64, "qa": 32}

DATA = Path("data")
RESULTS = Path("results")
CHECKPOINTS = Path("checkpoints")


def ckpt_dir(method: str, smoke: bool = False) -> Path:
    return CHECKPOINTS / (f"smoke_{method}" if smoke else method)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]
