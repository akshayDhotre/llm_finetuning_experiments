#!/usr/bin/env python3
"""
prepare_mlx_finance_dataset.py

Prepare MLX-compatible JSONL train/valid/test files from
Hugging Face dataset: Josephgflowers/Finance-Instruct-500k (streaming-friendly).

Features:
 - Uses streaming to handle large dataset without loading into memory.
 - Auto-detects common fields used in finance/instruction datasets:
   'system', 'user', 'assistant', 'instruction', 'input', 'output', 'response'
 - Supports output formats: 'completions' (prompt+completion) and 'chat' (messages).
 - Allows limiting total samples for quick tests (--max_samples).
 - Writes UTF-8 JSONL files: train.jsonl, valid.jsonl, test.jsonl

Example:
  python prepare_mlx_finance_dataset.py \
    --hf_id Josephgflowers/Finance-Instruct-500k \
    --out_dir ./data/finance_mlx \
    --format completions \
    --train_frac 0.9 --valid_frac 0.05 --test_frac 0.05 \
    --max_samples 20000
"""

import argparse
import json
import os
import random
from typing import Dict, Iterable, Optional

from datasets import load_dataset

COMMON_FIELDS = [
    "system",
    "user",
    "assistant",
    "instruction",
    "input",
    "output",
    "response",
    "prompt",
    "completion",
    "text",
]


def pick_fields_from_example(example: Dict) -> Dict[str, Optional[str]]:
    """
    Given a single example dict from the dataset, attempt to map onto:
      - src (prompt / user input)
      - tgt (assistant / output / completion)
    Strategy (priority):
      1. If 'user' and 'assistant' present -> user -> assistant
      2. If 'instruction' present, combine instruction + input (if any) -> output/response
      3. If 'prompt'/'completion' present -> prompt/completion
      4. If only 'text' present -> text as src (tgt empty)
      5. fallback: first string field -> treat as text
    Returns dict: {"src": str_or_none, "tgt": str_or_none}
    """

    # Ensure example is a dictionary
    if not isinstance(example, dict):
        return {"src": str(example), "tgt": ""}

    # helper to safely fetch string-like
    def sget(key):
        v = example.get(key)
        if v is None:
            return None
        if isinstance(v, list):
            # For lists, join them with double newlines
            try:
                return "\n\n".join([str(x) for x in v])
            except Exception:
                return None
        return str(v)

    # 1. user/assistant pair
    u = sget("user")
    a = sget("assistant")
    if u or a:
        return {"src": u or "", "tgt": a or ""}

    # 2. instruction [+ input] -> output/response
    instr = sget("instruction")
    inp = sget("input")
    out = sget("output") or sget("response")
    if instr:
        # combine instruction + input into a single prompt
        prompt = instr
        if inp:
            prompt = prompt.strip() + "\n\nInput:\n" + inp.strip()
        return {"src": prompt, "tgt": out or ""}

    # 3. prompt/completion
    prompt = sget("prompt")
    completion = sget("completion")
    if prompt or completion:
        return {"src": prompt or "", "tgt": completion or ""}

    # 4. text
    text = sget("text")
    if text:
        return {"src": text, "tgt": ""}

    # 5. fallback: any first string-like field
    for k, v in example.items():
        if isinstance(v, str):
            return {"src": v, "tgt": ""}
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str):
            return {"src": "\n\n".join(v), "tgt": ""}

    return {"src": None, "tgt": None}


def to_mlx_record(format_type: str, src: Optional[str], tgt: Optional[str]):
    if format_type == "completions":
        return {"prompt": src or "", "completion": tgt or ""}
    elif format_type == "chat":
        # represent as a two-message user->assistant exchange
        messages = []
        if src is not None:
            messages.append({"role": "user", "content": src})
        if tgt is not None and tgt != "":
            messages.append({"role": "assistant", "content": tgt})
        return {"messages": messages}
    elif format_type == "text":
        return {"text": src or ""}
    else:
        raise ValueError("Unsupported format_type: " + format_type)


def stream_dataset_examples(
    hf_id: str, split: Optional[str], streaming: bool = True
) -> Iterable[Dict]:
    """
    Yields examples from the Hugging Face dataset in streaming mode.
    If a split is provided and exists, stream that split, else stream the dataset (all).
    """
    if split:
        ds = load_dataset(hf_id, split=split, streaming=streaming)
    else:
        # if the dataset has default split 'train', load that in streaming mode
        ds = load_dataset(hf_id, streaming=streaming)
    return ds


def write_jsonl(path: str, records: Iterable[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_id",
        type=str,
        default="Josephgflowers/Finance-Instruct-500k",
        help="HuggingFace dataset id (default Josephgflowers/Finance-Instruct-500k)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="If dataset split to stream (e.g. 'train'). By default streams the dataset.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./data/finance_mlx",
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["completions", "chat", "text"],
        default="completions",
        help="MLX output format",
    )
    parser.add_argument("--train_frac", type=float, default=0.9)
    parser.add_argument("--valid_frac", type=float, default=0.05)
    parser.add_argument("--test_frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Total number of samples to process (useful for quick tests). If omitted, will stream entire dataset.",
    )
    parser.add_argument(
        "--shuffle_buffer",
        type=int,
        default=10000,
        help="In-memory shuffle buffer size (for streaming). Increase if you have RAM.",
    )
    args = parser.parse_args()

    assert abs(args.train_frac + args.valid_frac + args.test_frac - 1.0) < 1e-6, (
        "Fractions must sum to 1"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.jsonl")
    valid_path = os.path.join(args.out_dir, "valid.jsonl")
    test_path = os.path.join(args.out_dir, "test.jsonl")

    print(f"Streaming dataset {args.hf_id} (split={args.split}) ...")
    ds_iter = stream_dataset_examples(args.hf_id, args.split, streaming=True)

    # Use reservoir-like buffer for pseudo-shuffle while streaming
    buffer = []
    buf_size = args.shuffle_buffer
    random.seed(args.seed)

    out_train = []
    out_valid = []
    out_test = []

    total_processed = 0
    max_samples = args.max_samples

    # We'll produce records incrementally and write out periodically to avoid holding many records
    def flush_to_file(path, records):
        if not records:
            return
        with open(path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        records.clear()

    print(
        "Processing examples (this may take a while for full dataset). Use --max_samples for quick runs."
    )
    for example in ds_iter:
        total_processed += 1

        # pick src/tgt
        picked = pick_fields_from_example(example)
        src = picked.get("src")
        tgt = picked.get("tgt")

        # skip examples where we couldn't detect text
        if src is None and tgt is None:
            # ignore
            pass
        else:
            rec = to_mlx_record(args.format, src or "", tgt or "")
            # push to buffer
            buffer.append(rec)

        # maintain buffer
        if len(buffer) >= buf_size:
            # shuffle buffer and drain proportionally into train/valid/test
            random.shuffle(buffer)
            n = len(buffer)
            n_train = int(n * args.train_frac)
            n_valid = int(n * args.valid_frac)
            n_test = n - n_train - n_valid

            out_train.extend(buffer[:n_train])
            out_valid.extend(buffer[n_train : n_train + n_valid])
            out_test.extend(buffer[n_train + n_valid :])
            buffer.clear()

            # flush to disk if lists grow
            if len(out_train) >= 1000:
                flush_to_file(train_path, out_train)
            if len(out_valid) >= 500:
                flush_to_file(valid_path, out_valid)
            if len(out_test) >= 500:
                flush_to_file(test_path, out_test)

        # honor max_samples for quick tests
        if max_samples and total_processed >= max_samples:
            print(f"Reached max_samples={max_samples}. Stopping stream.")
            break

    # After streaming finished or stopped, handle remaining buffer
    if buffer:
        random.shuffle(buffer)
        n = len(buffer)
        n_train = int(n * args.train_frac)
        n_valid = int(n * args.valid_frac)
        n_test = n - n_train - n_valid
        out_train.extend(buffer[:n_train])
        out_valid.extend(buffer[n_train : n_train + n_valid])
        out_test.extend(buffer[n_train + n_valid :])
        buffer.clear()

    # Final flush
    flush_to_file(train_path, out_train)
    flush_to_file(valid_path, out_valid)
    flush_to_file(test_path, out_test)

    print("Done.")
    print("Wrote files (appended):")
    print(" -", train_path)
    print(" -", valid_path)
    print(" -", test_path)
    print(f"Total examples processed (streamed): {total_processed}")
    if args.max_samples:
        print(f"Wrote up to {args.max_samples} examples (according to fractions).")


if __name__ == "__main__":
    main()
