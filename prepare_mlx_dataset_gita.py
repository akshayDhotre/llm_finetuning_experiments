#!/usr/bin/env python3
"""
prepare_mlx_dataset.py

Download a Hugging Face dataset and produce MLX-compatible train/valid/test JSONL files.

Outputs:
    out_dir/train.jsonl
    out_dir/valid.jsonl
    out_dir/test.jsonl

MLX formats supported: "text", "completions", "chat"

Example:
    python prepare_mlx_dataset.py \
        --hf_id JDhruv14/Bhagavad-Gita_Dataset \
        --out_dir data/bhagavad_gita_mlx \
        --format completions \
        --train_frac 0.8 --valid_frac 0.1 --test_frac 0.1
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm import tqdm


# -----------------------
# Helpers
# -----------------------
def auto_pick_text_fields(ds) -> List[str]:
    """
    Inspect dataset column names and pick candidate text columns.
    Heuristic: prefer common names, else pick first string column.
    """
    cols = ds.column_names
    # Lower-case mapping
    lower = [c.lower() for c in cols]
    preferred = ["text", "sentence", "verse", "translation", "english", "sanskrit"]
    picks = []
    for p in preferred:
        for i, c in enumerate(cols):
            if p in c.lower() and c not in picks:
                picks.append(c)
    # add any other string columns
    for c in cols:
        # try to see if column dtype is string-compatible by sampling one value
        try:
            sample = ds[0][c]
            if isinstance(sample, str) and c not in picks:
                picks.append(c)
        except Exception:
            continue
    return picks


def to_mlx_record(
    format_type: str,
    record: Dict[str, Any],
    src_field: str = None,
    tgt_field: str = None,
) -> Dict[str, Any]:
    """
    Build MLX-format record depending on format_type:
      - text: { "text": "..."}
      - completions: { "prompt": "...", "completion": "..." }
      - chat: { "messages": [ {"role":"user","content":"..."} , {"role":"assistant","content":"..."} ] }
    If only one field exists, we treat it as raw text (text format) or prompt+completion with empty completion.
    """
    if format_type == "text":
        # choose tgt_field if present else src_field
        t = (
            record.get(tgt_field)
            if tgt_field and record.get(tgt_field)
            else record.get(src_field)
        )
        return {"text": t if t is not None else ""}
    elif format_type == "completions":
        prompt = record.get(src_field) or ""
        completion = record.get(tgt_field) or ""
        # Make sure completion often ends with newline or space for many training pipelines
        return {"prompt": f"What is sloka no. {prompt}?", "completion": completion}
    elif format_type == "chat":
        # simple chat format: user asks for translation/verse and assistant provides it
        user_content = record.get(src_field) or ""
        assistant_content = record.get(tgt_field) or ""
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        }
    else:
        raise ValueError("Unsupported format_type: " + format_type)


def write_jsonl(path: str, records: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_id",
        type=str,
        required=True,
        help="HuggingFace dataset id (e.g. JDhruv14/Bhagavad-Gita_Dataset)",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default=None,
        help="If dataset has splits (train/test/validation), you can specify a split to use (optional).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/mlx_dataset",
        help="Output directory for train/valid/test jsonl files",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "completions", "chat"],
        default="text",
        help="MLX dataset format",
    )
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--valid_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--src_field",
        type=str,
        default=None,
        help="Field to use as source/prompt (auto-detected if omitted)",
    )
    parser.add_argument(
        "--tgt_field",
        type=str,
        default=None,
        help="Field to use as target/completion (auto-detected if omitted)",
    )
    args = parser.parse_args()

    assert abs(args.train_frac + args.valid_frac + args.test_frac - 1.0) < 1e-6, (
        "Fractions must sum to 1"
    )

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading HF dataset: {args.hf_id} ...")
    ds = load_dataset(args.hf_id)
    # if dataset has multiple splits, optionally use a single split (e.g., 'train')
    # We'll concatenate all splits if split_name is None
    if args.split_name:
        if args.split_name not in ds:
            raise ValueError(
                f"Split {args.split_name} not found in dataset splits: {list(ds.keys())}"
            )
        raw = ds[args.split_name]
    else:
        # concatenate all splits into a single list
        if isinstance(ds, dict):
            # combine all splits
            concat = []
            for k in ds.keys():
                concat.append(ds[k])
            from datasets import concatenate_datasets

            raw = concatenate_datasets(concat)
        else:
            raw = ds

    print("Detected dataset columns:", raw.column_names)

    # auto-detect candidate text fields
    candidates = auto_pick_text_fields(raw)
    print("Auto-detected text-like fields (preferred order):", candidates)

    src_field = args.src_field
    tgt_field = args.tgt_field

    if not src_field:
        # If there are two useful fields -> assume first is source, second is target
        if len(candidates) >= 2:
            src_field = candidates[0]
            tgt_field = candidates[1]
            print(f"Selected src_field='{src_field}', tgt_field='{tgt_field}' (auto)")
        elif len(candidates) == 1:
            src_field = candidates[0]
            print(f"Selected single text field src_field='{src_field}' (auto)")
        else:
            raise ValueError(
                "Could not auto-detect any text field. Use --src_field to specify a column name."
            )

    # Build records list
    print("Building records...")
    all_records = []
    for i in tqdm(range(len(raw)), desc="reading"):
        item = raw[i]
        # normalize dict keys -> strings
        rec = {
            k: (v if not isinstance(v, bytes) else v.decode("utf-8"))
            for k, v in item.items()
        }
        mlx_rec = to_mlx_record(
            args.format, rec, src_field=src_field, tgt_field=tgt_field
        )
        all_records.append(mlx_rec)

    # Shuffle & split
    random.seed(args.seed)
    random.shuffle(all_records)
    n = len(all_records)
    n_train = int(n * args.train_frac)
    n_valid = int(n * args.valid_frac)
    n_test = n - n_train - n_valid

    train = all_records[:n_train]
    valid = all_records[n_train : n_train + n_valid]
    test = all_records[n_train + n_valid :]

    print(f"Total: {n}  -> train: {len(train)}, valid: {len(valid)}, test: {len(test)}")

    # Write files
    train_path = os.path.join(args.out_dir, "train.jsonl")
    valid_path = os.path.join(args.out_dir, "valid.jsonl")
    test_path = os.path.join(args.out_dir, "test.jsonl")

    print("Writing JSONL files...")
    write_jsonl(train_path, train)
    write_jsonl(valid_path, valid)
    write_jsonl(test_path, test)

    print("Done.")
    print(f"Wrote: {train_path}, {valid_path}, {test_path}")


if __name__ == "__main__":
    main()
