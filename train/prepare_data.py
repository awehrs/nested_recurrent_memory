"""Tokenize FineWeb-Edu into uint16 shards, optionally uploading them to GCS.

Run once; every arm trains on the same shards in the same order.

    python -m train.prepare_data --tokens 2_000_000                  # local check
    python -m train.prepare_data --gcs gs://bucket/path/             # the real thing

Resumable: shards already written are kept, and the stream skips the documents
they consumed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

SHARD = "shard_{:05d}.bin"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=lambda s: int(float(s.replace("_", ""))),
                   default=3_200_000_000, help="total tokens to write")
    p.add_argument("--shard-tokens", type=int, default=100_000_000)
    p.add_argument("--out", type=Path, default=Path("data/fineweb-edu-gpt2"))
    p.add_argument("--gcs", default="", help="gs:// prefix to upload each finished shard to")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--subset", default="sample-10BT")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--batch", type=int, default=1000, help="documents per tokenizer call")
    return p.parse_args()


def upload(path: Path, gcs: str):
    subprocess.run(["gsutil", "-q", "cp", str(path), gcs.rstrip("/") + "/" + path.name],
                   check=True)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "meta.json"

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id
    if tok.vocab_size >= 2**16:
        raise ValueError(f"{args.tokenizer} has {tok.vocab_size} tokens; uint16 holds 65536")

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    shards = list(meta.get("shards", []))
    docs_done = meta.get("documents", 0)
    tokens_done = meta.get("tokens", 0)
    if shards:
        print(f"Resuming after {len(shards)} shards, {tokens_done:,} tokens, {docs_done:,} documents")

    stream = load_dataset(args.dataset, name=args.subset, split="train", streaming=True)
    if docs_done:
        stream = stream.skip(docs_done)

    buf: list[np.ndarray] = []
    buf_len = 0
    batch: list[str] = []
    start = time.time()

    def write_shard():
        nonlocal buf, buf_len, tokens_done
        take = args.shard_tokens
        flat = np.concatenate(buf)
        path = args.out / SHARD.format(len(shards))
        flat[:take].tofile(path)
        if args.gcs:
            upload(path, args.gcs)
        shards.append(path.name)
        tokens_done += take
        buf = [flat[take:]]
        buf_len = len(buf[0])
        meta.update(
            dataset=args.dataset, subset=args.subset, tokenizer=args.tokenizer,
            vocab_size=tok.vocab_size, eos_token_id=eos, shard_tokens=args.shard_tokens,
            shards=shards, tokens=tokens_done, documents=docs_done, dtype="uint16",
        )
        meta_path.write_text(json.dumps(meta, indent=2))
        if args.gcs:
            upload(meta_path, args.gcs)
        rate = tokens_done / max(time.time() - start, 1e-9)
        print(f"  {path.name}  {tokens_done:,} / {args.tokens:,} tokens  {rate/1e6:.2f}M tok/s")

    def flush(texts):
        nonlocal buf_len, docs_done
        for ids in tok(texts, add_special_tokens=False)["input_ids"]:
            ids.append(eos)
            buf.append(np.asarray(ids, dtype=np.uint16))
            buf_len += len(ids)
        docs_done += len(texts)

    for row in stream:
        batch.append(row["text"])
        if len(batch) < args.batch:
            continue
        flush(batch)
        batch = []
        while buf_len >= args.shard_tokens and tokens_done < args.tokens:
            write_shard()
        if tokens_done >= args.tokens:
            break

    if batch and tokens_done < args.tokens:
        flush(batch)
    if buf_len and tokens_done < args.tokens:
        # Short final shard: whatever is left.
        args.shard_tokens = min(args.shard_tokens, buf_len)
        write_shard()

    print(f"{tokens_done:,} tokens in {len(shards)} shards at {args.out}"
          + (f", uploaded to {args.gcs}" if args.gcs else ""))
    print(f"Validation shard: {shards[-1]}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    main()
