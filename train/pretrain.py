"""Pretraining loop for the four arms.

    accelerate launch train/pretrain.py config=configs/nested-learned.yaml
    accelerate launch train/pretrain.py config=configs/nested-learned.yaml max_steps=100

Arguments are ``key=value``. A dotted key sets that path (``optim.lr=1e-4``); a
bare key is matched against the config sections and must be unique.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer

import nested_gdn2  # registers the architecture for save_pretrained  # noqa: F401
from nested_gdn2.models.configuration_nested_gdn2 import NestedGDN2Config
from nested_gdn2.models.modeling_nested_gdn2 import NestedGDN2ForCausalLM
from train.data import TokenShards, train_stream, val_batches

CLI_ONLY = {"config", "run_name", "gcs_dest", "resume_from", "out_dir", "data_dir"}


def _read(path: Path) -> dict:
    """Resolve `extends` chains, nearest file winning leaf by leaf."""
    cfg = yaml.safe_load(path.read_text())
    parent = cfg.pop("extends", None)
    if not parent:
        return cfg
    base = _read(path.parent / parent)
    for section, leaves in cfg.items():
        base.setdefault(section, {}).update(leaves)
    return base


def load_config(argv: list[str]) -> tuple[dict, dict]:
    args = {}
    for a in argv:
        if "=" not in a:
            raise SystemExit(f"arguments are key=value, got {a!r}")
        k, v = a.split("=", 1)
        args[k] = v

    path = Path(args.pop("config", "configs/nested-learned.yaml"))
    cfg = _read(path)

    cli = {k: args.pop(k) for k in list(args) if k in CLI_ONLY}
    for k, v in args.items():
        v = yaml.safe_load(v)
        if "." in k:
            section, leaf = k.split(".", 1)
            cfg[section][leaf] = v
            continue
        hits = [s for s, leaves in cfg.items() if k in leaves]
        if len(hits) != 1:
            raise SystemExit(f"{k!r} matches {len(hits)} sections; qualify it as section.{k}")
        cfg[hits[0]][k] = v
    return cfg, cli


def param_groups(model, weight_decay: float):
    """Decay matrices only. Norms, biases and the gate parameters stay out."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 and not getattr(p, "_no_weight_decay", False) else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_at(step: int, total: int, opt: dict) -> float:
    warmup = opt["warmup_steps"]
    if step < warmup:
        return opt["lr"] * (step + 1) / warmup
    t = (step - warmup) / max(total - warmup, 1)
    return opt["min_lr"] + 0.5 * (opt["lr"] - opt["min_lr"]) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def validate(model, shards, seq_len, micro_batch, acc, n_batches):
    model.eval()
    total = torch.zeros((), device=acc.device)
    n = 0
    for x in val_batches(shards, seq_len, micro_batch, acc.process_index,
                         acc.num_processes, n_batches):
        x = x.to(acc.device, non_blocking=True)
        total += model(input_ids=x, labels=x, use_cache=False).loss
        n += 1
    model.train()
    return (acc.reduce(total, "mean") / max(n, 1)).item()


def main():
    if not torch.cuda.is_available():
        raise SystemExit("pretraining needs CUDA")
    cfg, cli = load_config(sys.argv[1:])
    mcfg, dcfg, ocfg, rcfg = cfg["model"], cfg["data"], cfg["optim"], cfg["run"]

    if data_dir := cli.get("data_dir"):
        dcfg["dir"] = data_dir

    seq_len, micro_batch = dcfg["seq_len"], ocfg["micro_batch"]
    world = int(os.environ.get("WORLD_SIZE", 1))
    per_slot = micro_batch * seq_len * world
    if ocfg["tokens_per_step"] % per_slot:
        raise SystemExit(
            f"tokens_per_step {ocfg['tokens_per_step']} is not a multiple of "
            f"{micro_batch} x {seq_len} x {world} = {per_slot}"
        )
    accum = ocfg["tokens_per_step"] // per_slot
    acc = Accelerator(gradient_accumulation_steps=accum)
    assert acc.num_processes == world, f"WORLD_SIZE {world} but {acc.num_processes} processes"

    total_steps = rcfg.get("max_steps") or rcfg["tokens"] // ocfg["tokens_per_step"]
    set_seed(rcfg["seed"])

    run_name = cli.get("run_name", Path(cli.get("config", "run")).stem)
    out_dir = Path(cli.get("out_dir", "outputs")) / "checkpoints"

    model = NestedGDN2ForCausalLM(NestedGDN2Config(**mcfg))
    if rcfg.get("compile"):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        param_groups(model, ocfg["weight_decay"]), lr=ocfg["lr"], betas=tuple(ocfg["betas"])
    )
    model, optimizer = acc.prepare(model, optimizer)

    step = 0
    if resume := cli.get("resume_from"):
        acc.load_state(resume)
        step = json.loads((Path(resume) / "progress.json").read_text())["step"]
        acc.print(f"resumed {resume} at step {step}")

    train = TokenShards(dcfg["dir"], "train")
    val = TokenShards(dcfg["dir"], "val")
    stream = train_stream(train, seq_len, micro_batch, acc.process_index, world,
                          rcfg["seed"], start_slot=step * accum)

    if acc.is_main_process:
        import wandb
        wandb.init(project=rcfg["wandb_project"], name=run_name, config=cfg)
        params = sum(p.numel() for p in model.parameters())
        acc.print(f"{run_name}: {params/1e6:.1f}M params, {total_steps} steps, "
                  f"{accum} accum x {micro_batch} x {seq_len} x {world} GPUs, "
                  f"{train.tokens/1e9:.2f}B train tokens")

    model.train()
    t0 = time.time()
    while step < total_steps:
        lr = lr_at(step, total_steps, ocfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        for _ in range(accum):
            with acc.accumulate(model):
                x = next(stream).to(acc.device, non_blocking=True)
                loss = model(input_ids=x, labels=x, use_cache=False).loss
                acc.backward(loss)
                grad_norm = (
                    acc.clip_grad_norm_(model.parameters(), ocfg["grad_clip"])
                    if acc.sync_gradients else None
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % rcfg["log_every"] == 0 and acc.is_main_process:
            dt = time.time() - t0
            tps = rcfg["log_every"] * ocfg["tokens_per_step"] / dt
            wandb.log({
                "loss": loss.item(),
                "lr": lr,
                "grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
                "tokens_per_s": tps,
                "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                "tokens": step * ocfg["tokens_per_step"],
            }, step=step)
            acc.print(f"step {step:>6}  loss {loss.item():.4f}  lr {lr:.2e}  "
                      f"{tps/1e3:.1f}k tok/s  {torch.cuda.max_memory_allocated()/2**30:.1f} GB")
            t0 = time.time()

        if step % rcfg["val_every"] == 0 or step == total_steps:
            vloss = validate(model, val, seq_len, micro_batch, acc, rcfg["val_batches"])
            if acc.is_main_process:
                wandb.log({"val_loss": vloss, "val_ppl": math.exp(min(vloss, 20))}, step=step)
                acc.print(f"step {step:>6}  val {vloss:.4f}")
            t0 = time.time()

        if step % rcfg["save_every"] == 0 or step == total_steps:
            path = out_dir / f"step_{step:06d}"
            acc.save_state(str(path))
            if acc.is_main_process:
                (path / "progress.json").write_text(json.dumps({"step": step, "run": run_name}))
                # The final checkpoint also goes out in transformers format, so
                # evaluation loads it directly instead of reconstructing the
                # config from a YAML that may have moved on since.
                if step == total_steps:
                    hf = path / "hf"
                    acc.unwrap_model(model).save_pretrained(hf)
                    AutoTokenizer.from_pretrained(train.tokenizer).save_pretrained(hf)
                    acc.print(f"saved {hf} (transformers format)")
                acc.print(f"saved {path}")
            t0 = time.time()

    if acc.is_main_process:
        wandb.finish()
    acc.end_training()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
