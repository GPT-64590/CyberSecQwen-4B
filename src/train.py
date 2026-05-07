"""v2 SFT training — adapted from train_cyber_lora_plain.py.

Differences from v0.1:
  - Default --per-device-batch-size=8 (was 4) — verified VRAM headroom on MI300X
  - Default --grad-accum=2 (was 4) — same effective batch size 16
  - Default --max-seq-length=3072 (was 4096) — verified p99 corpus = 2565
  - Default --gradient-checkpointing=False (was True) — recompute tax wasted with 192GB
  - --resume-adapter <path>: load existing PEFT adapter and CONTINUE training on it
    (used to stack SFT on top of CPT adapter for v2 multi-stage training)
  - --lora-r configurable for r=32 vs r=64 ablation (default 64)

Same target_modules regex as v0.1: language_model only.
Same chat-template format: <|im_start|>user/...end/...model/...end.
Frozen base, LoRA-only training.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_cyber_lora_v2")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v2 cyber LoRA SFT training.")
    p.add_argument("--data", type=Path, default=Path("/shared-docker/sft/combined_v2.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("/shared-docker/output/adapter/cyber_v2"))
    p.add_argument("--base-model", type=str, default="google/gemma-4-E2B-it")
    p.add_argument(
        "--resume-adapter",
        type=Path,
        default=None,
        help="path to existing PEFT adapter to load + continue training (e.g. CPT adapter)",
    )
    p.add_argument("--lora-r", type=int, default=64, help="LoRA rank (32 for ablation, 64 default)")
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-seq-length", type=int, default=3072)
    p.add_argument("--per-device-batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--num-epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=-1, help="cap total steps (for ablation runs)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="opt-in (default OFF for v2 — v0.1 had it ON unnecessarily, wasting recompute)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="limit dataset rows (for ablation smoke runs, e.g. 5000)",
    )
    return p.parse_args(argv)


def _format_record(example: dict) -> dict:
    prompt = example.get("prompt", "")
    response = example.get("response", "")
    text = (
        "<|im_start|>user\n"
        f"{prompt}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{response}"
        "<|im_end|>\n"
    )
    return {"text": text}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.data.is_file():
        log.error("training data not found: %s", args.data)
        return 2

    log.info("loading base %s (bf16)", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Freeze base
    for p in model.parameters():
        p.requires_grad_(False)

    if args.resume_adapter is not None:
        log.info("resuming from adapter: %s", args.resume_adapter)
        model = PeftModel.from_pretrained(model, str(args.resume_adapter), is_trainable=True)
        log.info("LoRA params from %s loaded; continuing training", args.resume_adapter)
    else:
        log.info("attaching fresh PEFT LoRA r=%d alpha=%d", args.lora_r, args.lora_alpha)
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    model.print_trainable_parameters()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GEMMA4DEFENSE-V2: trainable params: {n_trainable:,}", flush=True)
    if n_trainable < 1_000_000:
        raise RuntimeError(f"only {n_trainable:,} trainable — abort")

    if args.gradient_checkpointing:
        log.info("gradient_checkpointing ENABLED (slower but lower VRAM)")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    else:
        log.info("gradient_checkpointing DISABLED — using VRAM headroom for speed")

    log.info("loading dataset %s", args.data)
    raw = load_dataset("json", data_files=str(args.data), split="train")
    if args.limit is not None:
        raw = raw.select(range(min(args.limit, len(raw))))
        log.info("dataset limited to %d rows (ablation mode)", len(raw))
    formatted = raw.map(_format_record, remove_columns=raw.column_names)
    log.info("dataset prepared: %d rows", len(formatted))

    use_wandb = os.environ.get("WANDB_API_KEY") is not None
    report_to = "wandb" if use_wandb else "none"

    sft_cfg = SFTConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=True,
        packing=True,
        max_length=args.max_seq_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        report_to=report_to,
        seed=args.seed,
        dataset_text_field="text",
        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=formatted,
        args=sft_cfg,
    )

    log.info("starting v2 SFT training")
    trainer.train()

    log.info("saving adapter to %s", args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    log.info("v2 SFT done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
