#!/usr/bin/env bash
# Fine-tune Qwen/Qwen3-4B-Instruct-2507 via LoRA on the assembled training corpus.
#
# Reproduces the released CyberSecQwen-4B checkpoint (within multi-trial
# noise) using the v8.0 recipe.
#
# Prerequisites:
#   - bash build_corpus.sh has been run (creates data/train/combined_train.jsonl)
#   - 1× GPU with ≥ 24 GB VRAM (training)
#   - Python 3.11+, torch>=2.6, transformers>=4.51, peft, trl==0.29.1, accelerate
#   - flash-attn>=2.8 for FA2 (preinstalled in vllm/vllm-openai-rocm Docker image
#     on AMD MI300X; on NVIDIA A100/H100 install via `pip install flash-attn --no-build-isolation`)
#   - HF auth: `huggingface-cli login` or HF_TOKEN env var
#
# Hyperparameters (LoRA r=64, alpha=64, dropout=0.05, lr=5e-5, 10 epochs, bf16, FA2):
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p output/adapter

if [ ! -f data/train/combined_train.jsonl ]; then
  echo "Running build_corpus.sh first..."
  bash build_corpus.sh
fi

python src/train.py \
  --data data/train/combined_train.jsonl \
  --output-dir output/adapter/cybersecqwen-4b \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --lora-r 64 --lora-alpha 64 --lora-dropout 0.05 \
  --max-seq-length 4096 \
  --per-device-batch-size 2 --grad-accum 8 \
  --num-epochs 10 \
  --lr 5e-5 --warmup-ratio 0.05 --weight-decay 0.01 \
  --logging-steps 10 --save-steps 200 --seed 42 \
  --gradient-checkpointing

echo
echo "Adapter saved to output/adapter/cybersecqwen-4b/"
echo "Next step: merge adapter into base for inference (see docs/HARDWARE.md for"
echo "the AMD MI300X serving variant)."
