# AMD MI300X setup notes

The released CyberSecQwen-4B checkpoint was trained, merged, and evaluated end-to-end on a single AMD Instinct MI300X 192 GB instance via the AMD Developer Cloud (`atl1` region). This document captures the setup specifics so others reproducing the work on AMD hardware don't hit the same papercuts we did.

## Stack used

| Component | Version |
|---|---|
| Hardware | AMD Instinct MI300X 192 GB (gfx942) |
| Cloud | DigitalOcean / AMD Developer Cloud (atl1) |
| OS | Ubuntu 22.04 |
| ROCm | 7.0 |
| Docker image | `vllm/vllm-openai-rocm:latest` |
| PyTorch | 2.6.0 (rocm) |
| transformers | 4.51+ |
| peft | 0.13+ |
| trl | 0.29.1 |
| flash-attn | 2.8.3 (preinstalled in the vLLM ROCm image) |
| vLLM | 0.10.1 |

## FlashAttention-2 viability bound on AMD MI300X

FA2 via the [ROCm/flash-attention](https://github.com/ROCm/flash-attention) Composable-Kernels backend is supported on `gfx942`, but is bounded at **head_dim ≤ 256** by the LDS (shared-memory) budget on this architecture.

| Model family | head_dim | FA2 on MI300X? |
|---|---:|---|
| Qwen3 (used here) | 128 | ✅ works |
| Llama-3 / Llama-3.1 | 128 | ✅ works |
| Mistral | 128 | ✅ works |
| Phi-3 / Phi-4 | 96 | ✅ works |
| Gemma-2 | 256 | ✅ works (boundary) |
| Gemma-4 | mixed 256 (sliding) + **512** (global) | ❌ FA2-CK fails on global layers |

For Gemma-4 we fell back to PyTorch `sdpa`, which is supported but slower (~1.6× per training step at the same precision). This is a known limitation; see the [companion Gemma4Defense-2B repo](https://github.com/GPT-64590/Gemma4Defense-2B) for the sdpa training path.

For Qwen3 (head_dim 128), FA2 is enabled simply by setting `attn_implementation="flash_attention_2"` in the trainer's `from_pretrained` call. No additional flags or kernel patches required. The flash-attn 2.8.3 wheel preinstalled in `vllm/vllm-openai-rocm:latest` works out of the box.

## Environment variables we set during training

```bash
# Inside the training Docker container:
export HF_HOME=/shared-docker/hf-cache
export VLLM_ROCM_USE_AITER=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HF_HUB_DISABLE_XET=1
export ROCM_PATH=/opt/rocm
export PYTORCH_ROCM_ARCH='gfx90a;gfx942;gfx950'
export AITER_ROCM_ARCH='gfx942;gfx950'
export HIP_FORCE_DEV_KERNARG=1
export HF_HUB_ENABLE_HF_TRANSFER=1   # only for HF push, not training
```

These are mostly drawn from AMD's recommended defaults for vLLM ROCm. The image preconfigures most of them; the explicit declaration above is for Docker `docker run -e` usage if you spin up a fresh container.

## vLLM ROCm gotcha: AITER and gpt-oss MoE

If you serve a gpt-oss-20b based model (e.g., the CyberPal-2.0-20B teacher we evaluated for distillation experiments), `VLLM_ROCM_USE_AITER` must be set to **0** (not 1) — the aiter MoE kernels do not support gpt-oss's expert layout and will error with `device_gemm with the specified compilation parameters does not support this GEMM problem`.

For Qwen3 dense models (the substrate used here), `VLLM_ROCM_USE_AITER=1` is correct.

## Training pipeline measurement

For reference, here is what the released CyberSecQwen-4B training run looked like on a single MI300X 192 GB:

| Phase | Wall time |
|---|---:|
| pip install + tokenizer setup | ~30 s |
| Tokenization + packing (14,776 records) | ~12 s |
| SFT (1,290 steps, LoRA r=64, FA2) | ~169 min @ ~7.85 s/step |
| Adapter merge into base + save | ~3 min |
| vLLM startup for eval | ~90 s |
| 5-trial benchmark (RCM 1k + MCQ 2.5k) | ~10 min |

Total: ~3 hours wall time end-to-end on a single MI300X 192 GB instance.

## Serving CyberSecQwen-4B on MI300X

```bash
docker run --rm --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri \
  -e VLLM_ROCM_USE_AITER=1 \
  -e TORCH_BLAS_PREFER_HIPBLASLT=1 \
  -e HF_HUB_DISABLE_XET=1 \
  -e ROCM_PATH=/opt/rocm \
  -e PYTORCH_ROCM_ARCH='gfx90a;gfx942;gfx950' \
  -e AITER_ROCM_ARCH='gfx942;gfx950' \
  -e HIP_FORCE_DEV_KERNARG=1 \
  vllm/vllm-openai-rocm:latest \
  --model athena129/CyberSecQwen-4B \
  --served-model-name cybersecqwen-4b \
  --attention-backend TRITON_ATTN \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 65536 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 --port 8001
```

Notes:
- `--attention-backend TRITON_ATTN` is the recommended inference attention backend on MI300X (separate from training's `flash_attention_2`)
- `--enable-prefix-caching` improves throughput when batch contains shared system prompts
- `--gpu-memory-utilization 0.9` is conservative; raise to 0.95 if you need more KV-cache headroom

## Hardware portability

The training recipe is hardware-agnostic. To run on NVIDIA A100 or H100, you only need to:
- Drop the AMD-specific environment variables (none of them harm NVIDIA, but they're no-ops there)
- Use the standard PyTorch + transformers + peft stack without the vLLM ROCm Docker image (a regular Python venv works)
- FA2 works the same way: `attn_implementation="flash_attention_2"` in the trainer

The 12 GB+ VRAM minimum for inference and 24 GB+ for training apply equally to NVIDIA hardware.
