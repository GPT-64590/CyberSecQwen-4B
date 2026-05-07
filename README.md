# CyberSecQwen-4B

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CyberSecQwen--4B-yellow)](https://huggingface.co/athena129/CyberSecQwen-4B)
[![Companion](https://img.shields.io/badge/Companion-Gemma4Defense--2B-blue)](https://github.com/GPT-64590/Gemma4Defense-2B)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![AMD MI300X](https://img.shields.io/badge/trained%20on-AMD%20MI300X-red)](docs/HARDWARE.md)
[![FA2](https://img.shields.io/badge/FlashAttention--2-enabled-purple)](docs/HARDWARE.md)

A 4B-parameter cybersecurity language model fine-tuned from [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) for CWE classification (CTI-RCM) and cyber threat intelligence multiple-choice (CTI-MCQ). Trained, merged, and evaluated end-to-end on a single AMD Instinct MI300X 192 GB instance using ROCm + vLLM + FlashAttention-2.

Under [Cisco's Foundation-Sec evaluation protocol (arXiv:2504.21039)](https://arxiv.org/abs/2504.21039), CyberSecQwen-4B retains 97.3% of Foundation-Sec-Instruct-8B's CTI-RCM accuracy at half the parameter count, and exceeds its CTI-MCQ by +8.7 points.

This repository contains everything needed to reproduce the model on AMD hardware: training corpus assembly (with explicit decontamination), supervised fine-tuning, multi-trial evaluation, and the released benchmark numbers.

---

## Contents

- [Headline benchmark results](#headline-benchmark-results)
- [Quick start (inference)](#quick-start-inference)
  - [vLLM serving (AMD MI300X)](#vllm-serving-amd-mi300x)
- [Reproducibility](#reproducibility)
- [AMD MI300X — what we used and what we optimized](#amd-mi300x--what-we-used-and-what-we-optimized)
- [Repository structure](#repository-structure)
- [Methodology summary](#methodology-summary)
- [Limitations and intended use](#limitations-and-intended-use)
- [Citation](#citation)
- [License](#license)
- [Companion model](#companion-model)

---

## Headline benchmark results

5 trials per cell, temperature 0.3, no system prompt, dataset-`Prompt`-column-as-user-message. Mean ± standard deviation.

| Benchmark | CyberSecQwen-4B (4B) | Foundation-Sec-Instruct-8B | Δ vs target |
|---|---:|---:|---:|
| CTI-MCQ (2,500 items) | **0.5868 ± 0.0029** | 0.4996 | **+8.7 pp** |
| CTI-RCM (1,000 items) | **0.6664 ± 0.0023** | 0.6850 | -1.9 pp |

A companion model trained with the **same recipe** on Gemma-4-E2B-it — [Gemma4Defense-2B](https://github.com/GPT-64590/Gemma4Defense-2B) — converges to within 0.9 points on CTI-RCM, demonstrating recipe portability across model families.

Full evaluation (more comparators, including independent reproduction of CyberPal-2.0-20B at our protocol): see [`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md).

---

## Quick start (inference)

```bash
pip install transformers torch accelerate
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "athena129/CyberSecQwen-4B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

cve = ("A deserialization vulnerability in the destruct() function of Laravel "
       "v8.5.9 allows attackers to execute arbitrary commands.")

messages = [{
    "role": "user",
    "content": (
        "Analyze the following CVE description and map it to the appropriate CWE. "
        "Provide a brief justification for your choice. "
        "Ensure the last line of your response contains only the CWE ID.\n\n"
        f"CVE Description: {cve}"
    ),
}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256, temperature=0.3, do_sample=True)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### vLLM serving (AMD MI300X)

```bash
docker run --rm --network=host --device=/dev/kfd --device=/dev/dri \
  -e VLLM_ROCM_USE_AITER=1 -e TORCH_BLAS_PREFER_HIPBLASLT=1 \
  vllm/vllm-openai-rocm:latest \
  --model athena129/CyberSecQwen-4B \
  --served-model-name cybersecqwen-4b \
  --attention-backend TRITON_ATTN \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9
```

See [`docs/HARDWARE.md`](docs/HARDWARE.md) for the full AMD MI300X setup, including FlashAttention-2 enablement, vLLM ROCm specifics, and verified-working environment variables.

---

## Reproducibility

Three commands, in order. Each is a thin wrapper around the underlying scripts in `src/`.

```bash
# 1. Assemble the training corpus (decontamination + composition).
#    Reads MITRE/NVD CVE→CWE mappings filtered to 2021-only with CTI-Bench
#    overlap items removed, plus synthetic CVE/CTI Q&A.
bash build_corpus.sh

# 2. Fine-tune Qwen3-4B-Instruct-2507 via LoRA on the assembled corpus.
#    LoRA r=64, alpha=64, dropout=0.05, lr=5e-5, 10 epochs, bf16, FA2.
bash train.sh

# 3. Evaluate the trained model under Cisco's Foundation-Sec protocol
#    on CTI-RCM and CTI-MCQ at 5 trials. Output: results/multi_trial_5x.json.
bash eval.sh
```

**System requirements** for full reproduction:
- 1× GPU with ≥ 24 GB VRAM (training) or ≥ 12 GB VRAM (inference only)
- Python 3.11+, PyTorch 2.6+, CUDA / ROCm 7
- ~50 GB disk for HF cache and intermediate artifacts

The released checkpoint was trained end-to-end on a single AMD Instinct MI300X 192 GB instance via the AMD Developer Cloud, using `vllm/vllm-openai-rocm:latest` Docker image. The recipe is hardware-agnostic and will run on NVIDIA A100/H100 with minor environment-variable changes; the AMD-specific paths are documented in [`docs/HARDWARE.md`](docs/HARDWARE.md).

---

## AMD MI300X — what we used and what we optimized

The released CyberSecQwen-4B checkpoint was trained, merged, and evaluated **end-to-end on a single AMD Instinct MI300X 192 GB instance** via the AMD Developer Cloud (`atl1` region). No multi-node, no NVIDIA hardware, no cross-platform porting. This section summarizes the optimization choices that made the pipeline work on the AMD stack; full setup specifics are in [`docs/HARDWARE.md`](docs/HARDWARE.md).

### Stack

| Component | Version |
|---|---|
| Hardware | AMD Instinct MI300X 192 GB (gfx942) |
| ROCm | 7.0 |
| Docker image | `vllm/vllm-openai-rocm:latest` |
| PyTorch | 2.6.0 (ROCm) |
| flash-attn | 2.8.3 (preinstalled in the vLLM ROCm image) |
| vLLM | 0.10.1 |

The vLLM ROCm Docker image ships everything needed (PyTorch ROCm, flash-attn 2.8.3, AMD libraries, transformers, peft, trl is `pip install`'d). One container, one GPU, full pipeline.

### Optimizations we used

1. **FlashAttention-2 enabled in training.** Qwen3-4B's attention head dimension (128) fits within the AMD `gfx942` LDS budget, so `attn_implementation="flash_attention_2"` works out of the box via flash-attn 2.8.3. Empirical measurement: ~7.85 s/step at LoRA r=64, max_seq_len=4096 — about **1.6× faster than the same recipe on Gemma-4 (head_dim=512, FA2 unavailable, falls back to sdpa)**.

2. **TRITON_ATTN backend for vLLM inference.** `--attention-backend TRITON_ATTN` is the recommended inference attention backend on MI300X for Qwen3-class models. Stable, no kernel-compile surprises.

3. **bf16 throughout.** No mixed-precision dance needed; bf16 is the native training and inference precision on MI300X.

4. **AITER kernels for matmul.** `VLLM_ROCM_USE_AITER=1` plus `TORCH_BLAS_PREFER_HIPBLASLT=1` for matmul throughput. (Note: this works for Qwen3 dense models — it does NOT work for gpt-oss-20b MoE models, where `AITER=0` is required. We hit this serving CyberPal-2.0-20B during recipe-development experiments.)

5. **Prefix caching for vLLM serving.** `--enable-prefix-caching` improves throughput when batches share a system or instruction prefix — relevant for any production deployment that wraps the model in a fixed-prompt template.

6. **`HF_HUB_ENABLE_HF_TRANSFER=1` for model push/pull.** The Rust-based multipart upload saturates the link to Hugging Face at ~240 MB/s on the AMD Developer Cloud `atl1` link — the 8 GB merged model uploads in ~36 seconds.

7. **`HIP_FORCE_DEV_KERNARG=1`** + AMD's recommended `PYTORCH_ROCM_ARCH='gfx90a;gfx942;gfx950'` for the device-kernel-args optimization.

### What does NOT work on the AMD stack (worth knowing)

These are not blockers for CyberSecQwen-4B, but they are real constraints that surfaced during the broader project:

- **FA2-CK on Gemma-4 (head_dim=512).** Falls back to sdpa. The companion [Gemma4Defense-2B](https://github.com/GPT-64590/Gemma4Defense-2B) repo trains with sdpa for this reason.
- **AITER kernels on gpt-oss-style MoE.** `VLLM_ROCM_USE_AITER=0` is required when serving CyberPal-2.0-20B (gpt-oss base) on MI300X.
- **bitsandbytes on ROCm.** Not officially supported. We did not test 4-bit/8-bit quantization on AMD; community ROCm forks exist but are not validated by the authors of this release.

### Inference (AMD MI300X)

```bash
docker run --rm --network=host --device=/dev/kfd --device=/dev/dri \
  -e VLLM_ROCM_USE_AITER=1 -e TORCH_BLAS_PREFER_HIPBLASLT=1 \
  vllm/vllm-openai-rocm:latest \
  --model athena129/CyberSecQwen-4B \
  --served-model-name cybersecqwen-4b \
  --attention-backend TRITON_ATTN \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.9
```

For the full set of environment variables, training-time gotchas, and pipeline timing measurements: see [`docs/HARDWARE.md`](docs/HARDWARE.md).

### Hardware portability

The training recipe in `train.sh` is hardware-agnostic. To run on NVIDIA A100 / H100, drop the AMD-specific environment variables (they're no-ops there) and use a regular Python venv with `pip install flash-attn --no-build-isolation` for FA2. NVIDIA users will need 24 GB+ VRAM for training and 12 GB+ for inference, same as MI300X minimums.

## Repository structure

```
CyberSecQwen-4B/
├── README.md                     # this file
├── LICENSE                       # Apache 2.0 (matches Qwen3 base)
├── CITATION.cff                  # GitHub-rendered citation
├── requirements.txt              # pinned Python dependencies
├── .gitignore
│
├── train.sh                      # single-command training reproducer
├── eval.sh                       # single-command 5-trial eval reproducer
├── build_corpus.sh               # single-command corpus assembly
│
├── src/
│   ├── train.py                  # LoRA SFT trainer (Qwen chat format, FA2)
│   ├── build_corpus.py           # corpus decontamination + composition
│   ├── cti_bench_eval.py         # Cisco-protocol benchmark harness
│   └── chat_template.jinja       # training-aligned minimal Qwen chat template
│
├── data/
│   ├── cti_bench/                # public eval data (TSV files)
│   │   ├── cti-rcm.tsv
│   │   └── cti-mcq.tsv
│   └── train/                    # training corpora (decontaminated)
│       ├── rcm_2021_train.jsonl  # CVE→CWE 2021 cohort, CTI-Bench overlap removed
│       └── cve_cti_synth.jsonl   # synthetic defensive-analyst Q&A
│
├── results/
│   ├── multi_trial_5x.json       # released benchmark numbers (5-trial mean ± std)
│   └── baseline_qwen3_it.json    # Qwen3-4B-Instruct-2507 raw baseline (pre-fine-tune)
│
└── docs/
    ├── RESEARCH_NOTES.md         # methodology, controlled experiments, lessons
    ├── RECIPE_PORTABILITY.md     # cross-substrate validation summary
    ├── HARDWARE.md               # AMD MI300X setup, FA2, vLLM ROCm specifics
    └── LIMITATIONS.md            # safety, ethics, abuse-prevention notes
```

---

## Methodology summary

This model uses **direct supervised fine-tuning (SFT)** of an instruction-tuned base via LoRA. Key design choices:

1. **Decontaminated training data.** An earlier internal iteration of this work showed roughly 72% test-set overlap when trained on undeduplicated CTI corpora. The released model trains exclusively on the 2021 CVE→CWE cohort with CTI-Bench overlap items removed, plus synthetic defensive-analyst Q&A grounded in CVE descriptions.
2. **Instruction-tuned base, not pre-trained base.** Direct SFT on the IT checkpoint preserves existing format priors (terse-answer multiple-choice convention) better than SFT on the pre-trained base. Notably, the IT base itself underperforms its corresponding pre-trained base on CTI-MCQ at our chat-template eval (Qwen3-4B-Base 0.667 vs Qwen3-4B-Instruct-2507 0.473) — the same MCQ-format collapse observed on Cisco's Foundation-Sec base→Instruct transition. Our SFT recovers and exceeds the IT starting point on both subsets.
3. **Direct SFT, not knowledge distillation.** We evaluated knowledge-distillation variants from a 20B teacher model (CyberPal-2.0-20B) earlier in the project. At our corpus scale (~15K supervised records) direct SFT outperformed distillation on the headline benchmarks. The released model is direct SFT only.
4. **Multi-trial benchmarking.** All headline numbers are means of 5 independent trials with random sampling seeds at temperature 0.3; standard deviations are reported alongside.
5. **Cross-substrate validation as built-in robustness check.** The identical training corpus and hyperparameters were applied independently to Gemma-4-E2B-it ([Gemma4Defense-2B](https://github.com/GPT-64590/Gemma4Defense-2B)). Both models converge within 0.9 points on CTI-RCM — strong evidence the result is recipe-driven, not substrate-specific.
6. **AMD MI300X end-to-end pipeline.** Training, adapter merging, and evaluation all run on a single MI300X 192 GB instance via the official `vllm/vllm-openai-rocm` Docker image. FlashAttention-2 is enabled because Qwen3-4B's attention head dimension (128) fits within the gfx942 LDS budget. See [`docs/HARDWARE.md`](docs/HARDWARE.md).

---

## Limitations and intended use

This is a defensive cybersecurity research artifact. It is not appropriate for:
- Generating exploit code, weaponized PoC, or attacker tradecraft
- Auto-executing security decisions without qualified human review
- Legal, medical, or regulated-advice contexts
- Tasks outside cybersecurity (general chat, code generation)

Full intended-use, out-of-scope-use, and limitations text is in the [Hugging Face model card](https://huggingface.co/athena129/CyberSecQwen-4B). Practical recommendations and recommended-use guardrails are in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

---

## Citation

```bibtex
@misc{cybersecqwen2026,
  title  = {CyberSecQwen-4B: A Compact CTI Specialist Fine-Tuned from Qwen3-4B-Instruct-2507 on AMD MI300X},
  author = {Mulia, Samuel},
  year   = {2026},
  publisher = {Hugging Face},
  url    = {https://huggingface.co/athena129/CyberSecQwen-4B}
}
```

The evaluation protocol is from [Foundation-Sec-8B (arXiv:2504.21039)](https://arxiv.org/abs/2504.21039); the benchmark is [CTI-Bench](https://github.com/xashru/cti-bench).

---

## License

- **Code in this repository:** Apache 2.0 — see [`LICENSE`](LICENSE)
- **The fine-tuned model weights** (hosted on Hugging Face): Apache 2.0 (matches Qwen3 base)

The model is a derivative of `Qwen/Qwen3-4B-Instruct-2507`, both code and weights are under Apache 2.0. The training data (decontaminated 2021 CVE→CWE mappings) is derived from public MITRE/NVD records; the synthetic CVE/CTI Q&A in `data/train/cve_cti_synth.jsonl` is original and released under Apache 2.0.

---

## Companion model

[Gemma4Defense-2B](https://github.com/GPT-64590/Gemma4Defense-2B) — sister release on Gemma-4-E2B-it. Same training recipe; converges to RCM 0.6754 ± 0.0035 / MCQ 0.6042 ± 0.0090. The Gemma variant ships under the Gemma Terms of Use; CyberSecQwen-4B (Apache 2.0) is appropriate for use cases where the Gemma license is not a fit.
