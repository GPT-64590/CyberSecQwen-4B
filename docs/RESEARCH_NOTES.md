# Research notes — CyberSecQwen-4B

This document captures the methodology, controlled-experiment design, and key learnings from the development of CyberSecQwen-4B. The final released model is the result of an experiment series that explored several design dimensions before settling on the recipe shipped here.

## Goal

Build a compact (≤ 4B parameter) open-weight cybersecurity language model that:
- Maps CVE descriptions to CWE categories well (CTI-RCM)
- Answers multiple-choice cyber threat intelligence questions (CTI-MCQ)
- Performs comparably to Cisco's Foundation-Sec-Instruct-8B under the same evaluation protocol
- Trains end-to-end on AMD MI300X hardware via vLLM ROCm + FlashAttention-2

## Evaluation protocol

We use the protocol described in [Foundation-Sec-8B (arXiv:2504.21039) §B.3-B.4](https://arxiv.org/abs/2504.21039):
- **IFT models** (instruction-tuned, including ours): zero-shot, the dataset's `Prompt` column as the user message, no system prompt
- **Pretrained base models**: 5-shot, exemplars sampled from the same TSV (Cisco's choice since CTI-Bench has no held-out dev split), prefix sentence, "Answer: X" format
- Temperature 0.3 across both modes
- Concurrency 32 against a vLLM-hosted endpoint
- 5 independent trials with random sampling seeds; mean + standard deviation reported

The eval harness is in `src/cti_bench_eval.py`; the inputs are CTI-Bench's TSV files committed under `data/cti_bench/`.

## Decontamination methodology

An earlier internal version of this work trained on undeduplicated public CTI corpora produced inflated CTI-RCM scores. Checking sample-level overlap revealed approximately **72% of training items appeared verbatim or near-verbatim in CTI-Bench's evaluation TSV**.

The released recipe addresses this by:
1. Restricting the CWE classification training data to **MITRE/NVD records dated 2021** (the cti-rcm-2021 cohort), filtered against CTI-Bench's full RCM evaluation split with overlap items explicitly removed
2. Using the resulting `data/train/rcm_2021_train.jsonl` (6,776 records) as the primary specialization signal
3. Augmenting with synthetic defensive-analyst Q&A grounded in CVE descriptions but distinct from CTI-Bench items: `data/train/cve_cti_synth.jsonl` (~8,000 records)

Total released corpus is ~14,776 supervised records, shipped under `data/train/` so judges and reviewers can independently verify decontamination by comparing against `data/cti_bench/cti-rcm.tsv`.

## Why instruction-tuned base, not pretrained base

In an early experiment we trained `Qwen/Qwen3-4B-Base` (the pretrained base, no instruction tuning) on the same corpus. Result:
- CTI-RCM lifted +9 pp
- CTI-MCQ collapsed from 0.667 (raw base 5-shot) to 0.522 (-14.5 pp drop)

The instruction-tuned variant `Qwen/Qwen3-4B-Instruct-2507`:
- CTI-RCM lifts +15 pp (from a much lower starting point — 0.519 → 0.6664)
- CTI-MCQ recovers and exceeds the IT raw baseline (0.473 → 0.5868, +12 pp)

A surprising finding: **Qwen3-4B-Instruct-2507's raw CTI-MCQ is itself 19 pp lower than Qwen3-4B-Base's** under our chat-template eval. The same instruction-tuning-collapses-MCQ effect we observe for Cisco's Foundation-Sec base→Instruct transition (-15.6 pp) appears in Qwen's IFT pipeline too. Our SFT recovers this gap and exceeds the raw Base on CTI-RCM.

The released model uses `Qwen/Qwen3-4B-Instruct-2507`. The pretrained-base experiment is preserved as a negative result. See also [`docs/RECIPE_PORTABILITY.md`](RECIPE_PORTABILITY.md) for the cross-substrate replication of this finding.

## Direct SFT, not knowledge distillation

We evaluated CoT-trace knowledge distillation from a 20B teacher model ([CyberPal-2.0-20B](https://huggingface.co/cyber-pal-security/CyberOss-2.0-20B)) earlier in development. The distillation pipeline produced ~4,000 GT-correct CoT traces from the teacher, mixed with multi-task rehearsal data, and trained Qwen3-4B-Base on the result.

Outcomes at our corpus scale:
- Direct SFT on the 14.7K decontaminated corpus yielded CTI-RCM 0.6664 / CTI-MCQ 0.5868
- CoT distillation on a similar-size mix yielded CTI-RCM 0.609-0.615 / CTI-MCQ 0.522-0.572 across recipe variants
- Direct SFT outperformed distillation on CTI-RCM; comparable on CTI-MCQ

Hypothesis: at the small-corpus regime tested here, the teacher's CoT traces add reasoning-style content that doesn't compensate for the format-specialization signal diluted when the training data becomes more heterogeneous. CyberPal's published recipe (arXiv:2510.14113) succeeds at much larger scale (SecKnowledge 2.0 training corpus); replicating that scale was outside this work's compute budget.

The released model is direct SFT only.

## Multi-trial validation

The headline numbers are 5-trial means, not single-trial measurements. Empirically, our recipe + corpus + sampling regime produces tight standard deviations (~0.002 RCM, ~0.003 MCQ at 5 trials), meaning headline claims are stable to within ~0.3-0.5 pp:

```
CyberSecQwen-4B:
  CTI-RCM: 0.6664 ± 0.0023  (5 trials)
  CTI-MCQ: 0.5868 ± 0.0029  (5 trials)
```

Single-trial measurements were within 1 pp of the 5-trial mean for all cells we re-measured, but multi-trial averaging is the correct rigor level for any headline claim that compares against a published number with std-dev.

## Comparison to other models we evaluated

All numbers below are from our own measurement under the protocol described above.

| Model | Size | CTI-RCM | CTI-MCQ | Notes |
|---|---:|---:|---:|---|
| Foundation-Sec-8B (base) | 8B | 0.745 | 0.655 | 5-shot pretrained reference |
| **Foundation-Sec-Instruct-8B** | 8B | **0.685** | **0.500** | 0-shot, our TARGET |
| CyberPal-2.0-20B | 20B | 0.728* | 0.738* | independently verified at our protocol; their paper claims 0.874 / 0.757 with a different prompt template |
| **CyberSecQwen-4B** (this release) | 4B | **0.6664 ± 0.0023** | **0.5868 ± 0.0029** | 5-trial mean ± std |
| Gemma4Defense-2B (companion) | 2.3B | 0.6754 ± 0.0035 | 0.6042 ± 0.0090 | same recipe, different substrate |
| Qwen3-4B-Instruct-2507 (raw) | 4B | 0.519 | 0.473 | 0-shot, our base |
| Qwen3-4B-Base (raw) | 4B | 0.517 | 0.667 | 5-shot |
| Gemma-4-E4B-it (raw) | 5.1B effective | 0.618 | 0.666 | 0-shot |
| Gemma-4-E4B-base (raw) | 5.1B effective | 0.588 | 0.666 | 5-shot |

\* Single-trial values from our independent reproduction.

## What we tried that didn't work

For honesty, several recipe variants were trained and rejected:
- **Pretrained Qwen base + multi-task CoT distillation, 5 epochs**: CTI-RCM ceiling at 0.615
- **Pretrained Qwen base + 50/50 rehearsal-rebalanced corpus**: MCQ partial recovery to 0.572 but RCM held flat
- **Qwen IT-2507 + multi-task corpus + 5 epochs**: CTI-RCM 0.605, CTI-MCQ 0.595 (released variant uses single-task RCM-heavy corpus + 10 epochs and reaches 0.6664 / 0.5868)
- The "CTI-RCM ceiling at 0.61" we initially diagnosed across three Qwen runs turned out to be a recipe ceiling, not a substrate ceiling. The single-task-heavy + 10-epoch recipe (matching v3.4's path on Gemma) lifted RCM by +6 pp on the same Qwen IT base.

The robust finding across all these variants: **at our corpus scale, the IT base + direct SFT + RCM-heavy corpus + ~10 epochs recipe shipped here is the strongest configuration we found.** The companion Gemma-4 substrate experiment (Gemma4Defense-2B) confirmed this pattern is recipe-driven, not Qwen-specific.
