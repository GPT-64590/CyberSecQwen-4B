"""CTIBench eval, Cisco protocol — apples-to-apples vs Foundation-Sec-8B.

Implements the exact protocol from arXiv:2504.21039 §B.3-B.4 (Figs 4-7) so we can
defend any "beat Cisco" claim. Two prompt modes:

  base  (pretrained models, e.g. Llama-3.1-8B, Foundation-Sec-8B base, Gemma-4-E2B
         non-it): 5-shot, exemplars sampled from the same TSV (Cisco's choice
         since CTIBench has no dev split), prefix sentence, "Answer: X" format.

  ift   (instruct-tuned, e.g. Foundation-Sec-8B-Instruct, Gemma-4-E2B-it,
         our v0.1/v3 SFT'd models): zero-shot, use the dataset's own `Prompt`
         column (which is the official IFT instruction), no system prompt.

Multi-trial averaging (default 10) with random exemplar/seed per trial — matches
Cisco's stochasticity treatment. For temperature=0 add `--trials 1`.

Inputs come from a local clone of github.com/xashru/cti-bench/data/*.tsv.
Scoring matches the original CTI-Bench notebook (case-insensitive letter for
MCQA, CWE-NNN string match for RCM) plus Cisco's regex fallbacks (Appendix B.4).

Outputs: JSON with per-trial scores + mean/std + raw predictions for audit.

Usage:
  python cti_bench_cisco.py \
      --base-url http://localhost:8001/v1 --api-key EMPTY \
      --model gemma4defense-e2b --protocol ift \
      --subsets cti-rcm cti-mcq --trials 10 --temperature 0.3 \
      --data-dir /shared-docker/project/eval/cti_bench_data \
      --output /shared-docker/eval-output/cisco_proto/v0.1.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import random
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import backoff
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cti_bench_cisco")

# Cisco's exact prefix for 5-shot pretrained eval (paper §B.3.1)
PRETRAINED_PREFIX = "The following are multiple choice questions about computer security."

# Per Cisco inference_clients.py — IFT models get full headroom
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.3
DEFAULT_N_SHOT = 5
DEFAULT_TRIALS = 10
DEFAULT_CONCURRENCY = 32

# ---------- Data loading ----------


def load_tsv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


# ---------- Prompt construction ----------


def format_mcq_exemplar(row: dict[str, str], include_answer: bool = True) -> str:
    """Cisco Fig 4 format: question / A. / B. / C. / D. / Answer: X."""
    parts = [
        row["Question"].strip(),
        f"A. {row['Option A'].strip()}",
        f"B. {row['Option B'].strip()}",
        f"C. {row['Option C'].strip()}",
        f"D. {row['Option D'].strip()}",
    ]
    if include_answer:
        parts.append(f"Answer: {row['GT'].strip().upper()}")
    return "\n".join(parts)


def format_rcm_exemplar(row: dict[str, str], include_answer: bool = True) -> str:
    """Cisco Fig 5 format: 'CVE Description: ...' / 'Answer: CWE-X'."""
    desc = row["Description"].strip()
    parts = [f"CVE Description: {desc}"]
    if include_answer:
        parts.append(f"Answer: {row['GT'].strip().upper()}")
    return "\n".join(parts)


def build_5shot_prompt(
    subset: str, target_row: dict[str, str], exemplars: list[dict[str, str]]
) -> str:
    """5-shot: prefix + 5 exemplars + target (no answer)."""
    if subset == "cti-mcq":
        chunks = [PRETRAINED_PREFIX]
        for ex in exemplars:
            chunks.append(format_mcq_exemplar(ex, include_answer=True))
        chunks.append(format_mcq_exemplar(target_row, include_answer=False))
        chunks.append("Answer:")
        return "\n\n".join(chunks)
    if subset == "cti-rcm":
        chunks = []  # RCM doesn't use the MCQ prefix per Cisco Fig 5
        for ex in exemplars:
            chunks.append(format_rcm_exemplar(ex, include_answer=True))
        chunks.append(format_rcm_exemplar(target_row, include_answer=False))
        chunks.append("Answer:")
        return "\n\n".join(chunks)
    raise ValueError(f"Unsupported subset: {subset}")


def build_ift_prompt(subset: str, target_row: dict[str, str]) -> str:
    """Zero-shot for IFT models: use the dataset's own Prompt column verbatim
    (which Cisco's paper §B.3.2 says they use directly for benchmarks that
    provide their own instruction — CTIBench does)."""
    return target_row["Prompt"].strip()


# ---------- Answer extraction (Cisco Appendix B.4) ----------

# B.4.1 step 0: "Answer: X" (case-insensitive, optional spaces, optional paren, A-D)
_ANSWER_PRIMARY = re.compile(r"answer\s*:\s*\(?([A-D])\)?", re.IGNORECASE)
# B.4.1 step 1: "answer is X" / "answer is: X"
_ANSWER_IS = re.compile(r"answer\s+is\s*:?\s*\(?([A-D])\)?", re.IGNORECASE)
# B.4.1 step 2: standalone uppercase A-D, optionally parenthesized — applied to last line only
_STANDALONE = re.compile(r"^\s*\(?([A-D])\)?\s*$")
# B.4.1 step 3: "Option X"
_OPTION = re.compile(r"\boption\s+\(?([A-D])\)?", re.IGNORECASE)

_CWE_RE = re.compile(r"CWE-(\d+)", re.IGNORECASE)


def extract_mcq_letter(text: str) -> str | None:
    """Cisco's regex cascade for MCQA answer extraction (paper §B.4.1)."""
    if not text:
        return None
    # Stage 1: the primary "Answer: X" pattern
    m = _ANSWER_PRIMARY.search(text)
    if m:
        return m.group(1).upper()
    # Stage 2: "answer is X"
    m = _ANSWER_IS.search(text)
    if m:
        return m.group(1).upper()
    # Stage 3: standalone letter on its own line (last-line check, common for IFT models
    # told "last line should contain only the letter")
    last_line = text.strip().splitlines()[-1] if text.strip() else ""
    m = _STANDALONE.match(last_line)
    if m:
        return m.group(1).upper()
    # Stage 4: "Option X"
    m = _OPTION.search(text)
    if m:
        return m.group(1).upper()
    return None


def extract_cwe(text: str) -> str | None:
    """Cisco uses CTI-Bench authors' codebase; pattern is `CWE-\\d+`."""
    if not text:
        return None
    matches = _CWE_RE.findall(text)
    if not matches:
        return None
    # Cisco's instruction says "last line contains only the CWE ID", so prefer
    # the last CWE-NNN in the response (handles multi-CWE reasoning text).
    return f"CWE-{matches[-1]}"


# ---------- Inference ----------


@backoff.on_exception(
    backoff.expo,
    (APIConnectionError, APIStatusError, RateLimitError, asyncio.TimeoutError),
    max_tries=3,
    jitter=backoff.full_jitter,
    giveup=lambda e: isinstance(e, APIStatusError) and 400 <= getattr(e, "status_code", 500) < 500,
)
async def _completion_call(
    client: AsyncOpenAI, model: str, prompt: str, max_tokens: int, temperature: float, seed: int | None, subset: str
) -> str:
    """Pretrained-model path: raw text completion, no chat template.
    Stop sequences halt the base model after the answer pattern so we don't
    waste tokens generating into the next exemplar."""
    if subset == "cti-mcq":
        # After "Answer: X", the model would naturally start the next exemplar
        # with a blank line. Stop on blank line or any new "Question/A./..." pattern.
        stop = ["\n\n", "\nQuestion:", "\n\nThe following"]
    else:  # cti-rcm
        stop = ["\n\n", "\nCVE Description:"]
    kwargs: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 1.0,
        "stop": stop,
    }
    if seed is not None:
        kwargs["seed"] = seed
    resp = await client.completions.create(**kwargs)
    return (resp.choices[0].text or "").strip()


@backoff.on_exception(
    backoff.expo,
    (APIConnectionError, APIStatusError, RateLimitError, asyncio.TimeoutError),
    max_tries=3,
    jitter=backoff.full_jitter,
    giveup=lambda e: isinstance(e, APIStatusError) and 400 <= getattr(e, "status_code", 500) < 500,
)
async def _chat_call(
    client: AsyncOpenAI, model: str, prompt: str, max_tokens: int, temperature: float, seed: int | None
) -> str:
    """IFT-model path: chat completions with the dataset's instruction as user msg.
    No system prompt per Cisco §B.3.2."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 1.0,
    }
    if seed is not None:
        kwargs["seed"] = seed
    resp = await client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


# ---------- Trial runner ----------


async def run_trial(
    *,
    client: AsyncOpenAI,
    model: str,
    subset: str,
    rows: list[dict[str, str]],
    protocol: str,
    n_shot: int,
    rng: random.Random,
    max_tokens: int,
    temperature: float,
    seed: int | None,
    concurrency: int,
) -> dict[str, Any]:
    """One trial = one pass over all rows, with one fresh exemplar set (base) or
    deterministic single-shot (ift)."""
    sem = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = [None] * len(rows)  # type: ignore[list-item]

    extract = extract_mcq_letter if subset == "cti-mcq" else extract_cwe

    async def one(idx: int, row: dict[str, str]) -> None:
        async with sem:
            if protocol == "base":
                # Sample n_shot exemplars from rows, excluding target. Cisco re-samples
                # per trial; within a trial we re-sample per item too for max diversity.
                pool = [r for j, r in enumerate(rows) if j != idx]
                exemplars = rng.sample(pool, n_shot)
                prompt = build_5shot_prompt(subset, row, exemplars)
                output = await _completion_call(client, model, prompt, max_tokens, temperature, seed, subset)
            elif protocol == "ift":
                prompt = build_ift_prompt(subset, row)
                output = await _chat_call(client, model, prompt, max_tokens, temperature, seed)
            else:
                raise ValueError(f"protocol must be base|ift, got {protocol!r}")

            pred = extract(output)
            gold = row["GT"].strip().upper()
            if subset == "cti-rcm":
                # Normalize gold to "CWE-NNN" form
                gold_match = _CWE_RE.search(gold)
                gold = f"CWE-{gold_match.group(1)}" if gold_match else gold
            correct = (pred == gold) if pred else False
            results[idx] = {
                "idx": idx,
                "url": row.get("URL", ""),
                "gold": gold,
                "pred_raw": output[:1000],  # truncate for log size
                "pred": pred,
                "correct": correct,
                "parseable": pred is not None,
            }

    t0 = time.time()
    await asyncio.gather(*[one(i, r) for i, r in enumerate(rows)])
    wall = time.time() - t0

    n = len(results)
    parseable = sum(1 for r in results if r["parseable"])
    correct = sum(1 for r in results if r["correct"])
    return {
        "n": n,
        "parseable": parseable,
        "correct": correct,
        "accuracy_total": correct / n if n else 0.0,
        "accuracy_parseable": correct / parseable if parseable else 0.0,
        "wall_seconds": round(wall, 1),
        "predictions": results,
    }


# ---------- Main ----------


async def amain(args: argparse.Namespace) -> None:
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        log.error("data dir %s does not exist; clone github.com/xashru/cti-bench first", data_dir)
        sys.exit(2)

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600.0)

    summary: dict[str, Any] = {
        "model": args.model,
        "base_url": args.base_url,
        "protocol": args.protocol,
        "n_shot": args.n_shot if args.protocol == "base" else 0,
        "trials": args.trials,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "subsets": {},
    }

    for subset in args.subsets:
        tsv_path = data_dir / f"{subset}.tsv"
        rows = load_tsv(tsv_path)
        if args.limit:
            rows = rows[: args.limit]
        log.info("[%s] loaded %d rows from %s", subset, len(rows), tsv_path)

        trial_accs: list[float] = []
        trial_records: list[dict[str, Any]] = []
        for t in range(args.trials):
            trial_seed = (args.seed or 0) + t
            rng = random.Random(trial_seed)
            log.info("[%s] trial %d/%d (seed=%d)", subset, t + 1, args.trials, trial_seed)
            res = await run_trial(
                client=client,
                model=args.model,
                subset=subset,
                rows=rows,
                protocol=args.protocol,
                n_shot=args.n_shot,
                rng=rng,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=trial_seed if args.temperature > 0 else None,
                concurrency=args.concurrency,
            )
            trial_accs.append(res["accuracy_total"])
            log.info(
                "[%s] trial %d acc=%.4f (parseable=%.4f, %ds)",
                subset,
                t + 1,
                res["accuracy_total"],
                res["parseable"] / res["n"] if res["n"] else 0.0,
                res["wall_seconds"],
            )
            # Drop predictions on all but the last trial to keep output reasonable
            if t < args.trials - 1:
                res.pop("predictions", None)
            trial_records.append(res)

        mean_acc = statistics.mean(trial_accs)
        std_acc = statistics.stdev(trial_accs) if len(trial_accs) > 1 else 0.0
        summary["subsets"][subset] = {
            "trials": trial_records,
            "trial_accuracies": trial_accs,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
        }
        log.info("[%s] DONE — mean=%.4f std=%.4f (n_trials=%d)", subset, mean_acc, std_acc, args.trials)

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    out_path.write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", out_path)

    # Also print a compact summary table for terminal review
    print("\n=== Cisco-protocol scoreboard ===")
    print(f"Model: {args.model}  |  Protocol: {args.protocol}  |  Trials: {args.trials}  |  Temp: {args.temperature}")
    print(f"{'Subset':<12} {'Mean':>8} {'Std':>8} {'Trials':>8}")
    for subset, rec in summary["subsets"].items():
        print(f"{subset:<12} {rec['mean_accuracy']:>8.4f} {rec['std_accuracy']:>8.4f} {len(rec['trial_accuracies']):>8d}")


def main() -> None:
    p = argparse.ArgumentParser(description="CTIBench eval, Cisco protocol (5-shot base / 0-shot IFT)")
    p.add_argument("--base-url", required=True, help="vLLM OpenAI-compatible endpoint, e.g. http://localhost:8001/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model", required=True, help="--served-model-name on the vLLM server")
    p.add_argument(
        "--protocol",
        required=True,
        choices=["base", "ift"],
        help="base = 5-shot text completion (Cisco Figs 4-5); ift = zero-shot chat (Figs 6-7)",
    )
    p.add_argument("--subsets", nargs="+", default=["cti-rcm", "cti-mcq"], choices=["cti-rcm", "cti-mcq"])
    p.add_argument("--data-dir", default="/shared-docker/project/eval/cti_bench_data", help="path to xashru/cti-bench/data")
    p.add_argument("--output", required=True)
    p.add_argument("--n-shot", type=int, default=DEFAULT_N_SHOT)
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Cisco uses 10 for temp>0")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="cap rows for smoke testing")
    args = p.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
