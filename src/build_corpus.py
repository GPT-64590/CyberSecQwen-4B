"""Assemble the Gemma4Defense-2B training corpus.

Combines two pre-curated, pre-decontaminated input files:
- data/train/rcm_2021_train.jsonl    (CVE→CWE 2021 cohort, CTI-Bench overlap removed)
- data/train/cve_cti_synth.jsonl     (synthetic defensive-analyst Q&A)

Outputs:
- data/train/combined_train.jsonl    (the actual SFT input for src/train.py)

Inputs are already filtered: rcm-2021 has been deduped against CTI-Bench's
RCM evaluation split, and cve_cti_synth contains only synthetic Q&A grounded
in CVE descriptions (not copied from CTI-Bench items). This script does NOT
re-run decontamination — it just shuffles and concatenates.

Re-running decontamination from raw NVD/MITRE feeds is documented in
docs/RESEARCH_NOTES.md (the "Decontamination methodology" section).

Usage:
    python src/build_corpus.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data" / "train"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "prompt" in row and "response" in row:
                rows.append({"prompt": row["prompt"], "response": row["response"]})
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rcm", type=Path, default=DATA / "rcm_2021_train.jsonl",
                   help="Path to the decontaminated CVE→CWE 2021 jsonl")
    p.add_argument("--cve-cti", type=Path, default=DATA / "cve_cti_synth.jsonl",
                   help="Path to the synthetic CVE/CTI Q&A jsonl")
    p.add_argument("--output", type=Path, default=DATA / "combined_train.jsonl",
                   help="Output path for the assembled training corpus")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.rcm.is_file():
        print(f"ERROR: missing input {args.rcm}")
        return 2
    if not args.cve_cti.is_file():
        print(f"ERROR: missing input {args.cve_cti}")
        return 2

    rcm = load_jsonl(args.rcm)
    cve = load_jsonl(args.cve_cti)
    print(f"loaded rcm_2021={len(rcm):>6}, cve_cti_synth={len(cve):>6}")

    random.seed(args.seed)
    combined = rcm + cve
    random.shuffle(combined)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for row in combined:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(combined)} records -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
