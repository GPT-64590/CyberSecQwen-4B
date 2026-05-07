#!/usr/bin/env bash
# Assemble the Gemma4Defense-2B training corpus.
#
# Inputs (committed to this repo):
#   data/train/rcm_2021_train.jsonl  — decontaminated CVE→CWE 2021 cohort
#   data/train/cve_cti_synth.jsonl   — synthetic defensive-analyst Q&A
#
# Output:
#   data/train/combined_train.jsonl  — the assembled SFT input
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f data/train/rcm_2021_train.jsonl ] || [ ! -f data/train/cve_cti_synth.jsonl ]; then
  echo "ERROR: required input files are missing under data/train/." >&2
  echo "       Expected rcm_2021_train.jsonl and cve_cti_synth.jsonl." >&2
  exit 1
fi

python src/build_corpus.py "$@"
