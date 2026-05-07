#!/usr/bin/env bash
# Multi-trial evaluation of Gemma4Defense-2B on CTI-Bench (RCM + MCQ)
# under Cisco's Foundation-Sec evaluation protocol (arXiv:2504.21039 §B.3-B.4).
#
# Modes:
#   bash eval.sh hf            (default) — pulls athena129/CyberSecQwen-4B from HF
#   bash eval.sh local <path>  — evaluates a locally-merged checkpoint
#
# Prerequisites:
#   - vLLM installed (recommended: official Docker image)
#   - 1× GPU with ≥ 12 GB VRAM
#   - 5 trials × (1000 RCM + 2500 MCQ) ≈ 8-10 minutes wall on a modern GPU
set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-hf}"
case "$MODE" in
  hf)
    MODEL_PATH="athena129/CyberSecQwen-4B"
    SERVED_NAME="cybersecqwen-4b"
    ;;
  local)
    if [ -z "${2:-}" ]; then
      echo "ERROR: local mode requires a path. Usage: bash eval.sh local <path-to-merged-checkpoint>" >&2
      exit 1
    fi
    MODEL_PATH="$2"
    SERVED_NAME="cybersecqwen-4b-local"
    ;;
  *)
    echo "Usage: bash eval.sh [hf|local <path>]" >&2
    exit 1
    ;;
esac

mkdir -p results

echo "Starting vLLM server with model: $MODEL_PATH"
# Background the server. User can also run this manually if they prefer.
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 --port 8001 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null || true" EXIT

# Wait for vLLM to be ready
echo "Waiting for vLLM..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:8001/v1/models > /dev/null 2>&1; then
    echo "vLLM ready (after ${i}×5s)"
    break
  fi
  sleep 5
done

python src/cti_bench_eval.py \
  --protocol ift --model "$SERVED_NAME" \
  --base-url http://localhost:8001/v1 --api-key EMPTY \
  --concurrency 32 --trials 5 --temperature 0.3 --max-tokens 512 \
  --subsets cti-rcm cti-mcq \
  --data-dir data/cti_bench \
  --output results/multi_trial_5x.json

echo
echo "Done. Results: results/multi_trial_5x.json"
python -c "
import json, statistics
r = json.load(open('results/multi_trial_5x.json'))
for sub in ['cti-rcm', 'cti-mcq']:
    if sub in r['subsets']:
        accs = [t['accuracy_total'] for t in r['subsets'][sub]['trials']]
        m = statistics.mean(accs)
        s = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(f'  {sub:10s}: {m:.4f} ± {s:.4f}  (n={len(accs)} trials)')
"
