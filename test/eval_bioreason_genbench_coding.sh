#!/bin/bash
#SBATCH --job-name=eval_bioreason_genbench
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=test/log/eval_bioreason_genbench_%j.out
#SBATCH --error=test/log/eval_bioreason_genbench_%j.err

## Test a BioReason-mode checkpoint on the coding-variant (snp_sequence) subset of
## iit-patna-cse-ai/GenBench_coding. See eval_bioreason_genbench_coding.py's docstring
## for why only that subset is used (the other 7 task types carry no DNA sequence).
##
## GenBench_coding is gated -- set HF_TOKEN (or run `huggingface-cli login` on this
## node beforehand) or the dataset load will fail/hang on auth.
##
## Usage:
##   HF_TOKEN=hf_xxx CKPT=<path/to/last.ckpt or model.pt> bash test/eval_bioreason_genbench_coding.sh [gpu_id]
##   LIMIT=50 CKPT=<...> bash test/eval_bioreason_genbench_coding.sh   # smoke test
##   SPLIT=both CKPT=<...> bash test/eval_bioreason_genbench_coding.sh   # train+test snp_sequence rows (bigger sample)
##   IGNORE_MODALITY=1 SPLIT=both CKPT=<...> bash test/eval_bioreason_genbench_coding.sh
##     # every MCQ-graded row, not just snp_sequence ones -- rows without DNA get an
##     # empty reference_sequence/variant_sequence (text-only prompt, same pipeline)
##   NO_THINK=1 CKPT=<...> bash test/eval_bioreason_genbench_coding.sh
##     # force an empty <think></think>, answer immediately -- fixes generations that
##     # ramble past --max_new_tokens without ever reaching an Answer: line

CONDA_ENV=${CONDA_ENV:-dna_env}
CACHE_DIR=${CACHE_DIR:-~/.cache/huggingface}
HF_TOKEN=${HF_TOKEN:-}
CKPT=${CKPT:-}
SPLIT=${SPLIT:-test}
LIMIT=${LIMIT:-}
IGNORE_MODALITY=${IGNORE_MODALITY:-0}
NO_THINK=${NO_THINK:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-600}
OUTPUT_DIR=${OUTPUT_DIR:-$(pwd)/test/log}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-bioreason_genbench_coding}

if [ -z "$CKPT" ]; then
    echo "ERROR: set CKPT=<path to a BioReason .ckpt or .pt checkpoint>"
    exit 1
fi

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/.."
mkdir -p "$OUTPUT_DIR"
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${1:-0}
export PYTORCH_ALLOC_CONF=expandable_segments:True

LOG="${LOG:-$OUTPUT_DIR/${OUTPUT_PREFIX}_$(date +%Y%m%d_%H%M%S)_run.log}"
exec > >(tee "$LOG") 2>&1
echo "Command:    bash $0 $*"
echo "Checkpoint: $CKPT"
echo "Split:      $SPLIT   limit=${LIMIT:-none}"
nvidia-smi

LIMIT_FLAG=""
[ -n "$LIMIT" ] && LIMIT_FLAG="--limit $LIMIT"
TOKEN_FLAG=""
[ -n "$HF_TOKEN" ] && TOKEN_FLAG="--hf_token $HF_TOKEN"
MODALITY_FLAG=""
[ "$IGNORE_MODALITY" = "1" ] && MODALITY_FLAG="--ignore_modality"
NO_THINK_FLAG=""
[ "$NO_THINK" = "1" ] && NO_THINK_FLAG="--no_think"

stdbuf -oL -eL python eval_bioreason_genbench_coding.py \
    --checkpoint            "$CKPT" \
    --split                 "$SPLIT" \
    --max_new_tokens        "$MAX_NEW_TOKENS" \
    --cache_dir             "$CACHE_DIR" \
    --output_dir            "$OUTPUT_DIR" \
    --output_prefix         "$OUTPUT_PREFIX" \
    $LIMIT_FLAG $TOKEN_FLAG $MODALITY_FLAG $NO_THINK_FLAG

echo ""
echo "=== Done. See $OUTPUT_DIR for ${OUTPUT_PREFIX}_predictions.csv / _metrics.json ==="
