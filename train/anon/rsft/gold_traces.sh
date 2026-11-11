#!/bin/bash
## ANON wrapper for Step 1c RSFT gold-trace backfill. Defaults the anon KEGG_CSV + output
## path and delegates to train/rsft/gold_traces.sh.
##
## Usage:
##   INDICES=train/rsft/samples_anon/uncovered_anon.txt bash train/anon/rsft/gold_traces.sh
set -euo pipefail
_REPO="$(cd "$(dirname "$0")/../../.." && pwd)"

export KEGG_CSV="${KEGG_CSV:-$_REPO/genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}"
export OUT="${OUT:-$_REPO/train/rsft/samples_anon/rsft_gold_anon_traces.jsonl}"

exec bash "$_REPO/train/rsft/gold_traces.sh" "$@"
