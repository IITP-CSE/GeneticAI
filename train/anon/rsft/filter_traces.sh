#!/bin/bash
## ANON wrapper for Step 1b RSFT filtering. Defaults the output path and delegates to
## train/rsft/filter_traces.sh. The filter itself is dataset-agnostic (operates on the
## sampled traces JSONL); this wrapper just keeps anon outputs separate.
##
## Usage:
##   TRACES=train/rsft/samples_anon/rsft_sample_anon_traces.jsonl \
##     bash train/anon/rsft/filter_traces.sh
set -euo pipefail
_REPO="$(cd "$(dirname "$0")/../../.." && pwd)"

export OUT="${OUT:-$_REPO/train/rsft/rsft_selfadaptive_anon.jsonl}"

exec bash "$_REPO/train/rsft/filter_traces.sh" "$@"
