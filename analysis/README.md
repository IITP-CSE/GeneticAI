# Supplementary analyses (Supplementary Tables S1–S4)

These scripts reproduce Supplementary Tables S1–S4, the unseen-variant rows of Table 9, and the Table 10 control
(RSFT seeded directly from Stage 1.5+Gate, skipping GRPO) of the revised GenoMorph manuscript (Briefings in
Bioinformatics, BIB-26-2135). They run on a CPU in under a minute and use only the saved per-example outputs below;
no model is loaded or run.

## Contents

| Path | What it is |
|---|---|
| `supplementary_analyses.py` | Table S1 (bootstrap 95% CIs, exact McNemar tests), Table S3 (class distribution), Table S4 (reasoning length), Table 9 unseen-variant rows, Table S2 gap interval, Table 10 control (optional, if its data files are present) |
| `faithfulness_check.py` | Table S2: rule-based pathway consistency of the written reasoning |
| `data/kegg_{named,anon}_{train,val,test}.csv` | Benchmark splits, as published on Hugging Face (`wanglab/kegg`, `iit-patna-cse-ai/kegg-anon-global`) |
| `data/rsft_{named,anon}_predictions.csv` | GenoMorph+ (RSFT) outputs (`rft_exact_selfadaptive_best_acc[_anon]_predictions.csv`) |
| `data/rsft_named_seed2_predictions.csv` | Second-seed repeat of the final RSFT stage (named benchmark) |
| `data/rsft_from_stage15_{named,anon}_predictions.csv` | RSFT seeded directly from Stage 1.5+Gate, skipping GRPO (Table 10 control) |
| `data/bioreason_{named,anon}_generations.csv` | BioReason (SFT) outputs |
| `data/llmonly_{named,anon}_generations.csv` | LLM-only (Qwen3-1.7B) outputs |
| `logs/test_06b_eval_stage3_named.log` | Named GenoMorph (GRPO) evaluation log (Table S1, GRPO row) |

## Requirements

Python 3.9 or later with `numpy`, `pandas`, `scikit-learn` and `statsmodels`:

```bash
pip install numpy pandas scikit-learn statsmodels
```

## Run

```bash
cd analysis
python supplementary_analyses.py --data data --out supp_out --grpo_log logs/test_06b_eval_stage3_named.log
python faithfulness_check.py \
    "GenoMorph+ (RSFT), named=data/rsft_named_predictions.csv" \
    "BioReason (SFT), named=data/bioreason_named_generations.csv" \
    "LLM-only (Qwen3), named=data/llmonly_named_generations.csv" \
    "GenoMorph+ (RSFT), anonymized=data/rsft_anon_predictions.csv" \
    "BioReason (SFT), anonymized=data/bioreason_anon_generations.csv" \
    "LLM-only (Qwen3), anonymized=data/llmonly_anon_generations.csv" --out faith_out
```

Results are printed and written to `supp_out/supplementary_analyses.json` and `faith_out/` (a LaTeX table, a JSON
summary and sample traces for manual checking).

## Notes

- **Scoring.** An example counted correct uses its ground-truth label as the prediction, so accuracy and F1 come from
  the same matched-prediction set, as in the paper.
- **Bootstrap.** 2,000 resamples of the 290 evaluation examples (validation 144 + test 146), random seed 0.
- **Example order.** The RSFT files list validation then test; the BioReason and LLM-only files list test then
  validation. The script reorders them and checks that every label aligns before comparing models example by example.
- **Pathway consistency.** Only written reasoning is checked; special tokens, including latent-step markers, are
  removed first. Entities are gene and molecule symbols (or anonymous identifiers) that occur in some pathway of the
  benchmark.
