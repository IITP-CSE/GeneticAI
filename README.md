# GenoMorph

> **Update 29-Sep-2026:** Paper archived on arXiv.
> **Update 01-Sep-2026:** Paper submitted to *Briefings in Bioinformatics*.

[![GitHub](https://img.shields.io/badge/GitHub-Code-4A90E2?style=for-the-badge&logo=github&logoColor=white)](https://github.com/IITP-CSE/GeneticAI)
[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-kegg--anon--global-FFBF00?style=for-the-badge)](https://huggingface.co/datasets/iit-patna-cse-ai/kegg-anon-global)
[![arXiv](https://img.shields.io/badge/arXiv-2609.34079-FF6B6B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.34079)
[![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-GenoMorph-FFBF00?style=for-the-badge)](https://huggingface.co/iit-patna-cse-ai/GenoMorph)

**GenoMorph: Pathway-Grounded Genomic Disease Reasoning via Adaptive Latent Computation**
Tanmoy Kanti Halder, Akash Ghosh, Arijit Roy, Sriparna Saha — IIT Patna. Preprint: [arXiv:2609.34079](https://arxiv.org/abs/2609.34079). Submitted to *Briefings in Bioinformatics* (Problem Solving Protocol format); not yet accepted or peer-reviewed. Manuscript, abstract, and citation: [`paper/README.md`](paper/README.md).

GenoMorph extends [BioReason](https://arxiv.org/abs/2505.23579) (Fallahpour et al., 2025) — a DNA-LLM model that fuses a frozen Evo2-7B encoder with Qwen3-1.7B via a static linear projection — with question-conditioned cross-attention fusion, self-adaptive latent reasoning (LatentSp), a residual reasoning gate for iterative genomic evidence reinjection, hierarchical optimal-transport (HiRef-OT) regularized GRPO, and rejection-sampling fine-tuning (RSFT) that distills the learned adaptive policy back into the model. On the KEGG benchmark this raises weighted-F1 from 0.7863 (BioReason) to 0.9725 while cutting inference latency by ~60%, and the framework retains 0.9465 weighted-F1 on a fully gene-name-anonymized benchmark — evidence that predictions follow pathway reasoning rather than memorized gene–disease associations. Full results, ablations, and qualitative analysis are in the paper.

> **Baseline**: `src/` contains the vendored original BioReason code (prior work, static linear projection, no cross-attention/gate/latent machinery). All GenoMorph contributions live in the root-level scripts and the `genomorph/` package.

---

## Repository Structure

```
GeneticAI/
├── paper/                              # Published manuscript + README/citation
│
├── genomorph/                          # GenoMorph package (our contributions)
│   ├── models/
│   │   ├── dna_llm.py                 # DNALLMModel + CrossAttentionFusion
│   │   ├── latent_reasoning.py        # GateNet + HRPO adaptive gate loop (Coconut-style)
│   │   ├── thinking_residual.py       # ThinkingResidualGate + DNAHiddenInjector
│   │   └── evo2_tokenizer.py          # HF-compatible Evo2 tokenizer wrapper
│   ├── hiref/                          # Hierarchical OT: HR_OT, FRLC, rank_annealing, ...
│   ├── trainer/                        # GRPO trainer + config (TRL-based)
│   ├── dataset/
│   │   ├── kegg.py                    # KEGG dataset loader (named + anonymized CSV)
│   │   └── global_stage1_anon_genes_mol_keep_chr.csv   # anonymized KEGG dataset
│   └── dna_modules/                    # NucleotideDNAModule: reward functions, answer extraction
│
├── src/                                 # Vendored BioReason baseline (prior work)
│   ├── bioreason/                      # Original BioReason package (static linear projection)
│   ├── train_dna_qwen.py               # BioReason SFT training script
│   └── scripts/{real,anon}/            # BioReason train/test launchers
│
├── train_dna_qwen.py                   # Stage 1: CrossAttn SFT entry point
├── train_latent_sft.py                 # Stage 1 LatentSp/Gate curriculum, and Stage 4 RSFT (--rsft_traces)
├── hiref_kegg_align.py                 # Stage 2: offline HiRef-OT manifold computation
├── train_grpo_latent_reasoning.py      # Stage 3: GRPO, fixed θ_low schedule
├── train_grpo_learned_theta.py         # Stage 3 (Option B): GRPO, SPSA-searched θ_low
├── eval_grpo_checkpoint_final.py       # GRPO / self-adaptive checkpoint eval, RSFT trace sampling
├── eval_selfadaptive_exact.py          # Exact-reproduction eval for Stage 4 RSFT checkpoints
├── eval_bioreason_genbench_coding.py   # BioReason eval on iit-patna-cse-ai/GenBench_coding
├── build_anon_dataset.py               # Build the anonymized KEGG CSV
├── upload_anon_dataset.py              # Push the anon CSV to the HF Hub (kegg-anon-global)
│
├── train/                               # Named-dataset training launchers (train_00 .. train_07)
│   ├── rsft/                           # Stage 4 pipeline: sample_traces → filter_traces → (gold_traces) → train_07
│   └── anon/                           # Anonymized-dataset mirror of train/ (incl. anon/rsft/)
│
├── test/                                # Named-dataset eval launchers
│   └── anon/                            # Anonymized-dataset mirror of test/
│
├── baselines/                             # Zero-shot baseline evals: GPT-4o-mini, Gemini, Claude,
│                                         # BioMedGPT, BioMistral, Meditron (local + anon variants)
│
└── RESULTS.md                           # Per-stage script/config reference for ablation runs
```

---

## Installation

### Requirements
- Python 3.11+
- CUDA 12.8
- SLURM cluster (scripts use `sbatch`; can also run with `bash` directly)

### Environment setup

```bash
# Create conda environment
conda create -n dna_env python=3.11 -y
conda activate dna_env

# Install PyTorch with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install GenoMorph package and dependencies
pip install -e .

# Install Evo2 DNA encoder
pip install -e ".[evo2]"
```

### Verified stack
(from `pyproject.toml`)
```
torch==2.9.0+cu128
transformers==4.57.6
trl==1.0.0
vllm==0.11.2
pytorch_lightning>=2.4.0
accelerate>=1.2.0
peft>=0.14.0
```

---

## Dataset

The KEGG variant-disease dataset (named) is loaded automatically from HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("wanglab/kegg")   # train / val / test splits
```

### Anonymized dataset

To test whether models rely on memorized gene names rather than DNA/pathway signal, every gene and molecular identifier is replaced with a globally consistent anonymous symbol (e.g. `GENE_1`, `MOL_1`) while DNA sequences and pathway topology are left unchanged. The anonymized benchmark is hosted on the HF Hub — the same one cited in the paper's Data Availability statement:

```python
ds = load_dataset("iit-patna-cse-ai/kegg-anon-global")
```

A local copy is also kept in-repo:
```
genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv
```

To rebuild it from scratch, or push a rebuilt copy back to the HF Hub:

```bash
python build_anon_dataset.py \
    --output genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv \
    --keep_chromosome \
    --anonymize_molecules \
    --seed 42

python upload_anon_dataset.py \
    --csv_path genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv \
    --repo_id  iit-patna-cse-ai/kegg-anon-global
```

---

## Training Pipeline

Four sequential stages, matching the paper (Section "GenoMorph Architecture"). Named and anonymized runs use the same scripts under `train/` and `train/anon/` respectively, and can be launched in parallel once a prerequisite checkpoint exists.

### Stage 1 — Supervised Fine-Tuning

Establishes the core reasoning architecture in three steps: `CrossAttentionFusion` for question-conditioned genomic alignment, curriculum-based `LatentSp` for latent reasoning initialization, and `ThinkingResidualGate` pretraining for genomic evidence reinjection.

```bash
# 1. CrossAttn SFT
bash train/train_02_stage1_sft.sh 0,1          # 2 GPUs

# 2. LatentSp curriculum (entropy-adaptive latent token replacement)
STAGE1_CKPT=<stage1 checkpoint> bash train/train_03b_stage1_50_cached.sh

# 3. ThinkingResidualGate + DNAHiddenInjector pretraining
STAGE15_CKPT=<stage1.5 model.pt> bash train/train_04_stage1_51.sh
```

Key settings (Table 1 in the paper): Qwen3-1.7B-Instruct backbone, frozen Evo2-7B encoder (`blocks.28.mlp.l3`), CrossAttentionFusion + LoRA (r=32, α=64) trainable, AdamW @ 5e-5, 6000-token text / 2048 bp DNA context (±1024 bp around the variant), BF16, 5 epochs.

### Stage 2 — Offline Hierarchical Optimal Transport (HiRef-OT)

Freezes the Stage-1 model and computes a fixed target manifold + Monge correspondence between genomic and disease-answer representations. The live OT distance to this fixed target is what actually feeds Stage 3's reward — see `paper/README.md` for the exact offline-vs-live distinction.

```bash
STAGE1_CKPT=<stage1.5.1 model.pt> bash train/train_05_stage2_hiref.sh
```

### Stage 3 — Group Relative Policy Optimization (GRPO)

Jointly optimizes correctness, format, reasoning quality, HiRef-OT alignment, and latent-computation efficiency via an 8-term weighted reward (see paper Eq. for `R(y)`). Two variants exist for the θ_low latent-decision threshold:

```bash
# Fixed schedule (θ_low ramped to a hand-set target)
STAGE1_CKPT=<...> STAGE2_DIR=<...> bash train/train_06_stage3_grpo.sh 0,1

# Option B: θ_low optimized online via a derivative-free sliding-window (SPSA) search
# — this is the variant that produced the reported GenoMorph-B / RSFT results.
# A plain REINFORCE update was tried and found uninformative here (GRPO's per-group
# advantage is zero-mean and the latent decision is shared across a group, so no
# per-example credit signal survives) — see the paper's Adaptive Latent Computation section.
STAGE1_CKPT=<...> STAGE2_DIR=<...> bash train/train_06b_stage3_grpo_optB.sh 0,1
```

Key settings (Table 2): LoRA (r=16, α=32), AdamW @ 2e-6, 8 generations/prompt, 800-token max completion, temperature 0.7 / top-p 0.95 / top-k 50, initial θ_low=1.0, θ_high=3.0, BF16.

### Stage 4 — Rejection Sampling Fine-Tuning (RSFT)

Samples multiple completions per prompt from the converged Stage-3 policy, keeps only trajectories that are correct **and** well-formed (a hard gate — no reward score is thresholded), prefers latent-using and then shortest among survivors, and fine-tunes the full backbone (no LoRA) on the selected trajectories — internalizing the adaptive latent policy so inference needs no online entropy estimation.

```bash
# 1. Sample candidate trajectories from the converged GRPO checkpoint
CKPT=<healthy GRPO checkpoint> bash train/rsft/sample_traces.sh 0,1

# 2. Filter to correct + well-formed, preferring latent-using / shortest
TRACES=<...>_traces.jsonl bash train/rsft/filter_traces.sh

# (optional) backfill prompts no sampled trace got right, from KEGG's own reasoning
INDICES=<uncovered.txt> bash train/rsft/gold_traces.sh

# 3. Fine-tune on the selected trajectories
STAGE1_CKPT=<GRPO checkpoint dir> RSFT_TRACES=<filtered.jsonl> bash train/train_07_selfadaptive.sh 0
```

Key settings (Table 3): initialized from the best GRPO checkpoint (LoRA merged), full-backbone fine-tune, AdamW @ 1e-5, FP16, 4 epochs.

---

## Testing and Evaluation

```bash
# BioReason baseline (vendored, src/)
bash src/scripts/real/sh_test_bioreason_sft.sh

# LLM-only baseline (no DNA encoder)
bash test/test_01_llm_only.sh

# Stage 1 / Stage 1.5 checkpoint sweeps + best-only eval
bash test/test_04_eval_stage1_5.sh        # sweep
bash test/test_04c_eval_stage1_5_best.sh  # best checkpoint only

# Stage 3 GRPO — one-directional and bidirectional scoring
CKPT=<checkpoint dir> bash test/test_06b_eval_stage3.sh
CKPT=<checkpoint dir> bash test/test_06b_eval_stage3_final.sh

# Stage 4 RSFT — exact-reproduction eval (reuses train_latent_sft.py's own
# evaluate_accuracy(), matches a checkpoint's reported number exactly)
CKPT=<...>/train_07_selfadaptive/best_acc bash train/rsft/eval_selfadaptive_exact.sh

# Zero-shot baselines (GPT-4o-mini, Gemini, Claude, BioMedGPT, BioMistral, Meditron)
bash baselines/run_eval_gpt4o_mini_local.sh
```

All named scripts above have an `anon/` counterpart (`test/anon/...`, `train/anon/rsft/...`) that runs the identical evaluation against the anonymized benchmark. See `RESULTS.md` for the full script/checkpoint-path reference per ablation row, and `eval_bioreason_genbench_coding.py` for cross-dataset testing against the unrelated multi-task `iit-patna-cse-ai/GenBench_coding` benchmark.

---

## Results

Full KEGG test+val set (290 samples). **Named** = original benchmark with explicit gene identifiers. **Anon** = every gene/molecular identifier replaced with a globally consistent anonymous symbol; DNA sequences and pathway topology unchanged.

| Model | Named Acc. | Named Weighted-F1 | Named Time (s) | Anon Acc. | Anon Weighted-F1 | Anon Time (s) |
|---|---|---|---|---|---|---|
| LLM-only (Qwen3-1.7B) | 0.8897 | 0.7461 | 21.45 | 0.4931 | 0.4256 | 25.93 |
| BioReason (SFT) | 0.9069 | 0.7863 | 23.74 | 0.6034 | 0.6032 | 28.83 |
| BioReason (GRPO) | 0.8275 | 0.8550 | 25.91 | 0.5172 | 0.4812 | 25.37 |
| GenoMorph (GRPO, learned θ_low) | 0.9552 | 0.9412 | 18.31 | 0.9207 | 0.9285 | 23.64 |
| **GenoMorph+ (Self-Adaptive RSFT)** | **0.9759** | **0.9725** | **9.75** | **0.9483** | **0.9465** | **10.73** |

Zero-shot LLM/biomedical-LM baselines (GPT-4o, Gemini, BioMedGPT, BioMistral, Meditron) score far lower on both benchmarks (weighted-F1 ≤ 0.26 named, ≤ 0.04 anon) — see the paper's Table 5 for the full breakdown.

**Robustness to gene-name anonymization** (accuracy retention = anon accuracy / named accuracy):

| Model | Δ Accuracy ↓ | Retention (%) ↑ |
|---|---|---|
| LLM-only (Qwen3) | 39.66 | 55.4 |
| BioReason (SFT) | 30.35 | 66.5 |
| GenoMorph | 3.45 | 96.4 |
| **GenoMorph+ (RSFT)** | **2.76** | **97.2** |

GenoMorph's small anonymization gap indicates it reasons primarily from genomic sequence and pathway evidence rather than memorized gene identities. The paper's ablation table (Table 7) further isolates the contribution of each component — CrossAttentionFusion, LatentSp, the gate, HiRef-OT, and self-adaptive RSFT each measurably improve the accuracy/inference-time trade-off, with removing HiRef-OT alone dropping named accuracy from 95.52% to 80.00%.

---

## Citation

If you use GenoMorph, please cite:

```bibtex
@misc{halder2026genomorph,
  title   = {GenoMorph: Pathway-Grounded Genomic Disease Reasoning via Adaptive Latent Computation},
  author  = {Halder, Tanmoy Kanti and Ghosh, Akash and Roy, Arijit and Saha, Sriparna},
  year    = {2026},
  eprint  = {2609.34079},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url     = {https://arxiv.org/abs/2609.34079},
  note    = {Submitted to Briefings in Bioinformatics (Problem Solving Protocol format);
             not yet accepted or peer-reviewed.}
}
```

and the BioReason paper this work builds on:

```bibtex
@misc{fallahpour2025bioreason,
  title   = {BioReason: Incentivizing Multimodal Biological Reasoning within a DNA-LLM Model},
  author  = {Adibvafa Fallahpour and Andrew Magnuson and Purav Gupta and Shihao Ma
             and Jack Naimer and Arnav Shah and Haonan Duan and Omar Ibrahim
             and Hani Goodarzi and Chris J. Maddison and Bo Wang},
  year    = {2025},
  eprint  = {2505.23579},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url     = {https://arxiv.org/abs/2505.23579}
}
```

---

## References

[1] Fallahpour et al., *BioReason: Incentivizing Multimodal Biological Reasoning within a DNA-LLM Model*, arXiv:2505.23579, 2025.
[2] Halmos et al., *Hierarchical Refinement: Optimal Transport to Infinity and Beyond*, arXiv:2503.03025, 2025.
[3] Kanehisa and Goto, *KEGG: Kyoto Encyclopedia of Genes and Genomes*, Nucleic Acids Research 28(1):27–30, 2000.
