# GenoMorph — Paper

> **Update 29-Sep-2026:** Paper archived on arXiv.
> **Update 01-Sep-2026:** Paper submitted to *Briefings in Bioinformatics*.

**GenoMorph: Pathway-Grounded Genomic Disease Reasoning via Adaptive Latent Computation**

Tanmoy Kanti Halder<sup>1,2</sup>, Akash Ghosh<sup>1</sup>, Arijit Roy<sup>1</sup>, Sriparna Saha<sup>1,\*</sup>

<sup>1</sup> Department of Computer Science and Engineering, Indian Institute of Technology Patna, Bihta, 801106, Bihar, India
<sup>2</sup> Prasannadeb Women's College, India

\* Corresponding author: sriparna@iitp.ac.in

[![arXiv](https://img.shields.io/badge/arXiv-2609.34079-FF6B6B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.34079)
[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-kegg--anon--global-FFBF00?style=for-the-badge)](https://huggingface.co/datasets/iit-patna-cse-ai/kegg-anon-global)
[![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-GenoMorph-FFBF00?style=for-the-badge)](https://huggingface.co/iit-patna-cse-ai/GenoMorph)

Preprint available on arXiv: [arXiv:2609.34079](https://arxiv.org/abs/2609.34079) (cs.AI; cross-listed q-bio.GN). Submitted to *Briefings in Bioinformatics* (Problem Solving Protocol format) — not yet accepted or peer-reviewed.

## Abstract

Large language models (LLMs) have demonstrated strong capabilities in biological reasoning; however, genomic disease inference remains largely dependent on memorized gene-disease associations rather than understanding of biological pathways. This shortcut learning undermines robustness and generalization, and it breaks down entirely when explicit molecular identifiers are unavailable. To overcome these limitations, we present GenoMorph, a multimodal genomic reasoning framework that shifts disease prediction from associative gene–disease mapping toward pathway-grounded reasoning. GenoMorph couples a frozen DNA foundation model with question-conditioned cross-attention fusion, self-adaptive latent reasoning (LatentSp), a residual reasoning gate for iterative genomic evidence reinjection, and rejection sampling fine-tuning regularized by hierarchical optimal transport (OT). Instead of learning direct mappings between genes and diseases, the proposed framework progressively aligns genomic sequence representations with latent pathway dynamics, enabling reasoning trajectories that follow underlying molecular interactions before producing disease predictions. Furthermore, the self-adaptive latent reasoning mechanism dynamically allocates computation according to reasoning confidence, reducing unnecessary latent reasoning steps and substantially improving inference efficiency. To understand the impact of our framework, we build an anonymized benchmark from the Kyoto Encyclopedia of Genes and Genomes (KEGG) pathway database that swaps every gene and molecular identifier for globally consistent anonymous symbols while preserving sequences and pathway topology, removing memorization shortcuts. GenoMorph raises the weighted F1 from 0.7863 (BioReason) to 0.9412, and rejection sampling fine-tuning with self-adaptive latent reasoning pushes it to 0.9725 while cutting latency nearly 60%. On the anonymized benchmark it reaches 0.9465 F1, substantially outperforming prior systems and confirming that accurate disease prediction can arise from pathway reasoning rather than memorized gene–disease associations.

**Keywords:** genomic reasoning, DNA foundation models, mechanistic inference, cross-attention fusion, adaptive latent reasoning, hierarchical optimal transport, residual reasoning gate, rejection sampling fine-tuning, gene-name anonymization

## Framework

GenoMorph is trained through a four-stage pipeline:

1. **Supervised Fine-Tuning (SFT)** — question-conditioned `CrossAttentionFusion` for genomic–language alignment, curriculum-based `LatentSp` for latent reasoning initialization, and `ThinkingResidualGate` pretraining for controlled genomic evidence reinjection.
2. **Offline Hierarchical Optimal Transport (HiRef-OT)** — freezes the Stage-1 model and estimates the geometric discrepancy between genomic and disease-answer representations, producing a fixed target manifold used as a geometry-aware reward signal.
3. **Group Relative Policy Optimization (GRPO)** — jointly optimizes prediction accuracy, reasoning quality, biological consistency, and adaptive latent computation through a multi-objective reward informed by the HiRef signal.
4. **Rejection Sampling Fine-Tuning (RSFT)** — filters GRPO-sampled reasoning trajectories with a two-step selection rule (correct + well-formed, then prefer latent-using/shortest) and fine-tunes on the survivors, consolidating the adaptive reasoning policy into the model parameters so inference needs no explicit online entropy estimation.

## Key results

| Model | Named Acc. | Named Weighted-F1 | Anon Acc. | Anon Weighted-F1 | Time (s), Named / Anon |
|---|---|---|---|---|---|
| LLM-only (Qwen3-1.7B) | 0.8897 | 0.7461 | 0.4931 | 0.4256 | 21.45 / 25.93 |
| BioReason (SFT) | 0.9069 | 0.7863 | 0.6034 | 0.6032 | 23.74 / 28.83 |
| BioReason (GRPO) | 0.8275 | 0.8550 | 0.5172 | 0.4812 | 25.91 / 25.37 |
| GenoMorph | 0.9552 | 0.9412 | 0.9207 | 0.9285 | 18.31 / 23.64 |
| **GenoMorph+ (Self-Adaptive RSFT)** | **0.9759** | **0.9725** | **0.9483** | **0.9465** | **9.75 / 10.73** |

GenoMorph+ retains 97.2% of its named-benchmark accuracy after full gene-name anonymization (vs. 55.4% for a text-only Qwen3-1.7B baseline), indicating that its predictions are driven by genomic sequence and pathway reasoning rather than memorized gene identities. Full ablations, Pareto-frontier efficiency analysis, and qualitative case studies are in the paper.

## Files

- [`GenoMorph_Briefings_in_Bioinformatics.pdf`](./GenoMorph_Briefings_in_Bioinformatics.pdf) — the manuscript.

## Code and data

- **Code** (this repository): [github.com/IITP-CSE/GeneticAI](https://github.com/IITP-CSE/GeneticAI)
- **Benchmark** (original + anonymized KEGG): [huggingface.co/datasets/iit-patna-cse-ai/kegg-anon-global](https://huggingface.co/datasets/iit-patna-cse-ai/kegg-anon-global)
- **Model checkpoint**: [huggingface.co/iit-patna-cse-ai/GenoMorph](https://huggingface.co/iit-patna-cse-ai/GenoMorph)

## Citation

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

## Acknowledgments

The authors gratefully acknowledge the Aryabhatta Supercomputing Centre (ASC) at the Indian Institute of Technology Patna, established under the National Supercomputing Mission (NSM), Government of India, for providing the computational resources utilized in this work.
