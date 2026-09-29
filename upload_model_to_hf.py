#!/usr/bin/env python3
"""
Upload a GenoMorph checkpoint directory to the Hugging Face Hub as a model repo.

This is NOT a transformers.PreTrainedModel -- DNALLMModel wraps a plain Qwen3ForCausalLM
and an Evo2 (StripedHyena) encoder loaded via the separate `evo2` package, neither of which
fits AutoModel.from_pretrained() cleanly. This script uploads the raw checkpoint file(s) plus
a model card that shows how to load them with this repo's own DNALLMModel class -- not a
plug-and-play AutoModel.from_pretrained(repo_id).

Expects a checkpoint directory as written by train_latent_sft.py's checkpoint-saving code
(train/train_07_selfadaptive.sh's best/ or best_acc/, or a GRPO checkpoint dir):
    model.pt            (required)  -- full DNALLMModel state dict
    thinking_gate.pt     (optional) -- ThinkingResidualGate state dict
    dna_injector.pt       (optional) -- DNAHiddenInjector state dict

Usage:
  python upload_model_to_hf.py \
      --checkpoint_dir /scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_07_selfadaptive/best_acc \
      --repo_id iit-patna-cse-ai/genomorph-rsft \
      [--token hf_xxx]      # optional if already logged in via huggingface-cli
      [--private]           # default is public
      [--variant "Self-Adaptive RSFT"]
"""

import argparse
import os

from huggingface_hub import HfApi, create_repo

MODEL_CARD_TEMPLATE = """---
license: mit
tags:
  - genomics
  - dna
  - biology
  - qwen3
  - evo2
---

# GenoMorph -- {variant}

Genomic disease-reasoning DNA-LLM checkpoint from *GenoMorph: Pathway-Grounded Genomic
Disease Reasoning via Adaptive Latent Computation* (Halder, Ghosh, Roy, Saha; IIT Patna;
Briefings in Bioinformatics, Problem Solving Protocol, in production).

- Code: https://github.com/IITP-CSE/GeneticAI
- Training data: https://huggingface.co/datasets/iit-patna-cse-ai/kegg-anon-global

## Files in this repo

{files_list}

## This is NOT a plug-and-play `AutoModel.from_pretrained()` checkpoint

`DNALLMModel` wraps a plain `Qwen3ForCausalLM` and an Evo2 (StripedHyena) encoder loaded via
the separate `evo2` package -- neither fits the standard `transformers` auto-loading
interface. Load it with this repo's own model class (see
[IITP-CSE/GeneticAI](https://github.com/IITP-CSE/GeneticAI)):

```python
import torch
from huggingface_hub import snapshot_download
from genomorph.models.dna_llm import DNALLMModel
from genomorph.models.evo2_tokenizer import register_evo2_tokenizer

register_evo2_tokenizer()
ckpt_dir = snapshot_download("{repo_id}")

model = DNALLMModel(
    text_model_name="Qwen/Qwen3-1.7B",
    dna_model_name="evo2_7b_base",
    dna_is_evo2=True,
    dna_embedding_layer="blocks.28.mlp.l3",
    use_cross_attention=True,
    use_hrpo_gate=True,   # only if thinking_gate.pt / dna_injector.pt are present above
)
model.load_state_dict(torch.load(f"{{ckpt_dir}}/model.pt", map_location="cpu"))
```

`eval_selfadaptive_exact.py` / `train/rsft/eval_selfadaptive_exact.sh` in the code repo show
the exact loading + generation path used to reproduce the paper's reported numbers.

## Citation

```bibtex
@article{{halder_genomorph,
  title   = {{GenoMorph: Pathway-Grounded Genomic Disease Reasoning via Adaptive Latent Computation}},
  author  = {{Halder, Tanmoy Kanti and Ghosh, Akash and Roy, Arijit and Saha, Sriparna}},
  journal = {{Briefings in Bioinformatics}},
  note    = {{Problem Solving Protocol. In production.}}
}}
```
"""


def upload(checkpoint_dir: str, repo_id: str, token: str = None, private: bool = False,
           variant: str = "Self-Adaptive RSFT"):
    candidate_files = ["model.pt", "thinking_gate.pt", "dna_injector.pt"]
    files = [f for f in candidate_files if os.path.exists(os.path.join(checkpoint_dir, f))]
    if "model.pt" not in files:
        raise FileNotFoundError(f"No model.pt found in {checkpoint_dir}")

    api = HfApi(token=token)
    create_repo(repo_id, repo_type="model", private=private, token=token, exist_ok=True)

    for fname in files:
        size_mb = os.path.getsize(os.path.join(checkpoint_dir, fname)) / (1024 * 1024)
        print(f"Uploading {fname} ({size_mb:.1f} MB) ...")
        api.upload_file(
            path_or_fileobj=os.path.join(checkpoint_dir, fname),
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )

    files_list = "\n".join(f"- `{f}`" for f in files)
    card = MODEL_CARD_TEMPLATE.format(variant=variant, files_list=files_list, repo_id=repo_id)
    readme_path = os.path.join(checkpoint_dir, "_upload_README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(card)
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    os.remove(readme_path)

    print(f"\nUploaded {len(files)} checkpoint file(s) -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a GenoMorph checkpoint to the HF Hub.")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory containing model.pt (+ thinking_gate.pt, dna_injector.pt).")
    parser.add_argument("--repo_id", required=True,
                        help="HF Hub repo to create, e.g. iit-patna-cse-ai/genomorph-rsft")
    parser.add_argument("--token", default=None, help="HF write token (optional if already logged in).")
    parser.add_argument("--private", action="store_true", help="Make the model repo private.")
    parser.add_argument("--variant", default="Self-Adaptive RSFT",
                        help="Short label for the model card, e.g. 'Self-Adaptive RSFT' or "
                             "'GRPO (learned theta_low)'.")
    args = parser.parse_args()
    upload(args.checkpoint_dir, args.repo_id, args.token, args.private, args.variant)
