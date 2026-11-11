#!/usr/bin/env python3
"""
Test a BioReason-mode (use_cross_attention=False, no gate, no LoRA) DNA-LLM checkpoint
on iit-patna-cse-ai/GenBench_coding.

GenBench_coding's schema does NOT match wanglab/kegg: modalities vary per row, and
`reasoning_chain` is a structured multi-hop graph path, not free text. Only rows tagged
with a "snp_sequence" modality carry actual DNA sequence content Evo2 can encode -- the
other seven task types (protein_sequence / expression / conservation-only reasoning)
have no nucleotide input at all and are skipped by default. Pass --ignore_modality to
evaluate every row instead: rows without a DNA sequence get empty reference_sequence/
variant_sequence, which resolves to a pure text-only prompt through the same pipeline
(DLProcessor maps each <|dna_pad|> placeholder to its DNA sequence's tokenized length --
zero for an empty sequence, so no DNA content gets injected, no special-casing needed).

Each kept row's modality_data entry with tag=="snp_sequence" has:
    payload.ref_context -> reference_sequence
    payload.alt_context -> variant_sequence
matching the reference/variant pair convention used everywhere else in this repo.

Not every row is MCQ-graded either -- rows with a `choices` list are graded on whether
the generated "Answer: X" span resolves to the ground-truth letter; rows with no
choices are open-ended, graded by substring match of the free-text answer (same
convention as NucleotideDNAModule.correctness_reward_func elsewhere in this repo).
Both kinds are evaluated automatically and reported separately.

Metrics (mcq / open_ended blocks, plus overall_accuracy across both):
    accuracy            - MCQ: exact letter match. Open-ended: substring match.
    f1_macro            - MCQ only: macro-F1 over the observed answer-letter classes.
    cosine_similarity   - mean cosine similarity between the model's own mean-pooled
                          hidden state for its generated answer span and for the
                          ground-truth text (the correct choice's text for MCQ, the raw
                          answer for open-ended) -- a soft-match signal for near-misses
                          that name the right concept without an exact-format match.

Usage:
  python eval_bioreason_genbench_coding.py \
      --checkpoint /scratch/.../bioreason/.../last.ckpt \
      --split test --limit 50 \
      --output_dir test/log --output_prefix bioreason_genbench_coding
"""
import argparse
import csv
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset

from genomorph.models.dna_llm import DNALLMModel
from genomorph.models.dl.processing_dl import DLProcessor
from genomorph.models.evo2_tokenizer import register_evo2_tokenizer
from genomorph.dataset.kegg import get_format_kegg_function, qwen_dna_collate_fn
from genomorph.dataset.utils import truncate_dna
from genomorph.dna_modules.nucleotide_module import NucleotideDNAModule

register_evo2_tokenizer()

_LETTER_RE  = re.compile(r'^\(?([A-Za-z])\)?[.\):]?\s*')
_CHOICE_RE  = re.compile(r'^([A-Za-z])[.\):]\s*(.*)$')


# ── Data: filter + reshape GenBench_coding to the kegg-format contract ────────────────

def _load_one_split(
    split: str, cache_dir: Optional[str], token: Optional[str], ignore_modality: bool
) -> List[Dict[str, Any]]:
    ds = load_dataset("iit-patna-cse-ai/GenBench_coding", split=split, cache_dir=cache_dir, token=token)

    rows = []
    n_no_snp, n_choices, n_open_ended, n_text_only = 0, 0, 0, 0
    for ex in ds:
        # modality_data elements come back as JSON-encoded strings, not parsed dicts --
        # Arrow can't natively type the heterogeneous per-tag payload structs, so HF
        # stores each entry as a JSON string (the datasets-server API preview decodes
        # these for display, which is misleading about the actual Python-side type).
        modality_data = [json.loads(m) if isinstance(m, str) else m for m in ex["modality_data"]]
        snp = next((m for m in modality_data if m.get("tag") == "snp_sequence"), None)

        if snp is not None:
            ref_seq, var_seq = snp["payload"]["ref_context"], snp["payload"]["alt_context"]
        elif ignore_modality:
            # No DNA sequence for this row (other 7 task types: protein/expression/
            # conservation-only). Empty strings -> DLProcessor resolves each <|dna_pad|>
            # placeholder to its tokenized DNA length (processing_dl.py:200-208), so an
            # empty sequence naturally renders zero dna_pad tokens -- falls through to a
            # pure text-only prompt, same pipeline, no DNA content injected.
            ref_seq, var_seq = "", ""
            n_text_only += 1
        else:
            n_no_snp += 1
            continue

        # Not every task type in this benchmark is MCQ-graded -- choices is None for
        # some rows. Those are open-ended: graded by substring match on the raw answer
        # text (same convention as NucleotideDNAModule.correctness_reward_func
        # elsewhere in this repo: `answer.lower() in extracted_response.lower()`),
        # not by letter.
        is_mcq = bool(ex.get("choices"))
        if is_mcq:
            n_choices += 1
            choices_block = "\n".join(ex["choices"])
            question = (f"{ex['question']}\nOptions:\n{choices_block}\n"
                        f"Answer directly with the letter of the correct option, "
                        f"formatted exactly as: Answer: X")
            answer = ex["answer"].strip().upper()   # ground-truth letter
        else:
            n_open_ended += 1
            question = (f"{ex['question']}\n"
                        f"Answer directly, formatted exactly as: Answer: <your answer>")
            answer = ex["answer"].strip()            # free-text ground truth, keep case

        rows.append({
            "id":                 ex["id"],
            "source_split":       split,
            "is_mcq":             is_mcq,
            "question":           question,
            "reasoning":          "",   # unused (is_sft=False); kept only so
                                          # get_format_kegg_function doesn't KeyError.
            "reference_sequence": ref_seq,
            "variant_sequence":   var_seq,
            "answer":             answer,
            "choices":            ex["choices"] or [],
        })
    print(f"  [{split}] {len(ds)} total -- kept {len(rows)} "
          f"({n_choices} MCQ, {n_open_ended} open-ended, {n_text_only} text-only), "
          f"no snp_sequence={n_no_snp}")
    return rows


def load_genbench_coding_dna_subset(
    split: str, cache_dir: Optional[str] = None, token: Optional[str] = None,
    ignore_modality: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load iit-patna-cse-ai/GenBench_coding. Returns rows shaped for
    get_format_kegg_function("dna-llm", is_sft=False): question/reference_sequence/
    variant_sequence/answer, plus `is_mcq` and (for MCQ rows) the raw `choices` list.

    Not every row in this benchmark is MCQ-graded -- rows with no `choices` are kept
    as open-ended: `answer` is the free-text ground truth, graded by substring match
    (same convention as NucleotideDNAModule.correctness_reward_func elsewhere in this
    repo) rather than by letter. main() reports MCQ and open-ended metrics separately.

    ignore_modality=False (default): only rows whose modality_data contains a
    "snp_sequence" entry (the other 7 task types have no nucleotide input at all).
    ignore_modality=True: every row regardless of modality -- rows without snp_sequence
    get empty reference_sequence/variant_sequence (a pure text-only prompt through the
    same pipeline; see the DLProcessor note above).

    split="both" concatenates train+test (mirrors the merge_val_test_set convention
    used elsewhere in this repo for small eval splits) -- useful since the
    snp_sequence-bearing subset of either split alone is small.

    This dataset is gated -- token defaults to None (ambient huggingface-cli login /
    HF_TOKEN env var), or pass --hf_token explicitly if the node running this script
    doesn't share login state with wherever `huggingface-cli login` was run.
    """
    if split == "both":
        return (_load_one_split("train", cache_dir, token, ignore_modality)
                + _load_one_split("test", cache_dir, token, ignore_modality))
    return _load_one_split(split, cache_dir, token, ignore_modality)


def choice_text_for_letter(choices: List[str], letter: str) -> Optional[str]:
    """"B. Downregulation of ERBB2:ERBB3 signaling" -> text after 'B. ' for letter 'B'."""
    for c in choices:
        m = _CHOICE_RE.match(c.strip())
        if m and m.group(1).upper() == letter.upper():
            return m.group(2).strip()
    return None


def extract_predicted_letter(generated_text: str, choices: List[str]) -> Optional[str]:
    """
    Parse the model's own 'Answer: X' span (same convention used across this repo,
    NucleotideDNAModule._extract_xml_answer) and resolve it to a choice letter.
    Falls back to matching the extracted text against each choice's own content when
    the model writes the option's text instead of its letter.
    """
    raw = NucleotideDNAModule._extract_xml_answer(generated_text).strip()
    if not raw:
        return None

    m = _LETTER_RE.match(raw)
    if m:
        letter = m.group(1).upper()
        if any(_CHOICE_RE.match(c.strip()) and _CHOICE_RE.match(c.strip()).group(1).upper() == letter
               for c in choices):
            return letter

    raw_low = raw.lower()
    for c in choices:
        cm = _CHOICE_RE.match(c.strip())
        if cm and cm.group(2).strip().lower() in raw_low:
            return cm.group(1).upper()
    return None


# ── Model ──────────────────────────────────────────────────────────────────────────────

def _merge_lora_into_base(state: Dict[str, torch.Tensor], prefix: str,
                           lora_alpha: float, lora_r: Optional[int] = None) -> None:
    """
    Collapse a PEFT-wrapped `<prefix>.base_model.model.*` sub-tree (from
    get_peft_model(text_model, LoraConfig(...)) during BioReason SFT training -- the
    LoRA config there targets every Linear layer, see get_target_modules()) back into
    plain `<prefix>.model.*` keys matching an unwrapped model, IN PLACE on `state`.

    Each LoRA-targeted Linear's effective (actually fine-tuned) weight is
        base_layer.weight + (lora_B.weight @ lora_A.weight) * (lora_alpha / r)
    Modules PEFT didn't wrap (embed_tokens, lm_head, norms) are just relocated,
    stripping the `base_model.model.` prefix. No-op if `prefix` isn't PEFT-wrapped.
    """
    wrapped_prefix = f"{prefix}.base_model.model."
    wrapped_keys = [k for k in state if k.startswith(wrapped_prefix)]
    if not wrapped_keys:
        return

    bases:  Dict[str, torch.Tensor] = {}
    lora_a: Dict[str, torch.Tensor] = {}
    lora_b: Dict[str, torch.Tensor] = {}
    passthrough: Dict[str, torch.Tensor] = {}

    for k in wrapped_keys:
        v = state.pop(k)
        rest = k[len(wrapped_prefix):]
        if rest.endswith(".base_layer.weight"):
            bases[rest[: -len(".base_layer.weight")]] = v
        elif ".lora_A." in rest and rest.endswith(".weight"):
            lora_a[rest.split(".lora_A.")[0]] = v
        elif ".lora_B." in rest and rest.endswith(".weight"):
            lora_b[rest.split(".lora_B.")[0]] = v
        else:
            passthrough[rest] = v

    n_merged = 0
    for path, base_w in bases.items():
        if path in lora_a and path in lora_b:
            r = lora_r or lora_a[path].shape[0]
            scaling = lora_alpha / r
            delta = (lora_b[path].float() @ lora_a[path].float()) * scaling
            passthrough[path + ".weight"] = (base_w.float() + delta).to(base_w.dtype)
            n_merged += 1
        else:
            passthrough[path + ".weight"] = base_w

    for rest, v in passthrough.items():
        state[f"{prefix}.{rest}"] = v

    print(f"[load] merged {n_merged} LoRA-adapted layers into base weights under '{prefix}' "
          f"(lora_alpha={lora_alpha}, {len(lora_a)} lora_A / {len(lora_b)} lora_B tensors found)")


def load_model(checkpoint: str, args) -> DNALLMModel:
    """Load a BioReason-mode (use_cross_attention=False) checkpoint. Accepts a plain
    .pt state dict or a Lightning .ckpt -- mirrors hiref_kegg_align.py::load_model_for_hiref.

    BioReason SFT wraps text_model in LoRA during training (see get_target_modules(),
    applied to every Linear layer) -- checkpoints therefore have text_model.base_model.
    model.* (PEFT-wrapped) keys, not plain text_model.model.*. _merge_lora_into_base
    collapses these back to plain keys with the LoRA deltas merged in, so what actually
    loads is the real fine-tuned weights, not just the frozen pretrained base."""
    model = DNALLMModel(
        text_model_name=args.text_model_name,
        dna_model_name=args.dna_model_name,
        cache_dir=args.cache_dir,
        max_length_text=args.max_length_text,
        max_length_dna=args.max_length_dna,
        text_model_finetune=False,
        dna_model_finetune=False,
        dna_is_evo2=True,
        dna_embedding_layer=args.dna_embedding_layer,
        use_cross_attention=False,   # BioReason: plain linear projection, no cross-attn
        use_hrpo_gate=False,
        device="cpu",
    )
    raw = torch.load(checkpoint, map_location="cpu")
    state = raw.get("state_dict", raw)
    state = {(k[6:] if k.startswith("model.") else k): v for k, v in state.items()}

    hp = raw.get("hyper_parameters", {}) or {}
    lora_alpha = float(hp.get("lora_alpha", 32))
    lora_r     = hp.get("lora_rank")
    _merge_lora_into_base(state, "text_model", lora_alpha, lora_r)

    # Qwen's config.json vocab_size (151936) is padded beyond its tokenizer's actual
    # entry count; resize_token_embeddings(len(tokenizer)) truncates a freshly built
    # model down to the real count + added DNA tokens. Older checkpoints (or ones
    # built against a different tokenizer snapshot) may still have the full padded
    # table. Resize to match whatever this checkpoint actually saved before loading,
    # rather than assume a fresh model's size is right.
    ckpt_embed = state.get("text_model.model.embed_tokens.weight")
    if ckpt_embed is not None:
        cur_size = model.text_model.get_input_embeddings().weight.shape[0]
        if ckpt_embed.shape[0] != cur_size:
            print(f"[load] resizing text_model embeddings {cur_size} -> {ckpt_embed.shape[0]} "
                  f"to match checkpoint")
            model.text_model.resize_token_embeddings(ckpt_embed.shape[0])

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  missing (first 5): {missing[:5]}")
    if unexpected:
        print(f"  unexpected (first 5): {unexpected[:5]}")
    return model


@torch.no_grad()
def mean_pooled_embedding(model: DNALLMModel, text: str, device: str) -> torch.Tensor:
    """Mean-pooled, L2-normalized last-layer hidden state for a plain text string (no
    DNA) -- a lightweight embedding for the cosine-similarity metric, reusing the same
    masked-mean pattern hiref_kegg_align.py uses for the offline HiRef embeddings."""
    ids = model.text_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    ids = {k: v.to(device) for k, v in ids.items()}
    out = model.text_model(**ids, output_hidden_states=True, use_cache=False)
    pooled = out.hidden_states[-1][0].mean(dim=0)
    return F.normalize(pooled.float(), dim=-1)


def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    try:
        from sklearn.metrics import f1_score
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    except ImportError:
        labels = sorted(set(y_true))
        f1s = []
        for lab in labels:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        return sum(f1s) / len(f1s) if f1s else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Plain .pt state dict or Lightning .ckpt.")
    p.add_argument("--split", default="test", choices=["train", "test", "both"])
    p.add_argument("--ignore_modality", action="store_true",
                   help="Include every MCQ-graded row, not just snp_sequence ones. Rows "
                        "without a DNA sequence get empty reference_sequence/"
                        "variant_sequence (pure text-only prompt through the same "
                        "pipeline -- see load_genbench_coding_dna_subset docstring).")
    p.add_argument("--limit", type=int, default=0,
                   help="Cap the number of kept rows (smoke test). 0 = no limit, run everything.")
    p.add_argument("--text_model_name", default="Qwen/Qwen3-1.7B")
    p.add_argument("--dna_model_name", default="evo2_7b_base")
    p.add_argument("--dna_embedding_layer", default="blocks.28.mlp.l3")
    p.add_argument("--max_length_text", type=int, default=6000)
    p.add_argument("--max_length_dna", type=int, default=2048)
    p.add_argument("--truncate_dna_per_side", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=600)
    p.add_argument("--no_think", action="store_true",
                   help="Skip chain-of-thought: force an empty <think></think> block "
                        "so the model must answer immediately. Matches Qwen3's "
                        "enable_thinking=False. Fixes truncated/incomplete generations "
                        "on this out-of-distribution dataset without needing a much "
                        "larger --max_new_tokens.")
    p.add_argument("--cache_dir", default=os.path.expanduser("~/.cache/huggingface"))
    p.add_argument("--hf_token", default=None,
                   help="HF token for the gated iit-patna-cse-ai/GenBench_coding dataset. "
                        "Omit to use ambient huggingface-cli login / HF_TOKEN env var.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="test/log")
    p.add_argument("--output_prefix", default="bioreason_genbench_coding")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mode_desc = "all MCQ-graded rows (ignoring modality)" if args.ignore_modality else "snp_sequence rows"
    print(f"Loading GenBench_coding[{args.split}] -- {mode_desc}...")
    rows = load_genbench_coding_dna_subset(args.split, cache_dir=args.cache_dir, token=args.hf_token,
                                            ignore_modality=args.ignore_modality)
    print(f"  kept {len(rows)} rows total")
    if args.limit > 0:
        rows = rows[: args.limit]
        print(f"  --limit applied: {len(rows)} rows")
    if not rows:
        raise RuntimeError("No usable rows found -- nothing to evaluate.")

    if args.truncate_dna_per_side:
        rows = [truncate_dna(r, truncate_dna_per_side=args.truncate_dna_per_side) for r in rows]

    print(f"Loading model from {args.checkpoint} ...")
    model = load_model(args.checkpoint, args)
    model.to(args.device)
    model.eval()

    processor = DLProcessor(tokenizer=model.text_tokenizer, dna_tokenizer=model.dna_tokenizer)
    fmt = get_format_kegg_function("dna-llm", is_sft=False)

    records: List[Dict[str, Any]] = []
    n_correct_mcq, n_correct_oe = 0, 0
    y_true_mcq: List[str] = []
    y_pred_mcq: List[str] = []
    cos_sims_mcq: List[float] = []
    cos_sims_oe: List[float] = []
    n_mcq_seen, n_oe_seen = 0, 0

    for i, row in enumerate(rows):
        print(f"  [{i + 1}/{len(rows)}] generating for {row['id']} ({'MCQ' if row['is_mcq'] else 'open-ended'}) ...",
              flush=True)
        item = fmt(row)
        batch = qwen_dna_collate_fn(
            [item], processor=processor,
            max_length_text=args.max_length_text, max_length_dna=args.max_length_dna,
            return_answer_in_batch=False, truncate_for_generation=True,
        )
        input_ids      = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        dna_tokenized  = {k: v.to(args.device) for k, v in batch["dna_tokenized"].items()}
        batch_idx_map  = batch["batch_idx_map"]

        if args.no_think:
            # Qwen3's chat template supports enable_thinking=False (renders an empty
            # <think></think> right after <|im_start|>assistant\n) but
            # qwen_dna_collate_fn/TRL's maybe_apply_chat_template doesn't expose a way
            # to pass that flag through. Same effect done manually: append the same
            # empty block's tokens directly onto the already-built prompt so the model
            # has no room to ramble and must answer immediately. batch_size=1 here, so
            # appending to the right of a left-padded sequence is safe.
            think_ids = model.text_tokenizer("<think>\n\n</think>\n\n", add_special_tokens=False,
                                              return_tensors="pt")["input_ids"].to(args.device)
            input_ids      = torch.cat([input_ids, think_ids], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(think_ids)], dim=1)

        t0 = time.time()
        out_ids = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            dna_tokenized=dna_tokenized, batch_idx_map=batch_idx_map,
            max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=model.text_tokenizer.pad_token_id,
        )
        gen_time = time.time() - t0
        gen_text = model.text_tokenizer.decode(out_ids[0], skip_special_tokens=False)
        pred_span = NucleotideDNAModule._extract_xml_answer(gen_text).strip() or gen_text

        if row["is_mcq"]:
            n_mcq_seen += 1
            prediction   = extract_predicted_letter(gen_text, row["choices"])
            ground_truth = row["answer"]
            is_correct   = prediction == ground_truth
            n_correct_mcq += int(is_correct)
            y_true_mcq.append(ground_truth)
            y_pred_mcq.append(prediction or "")
            cos_target = choice_text_for_letter(row["choices"], ground_truth) or ground_truth
        else:
            n_oe_seen += 1
            prediction   = pred_span
            ground_truth = row["answer"]
            # Same convention as NucleotideDNAModule.correctness_reward_func: substring
            # match, case-insensitive -- no letter to match against for open-ended rows.
            is_correct   = ground_truth.lower() in prediction.lower()
            n_correct_oe += int(is_correct)
            cos_target = ground_truth

        emb_pred = mean_pooled_embedding(model, pred_span, args.device)
        emb_true = mean_pooled_embedding(model, cos_target, args.device)
        cos_sim  = torch.dot(emb_pred, emb_true).item()
        (cos_sims_mcq if row["is_mcq"] else cos_sims_oe).append(cos_sim)

        records.append({
            "id": row["id"], "source_split": row["source_split"], "is_mcq": row["is_mcq"],
            "question": row["question"], "ground_truth": ground_truth, "prediction": prediction,
            "is_correct": is_correct, "cosine_similarity": round(cos_sim, 4),
            "gen_time_sec": round(gen_time, 3), "full_generation": gen_text,
        })

        running_acc = ((n_correct_mcq + n_correct_oe) / (n_mcq_seen + n_oe_seen))
        print(f"  [{i + 1}/{len(rows)}] done in {gen_time:.1f}s -- "
              f"pred={prediction!r} true={ground_truth!r} correct={is_correct} "
              f"running_acc={running_acc:.4f}", flush=True)
        if prediction is None:
            # Show what the model actually said whenever letter-extraction failed
            # entirely -- otherwise "why is pred None" requires digging through the
            # CSV every time. A wrong-but-extracted MCQ answer isn't logged here;
            # that's an expected outcome, not an extraction failure.
            print(f"    raw generation (last 300 chars): ...{gen_text[-300:]!r}", flush=True)

    metrics: Dict[str, Any] = {"n_examples": len(rows)}
    if n_mcq_seen:
        metrics["mcq"] = {
            "n":                     n_mcq_seen,
            "accuracy":              round(n_correct_mcq / n_mcq_seen, 4),
            "f1_macro":              round(macro_f1(y_true_mcq, y_pred_mcq), 4),
            "cosine_similarity_mean": round(sum(cos_sims_mcq) / len(cos_sims_mcq), 4),
        }
    if n_oe_seen:
        metrics["open_ended"] = {
            "n":                     n_oe_seen,
            "accuracy":              round(n_correct_oe / n_oe_seen, 4),
            "cosine_similarity_mean": round(sum(cos_sims_oe) / len(cos_sims_oe), 4),
        }
    metrics["overall_accuracy"] = round((n_correct_mcq + n_correct_oe) / len(rows), 4)
    print(json.dumps(metrics, indent=2))

    out_csv  = os.path.join(args.output_dir, f"{args.output_prefix}_predictions.csv")
    out_json = os.path.join(args.output_dir, f"{args.output_prefix}_metrics.json")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Predictions -> {out_csv}")
    print(f"Metrics     -> {out_json}")


if __name__ == "__main__":
    main()
