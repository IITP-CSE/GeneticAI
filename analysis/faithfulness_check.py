#!/usr/bin/env python3
"""Rule-based pathway-faithfulness check of reasoning traces (BIB-26-2135, Reviewer 3 major comment 9).

For each evaluated example, the KEGG pathway given in the question (its network definition and gene list) is compared
with the entities the model names in its written reasoning:
  * variant gene mentioned  - the starred (variant) gene of the network definition appears in the trace;
  * entity precision        - of the pathway-vocabulary entities named in the trace, the share that belong to this
                              question's pathway (1 - this share = entities from other pathways);
  * order consistency       - of all pairs of this pathway's nodes named in the trace, the share mentioned in the
                              same order as the network definition (upstream before downstream).
Only written text is checked; latent steps are invisible. The check covers entities and their order, not the
biological correctness of each stated mechanism.

Input: one or more files given as "Label=path". Accepted formats:
  * CSV with a question column (question / user_input / prompt) and a generation column
    (full_generation / generation / prediction / output / response); an optional correctness column
    (is_correct / contains_ground_truth / correct) splits the results by correct vs incorrect.
  * JSONL with the same field names.
Usage:
  python faithfulness_check.py "GenoMorph+ (RSFT), named=rft_..._predictions.csv" "BioReason (SFT), named=..." \
      --out faithfulness_out [--audit 50]
"""
import argparse, itertools, json, os, re, sys
import pandas as pd

NET = re.compile(r"Network Definition of the pathway:\s*(.*?)(?:\n|\\n|$)")
GL = re.compile(r"Genes in the pathway:\s*(.*?)(?:\n\n|\\n\\n|\nGiven|$)", re.S)
ANON = re.compile(r"(?:GENE|MOL)_\d+")
SPLIT = re.compile(r"\s*(?:->|-\||//|=>|--|\||,|\+|/|\(|\)|\[|\]|\s)\s*")
QCOLS = ["question", "user_input", "prompt", "input"]
GCOLS = ["full_generation", "generation", "prediction", "output", "response", "answer_text"]
CCOLS = ["is_correct", "contains_ground_truth", "correct"]


def nodes(net):
    out = []
    for t in SPLIT.split(net.replace("*", "")):
        if t and t not in out:
            out.append(t)
    return out


def gene_list(q):
    g = GL.search(q)
    return {x.split(";")[0].strip() for x in g.group(1).split("|")} if g else set()


def load(path):
    df = pd.read_json(path, lines=True) if path.endswith((".jsonl", ".json")) else pd.read_csv(path)
    cols = list(df.columns)
    q = next((c for c in QCOLS if c in cols), None)
    g = next((c for c in GCOLS if c in cols), None)
    if not q or not g:
        sys.exit(f"{path}: need a question column {QCOLS} and a generation column {GCOLS}; found {cols}")
    c = next((c for c in CCOLS if c in cols), None)
    ok = None
    if c:
        ok = df[c].map(lambda v: str(v).strip().lower() in ("true", "1", "yes")) if df[c].dtype == object else df[c].astype(bool)
    return df.rename(columns={q: "q", g: "g"}).assign(ok=ok)


def trace_text(t):
    t = str(t)
    if "</think>" in t:                      # BioReason-style: reasoning inside <think> ... </think>
        t = t.split("</think>")[0]
    return re.sub(r"<[^>]+>", " ", t)         # drop special tokens, incl. latent markers


def analyse(df):
    vocab = set()
    for q in df.q:
        m = NET.search(str(q))
        if m:
            vocab |= set(nodes(m.group(1)))
        vocab |= gene_list(str(q))
    vocab = {v for v in vocab if len(v) > 1 and not v.isdigit()}
    rows = []
    for i, (q, g, ok) in enumerate(zip(df.q.astype(str), df.g.fillna(""), df.ok)):
        m = NET.search(q)
        if not m:
            continue
        nd = nodes(m.group(1))
        star = re.search(r"([A-Za-z0-9_\-]+)\*", m.group(1))
        pw = set(nd) | gene_list(q)
        words = re.findall(r"[A-Za-z0-9_\-]+", trace_text(g))
        order = []
        for w in words:
            # entities = identifiers/symbols that occur in some pathway of the benchmark. Other anonymized IDs are
            # not counted: the anonymization also replaced non-pathway terms (e.g. disease abbreviations, "dopamine")
            if w in vocab and w not in order:
                order.append(w)
        pos = {n: k for k, n in enumerate(nd)}
        pairs = list(itertools.combinations([w for w in order if w in pos], 2))
        rows.append(dict(
            idx=i, ok=ok, n_entities=len(order),
            variant_mentioned=(star.group(1) in order) if star else None,
            precision=(sum(w in pw for w in order) / len(order)) if order else None,
            order_consistency=(sum(pos[a] < pos[b] for a, b in pairs) / len(pairs)) if pairs else None,
            off_pathway=[w for w in order if w not in pw], pathway=m.group(1).strip(), mentioned=order))
    return pd.DataFrame(rows)


def summarise(r):
    return dict(n=len(r),
                variant_mentioned=100 * r.variant_mentioned.dropna().astype(float).mean(),
                precision=100 * r.precision.dropna().mean(),
                order_consistency=100 * r.order_consistency.dropna().mean(),
                n_with_pairs=int(r.order_consistency.notna().sum()),
                mean_entities=r.n_entities.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help='"Label=path"')
    ap.add_argument("--out", default="faithfulness_out")
    ap.add_argument("--audit", type=int, default=50, help="examples per model exported for manual checking")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    table, report = [], {}
    for item in a.inputs:
        label, path = item.rsplit("=", 1)
        r = analyse(load(path))
        s = summarise(r)
        report[label] = {"all": s}
        if r.ok.notna().any():
            for k, sub in r.groupby(r.ok.astype(bool)):
                report[label]["correct" if k else "incorrect"] = summarise(sub)
        table.append((label, s))
        print(f"{label}: n={s['n']} | variant gene {s['variant_mentioned']:.1f}% | within-pathway entities "
              f"{s['precision']:.1f}% | order {s['order_consistency']:.1f}% (pairs in {s['n_with_pairs']}) | "
              f"entities/trace {s['mean_entities']:.1f}")
        r.sample(min(a.audit, len(r)), random_state=0).to_csv(
            os.path.join(a.out, f"audit_{re.sub(r'[^A-Za-z0-9]+', '_', label)}.csv"), index=False)
    T = ["\\begin{tabular}{lcccc}", "\\toprule",
         "Model & Variant gene named & Within-pathway entities & Pathway order & Entities per trace \\\\", "\\midrule"]
    T += [f"{l} & {s['variant_mentioned']:.1f}\\% & {s['precision']:.1f}\\% & {s['order_consistency']:.1f}\\% & "
          f"{s['mean_entities']:.1f} \\\\" for l, s in table]
    T += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(a.out, "faithfulness_table.tex"), "w").write("\n".join(T) + "\n")
    json.dump(report, open(os.path.join(a.out, "faithfulness.json"), "w"), indent=1, default=float)
    print(f"Wrote {a.out}/faithfulness_table.tex, faithfulness.json and audit_*.csv")


if __name__ == "__main__":
    main()
