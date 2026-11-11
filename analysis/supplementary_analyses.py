#!/usr/bin/env python3
"""Supplementary analyses for GenoMorph (BIB-26-2135, revision): Supplementary Tables S1-S4 and the
held-out-variant rows of Table 9. CPU only; computed from saved per-example outputs, no model is run.

Inputs (one directory, default ./data):
  kegg_{named,anon}_{train,val,test}.csv          benchmark splits (Hugging Face: wanglab/kegg, iit-patna-cse-ai/kegg-anon-global)
  rsft_{named,anon}_predictions.csv               GenoMorph+ (RSFT) exact-eval outputs (validation then test order)
  bioreason_{named,anon}_generations.csv          BioReason (SFT) test_generations outputs (test then validation order)
  llmonly_{named,anon}_generations.csv            LLM-only (Qwen3) test_generations outputs (test then validation order)
  rsft_named_seed2_predictions.csv (optional)     repeat of the final RSFT stage with seed 2 (named)
  rsft_from_stage15_{named,anon}_predictions.csv (optional)  RSFT seeded from Stage 1.5+Gate, skipping GRPO (Table 10)
Optional:
  --grpo_log <file>   named GenoMorph (GRPO) evaluation log with per-example "[+]/[-] gt=... pred=..." lines (S1 row)

Usage:
  python supplementary_analyses.py --data data --out supp_out [--grpo_log test_06b_eval_stage3_*.log]
Pathway consistency (Table S2) is computed by faithfulness_check.py, imported here for its bootstrap interval.

Scoring follows the paper: an example counted correct uses its ground-truth label as the prediction, so accuracy and
F1 come from the same matched-prediction set. Bootstrap: 2,000 resamples of the 290 evaluation examples, seed 0.
"""
import argparse, json, os, re
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import mcnemar

B, SEED = 2000, 0
norm = lambda s: re.sub(r'\s+', ' ', str(s).strip().lower())
truthy = lambda v: str(v).strip().lower() in ('true', '1', 'yes')


def macro(g, p):
    return f1_score(g, p, labels=sorted(set(g)), average='macro', zero_division=0)


def weighted(g, p):
    return f1_score(g, p, labels=sorted(set(g)), average='weighted', zero_division=0)


def boot(g, p, ok, seed=SEED):
    rng = np.random.default_rng(seed); n = len(g)
    idx = [rng.integers(0, n, n) for _ in range(B)]
    acc = np.array([ok[i].mean() for i in idx]); mac = np.array([macro(g[i], p[i]) for i in idx])
    return acc, mac


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def load(d, t):
    tr, va, te = (pd.read_csv(os.path.join(d, f'kegg_{t}_{s}.csv')) for s in ('train', 'val', 'test'))
    ev = pd.concat([va, te], ignore_index=True)                       # evaluation order: validation, then test
    gt = ev.answer.map(norm).values
    swap = lambda df: pd.concat([df.iloc[len(te):], df.iloc[:len(te)]], ignore_index=True)  # test+val -> val+test
    r = pd.read_csv(os.path.join(d, f'rsft_{t}_predictions.csv'))
    b = swap(pd.read_csv(os.path.join(d, f'bioreason_{t}_generations.csv')))
    l = swap(pd.read_csv(os.path.join(d, f'llmonly_{t}_generations.csv')))
    M = {}
    for name, df, okc, prc, genc in [('GenoMorph+ (RSFT)', r, 'is_correct', 'predicted_answer', 'full_generation'),
                                     ('BioReason (SFT)', b, 'contains_ground_truth', 'predicted_label', 'generation'),
                                     ('LLM-only (Qwen3)', l, 'correct', 'predicted_label', 'generation')]:
        assert (df.ground_truth.map(norm).values == gt).all(), f'{name} ({t}): example order does not match the split'
        ok = df[okc].map(truthy).values
        pred = np.where(ok, gt, df[prc].fillna('none').map(norm).values)
        M[name] = dict(ok=ok, pred=pred, gen=df[genc].fillna('').astype(str).values)
    return tr, va, te, ev, gt, M


def grpo_from_log(path):
    L = open(path).read()
    m = re.findall(r"\[rank\d\s+\d+/\d+\] \[([+-])\]\s+gt=['\"](.*?)['\"]\s+pred=['\"](.*?)['\"]\s+latent", L)
    g = np.array([norm(x[1]) for x in m]); ok = np.array([x[0] == '+' for x in m])
    return g, np.where(ok, g, np.array([norm(x[2]) for x in m])), ok


def words(text):
    return len(re.findall(r'\S+', re.sub(r'<[^>]+>', ' ', text)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='data'); ap.add_argument('--out', default='supp_out'); ap.add_argument('--grpo_log')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    R = {'S1': {}, 'S1b': {}, 'heldout': {}, 'S3': {}, 'S4': {}}
    for t, T in (('named', 'Named'), ('anon', 'Anonymized')):
        tr, va, te, ev, gt, M = load(a.data, t)
        boots = {}
        for name, m in M.items():
            acc, mac = boot(gt, m['pred'], m['ok']); boots[name] = acc
            R['S1'][f'{name}, {T}'] = dict(acc=float(m['ok'].mean()), acc_ci=ci(acc), macro=macro(gt, m['pred']),
                                           macro_ci=ci(mac), weighted=weighted(gt, m['pred']))
        gm = M['GenoMorph+ (RSFT)']['ok']
        seen = ev.variant_sequence.astype(str).isin(set(tr.variant_sequence.astype(str))).values
        R['heldout'][T] = {'n_heldout': int((~seen).sum())}
        for name in ('BioReason (SFT)', 'LLM-only (Qwen3)'):
            o = M[name]['ok']
            for key, mask in (('all', np.ones_like(seen)), ('heldout', ~seen)):
                x, y = gm[mask], o[mask]
                tab = [[int((x & y).sum()), int((x & ~y).sum())], [int((~x & y).sum()), int((~x & ~y).sum())]]
                res = dict(only_gm=tab[0][1], only_baseline=tab[1][0], p=float(mcnemar(tab, exact=True).pvalue))
                if key == 'all': R['S1b'].setdefault(T, {})[name] = res
                else: R['heldout'][T][f'McNemar vs {name}'] = res
            d = boots['GenoMorph+ (RSFT)'] - boots[name]
            R['S1b'][T][name].update(diff=float(gm.mean() - o.mean()), diff_ci=ci(d))
        for name, m in M.items():
            R['heldout'][T][name] = dict(all=float(m['ok'].mean()), heldout=float(m['ok'][~seen].mean()))
            w = np.array([words(s) for s in m['gen']])
            R['S4'][f'{name}, {T}'] = dict(mean=float(w.mean()), median=float(np.median(w)),
                                           q1=float(np.percentile(w, 25)), q3=float(np.percentile(w, 75)))
        if t == 'named':
            c = {s: df.answer.value_counts() for s, df in (('train', tr), ('val', va), ('test', te))}
            tab = pd.DataFrame(c).fillna(0).astype(int); tab['total'] = tab.sum(axis=1)
            R['S3'] = tab.sort_values('total', ascending=False).to_dict(orient='index')
    # Table S2 footnote: within-pathway gap, anonymized, GenoMorph+ (RSFT) vs BioReason (SFT)
    import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import faithfulness_check as fc
        _, _, _, _, _, M = load(a.data, 'anon'); ev_q = pd.concat([pd.read_csv(os.path.join(a.data, f'kegg_anon_{s}.csv'))
                                                                  for s in ('val', 'test')], ignore_index=True).question
        prec = {}
        for name in ('GenoMorph+ (RSFT)', 'BioReason (SFT)'):
            r = fc.analyse(pd.DataFrame({'q': ev_q.values, 'g': M[name]['gen'], 'ok': M[name]['ok']}))
            prec[name] = r.precision.dropna().values
        rng = np.random.default_rng(SEED); x, y = prec['GenoMorph+ (RSFT)'], prec['BioReason (SFT)']
        d = np.array([x[rng.integers(0, len(x), len(x))].mean() - y[rng.integers(0, len(y), len(y))].mean() for _ in range(B)])
        R['S2_gap_anon'] = dict(gap=float(x.mean() - y.mean()), ci=ci(d))
        print(f"S2  within-pathway gap (anonymized): {100*(x.mean()-y.mean()):+.1f} pp [{100*ci(d)[0]:.1f}, {100*ci(d)[1]:.1f}]")
    except ImportError:
        print('S2  faithfulness_check.py not found next to this script; gap skipped')
    # Repeat of the final RSFT stage with a second seed (named benchmark), if its predictions are present
    p2 = os.path.join(a.data, 'rsft_named_seed2_predictions.csv')
    if os.path.exists(p2):
        tr, va, te, ev, gt, M = load(a.data, 'named')
        d2 = pd.read_csv(p2)
        assert (d2.ground_truth.map(norm).values == gt).all(), 'seed-2 file: example order does not match the split'
        ok2 = d2.is_correct.map(truthy).values
        pr2 = np.where(ok2, gt, d2.predicted_answer.fillna('none').map(norm).values)
        acc, mac = boot(gt, pr2, ok2)
        R['S1']['GenoMorph+ (RSFT), seed 2, Named'] = dict(acc=float(ok2.mean()), acc_ci=ci(acc), macro=macro(gt, pr2),
                                                          macro_ci=ci(mac), weighted=weighted(gt, pr2))
        seen = ev.variant_sequence.astype(str).isin(set(tr.variant_sequence.astype(str))).values
        R['seed2'] = {'heldout_acc': float(ok2[~seen].mean())}
        for name in ('GenoMorph+ (RSFT)', 'BioReason (SFT)', 'LLM-only (Qwen3)'):
            o = M[name]['ok']
            tab = [[int((ok2 & o).sum()), int((ok2 & ~o).sum())], [int((~ok2 & o).sum()), int((~ok2 & ~o).sum())]]
            R['seed2'][f'vs {name}'] = dict(only_seed2=tab[0][1], only_other=tab[1][0], p=float(mcnemar(tab, exact=True).pvalue))
        print('Seed 2 (RSFT, named):', json.dumps(R['seed2']))
    # Table 10 control: RSFT seeded directly from Stage 1.5+Gate, skipping GRPO (if its predictions are present)
    for t, T in (('named', 'Named'), ('anon', 'Anonymized')):
        pc = os.path.join(a.data, f'rsft_from_stage15_{t}_predictions.csv')
        if not os.path.exists(pc):
            continue
        tr, va, te, ev, gt, M = load(a.data, t)
        dc = pd.read_csv(pc)
        assert (dc.ground_truth.map(norm).values == gt).all(), f'{pc}: example order does not match the split'
        okc = dc.is_correct.map(truthy).values
        prc = np.where(okc, gt, dc.predicted_answer.fillna('none').map(norm).values)
        pa = dc.predicted_answer.fillna('').astype(str).str.strip()
        empty = ((pa == '') | pa.str.lower().isin(['none', 'nan'])).values
        nlat = dc.full_generation.fillna('').astype(str).str.count('<start-latent>').values
        o = M['GenoMorph+ (RSFT)']['ok']
        tab = [[int((okc & o).sum()), int((okc & ~o).sum())], [int((~okc & o).sum()), int((~okc & ~o).sum())]]
        R.setdefault('stage15_control', {})[T] = dict(
            acc=float(okc.mean()), macro=macro(gt, prc), weighted=weighted(gt, prc), errors=int((~okc).sum()),
            empty=int(empty.sum()), empty_errors=int((empty & ~okc).sum()),
            latent_blocks_in_empty=[int(nlat[empty].min()), int(nlat[empty].max())] if empty.any() else None,
            vs_reported=dict(only_control=tab[0][1], only_reported=tab[1][0], p=float(mcnemar(tab, exact=True).pvalue)))
        print(f'RSFT from Stage 1.5+Gate ({T}):', json.dumps(R['stage15_control'][T]))
    if a.grpo_log:
        g, p, ok = grpo_from_log(a.grpo_log); acc, mac = boot(g, p, ok)
        R['S1']['GenoMorph (GRPO), Named'] = dict(acc=float(ok.mean()), acc_ci=ci(acc), macro=macro(g, p),
                                                  macro_ci=ci(mac), weighted=weighted(g, p))
    json.dump(R, open(os.path.join(a.out, 'supplementary_analyses.json'), 'w'), indent=1)
    print('S1  accuracy and macro-F1 with bootstrap 95% CI')
    for k, v in R['S1'].items():
        print(f"  {k:38s} acc {v['acc']:.4f} [{v['acc_ci'][0]:.4f}, {v['acc_ci'][1]:.4f}]  macro {v['macro']:.4f} "
              f"[{v['macro_ci'][0]:.4f}, {v['macro_ci'][1]:.4f}]  weighted {v['weighted']:.4f}")
    print('S1b paired comparison with GenoMorph+ (RSFT)')
    for T, d in R['S1b'].items():
        for k, v in d.items():
            print(f"  {T:10s} vs {k:16s} only GM {v['only_gm']:3d} / only baseline {v['only_baseline']:2d}  p={v['p']:.2g}  "
                  f"diff {100*v['diff']:+.1f} pp [{100*v['diff_ci'][0]:.1f}, {100*v['diff_ci'][1]:.1f}]")
    print('Table 9  held-out variant sequences')
    for T, d in R['heldout'].items():
        print(f"  {T}: {d['n_heldout']} examples")
        for k, v in d.items():
            if k == 'n_heldout': continue
            if 'all' in v: print(f"    {k:18s} all {100*v['all']:.2f}%  held-out {100*v['heldout']:.2f}%")
            else: print(f"    {k}: only GM {v['only_gm']} / only baseline {v['only_baseline']}, p={v['p']:.2g}")
    print('S4  written words per output (mean / median / IQR)')
    for k, v in R['S4'].items():
        print(f"  {k:32s} {v['mean']:.0f} / {v['median']:.0f} / {v['q1']:.0f}-{v['q3']:.0f}")
    print(f"S3  class distribution: {len(R['S3'])} classes (see JSON)")


if __name__ == '__main__':
    main()
