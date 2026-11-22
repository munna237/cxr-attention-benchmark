"""Seed-level cross-check of the attention effects (Supplementary Table S6).

Reads results/round3/per_seed_all_runs.csv (macro F1 of every run at the selected checkpoint) and,
for each of the 16 main and 8 robustness attention effects, computes a paired t-test on the five
per-seed differences and an exact sign-flip permutation test over the 2^5 sign assignments,
with Holm correction within each family. Also reports the seed-level t-interval of the pooled
(mean of 16) main effect. Usage:  python scripts/seed_level_crosscheck.py
"""
import itertools, os
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, '..', 'results', 'round3', 'per_seed_all_runs.csv'))
d['variant'] = d.variant.fillna('')
AL = {'resnet50': 'ResNet50', 'densenet121': 'DenseNet121', 'mobilenetv2': 'MobileNetV2', 'customcnn': 'Custom CNN'}
TL = {'se': 'SE', 'cbam': 'CBAM'}
f = lambda a, t, c, v='': d[(d.arch == a) & (d.attention == t) & (d.condition == c) & (d.variant == v)].set_index('seed').f1_best

def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, (m - k) * p[i]); adj[i] = min(1, run)
    return adj

def signflip(x):
    x = np.asarray(x); obs = abs(x.mean())
    return np.mean([abs((x * np.array(s)).mean()) >= obs - 1e-12 for s in itertools.product([1, -1], repeat=len(x))])

def row(family, label, x):
    n = len(x); se = x.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(.975, n - 1)
    return dict(family=family, effect=label, n=n, delta=x.mean(), sd=x.std(ddof=1), t_lo=x.mean() - q * se, t_hi=x.mean() + q * se,
                p_t=stats.ttest_1samp(x, 0).pvalue, p_perm=signflip(x.values))

rows, main_diffs = [], []
for c in ['imbalanced', 'balanced']:
    for a in AL:
        for t in TL:
            x = (f(a, t, c) - f(a, 'none', c)).dropna(); main_diffs.append(x)
            rows.append(row('main', f'{TL[t]}, {AL[a]}, {c}', x))
for a, v, lab, bv in [('resnet50', '_lungcrop-True', 'lung crop', '_lungcrop-True'), ('resnet50', '_val-grouped', 'grouped validation', '_val-grouped'),
                      ('resnet50', '_placement-multi', 'multi-depth', ''), ('densenet121', '_pretrain-xrv', 'CXR pretraining', '_pretrain-xrv')]:
    for t in TL:
        rows.append(row('robustness', f'{TL[t]}, {AL[a]}, {lab}', (f(a, t, 'imbalanced', v) - f(a, 'none', 'imbalanced', bv)).dropna()))
R = pd.DataFrame(rows)
for fam in ['main', 'robustness']:
    m = R.family == fam; R.loc[m, 'p_t_holm'] = holm(R.loc[m, 'p_t']); R.loc[m, 'p_perm_holm'] = holm(R.loc[m, 'p_perm'])
print(R.round(4).to_string())
P = pd.concat(main_diffs, axis=1).mean(axis=1); n = len(P); se = P.std(ddof=1) / np.sqrt(n)
print(f'\npooled main effect: {P.mean():+.4f}  95% t-CI [{P.mean() - stats.t.ppf(.975, n - 1) * se:+.4f}, {P.mean() + stats.t.ppf(.975, n - 1) * se:+.4f}]'
      f'  90% t-CI [{P.mean() - stats.t.ppf(.95, n - 1) * se:+.4f}, {P.mean() + stats.t.ppf(.95, n - 1) * se:+.4f}]')
R.to_csv(os.path.join(HERE, '..', 'results', 'supplementary', 'table_s6_seed_level_crosscheck.csv'), index=False)
