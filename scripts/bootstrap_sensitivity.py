"""Paired hierarchical bootstrap of the attention effects (sensitivity analysis; Supplementary Tables S4-S5).
Same procedure as notebook 08 (B = 2,000; stratified image resamples from RNG seed 12345, shared by all contrasts;
one seed resample per replicate shared within each pooled estimate), but MobileNetV2 effects are computed against
the dropout-matched baseline (variant _head_dropout-0.0); effects against the original baseline are added as a
separate family. Input: outputs/runs/<run_id>/{metrics.json,preds.csv}. Output: results/round5/bootstrap/."""
import os, glob, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results', 'round5', 'bootstrap'); os.makedirs(OUT, exist_ok=True)
N_BOOT, CKPT, MARGINS = 2000, 'best', [0.01, 0.02]
RNG = np.random.default_rng(12345)
CLASS_NAMES = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']; K = 4
ARCH_ORDER = ['resnet50', 'densenet121', 'mobilenetv2', 'customcnn']
ARCH_LABEL = {'resnet50': 'ResNet50', 'densenet121': 'DenseNet121', 'mobilenetv2': 'MobileNetV2', 'customcnn': 'Custom CNN'}
ATT_LABEL = {'none': 'Baseline', 'se': 'SE', 'cbam': 'CBAM'}

RUNS, ref = {}, None
for f in sorted(glob.glob(os.path.join(ROOT, 'outputs', 'runs', '*', 'metrics.json'))):
    m = json.load(open(f))
    if m.get('status') != 'done' or m.get('smoke_test'): continue
    P = pd.read_csv(os.path.join(os.path.dirname(f), 'preds.csv'))
    rec = dict(meta=m, variant=''.join(f'_{k}-{v}' for k, v in sorted(m.get('opts', {}).items())))
    for ck in ['best', 'final']:
        d = P[(P.split == 'test') & (P.ckpt == ck)].sort_values('idx')
        rec[ck] = d[[f'p_{c}' for c in CLASS_NAMES]].values.argmax(1); rec['y'] = d.y_true.values
        if ref is None: ref = d.relpath.tolist()
        assert d.relpath.tolist() == ref, f"{m['run_id']}: test order differs"
    RUNS[m['run_id']] = rec
print(len(RUNS), 'runs')
Y = RUNS[next(iter(RUNS))]['y']
CFG = {}
for rid, r in RUNS.items(): CFG.setdefault((r['meta']['arch'], r['meta']['attention'], r['meta']['condition'], r['variant']), {})[r['meta']['seed']] = rid
seeds_of = lambda c: sorted(CFG.get(c, {}))

def macro_f1(cm):
    cm = cm.astype(float); tp = np.diagonal(cm, axis1=-2, axis2=-1); fp = cm.sum(-2) - tp; fn = cm.sum(-1) - tp
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.nan_to_num(tp / (tp + fp)); r = np.nan_to_num(tp / (tp + fn)); f = np.nan_to_num(2 * p * r / (p + r))
    return f.mean(-1)
def cm_of(y, p, idx=None):
    if idx is None: return np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    code = y[idx] * K + p[idx]; off = (np.arange(idx.shape[0]) * K * K)[:, None]
    return np.bincount((code + off).ravel(), minlength=idx.shape[0] * K * K).reshape(-1, K, K)
strata = [np.where(Y == c)[0] for c in range(K)]
BOOT = np.concatenate([s[RNG.integers(0, len(s), (N_BOOT, len(s)))] for s in strata], axis=1)   # same draw as notebooks 03/08/09
POINT = {rid: macro_f1(cm_of(Y, r[CKPT])) for rid, r in RUNS.items()}
BOOTM = {rid: macro_f1(cm_of(Y, r[CKPT], BOOT)) for rid, r in RUNS.items()}

def contrast_dist(pairs, seeds, pick=None):
    S = len(seeds)
    if pick is None: pick = RNG.integers(0, S, (N_BOOT, S))
    tb = np.zeros(N_BOOT); tp = 0.0
    for a, b, w in pairs:
        Da = np.stack([BOOTM[CFG[a][s]] for s in seeds]); Db = np.stack([BOOTM[CFG[b][s]] for s in seeds])
        tb += w * (Da - Db)[pick, np.arange(N_BOOT)[:, None]].mean(1)
        tp += w * np.mean([POINT[CFG[a][s]] - POINT[CFG[b][s]] for s in seeds])
    return tp, tb
def summarise(label, point, d, extra=None):
    r = dict(effect=label, delta=point, ci95_lo=np.percentile(d, 2.5), ci95_hi=np.percentile(d, 97.5),
             ci90_lo=np.percentile(d, 5), ci90_hi=np.percentile(d, 95), n_le0=int((d <= 0).sum()), n_ge0=int((d >= 0).sum()), B=len(d),
             p_boot=min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    for m in MARGINS:
        r[f'equiv_{m}'] = bool(r['ci90_lo'] > -m and r['ci90_hi'] < m); r[f'p_tost_{m}'] = max((d <= -m).mean(), (d >= m).mean())
    if extra: r.update(extra)
    return r

base_of = lambda a, c: (a, 'none', c, '_head_dropout-0.0') if a == 'mobilenetv2' else (a, 'none', c, '')
MAIN = [((a, t, c, ''), base_of(a, c), f'{ATT_LABEL[t]} | {ARCH_LABEL[a]} | {c}', a, t, c)
        for c in ['imbalanced', 'balanced'] for a in ARCH_ORDER for t in ['se', 'cbam']]
MBDROP = [((a, t, c, ''), (a, 'none', c, ''), f'{ATT_LABEL[t]} | {ARCH_LABEL[a]} | {c} | original baseline', a, t, c)
          for c in ['imbalanced', 'balanced'] for a in ['mobilenetv2'] for t in ['se', 'cbam']]
ROB = []
for a, v, lab in [('resnet50', '_lungcrop-True', 'lung crop'), ('resnet50', '_val-grouped', 'grouped validation'),
                  ('resnet50', '_placement-multi', 'multi-depth'), ('densenet121', '_pretrain-xrv', 'CXR pretraining')]:
    for t in ['se', 'cbam']:
        ROB.append(((a, t, 'imbalanced', v), (a, 'none', 'imbalanced', v if v != '_placement-multi' else ''), f'{ATT_LABEL[t]} | {ARCH_LABEL[a]} | {lab}', a, t, lab))

per = []
for fam, items in [('main', MAIN), ('robustness', ROB), ('mbdrop', MBDROP)]:
    for att, base, lab, a, t, c in items:
        seeds = sorted(set(seeds_of(att)) & set(seeds_of(base))); pt, d = contrast_dist([(att, base, 1.0)], seeds)
        per.append(summarise(lab, pt, d, dict(family=fam, n_seeds=len(seeds))))
EQ = pd.DataFrame(per); EQ.to_csv(f'{OUT}/equivalence_per_contrast.csv', index=False)

pooled = []
def pool(label, items, seeds=None):
    seeds = seeds or sorted(set.intersection(*[set(seeds_of(x[0])) & set(seeds_of(x[1])) for x in items]))
    pt, d = contrast_dist([(x[0], x[1], 1.0 / len(items)) for x in items], seeds)
    pooled.append(summarise(label, pt, d, dict(n_effects=len(items), n_seeds=len(seeds))))
pool('All 16 main effects', MAIN)
pool('Main: SE (8)', [x for x in MAIN if x[4] == 'se']); pool('Main: CBAM (8)', [x for x in MAIN if x[4] == 'cbam'])
pool('Main: imbalanced (8)', [x for x in MAIN if x[5] == 'imbalanced']); pool('Main: balanced (8)', [x for x in MAIN if x[5] == 'balanced'])
pool('Main: pretrained backbones (12)', [x for x in MAIN if x[3] != 'customcnn'])
for a in ARCH_ORDER: pool(f'Main: {ARCH_LABEL[a]} (4)', [x for x in MAIN if x[3] == a])
pool('All 8 robustness effects', ROB)
pool('All 16 main effects, original MobileNetV2 baseline', [x for x in MAIN if x[3] != 'mobilenetv2'] + MBDROP)
pool('12 main effects, MobileNetV2 excluded', [x for x in MAIN if x[3] != 'mobilenetv2'])
# without the collapsed custom-CNN + SE run (seed 1): drop that seed for the two custom-CNN imbalanced effects
cc = [x for x in MAIN if x[3] == 'customcnn' and x[5] == 'imbalanced']; rest = [x for x in MAIN if x not in cc]
pa, da = contrast_dist([(x[0], x[1], 1.0) for x in rest], [42, 0, 1, 2, 3])
pc, dc = contrast_dist([(x[0], x[1], 1.0) for x in cc], [42, 0, 2, 3])
pooled.append(summarise('All 16 main effects, without collapsed run', (pa + pc) / 16, (da + dc) / 16, dict(n_effects=16)))
POOL = pd.DataFrame(pooled); POOL.to_csv(f'{OUT}/pooled_attention.csv', index=False)
pd.set_option('display.width', 250)
print(POOL.round(4).to_string()); print(EQ.round(4).to_string())
