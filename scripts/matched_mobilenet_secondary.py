"""Secondary analyses that involve the MobileNetV2 baseline, recomputed with the dropout-matched baseline
(variant _head_dropout-0.0), plus the Normal-vs-Pneumonia sub-endpoint and the checkpoint-selection statistics.
- Table 6 rows for the matched baseline (accuracy, macro F1, hierarchical-bootstrap CI as in notebook 03);
- Table 9: MobileNetV2 balancing effect of the matched baseline and attention x balancing interactions
  (notebook-03 procedure, RNG seed 12345); Holm within the BALANCING (12) and INTERACTION (8) families,
  combined with the published p-values of the other backbones (results/main_analysis/effects_labeled.csv);
- Table 14 / calibration: ECE, classwise ECE, Brier and NLL attention effects of MobileNetV2 against the matched
  baseline (notebook-06 procedure, RNG seed 2026); Holm within each metric over the 24 effects, combined with the
  published values of the other 20 effects (results/calibration/calibration_attention_effects.csv);
- Normal-vs-Pneumonia binary sub-endpoint: test images whose true class is Normal or Pneumonia, predicted Pneumonia
  when p(Pneumonia) > p(Normal); balanced accuracy and AUC of p(P)/(p(N)+p(P)); 16 attention effects at the seed level;
- Checkpoint selection (Section IV-E) over the 157 main-study runs.
Input: outputs/runs/. Output: results/round5/secondary/."""
import os, glob, json, itertools
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results', 'round5', 'secondary'); os.makedirs(OUT, exist_ok=True)
N_BOOT, N_BINS, K = 2000, 15, 4
CLASS_NAMES = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']
AL = {'resnet50': 'ResNet50', 'densenet121': 'DenseNet121', 'mobilenetv2': 'MobileNetV2', 'customcnn': 'Custom CNN'}
TL = {'none': 'Baseline', 'se': 'SE', 'cbam': 'CBAM'}
MB = '_head_dropout-0.0'

RUNS, ref = {}, None
for f in sorted(glob.glob(os.path.join(ROOT, 'outputs', 'runs', '*', 'metrics.json'))):
    m = json.load(open(f)); P = pd.read_csv(os.path.join(os.path.dirname(f), 'preds.csv'))
    rec = dict(meta=m, variant=''.join(f'_{k}-{v}' for k, v in sorted(m.get('opts', {}).items())))
    for sp in ['val', 'test']:
        for ck in ['best', 'final']:
            d = P[(P.split == sp) & (P.ckpt == ck)].sort_values('idx')
            rec[(sp, ck)] = np.clip(d[[f'p_{c}' for c in CLASS_NAMES]].values, 1e-12, 1); rec[(sp, 'y')] = d.y_true.values
    t = P[(P.split == 'test') & (P.ckpt == 'best')].sort_values('idx').relpath.tolist()
    if ref is None: ref = t
    assert t == ref
    RUNS[m['run_id']] = rec
Y = RUNS[next(iter(RUNS))][('test', 'y')]
CFG = {}
for rid, r in RUNS.items(): CFG.setdefault((r['meta']['arch'], r['meta']['attention'], r['meta']['condition'], r['variant']), {})[r['meta']['seed']] = rid
seeds_of = lambda c: sorted(CFG.get(c, {}))
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, (m - k) * p[i]); adj[i] = min(1, run)
    return adj
def cm_of(y, p, idx=None):
    if idx is None: return np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    code = y[idx] * K + p[idx]; off = (np.arange(idx.shape[0]) * K * K)[:, None]
    return np.bincount((code + off).ravel(), minlength=idx.shape[0] * K * K).reshape(-1, K, K)
def metrics_batch(cm):
    cm = cm.astype(float); tp = np.diagonal(cm, axis1=-2, axis2=-1); fp = cm.sum(-2) - tp; fn = cm.sum(-1) - tp
    with np.errstate(divide='ignore', invalid='ignore'):
        pr = np.nan_to_num(tp / (tp + fp)); rc = np.nan_to_num(tp / (tp + fn)); f1 = np.nan_to_num(2 * pr * rc / (pr + rc))
    return {'acc': tp.sum(-1) / cm.sum((-1, -2)), 'macro_f1': f1.mean(-1)}

# ── notebook-03 bootstrap (RNG 12345): Table 6 rows, balancing, interaction ─────
RNG = np.random.default_rng(12345)
strata = [np.where(Y == c)[0] for c in range(K)]
BOOT = np.concatenate([s[RNG.integers(0, len(s), (N_BOOT, len(s)))] for s in strata], axis=1)
POINT = {rid: metrics_batch(cm_of(Y, r[('test', 'best')].argmax(1))) for rid, r in RUNS.items()}
BOOTM = {rid: metrics_batch(cm_of(Y, r[('test', 'best')].argmax(1), BOOT)) for rid, r in RUNS.items()}
def hier(cfg, metric):
    s = seeds_of(cfg); M = np.stack([BOOTM[CFG[cfg][x]][metric] for x in s]); pick = RNG.integers(0, len(s), (N_BOOT, len(s)))
    return M[pick, np.arange(N_BOOT)[:, None]].mean(1)
def summ(cfg, metric):
    v = np.array([POINT[CFG[cfg][s]][metric] for s in seeds_of(cfg)]); b = hier(cfg, metric)
    return dict(mean=v.mean(), sd=v.std(ddof=1), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5))
rows = []
for cond in ['imbalanced', 'balanced']:
    c = ('mobilenetv2', 'none', cond, MB); a = summ(c, 'acc'); f = summ(c, 'macro_f1')
    rows.append(dict(condition=cond, acc_mean=a['mean'], acc_sd=a['sd'], f1_mean=f['mean'], f1_sd=f['sd'], f1_lo=f['lo'], f1_hi=f['hi']))
pd.DataFrame(rows).to_csv(f'{OUT}/table6_matched_baseline.csv', index=False)

def boot_contrast(terms, seeds):
    """terms: list of (cfg, sign). Seed-paired, one seed resample per replicate."""
    pick = RNG.integers(0, len(seeds), (N_BOOT, len(seeds)))
    S = sum(sg * np.stack([BOOTM[CFG[c][s]]['macro_f1'] for s in seeds]) for c, sg in terms)
    d = S[pick, np.arange(N_BOOT)[:, None]].mean(1)
    pv = np.array([sum(sg * POINT[CFG[c][s]]['macro_f1'] for c, sg in terms) for s in seeds])
    return dict(delta_mean=pv.mean(), n_seeds=len(seeds), seeds_positive=int((pv > 0).sum()), ci_lo=np.percentile(d, 2.5),
                ci_hi=np.percentile(d, 97.5), p_boot=min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
new = {}
mb_b, mb_i = ('mobilenetv2', 'none', 'balanced', MB), ('mobilenetv2', 'none', 'imbalanced', MB)
new['Balancing effect | MobileNetV2 Baseline'] = boot_contrast([(mb_b, 1), (mb_i, -1)], seeds_of(mb_b))
for t in ['se', 'cbam']:
    tb, ti = ('mobilenetv2', t, 'balanced', ''), ('mobilenetv2', t, 'imbalanced', '')
    new[f'Attention×Balancing | MobileNetV2 {TL[t]}  [(bal−imb)_att − (bal−imb)_base]'] = boot_contrast([(tb, 1), (ti, -1), (mb_b, -1), (mb_i, 1)], seeds_of(tb))
E = pd.read_csv(os.path.join(ROOT, 'results', 'main_analysis', 'effects_labeled.csv'))
E = E[E.family.isin(['BALANCING', 'INTERACTION'])].copy(); E['baseline'] = 'original'
for k, v in new.items():
    i = E.index[E.effect == k][0]
    for kk, vv in v.items(): E.loc[i, kk] = vv
    E.loc[i, 'baseline'] = 'dropout-matched'
for fam in ['BALANCING', 'INTERACTION']:
    m = E.family == fam; E.loc[m, 'p_holm'] = holm(E.loc[m, 'p_boot'])
E[['family', 'effect', 'baseline', 'delta_mean', 'n_seeds', 'seeds_positive', 'ci_lo', 'ci_hi', 'p_boot', 'p_holm']].to_csv(f'{OUT}/balancing_interaction.csv', index=False)

# ── notebook-06 calibration (RNG 2026) ─────────────────────────────────────────
RNG = np.random.default_rng(2026)
BOOTC = np.concatenate([s[RNG.integers(0, len(s), (N_BOOT, len(s)))] for s in strata], axis=1); ALL = np.arange(len(Y))[None, :]
def ece_rows(conf, corr, idx, nb=N_BINS):
    b = np.minimum((conf * nb).astype(int), nb - 1); R = idx.shape[0]; off = (np.arange(R) * nb)[:, None]; key = (b[idx] + off).ravel()
    sc = np.bincount(key, weights=conf[idx].ravel(), minlength=R * nb).reshape(R, nb)
    sk = np.bincount(key, weights=corr[idx].ravel(), minlength=R * nb).reshape(R, nb)
    return np.abs(sc - sk).sum(1) / idx.shape[1]
def cal_metrics(prob, idx):
    pred = prob.argmax(1); conf = prob.max(1); corr = (pred == Y).astype(float); oh = np.eye(K)[Y]
    return {'ece': ece_rows(conf, corr, idx), 'cw_ece': np.mean([ece_rows(prob[:, c], (Y == c).astype(float), idx) for c in range(K)], 0),
            'brier': ((prob - oh) ** 2).sum(1)[idx].mean(1), 'nll': (-np.log(prob[np.arange(len(Y)), Y]))[idx].mean(1)}
CP = {rid: {k: v[0] for k, v in cal_metrics(r[('test', 'best')], ALL).items()} for rid, r in RUNS.items()}
CB = {rid: cal_metrics(r[('test', 'best')], BOOTC) for rid, r in RUNS.items()}
C = pd.read_csv(os.path.join(ROOT, 'results', 'calibration', 'calibration_attention_effects.csv')); C['baseline'] = 'original'
for cond in ['imbalanced', 'balanced']:
    base = ('mobilenetv2', 'none', cond, MB)
    for t in ['se', 'cbam']:
        att = ('mobilenetv2', t, cond, ''); sd = sorted(set(seeds_of(att)) & set(seeds_of(base)))
        for k in ['ece', 'cw_ece', 'brier', 'nll']:
            D = np.stack([CB[CFG[att][s]][k] - CB[CFG[base][s]][k] for s in sd]); pick = RNG.integers(0, len(sd), (N_BOOT, len(sd)))
            d = D[pick, np.arange(N_BOOT)[:, None]].mean(1); pv = np.array([CP[CFG[att][s]][k] - CP[CFG[base][s]][k] for s in sd])
            i = C.index[(C.metric == k) & (C.effect == f'{TL[t]} − Baseline | MobileNetV2 | {cond}')][0]
            C.loc[i, ['delta', 'ci_lo', 'ci_hi', 'n_seeds', 'seeds_positive', 'p_boot']] = [pv.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), len(sd), int((pv > 0).sum()), min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))]
            C.loc[i, 'baseline'] = 'dropout-matched'
for k, idx in C.groupby('metric').groups.items(): C.loc[idx, 'p_holm'] = holm(C.loc[idx, 'p_boot'])
C['CI_excludes_0'] = (C.ci_lo > 0) | (C.ci_hi < 0)
C.to_csv(f'{OUT}/calibration_attention_effects_matched.csv', index=False)
cal_tab = []
for cond in ['imbalanced', 'balanced']:
    rids = CFG[('mobilenetv2', 'none', cond, MB)].values()
    cal_tab.append(dict(condition=cond, **{k: np.mean([CP[r][k] for r in rids]) for k in ['ece', 'cw_ece', 'brier', 'nll']}))
pd.DataFrame(cal_tab).to_csv(f'{OUT}/calibration_matched_baseline.csv', index=False)

# ── Normal-vs-Pneumonia sub-endpoint ───────────────────────────────────────────
NP = np.isin(Y, [1, 2]); yb = (Y[NP] == 2).astype(int)
def np_metrics(rid, ck='best'):
    p = RUNS[rid][('test', ck)][NP]; s = p[:, 2] / (p[:, 1] + p[:, 2]); pred = (s > 0.5).astype(int)
    sens = (pred[yb == 1] == 1).mean(); spec = (pred[yb == 0] == 0).mean()
    return dict(bacc=(sens + spec) / 2, auc=roc_auc_score(yb, s), sens=sens, spec=spec)
NPM = {(rid, ck): np_metrics(rid, ck) for rid in RUNS for ck in ['best', 'final']}
def tci(x, lvl=.95):
    n = len(x); se = x.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(1 - (1 - lvl) / 2, n - 1); return x.mean() - q * se, x.mean() + q * se
def signflip(x):
    x = np.asarray(x); obs = abs(x.mean()); return np.mean([abs((x * np.array(s)).mean()) >= obs - 1e-12 for s in itertools.product([1, -1], repeat=len(x))])
rows = []
for ck in ['best', 'final']:
    for metric in ['bacc', 'auc']:
        for cond in ['imbalanced', 'balanced']:
            for a in AL:
                base = (a, 'none', cond, MB if a == 'mobilenetv2' else '')
                for t in ['se', 'cbam']:
                    att = (a, t, cond, ''); sd = sorted(set(seeds_of(att)) & set(seeds_of(base)))
                    x = pd.Series([NPM[(CFG[att][s], ck)][metric] - NPM[(CFG[base][s], ck)][metric] for s in sd])
                    lo, hi = tci(x)
                    rows.append(dict(ckpt=ck, metric=metric, arch=a, att=t, cond=cond, delta=x.mean(), t_lo=lo, t_hi=hi,
                                     p_t=stats.ttest_1samp(x, 0).pvalue, p_perm=signflip(x.values),
                                     base_mean=np.mean([NPM[(CFG[base][s], ck)][metric] for s in sd])))
NPR = pd.DataFrame(rows)
for (ck, met), idx in NPR.groupby(['ckpt', 'metric']).groups.items(): NPR.loc[idx, 'p_holm'] = holm(NPR.loc[idx, 'p_t'])
NPR.to_csv(f'{OUT}/normal_pneumonia_attention_effects.csv', index=False)
lev = []
for (a, t, cond, v), d in CFG.items():
    if v and not (a == 'mobilenetv2' and t == 'none' and v == MB): continue
    if cond not in ('imbalanced', 'balanced'): continue
    for ck in ['best']:
        b = np.array([NPM[(r, ck)]['bacc'] for r in d.values()]); u = np.array([NPM[(r, ck)]['auc'] for r in d.values()])
        sp = np.array([NPM[(r, ck)]['spec'] for r in d.values()]); se = np.array([NPM[(r, ck)]['sens'] for r in d.values()])
        lev.append(dict(arch=a, att=t, cond=cond, variant=v, bacc=b.mean(), bacc_sd=b.std(ddof=1), auc=u.mean(), auc_sd=u.std(ddof=1), sens=se.mean(), spec=sp.mean()))
pd.DataFrame(lev).to_csv(f'{OUT}/normal_pneumonia_levels.csv', index=False)

# ── checkpoint selection over the 157 main-study runs ─────────────────────────
ck = []
for rid, r in RUNS.items():
    m = r['meta']; v = r['variant']
    main = (v == '' and m['condition'] != 'downsample') or v in ('_attn_lr-0.0001', '_unfreeze-last2', MB)
    if not main: continue
    yv = r[('val', 'y')]; ck.append(dict(run_id=rid, val_acc_best=(r[('val', 'best')].argmax(1) == yv).mean(),
                                         delta=m['test_final']['macro_f1'] - m['test_best']['macro_f1']))
CK = pd.DataFrame(ck)
pd.DataFrame([dict(n_runs=len(CK), frac_val_acc_100=(CK.val_acc_best == 1).mean(), frac_final_gt_best=(CK.delta > 0).mean(),
                   mean_abs_delta=CK.delta.abs().mean(), max_abs_delta=CK.delta.abs().max())]).to_csv(f'{OUT}/checkpoint_selection_157.csv', index=False)
pd.set_option('display.width', 250)
for f in sorted(os.listdir(OUT)):
    print('==', f); print(pd.read_csv(os.path.join(OUT, f)).round(4).to_string(max_rows=80))

# ── balancing and attention x balancing interaction at the seed level (primary), both checkpoints ──
PS = pd.read_csv(os.path.join(ROOT, 'results', 'round5', 'per_seed_all_runs.csv')); PS['variant'] = PS.variant.fillna('')
def fs(a, t, c, v, ck): return PS[(PS.arch == a) & (PS.attention == t) & (PS.condition == c) & (PS.variant == v)].set_index('seed')[f'f1_{ck}']
rows = []
for ck in ['best', 'final']:
    for a in AL:
        bv = MB if a == 'mobilenetv2' else ''
        for t in ['none', 'se', 'cbam']:
            v = bv if t == 'none' else ''
            x = (fs(a, t, 'balanced', v, ck) - fs(a, t, 'imbalanced', v, ck)).dropna(); lo, hi = tci(x)
            loo = [x.drop(s).mean() for s in x.index]
            rows.append(dict(ckpt=ck, family='BALANCING', arch=a, att=t, baseline='dropout-matched' if v else 'standard', delta=x.mean(), t_lo=lo, t_hi=hi,
                             p_t=stats.ttest_1samp(x, 0).pvalue, p_perm=signflip(x.values), seeds_positive=int((x > 0).sum()), loo_min=min(loo), loo_max=max(loo)))
        for t in ['se', 'cbam']:
            x = ((fs(a, t, 'balanced', '', ck) - fs(a, t, 'imbalanced', '', ck)) - (fs(a, 'none', 'balanced', bv, ck) - fs(a, 'none', 'imbalanced', bv, ck))).dropna()
            lo, hi = tci(x); loo = [x.drop(s).mean() for s in x.index]
            rows.append(dict(ckpt=ck, family='INTERACTION', arch=a, att=t, baseline='dropout-matched' if bv else 'standard', delta=x.mean(), t_lo=lo, t_hi=hi,
                             p_t=stats.ttest_1samp(x, 0).pvalue, p_perm=signflip(x.values), seeds_positive=int((x > 0).sum()), loo_min=min(loo), loo_max=max(loo)))
    # sensitivity: MobileNetV2 with the original baseline
    x = (fs('mobilenetv2', 'none', 'balanced', '', ck) - fs('mobilenetv2', 'none', 'imbalanced', '', ck)); lo, hi = tci(x)
    rows.append(dict(ckpt=ck, family='BALANCING-original-MobileNetV2', arch='mobilenetv2', att='none', baseline='original', delta=x.mean(), t_lo=lo, t_hi=hi,
                     p_t=stats.ttest_1samp(x, 0).pvalue, p_perm=signflip(x.values), seeds_positive=int((x > 0).sum())))
BI = pd.DataFrame(rows)
for (ck, fam), idx in BI[BI.family.isin(['BALANCING', 'INTERACTION'])].groupby(['ckpt', 'family']).groups.items():
    BI.loc[idx, 'p_t_holm'] = holm(BI.loc[idx, 'p_t']); BI.loc[idx, 'p_perm_holm'] = holm(BI.loc[idx, 'p_perm'])
BI.to_csv(f'{OUT}/balancing_interaction_seed_level.csv', index=False)
print(BI.round(4).to_string())

# ── seed-level check of the SE-versus-CBAM and ablation families (bootstrap-primary in the paper) ──
import ast
E2 = pd.read_csv(os.path.join(ROOT, 'results', 'main_analysis', 'effects_labeled.csv'))
def f2(c): a, t, cond, v = c; return PS[(PS.arch == a) & (PS.attention == t) & (PS.condition == cond) & (PS.variant == v)].set_index('seed').f1_best
rows = []
for fam in ['SE-vs-CBAM', 'ABLATION']:
    for r in E2[E2.family == fam].itertuples():
        a, b = [ast.literal_eval(x) for x in r.compare.split('  vs  ')]; x = (f2(a) - f2(b)).dropna(); lo, hi = tci(x)
        rows.append(dict(family=fam, effect=r.effect, delta=x.mean(), t_lo=lo, t_hi=hi, p_t=stats.ttest_1samp(x, 0).pvalue, p_boot_holm=r.p_holm))
SC = pd.DataFrame(rows)
for fam, idx in SC.groupby('family').groups.items(): SC.loc[idx, 'p_t_holm'] = holm(SC.loc[idx, 'p_t'])
SC.to_csv(f'{OUT}/se_cbam_ablation_seed_level.csv', index=False)
print(SC.groupby('family')[['p_t', 'p_t_holm']].min())
