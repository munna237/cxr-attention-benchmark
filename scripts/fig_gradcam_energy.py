"""Fig. 8: fraction of Grad-CAM energy inside the lung masks per model and class (five seeds pooled),
with per-class chance lines (mean lung-area fraction) and n per box in the legend.
Input: outputs/gradcam/gradcam_per_image.csv. Output: figures/fig_gradcam_energy_v3.png."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = pd.read_csv(os.path.join(ROOT, 'outputs', 'gradcam', 'gradcam_per_image.csv')); g = G[G.qc_ok]
CLS = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']; LAB = ['COVID-19', 'Normal', 'Pneumonia', 'Tuberculosis']
cols = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
names = ['ResNet50 + Baseline', 'ResNet50 + SE', 'ResNet50 + CBAM', 'DenseNet121 + CBAM', 'MobileNetV2 + Baseline', 'Custom CNN + CBAM']
chance = g.drop_duplicates('img').groupby('true').lung_area_frac.mean()
nbox = g.groupby(['model', 'true']).size()
assert all(nbox.xs(c, level='true').nunique() == 1 for c in CLS)          # same n for every model
plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots(figsize=(7.2, 3.7)); ticks = []
for k, name in enumerate(names):
    for ci, c in enumerate(CLS):
        x = k * 4.8 + ci; dat = g[(g.model == name) & (g.true == c)].energy_in_lung.values
        b = ax.boxplot([dat], positions=[x], widths=.72, showfliers=False, patch_artist=True, medianprops=dict(color='k', lw=1.2))
        b['boxes'][0].set(facecolor=cols[ci], alpha=.6)
        ax.plot([x - .42, x + .42], [chance[c]] * 2, ls=':', c='k', lw=1.4)
    ticks.append(k * 4.8 + 1.5)
ax.set_xticks(ticks); ax.set_xticklabels([n.replace(' + ', '\n+ ') for n in names], fontsize=9.5)
ax.set_ylim(0, 1.08); ax.set_yticks(np.arange(0, 1.01, .2)); ax.set_ylabel('Grad-CAM energy inside lungs')
ax.spines[['top', 'right']].set_visible(False)
handles = [Patch(facecolor=cols[i], alpha=.6, label=f'{LAB[i]} (n = {nbox.xs(CLS[i], level="true").iloc[0]})') for i in range(4)] + \
          [Line2D([], [], ls=':', c='k', lw=1.4, label='class lung-area fraction (chance)')]
ax.legend(handles=handles, fontsize=8.5, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.13), frameon=False, columnspacing=1.2, handlelength=1.4)
plt.tight_layout(); os.makedirs(os.path.join(ROOT, 'figures'), exist_ok=True)
plt.savefig(os.path.join(ROOT, 'figures', 'fig_gradcam_energy_v3.png'), dpi=300); plt.savefig(os.path.join(ROOT, 'figures', 'fig_gradcam_energy_v3.pdf'))
print(chance.round(3).to_dict(), nbox.xs('ResNet50 + Baseline').to_dict())
