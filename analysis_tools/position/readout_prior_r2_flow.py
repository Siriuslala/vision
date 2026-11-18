"""Experiment 3.16 round 2: schematic of the coordinate-readout mechanism (docs figure; no data)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(15, 5.2)); ax.set_xlim(0, 15); ax.set_ylim(0, 5.2); ax.axis('off')
def box(x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05', fc=c, ec='0.3', lw=0.8)); ax.text(x + w/2, y + h/2, t, ha='center', va='center', fontsize=8.3)
def arr(x0, y0, x1, y1, t='', c='0.25'):
    ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='->', color=c, lw=1.2)); 
    if t: ax.text((x0+x1)/2, (y0+y1)/2 + 0.12, t, ha='center', fontsize=7.5, color=c)
B, O, G, Y = '#d6e6f8', '#fbe0d3', '#d4f0e3', '#fbefc9'
box(0.2, 3.6, 2.0, 1.2, 'ViT + merger\nposition of the disc\nwritten into image tokens\n(long-period code, T >= 500)', B)
box(2.8, 3.6, 2.5, 1.2, 'decoder layer 1, first image token\n(+ fixed vector v, 4096 params)\n[fix]', O)
box(5.9, 3.6, 2.7, 1.2, 'layers 2-13: other image tokens read\nthe first token (Gated DeltaNet state,\nfull attention L4/8/12)', Y)
box(9.2, 3.6, 2.6, 1.2, 'layers 14-19 on image tokens:\nperiod-100 pair cos/sin(2πx/100)\nappears on target tokens (L15-16)', G)
box(12.3, 3.6, 2.5, 1.2, 'full attention L20 / L24:\nq0 reads target tokens\n(same heads as for text digits)', B)
box(12.3, 1.4, 2.5, 1.2, 'layers 21-32 at q0 / q1:\nlinear readout + softmax\nfirst digit: arc (< 1 turn)\nsecond digit: 10 sectors of circle', B)
box(8.9, 1.4, 2.9, 1.2, 'base model (no fix):\nq0 receives only the long-period code\n-> second digit via neighbour-logit\ndifference m (3.12), per-hundred bias', '#eeeeee')
arr(2.2, 4.2, 2.8, 4.2); arr(5.3, 4.2, 5.9, 4.2); arr(8.6, 4.2, 9.2, 4.2); arr(11.8, 4.2, 12.3, 4.2)
arr(13.55, 3.6, 13.55, 2.6, 'period-100 code at q0')
arr(12.3, 2.0, 11.8, 2.0, '', '0.5')
ax.text(13.55, 0.9, 'digits "3" "1" "2"', ha='center', fontsize=9); arr(13.55, 1.4, 13.55, 1.1)
ax.text(0.2, 0.4, 'Blue: unchanged model computation; orange: the located change; yellow: transfer from the first image token; green: where the new code is built; grey: base-model fallback.', fontsize=8, color='0.3')
fig.savefig('docs/analysis/position/figures/readout_prior_r2/mechanism_flow.png', dpi=150, bbox_inches='tight')
