"""Figures and numerical summaries from saved representation-layout experiments."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


STAGES = ['patch_embed','vit_03','vit_09','vit_18','vit_27','merger','decoder_01','decoder_04','decoder_08','decoder_16','decoder_24','decoder_28','decoder_31','decoder_32']


def summarize(data: dict) -> dict:
    aggregate = {}
    for kind in ['all','standard','extension']:
        groups = defaultdict(list)
        for row in data['records']:
            standard = row['case'].startswith('grid')
            if kind == 'standard' and not standard or kind == 'extension' and standard:
                continue
            groups[row['stage']].append(row)
        aggregate[kind] = {}
        for stage, rows in groups.items():
            aggregate[kind][stage] = {'n':len(rows),'paired_rms':sum(r['paired_rms_correct'] for r in rows),
                'paired_cosine':sum(r['paired_cosine_correct'] for r in rows),
                'same_normal_peak':sum(r['scores']['all_rms']['same_normal_peak'] for r in rows),
                'correct':{k:sum(r['scores'][k]['correct'] for r in rows) for k in rows[0]['scores']},
                'empty_objects':sum(r['support']['exclude_high5']['empty_objects'] for r in rows)}
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--docs-dir',type=Path,required=True)
    args=parser.parse_args()
    data=json.loads((args.input/'results.json').read_text())
    args.docs_dir.mkdir(parents=True,exist_ok=True)
    layout_by_name={r['name']:r for r in data['metadata']['layouts']}
    normals={(r['case'],r['stage']):r for r in data['normals']}
    names=list(layout_by_name)
    fig,axes=plt.subplots(len(names),5,figsize=(15,3*len(names)),layout='constrained')
    for row,name in enumerate(names):
        case=f'{name}_person';layout=layout_by_name[name]
        im=Image.open(args.input/case/'normal.png')
        axes[row,0].imshow(im);axes[row,0].set_title(name)
        for col,stage in enumerate(['vit_18','merger','decoder_24','decoder_32'],1):
            ax=axes[row,col];record=normals[case,stage]
            ax.imshow(im)
            boxes=np.array(layout['boxes']);centers=(boxes[:,:2]+boxes[:,2:])/2
            scores=np.array(record['scores']['all_rms']['scores'])
            colors=(scores-scores.min())/max(float(scores.max()-scores.min()),1e-20)
            ax.scatter(centers[:,0],centers[:,1],c=colors,cmap='coolwarm',s=45 if len(boxes)<100 else 8,vmin=0,vmax=1)
            peak=record['scores']['all_rms']['peak']
            if peak is not None:
                x,y=centers[peak];ax.scatter([x],[y],marker='s',facecolors='none',edgecolors='lime',s=180,linewidths=2)
            ax.set_title(f'{stage}: peak object {peak}')
        for ax in axes[row]:ax.set_axis_off()
    fig.suptitle('Normal images: object RMS score within each image (green box = unique maximum)',fontsize=13)
    fig.savefig(args.input/'normal_patterns.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(len(names),4,figsize=(12,3*len(names)),layout='constrained')
    for row,name in enumerate(names):
        for col,stage in enumerate(['vit_18','merger','decoder_24','decoder_32']):
            record=normals[f'{name}_person',stage];values=np.array(record['norm_map']);ax=axes[row,col]
            ax.imshow(values,cmap='magma',interpolation='nearest')
            y,x=np.unravel_index(values.argmax(),values.shape)
            ax.scatter([x],[y],marker='s',facecolors='none',edgecolors='cyan',s=60)
            ax.set_title(f'{name}\n{stage}: max/median={record["norm_max_over_median"]:.2f}')
            ax.set_axis_off()
    fig.suptitle('Normal images: token L2 norms; each panel independently scaled',fontsize=13)
    fig.savefig(args.input/'normal_norms.png',dpi=150);plt.close(fig)
    shared={(r['case'],r['target']) for r in data['records'] if r['stage']=='decoder_32'}
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
    for ax,standard,title in zip(axes,[True,False],['Standard rectangular grids','Staggered and circular layouts']):
        rows=[r for r in data['records'] if (r['case'],r['target']) in shared and r['case'].startswith('grid')==standard]
        groups={stage:[r for r in rows if r['stage']==stage] for stage in STAGES}
        for key,label in [('all_rms','RMS: all tokens'),('all_cosine','Cosine: all tokens'),('exclude_high5_rms','RMS: exclude high-norm 5%')]:
            values=[np.mean([r['scores'][key]['correct'] for r in groups[stage]])*100 for stage in STAGES]
            ax.plot(range(len(STAGES)),values,'o-',label=label,markersize=3)
        random=[np.mean([[r['scores'][f'exclude_random5_seed{s}_rms']['correct'] for s in range(3)] for r in groups[stage]])*100 for stage in STAGES]
        ax.plot(range(len(STAGES)),random,'o--',label='RMS: random exclusion (3 seeds)',markersize=3)
        ax.set_title(f'{title}: same {len(groups[STAGES[0]])} images at every layer')
        ax.set_ylabel('Correctly localized images (%)');ax.set_ylim(-3,103);ax.grid(alpha=.2)
        ax.set_xticks(range(len(STAGES)),[s.replace('decoder_','D').replace('vit_','V').replace('patch_embed','patch') for s in STAGES],rotation=65)
        ax.legend(fontsize=8)
    fig.savefig(args.input/'layer_accuracy.png',dpi=150);plt.close(fig)
    summary=summarize(data)
    (args.input/'aggregate.json').write_text(json.dumps(summary,indent=2))
    for name in ['normal_patterns.png','normal_norms.png','layer_accuracy.png']:
        shutil.copyfile(args.input/name,args.docs_dir/f'2026-10-01_{name}')


if __name__=='__main__':main()
