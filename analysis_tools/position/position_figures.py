"""Publication-style position probe figures from saved experiment records."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis_tools.position.position_real_probe import labels


def save(fig, path):
    fig.savefig(path,dpi=180,bbox_inches='tight'); plt.close(fig)


def legacy(root: Path, out: Path):
    results=json.loads((root/'results.json').read_text())
    cases=json.loads((root/'cases.json').read_text())
    fig,axes=plt.subplots(2,3,layout='constrained',figsize=(12,8))
    for ax,group in zip(axes.flat,['train','validation','test_same','test_shift','test_template','test_size_aspect']):
        case=next(x for x in cases if x['split']==group)
        ax.imshow(Image.open(root/(case['id']+'.png')));ax.set_title(f"{group}\n{case['size'][0]} x {case['size'][1]} pixels");ax.axis('off')
    save(fig,out/'synthetic_examples.png')
    x=[0]+[int(k.split('_')[1]) for k in results if k!='visual_embedding']
    fig,axes=plt.subplots(1,2,layout='constrained',figsize=(13,4))
    for split,label in [('test_same','Same distribution'),('test_shift','Shifted objects'),('test_template','Unseen triangles'),('test_size_aspect','Unseen image sizes')]:
        axes[0].plot(x,[v['splits'][split]['probe']['mean_coordinate_mae']*1000 for v in results.values()],'-o',label=label)
    axes[0].axhline(250,ls='--',color='gray',label='Constant baseline');axes[0].set_ylim(0,270)
    axes[0].set(title='A: coordinate recovery',xlabel='Decoder layer (0 = visual embedding)',ylabel='Coordinate MAE (0-1000 units)');axes[0].legend(fontsize=8)
    for key,label in [('prefix_paired_short_mae','Short prefix'),('prefix_paired_long_mae','Long prefix')]:
        axes[1].plot(x,[v[key]*1000 for v in results.values()],'-o',label=label)
    axes[1].set(title='B: same images, changed text prefix',xlabel='Decoder layer (0 = visual embedding)',ylabel='Coordinate MAE (0-1000 units)');axes[1].legend()
    save(fig,out/'synthetic_probe_curves.png')
    fig,ax=plt.subplots(layout='constrained',figsize=(12,3));ax.axis('off')
    for yy,title,txt in [(.72,'Short prefix','[image]  Describe the image briefly.'),(.25,'Long prefix','This is a controlled visual observation. (repeated 24 times)\n[image]  Describe the image briefly.')]:
        ax.text(.02,yy,title,weight='bold',fontsize=12);ax.text(.18,yy,txt,fontsize=11,bbox=dict(facecolor='#edf3f8',edgecolor='#7798af',pad=9),va='center')
    ax.text(.02,-.12,'Same pixels and targets; use the probe trained on short-prefix images without refitting.',fontsize=11)
    save(fig,out/'prefix_design.png')


def real(root: Path,out: Path):
    results=json.loads((root/'results.json').read_text());cases=json.loads((root/'cases.json').read_text());pred=np.load(root/'predictions.npz')
    test=[r for r in cases if r['split']=='test']
    fig,axes=plt.subplots(1,3,layout='constrained',figsize=(12,4))
    for ax,split in zip(axes,['train','validation','test']):
        c=next(r for r in cases if r['split']==split);ax.imshow(Image.open(root/'images'/f"{c['id']:04d}.png"))
        _,xy=labels(tuple(c['patch_grid']),np.array(c['merged_indices']),False)
        ax.scatter(xy[:,0]/1000*c['input_size'][0],xy[:,1]/1000*c['input_size'][1],s=10,c=xy[:,1],cmap='cool',edgecolors='white',linewidths=.3)
        ax.set_title(f"{split}: image {c['id']}\n{c['input_size'][0]} x {c['input_size'][1]}, 48 sampled tokens");ax.axis('off')
    save(fig,out/'refcoco_examples.png')
    stages=[s for s in results if s.startswith('vit_')]; decoder=['merger']+[s for s in results if s.startswith('decoder')]
    fig,axes=plt.subplots(2,2,layout='constrained',figsize=(14,8))
    for col,seq,title in [(0,stages,'Vision Transformer'),(1,decoder,'VLM decoder')]:
        xx=np.arange(len(seq))
        for row,target in [(0,'image1000'),(1,'grid')]:
            vals=np.array([results[s][target]['probe']['mean_coordinate_mae'] for s in seq]);ci=np.array([results[s][target]['image_bootstrap_ci95'] for s in seq])
            axes[row,col].plot(xx,vals,'-o',ms=3,label='Probe');axes[row,col].fill_between(xx,ci[:,0],ci[:,1],alpha=.2)
            axes[row,col].plot(xx,[results[s][target]['constant']['mean_coordinate_mae'] for s in seq],'--',color='gray',label='Constant')
            axes[row,col].plot(xx,[results[s][target]['shuffled']['mean_coordinate_mae'] for s in seq],':',color='brown',label='Shuffled train labels')
            axes[row,col].set(title=title,xlabel='Layer (0 = position addition)' if col==0 else 'Layer (0 = merger)',ylabel='MAE (0-1000 units)' if row==0 else ('MAE (patch cells)' if col==0 else 'MAE (merged cells)'))
            if col==1:
                for n in range(4,33,4): axes[row,col].axvline(n,color='gray',alpha=.15,lw=2)
            axes[row,col].legend(fontsize=8)
    save(fig,out/'refcoco_all_layers.png')
    # A closer scale makes the non-monotonic trends visible without baselines.
    fig,axes=plt.subplots(1,2,layout='constrained',figsize=(13,4))
    for ax,seq,title in zip(axes,[stages,decoder],['ViT','Decoder (gray = full attention)']):
        for target,linestyle in [('x_mae','-'),('y_mae','--')]:
            ax.plot(range(len(seq)),[results[s]['image1000']['probe'][target] for s in seq],linestyle,label=target)
        ax.set(title=title,xlabel='Layer (0 = input with position / merger)',ylabel='Coordinate MAE (0-1000 units)');ax.legend()
        if seq==decoder:
            for n in range(4,33,4): ax.axvline(n,color='gray',alpha=.15,lw=2)
    save(fig,out/'refcoco_layer_detail.png')
    low=json.loads((root/'results_dim128.json').read_text())
    fig,axes=plt.subplots(1,2,layout='constrained',figsize=(13,4))
    for ax,seq,title in zip(axes,[stages,decoder],['ViT','Decoder']):
        for source,label in [(results,'256 dimensions'),(low,'128 dimensions')]:
            ax.plot(range(len(seq)),[source[s]['image1000']['probe']['mean_coordinate_mae'] for s in seq],'-o',ms=3,label=label)
        ax.set(title=title,xlabel='Layer (0 = input with position / merger)',ylabel='Coordinate MAE (0-1000 units)');ax.legend()
    save(fig,out/'refcoco_probe_capacity.png')

    seq=['merger','decoder_08_full_attention','decoder_16_full_attention','decoder_24_full_attention','decoder_31_linear_attention','decoder_32_full_attention']
    fig,axes=plt.subplots(2,3,layout='constrained',figsize=(13,8))
    c=test[0];_,truth=labels(tuple(c['patch_grid']),np.array(c['merged_indices']),False)
    for ax,s in zip(axes.flat,seq):
        ax.imshow(Image.open(root/'images'/f"{c['id']:04d}.png"),extent=(0,1000,1000,0)); p=pred[s+'__image1000'][0]; ids=np.arange(0,len(truth),2)
        ax.quiver(truth[ids,0],truth[ids,1],p[ids,0]-truth[ids,0],p[ids,1]-truth[ids,1],angles='xy',scale_units='xy',scale=1,color='#ff3030',width=.004)
        ax.scatter(truth[ids,0],truth[ids,1],c='cyan',s=8);ax.set(xlim=(0,1000),ylim=(1000,0),title=s.replace('_',' '));ax.set_aspect('equal')
    fig.suptitle('One held-out image: cyan = true token center; red arrow = prediction error')
    save(fig,out/'refcoco_prediction_arrows.png')
    # Aggregate spatial residuals over equally sampled held-out images.
    fig,axes=plt.subplots(1,3,layout='constrained',figsize=(12,4))
    for ax,s in zip(axes,['merger','decoder_16_full_attention','decoder_32_full_attention']):
        counts=np.zeros((8,8)); sums=np.zeros((8,8))
        for j,c in enumerate(test):
            _,truth=labels(tuple(c['patch_grid']),np.array(c['merged_indices']),False);err=np.abs(pred[s+'__image1000'][j]-truth).mean(-1)
            bins=np.minimum((truth/125).astype(int),7)
            for (bx,by),e in zip(bins,err): sums[by,bx]+=e;counts[by,bx]+=1
        im=ax.imshow(sums/np.maximum(counts,1),vmin=0,vmax=350,cmap='magma',extent=(0,1000,1000,0));ax.set_title(s.replace('_',' '))
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Mean coordinate absolute error',shrink=.8)
    save(fig,out/'refcoco_spatial_error.png')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--legacy',type=Path);parser.add_argument('--real',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.legacy: legacy(args.legacy,args.output)
    if args.real: real(args.real,args.output)

if __name__=='__main__': main()
