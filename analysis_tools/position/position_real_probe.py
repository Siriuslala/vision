"""Image-disjoint position probes across every vision and decoder block."""
import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from analysis_tools.position.position_probe import fit_ridge, metrics


def patch_grid(height: int, width: int, merge: int = 2) -> np.ndarray:
    """Map Qwen merge-group-major patch sequence to zero-based (column, row)."""
    if height % merge or width % merge:
        raise ValueError('Patch grid must be divisible by spatial merge size.')
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    return np.stack([x, y], -1).reshape(height//merge, merge, width//merge, merge, 2).transpose(0, 2, 1, 3, 4).reshape(-1, 2)


def labels(grid: tuple[int, int], indices: np.ndarray, vision: bool) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid
    if vision:
        xy = patch_grid(h, w)[indices].astype(np.float32)
    else:
        h, w = h//2, w//2
        xy = np.stack([indices % w, indices // w], -1).astype(np.float32)
    return xy, (xy + .5) / np.array([w, h], np.float32) * 1000


def evaluate(output: Path, projection_dim: int = 256) -> None:
    import torch
    torch.set_num_threads(4)
    cases = json.loads((output/'cases.json').read_text())
    metadata = json.loads((output/'experiment.json').read_text())
    data = np.load(output/'features.npz')
    if set(data.files) != set(metadata['stages']):
        raise ValueError('Feature stages do not match extraction metadata.')
    if projection_dim > metadata['projection_dimension']:
        raise ValueError('Requested projection exceeds cached dimensions.')
    if not metadata['model_path_name'] or metadata['thinking'] != 'off':
        raise ValueError('Unexpected model snapshot or thinking configuration.')
    splits = np.array([r['split'] for r in cases])
    if len({r['source_sha256'] for r in cases}) != len(cases):
        raise ValueError('Duplicate source images in probe data.')
    if set(splits) != {'train', 'validation', 'test'}:
        raise ValueError('Expected train, validation, and test image splits.')
    results, predictions = {}, {}
    for stage in data.files:
        vision = stage.startswith('vit') or stage == 'patch_content'
        cached = data[stage]
        expected = (len(cases), metadata['tokens_per_image'], metadata['projection_dimension'])
        if cached.shape != expected or not np.isfinite(cached).all():
            raise ValueError(f'Invalid cached feature shape or nonfinite values at {stage}.')
        xx = cached[..., :projection_dim].astype(np.float64)
        ys = [labels(tuple(r['patch_grid']), np.array(r['patch_indices' if vision else 'merged_indices']), vision) for r in cases]
        scale = np.maximum(xx[splits=='train'].reshape(-1, xx.shape[-1]).std(0), 1e-6)
        x = xx / scale
        result = {}
        for label_id, name in enumerate(['grid', 'image1000']):
            y = np.stack([v[label_id] for v in ys])
            tx, ty = x[splits=='train'].reshape(-1, x.shape[-1]), y[splits=='train'].reshape(-1, 2)
            vx, vy = x[splits=='validation'].reshape(-1, x.shape[-1]), y[splits=='validation'].reshape(-1, 2)
            fits = [(a, fit_ridge(tx,ty,a)) for a in (.01,1.,100.)]
            alpha,(weight,bias) = min(fits,key=lambda z: np.square(vx@z[1][0]+z[1][1]-vy).mean())
            rng=np.random.default_rng(9401)
            shuffled=ty.copy(); rng.shuffle(shuffled,axis=0)
            sw,sb=fit_ridge(tx,shuffled,alpha)
            px=x[splits=='test']; yy=y[splits=='test']; pred=px@weight+bias
            image_mae=np.abs(pred-yy).mean(axis=(1,2))
            boot=rng.choice(image_mae,(2000,len(image_mae)),replace=True).mean(1)
            entry=dict(alpha=alpha,probe=metrics(pred.reshape(-1,2),yy.reshape(-1,2)),
                constant=metrics(np.broadcast_to(ty.mean(0),yy.shape).reshape(-1,2),yy.reshape(-1,2)),
                shuffled=metrics((px@sw+sb).reshape(-1,2),yy.reshape(-1,2)),
                image_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist(), per_image_mae=image_mae.tolist())
            if name=='grid':
                entry['exact_grid_accuracy']=float(np.all(np.rint(pred)==yy,axis=-1).mean())
                entry['within_one_cell_accuracy']=float(np.all(np.abs(np.rint(pred)-yy)<=1,axis=-1).mean())
            result[name]=entry
            predictions[stage+'__'+name]=pred.astype(np.float32)
        results[stage]=result
        print('Evaluated',stage,round(result['image1000']['probe']['mean_coordinate_mae'],3),flush=True)
    suffix = '' if projection_dim == 256 else f'_dim{projection_dim}'
    (output/f'results{suffix}.json').write_text(json.dumps(results,indent=2)+'\n')
    np.savez_compressed(output/f'predictions{suffix}.npz',**predictions)
    (output/f'evaluation{suffix}.json').write_text(json.dumps(dict(projection_dimension=projection_dim, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), evaluated_utc=datetime.now(timezone.utc).isoformat(), bootstrap_samples=2000, bootstrap_unit='image', regularization_candidates=[.01,1.,100.]),indent=2)+'\n')


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--refcoco',type=Path)
    parser.add_argument('--model',default='Qwen3.5-9B')
    parser.add_argument('--images',type=int,default=360)
    parser.add_argument('--tokens',type=int,default=48)
    parser.add_argument('--evaluate-only',action='store_true')
    parser.add_argument('--projection-dim',type=int,choices=[128,256],default=256)
    args=parser.parse_args()
    if args.evaluate_only:
        evaluate(args.output,args.projection_dim); return
    if args.images<30 or args.images%6:
        raise ValueError('Image count must be >=30 and divisible by six.')
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from eva.inference import resolve_model
    from utils.project_paths import load_paths
    torch.set_num_threads(4)
    started=datetime.now(timezone.utc).isoformat()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'images').mkdir()
    seen=set(); candidates=[]
    for shard in sorted((args.refcoco/'data').glob('val-*.parquet')):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                raw=row['image']['bytes']; digest=hashlib.sha256(raw).hexdigest()
                if digest in seen: continue
                seen.add(digest)
                image=Image.open(io.BytesIO(raw)).convert('RGB')
                # Keep realistic aspect variation without allowing extreme tiny axes.
                if max(image.size)/min(image.size)>3: continue
                candidates.append((digest,image,row['question_id']))
                if len(candidates)>=args.images: break
            if len(candidates)>=args.images: break
        if len(candidates)>=args.images: break
    if len(candidates)!=args.images: raise ValueError('Insufficient unique images.')
    order=np.random.default_rng(20261001).permutation(args.images)
    groups=['train']*(args.images*2//3)+['validation']*(args.images//6)+['test']*(args.images//6)
    assignments={int(k):groups[i] for i,k in enumerate(order)}
    model_path=resolve_model(args.model,load_paths()['HF_HUB_CACHE'])
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(model_path,local_files_only=True,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
    projections={}; captured={}; handles=[]; current={}
    def capture(name,vision=False,inputs=False):
        def hook(module,inp,out):
            value=inp[0] if inputs else out
            if not isinstance(value,torch.Tensor): value=value[0]
            if value.ndim==3: value=value[0,current['mask']]
            ids=current['patch_ids' if vision else 'merged_ids']
            value=value[ids].float()
            dim=value.shape[-1]
            if dim not in projections:
                matrix=np.random.default_rng(20261001).normal(size=(dim,256)).astype(np.float32)/np.float32(np.sqrt(dim))
                projections[dim]=torch.as_tensor(matrix,device=value.device)
            captured[name]=(value@projections[dim]).cpu().numpy().astype(np.float16)
        return hook
    visual=model.model.visual
    if visual.config.spatial_merge_size != 2:
        raise ValueError('This experiment requires the Qwen 2x2 spatial merger.')
    handles.append(visual.patch_embed.register_forward_hook(capture('patch_content',True)))
    handles.append(visual.blocks[0].register_forward_hook(capture('vit_00_position',True,True)))
    for i,block in enumerate(visual.blocks): handles.append(block.register_forward_hook(capture(f'vit_{i+1:02d}',True)))
    handles.append(visual.merger.register_forward_hook(capture('merger')))
    for i,block in enumerate(model.model.language_model.layers): handles.append(block.register_forward_hook(capture(f'decoder_{i+1:02d}_{block.block_type}')))
    records=[]; features={}
    for i,(digest,original,qid) in enumerate(candidates):
        ratio=512/max(original.size)
        size=tuple(max(96,int(round(v*ratio/32))*32) for v in original.size)
        image=original.resize(size,Image.Resampling.LANCZOS)
        image.save(args.output/'images'/f'{i:04d}.png')
        content=[{'type':'image','image':image},{'type':'text','text':'Describe the image briefly.'}]
        text=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        batch=processor(text=[text],images=[image],return_tensors='pt').to('cuda:0')
        _,h,w=batch.image_grid_thw[0].tolist()
        rng=np.random.default_rng(2000+i)
        m=min(args.tokens,h*w//4)
        if m!=args.tokens: raise ValueError('Image contains fewer tokens than requested.')
        mids=np.sort(rng.choice(h*w//4,m,replace=False)); pids=mids*4+rng.integers(0,4,m)
        current.update(mask=batch.input_ids[0]==model.config.image_token_id,merged_ids=torch.tensor(mids,device='cuda'),patch_ids=torch.tensor(pids,device='cuda'))
        if int(current['mask'].sum())!=h*w//4: raise ValueError('Unexpected visual sequence length.')
        captured.clear()
        with torch.inference_mode(): model.model(**batch,use_cache=False)
        for stage,value in captured.items(): features.setdefault(stage,[]).append(value)
        records.append(dict(id=i,split=assignments[i],source_sha256=digest,question_id=qid,source_size=original.size,input_size=size,patch_grid=[h,w],patch_indices=pids.tolist(),merged_indices=mids.tolist(),visual_start=int(current['mask'].nonzero()[0]),prompt='Describe the image briefly.'))
        if (i+1)%10==0: print(f'Encoded {i+1}/{args.images}',flush=True)
    for handle in handles: handle.remove()
    np.savez_compressed(args.output/'features.npz',**{k:np.stack(v) for k,v in features.items()})
    (args.output/'cases.json').write_text(json.dumps(records,indent=2)+'\n')
    (args.output/'experiment.json').write_text(json.dumps(dict(started_utc=started,finished_encoding_utc=datetime.now(timezone.utc).isoformat(),model=args.model,model_path_name=Path(model_path).name,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),thinking='off',dtype='bfloat16',attention='sdpa',projection_dimension=256,tokens_per_image=args.tokens,split_seed=20261001,stages=list(features),text_config=model.config.text_config.to_dict()),indent=2)+'\n')
    del model; torch.cuda.empty_cache()
    evaluate(args.output)


if __name__=='__main__': main()
