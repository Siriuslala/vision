"""Image-disjoint, low-capacity probes of visual slot coordinates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def coordinates(height: int, width: int) -> np.ndarray:
    """Return normalized centers in row-major merged visual-token order."""
    y, x = np.meshgrid((np.arange(height) + .5) / height,
                       (np.arange(width) + .5) / width, indexing='ij')
    return np.stack((x, y), axis=-1).reshape(-1, 2)


def make_image(size: tuple[int, int], shape: str, seed: int, shifted=False) -> Image.Image:
    rng = np.random.default_rng(seed)
    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)
    dx, dy = rng.integers(-15, 16, size=2) if shifted else (0, 0)
    for y in range(32, size[1], 64):
        for x in range(32, size[0], 64):
            radius = int(rng.integers(9, 20))
            color = tuple(int(v) for v in rng.integers(0, 160, size=3))
            x1, y1 = int(x + dx), int(y + dy)
            box = (x1-radius, y1-radius, x1+radius, y1+radius)
            if shape == 'circle':
                draw.ellipse(box, outline=color, width=3)
            elif shape == 'square':
                draw.rectangle(box, outline=color, width=3)
            elif shape == 'triangle':
                draw.polygon([(x1,y1-radius),(x1-radius,y1+radius),(x1+radius,y1+radius)], outline=color, width=3)
            else:
                raise ValueError(f'Unknown shape: {shape}')
    return image


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit an intercept without regularizing it; alpha applies to summed loss."""
    mean_x, mean_y = x.mean(0), y.mean(0)
    centered = x - mean_x
    gram = centered.T @ centered
    weights = np.linalg.solve(gram + alpha * np.eye(x.shape[1]), centered.T @ (y-mean_y))
    return weights, mean_y - mean_x @ weights


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {'mean_coordinate_mae': float(np.abs(prediction-target).mean()),
            'x_mae': float(np.abs(prediction[:,0]-target[:,0]).mean()),
            'y_mae': float(np.abs(prediction[:,1]-target[:,1]).mean()),
            'mean_point_l2': float(np.linalg.norm(prediction-target, axis=1).mean()),
            'p90_point_l2': float(np.quantile(np.linalg.norm(prediction-target, axis=1), .9))}


def evaluate(records: list[dict], features: list[dict], output: Path) -> None:
    groups = sorted({r['split'] for r in records})
    results = {}
    for stage in features[0]:
        rng = np.random.default_rng(2026)
        dim = features[0][stage].shape[1]
        projection = rng.normal(size=(dim, min(dim, 256))).astype(np.float32)/np.sqrt(dim)
        xs = [f[stage].astype(np.float32) @ projection for f in features]
        ys = [coordinates(r['grid'][0], r['grid'][1]) for r in records]
        def collect(split):
            ids = [i for i,r in enumerate(records) if r['split']==split]
            return np.concatenate([xs[i] for i in ids]), np.concatenate([ys[i] for i in ids])
        train_x, train_y = collect('train')
        val_x, val_y = collect('validation')
        scale = np.maximum(train_x.std(0), 1e-6)
        train_x, val_x = train_x/scale, val_x/scale
        fits = [(a, fit_ridge(train_x, train_y, a)) for a in (.01, 1., 100.)]
        alpha, (weights, bias) = min(fits, key=lambda f: np.square(val_x@f[1][0]+f[1][1]-val_y).mean())
        shuffled = train_y.copy()
        rng.shuffle(shuffled, axis=0)
        random_weights, random_bias = fit_ridge(train_x, shuffled, alpha)
        result = {'alpha': alpha, 'input_dimension': dim, 'projection_dimension': projection.shape[1], 'splits': {}}
        for split in groups:
            x,y = collect(split)
            pred = x/scale@weights+bias
            result['splits'][split] = {'n_images':sum(r['split']==split for r in records), 'n_tokens': len(y),
                'probe':metrics(pred,y), 'shuffled_training_labels':metrics(x/scale@random_weights+random_bias,y),
                'constant':metrics(np.broadcast_to(train_y.mean(0), y.shape),y)}
        # Paired prefix intervention uses exactly the same pixels and slot targets.
        paired = []
        paired_base_errors, paired_long_errors = [], []
        for i,r in enumerate(records):
            if r['split'] == 'long_prefix':
                j = next(j for j,s in enumerate(records) if s['id']==r['pair_id'])
                a,b = xs[i]/scale@weights+bias,xs[j]/scale@weights+bias
                paired.append(float(np.linalg.norm(a-b,axis=1).mean()))
                paired_base_errors.append(metrics(b,ys[j])['mean_coordinate_mae'])
                paired_long_errors.append(metrics(a,ys[i])['mean_coordinate_mae'])
        result['prefix_mean_prediction_shift'] = float(np.mean(paired))
        result['prefix_paired_short_mae'] = float(np.mean(paired_base_errors))
        result['prefix_paired_long_mae'] = float(np.mean(paired_long_errors))
        results[stage] = result
        print(stage, {k:round(v['probe']['mean_coordinate_mae'],4) for k,v in result['splits'].items()}, flush=True)
    (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen3.5-9B')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-features', action='store_true')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    if args.reuse_features:
        records = json.loads((args.output/'cases.json').read_text())
        features = torch.load(args.output/'features.pt', weights_only=False)
        evaluate(records, features, args.output)
        return
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from eva.inference import resolve_model
    from utils.project_paths import load_paths
    args.output.mkdir(parents=True, exist_ok=False)
    model_path = resolve_model(args.model, load_paths()['HF_HUB_CACHE'])
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(model_path, local_files_only=True,
        dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa').eval()
    records = []
    for split,n in [('train',24),('validation',6),('test_same',8),('test_shift',8),('test_template',8),('test_size_aspect',8)]:
        for i in range(n):
            size = ((256,256),(384,384))[i%2]
            if split=='test_size_aspect':
                size = ((512,512),(512,256),(256,512),(640,384))[i%4]
            records.append(dict(id=f'{split}_{i}', split=split, seed=len(records)+10, size=size,
                shape='triangle' if split=='test_template' else ('circle','square')[(i//2)%2], shifted=split=='test_shift'))
    for record in [r.copy() for r in records if r['split']=='test_same'][:4]:
        records.append(dict(record, id='prefix_'+record['id'], split='long_prefix', pair_id=record['id']))
    captured, handles, mask = {}, [], None
    def capture(name):
        def hook(module, inputs, output):
            value = output if isinstance(output, torch.Tensor) else output[0]
            if value.ndim==3:
                value=value[0, mask]
            captured[name]=value.detach().float().cpu().numpy().astype(np.float16)
        return hook
    handles.append(model.model.visual.merger.register_forward_hook(capture('visual_embedding')))
    layers = model.model.language_model.layers
    for i in (0,2,3,4,6,7,8,15,31):
        if i<len(layers):
            handles.append(layers[i].register_forward_hook(capture(f'decoder_{i+1}_{layers[i].block_type}')))
    features=[]
    for i,record in enumerate(records):
        im=make_image(tuple(record['size']),record['shape'],record['seed'],record['shifted'])
        im.save(args.output/f"{record['id']}.png")
        content=[]
        if record['split']=='long_prefix':
            content.append({'type':'text','text':'This is a controlled visual observation. '*24})
        content.extend([{'type':'image','image':im},{'type':'text','text':'Describe the image briefly.'}])
        text=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        batch=processor(text=[text], images=[im], return_tensors='pt').to('cuda:0')
        grid=batch.image_grid_thw[0].tolist()
        record['grid']=[grid[1]//2,grid[2]//2]
        record['input_tokens']=int(batch.input_ids.shape[-1])
        mask=batch.input_ids[0]==model.config.image_token_id
        record['visual_start']=int(mask.nonzero()[0])
        if int(mask.sum())!=record['grid'][0]*record['grid'][1]:
            raise ValueError('Visual token/grid mismatch')
        captured.clear()
        with torch.inference_mode():
            model.model(**batch, use_cache=False)
        features.append(captured.copy())
        print(f'Encoded {i+1}/{len(records)} {record["id"]} grid={record["grid"]}',flush=True)
    for handle in handles:
        handle.remove()
    (args.output/'cases.json').write_text(json.dumps(records,indent=2)+'\n')
    (args.output/'experiment.json').write_text(json.dumps(dict(model=args.model, thinking='off',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),dtype='bfloat16',attention='sdpa',
        stages=list(features[0]), text_config=model.config.text_config.to_dict()),indent=2)+'\n')
    torch.save(features,args.output/'features.pt')
    del model
    torch.cuda.empty_cache()
    evaluate(records,features,args.output)


if __name__=='__main__':
    main()
