"""Batched offline Hugging Face inference shared by simple_test and evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from data.prepare_data import snapshot
from utils.project_paths import load_paths
from eva.benchmark import load_cases, parse_ids, prompt_for, score, summarize
from eva.inputs import load_image_prompts, summarize_inputs

MODELS = ['Qwen3.5-4B', 'Qwen3.5-9B', 'Qwen3-VL-4B-Instruct', 'Qwen3-VL-8B-Instruct']


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def parser(full=False):
    paths = load_paths()
    p = argparse.ArgumentParser(description='Offline local VLM benchmark inference')
    p.add_argument('--model', '--model-name', '--model_name', required=True)
    if full:
        p.add_argument('--bench', '--bench-name', '--bench_name', type=str.lower,
                       choices=['babyvision', 'babyvision-gen'], default='babyvision')
        p.add_argument('--case-ids', '--case_ids', nargs='+',
                       help='Original taskIds: 445 464, 445,464, or "[445,464]"')
        p.add_argument('--data-root', default=paths['DATA_DIR'])
        p.add_argument('--split', default='train')
        p.add_argument('--gen-text-only', action='store_true',
                       help='Explicitly opt into unscored BabyVision-Gen text diagnostics')
    else:
        p.add_argument('--image-path', '--image_path', '--image-paths', '--image_paths',
                       dest='image_paths', nargs='+', required=True,
                       help='List of local image paths, paired with prompts by index')
        p.add_argument('--prompt', '--prompts', dest='prompts', nargs='+', required=True,
                       help='List of prompt strings; quote each prompt separately')
    p.add_argument('--output', '--output-file', '--output_file', required=True,
                   help='JSONL predictions; companion .meta.json and .summary.json are created')
    p.add_argument('--batchsize', '--batch-size', type=positive, default=1)
    p.add_argument('--gpu-ids', '--gpu_ids', default='0', help='Physical GPU IDs, e.g. 0,1 or "[0,1]"; cpu is supported')
    p.add_argument('--cache-root', default=paths['HF_HUB_CACHE'])
    p.add_argument('--max-new-tokens', type=positive, default=4096)
    p.add_argument('--max-pixels', type=positive, default=None,
                   help='Optional processor pixel cap; default preserves the model processor settings')
    p.add_argument('--thinking', choices=['default', 'on', 'off'], default='off')
    p.add_argument('--dtype', choices=['auto', 'bfloat16', 'float16', 'float32'], default='bfloat16')
    p.add_argument('--attn-implementation', choices=['sdpa', 'eager', 'flash_attention_2'], default='sdpa')
    p.add_argument('--seed', '--random-seed', '--random_seed', dest='seed', type=int, default=0,
                   help='Random seed for Python, NumPy, and PyTorch CPU/CUDA RNGs (default: 0)')
    p.add_argument('--resume', action='store_true', help='Resume only with exactly matching run configuration')
    p.add_argument('--dry-run', action='store_true', help='Validate inputs and write prompts without loading model weights')
    return p


def resolve_model(name, cache_root):
    supplied = Path(name).expanduser()
    if supplied.is_dir():
        result = snapshot(supplied)
    else:
        short = name.removeprefix('models--Qwen--').removeprefix('Qwen/')
        if short not in MODELS:
            raise ValueError(f'Unknown model {name}; use {MODELS}, Qwen/<name>, or a local directory')
        local = Path(cache_root) / f'models--Qwen--{short}'
        if local.is_dir():
            result = snapshot(local)
        else:
            from huggingface_hub import snapshot_download
            result = Path(snapshot_download(f'Qwen/{short}', local_files_only=True))
    if not (result / 'config.json').is_file():
        raise FileNotFoundError(f'Missing model config in {result}')
    return result.resolve()


def write_json(path, obj):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def main(full=False):
    p = parser(full)
    args = p.parse_args()
    if full and args.bench == 'babyvision-gen' and not args.gen_text_only:
        p.error('These Qwen models cannot generate images. Use --gen-text-only for unscored text diagnostics.')
    gpu = args.gpu_ids.strip()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    if gpu != 'cpu':
        gpu_ids = parse_ids([gpu])
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # Set visibility before importing torch/transformers.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    manifest = None
    if full:
        rows, manifest = load_cases(args.data_root, args.bench, args.split, parse_ids(args.case_ids))
    else:
        rows = load_image_prompts(args.image_paths, args.prompts)
    id_key = 'taskId' if full else 'sample_id'
    summarize_run = summarize if full else summarize_inputs
    model_path = resolve_model(args.model, args.cache_root)
    output = Path(args.output).expanduser().resolve()
    meta_path = output.with_suffix(output.suffix + '.meta.json')
    summary_path = output.with_suffix(output.suffix + '.summary.json')
    protected = {Path(r['image_path']) for r in rows}
    if manifest is not None:
        protected.add(manifest)
    if any(path.resolve() in protected for path in (output, meta_path, summary_path)):
        raise ValueError('Output files cannot replace an input image or manifest')
    output.parent.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items()
              if k not in ('resume', 'output', 'case_ids', 'image_paths', 'prompts')}
    config.update(model_path=str(model_path), selected_ids=[r[id_key] for r in rows],
                  prompt_version=1, decoding='greedy')
    if full:
        config['manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    else:
        config.update(input_mode='image_prompt', inputs=rows)
    records = []
    if output.exists():
        if not args.resume:
            raise FileExistsError(f'{output} exists; choose another file or --resume')
        if json.loads(meta_path.read_text())['config'] != config:
            raise ValueError('Resume configuration differs from the original run')
        # Fail explicitly on torn JSONL; do not silently lose or double-count records.
        records = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
        done = [r[id_key] for r in records]
        if len(done) != len(set(done)) or not set(done).issubset(config['selected_ids']):
            raise ValueError('Invalid IDs in existing output')
    else:
        if args.resume:
            raise FileNotFoundError('--resume requires an existing output')
        if meta_path.exists() or summary_path.exists():
            raise FileExistsError('Companion output files already exist; choose another output')
        write_json(meta_path, {'config': config, 'created_utc': datetime.now(timezone.utc).isoformat()})
        output.touch(exist_ok=False)
    done = {r[id_key] for r in records}
    remaining = [r for r in rows if r[id_key] not in done]
    if not remaining:
        write_json(summary_path, summarize_run(records))
        print(f'Already complete: {output}')
        return
    model = processor = None
    if not args.dry_run:
        import torch
        import transformers
        from transformers import AutoProcessor, AutoModelForImageTextToText, set_seed
        if gpu != 'cpu' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Check GPU access/driver; use --gpu-ids cpu only for deliberate CPU inference.')
        set_seed(args.seed)
        print(f'Random seed: {args.seed}', flush=True)
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        processor.tokenizer.padding_side = 'left'
        if args.max_pixels:
            # Qwen fast processors use size.longest_edge; also retain max_pixels for older versions.
            processor.image_processor.max_pixels = args.max_pixels
            processor.image_processor.size['longest_edge'] = args.max_pixels
            if args.max_pixels < processor.image_processor.size.get('shortest_edge', 0):
                raise ValueError('--max-pixels is smaller than the processor minimum pixel count')
        dtype = args.dtype if args.dtype == 'auto' else getattr(torch, args.dtype)
        print(f'Loading {model_path} on {gpu}; {len(remaining)} cases', flush=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, local_files_only=True, dtype=dtype,
            device_map='cpu' if gpu == 'cpu' else 'auto',
            attn_implementation=args.attn_implementation).eval()
        meta = json.loads(meta_path.read_text())
        meta['runtime'] = {'torch': torch.__version__, 'transformers': transformers.__version__,
                           'devices': {str(k): str(v) for k, v in getattr(model, 'hf_device_map', {}).items()},
                           'image_processor': processor.image_processor.to_dict()}
        write_json(meta_path, meta)
    with output.open('a', encoding='utf-8') as handle:
        for start in range(0, len(remaining), args.batchsize):
            batch = remaining[start:start + args.batchsize]
            prompts = [prompt_for(row, args.bench) if full else row['prompt'] for row in batch]
            started = time.perf_counter()
            if args.dry_run:
                responses = [''] * len(batch)
                token_counts = [0] * len(batch)
                finishes = ['dry_run'] * len(batch)
            else:
                from PIL import Image
                images = []
                for row in batch:
                    with Image.open(row['image_path']) as im:
                        images.append(im.convert('RGB'))
                chats = [[{'role': 'user', 'content': [
                    {'type': 'image'}, {'type': 'text', 'text': prompt}]}] for prompt in prompts]
                template_kwargs = {}
                if args.thinking != 'default' and model.config.model_type == 'qwen3_5':
                    template_kwargs['enable_thinking'] = args.thinking == 'on'
                texts = [processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                                        **template_kwargs) for chat in chats]
                inputs = processor(text=texts, images=images, padding=True, return_tensors='pt')
                inputs = inputs.to(model.device)
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                suffix = generated[:, inputs['input_ids'].shape[1]:]
                responses = processor.batch_decode(suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                eos = model.generation_config.eos_token_id
                eos = set(eos if isinstance(eos, list) else [eos])
                token_counts, finishes = [], []
                for sequence in suffix.tolist():
                    stop = next((i for i, token in enumerate(sequence) if token in eos), None)
                    token_counts.append(stop + 1 if stop is not None else len(sequence))
                    finishes.append('eos' if stop is not None else 'length')
            elapsed = time.perf_counter() - started
            for row, prompt, response, count, finish in zip(batch, prompts, responses, token_counts, finishes):
                result = ({key: row.get(key) for key in ('taskId', 'type', 'subtype', 'ansType')}
                          if full else {'sample_id': row['sample_id'], 'image_path': row['image_path']})
                if full:
                    result.update(benchmark=args.bench, image=row['image_path'])
                result.update(model=args.model,
                              prompt=prompt, response=response, generated_tokens=count, finish_reason=finish,
                              batch_seconds=elapsed, batch_size=len(batch))
                if full:
                    result['score'] = (score(row, response, args.bench) if not args.dry_run else
                                       {'prediction': None, 'reference': None, 'correct': None, 'metric': 'dry_run'})
                handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                handle.flush()
                records.append(result)
                print(f"[{len(records)}/{len(rows)}] {id_key}={row[id_key]} {finish}: {response[:120]!r}", flush=True)
            write_json(summary_path, summarize_run(records))
    print(f'Predictions: {output}\nSummary: {summary_path}', flush=True)
