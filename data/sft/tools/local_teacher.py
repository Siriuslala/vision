"""Offline Qwen3.5 inference shared by generation and quality checks."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path


def json_object(response: str) -> dict:
    text = response.strip()
    fenced = re.search(r'```(?:json)?\s*\n(.*?)\n```\s*$',text,re.S|re.I)
    if fenced:
        text = fenced[1].strip()
    else:
        text = text.rstrip('`').rstrip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError as error:
        if error.msg=='Extra data' and text[error.pos:].strip()=='}':
            result = json.loads(text[:error.pos])
            if not isinstance(result,dict):
                raise ValueError('Teacher response must be a JSON object')
            return result
        # Some local outputs place explanatory prose before the final JSON. Accept
        # exactly one complete trailing object; keep the untouched output in logs.
        candidates = []
        for index, char in enumerate(text):
            if char != '{':
                continue
            try:
                value, end = json.JSONDecoder().raw_decode(text[index:])
                if isinstance(value, dict) and not text[index+end:].strip():
                    candidates.append(value)
            except json.JSONDecodeError:
                pass
        if len(candidates) != 1:
            raise ValueError('Teacher response has no unambiguous final JSON object')
        result = candidates[0]
    if not isinstance(result, dict):
        raise ValueError('Teacher response must be a JSON object')
    return result


class LocalTeacher:
    def __init__(self, model_name: str, cache_root: Path, gpu: str, max_pixels: int = 1048576):
        if model_name.removeprefix('models--Qwen--').removeprefix('Qwen/') not in ('Qwen3.5-4B', 'Qwen3.5-9B'):
            raise ValueError('SFT teachers must use Qwen3.5-4B or Qwen3.5-9B')
        os.environ.update(CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed
        from eva.inference import resolve_model
        if gpu != 'cpu' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        set_seed(0)
        path = resolve_model(model_name, cache_root)
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.processor.image_processor.max_pixels = max_pixels
        self.processor.image_processor.size['longest_edge'] = max_pixels
        self.model = AutoModelForImageTextToText.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16,
                         device_map='cpu' if gpu == 'cpu' else 'auto', attn_implementation='sdpa').eval()
        self.info = {'model': model_name, 'snapshot': str(path), 'transformers': transformers.__version__,
                     'torch': torch.__version__, 'seed': 0, 'thinking': False, 'max_pixels': max_pixels}
        self.calls = 0

    def generate(self, images: list, prompt: str, max_new_tokens: int = 2048) -> dict:
        import torch
        chat = [{'role':'system','content':'Follow the requested JSON schema exactly. Return one valid JSON object only, with no Markdown or surrounding prose.'},
                {'role': 'user', 'content': [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': prompt}]}]
        text = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.processor(text=[text], **({'images':images} if images else {}), return_tensors='pt').to(self.model.device)
        started = time.monotonic()
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        tokens = generated[0, inputs['input_ids'].shape[1]:].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        response = self.processor.decode(tokens, skip_special_tokens=True)
        self.calls += 1
        return {'response': response, 'finish_reason': 'eos' if any(t in eos for t in tokens) else 'length',
                'generated_tokens': len(tokens), 'seconds': time.monotonic()-started, 'prompt': prompt,
                'image_sizes': [list(image.size) for image in images], 'call': self.calls}
