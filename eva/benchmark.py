"""Benchmark selection, safe prompt construction, and conservative local scoring."""
import json
from pathlib import Path
import re
import unicodedata
from data.prepare_data import BENCHES


def parse_ids(values):
    if values is None:
        return None
    text = ' '.join(values).strip()
    ids = json.loads(text) if text.startswith('[') else re.split(r'[,\s]+', text)
    if not isinstance(ids, list) or not ids:
        raise ValueError('case_ids must be a nonempty list of taskId values')
    result = []
    for value in ids:
        if isinstance(value, bool) or not re.fullmatch(r'\d+', str(value)):
            raise ValueError(f'Invalid taskId: {value!r}')
        result.append(int(value))
    if len(result) != len(set(result)):
        raise ValueError('Duplicate case_ids are not allowed')
    return result


def load_cases(root, bench, split, ids=None):
    base = Path(root).expanduser().resolve() / BENCHES[bench]
    path = base / f'{split}.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    index = {row['taskId']: row for row in rows}
    if len(index) != len(rows):
        raise ValueError('Duplicate taskId in manifest')
    if ids is not None:
        missing = sorted(set(ids) - index.keys())
        if missing:
            raise ValueError(f'Unknown taskId values: {missing}; IDs are not row indices')
        rows = [index[i] for i in ids]
    for row in rows:
        image = (base / row['image']).resolve()
        if not image.is_relative_to(base) or not image.is_file():
            raise ValueError(f'Invalid image path: {image}')
        row['image_path'] = str(image)
    if not rows:
        raise ValueError('No cases selected')
    return rows, path


def prompt_for(row, bench):
    if bench == 'babyvision-gen':
        return (row['generationPrompt'] + '\nThis is a text-only diagnostic: describe precisely '
                'where and how you would annotate the image. You cannot output an edited image.')
    text = row['question'].strip()
    if row['ansType'] == 'choice':
        text += '\nOptions:\n' + '\n'.join(f'{chr(65+i)}. {opt}' for i, opt in enumerate(row['options']))
        text += '\nGive your final answer on its own last line as "Final answer: X", where X is one option letter.'
    else:
        text += '\nGive your final answer on its own last line as "Final answer: <answer>". Follow the question\'s answer format.'
    return text


def normalize(value):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(value))).casefold().rstrip('.')


def score(row, response, bench):
    if bench == 'babyvision-gen':
        return {'prediction': None, 'reference': None, 'correct': None,
                'metric': 'not_scored_text_only_diagnostic'}
    final = response.split('</think>')[-1].strip()
    matches = re.findall(r'^\s*Final answer\s*:\s*(.+?)\s*$', final, flags=re.I | re.M)
    candidate = matches[-1].strip() if matches else final
    # Without the requested marker, accept only a standalone answer, never a letter in reasoning.
    if row['ansType'] == 'choice':
        reference = chr(65 + row['choiceAns'])
        match = re.fullmatch(r'\s*([A-Z])[.)]?\s*', candidate, flags=re.I)
        prediction = match[1].upper() if match else None
        if prediction and ord(prediction) - 65 >= len(row['options']):
            prediction = None
    else:
        reference = row['blankAns']
        prediction = candidate if candidate and '\n' not in candidate else None
    return {'prediction': prediction, 'reference': reference,
            'correct': prediction is not None and normalize(prediction) == normalize(reference),
            'metric': 'local_normalized_exact_match'}


def summarize(records):
    def group(items):
        scored = [r for r in items if r['score']['correct'] is not None]
        return {'count': len(items), 'scored': len(scored),
                'correct': sum(r['score']['correct'] for r in scored),
                'accuracy': sum(r['score']['correct'] for r in scored) / len(scored) if scored else None,
                'unparsed': sum(r['score']['prediction'] is None for r in scored),
                'truncated': sum(r.get('finish_reason') == 'length' for r in items)}
    result = group(records)
    for key in ('type', 'subtype', 'ansType'):
        result['by_' + key] = {v: group([r for r in records if r.get(key) == v])
                              for v in sorted({r[key] for r in records if r.get(key)})}
    result['metric_note'] = 'Local strict scoring, not the official LLM-judge metric; inspect unparsed/blank answers.'
    return result
