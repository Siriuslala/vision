"""Read-only local viewer for prediction JSONL and benchmark reference data."""
import argparse
import json
import mimetypes
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from markdown_it import MarkdownIt

STATIC = Path(__file__).with_name('static')
MARKDOWN = MarkdownIt('commonmark', {'html': False, 'breaks': True}).enable(['table', 'strikethrough']).disable('image')


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('Expected a JSON object')
                rows.append(row)
            except ValueError as exc:
                raise ValueError(f'{path.name}:{number}: {exc}') from exc
    return rows


def display(value) -> str:
    if value is None:
        return ''
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)


def first_value(row: dict, *keys):
    """Select a populated field without discarding numeric answers such as zero."""
    return next((row[k] for k in keys if row.get(k) is not None and row[k] != ''), None)


def reference_answer(row: dict):
    answer = first_value(row, 'reference', 'reference_answer', 'ground_truth', 'answer')
    if answer is not None:
        return answer
    if row.get('blankAns') is not None:
        return row['blankAns']
    if row.get('choiceAns') is not None:
        return chr(65 + int(row['choiceAns']))
    return None


class Results:
    def __init__(self, paths: list[Path], manifest: Path | None = None):
        self.paths = [path.expanduser().resolve() for path in paths]
        self.manifest = manifest
        references = {}
        if manifest:
            for row in read_rows(manifest):
                key = str(row['taskId'])
                if key in references:
                    raise ValueError(f'Duplicate taskId in manifest: {key}')
                references[key] = row
        self.runs = []
        for path in self.paths:
            records, seen = [], set()
            for row in read_rows(path):
                key = row.get('taskId', row.get('sample_id'))
                if key is None or str(key) in seen:
                    raise ValueError(f'{path.name}: missing or duplicate sample ID: {key}')
                seen.add(str(key))
                if not isinstance(row.get('response'), str):
                    raise ValueError(f'{path.name}: sample {key} requires a response string')
                ref = references.get(str(key), {}) if 'taskId' in row else {}
                score = row.get('score') or {}
                answer = reference_answer(row)
                if answer is None:
                    answer = score.get('reference')
                if answer is None:
                    answer = reference_answer(ref)
                image = row.get('image_path') or row.get('image')
                image_base = path.parent
                if not image and manifest:
                    image = ref.get('image_path') or ref.get('image')
                    image_base = manifest.expanduser().resolve().parent
                if not isinstance(image, str) or not image:
                    raise ValueError(f'{path.name}: sample {key} has no image path')
                image_path = (image_base / Path(image).expanduser()).resolve()
                if not image_path.is_file():
                    raise ValueError(f'{path.name}: missing image for sample {key}: {image}')
                question = first_value(row, 'question')
                question_source = row
                if question is None:
                    question = first_value(ref, 'question')
                    question_source = ref
                if question is None:
                    question = row.get('prompt', '')
                    options = []  # The saved prompt already includes its options.
                else:
                    options = row.get('options', question_source.get('options')) or []
                question = display(question)
                if options:
                    question += '\n\n' + '\n'.join(f'{chr(65+i)}. {option}' for i, option in enumerate(options))
                reasoning = first_value(row, 'reference_reasoning', 'coT', 'cot', 'CoT', 'rationale', 'solution')
                if reasoning is None:
                    reasoning = first_value(ref, 'reference_reasoning', 'coT', 'reasoning', 'rationale', 'cot', 'CoT', 'solution')
                record = dict(id=str(key), question=display(question), reference=display(answer),
                              reference_reasoning=display(reasoning), response=row['response'],
                              reasoning=display(row.get('reasoning') or row.get('reasoning_content')),
                              prompt=display(row.get('prompt')), correct=score.get('correct'),
                              prediction=display(score.get('prediction')), model=row.get('model', ''),
                              finish_reason=row.get('finish_reason', ''), tokens=row.get('generated_tokens'),
                              category=' / '.join(str(row[k]) for k in ('type', 'subtype') if row.get(k)),
                              image_path=image_path)
                records.append(record)
            self.runs.append({'name': path.name, 'records': records})

    def record(self, run: int, index: int) -> dict:
        if run < 0 or run >= len(self.runs) or index < 0 or index >= len(self.runs[run]['records']):
            raise IndexError('Unknown run or sample')
        return self.runs[run]['records'][index]

    def reload(self) -> None:
        fresh = Results(self.paths, self.manifest)
        self.runs = fresh.runs

    def catalog(self) -> list[dict]:
        return [{'name': run['name'], 'samples': [dict(index=i, id=r['id'], correct=r['correct'],
                 finish_reason=r['finish_reason'], question=r['question']) for i,r in enumerate(run['records'])]}
                for run in self.runs]


def handler_for(results: Results):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body, mime='application/json; charset=utf-8', status=200):
            payload = json.dumps(body, ensure_ascii=False).encode() if isinstance(body, (dict,list)) else body
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self'; object-src 'none'; frame-src 'none'; base-uri 'none'")
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path == '/api/runs':
                    results.reload()
                    return self.send(results.catalog())
                if url.path in ('/api/sample', '/api/image'):
                    query = parse_qs(url.query)
                    record = results.record(int(query['run'][0]), int(query['index'][0]))
                    if url.path == '/api/image':
                        path = record['image_path'].resolve()
                        mime = mimetypes.guess_type(path.name)[0]
                        if mime not in ('image/png','image/jpeg','image/webp','image/gif','image/bmp'):
                            raise ValueError('Unsupported image format')
                        return self.send(path.read_bytes(), mime)
                    data = {k:v for k,v in record.items() if k != 'image_path'}
                    data['html'] = {k: MARKDOWN.render(data[k]) for k in
                                    ('question','reference','reference_reasoning','response','reasoning')}
                    return self.send(data)
                files = {'/':'index.html', '/app.js':'app.js', '/style.css':'style.css'}
                if url.path not in files:
                    return self.send({'error':'Not found'}, status=404)
                path = STATIC / files[url.path]
                return self.send(path.read_bytes(), mimetypes.guess_type(path.name)[0] + '; charset=utf-8')
            except (ValueError, KeyError, IndexError, OSError) as exc:
                self.send({'error':str(exc)}, status=400)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jsonl', nargs='+', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=1044)
    args = parser.parse_args()
    results = Results(args.jsonl, args.manifest)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(results))
    print(f'Prediction viewer: http://{args.host}:{server.server_port} ({len(results.runs)} runs)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
