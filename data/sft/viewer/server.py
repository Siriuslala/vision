"""Local JSONL-backed SFT inspection server with indexed paired reasoning modes."""
from __future__ import annotations
import argparse
import json
import mimetypes
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

STATIC = Path(__file__).with_name('static')


class Dataset:
    def __init__(self, config: dict):
        self.name = config['name']
        self.data_root = Path(config['data_root']).expanduser().resolve()
        paths = config.get('jsonl_paths') or [config['jsonl_path']]
        self.paths = [Path(p).expanduser().resolve() for p in paths]
        self.snapshots = {}
        self.groups = []
        self.ids = {}
        keys = {}
        for path in self.paths:
            stat = path.stat()
            self.snapshots[path] = (stat.st_size, stat.st_mtime_ns)
            with path.open('rb') as stream:
                line_number = 0
                while True:
                    offset = stream.tell(); line = stream.readline()
                    if not line: break
                    line_number += 1
                    if not line.strip(): continue
                    try:
                        row = json.loads(line)
                        self.validate(row)
                        sample_id = row['sample_id']
                        if sample_id in self.ids: raise ValueError('duplicate sample_id '+sample_id)
                        provenance = row.get('provenance', {})
                        # Only source-identical versions are shown side by side.
                        source_id = provenance.get('source_sample_id')
                        key = json.dumps([provenance.get('source_name'), provenance.get('source_revision'), source_id,
                                          row['image'].get('sha256') or row['image']['path'],row['query'],row['answer']],sort_keys=True,ensure_ascii=False) if source_id else sample_id
                        mode = row['reasoning_mode']
                        if key in keys and mode in self.groups[keys[key]]:
                            # Distinct demonstrations stay independently inspectable.
                            key += ':' + sample_id
                        if key not in keys:
                            keys[key] = len(self.groups); self.groups.append({})
                        index = keys[key]
                        self.groups[index][mode] = (path,offset)
                        self.ids[sample_id] = index
                    except (ValueError,KeyError,TypeError) as exc:
                        raise ValueError(f'{path.name}:{line_number}: {exc}') from exc
        if not self.groups: raise ValueError(f'{self.name}: JSONL contains no samples')

    @staticmethod
    def validate(row):
        if row.get('schema_version')!='1.1' or row.get('coordinate_system')!='image_long_side_normalized_xy':
            raise ValueError('Viewer requires schema 1.1 long-side coordinates')
        if row.get('reasoning_mode') not in ('interleaved','visual_only'): raise ValueError('Unknown reasoning mode')
        if not isinstance(row.get('sample_id'),str) or not row['sample_id']:raise ValueError('Missing sample_id')
        if not isinstance(row.get('trajectory'),list):raise ValueError('Expected an SFT trajectory array')
        if any(type(row['image'].get(k)) is not int or row['image'][k]<=0 for k in ('width','height')):raise ValueError('Invalid image dimensions')

    def record(self, index: int):
        if not 0<=index<len(self.groups): raise IndexError('Sample index outside dataset')
        result = {}
        for mode,(path,offset) in self.groups[index].items():
            stat = path.stat()
            if (stat.st_size,stat.st_mtime_ns)!=self.snapshots[path]:
                raise ValueError('Input JSONL changed; restart viewer to rebuild its index')
            with path.open('rb') as stream:
                stream.seek(offset);result[mode]=json.loads(stream.readline())
        return result

    def image_path(self,index):
        row = next(iter(self.record(index).values()))
        path=(self.data_root/row['image']['path']).resolve()
        if not path.is_relative_to(self.data_root):raise ValueError('Image path escapes data_root')
        if not path.is_file():raise FileNotFoundError('Image is missing under configured data_root')
        return path


def handler_for(datasets, font_path=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def send(self, data, content_type='application/json; charset=utf-8', status=200):
            payload=json.dumps(data,ensure_ascii=False).encode() if isinstance(data,(dict,list)) else data
            self.send_response(status);self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(payload)
        def do_GET(self):
            url=urlparse(self.path);query=parse_qs(url.query)
            try:
                if url.path=='/api/datasets':
                    self.send([{'id':i,'name':d.name,'count':len(d.groups),'records':len(d.ids)} for i,d in enumerate(datasets)]);return
                if url.path in ('/api/sample','/api/image','/api/search'):
                    dataset_id=int(query.get('dataset',['0'])[0])
                    if not 0<=dataset_id<len(datasets):raise IndexError('Unknown dataset')
                    dataset=datasets[dataset_id]
                    if url.path=='/api/search':
                        sample_id=query.get('sample_id',[''])[0]
                        if sample_id not in dataset.ids:raise KeyError('sample_id not found in this dataset')
                        self.send({'index':dataset.ids[sample_id]});return
                    index=int(query.get('index',['0'])[0])
                    if url.path=='/api/image':
                        path=dataset.image_path(index)
                        self.send(path.read_bytes(),mimetypes.guess_type(path.name)[0] or 'application/octet-stream');return
                    self.send({'index':index,'count':len(dataset.groups),'modes':dataset.record(index),'image_url':f'/api/image?dataset={dataset_id}&index={index}'});return
                if url.path=='/font.otf' and font_path:
                    self.send(font_path.read_bytes(),'font/otf');return
                name={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}.get(url.path)
                if not name:self.send({'error':'Not found'},status=404);return
                content=(STATIC/name).read_bytes()
                if name=='style.css' and font_path:
                    content=b'@font-face{font-family:AuditCJK;src:url(/font.otf);font-display:swap}body{font-family:Inter,-apple-system,BlinkMacSystemFont,Segoe UI,AuditCJK,sans-serif}'+content
                self.send(content,mimetypes.guess_type(name)[0]+'; charset=utf-8')
            except (ValueError,KeyError,IndexError,FileNotFoundError) as exc:
                self.send({'error':str(exc)},status=404 if isinstance(exc,(KeyError,IndexError,FileNotFoundError)) else 409)
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=18765)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    datasets=[Dataset(d) for d in config['datasets']]
    if not datasets:raise ValueError('Configure at least one dataset')
    font_path=Path(config['font_path']).expanduser().resolve() if config.get('font_path') else None
    if font_path and not font_path.is_file():raise FileNotFoundError('Configured font_path does not exist')
    server=ThreadingHTTPServer((args.host,args.port),handler_for(datasets,font_path))
    print(f'SFT viewer: http://{args.host}:{server.server_port} ({len(datasets)} datasets)',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
