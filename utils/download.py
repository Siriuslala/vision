"""Download pinned Hugging Face datasets with persistent HTTP Range recovery."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import sys
import threading
from urllib.parse import parse_qs, urlsplit
import uuid

import httpx
from filelock import FileLock, Timeout as LockTimeout
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import build_hf_headers
from tqdm import tqdm

STATE_DIR = '.download-state'
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class DownloadError(RuntimeError):
    """A download could not safely continue; local bytes are retained."""


class RetryableDownloadError(DownloadError):
    pass


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    digest: str
    algorithm: str

    def validate(self) -> None:
        path = PurePosixPath(self.path)
        if (not self.path or path.is_absolute() or path.as_posix() != self.path
                or any(part in ('.', '..') for part in path.parts)
                or path.parts[0] in (STATE_DIR, '.cache') or '\\' in self.path):
            raise DownloadError(f'Unsafe repository path: {self.path!r}')
        length = {'sha256': 64, 'git-sha1': 40}.get(self.algorithm)
        if (type(self.size) is not int or self.size < 0 or length is None or not isinstance(self.digest, str)
                or re.fullmatch(r'[0-9a-f]{%d}' % length, self.digest) is None):
            raise DownloadError(f'Invalid remote metadata: {self.path}')


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def local_path(root: Path, relative: str) -> Path:
    """Reject symlinks, including links that redirect state or file writes."""
    result = root / relative
    if not result.is_relative_to(root) or '..' in Path(relative).parts:
        raise DownloadError(f'Unsafe local path: {relative}')
    for path in (result, *result.parents):
        if path == root:
            break
        if path.is_symlink():
            raise DownloadError(f'Symlink is not supported: {path}')
    return result


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    if temp.is_symlink():
        raise DownloadError(f'Symlink is not supported: {temp}')
    with temp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    sync_directory(path.parent)


@contextmanager
def dataset_lock(root: Path):
    state = local_path(root, STATE_DIR)
    state.mkdir(parents=True, exist_ok=True)
    lock = local_path(root, f'{STATE_DIR}/download.lock')
    with lock.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DownloadError('Another downloader holds this destination lock.') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load_manifest(root: Path, repo_id: str, revision: str, endpoint: str,
                  attempts: int, timeout: int, stop: threading.Event) -> tuple[str, list[RemoteFile]]:
    path = local_path(root, f'{STATE_DIR}/manifest.json')
    if path.exists():
        data = json.loads(path.read_text())
        if (data.get('version') != 1 or data.get('repo_id') != repo_id
                or data.get('revision') != revision):
            raise DownloadError('Destination manifest belongs to another repository/revision; use a new directory.')
    else:
        for attempt in range(attempts):
            try:
                info = HfApi(endpoint=endpoint).repo_info(
                    repo_id, repo_type='dataset', revision=revision,
                    files_metadata=True, timeout=timeout)
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
                if (status is not None and status not in RETRY_STATUS) or attempt + 1 == attempts:
                    raise DownloadError(f'Cannot fetch repository metadata ({type(error).__name__}, HTTP {status}).') from None
                if stop.wait(min(5 * (attempt + 1), 60)):
                    raise DownloadError('Stopped; existing files retained.') from None
        files = []
        for sibling in info.siblings or []:
            lfs = sibling.lfs
            files.append(RemoteFile(sibling.rfilename, sibling.size,
                                    lfs.sha256 if lfs else sibling.blob_id,
                                    'sha256' if lfs else 'git-sha1'))
        data = {'version': 1, 'repo_id': repo_id, 'revision': revision,
                'commit': info.sha, 'files': [asdict(file) for file in files]}
    if not isinstance(data.get('commit'), str) or not re.fullmatch('[0-9a-f]{40}', data['commit']):
        raise DownloadError('Invalid pinned commit in manifest.')
    files = [RemoteFile(**item) for item in data['files']]
    if not files or len({file.path for file in files}) != len(files):
        raise DownloadError('Empty or duplicate file list in manifest.')
    names = {file.path for file in files}
    for file in files:
        file.validate()
        if any(parent.as_posix() in names for parent in PurePosixPath(file.path).parents):
            raise DownloadError(f'Conflicting repository paths: {file.path}')
    if not path.exists():
        atomic_json(path, data)
    return data['commit'], files


def verify_file(path: Path, file: RemoteFile, stop: threading.Event) -> None:
    if path.stat().st_size != file.size:
        raise DownloadError(f'Size mismatch; preserved for inspection: {path}')
    digest = hashlib.sha256() if file.algorithm == 'sha256' else hashlib.sha1()
    if file.algorithm == 'git-sha1':
        digest.update(f'blob {file.size}\0'.encode())
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            if stop.is_set():
                raise DownloadError('Stopped during verification; files retained.')
            digest.update(block)
    if digest.hexdigest() != file.digest:
        raise DownloadError(f'Checksum mismatch; preserved for inspection: {path}')


def receipt_data(path: Path, file: RemoteFile) -> dict:
    stat = path.stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'digest': file.digest, 'algorithm': file.algorithm}


class Downloader:
    """Keep a contiguous local prefix across failures; publish only verified files."""

    def __init__(self, root: Path, repo_id: str, commit: str, endpoint: str,
                 stop: threading.Event, *, attempts: int = 10, timeout: int = 120,
                 range_bytes: int = 64 * 1024 * 1024, import_http_partials: bool = False,
                 client_factory=None):
        self.root, self.repo_id, self.commit, self.endpoint = root, repo_id, commit, endpoint
        self.stop, self.attempts, self.timeout, self.range_bytes = stop, attempts, timeout, range_bytes
        self.import_http_partials = import_http_partials
        self.client_factory = client_factory or (lambda: httpx.Client(
            follow_redirects=True, timeout=httpx.Timeout(timeout, connect=30), trust_env=True))
        self.progress = None

    def paths(self, file: RemoteFile) -> tuple[Path, Path, Path]:
        key = hashlib.sha256(file.path.encode()).hexdigest()
        return (local_path(self.root, file.path),
                local_path(self.root, f'{STATE_DIR}/partials/{key}.part'),
                local_path(self.root, f'{STATE_DIR}/verified/{key}.json'))

    def inspect(self, file: RemoteFile) -> tuple[bool, int]:
        final, part, receipt = self.paths(file)
        if final.exists():
            expected = receipt_data(final, file)
            if final.stat().st_size != file.size:
                raise DownloadError(f'Existing file has wrong size; not overwriting: {final}')
            if not receipt.exists() or json.loads(receipt.read_text()) != expected:
                tqdm.write(f'Verifying existing file: {file.path}', file=sys.stderr)
                verify_file(final, file, self.stop)
                atomic_json(receipt, expected)
            return True, file.size
        if not part.exists() and self.import_http_partials:
            legacy_dir = local_path(self.root, '.cache/huggingface/download/' +
                                    str(PurePosixPath(file.path).parent))
            candidates = [path for path in legacy_dir.glob('*.incomplete')
                          if file.digest in path.name.split('.') and not path.is_symlink()]
            if candidates:
                # This is opt-in: Xet's sparse, out-of-order files are not HTTP prefixes.
                lock_path = local_path(self.root, '.cache/huggingface/download/' + file.path + '.lock')
                try:
                    with FileLock(str(lock_path), timeout=0):
                        source = max(candidates, key=lambda path: path.stat().st_size)
                        if source.stat().st_size > file.size:
                            raise DownloadError(f'Legacy partial exceeds expected size: {source}')
                        part.parent.mkdir(parents=True, exist_ok=True)
                        os.rename(source, part)
                        sync_directory(part.parent)
                        tqdm.write(f'Imported HTTP partial: {file.path} ({part.stat().st_size:,} bytes)', file=sys.stderr)
                except LockTimeout:
                    raise DownloadError(f'Legacy downloader is still writing {file.path}; stop it before migration.') from None
        size = part.stat().st_size if part.exists() else 0
        if size > file.size:
            raise DownloadError(f'Partial file exceeds expected size: {part}')
        return False, size

    def transfer_range(self, client: httpx.Client, file: RemoteFile, part: Path) -> None:
        start = part.stat().st_size if part.exists() else 0
        end = min(start + self.range_bytes, file.size) - 1
        # Always start at the resolver, never persist a signed CDN URL. The nonce avoids
        # stale redirect responses in intermediaries; it is not added to the signed URL.
        url = hf_hub_url(self.repo_id, file.path, repo_type='dataset',
                         revision=self.commit, endpoint=self.endpoint)
        headers = build_hf_headers()
        headers.update({'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity',
                        'Cache-Control': 'no-cache'})
        with client.stream('GET', url, params={'download': 'true', '_download_nonce': uuid.uuid4().hex},
                           headers=headers) as response:
            status = response.status_code
            if status in RETRY_STATUS:
                raise RetryableDownloadError(f'HTTP {status}')
            if status == 403:
                # Read only a bounded error body. Do not print signed URLs or credentials.
                body = b''
                for chunk in response.iter_bytes():
                    body += chunk[:8192 - len(body)]
                    if len(body) >= 8192:
                        break
                parsed = urlsplit(str(response.url))
                query = parse_qs(parsed.query)
                if ('Signature' in query and ('Expires' in query or 'Policy' in query)
                        and b'expir' in body.lower()):
                    raise RetryableDownloadError('CDN signature expired; refreshing resolver URL')
                raise DownloadError(f'HTTP 403 permission denied for {file.path}')
            if status not in (200, 206):
                raise DownloadError(f'HTTP {status} for {file.path}; partial file retained')
            if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
                raise DownloadError(f'Encoded response cannot be safely resumed: {file.path}')
            if status == 206:
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                if (not match or tuple(map(int, match.groups()))[0] != start
                        or int(match[3]) != file.size or not start <= int(match[2]) <= end):
                    raise DownloadError(f'Invalid Content-Range for {file.path}; local bytes retained')
                response_end = int(match[2])
            elif start != 0:
                raise DownloadError(f'Server ignored Range for {file.path}; refusing to truncate existing bytes')
            else:
                # Small Git files may ignore Range. A complete response at offset zero is safe.
                response_end = file.size - 1
            part.parent.mkdir(parents=True, exist_ok=True)
            with part.open('ab', buffering=0) as stream:
                offset = start
                try:
                    for chunk in response.iter_raw():
                        if self.stop.is_set():
                            raise DownloadError('Stopped; partial files retained.')
                        if offset + len(chunk) > response_end + 1:
                            raise DownloadError(f'Response exceeds declared range: {file.path}')
                        written = stream.write(chunk)
                        if written != len(chunk):
                            raise OSError('Short local write; partial file retained')
                        offset += written
                        if self.progress is not None:
                            self.progress.update(written)
                finally:
                    os.fsync(stream.fileno())
            if offset != response_end + 1:
                raise RetryableDownloadError(f'Truncated response for {file.path}')

    def download(self, file: RemoteFile) -> None:
        final, part, receipt = self.paths(file)
        failures = 0
        with self.client_factory() as client:
            while not part.exists() or part.stat().st_size < file.size:
                if self.stop.is_set():
                    raise DownloadError('Stopped; partial files retained.')
                if file.size == 0:
                    part.parent.mkdir(parents=True, exist_ok=True)
                    part.touch()
                    break
                try:
                    self.transfer_range(client, file, part)
                    failures = 0
                except (httpx.TransportError, RetryableDownloadError) as error:
                    failures += 1
                    offset = part.stat().st_size if part.exists() else 0
                    if failures >= self.attempts:
                        raise DownloadError(f'{file.path}: retry limit reached; retained {offset}/{file.size} bytes') from None
                    # Our retryable errors contain sanitized reasons; transport exceptions
                    # may contain signed URLs, so print only their class name.
                    reason = str(error) if isinstance(error, RetryableDownloadError) else type(error).__name__
                    tqdm.write(f'{file.path}: {reason}; retained {offset}/{file.size} bytes; '
                               f'refreshing URL ({failures}/{self.attempts})', file=sys.stderr)
                    if self.stop.wait(min(5 * failures, 60)):
                        raise DownloadError('Stopped; partial files retained.') from None
        verify_file(part, file, self.stop)
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise DownloadError(f'Destination appeared during download; not overwriting: {final}')
        os.replace(part, final)
        sync_directory(final.parent)
        atomic_json(receipt, receipt_data(final, file))
        tqdm.write(f'Verified: {file.path}', file=sys.stderr)


def run(args, stop: threading.Event) -> None:
    root = args.local_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    endpoint = args.endpoint or os.environ.get('HF_ENDPOINT') or 'https://huggingface.co'
    if urlsplit(endpoint).scheme not in ('http', 'https'):
        raise DownloadError('Endpoint must be an HTTP(S) URL.')
    with dataset_lock(root):
        commit, files = load_manifest(root, args.repo_id, args.revision, endpoint,
                                      args.attempts, args.etag_timeout, stop)
        print(f'Dataset: {args.repo_id}@{commit}\nDestination: {root}\nEndpoint: {endpoint}', flush=True)
        downloader = Downloader(root, args.repo_id, commit, endpoint, stop,
                                attempts=args.attempts, timeout=args.download_timeout,
                                range_bytes=args.range_mib * 1024 * 1024,
                                import_http_partials=args.import_http_partials)
        pending, initial, verified = [], 0, 0
        for file in files:
            ready, size = downloader.inspect(file)
            initial += size
            if ready:
                verified += file.size
            else:
                pending.append(file)
        total = sum(file.size for file in files)
        print(f'Verified files: {len(files) - len(pending)}/{len(files)}; verified bytes: {verified:,}; '
              f'partial bytes: {initial - verified:,}; total bytes: {total:,}', flush=True)
        legacy = list(root.glob('.cache/huggingface/download/**/*.incomplete'))
        if legacy:
            print(f'Found {len(legacy)} remaining legacy Hub partial files; these are left untouched. '
                  'Only use --import-http-partials for stopped HTTP downloads, never Xet.', flush=True)
        if args.dry_run:
            return
        with tqdm(total=total, initial=initial, unit='B', unit_scale=True,
                  desc='Local bytes (including unverified partials)') as progress:
            downloader.progress = progress
            with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                futures = [pool.submit(downloader.download, file) for file in pending]
                try:
                    for future in as_completed(futures):
                        future.result()
                except BaseException:
                    stop.set()
                    for future in futures:
                        future.cancel()
                    raise
        print(f'All {len(files)} files verified: {root}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--local-dir', required=True, type=Path)
    parser.add_argument('--revision', default='main')
    parser.add_argument('--endpoint', default='', help='Empty inherits HF_ENDPOINT; use official endpoint to bypass mirror')
    parser.add_argument('--max-workers', type=positive_int, default=2)
    parser.add_argument('--range-mib', type=positive_int, default=64)
    parser.add_argument('--download-timeout', type=positive_int, default=120)
    parser.add_argument('--etag-timeout', type=positive_int, default=60)
    parser.add_argument('--attempts', type=positive_int, default=10,
                        help='Consecutive failures per file; successful range resets this counter')
    parser.add_argument('--dry-run', action='store_true', help='Pin manifest and verify existing files without downloading content')
    parser.add_argument('--import-http-partials', action='store_true',
                        help='Adopt matching legacy .incomplete files from stopped HTTP downloads (never Xet)')
    args = parser.parse_args()
    if not args.local_dir.is_absolute():
        parser.error('--local-dir must be absolute')
    stop = threading.Event()
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, lambda *_: stop.set())
    try:
        run(args, stop)
    except (DownloadError, OSError, ValueError, TypeError, KeyError) as error:
        print(f'Download stopped: {error}', file=sys.stderr)
        raise SystemExit(1) from None
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == '__main__':
    main()
