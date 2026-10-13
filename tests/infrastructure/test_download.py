import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from filelock import FileLock
import httpx

from utils.download import (DownloadError, Downloader, RemoteFile, STATE_DIR,
                            atomic_json, dataset_lock, load_manifest, verify_file)


class BrokenStream(httpx.SyncByteStream):
    def __iter__(self):
        yield b'abcd'
        raise httpx.RemoteProtocolError('connection interrupted')


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = b'abcdefghijklmnop'
        self.file = RemoteFile('data/train.parquet', len(self.data),
                               hashlib.sha256(self.data).hexdigest(), 'sha256')
        self.stop = threading.Event()
        self.wait = patch.object(self.stop, 'wait', return_value=False)
        self.wait.start()
        self.addCleanup(self.wait.stop)

    def downloader(self, handler, **kwargs):
        return Downloader(self.root, 'test/data', 'a' * 40, 'https://hub.test', self.stop,
                          range_bytes=8, client_factory=lambda: httpx.Client(
                              transport=httpx.MockTransport(handler), follow_redirects=True), **kwargs)

    def response(self, request):
        start, end = map(int, request.headers['range'].removeprefix('bytes=').split('-'))
        return httpx.Response(206, headers={'Content-Range': f'bytes {start}-{end}/{len(self.data)}'},
                              stream=httpx.ByteStream(self.data[start:end + 1]))

    def test_disconnect_expiry_and_new_url_preserve_prefix(self):
        requests = []
        nonces = []
        def handler(request):
            requests.append(request.headers['range'])
            nonces.append(request.url.params['_download_nonce'])
            if len(requests) == 1:
                return httpx.Response(206, headers={'Content-Range': 'bytes 0-7/16'}, stream=BrokenStream())
            if len(requests) == 2:
                return httpx.Response(302, headers={'Location': 'https://us.aws.cdn.hf.co/file?Expires=1&Signature=test'})
            if request.url.host == 'us.aws.cdn.hf.co':
                return httpx.Response(403, text='Auth failed: credentials expired')
            return self.response(request)
        # Redirects have no resolver nonce, so record only resolver requests.
        def wrapped(request):
            if request.url.host == 'us.aws.cdn.hf.co':
                return httpx.Response(403, text='Auth failed: credentials expired')
            return handler(request)
        downloader = self.downloader(wrapped)
        downloader.download(self.file)
        self.assertEqual((self.root / self.file.path).read_bytes(), self.data)
        self.assertEqual(requests[:3], ['bytes=0-7', 'bytes=4-11', 'bytes=4-11'])
        self.assertEqual(len(nonces), len(set(nonces)))

    def test_retry_exhaustion_then_new_instance_resumes(self):
        first = True
        def handler(request):
            nonlocal first
            if first:
                first = False
                return httpx.Response(206, headers={'Content-Range': 'bytes 0-7/16'}, stream=BrokenStream())
            return httpx.Response(503)
        downloader = self.downloader(handler, attempts=2)
        with self.assertRaisesRegex(DownloadError, 'retained 4/16'):
            downloader.download(self.file)
        part = downloader.paths(self.file)[1]
        self.assertEqual(part.read_bytes(), b'abcd')
        ranges = []
        def resume(request):
            ranges.append(request.headers['range'])
            return self.response(request)
        second = self.downloader(resume)
        self.assertEqual(second.inspect(self.file), (False, 4))
        second.download(self.file)
        self.assertEqual(ranges[0], 'bytes=4-11')
        self.assertEqual((self.root / self.file.path).read_bytes(), self.data)

    def test_bad_ranges_and_permission_errors_keep_partial(self):
        for status, headers, body in [
            (200, {}, self.data),
            (206, {'Content-Range': 'bytes 0-7/16'}, self.data[:8]),
            (206, {'Content-Range': 'bytes 4-11/17'}, self.data[4:12]),
            (403, {}, b'Permission denied'),
            (416, {}, b''),
        ]:
            with self.subTest(status=status, headers=headers):
                calls = []
                def handler(request):
                    calls.append(request)
                    return httpx.Response(status, headers=headers, content=body)
                downloader = self.downloader(handler)
                part = downloader.paths(self.file)[1]
                part.parent.mkdir(parents=True, exist_ok=True)
                part.write_bytes(b'abcd')
                with self.assertRaises(DownloadError):
                    downloader.download(self.file)
                self.assertEqual(part.read_bytes(), b'abcd')
                self.assertEqual(len(calls), 1)

    def test_hash_failure_never_publishes(self):
        downloader = self.downloader(lambda request: httpx.Response(200, stream=httpx.ByteStream(b'x' * len(self.data))))
        with self.assertRaisesRegex(DownloadError, 'Checksum mismatch'):
            downloader.download(self.file)
        self.assertFalse((self.root / self.file.path).exists())
        self.assertTrue(downloader.paths(self.file)[1].exists())

    def test_complete_files_verified_and_reused_without_network(self):
        final = self.root / self.file.path
        final.parent.mkdir()
        final.write_bytes(self.data)
        downloader = self.downloader(lambda _: self.fail('No network expected'))
        self.assertEqual(downloader.inspect(self.file), (True, len(self.data)))
        self.assertEqual(downloader.inspect(self.file), (True, len(self.data)))
        final.write_bytes(b'x' * len(self.data))
        with self.assertRaisesRegex(DownloadError, 'Checksum mismatch'):
            downloader.inspect(self.file)

    def test_git_blob_checksum(self):
        final = self.root / 'README.md'
        final.write_bytes(b'hello')
        file = RemoteFile('README.md', 5, hashlib.sha1(b'blob 5\0hello').hexdigest(), 'git-sha1')
        verify_file(final, file, self.stop)

    def test_legacy_http_import_selects_longest_matching_hash(self):
        folder = self.root / '.cache/huggingface/download/data'
        folder.mkdir(parents=True)
        (folder / f'old.{self.file.digest}.one.incomplete').write_bytes(b'ab')
        longest = folder / f'old.{self.file.digest}.two.incomplete'
        longest.write_bytes(b'abcd')
        unrelated = folder / ('other.' + '0' * 64 + '.one.incomplete')
        unrelated.write_bytes(b'wrong')
        downloader = self.downloader(self.response, import_http_partials=True)
        self.assertEqual(downloader.inspect(self.file), (False, 4))
        self.assertFalse(longest.exists())
        self.assertTrue(unrelated.exists())
        downloader.download(self.file)
        self.assertEqual((self.root / self.file.path).read_bytes(), self.data)

    def test_active_legacy_download_is_not_imported(self):
        folder = self.root / '.cache/huggingface/download/data'
        folder.mkdir(parents=True)
        (folder / f'old.{self.file.digest}.one.incomplete').write_bytes(b'ab')
        downloader = self.downloader(self.response, import_http_partials=True)
        with FileLock(str(folder / 'train.parquet.lock')):
            with self.assertRaisesRegex(DownloadError, 'still writing'):
                downloader.inspect(self.file)

    def test_destination_lock_and_manifest_identity(self):
        with dataset_lock(self.root):
            with self.assertRaisesRegex(DownloadError, 'holds this destination'):
                with dataset_lock(self.root):
                    pass
        path = self.root / STATE_DIR / 'manifest.json'
        atomic_json(path, {'version': 1, 'repo_id': 'test/data', 'revision': 'main',
                          'commit': 'a' * 40, 'files': [self.file.__dict__]})
        commit, files = load_manifest(self.root, 'test/data', 'main', 'https://unused', 1, 1, self.stop)
        self.assertEqual(commit, 'a' * 40)
        self.assertEqual(files, [self.file])
        with self.assertRaisesRegex(DownloadError, 'another repository'):
            load_manifest(self.root, 'different/data', 'main', 'https://unused', 1, 1, self.stop)

    def test_unsafe_paths_and_symlinks(self):
        for path in ('../escape', '/absolute', '.download-state/manifest.json', 'a/../b', '.cache/file'):
            with self.subTest(path=path), self.assertRaises(DownloadError):
                RemoteFile(path, 16, self.file.digest, 'sha256').validate()
        (self.root / 'data').symlink_to('/tmp')
        with self.assertRaisesRegex(DownloadError, 'Symlink'):
            self.downloader(self.response).inspect(self.file)

    def test_real_http_sigterm_and_process_restart(self):
        self.check_process_restart(force=False)

    def test_real_http_sigkill_and_process_restart(self):
        self.check_process_restart(force=True)

    def check_process_restart(self, force):
        data = b'a' * (1024 * 1024 + 17)
        file = RemoteFile('file.bin', len(data), hashlib.sha256(data).hexdigest(), 'sha256')
        atomic_json(self.root / STATE_DIR / 'manifest.json', {
            'version': 1, 'repo_id': 'test/data', 'revision': 'main',
            'commit': 'a' * 40, 'files': [file.__dict__]})
        requests = []
        release = threading.Event()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_GET(self):
                start, end = map(int, self.headers['Range'].removeprefix('bytes=').split('-'))
                requests.append(start)
                self.send_response(206)
                self.send_header('Content-Range', f'bytes {start}-{end}/{len(data)}')
                self.send_header('Content-Length', str(end - start + 1))
                self.end_headers()
                try:
                    if len(requests) == 1:
                        self.wfile.write(data[:262144])
                        self.wfile.flush()
                        release.wait(8)
                        return
                    self.wfile.write(data[start:end + 1])
                except (BrokenPipeError, ConnectionResetError):
                    pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        command = [sys.executable, '-m', 'utils.download', '--repo-id', 'test/data',
                   '--local-dir', str(self.root), '--endpoint', f'http://127.0.0.1:{server.server_port}',
                   '--download-timeout', '1', '--range-mib', '1', '--attempts', '2']
        env = dict(os.environ, NO_PROXY='127.0.0.1', no_proxy='127.0.0.1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        try:
            deadline = time.monotonic() + 8
            part = self.root / STATE_DIR / 'partials' / (hashlib.sha256(b'file.bin').hexdigest() + '.part')
            while not part.exists() or part.stat().st_size < 262144:
                if time.monotonic() > deadline:
                    self.fail('First process did not persist its partial response')
                time.sleep(.02)
            if force:
                process.kill()
            else:
                process.terminate()
            process.communicate(timeout=6)
            self.assertNotEqual(process.returncode, 0)
            retained = part.stat().st_size
            self.assertEqual(retained, 262144)
            release.set()
            result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=12)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(requests[1], retained)
            self.assertEqual((self.root / 'file.bin').read_bytes(), data)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
            release.set()
            server.shutdown()
            server.server_close()
            thread.join()
