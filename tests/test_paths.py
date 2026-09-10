import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from utils.project_paths import load_paths


class PathConfigTests(unittest.TestCase):
    def test_private_paths_and_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'code with spaces'
            (root / 'scripts').mkdir(parents=True)
            (root / 'pyproject.toml').touch()
            loader = Path(__file__).resolve().parents[1] / 'scripts/load_env.sh'
            (root / 'scripts/load_env.sh').write_text(loader.read_text())
            env = root / '.env'
            expected = {'ROOT_DIR': str(root), 'DATA_DIR': str(Path(tmp) / 'data with spaces'),
                        'WORK_DIR': str(Path(tmp) / 'results'), 'HF_HUB_CACHE': str(Path(tmp) / 'cache')}
            env.write_text('\n'.join(f'{k}="{v}" # comment' for k, v in expected.items()))
            self.assertEqual(load_paths(env), expected)
            command = 'set -euo pipefail; source "$1"; printf "%s\\n" "$ROOT_DIR" "$DATA_DIR" "$WORK_DIR" "$HF_HUB_CACHE"'
            run = subprocess.run(['bash', '-c', command, 'check', str(root / 'scripts/load_env.sh')],
                                 check=True, text=True, capture_output=True)
            self.assertEqual(run.stdout.splitlines(), list(expected.values()))

    def test_missing_and_empty_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / '.env'
            with self.assertRaisesRegex(ValueError, 'Missing .env'):
                load_paths(env)
            env.write_text('ROOT_DIR=""\nDATA_DIR=""\nWORK_DIR=""\n')
            with self.assertRaisesRegex(ValueError, 'ROOT_DIR'):
                load_paths(env)

    def test_standard_cache_and_relative_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / '.env'
            env.write_text(f'ROOT_DIR="{tmp}"\nDATA_DIR="{tmp}"\nWORK_DIR="{tmp}"\n')
            with patch.dict(os.environ, {'HF_HOME': str(Path(tmp) / 'hf')}, clear=True):
                self.assertEqual(load_paths(env)['HF_HUB_CACHE'], str(Path(tmp) / 'hf/hub'))
            env.write_text('ROOT_DIR="relative"\nDATA_DIR="/data"\nWORK_DIR="/results"\n')
            with self.assertRaisesRegex(ValueError, 'absolute'):
                load_paths(env)
