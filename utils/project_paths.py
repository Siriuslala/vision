"""Machine-specific paths loaded from the private project .env file."""
from pathlib import Path
import os
import shlex

ENV_KEYS = ('ROOT_DIR', 'DATA_DIR', 'WORK_DIR', 'HF_HUB_CACHE', 'HF_HOME')


def load_paths(env_file=None):
    env_file = Path(env_file) if env_file else Path(__file__).resolve().parents[1] / '.env'
    if not env_file.is_file():
        raise ValueError('Missing .env. Copy .env_example to .env and fill in the paths.')
    values = {}
    for number, line in enumerate(env_file.read_text().splitlines(), 1):
        parts = shlex.split(line, comments=True)
        if not parts:
            continue
        if parts[0] == 'export':
            parts = parts[1:]
        if len(parts) != 1 or '=' not in parts[0]:
            raise ValueError(f'Invalid .env assignment on line {number}; use KEY="absolute path".')
        key, value = parts[0].split('=', 1)
        if key in ENV_KEYS:
            values[key] = value
    for key in ('ROOT_DIR', 'DATA_DIR', 'WORK_DIR'):
        if not values.get(key):
            raise ValueError(f'Fill in {key} in .env.')
        path = Path(values[key]).expanduser()
        if not path.is_absolute():
            raise ValueError(f'{key} must be an absolute path in .env.')
        values[key] = str(path)
    cache = values.get('HF_HUB_CACHE') or os.environ.get('HF_HUB_CACHE')
    if not cache:
        hf_home = values.get('HF_HOME') or os.environ.get('HF_HOME')
        cache_home = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache')))
        cache = str((Path(hf_home).expanduser() if hf_home else cache_home / 'huggingface') / 'hub')
    cache_path = Path(cache).expanduser()
    if not cache_path.is_absolute():
        raise ValueError('HF_HUB_CACHE/HF_HOME must specify an absolute path.')
    values['HF_HUB_CACHE'] = str(cache_path)
    return values
