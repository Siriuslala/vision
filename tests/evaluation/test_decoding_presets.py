"""Validate Bash model presets, fixed checkpoint modes, and generated CLI arguments."""
from pathlib import Path
import re
import shlex
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]


class PresetTests(unittest.TestCase):
    def select(self, model, thinking):
        command = '''set -euo pipefail
source "$1/scripts/decoding_presets.sh"
MODEL="$2"
THINKING="$3"
MAX_NEW_TOKENS=8192
select_decoding_preset
printf '%s ' "$THINKING" "$DO_SAMPLE" "$TEMPERATURE" "$TOP_P" "$TOP_K" "$MIN_P" "$REPETITION_PENALTY" "$MAX_NEW_TOKENS"
'''
        return subprocess.run(['bash', '-c', command, 'test', str(ROOT), model, thinking],
                              capture_output=True, text=True)

    def test_presets_and_aliases(self):
        cases = [('Qwen3.8-27B', 'on', 'on true 1.0 0.95 20 0.0 1.0 8192'),
                 ('Qwen/Qwen3.8-27B', 'off', 'off true 0.7 0.80 20 0.0 1.0 8192'),
                 ('models--Qwen--Qwen3.8-27B', 'default', 'on true 1.0 0.95 20 0.0 1.0 8192'),
                 ('Qwen3.5-9B', 'on', 'on true 1.0 0.95 20 0.0 1.0 8192'),
                 ('Qwen/Qwen3.5-9B', 'off', 'off true 1.0 1.0 40 0.0 1.0 8192'),
                 ('models--Qwen--Qwen3.5-4B', 'default', 'on true 1.0 0.95 20 0.0 1.0 8192'),
                 ('Qwen3-VL-8B-Instruct', 'default', 'off true 0.7 0.8 20 0.0 1.0 8192'),
                 ('Qwen3-VL-4B-Instruct', 'off', 'off true 0.7 0.8 20 0.0 1.0 8192'),
                 ('Qwen/Qwen3-VL-8B-Thinking', 'on', 'on true 1.0 0.95 20 0.0 1.0 8192'),
                 ('Qwen3-VL-4B-Thinking', 'default', 'on true 1.0 0.95 20 0.0 1.0 8192')]
        for model, thinking, expected in cases:
            with self.subTest(model=model, thinking=thinking):
                result = self.select(model, thinking)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)

    def test_invalid_or_conflicting_modes(self):
        for model, thinking in [('Qwen3-VL-8B-Instruct', 'on'),
                                ('Qwen3-VL-8B-Thinking', 'off'),
                                ('unknown', 'off'), ('Qwen3.5-9B', 'yes')]:
            result = self.select(model, thinking)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(result.stderr)

    def test_evaluation_filename_and_cli_use_selected_values(self):
        script = (ROOT / 'scripts/eva/evaluation.sh').read_text()
        # Replace machine configuration and inference with argument capture only.
        script = re.sub(r'^source .*load_env.sh"$',
                        'ROOT_DIR=' + shlex.quote(str(ROOT)) + '\nWORK_DIR=/work\nDATA_DIR=/data\nHF_HUB_CACHE=/cache',
                        script, count=1, flags=re.M)
        script = re.sub(r'^MODEL=.*$', 'MODEL="Qwen/Qwen3-VL-8B-Thinking"', script, count=1, flags=re.M)
        script = re.sub(r'^THINKING=.*$', 'THINKING="on"', script, count=1, flags=re.M)
        script = script.replace('exec uv run --frozen python -m eva.evaluation', "printf '%s\\n'")
        result = subprocess.run(['bash', '-c', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertEqual(args[args.index('--top-p') + 1], '0.95')
        self.assertEqual(args[args.index('--top-k') + 1], '20')
        self.assertIn('--do-sample', args)
        output = args[args.index('--output') + 1]
        self.assertIn('qwen3_vl_8b_thinking_babyvision-thinkingon', output)
        self.assertIn('-top_p0.95-top_k20-', output)
