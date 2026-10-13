"""Validate sampling partitions used in norm-conditioned summaries."""
import unittest
from analysis_tools.token_lens import norm_groups


class TokenLensTests(unittest.TestCase):
    def test_norm_partition_preserves_all_tokens(self):
        high, other = norm_groups([2, 1, 9, 3, 4, 5, 6, 7, 8, 0, 10])
        self.assertEqual(high, [2, 10])
        self.assertTrue(set(high).isdisjoint(other))
        self.assertEqual(sorted(high+other), list(range(11)))


    def test_norm_partition_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            norm_groups([float('nan'), 1])

    def test_unicode_tokens_are_not_escaped(self):
        from analysis_tools.token_lens import visible_token
        self.assertEqual(visible_token(' 天空\n'), '·天空[LF]')
        self.assertNotIn('\\u', visible_token('道路'))

    def test_render_requires_explicit_available_font(self):
        from pathlib import Path
        from analysis_tools.token_lens import render_case
        with self.assertRaises(ValueError):
            render_case({'grid': [1, 1], 'layers': {}}, Path('/tmp'), '/missing/font.otf')
