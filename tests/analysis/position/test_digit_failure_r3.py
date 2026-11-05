"""Model-free checks of experiment 3.12 round 3: digit decomposition, row plans, pair selection, slot copies."""
import unittest

import numpy as np
import torch

from analysis_tools.position import digit_failure as df
from analysis_tools.position import digit_failure_r3 as R3


class FakeLinearLayer:
    def __init__(self, rows: int):
        self.recurrent_states = {0: torch.arange(rows, dtype=torch.float32).view(rows, 1, 1).repeat(1, 2, 2)}
        self.conv_states = {0: torch.arange(rows, dtype=torch.float32).view(rows, 1, 1).repeat(1, 3, 4)}
        self.is_recurrent_states_initialized = {0: True}
        self.is_conv_states_initialized = {0: True}


class R3Tests(unittest.TestCase):
    def test_digits_of(self):
        self.assertEqual(R3.digits_of(312.5), (3, 1, 2))      # round half to even -> 312
        self.assertEqual(R3.digits_of(498.6), (4, 9, 9))
        with self.assertRaises(ValueError):
            R3.digits_of(99.2)

    def test_conv_slots(self):
        lin = {25: FakeLinearLayer(3), 3: FakeLinearLayer(3)}
        R3.copy_conv_slots(lin, 2, 1, (-1, -2), R3.LATE)
        c = lin[25].conv_states[0]
        self.assertTrue(torch.all(c[2, :, -2:] == 1))
        self.assertTrue(torch.all(c[2, :, :-2] == 2))
        self.assertTrue(torch.all(lin[3].conv_states[0][2] == 2))
        self.assertTrue(torch.all(lin[25].recurrent_states[0][2] == 2))

    def test_relay_rows(self):
        rows = R3.relay_rows()
        names = [r[0] for r in rows]
        self.assertEqual(names[0], 'base')
        self.assertEqual(len(set(names)), len(names))
        for name, src, edit, esrc in rows:
            if name.startswith('steer') and edit:
                self.assertEqual(esrc, 0)                       # restored from the base row
            if name.startswith('base+'):
                self.assertEqual(src, 0)
                self.assertIn(esrc, (1, 2))
        # every edit source has an unedited row
        for _, _, edit, esrc in rows:
            if edit:
                self.assertTrue(any(r[1] == esrc and not r[2] for r in rows))

    def test_patch3_pairs(self):
        units = [df.Unit(f'sweep_h_{i}', 'sweep_h', 0, float(v), '', '') for i, v in enumerate(np.arange(100, 1000, 8.3))]
        pairs = R3.patch3_pairs(units, per_axis=10)
        by_id = {u.id: u for u in units}
        self.assertEqual(len(pairs), 10)
        for p in pairs:
            a, b = by_id[p['A']].truth, by_id[p['B']].truth
            self.assertTrue(15 <= abs(b - a) <= 35)
            self.assertEqual((p['k'], p['t']), R3.digits_of(a)[:2])

    def test_scaled_add_requires_unsteered_row0(self):
        class RT:
            layers = [torch.nn.Identity()] * 30
        with self.assertRaises(ValueError):
            R3.ScaledAdd(RT(), 24, torch.tensor([0.1, 0.0]), torch.zeros(2, 4), 0)


if __name__ == '__main__':
    unittest.main()
