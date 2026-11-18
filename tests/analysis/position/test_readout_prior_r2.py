"""Unit tests of the model-free helpers of experiment 3.16 round 2."""
import unittest

import numpy as np

from analysis_tools.position import readout_prior_r2 as r2


def _features(x, kind, rng):
    if kind == 'circle':
        F = np.stack([np.cos(2 * np.pi * x / 100), np.sin(2 * np.pi * x / 100)], 1)
    else:
        F = np.stack([x / 1000, (x / 1000) ** 2], 1)
    M = rng.standard_normal((F.shape[1], 64))
    return F @ M + 0.01 * rng.standard_normal((len(x), 64))


class ReadoutPriorR2Test(unittest.TestCase):
    def test_spectrum_separates_circle_and_ruler(self):
        rng = np.random.default_rng(0)
        x = np.linspace(100, 999, 300)
        sc = r2.spectrum(_features(x, 'circle', rng), x)
        sr = r2.spectrum(_features(x, 'ruler', rng), x)
        i100, i1000 = r2.SPECTRUM_PERIODS.index(100.0), r2.SPECTRUM_PERIODS.index(1000.0)
        assert sc[i100] > 0.9 and sr[i100] < 0.1
        assert sr[i1000] > 0.3 > sr[i100]


    def test_digit_decoding_needs_period_100_with_hundreds_held_out(self):
        rng = np.random.default_rng(1)
        x = np.linspace(100, 999, 400)
        assert r2.digit_decoding(_features(x, "circle", rng), x, 1, True) > 0.8
        assert r2.digit_decoding(_features(x, 'ruler', rng), x, 1, True) < 0.3


    def test_simulated_codes(self):
        x = np.arange(100, 1000, 0.5) + 0.25
        s = (np.floor(x).astype(int) // 10) % 10
        sims = r2.simulate_codes(x)
        assert r2.argmax_changes(sims['monotone']) <= 9
        assert (sims['circle'] == s).mean() > 0.99
        assert (sims['conditional'] == s).mean() > 0.99          # no bias, no noise: exact


    def test_donor_shift_recovers_donor_slope(self):
        truth = np.linspace(100, 999, 200)
        donor = truth[::-1].copy()
        n = np.round(donor).astype(int)
        logp = np.full((200, 10), -20.0)
        logp[np.arange(200), (n // 10) % 10] = 0.0
        d = r2.donor_shift(truth, donor, logp)
        assert d['argmax_donor'] == 1.0 and d['b_donor'] > 0.9


    def test_jacobian_alignment_identity_and_change_split(self):
        rng = np.random.default_rng(2)
        J = rng.standard_normal((20, 10, 50))
        a = r2.jacobian_alignment(J, J)
        assert abs(a['matched_cos'] - 1) < 1e-9 and abs(a['overlap'] - 1) < 1e-9
        Hb = rng.standard_normal((30, 8)) + 5
        d = r2.change_split(Hb, Hb + 1.0)
        assert d['vary'] < 1e-9 and d['const'] > 0


    def test_clock_plane_slope(self):
        rng = np.random.default_rng(3)
        x = np.linspace(100, 999, 300)
        P, ang = r2.clock_plane(_features(x, 'circle', rng), x)
        slope, r = r2.circular_slope(ang, x)
        assert abs(abs(slope) - 3.6) < 0.05 and r > 0.999


if __name__ == '__main__':
    unittest.main()
