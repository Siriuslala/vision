import unittest

import numpy as np
import torch

from analysis_tools.position import position_probe_refinement as pr


def moments(x: np.ndarray, y: np.ndarray) -> dict:
    m = pr.Moments(x.shape[1], y.shape[1], 'cpu')
    for chunk in np.array_split(np.arange(len(x)), 3):
        m.add(torch.as_tensor(x[chunk]), torch.as_tensor(y[chunk]))
    return m.state()


class ProbeRefinementTests(unittest.TestCase):
    def test_moment_ridge_matches_direct_standardized_ridge(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(300, 12)) * rng.uniform(.5, 5, 12) + rng.normal(size=12) * 10
        y = x[:, :2] @ rng.normal(size=(2, 2)) + rng.normal(size=(300, 2))
        xv = rng.normal(size=(80, 12)) * 2
        yv = rng.normal(size=(80, 2))
        path = pr.ridge_path(moments(x, y), moments(xv, yv), [0, 1], lambdas=[.1, 10.])
        mu, sd = x.mean(0), x.std(0)
        z = (x - mu) / sd
        for i, lam in enumerate([.1, 10.]):
            b = np.linalg.solve(z.T @ z + lam * np.eye(12), z.T @ (y - y.mean(0)))
            pred = (xv - mu) / sd @ b + y.mean(0)
            np.testing.assert_allclose((xv @ path['w'][i].numpy() + path['b'][i].numpy()), pred, rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(path['val_mse'][i].numpy(), ((pred - yv) ** 2).mean(0), rtol=1e-8)

    def test_shifted_float32_moments_flush_to_raw_moments(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(200, 5)) * 3 + 40
        y = rng.uniform(0, 1000, size=(200, 3))
        acc = pr.ShiftedMoments(5, 3, 'cpu')
        for chunk in np.array_split(np.arange(200), 4):
            acc.add(torch.as_tensor(x[chunk]), torch.as_tensor(y[chunk]))
            if chunk[0] == 100:
                acc.flush()
        acc.flush()
        ref = moments(x, y)
        got = acc.total.state()
        for k in ref:
            np.testing.assert_allclose(got[k].numpy(), ref[k].numpy(), rtol=2e-5, atol=1e-2)

    def test_projected_moments_equal_moments_of_projection(self):
        rng = np.random.default_rng(1)
        x, y = rng.normal(size=(50, 6)), rng.normal(size=(50, 2))
        p = rng.normal(size=(6, 3))
        a = pr.project_moments(moments(x, y), torch.as_tensor(p))
        b = moments(x @ p, y)
        for k in ('s', 'c', 'u'):
            np.testing.assert_allclose(a[k].numpy(), b[k].numpy(), rtol=1e-10, atol=1e-10)

    def test_labels_and_sampling(self):
        grid, image = pr.token_labels((4, 8), vision=False)
        np.testing.assert_array_equal(grid[[0, 3, 4]], [[0, 0], [3, 0], [0, 1]])
        np.testing.assert_allclose(image[0], [125, 250])
        mids, pids = pr.sample_tokens((32, 32), 7)
        self.assertEqual(len(set(mids.tolist())), pr.TOKENS_PER_IMAGE)
        np.testing.assert_array_equal(pids // 4, mids)
        np.testing.assert_array_equal(pr.sample_tokens((32, 32), 7)[1], pids)
        with self.assertRaises(ValueError):
            pr.sample_tokens((8, 8), 0)

    def test_stage_layout(self):
        stages = pr.stage_names(27, ['linear_attention'] * 3 + ['full_attention'])
        self.assertEqual(len(stages), 34)
        self.assertEqual(stages[30], 'decoder_01_linear_attention')
        self.assertEqual(pr.decoder_layer('decoder_04_full_attention'), 4)
        self.assertTrue(pr.is_vision('vit_00_position') and not pr.is_vision('merger'))

    def test_synthetic_specs_render_on_token_grid(self):
        specs = pr.synthetic_specs()
        self.assertEqual(len(specs), 6 * 3 * 16 + 16 * 2)
        sizes = [tuple(r['size']) for r in specs if r['kind'] == 'uniform' and r['color'] == 'white']
        self.assertEqual(len(set(sizes)), 16)
        for row in (specs[0], specs[47], specs[-1]):
            image, _ = pr.render_synthetic(row)
            self.assertEqual((image.width % 32, image.height % 32), (0, 0))


if __name__ == '__main__':
    unittest.main()
