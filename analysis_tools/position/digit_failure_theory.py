"""Experiment 3.12: toy readout theory for digit-by-digit output of a continuous coordinate.

A coordinate x in [0, 1000) is available to the readout as a one-dimensional monotone code
z = x + noise. A softmax readout over C classes with logits l_c = a_c * f(z) + b_c (f a fixed feature
map) is fitted by multinomial logistic regression. The module answers three questions:

1. Which digit patterns can a linear readout of the monotone code produce? With f(z) = z each class
   wins one contiguous interval (the arg-max of affine functions in one variable is the upper
   envelope of lines), so the hundreds digit (one band per digit) is producible, the tens digit (ten
   disjoint bands per digit) is not.
2. What does conditioning on the already emitted first digit k add? Features [z, onehot(k)] make the
   tens digit producible in principle, but only if the class biases b_{d,k} cancel a_d * 100 k, a term
   whose size grows with the required gain; `cancellation_ratio` reports how large the cancelled terms
   are relative to the logit differences that decide the digit.
3. A single "location token" softmax (100 classes of width 10) on f(z) = z is again a band readout.
"""
from __future__ import annotations

import numpy as np
import torch


def fit_softmax(X: np.ndarray, y: np.ndarray, n_classes: int, l2: float = 1e-4, iters: int = 300) -> dict:
    """Multinomial logistic regression (L-BFGS, float64). X (N, F) is standardized internally."""
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.long)
    W = torch.zeros(X.shape[1], n_classes, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(n_classes, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xt @ W + b, yt) + l2 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return {'W': W.detach().numpy(), 'b': b.detach().numpy(), 'mu': mu, 'sd': sd}


def predict_logits(m: dict, X: np.ndarray) -> np.ndarray:
    return ((X - m['mu']) / m['sd']) @ m['W'] + m['b']


def features(z: np.ndarray, kind: str, k: np.ndarray | None = None) -> np.ndarray:
    """Feature maps of the toy readouts.

    'z':        the monotone code alone;
    'z+k':      code and a one-hot of the emitted first digit (additive class biases per k);
    'periodic': cos / sin of 2 pi z / 100 (a code that is already cyclic with period 100).
    """
    if kind == 'z':
        return z[:, None]
    if kind == 'z+k':
        return np.concatenate([z[:, None], np.eye(10)[k]], 1)
    if kind == 'periodic':
        return np.stack([np.cos(2 * np.pi * z / 100), np.sin(2 * np.pi * z / 100)], 1)
    raise ValueError(kind)


def cancellation_ratio(m: dict, X: np.ndarray) -> float:
    """Median over samples of (largest |a_c f(z)| term) / (gap between the top two logits).

    Measures how large the individual logit terms are relative to the margin that decides the class.
    Values far above the bfloat16 resolution limit (about 1 / 2^-8 = 256) mean the readout relies on
    cancellations a bfloat16 network cannot represent in one linear step.
    """
    Xs = (X - m['mu']) / m['sd']
    terms = np.abs(Xs[:, :, None] * m['W'][None]).max((1, 2))
    lg = Xs @ m['W'] + m['b']
    top2 = np.sort(lg, 1)[:, -2:]
    return float(np.median(terms / np.maximum(top2[:, 1] - top2[:, 0], 1e-9)))


def toy_experiment(sigma: float = 5.0, n: int = 20000, seed: int = 0) -> dict:
    """Accuracy of the toy readouts on x ~ U[100, 1000) with code noise sigma (0-1000 units)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(100, 1000, n)
    z = x + rng.normal(0, sigma, n)
    hund = (x // 100).astype(int)
    tens = ((x // 10) % 10).astype(int)
    bins = (x // 10).astype(int) - 10                                  # 90 location bins of width 10
    tr, te = slice(0, n // 2), slice(n // 2, n)
    out = {'sigma': sigma}
    cases = {'hundreds|z': ('z', hund, 10), 'tens|z': ('z', tens, 10), 'tens|z+k': ('z+k', tens, 10),
             'tens|periodic': ('periodic', tens, 10), 'location|z': ('z', bins, 90)}
    for name, (kind, y, C) in cases.items():
        X = features(z, kind, hund)
        m = fit_softmax(X[tr], y[tr], C, iters=500 if C > 10 else 300)
        pred = predict_logits(m, X[te]).argmax(1)
        diff = np.abs(pred - y[te])
        circ = np.minimum(diff, 10 - diff) if name.startswith('tens') else diff
        out[name] = {'acc': float((pred == y[te]).mean()), 'within1': float((circ <= 1).mean()),
                     'cancellation': cancellation_ratio(m, X[te])}
    return out


def envelope_demo(n_points: int = 2001) -> dict:
    """Ideal digit patterns and the best affine-in-x readouts, for the explanatory figure.

    Returns x, the ideal hundreds / tens digits, the logits of the fitted affine readouts and their
    arg-max, plus the conditioned (z+k) readout of the tens digit.
    """
    x = np.linspace(100, 999.9, n_points)
    hund = (x // 100).astype(int)
    tens = ((x // 10) % 10).astype(int)
    res = {'x': x, 'hundreds': hund, 'tens': tens}
    for name, kind, y in (('hundreds_z', 'z', hund), ('tens_z', 'z', tens), ('tens_zk', 'z+k', tens)):
        X = features(x, kind, hund)
        m = fit_softmax(X, y, 10, l2=1e-6, iters=500)
        lg = predict_logits(m, X)
        res[name] = {'logits': lg, 'argmax': lg.argmax(1), 'acc': float((lg.argmax(1) == y).mean())}
    return res
