"""DP-SGD (Abadi et al. 2016) with per-example gradients from ``torch.func``.

One step:
1. **Poisson sampling.** Each example joins the batch independently with probability q.
   This is the sampling the accountant analyses; a fixed-size shuffled batch is not.
2. **Per-example gradients**, with ``vmap(grad)``: one forward and backward for the
   whole batch.
3. **Clipping.** Each example's gradient is scaled to norm at most C, across all
   parameters jointly, so one example moves the sum by at most C.
4. **Noise.** Gaussian noise of standard deviation σ·C is added to the *sum*, which is then
   divided by the *expected* batch size q·N. Dividing by the realised batch size would
   leak the batch size.

``noise_bug`` reproduces a common mistake: the noise's standard deviation is σ instead of
σ·C. The audit in :mod:`dpsgd.audit` catches it.
"""

from __future__ import annotations

import torch
from torch.func import functional_call, grad, vmap


def per_example_grads(model, loss_fn, params, x, y):
    def one(p, xi, yi):
        out = functional_call(model, p, (xi.unsqueeze(0),))
        return loss_fn(out, yi.unsqueeze(0))

    return vmap(grad(one), in_dims=(None, 0, 0))(params, x, y)


def clip(grads, C):
    """Scale each example's gradient (all parameters jointly) to norm ≤ C."""
    flat = torch.cat([g.flatten(1) for g in grads.values()], dim=1)
    norms = flat.norm(dim=1)
    scale = (C / (norms + 1e-12)).clamp(max=1.0)
    return {k: g * scale.view(-1, *[1] * (g.dim() - 1)) for k, g in grads.items()}, norms


class DPSGD:
    def __init__(self, model, loss_fn, n, q, sigma, C, lr, generator=None, noise_bug=False):
        self.model, self.loss_fn = model, loss_fn
        self.n, self.q, self.sigma, self.C, self.lr = n, q, sigma, C, lr
        self.g = generator or torch.Generator().manual_seed(0)
        self.noise_bug = noise_bug
        self.steps = 0

    def sample(self):
        return torch.nonzero(torch.rand(self.n, generator=self.g) < self.q).flatten()

    def step(self, X, Y):
        params = {k: v.detach() for k, v in self.model.named_parameters()}
        idx = self.sample()
        if len(idx):
            grads, _ = clip(
                per_example_grads(self.model, self.loss_fn, params, X[idx], Y[idx]), self.C
            )
            total = {k: g.sum(0) for k, g in grads.items()}
        else:
            total = {k: torch.zeros_like(v) for k, v in params.items()}
        std = self.sigma if self.noise_bug else self.sigma * self.C
        with torch.no_grad():
            for k, p in self.model.named_parameters():
                noise = torch.randn(p.shape, generator=self.g, dtype=p.dtype) * std
                p -= self.lr * (total[k] + noise) / (self.q * self.n)
        self.steps += 1
        return len(idx)
