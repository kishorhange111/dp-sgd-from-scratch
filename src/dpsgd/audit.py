"""An empirical lower bound on ε, from running the real DP-SGD code.

The worst case for DP-SGD:
- one *canary* example whose gradient, after clipping, is exactly C·u, for a fixed unit
  vector u;
- every other example has gradient 0;
- the attacker sees every intermediate model (Nasr et al. 2021, 2023).

The model is linear with loss θ·x, so an example's gradient is its input. Neighbouring
datasets differ in one slot: the canary, or a zero example. Both have N examples, so
the normalisation q·N is identical.

From each step's update the attacker reads y_t = ⟨θ_t − θ_{t+1}, u⟩ · qN / lr. That is
C·b_t + noise, where b_t ~ Bernoulli(q) says whether the canary was sampled. It scores a
run by the log-likelihood ratio of the *claimed* mechanism, summed over steps. Any
threshold τ on the score is a test with false-positive rate α and false-negative rate β,
and (ε, δ)-DP forces

    ε ≥ log((1 − δ − α) / β)   and   ε ≥ log((1 − δ − β) / α).

The threshold is chosen on half the runs. α and β are then measured on the other half,
with Clopper–Pearson upper bounds at 95% confidence, so the reported ε_lb is a lower
bound that holds with 95% confidence.

A correct implementation has ε_lb ≤ the accountant's ε. An implementation whose noise is
too small exceeds it.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import beta as beta_dist

from .train import DPSGD


def _run(canary, q, sigma, C, T, N, d, seed, noise_bug):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = torch.nn.Linear(d, 1, bias=False).double()
    u = torch.zeros(d, dtype=torch.float64)
    u[0] = 1.0
    X = torch.zeros(N, d, dtype=torch.float64)
    if canary:
        X[0] = 10 * C * u  # clipped to exactly C·u
    Y = torch.zeros(N, dtype=torch.float64)
    opt = DPSGD(model, lambda out, y: out.sum(), N, q, sigma, C, lr=0.1, generator=g,
                noise_bug=noise_bug)  # fmt: skip
    ys = []
    for _ in range(T):
        before = model.weight.detach()[0].clone()
        opt.step(X, Y)
        ys.append(float((before - model.weight.detach()[0]) @ u) * q * N / 0.1)
    return np.array(ys)


def score(ys, q, sigma, C):
    """Log-likelihood ratio of 'canary present' under the claimed mechanism."""
    z = np.asarray(ys) / C  # N(b, σ²) with b ∈ {0, 1}
    return float(np.sum(np.logaddexp(np.log1p(-q), np.log(q) + (2 * z - 1) / (2 * sigma**2))))


def _cp_upper(k, n, conf):
    return 1.0 if k >= n else float(beta_dist.ppf(conf, k + 1, n - k))


def epsilon_lower_bound(s_out, s_in, delta, conf=0.975):
    """ε_lb from scores without (``s_out``) and with (``s_in``) the canary."""
    h = len(s_out) // 2
    sel_out, ev_out = s_out[:h], s_out[h:]
    sel_in, ev_in = s_in[: len(s_in) // 2], s_in[len(s_in) // 2 :]

    def eps_at(tau, out, inn, upper):
        n0, n1 = len(out), len(inn)
        fp, fn = int(np.sum(out > tau)), int(np.sum(inn <= tau))
        a = _cp_upper(fp, n0, conf) if upper else fp / n0
        b = _cp_upper(fn, n1, conf) if upper else fn / n1
        vals = []
        if b > 0 and 1 - delta - a > 0:
            vals.append(math.log((1 - delta - a) / b))
        if a > 0 and 1 - delta - b > 0:
            vals.append(math.log((1 - delta - b) / a))
        return max(vals, default=0.0)

    taus = np.unique(np.concatenate([sel_out, sel_in]))
    best = max(taus, key=lambda t: eps_at(t, sel_out, sel_in, upper=False))
    return max(eps_at(best, ev_out, ev_in, upper=True), 0.0), float(best)


def empirical_mu(s_in, s_out, boot=2000, seed=0):
    """For q = 1 the mechanism is μ-GDP with μ = √T/σ and both score distributions are
    Gaussians μ apart in standard-deviation units. Estimate μ, with a 95% bootstrap CI."""
    s_in, s_out = np.asarray(s_in), np.asarray(s_out)

    def mu(a, b):
        return (a.mean() - b.mean()) / math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)

    rng = np.random.default_rng(seed)
    bs = [mu(rng.choice(s_in, len(s_in)), rng.choice(s_out, len(s_out))) for _ in range(boot)]
    return float(mu(s_in, s_out)), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def audit(q, sigma, C, T, trials, delta=1e-5, N=20, d=4, noise_bug=False, seed=0):
    s_in = [score(_run(True, q, sigma, C, T, N, d, seed + 2 * i, noise_bug), q, sigma, C)
            for i in range(trials)]  # fmt: skip
    s_out = [score(_run(False, q, sigma, C, T, N, d, seed + 2 * i + 1, noise_bug), q, sigma, C)
             for i in range(trials)]  # fmt: skip
    eps, tau = epsilon_lower_bound(np.array(s_out), np.array(s_in), delta)
    return {"eps_lb": eps, "tau": tau, "s_in": s_in, "s_out": s_out}
