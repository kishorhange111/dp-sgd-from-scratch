import math
import warnings

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.stats import norm

from dpsgd.accountant import (
    gaussian_epsilon,
    pld_epsilon,
    rdp_epsilon,
    rdp_sgm,
    sgm_pld,
)
from dpsgd.audit import audit, empirical_mu, epsilon_lower_bound
from dpsgd.train import DPSGD, clip, per_example_grads

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -- accountants ------------------------------------------------------------------------


def renyi_numeric(q, sigma, a):
    """D_α(P‖Q) for P = (1−q)N(0,σ²) + qN(1,σ²), Q = N(0,σ²), by quadrature."""
    ratio = lambda x: (1 - q) + q * math.exp((2 * x - 1) / (2 * sigma**2))  # P/Q
    f = lambda x: norm.pdf(x, 0, sigma) * ratio(x) ** a
    hi = 1 + a * 1.0 + 12 * sigma  # the integrand's mass sits near x ≈ α (P/Q tilts it)
    return math.log(quad(f, -12 * sigma, hi, limit=500, points=[0.5, a])[0]) / (a - 1)


@pytest.mark.parametrize("q,sigma", [(0.01, 1.0), (0.1, 2.0), (0.5, 0.8)])
def test_rdp_closed_form_matches_quadrature(q, sigma):
    orders = [2, 3, 5, 8]
    for a, eps in zip(orders, rdp_sgm(q, sigma, orders), strict=True):
        assert eps == pytest.approx(renyi_numeric(q, sigma, a), rel=1e-6)


def test_rdp_without_subsampling_is_the_gaussian_mechanism():
    np.testing.assert_allclose(rdp_sgm(1.0, 2.0, [2, 10]), [2 / 8, 10 / 8])


@pytest.mark.parametrize("sigma,T", [(1.0, 1), (2.0, 10), (5.0, 100)])
def test_pld_brackets_the_exact_composed_gaussian(sigma, T):
    exact = gaussian_epsilon(math.sqrt(T) / sigma, 1e-5)
    upper, lower = pld_epsilon(1.0, sigma, T, 1e-5, h=1e-4)
    assert lower <= exact <= upper
    assert upper - lower <= T * 1e-4 + 1e-6  # the gap is at most T·h
    assert rdp_epsilon(1.0, sigma, T, 1e-5)[0] > upper  # RDP is looser


def test_too_narrow_window_is_widened_not_reported_as_infinite():
    """ε ≈ 10 here; a window of 2 pushes more than δ to +∞ until it is widened."""
    upper, lower = pld_epsilon(1.0, 5.0, 100, 1e-5, h=1e-3, window=2.0)
    exact = gaussian_epsilon(2.0, 1e-5)
    assert math.isfinite(upper) and lower <= exact <= upper


def test_pld_is_tighter_than_rdp_for_subsampled_gaussian():
    for q, sigma, T in [(0.01, 1.0, 200), (0.05, 1.5, 100)]:
        upper, lower = pld_epsilon(q, sigma, T, 1e-5, h=2e-5)
        assert lower <= upper < rdp_epsilon(q, sigma, T, 1e-5)[0]


def test_single_step_pld_is_a_distribution_and_delta_is_monotone():
    for direction in ("remove", "add"):
        for pess in (True, False):
            p = sgm_pld(0.02, 1.0, 1e-4, pess, direction)
            assert p.pmf.sum() + p.inf_mass == pytest.approx(1.0, abs=1e-12)
            ds = [p.delta(e) for e in np.linspace(0, 3, 13)]
            assert all(x >= y for x, y in zip(ds, ds[1:], strict=False))


def test_more_noise_or_fewer_steps_means_less_epsilon():
    e = lambda s, T: rdp_epsilon(0.01, s, T, 1e-5)[0]
    assert e(1.0, 1000) > e(1.5, 1000) > e(2.0, 1000)
    assert e(1.0, 100) < e(1.0, 1000) < e(1.0, 5000)


# -- the DP-SGD step ------------------------------------------------------------------


def small():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.Tanh(), torch.nn.Linear(7, 3))
    X, Y = torch.randn(12, 5), torch.randint(0, 3, (12,))
    return model, X, Y


def test_per_example_gradients_equal_a_loop():
    model, X, Y = small()
    loss = torch.nn.functional.cross_entropy
    params = {k: v.detach() for k, v in model.named_parameters()}
    g = per_example_grads(model, loss, params, X, Y)
    for i in range(len(X)):
        model.zero_grad()
        loss(model(X[i : i + 1]), Y[i : i + 1]).backward()
        for k, p in model.named_parameters():
            torch.testing.assert_close(g[k][i], p.grad, rtol=1e-5, atol=1e-6)


def test_clipping_bounds_each_example_and_leaves_small_ones_alone():
    model, X, Y = small()
    params = {k: v.detach() for k, v in model.named_parameters()}
    g = per_example_grads(model, torch.nn.functional.cross_entropy, params, X, Y)
    clipped, norms = clip(g, C=0.1)
    flat = torch.cat([v.flatten(1) for v in clipped.values()], 1)
    assert torch.all(flat.norm(dim=1) <= 0.1 + 1e-6)
    same, _ = clip(g, C=float(norms.max()) * 2)
    for k in g:
        torch.testing.assert_close(same[k], g[k])


def test_without_noise_or_clipping_and_q_one_it_is_full_batch_gradient_descent():
    model, X, Y = small()
    ref = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.Tanh(), torch.nn.Linear(7, 3))
    ref.load_state_dict(model.state_dict())
    loss = torch.nn.functional.cross_entropy
    opt = DPSGD(model, loss, len(X), q=1.0, sigma=0.0, C=1e9, lr=0.3)
    sgd = torch.optim.SGD(ref.parameters(), lr=0.3)
    for _ in range(3):
        opt.step(X, Y)
        sgd.zero_grad()
        loss(ref(X), Y).backward()
        sgd.step()
    for p, r in zip(model.parameters(), ref.parameters(), strict=True):
        torch.testing.assert_close(p, r, rtol=1e-5, atol=1e-6)


def test_noise_scale_and_poisson_sampling_rate():
    model = torch.nn.Linear(2000, 1, bias=False).double()
    with torch.no_grad():
        model.weight.zero_()
    X, Y = torch.zeros(100, 2000, dtype=torch.float64), torch.zeros(100, dtype=torch.float64)
    opt = DPSGD(model, lambda o, y: o.sum(), 100, q=0.3, sigma=1.5, C=2.0, lr=1.0)
    sizes = [opt.step(X, Y) for _ in range(200)]
    # 200 steps of N(0, (σC)²) noise, each divided by qN, per coordinate
    noise = model.weight.detach().flatten() * (0.3 * 100) / math.sqrt(200)
    assert float(noise.std()) == pytest.approx(1.5 * 2.0, rel=0.06)
    assert np.mean(sizes) == pytest.approx(30, rel=0.05)


# -- the audit ---------------------------------------------------------------------------


def test_lower_bound_is_zero_for_indistinguishable_scores():
    rng = np.random.default_rng(0)
    eps, _ = epsilon_lower_bound(rng.normal(size=4000), rng.normal(size=4000), 1e-5)
    assert eps < 0.2


def test_audit_accepts_the_correct_code_and_refutes_the_noise_bug():
    claimed = gaussian_epsilon(1 / 4.0, 1e-5)  # q = 1, one step, σ = 4
    good = audit(1.0, 4.0, 4.0, 1, 800)
    bad = audit(1.0, 4.0, 4.0, 1, 800, noise_bug=True)
    assert good["eps_lb"] <= claimed < bad["eps_lb"]
    mu, (lo, hi) = empirical_mu(good["s_in"], good["s_out"])
    assert lo <= 0.25 <= hi
    mu_bad, (lo_b, _) = empirical_mu(bad["s_in"], bad["s_out"])
    assert lo_b > 0.25 * 3  # the bug's noise is C = 4 times too small
