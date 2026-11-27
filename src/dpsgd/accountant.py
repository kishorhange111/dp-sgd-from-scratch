"""Privacy accounting for DP-SGD, two ways.

DP-SGD with Poisson sampling rate q and noise multiplier σ is T compositions of the
*sampled Gaussian mechanism* (SGM). With the clipped gradient scaled to sensitivity 1,
neighbouring datasets give the output distributions

    P = (1 − q)·N(0, σ²) + q·N(1, σ²)    and    Q = N(0, σ²)

("remove" one example; "add" swaps the roles).

**Rényi DP** (Mironov, Talwar & Zhang 2019). For integer α the SGM's Rényi divergence has
a closed form:

    exp((α − 1)·ε_α) = Σ_k C(α, k) (1 − q)^{α−k} q^k exp((k² − k) / 2σ²).

It composes by addition over steps. It is converted to (ε, δ) with the improved bound of
Balle et al. (2020), minimised over α. The result is always an upper bound, and usually a
loose one.

**Privacy loss distribution (PLD).** The privacy loss L = log P(x)/Q(x), x ~ P, has a
distribution whose T-fold convolution is the loss of T steps, and

    δ(ε) = E[(1 − e^{ε − L})₊] + P(L = ∞).

L is monotone in x, so its CDF is exact in closed form. Discretising L on a grid of
width h, rounding every value **up** gives a distribution that dominates the true one,
and this survives convolution. So the *pessimistic* PLD is a certified upper bound on
δ(ε). Rounding **down** gives a lower bound. The two bracket the exact curve, and the gap
shrinks with h. Composition is by FFT and repeated squaring. Losses above a window are
moved to +∞ (pessimistic) or to the window's edge (optimistic); losses below it are moved
up to its edge.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import gammaln, logsumexp
from scipy.stats import norm

ORDERS = list(range(2, 257))


# -- Rényi DP ---------------------------------------------------------------------------
def rdp_sgm(q, sigma, orders=ORDERS):
    """Per-step RDP ε_α of the sampled Gaussian mechanism, at integer orders α ≥ 2."""
    if q == 1.0:  # the Gaussian mechanism: ε_α = α / 2σ²
        return np.array([a / (2 * sigma**2) for a in orders])
    out = []
    for a in orders:
        k = np.arange(a + 1)
        log_binom = gammaln(a + 1) - gammaln(k + 1) - gammaln(a - k + 1)
        with np.errstate(divide="ignore"):
            log_terms = (
                log_binom + (a - k) * np.log1p(-q) + k * np.log(q) + (k * k - k) / (2 * sigma**2)
            )
        out.append(logsumexp(log_terms) / (a - 1))
    return np.array(out)


def rdp_epsilon(q, sigma, steps, delta, orders=ORDERS):
    """(ε, best α) for ``steps`` compositions, by Balle et al.'s RDP → (ε, δ) conversion."""
    a = np.array(orders, dtype=float)
    rdp = steps * rdp_sgm(q, sigma, orders)
    eps = rdp + np.log1p(-1 / a) - (math.log(delta) + np.log(a)) / (a - 1)
    i = int(np.argmin(eps))
    return float(max(eps[i], 0.0)), orders[i]


# -- privacy loss distributions -------------------------------------------------------
class PLD:
    """Probability mass ``pmf[i]`` at loss (``lo`` + i)·h, plus ``inf_mass`` at +∞."""

    def __init__(self, pmf, lo, h, inf_mass, pessimistic):
        self.pmf, self.lo, self.h, self.inf_mass = pmf, lo, h, inf_mass
        self.pessimistic = pessimistic

    @property
    def losses(self):
        return (self.lo + np.arange(len(self.pmf))) * self.h

    def delta(self, eps):
        l = self.losses
        m = l > eps
        return float(np.sum(self.pmf[m] * -np.expm1(eps - l[m])) + self.inf_mass)

    def epsilon(self, delta, hi=100.0):
        """Smallest ε with δ(ε) ≤ delta (δ(ε) is non-increasing), by bisection."""
        if self.delta(0.0) <= delta:
            return 0.0
        lo_e, hi_e = 0.0, hi
        if self.delta(hi_e) > delta:
            return math.inf
        for _ in range(60):
            mid = (lo_e + hi_e) / 2
            if self.delta(mid) > delta:
                lo_e = mid
            else:
                hi_e = mid
        return hi_e

    def compose(self, other, window):
        pmf = np.clip(fftconvolve(self.pmf, other.pmf), 0.0, None)
        inf = 1 - (1 - self.inf_mass) * (1 - other.inf_mass)
        return _truncate(pmf, self.lo + other.lo, self.h, inf, self.pessimistic, window)

    def self_compose(self, T, window=30.0):
        result, base = None, self
        while T:
            if T & 1:
                result = base if result is None else result.compose(base, window)
            T >>= 1
            if T:
                base = base.compose(base, window)
        return result


def _truncate(pmf, lo, h, inf_mass, pessimistic, window):
    """Keep losses in [−window, window]: mass below moves up to the edge; mass above goes
    to +∞ (pessimistic) or down to the edge (optimistic)."""
    w = int(math.ceil(window / h))
    i_lo, i_hi = -w - lo, w - lo  # array indices of the window's edges
    if i_lo > 0:
        pmf = pmf.copy()
        pmf[i_lo] += pmf[:i_lo].sum()
        pmf, lo = pmf[i_lo:], -w
        i_hi = w - lo
    if i_hi < len(pmf) - 1:
        above = pmf[i_hi + 1 :].sum()
        pmf = pmf[: i_hi + 1].copy()
        if pessimistic:
            inf_mass = inf_mass + above
        else:
            pmf[-1] += above
    return PLD(pmf, lo, h, inf_mass, pessimistic)


def _x_of_loss(l, q, sigma):
    """The x at which log((1 − q) + q·exp((2x − 1)/2σ²)) equals l (needs e^l > 1 − q)."""
    if q == 1.0:
        return sigma**2 * l + 0.5
    return sigma**2 * np.log((np.exp(l) - (1 - q)) / q) + 0.5


def sgm_pld(q, sigma, h=1e-4, pessimistic=True, direction="remove", tail=1e-16):
    """Discretised PLD of one step of the sampled Gaussian mechanism."""
    z = norm.isf(tail)
    if direction == "remove":  # x ~ (1 − q)N(0, σ²) + qN(1, σ²); L increasing in x

        def sf(l):  # P(L > l)
            x = _x_of_loss(l, q, sigma)
            return (1 - q) * norm.sf(x / sigma) + q * norm.sf((x - 1) / sigma)

        floor = math.log1p(-q) if q < 1 else -math.inf  # L > floor always
        top = math.log1p(-q + q * math.exp((2 * (1 + z * sigma) - 1) / (2 * sigma**2)))
        bottom = floor if q < 1 else (2 * (-z * sigma) - 1) / (2 * sigma**2)
    else:  # x ~ N(0, σ²); L = −log((1 − q) + q·exp((2x − 1)/2σ²)), decreasing in x

        def sf(l):  # P(L > l) = P(x < x(−l))
            return norm.cdf(_x_of_loss(-l, q, sigma) / sigma)

        ceiling = -math.log1p(-q) if q < 1 else math.inf  # L < ceiling always
        top = ceiling if q < 1 else -(2 * (-z * sigma) - 1) / (2 * sigma**2)
        bottom = -math.log1p(-q + q * math.exp((2 * (z * sigma) - 1) / (2 * sigma**2)))
    i0 = math.floor(bottom / h) if math.isfinite(bottom) else None
    i1 = math.ceil(top / h)
    grid = np.arange(i0, i1 + 1) * h
    s = np.ones_like(grid)
    if direction == "remove":
        ok = grid > (math.log1p(-q) if q < 1 else -math.inf)
    else:
        ok = grid < (-math.log1p(-q) if q < 1 else math.inf)
    s[ok] = sf(grid[ok])
    if direction != "remove":
        s[~ok] = 0.0
    # mass in (grid[j-1], grid[j]] = s[j-1] − s[j]; below grid[0]: 1 − s[0]; above top: s[-1]
    inner = np.clip(s[:-1] - s[1:], 0.0, None)
    below, above = max(1 - s[0], 0.0), max(s[-1], 0.0)
    if pessimistic:  # round up: each interval to its right edge
        pmf = np.concatenate([[below], inner])
        return PLD(pmf, i0, h, above, True)
    pmf = np.concatenate([inner, [above]])  # round down: each interval to its left edge
    pmf[0] += below
    return PLD(pmf, i0, h, 0.0, False)


def pld_epsilon(q, sigma, steps, delta, h=1e-4, window=30.0):
    """(pessimistic ε, optimistic ε): the exact ε lies between, for both adjacency
    directions (the worse direction is reported).

    If the pessimistic bound comes out infinite, the window was too narrow: the loss
    mass it moved to +∞ exceeds δ. The window is then doubled, up to 4 times."""
    for _ in range(5):
        out = []
        for pess in (True, False):
            eps = max(
                sgm_pld(q, sigma, h, pess, d).self_compose(steps, window).epsilon(delta)
                for d in ("remove", "add")
            )
            out.append(eps)
        if math.isfinite(out[0]):
            break
        window *= 2
    return tuple(out)


def gaussian_delta(eps, mu):
    """Exact δ(ε) of the Gaussian mechanism with sensitivity / σ = μ (Balle & Wang 2018)."""
    return norm.cdf(-eps / mu + mu / 2) - math.exp(eps) * norm.cdf(-eps / mu - mu / 2)


def gaussian_epsilon(mu, delta):
    lo, hi = 0.0, 100.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if gaussian_delta(mid, mu) > delta:
            lo = mid
        else:
            hi = mid
    return hi
