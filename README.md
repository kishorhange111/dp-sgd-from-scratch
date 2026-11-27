# DP-SGD from scratch

DP-SGD (Abadi et al. 2016) is built here end to end, with nothing taken from Opacus or
`dp-accounting`:

- **Training.** Poisson sampling, per-example gradients with `torch.func.vmap(grad)`,
  per-example clipping across all parameters jointly, and Gaussian noise on the sum.
- **Two privacy accountants.** Rényi DP uses the closed form for the sampled Gaussian
  mechanism. The privacy-loss-distribution (PLD) accountant composes by FFT and returns a
  *certified upper bound and a lower bound* on ε.
- **An empirical privacy audit.** It runs the real training code thousands of times,
  with and without a worst-case canary. It lower-bounds ε with 95% confidence, and it
  catches a deliberately planted noise-scaling bug that the unit tests would not.

```
src/dpsgd/
  accountant.py   RDP of the sampled Gaussian; PLD with pessimistic/optimistic discretisation
  train.py        DP-SGD: Poisson sampling, vmap(grad) per-example grads, clipping, noise
  audit.py        canary attack, likelihood-ratio test, Clopper–Pearson ε lower bound, μ-GDP fit
  bench.py        the tables below
```

## Results

All numbers come from `python -m dpsgd.bench` (`results/bench.md`, `results/bench.jsonl`),
run on a CPU. δ = 1e-5 throughout.

### 1. Accounting: RDP overstates ε by 8–15%

| q | σ | steps | RDP ε | PLD ε, certified upper | PLD ε, lower | RDP / PLD | exact |
|---|---|---|---|---|---|---|---|
| 1.0 | 5.0 | 100 | 10.802 | 10.002 | 9.992 | 1.080 | 9.997 |
| 0.01 | 1.0 | 1,000 | 2.108 | 1.833 | 1.823 | 1.150 | — |
| 0.004 | 1.1 | 10,000 | 2.014 | 1.851 | 1.831 | 1.088 | — |
| 0.01 | 4.0 | 10,000 | 1.036 | 0.957 | 0.937 | 1.082 | — |

- **The PLD brackets the truth.** Where an exact answer exists (q = 1 is the plain
  Gaussian mechanism, μ = √T/σ), the PLD's upper and lower bounds bracket it: 9.992 ≤
  9.997 ≤ 10.002. The tests check this bracket, and check that the gap is at most T·h for
  grid width h.
- **RDP is always looser.** It can't be tight, because converting RDP to (ε, δ) loses
  information, even with Balle et al.'s improved conversion. The same privacy budget
  therefore buys 8–15% less noise under PLD accounting, at no cost in privacy.

### 2. Training: digits, MLP 64 → 64 → 10

N = 1,437, expected batch 64 (q = 0.0445), 673 steps (30 epochs), C = 1, lr = 0.5,
3 seeds.

| noise σ | test accuracy | RDP ε | PLD ε, certified | PLD ε, lower |
|---|---|---|---|---|
| none (no clipping, no noise) | 0.975 ± 0.004 | ∞ | ∞ | ∞ |
| 0.6 | 0.950 ± 0.004 | 29.97 | 25.42 | 25.41 |
| 0.8 | 0.950 ± 0.002 | 15.06 | 12.42 | 12.41 |
| 1.0 | 0.948 ± 0.001 | 8.64 | 7.74 | 7.74 |
| 1.5 | 0.937 ± 0.005 | 4.29 | 3.90 | 3.90 |
| 2.0 | 0.926 ± 0.009 | 2.86 | 2.62 | 2.61 |

- **Privacy cost.** Going from σ = 0.6 to σ = 2 lowers ε by 10× and costs 2.4 points of
  accuracy.
- **The window fix.** The σ = 0.6 row first came out as *certified ε = ∞*. The PLD
  truncates losses to a window [−20, 20], and at ε ≈ 25 more than δ of the mass lies
  above it. The pessimistic bound correctly moves that mass to +∞, so it reported ∞.
  `pld_epsilon` now doubles the window when that happens. Only this row needed it;
  every other row has the same value under both code paths.

### 3. Audit: the correct code passes, the noise bug is caught

The audit uses the worst case for DP-SGD. One canary's clipped gradient is exactly C·u,
every other example's gradient is 0, and the attacker sees every update (Nasr et al.
2021). Each of 2,000 runs with the canary and 2,000 without is scored by the
likelihood ratio of the *claimed* mechanism. The threshold is chosen on one half of the
runs. Clopper–Pearson bounds on the other half then give an ε lower bound that holds
with 95% confidence.

The planted bug, `noise_bug=True`, is a common one: noise std σ instead of σ·C. With
C = 4 that is 4× too little noise.

| setting | claimed ε | ε lower bound, correct | ε lower bound, bug | empirical μ, correct | empirical μ, bug | claimed μ |
|---|---|---|---|---|---|---|
| q = 1, σ = 4, 1 step | 0.926 | 0.000 | **1.755** | 0.30 (0.24–0.36) | 1.06 (1.00–1.12) | 0.25 |
| q = 1, σ = 8, 4 steps | 0.927 | 0.000 | 0.597 | 0.27 (0.21–0.33) | 1.03 (0.97–1.09) | 0.25 |
| q = 0.2, σ = 2, 10 steps | 1.618 | 0.000 | **2.781** | — | — | — |

- **The bug is refuted where the bound beats the claim.** In two of the three settings,
  the buggy code's ε lower bound is above the claimed ε, which is a 95%-confidence proof
  that it is not as private as claimed.
- **The μ estimate works even when the ε bound doesn't.** In the middle row the lower
  bound (0.597) does not exceed the claim. A threshold test with 2,000 runs per side has
  little power at δ = 1e-5. The μ-GDP estimate still catches it: for q = 1 the two score
  distributions are Gaussians μ apart. The correct code's CI contains the claimed 0.25,
  and the bug's μ is ≈ 1.0, which is exactly the 4× from C = 4.
- **What the audit cannot show.** The correct code's lower bound of 0 is consistent with
  the claim. It does not prove the claim, because an audit can only refute a privacy
  guarantee, never certify one. The accountant is what certifies it.

## Design notes

- **Poisson sampling, not shuffling.** Each example joins a batch independently with
  probability q, which is the sampling the accountant analyses. The noisy sum is divided
  by the *expected* batch size q·N. Dividing by the realised size would leak it.
- **Exact discretisation.** The privacy loss is monotone in the mechanism's output, so
  its CDF is exact in closed form. Rounding each grid cell's mass up gives a distribution
  that dominates the true one, and domination survives convolution. That is why the
  pessimistic PLD is a certified bound, not an approximation.
- **The RDP closed form** is checked against numerical quadrature of the Rényi divergence
  to 1e-6 relative error.

## Run it

```
pip install -e ".[dev]"
pytest -q                          # 17 tests
python -m dpsgd.bench --out results
```
