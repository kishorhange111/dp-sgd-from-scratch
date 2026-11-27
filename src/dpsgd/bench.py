"""DP-SGD end to end: accuracy against ε on a real dataset, both accountants, and audits.

    python -m dpsgd.bench --out results

1. **Accounting.** ε at δ = 1e-5 by RDP and by PLD (pessimistic and optimistic bounds) for
   standard DP-SGD settings, plus the exact Gaussian mechanism as a check.
2. **Training.** An MLP on scikit-learn's digits (8 × 8 images, 10 classes): non-private
   SGD, then DP-SGD at several noise multipliers. Each run reports test accuracy and the
   ε of that same run under each accountant.
3. **Audit.** Lower bounds on ε for the correct implementation and for one with
   too little noise.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from .accountant import gaussian_epsilon, pld_epsilon, rdp_epsilon
from .audit import audit, empirical_mu
from .train import DPSGD

DELTA = 1e-5


def mlp():
    return torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 10))


def train(X, Y, Xt, Yt, q, sigma, C, lr, steps, seed):
    torch.manual_seed(seed)
    model = mlp()
    loss = torch.nn.functional.cross_entropy
    opt = DPSGD(model, loss, len(X), q, sigma, C, lr, torch.Generator().manual_seed(seed))
    for _ in range(steps):
        opt.step(X, Y)
    with torch.no_grad():
        return float((model(Xt).argmax(1) == Yt).float().mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--trials", type=int, default=2000)
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    rows, md = [], []

    # 1. accounting
    md += ["## Accounting: ε at δ = 1e-5", "",
           "| q | σ | steps | RDP ε | PLD ε (certified upper) | PLD ε (lower) | RDP / PLD | exact |",
           "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for q, s, T, h in [(1.0, 5.0, 100, 1e-4), (0.01, 1.0, 1000, 1e-5),
                       (0.004, 1.1, 10000, 2e-6), (0.01, 4.0, 10000, 2e-6)]:  # fmt: skip
        t0 = time.time()
        pess, opt_ = pld_epsilon(q, s, T, DELTA, h=h, window=12.0 if T > 1000 else 30.0)
        r = rdp_epsilon(q, s, T, DELTA)[0]
        exact = gaussian_epsilon(math.sqrt(T) / s, DELTA) if q == 1.0 else None
        rows.append({"bench": "accounting", "q": q, "sigma": s, "steps": T, "h": h, "rdp": r,
                     "pld_upper": pess, "pld_lower": opt_, "exact": exact,
                     "seconds": time.time() - t0})  # fmt: skip
        ex = f"{exact:.4f}" if exact is not None else "—"
        md.append(
            f"| {q} | {s} | {T} | {r:.4f} | {pess:.4f} | {opt_:.4f} | {r / pess:.3f} | {ex} |"
        )
        print(md[-1], flush=True)

    # 2. training on digits
    X, Y = load_digits(return_X_y=True)
    X = (X / 16.0).astype(np.float32)
    X, Xt, Y, Yt = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)
    X, Xt = torch.tensor(X), torch.tensor(Xt)
    Y, Yt = torch.tensor(Y), torch.tensor(Yt)
    N, batch, epochs, C, lr = len(X), 64, 30, 1.0, 0.5
    q = batch / N
    steps = int(epochs / q)
    md += ["", f"## DP-SGD on digits (MLP 64 → 64 → 10; N = {N}, expected batch {batch}, "
           f"q = {q:.4f}, {steps} steps = {epochs} epochs, C = {C}, lr = {lr}, "
           f"{a.seeds} seeds)", "",
           "| noise σ | test accuracy (mean ± sd) | RDP ε | PLD ε (certified) | PLD ε (lower) |",
           "|---|---|---|---|---|"]  # fmt: skip
    for sigma in (0.0, 0.6, 0.8, 1.0, 1.5, 2.0):
        accs = [train(X, Y, Xt, Yt, q, sigma, C if sigma else 1e9, lr, steps, s)
                for s in range(a.seeds)]  # fmt: skip
        if sigma:
            r = rdp_epsilon(q, sigma, steps, DELTA)[0]
            pess, opt_ = pld_epsilon(q, sigma, steps, DELTA, h=1e-5, window=20.0)
        else:
            r = pess = opt_ = math.inf
        rows.append({"bench": "digits", "sigma": sigma, "accs": accs, "rdp": r,
                     "pld_upper": pess, "pld_lower": opt_, "steps": steps, "q": q})  # fmt: skip
        e = lambda v: "∞" if math.isinf(v) else f"{v:.3f}"
        md.append(f"| {sigma or 'none (non-private, no clipping)'} | {np.mean(accs):.3f} ± "
                  f"{np.std(accs):.3f} | {e(r)} | {e(pess)} | {e(opt_)} |")  # fmt: skip
        print(md[-1], flush=True)

    # 3. audits
    md += ["", f"## Audit ({a.trials} runs with and {a.trials} without the canary, C = 4)", "",
           "| setting | claimed ε (PLD) | ε lower bound, correct | ε lower bound, noise bug "
           "| empirical μ, correct | empirical μ, noise bug | claimed μ |",
           "|---|---|---|---|---|---|---|"]  # fmt: skip
    for q, s, T in [(1.0, 4.0, 1), (1.0, 8.0, 4), (0.2, 2.0, 10)]:
        claimed = pld_epsilon(q, s, T, DELTA, h=1e-4)[0]
        good = audit(q, s, 4.0, T, a.trials)
        bad = audit(q, s, 4.0, T, a.trials, noise_bug=True)
        row = {"bench": "audit", "q": q, "sigma": s, "steps": T, "claimed": claimed,
               "eps_lb": good["eps_lb"], "eps_lb_bug": bad["eps_lb"]}  # fmt: skip
        mu_cells = ["—", "—", "—"]
        if q == 1.0:
            mg, cg = empirical_mu(good["s_in"], good["s_out"])
            mb, cb = empirical_mu(bad["s_in"], bad["s_out"])
            row |= {
                "mu": mg,
                "mu_ci": cg,
                "mu_bug": mb,
                "mu_bug_ci": cb,
                "mu_claimed": math.sqrt(T) / s,
            }
            mu_cells = [f"{mg:.2f} ({cg[0]:.2f}–{cg[1]:.2f})", f"{mb:.2f} ({cb[0]:.2f}–{cb[1]:.2f})",
                        f"{math.sqrt(T) / s:.2f}"]  # fmt: skip
        rows.append(row)
        md.append(
            f"| q = {q}, σ = {s}, {T} step{'s' * (T > 1)} | {claimed:.3f} | "
            f"{good['eps_lb']:.3f} | **{bad['eps_lb']:.3f}** | " + " | ".join(mu_cells) + " |"
        )
        print(md[-1], flush=True)

    with open(os.path.join(a.out, "bench.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(a.out, "bench.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
