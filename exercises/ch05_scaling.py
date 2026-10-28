# Run: ~/.venvs/llm/bin/python exercises/ch05_scaling.py
"""Chapter 5 exercise: scaling-law calculator and a small curve fit. NumPy only, runs in about a second.

Part 1: fit a power law N_opt ∝ C^a to noisy "training runs", the way Chinchilla's iso-FLOP method does.
Part 2: compute-optimal model size and tokens for a few compute budgets, from a fitted loss formula.
Part 3: what "overtraining" a small model costs in loss, and what it saves at inference.
Part 4: FLOPs, GPU-hours and dollars for a Llama 3.1 405B-sized run.
"""
import numpy as np

rng = np.random.default_rng(0)

# Loss formula L(N, D) = E + A / N^alpha + B / D^beta.
# N = parameters, D = training tokens. Constants: Besiroglu et al. 2024's refit of Chinchilla
# (arXiv 2404.10102), which implies about 20 tokens per parameter at the optimum.
E, A, B, ALPHA, BETA = 1.8172, 482.01, 2085.43, 0.3478, 0.3658


def loss(n, d):
    return E + A / n**ALPHA + B / d**BETA


def optimal_n_d(c):
    """Minimise L(N, D) subject to 6 * N * D = C. Closed form from setting dL/dN = 0."""
    g = (ALPHA * A / (BETA * B)) ** (1 / (ALPHA + BETA))
    n = g * (c / 6) ** (BETA / (ALPHA + BETA))
    return n, c / (6 * n)


def fmt(x):
    return f"{x:.2e}"


# ---- Part 1: iso-FLOP curve fit -------------------------------------------------------------
print("Part 1: fit N_opt = k * C^a from noisy iso-FLOP sweeps")
budgets = np.logspace(18, 22, 9)  # FLOPs: tiny research runs
best_n = []
for c in budgets:
    # For each budget, "train" 15 models of different sizes. Tokens are whatever the budget allows.
    n_grid = np.logspace(np.log10(optimal_n_d(c)[0]) - 1.2, np.log10(optimal_n_d(c)[0]) + 1.2, 15)
    d_grid = c / (6 * n_grid)
    observed = loss(n_grid, d_grid) + rng.normal(0, 0.01, n_grid.size)  # run-to-run noise
    # Fit a parabola to loss vs log(N) and take its minimum, as Hoffmann et al. did.
    p = np.polyfit(np.log10(n_grid), observed, 2)
    best_n.append(10 ** (-p[1] / (2 * p[0])))
slope, intercept = np.polyfit(np.log10(budgets), np.log10(best_n), 1)
print(f"  fitted exponent a = {slope:.3f}  (true value from the formula: {BETA / (ALPHA + BETA):.3f})")
print(f"  so tokens D_opt = C / (6 N_opt) grow as C^{1 - slope:.3f}: both grow about as sqrt(C)")
print()

# ---- Part 2: compute-optimal sizes --------------------------------------------------------------
print("Part 2: compute-optimal N and D for a budget")
print(f"  {'FLOPs':>9}  {'params N':>9}  {'tokens D':>9}  {'D/N':>5}  {'loss':>5}")
for c in [1e21, 1e23, 5.76e23, 1e25, 3.8e25]:
    n, d = optimal_n_d(c)
    print(f"  {fmt(c):>9}  {fmt(n):>9}  {fmt(d):>9}  {d / n:5.1f}  {loss(n, d):5.3f}")
print("  (5.76e23 is Chinchilla's own budget; it used 70B params and 1.4T tokens.)")
print()

# ---- Part 3: overtraining -------------------------------------------------------------------------
print("Part 3: overtraining an 8B model on 15T tokens (Llama 3 8B's recipe)")
n_small, d_small = 8e9, 15e12
c = 6 * n_small * d_small
n_opt, d_opt = optimal_n_d(c)
print(f"  compute: {fmt(c)} FLOPs, tokens per param = {d_small / n_small:.0f}")
print(f"  8B on 15T:              loss {loss(n_small, d_small):.3f}")
print(f"  compute-optimal {n_opt / 1e9:.0f}B on {d_opt / 1e12:.1f}T:  loss {loss(n_opt, d_opt):.3f}")
print(f"  inference cost per token is about 2N FLOPs, so the optimal model costs "
      f"{n_opt / n_small:.1f}x more to serve")
print("  extra loss from choosing 8B: "
      f"{loss(n_small, d_small) - loss(n_opt, d_opt):.3f} nats per token (small, but not zero)")
print()

# ---- Part 4: cost of a 405B run ---------------------------------------------------------------------
print("Part 4: cost of Llama 3.1 405B pretraining from 6ND")
n, d = 405e9, 15.6e12
flops = 6 * n * d
peak = 989e12  # H100 SXM dense bf16 peak, FLOP/s
mfu = 0.40  # model FLOPs utilisation: share of peak the run achieves
gpu_seconds = flops / (peak * mfu)
gpu_hours = gpu_seconds / 3600
print(f"  FLOPs = 6 * {n:.3g} * {d:.3g} = {fmt(flops)}")
print(f"  at {mfu:.0%} MFU: {gpu_hours / 1e6:.1f}M GPU-hours "
      f"= {gpu_hours / 16384 / 24:.0f} days on 16,384 GPUs")
for price in [2.0, 4.0]:
    print(f"  at ${price:.2f}/GPU-hour: ${gpu_hours * price / 1e6:.0f}M")
print("  Meta reports 30.84M GPU-hours for the 405B model (it does not break down the gap).")
