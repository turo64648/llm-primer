# Run: ~/.venvs/llm/bin/python exercises/ch09_evals.py
"""Chapter 9 exercise: the statistics of evals. NumPy only, runs in a few seconds.

Part 1: pass@k. Compare the unbiased estimator (Chen et al. 2021) with the naive plug-in estimator.
Part 2: one 200-question benchmark, two models. Standard error and a paired bootstrap confidence interval.
Part 3: how often a "2-point improvement" on 200 questions is noise, and how often a real one is missed.
Part 4: how many questions you need to detect a given gap.
"""
import numpy as np
from math import comb, erf, sqrt

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- Part 1: pass@k
print("Part 1: pass@k estimators")


def pass_at_k_unbiased(n, c, k):
    """1 - C(n-c, k) / C(n, k): chance that k samples drawn without replacement from the n include a correct one."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_at_k_naive(n, c, k):
    """Plug in p_hat = c/n and pretend the k tries are independent: 1 - (1 - p_hat)^k."""
    return 1.0 - (1.0 - c / n) ** k


# 500 problems, each with its own true chance p of a sample being correct. Many are hard (p near 0).
p_true = rng.beta(0.3, 1.0, size=500)
n = 20  # samples generated per problem
print(f"  {len(p_true)} problems, n = {n} samples each, averaged over 200 repeats of the whole eval")
print(f"  {'k':>3} {'true':>7} {'unbiased':>9} {'naive':>7}")
for k in (1, 5, 10):
    truth = np.mean(1 - (1 - p_true) ** k)
    unb, nai = [], []
    for _ in range(200):
        c = rng.binomial(n, p_true)  # correct samples per problem
        unb.append(np.mean([pass_at_k_unbiased(n, ci, k) for ci in c]))
        nai.append(np.mean([pass_at_k_naive(n, ci, k) for ci in c]))
    print(f"  {k:>3} {truth:7.3f} {np.mean(unb):9.3f} {np.mean(nai):7.3f}")

# ---------------------------------------------------------------- Part 2: one benchmark, two models
print("\nPart 2: one 200-question run, model A vs model B")
Q = 200


def threshold(acc):
    """Value t with P(z + noise < t) = acc, where z + noise has standard deviation sqrt(2). Bisection."""
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if 0.5 * (1 + erf(mid / 2)) < acc:  # normal CDF with sd sqrt(2): erf(x / (sqrt(2) * sqrt(2)))
            lo = mid
        else:
            hi = mid
    return mid


def simulate_pair(acc_a, acc_b, q=Q):
    """Correct/incorrect per question for two models that share question difficulty (paired data).

    Each question gets a difficulty z. A model answers correctly if z + noise < threshold, with the
    threshold set so its average accuracy is acc_a or acc_b. Hard questions are hard for both models.
    """
    z = rng.standard_normal(q)
    ea, eb = rng.standard_normal(q), rng.standard_normal(q)
    a = (z + ea) < threshold(acc_a)
    b = (z + eb) < threshold(acc_b)
    return a.astype(float), b.astype(float)


a, b = simulate_pair(0.70, 0.72)
pa, pb = a.mean(), b.mean()
se_a = np.sqrt(pa * (1 - pa) / Q)
print(f"  observed accuracy: A = {pa:.1%}, B = {pb:.1%}, gap = {100 * (pb - pa):+.1f} points")
print(f"  binomial standard error of A alone: sqrt(p(1-p)/n) = {100 * se_a:.1f} points")
print(f"  95% interval for A alone: {100 * (pa - 1.96 * se_a):.1f} to {100 * (pa + 1.96 * se_a):.1f}")

B_RESAMPLES = 10_000
idx = rng.integers(0, Q, size=(B_RESAMPLES, Q))  # resample questions with replacement
gaps_paired = (b[idx] - a[idx]).mean(axis=1)  # same questions for both: paired
idx2 = rng.integers(0, Q, size=(B_RESAMPLES, Q))
gaps_unpaired = b[idx].mean(axis=1) - a[idx2].mean(axis=1)  # ignore pairing
lo, hi = np.percentile(gaps_paired, [2.5, 97.5])
ulo, uhi = np.percentile(gaps_unpaired, [2.5, 97.5])
print(f"  paired bootstrap 95% CI for the gap:   {100 * lo:+.1f} to {100 * hi:+.1f} points")
print(f"  unpaired bootstrap 95% CI for the gap: {100 * ulo:+.1f} to {100 * uhi:+.1f} points")
print(f"  questions where A and B disagree: {int((a != b).sum())} of {Q}")

# ---------------------------------------------------------------- Part 3: is +2 points real?
print("\nPart 3: 5,000 simulated 200-question evals")
TRIALS = 5_000
same = np.array([np.subtract(*simulate_pair(0.70, 0.70)[::-1]).mean() for _ in range(TRIALS)])
real = np.array([np.subtract(*simulate_pair(0.70, 0.72)[::-1]).mean() for _ in range(TRIALS)])
print(f"  both models truly 70%:    B looks >= 2 points better in {np.mean(same >= 0.02):.0%} of runs")
print(f"                            one of them looks >= 2 points better in {np.mean(np.abs(same) >= 0.02):.0%}")
print(f"  B truly 72% (A 70%):      B looks better at all in {np.mean(real > 0):.0%} of runs")
print(f"                            B looks worse or tied in {np.mean(real <= 0):.0%} of runs")

# ---------------------------------------------------------------- Part 4: questions needed
print("\nPart 4: questions needed so a 95% interval on the gap excludes zero")
print("  (unpaired worst case: two independent accuracies near 70%, gap SE = sqrt(2 p(1-p)/n))")
for gap in (0.10, 0.05, 0.02, 0.01):
    # want 1.96 * SE <= gap  ->  n >= 2 p(1-p) (1.96 / gap)^2
    need = 2 * 0.7 * 0.3 * (1.96 / gap) ** 2
    print(f"  gap {100 * gap:4.0f} points: about {need:8,.0f} questions")
