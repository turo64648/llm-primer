# Run: ~/.venvs/llm/bin/python exercises/ch13_fast_weights.py
# Fast-weight memory: store key->value pairs in one fixed-size matrix.
# Part 1: Hebbian (outer-product) writes, as in linear attention. Recall degrades as pairs approach
#         and exceed the dimension, and collapses when keys resemble each other.
# Part 2: the delta rule writes only the error: it can overwrite a key and copes with similar keys.
# Part 3: a tiny test-time-training memory: an MLP updated by gradient steps on its own "surprise".
import numpy as np
import torch

rng = np.random.default_rng(0)
torch.manual_seed(0)
D = 64  # key and value size; the memory matrix is D x D = 4096 numbers, however many pairs we store


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def make_keys(N, shared):
    """Unit keys. shared>0 mixes in one common direction, so keys look alike (as real keys often do)."""
    common = rng.standard_normal(D)
    return unit(shared * unit(common) + (1 - shared) * unit(rng.standard_normal((N, D))))


def recall_acc(M, K, V):
    """Read every key back. % of reads whose closest stored value is the right one."""
    R = unit(K @ M.T)                             # row i is M @ k_i, normalised
    return 100 * np.mean(np.argmax(R @ unit(V).T, axis=1) == np.arange(len(K)))


def write_hebbian(K, V):
    M = np.zeros((D, D))
    for k, v in zip(K, V):
        M += np.outer(v, k)                       # add v k^T: "when you see k, output v"
    return M


def write_delta(K, V, beta=1.0, passes=1):
    M = np.zeros((D, D))
    for _ in range(passes):
        for k, v in zip(K, V):
            err = v - M @ k                       # what the memory gets wrong for this key right now
            M += beta * np.outer(err, k)          # write only the correction
    return M


for shared, label in [(0.0, "random keys"), (0.5, "similar keys (half shared direction)")]:
    print(f"Recall accuracy (%) from a 64x64 memory, {label}")
    print(f"{'pairs N':>8} | {'Hebbian':>7} | {'delta, 1 pass':>13} | {'delta, 5 passes':>15}")
    for N in [16, 32, 64, 128, 256]:
        K = make_keys(N, shared)
        V = rng.standard_normal((N, D))
        h = recall_acc(write_hebbian(K, V), K, V)
        d1 = recall_acc(write_delta(K, V), K, V)
        d5 = recall_acc(write_delta(K, V, passes=5), K, V)
        print(f"{N:>8} | {h:>7.0f} | {d1:>13.0f} | {d5:>15.0f}")
    print()

print("Overwrite test: write key k with value A, then 20 other pairs, then k again with value B")
k = unit(rng.standard_normal(D))
A, B = rng.standard_normal(D), rng.standard_normal(D)
Ks = np.vstack([k, unit(rng.standard_normal((20, D))), k])
Vs = np.vstack([A, rng.standard_normal((20, D)), B])
for name, M in [("Hebbian", write_hebbian(Ks, Vs)), ("delta", write_delta(Ks, Vs))]:
    out = unit(M @ k)
    print(f"  {name:>7}: similarity of read to old A = {out @ unit(A):+.2f}, to new B = {out @ unit(B):+.2f}")

print("\nGradient check: one SGD step (lr 0.5) on ||M k - v||^2 equals one delta-rule write (beta 1)")
M0 = rng.standard_normal((D, D)) * 0.1
kk, vv = unit(rng.standard_normal(D)), rng.standard_normal(D)
Mt = torch.tensor(M0, requires_grad=True)
((Mt @ torch.tensor(kk) - torch.tensor(vv)) ** 2).sum().backward()
M_sgd = M0 - 0.5 * Mt.grad.numpy()
M_delta = M0 + np.outer(vv - M0 @ kk, kk)
print(f"  max difference: {np.abs(M_sgd - M_delta).max():.1e}")

print("\nPart 3: an MLP memory learning at test time from its own surprise (Titans-style, tiny)")
# The stream repeats 32 key->value pairs three times, then shows 8 new pairs.
# Surprise = the memory's loss ||memory(k) - v||^2 before it updates on that pair.
# Update: a few SGD steps with momentum; weight decay plays the role of slow forgetting.
torch.manual_seed(0)
Dm = 32
K_old = torch.nn.functional.normalize(torch.randn(32, Dm), dim=1)
V_old = torch.randn(32, Dm)
K_new = torch.nn.functional.normalize(torch.randn(8, Dm), dim=1)
V_new = torch.randn(8, Dm)
memory = torch.nn.Sequential(torch.nn.Linear(Dm, 256), torch.nn.SiLU(), torch.nn.Linear(256, Dm))
opt = torch.optim.SGD(memory.parameters(), lr=0.3, momentum=0.6, weight_decay=1e-3)


def see(K, V, steps=4):
    surprises = []
    for k, v in zip(K, V):
        for s in range(steps):
            loss = ((memory(k) - v) ** 2).mean()
            if s == 0:
                surprises.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
    return np.mean(surprises)


for r in range(3):
    print(f"  round {r + 1} over the 32 old pairs: mean surprise {see(K_old, V_old):.2f}")
print(f"  8 never-seen pairs:              mean surprise {see(K_new, V_new):.2f}")
with torch.no_grad():
    for name, K, V in [("old", K_old, V_old), ("new", K_new, V_new)]:
        cos = torch.nn.functional.cosine_similarity(memory(K), V, dim=1).mean().item()
        print(f"  final recall, {name} pairs: mean similarity to true value {cos:.2f}")
