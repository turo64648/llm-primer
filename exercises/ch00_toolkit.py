# Run: ~/.venvs/llm/bin/python exercises/ch00_toolkit.py
import time

import numpy as np

rng = np.random.default_rng(0)

# 1. Dot product as similarity: same direction -> large, unrelated -> near 0.
a = np.array([1.0, 2.0, 3.0])
print("1. dot(a, a)  =", a @ a)
print("   dot(a, -a) =", a @ -a)
u, v = rng.standard_normal(512), rng.standard_normal(512)
cos = (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
print(f"   cosine of two random 512-d vectors = {cos:.3f}")

# 2. Matrix multiply is many dot products. Check one entry, then count FLOPs.
m, k, n = 256, 512, 128
A = rng.standard_normal((m, k))
B = rng.standard_normal((k, n))
C = A @ B
print("2. C[3, 7] == row 3 of A . column 7 of B:", np.isclose(C[3, 7], A[3, :] @ B[:, 7]))
flops = 2 * m * n * k
reps = 50
t0 = time.perf_counter()
for _ in range(reps):
    A @ B
dt = (time.perf_counter() - t0) / reps
print(f"   2*m*n*k = {flops:,} FLOPs; ~{flops / dt / 1e9:.0f} GFLOP/s on this CPU")

# 3. Derivative as sensitivity, and the chain rule, checked numerically.
#    y = f(g(x)) with g(x) = 3x + 1 and f(u) = u**2, so dy/dx = 2*g(x) * 3.
x, h = 2.0, 1e-6
y = lambda x: (3 * x + 1) ** 2
numeric = (y(x + h) - y(x - h)) / (2 * h)
print(f"3. dy/dx numeric = {numeric:.4f}, chain rule = {2 * (3 * x + 1) * 3:.4f}")

# 4. Softmax: the naive version overflows; subtracting the max fixes it.
def softmax_naive(z):
    e = np.exp(z)
    return e / e.sum()

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

z = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="ignore", invalid="ignore"):
    print("4. naive  :", softmax_naive(z))
print("   stable :", softmax(z).round(4))
print("   same as softmax([0, 1, 2]):", softmax(np.array([0.0, 1.0, 2.0])).round(4))

# 5. Cross-entropy = surprise = -log(probability given to the right answer).
for p in [0.9, 0.5, 0.01]:
    print(f"5. p(correct) = {p:<4} -> loss = {-np.log(p):.3f} nats")
uniform = -np.log(1 / 50_000)
print(f"   uniform guess over 50,000 tokens -> loss = {uniform:.2f}")

# 6. Shapes and broadcasting: a batch of 4 sequences, 3 tokens each, 5 numbers per token.
X = rng.standard_normal((4, 3, 5))
bias = np.arange(5.0)                     # shape (5,) is stretched over (4, 3, 5)
W = rng.standard_normal((5, 2))
print("6. (X + bias).shape =", (X + bias).shape)
print("   (X @ W).shape     =", (X @ W).shape)
row_max = X.max(axis=-1, keepdims=True)   # (4, 3, 1): one max per token
print("   max keepdims      =", row_max.shape, "-> (X - max).shape =", (X - row_max).shape)
try:
    X + np.ones(3)
except ValueError as e:
    print("   X + ones(3) fails:", str(e).split(":")[0])
