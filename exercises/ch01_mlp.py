# Run: ~/.venvs/llm/bin/python exercises/ch01_mlp.py
"""Chapter 1: a neural network forward pass in plain NumPy.

Part 1: one neuron, hand-set weights, computes AND.
Part 2: no straight line separates XOR; the best linear model outputs 0.5 everywhere.
Part 3: a 2-2-1 network with ReLU and hand-set weights solves XOR.
Part 4: two linear layers without ReLU collapse into one.
Part 5: a random multi-layer perceptron (MLP) on a batch: shapes, parameter count, and the model as a file.
"""
import os
import tempfile

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)


def relu(z):
    return np.maximum(0.0, z)


def step(z):
    return (z > 0).astype(float)


# The four possible inputs, one per row: shape (4, 2).
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)

print("== Part 1: one neuron computes AND ==")
w = np.array([1.0, 1.0])  # one weight per input
b = -1.5                  # bias: the threshold
z = X @ w + b             # weighted sum for all 4 inputs at once
for x, zi, yi in zip(X, z, step(z)):
    print(f"x={x}  z={zi:+.1f}  out={yi:.0f}")

print("\n== Part 2: the best purely linear model for XOR ==")
y_xor = np.array([0, 1, 1, 0], dtype=float)
X1 = np.hstack([X, np.ones((4, 1))])  # extra column of 1s plays the role of the bias
coef, *_ = np.linalg.lstsq(X1, y_xor, rcond=None)
print("best weights and bias:", np.round(coef, 3) + 0.0)  # + 0.0 tidies -0.0
print("its outputs:          ", X1 @ coef, " wanted:", y_xor)

print("\n== Part 3: a 2-2-1 network with ReLU solves XOR ==")
W1 = np.array([[1.0, 1.0],    # hidden neuron 1: x1 + x2
               [1.0, 1.0]])   # hidden neuron 2: x1 + x2 - 1
b1 = np.array([0.0, -1.0])
W2 = np.array([[1.0, -2.0]])  # output: h1 - 2*h2
b2 = np.array([0.0])

H = relu(X @ W1.T + b1)       # (4, 2) @ (2, 2) -> (4, 2)
Y = H @ W2.T + b2             # (4, 2) @ (2, 1) -> (4, 1)
for x, h, yi in zip(X, H, Y[:, 0]):
    print(f"x={x}  hidden={h}  out={yi:.0f}")

print("\n== Part 4: without ReLU, two layers collapse into one ==")
A = np.random.randn(5, 3)     # layer 1: 3 inputs -> 5 outputs
B = np.random.randn(2, 5)     # layer 2: 5 inputs -> 2 outputs
x = np.random.randn(3)
two_layers = B @ (A @ x)
C = B @ A                     # one 2x3 matrix that does the same job
one_layer = C @ x
print("two layers:", two_layers)
print("one layer: ", one_layer)
print("same?", np.allclose(two_layers, one_layer))

print("\n== Part 5: a random MLP on a batch ==")
sizes = [8, 32, 32, 3]        # input width, two hidden layers, output width
params = []
for n_in, n_out in zip(sizes[:-1], sizes[1:]):
    W = (np.random.randn(n_out, n_in) / np.sqrt(n_in)).astype(np.float32)
    b = np.zeros(n_out, dtype=np.float32)
    params += [W, b]

x = np.random.randn(4, sizes[0]).astype(np.float32)  # a batch of 4 examples
print("input        ", x.shape)
h = x
n_layers = len(params) // 2
for i in range(n_layers):
    W, b = params[2 * i], params[2 * i + 1]
    h = h @ W.T + b
    if i < n_layers - 1:
        h = relu(h)
    print(f"after layer {i + 1}", h.shape, f"  W{i + 1} {W.shape}, b{i + 1} {b.shape}")
print("output for example 0:", h[0])

n_params = sum(p.size for p in params)
print("\nparameter count:", n_params)
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "model.npy")
    np.save(path, np.concatenate([p.ravel() for p in params]))
    print(f"saved as float32: {os.path.getsize(path)} bytes "
          f"({n_params} params x 4 bytes = {n_params * 4}, plus a small header)")
