# Run: ~/.venvs/llm/bin/python exercises/ch02_autograd.py
"""Chapter 2: a tiny automatic-differentiation (autograd) engine, built from scratch.

Every number is a Value that remembers which Values made it and how. backward() walks that
graph from the loss back to the inputs, applying the chain rule at each node.

Part 1: the chain-rule example from Chapter 0, y = (3x + 1)^2, by hand and by the engine.
Part 2: a 2-4-1 network on XOR. Check every gradient against finite differences.
Part 3: train it with plain gradient descent and watch the loss fall.
"""
import math
import random

random.seed(0)


class Value:
    """One scalar in the computation graph."""

    def __init__(self, data, parents=(), local_grads=()):
        self.data = data                # the number computed in the forward pass
        self.grad = 0.0                 # d(loss)/d(this), filled in by backward()
        self.parents = parents          # the Values this one was computed from
        self.local_grads = local_grads  # d(this)/d(parent), one per parent, saved now

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, k):  # k is a plain number
        return Value(self.data ** k, (self,), (k * self.data ** (k - 1),))

    def tanh(self):
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t * t,))

    def exp(self):
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    __radd__ = __add__
    __rmul__ = __mul__
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return (-self) + other
    def __truediv__(self, other): return self * other ** -1
    def __rtruediv__(self, other): return Value(other) * self ** -1

    def backward(self):
        # Order the graph so each node comes after everything it depends on (a topological sort).
        order, seen = [], set()
        def visit(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v.parents:
                    visit(p)
                order.append(v)
        visit(self)
        # Walk it backwards: each node passes its gradient to its parents (the chain rule).
        self.grad = 1.0
        for v in reversed(order):
            for p, local in zip(v.parents, v.local_grads):
                p.grad += v.grad * local  # += because a Value can feed several later nodes
        return len(order)


print("== Part 1: the chain rule, by hand and by the engine ==")
x = Value(2.0)
y = (3 * x + 1) ** 2
y.backward()
print(f"y = {y.data},  dy/dx by hand = 2*(3*2+1)*3 = 42,  by the engine = {x.grad}")

# A 2-4-1 network: 4 tanh hidden neurons, one output score (a logit).
H = 4
W1 = [[Value(random.gauss(0, 1)) for _ in range(2)] for _ in range(H)]
b1 = [Value(0.0) for _ in range(H)]
W2 = [Value(random.gauss(0, 1)) for _ in range(H)]
b2 = Value(0.0)
params = [w for row in W1 for w in row] + b1 + W2 + [b2]

data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]  # XOR


def forward(x):
    h = [sum((w * xi for w, xi in zip(W1[j], x)), b1[j]).tanh() for j in range(H)]
    z = sum((w * hj for w, hj in zip(W2, h)), b2)
    return 1 / (1 + (-z).exp())  # sigmoid: probability that the answer is 1


def loss_fn():
    # Binary cross-entropy: -log(probability given to the right answer), averaged.
    total = 0
    for x, t in data:
        p = forward(x)
        total = total + (-(p.log()) if t == 1 else -((1 - p).log()))
    return total / len(data)


print(f"\n== Part 2: gradient check on {len(params)} parameters ==")
loss = loss_fn()
n_nodes = loss.backward()
print(f"loss = {loss.data:.4f}; the graph for one loss has {n_nodes} nodes, all kept for backward")
eps, worst = 1e-6, 0.0
for p in params:
    old = p.data
    p.data = old + eps; up = loss_fn().data
    p.data = old - eps; down = loss_fn().data
    p.data = old
    numeric = (up - down) / (2 * eps)  # finite difference: nudge one weight, watch the loss
    worst = max(worst, abs(numeric - p.grad))
print("first 3 grads (engine):  ", [round(p.grad, 6) for p in params[:3]])
print(f"largest |engine - finite difference| over all params: {worst:.1e}")

print("\n== Part 3: train with gradient descent (learning rate 0.5) ==")
lr = 0.5
for step in range(501):
    for p in params:
        p.grad = 0.0              # gradients add up, so clear them every step
    loss = loss_fn()
    loss.backward()
    for p in params:
        p.data -= lr * p.grad     # step against the gradient
    if step % 100 == 0:
        print(f"step {step:3d}  loss {loss.data:.4f}")
for x, t in data:
    print(f"x={x}  target={t}  predicted P(1)={forward(x).data:.3f}")
