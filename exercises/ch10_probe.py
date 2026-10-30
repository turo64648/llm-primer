# Run: ~/.venvs/llm/bin/python exercises/ch10_probe.py
"""Chapter 10 exercise: linear probes and superposition. CPU PyTorch, runs in well under a minute.

Part 1: train a small network on a task that needs a hidden variable, then fit linear probes on its
        hidden activations to see whether the variable can be read out. Compare against two controls:
        the raw input, and the same network before training.
Part 2: the toy model of superposition (Elhage et al. 2022). Squeeze 20 features into 5 dimensions and
        count how many features the model manages to store, for dense and for sparse features.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

# ---------------------------------------------------------------- Part 1: probing
# Input: 6 bits, written as -1/+1, plus a little noise.
# Label: 1 if (x0 XOR x1) OR (x2 XOR x3). Bits x4, x5 are distractors the task ignores.
# Hidden variables we will probe for:
#   h_needed = x0 XOR x1   (the task needs it; no straight line through the input separates it)
#   h_unused = x4 XOR x5   (the task does not need it)


def make_data(n):
    bits = torch.randint(0, 2, (n, 6))
    x = (2 * bits - 1).float() + 0.2 * torch.randn(n, 6)
    xor01 = bits[:, 0] ^ bits[:, 1]
    xor23 = bits[:, 2] ^ bits[:, 3]
    xor45 = bits[:, 4] ^ bits[:, 5]
    y = (xor01 | xor23).float()
    return x, y, xor01.float(), xor45.float()


def make_net():
    return nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 1))


def fit_probe(feats, target, feats_test, target_test, steps=500):
    """Logistic regression: one weight per feature plus a bias. Returns test accuracy."""
    mu, sd = feats.mean(0), feats.std(0) + 1e-6
    f, ft = (feats - mu) / sd, (feats_test - mu) / sd
    probe = nn.Linear(f.shape[1], 1)
    opt = torch.optim.Adam(probe.parameters(), lr=0.05)
    for _ in range(steps):
        loss = nn.functional.binary_cross_entropy_with_logits(probe(f).squeeze(1), target)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return ((probe(ft).squeeze(1) > 0).float() == target_test).float().mean().item()


x_tr, y_tr, need_tr, unused_tr = make_data(4000)
x_te, y_te, need_te, unused_te = make_data(2000)

net = make_net()
untrained = make_net()
untrained.load_state_dict(net.state_dict())  # same starting weights, never trained

opt = torch.optim.Adam(net.parameters(), lr=0.01)
for step in range(1500):
    loss = nn.functional.binary_cross_entropy_with_logits(net(x_tr).squeeze(1), y_tr)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    task_acc = ((net(x_te).squeeze(1) > 0).float() == y_te).float().mean().item()
print("Part 1: linear probes")
print(f"  task accuracy of the trained network: {task_acc:.3f}")


def hidden(model, x):
    with torch.no_grad():
        return model[1](model[0](x))  # activations after the ReLU


print(f"  {'probe reads from':<28}{'x0 XOR x1 (needed)':>20}{'x4 XOR x5 (unused)':>20}")
for name, ftr, fte in [
    ("raw input", x_tr, x_te),
    ("hidden layer, untrained", hidden(untrained, x_tr), hidden(untrained, x_te)),
    ("hidden layer, trained", hidden(net, x_tr), hidden(net, x_te)),
]:
    a_need = fit_probe(ftr, need_tr, fte, need_te)
    a_unused = fit_probe(ftr, unused_tr, fte, unused_te)
    print(f"  {name:<28}{a_need:>20.3f}{a_unused:>20.3f}")

# ---------------------------------------------------------------- Part 2: superposition
# n features, m < n dimensions. Each feature is 0 with probability `sparsity`, else uniform in [0, 1].
# Model: h = W x (compress to m numbers), x_hat = ReLU(Wᵀ h + b) (try to rebuild all n features).
# Feature i matters less as i grows: importance_i = 0.9 ** i.
n, m = 20, 5
importance = 0.9 ** torch.arange(n).float()


def train_toy(sparsity, steps=4000):
    W = nn.Parameter(0.1 * torch.randn(m, n))
    b = nn.Parameter(torch.zeros(n))
    opt = torch.optim.Adam([W, b], lr=0.01)
    for _ in range(steps):
        x = torch.rand(1024, n) * (torch.rand(1024, n) > sparsity).float()
        x_hat = torch.relu(x @ W.T @ W + b)
        loss = (importance * (x - x_hat) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return W.detach()


print("\nPart 2: toy model of superposition (20 features, 5 dimensions)")
print(f"  {'sparsity':>9}{'features stored':>17}{'max overlap':>13}   norms of the 20 feature directions")
for sparsity in [0.0, 0.7, 0.9, 0.97]:
    W = train_toy(sparsity)
    norms = W.norm(dim=0)                       # length of each feature's direction
    stored = (norms > 0.5)
    unit = W / norms.clamp(min=1e-6)
    overlap = (unit.T @ unit).abs().fill_diagonal_(0)
    max_ov = overlap[stored][:, stored].max().item() if stored.sum() > 1 else 0.0
    bar = "".join("#" if v > 0.5 else "." for v in norms)
    print(f"  {sparsity:>9.2f}{int(stored.sum()):>17}{max_ov:>13.2f}   {bar}")
