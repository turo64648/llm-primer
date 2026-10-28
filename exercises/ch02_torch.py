# Run: ~/.venvs/llm/bin/python exercises/ch02_torch.py
"""Chapter 2: the same ideas in PyTorch, which does the autograd for you.

Part 1: XOR again. PyTorch builds the graph and runs backward(); we only write the forward pass.
Part 2: a harder toy problem (two interleaved spirals) with minibatches.
        Compare plain SGD, SGD with momentum, and Adam on the same budget of steps.
Part 3: overfitting. A big network, little noisy data: training loss falls while validation
        loss rises. Compare weight decay (AdamW) and more data.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(2)

print("== Part 1: XOR in PyTorch ==")
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([0., 1., 1., 0.])
net = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
opt = torch.optim.SGD(net.parameters(), lr=0.5)
for step in range(501):
    logit = net(X).squeeze(1)
    loss = F.binary_cross_entropy_with_logits(logit, y)  # sigmoid + cross-entropy, fused
    opt.zero_grad()   # clear old gradients
    loss.backward()   # reverse-mode autodiff through the graph
    opt.step()        # w -= lr * grad
    if step % 100 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}")
print("P(1):", torch.sigmoid(net(X).squeeze(1)).detach().numpy().round(3))


def spirals(n, noise, gen):
    """Two interleaved spirals: label 0 and label 1. Not separable by any straight line."""
    t = torch.rand(n, generator=gen) * 3 * math.pi + 0.5
    label = torch.randint(0, 2, (n,), generator=gen)
    angle = t + label * math.pi
    pts = torch.stack([t * torch.cos(angle), t * torch.sin(angle)], 1) / 5
    return pts + noise * torch.randn(n, 2, generator=gen), label


def mlp(width):
    return nn.Sequential(nn.Linear(2, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                         nn.Linear(width, 2))  # 2 logits: one per class


def evaluate(net, X, y):
    with torch.no_grad():  # no graph needed: nothing will call backward
        logits = net(X)
        return F.cross_entropy(logits, y).item(), (logits.argmax(1) == y).float().mean().item()


print("\n== Part 2: SGD vs momentum vs Adam on two spirals (batch 32, 1500 steps) ==")
gen = torch.Generator().manual_seed(1)
Xtr, ytr = spirals(1000, 0.02, gen)
Xva, yva = spirals(1000, 0.02, gen)
configs = [("SGD        lr=0.05", lambda p: torch.optim.SGD(p, lr=0.05)),
           ("SGD+moment lr=0.05", lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9)),
           ("Adam       lr=0.01", lambda p: torch.optim.Adam(p, lr=0.01))]
print(f"{'optimizer':20s}{'loss@100':>10s}{'@500':>8s}{'@1500':>8s}{'val acc':>9s}")
for name, make_opt in configs:
    torch.manual_seed(0)  # same starting weights and same batches for every optimizer
    net = mlp(64)
    opt = make_opt(net.parameters())
    record = []
    for step in range(1, 1501):
        idx = torch.randint(0, len(Xtr), (32,))  # a random minibatch
        loss = F.cross_entropy(net(Xtr[idx]), ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (100, 500, 1500):
            record.append(evaluate(net, Xtr, ytr)[0])  # loss on the full training set
    acc = evaluate(net, Xva, yva)[1]
    print(f"{name:20s}{record[0]:10.3f}{record[1]:8.3f}{record[2]:8.3f}{acc:9.1%}")
n_params = sum(p.numel() for p in net.parameters())
state = sum(t.numel() for s in opt.state.values() for t in s.values() if t.dim() > 0)
print(f"parameters: {n_params}; Adam state tensors hold {state} numbers ({state / n_params:.0f} per parameter)")

print("\n== Part 3: overfitting; weight decay; more data ==")
gen = torch.Generator().manual_seed(2)
Xva, yva = spirals(1000, 0.15, gen)        # a large held-out set: never trained on
Xbig, ybig = spirals(1000, 0.15, gen)      # a large training set
Xsmall, ysmall = Xbig[:60], ybig[:60]      # only 60 noisy training points
runs = [("60 points, no weight decay  ", Xsmall, ysmall, 0.0),
        ("60 points, weight decay 1.0 ", Xsmall, ysmall, 1.0),
        ("1000 points, no weight decay", Xbig, ybig, 0.0)]
for name, Xtr, ytr, wd in runs:
    torch.manual_seed(0)
    net = mlp(256)
    opt = torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=wd)
    print(name)
    for step in range(1, 3001):
        loss = F.cross_entropy(net(Xtr), ytr)  # full batch: every training point, every step
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (100, 1000, 3000):
            va_loss, va_acc = evaluate(net, Xva, yva)
            tr_loss, tr_acc = evaluate(net, Xtr, ytr)
            print(f"  step {step:4d}  train loss {tr_loss:.3f} acc {tr_acc:6.1%}   "
                  f"val loss {va_loss:.3f} acc {va_acc:6.1%}")
