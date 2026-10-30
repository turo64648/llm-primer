# Run: ~/.venvs/llm/bin/python exercises/ch12_forgetting.py   (CPU, under a minute)
# Catastrophic forgetting on two toy tasks, and two fixes: replay and EWC.
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(2)
DIM, CLASSES, PER_CLASS_CLUSTERS = 30, 5, 4


def make_task(seed, n_train=4000, n_test=2000):
    """Each class is a few random cluster centres in 30-D space; points are noisy copies."""
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(CLASSES * PER_CLASS_CLUSTERS, DIM, generator=g) * 1.0
    labels_of_centre = torch.arange(CLASSES).repeat_interleave(PER_CLASS_CLUSTERS)

    def sample(n):
        idx = torch.randint(len(centres), (n,), generator=g)
        x = centres[idx] + 0.6 * torch.randn(n, DIM, generator=g)
        return x, labels_of_centre[idx]

    return sample(n_train), sample(n_test)


def mlp():
    return nn.Sequential(nn.Linear(DIM, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, CLASSES))


def accuracy(model, data):
    x, y = data
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item()


def train(model, data, steps=600, replay=None, ewc=None, lr=0.05):
    """Plain SGD on one task. Optionally mix in replayed old examples, or add the EWC penalty."""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    x, y = data
    for _ in range(steps):
        i = torch.randint(len(x), (64,))
        xb, yb = x[i], y[i]
        if replay is not None:                       # half of every batch is old data
            rx, ry = replay
            j = torch.randint(len(rx), (64,))
            xb, yb = torch.cat([xb, rx[j]]), torch.cat([yb, ry[j]])
        loss = F.cross_entropy(model(xb), yb)
        if ewc is not None:                          # pull important weights back to their old values
            fisher, old, lam = ewc
            for p, f, o in zip(model.parameters(), fisher, old):
                loss = loss + lam / 2 * (f * (p - o) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()


def fisher_diagonal(model, data, n=1000):
    """How much each weight matters for task A: average squared gradient of the log-likelihood,
    with labels sampled from the model's own predictions (the 'true' Fisher)."""
    x, _ = data
    fisher = [torch.zeros_like(p) for p in model.parameters()]
    for k in range(n):
        logp = F.log_softmax(model(x[k:k + 1]), 1)
        label = torch.multinomial(logp.exp(), 1).item()
        model.zero_grad()
        logp[0, label].backward()
        for f, p in zip(fisher, model.parameters()):
            f += p.grad ** 2 / n
    return fisher


(trainA, testA), (trainB, testB) = make_task(1), make_task(2)
results = []

for method in ["naive (A then B)", "replay 200 old examples", "EWC (lambda=300)",
               "EWC (lambda=3000)", "EWC (lambda=10000)", "joint (A and B together)"]:
    accs = []
    for seed in range(3):
        torch.manual_seed(seed)
        model = mlp()
        if method.startswith("joint"):
            both = (torch.cat([trainA[0], trainB[0]]), torch.cat([trainA[1], trainB[1]]))
            train(model, both, steps=1200)
            accs.append((accuracy(model, testA), accuracy(model, testA), accuracy(model, testB)))
            continue
        train(model, trainA)
        a_before = accuracy(model, testA)
        if method.startswith("naive"):
            train(model, trainB)
        elif method.startswith("replay"):
            train(model, trainB, replay=(trainA[0][:200], trainA[1][:200]))
        else:
            fisher = fisher_diagonal(model, trainA)
            old = [p.detach().clone() for p in model.parameters()]
            lam = float(method.split("=")[1].rstrip(")"))
            train(model, trainB, ewc=(fisher, old, lam))
        accs.append((a_before, accuracy(model, testA), accuracy(model, testB)))
    mean = [sum(a[i] for a in accs) / len(accs) for i in range(3)]
    results.append((method, *mean))

print("Task A and task B: 5-way classification of different random clusters (mean of 3 seeds)")
print("chance = 0.20\n")
print(f"{'method':<26}{'A after A':>10}{'A after B':>11}{'B after B':>11}{'forgot':>8}")
for name, a0, a1, b in results:
    forgot = "-" if name.startswith("joint") else f"{a0 - a1:.2f}"
    print(f"{name:<26}{a0:>10.2f}{a1:>11.2f}{b:>11.2f}{forgot:>8}")
