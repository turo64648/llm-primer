# Run: ~/.venvs/llm/bin/python exercises/ch07_grpo.py
# Part 1: GRPO-style RL with a verifiable reward on a toy task (add two digits).
# Part 2: the DPO loss on toy preference pairs.
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Part 1: a tiny "policy" learns to add two digits from reward alone.
# Prompt: two digits a, b (0-9). Action: one answer token out of 0..18.
# Reward: 1 if the answer equals a + b, else 0. A checker, not a human.
# ---------------------------------------------------------------------------
N_ANS = 19


def encode(a, b):
    return torch.cat([F.one_hot(a, 10), F.one_hot(b, 10)], dim=-1).float()


policy = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, N_ANS))
reference = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, N_ANS))
reference.load_state_dict(policy.state_dict())  # frozen copy of the starting policy
for p in reference.parameters():
    p.requires_grad_(False)

opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
PROMPTS, GROUP, BETA_KL = 32, 8, 0.01  # prompts per step, samples per prompt, KL weight


@torch.no_grad()
def greedy_accuracy():
    a, b = torch.meshgrid(torch.arange(10), torch.arange(10), indexing="ij")
    a, b = a.flatten(), b.flatten()
    return (policy(encode(a, b)).argmax(-1) == a + b).float().mean().item()


print("== Part 1: GRPO on 'add two digits', reward = 1 if correct ==")
print(f"{'step':>4}  {'mean reward':>11}  {'groups with no signal':>21}  {'greedy acc':>10}")
for step in range(301):
    a = torch.randint(0, 10, (PROMPTS,))
    b = torch.randint(0, 10, (PROMPTS,))
    x = encode(a, b).repeat_interleave(GROUP, dim=0)        # each prompt GROUP times
    target = (a + b).repeat_interleave(GROUP)

    logits = policy(x)
    dist = torch.distributions.Categorical(logits=logits)
    answer = dist.sample()                                   # the policy "writes" an answer
    reward = (answer == target).float()                      # verifiable reward

    # Group-relative advantage: compare each sample with its siblings on the same prompt.
    r = reward.view(PROMPTS, GROUP)
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
    adv = adv.view(-1)

    # Policy gradient: push up log-prob of above-average answers, push down below-average ones.
    pg_loss = -(adv * dist.log_prob(answer)).mean()

    # KL penalty to the frozen reference keeps the policy from drifting too far.
    logp = F.log_softmax(logits, -1)
    logp_ref = F.log_softmax(reference(x), -1)
    kl = (logp.exp() * (logp - logp_ref)).sum(-1).mean()

    loss = pg_loss + BETA_KL * kl
    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 50 == 0:
        no_signal = (r.std(1) == 0).float().mean().item()
        print(f"{step:>4}  {reward.mean().item():>11.3f}  {no_signal:>21.2f}  {greedy_accuracy():>10.2f}")

# ---------------------------------------------------------------------------
# Part 2: DPO on preference pairs. One prompt, 4 possible responses.
# Hidden "true quality" scores generate preferences via Bradley-Terry.
# DPO never sees the scores, never trains a reward model.
# ---------------------------------------------------------------------------
print("\n== Part 2: DPO on toy preference pairs ==")
true_score = torch.tensor([0.0, 2.0, -1.0, 1.0])            # response 1 is best, 2 is worst
ref_logits = torch.zeros(4)                                   # reference policy: uniform
pol_logits = torch.zeros(4, requires_grad=True)
opt2 = torch.optim.Adam([pol_logits], lr=0.05)
BETA = 0.5

# Make 2,000 labelled pairs: P(i preferred over j) = sigmoid(score_i - score_j).
i = torch.randint(0, 4, (2000,))
j = (i + torch.randint(1, 4, (2000,))) % 4                    # j != i
i_wins = torch.rand(2000) < torch.sigmoid(true_score[i] - true_score[j])
chosen, rejected = torch.where(i_wins, i, j), torch.where(i_wins, j, i)

for step in range(201):
    logp = F.log_softmax(pol_logits, -1)
    logp_ref = F.log_softmax(ref_logits, -1)
    # Implied reward of a response = BETA * (log pi - log pi_ref).
    margin = BETA * ((logp[chosen] - logp_ref[chosen]) - (logp[rejected] - logp_ref[rejected]))
    loss = -F.logsigmoid(margin).mean()
    opt2.zero_grad()
    loss.backward()
    opt2.step()
    if step % 50 == 0:
        print(f"step {step:>3}  DPO loss {loss.item():.3f}  policy probs "
              f"{[round(v, 2) for v in F.softmax(pol_logits, -1).tolist()]}")

implied = BETA * (F.log_softmax(pol_logits, -1) - F.log_softmax(ref_logits, -1)).detach()
implied = implied - implied.mean()
print("true scores (centred):    ", [round(v, 2) for v in (true_score - true_score.mean()).tolist()])
print("DPO implied reward (cent.):", [round(v, 2) for v in implied.tolist()])
