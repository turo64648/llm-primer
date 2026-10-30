# Run: ~/.venvs/llm/bin/python exercises/ch04_gpt.py
"""Chapter 4: a tiny GPT. A decoder-only transformer with causal self-attention written out by hand
(no nn.MultiheadAttention), rotary position embeddings (RoPE), pre-norm RMSNorm blocks and a GELU MLP.
Trained on the same Shakespeare BPE tokens as the bigram model in ch03_bigram.py, on CPU, in a few
minutes. Prints the parameter count, falling train/val loss, an attention sanity check and a sample.
"""
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ch03_bpe import get_tokenizer, load_text  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(2)

# Model size: 4 layers, width d = 128, 4 heads of 32 dims each, context of 128 tokens.
N_LAYER, D, N_HEAD, CTX = 4, 128, 4, 128
STEPS, BATCH, LR = 1000, 12, 3e-3


class RMSNorm(nn.Module):
    """Rescale each vector to root-mean-square 1, then multiply by a learned per-dimension gain."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.gain = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.gain


def rope_tables(ctx, head_dim, base=10000.0):
    """Angle for position p and pair i: p * base^(-2i/head_dim). Fast-turning pairs see short
    distances, slow-turning pairs see long ones."""
    freqs = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)   # (head_dim/2,)
    angles = torch.outer(torch.arange(ctx).float(), freqs)                # (ctx, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate each pair (x1, x2) of a query or key vector by its position's angle."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D, 3 * D, bias=False)   # W_Q, W_K, W_V side by side
        self.out = nn.Linear(D, D, bias=False)       # W_O: mixes the heads back together
        mask = torch.tril(torch.ones(CTX, CTX, dtype=torch.bool))
        self.register_buffer("mask", mask)            # True where token i may look at token j (j <= i)

    def forward(self, x, cos, sin, return_weights=False):
        B, T, _ = x.shape
        hd = D // N_HEAD
        q, k, v = self.qkv(x).split(D, dim=-1)                       # each (B, T, D)
        # Split the width into heads: (B, T, D) -> (B, heads, T, head_dim).
        q, k, v = (t.view(B, T, N_HEAD, hd).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos[:T], sin[:T]), apply_rope(k, cos[:T], sin[:T])
        scores = q @ k.transpose(-2, -1) / math.sqrt(hd)               # (B, heads, T, T)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))  # no peeking at the future
        weights = F.softmax(scores, dim=-1)                            # each row sums to 1
        y = weights @ v                                                 # weighted sum of values
        y = y.transpose(1, 2).reshape(B, T, D)                          # glue the heads back together
        y = self.out(y)
        return (y, weights) if return_weights else y


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.attn = RMSNorm(D), CausalSelfAttention()
        self.norm2 = RMSNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D, bias=False), nn.GELU(),
                                 nn.Linear(4 * D, D, bias=False))

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)   # read from the residual stream, add back
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, D)
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYER))
        self.norm = RMSNorm(D)
        self.unembed = nn.Linear(D, vocab_size, bias=False)
        cos, sin = rope_tables(CTX, D // N_HEAD)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        for p in self.parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)

    def forward(self, idx):                     # idx: (B, T) token ids
        x = self.embed(idx)                     # (B, T, D): the residual stream starts here
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        return self.unembed(self.norm(x))       # (B, T, vocab) logits for the next token

    @torch.no_grad()
    def generate(self, idx, n):
        for _ in range(n):
            logits = self(idx[:, -CTX:])[:, -1]          # only the last position predicts the next token
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


tok = get_tokenizer()
text = load_text()
data = torch.tensor(tok.encode(text))
vocab = len(tok.vocab) + 1                      # 256 bytes + 512 merges + 1 special token
n = int(0.9 * len(data))
train_ids, val_ids = data[:n], data[n:]          # same split as ch03_bigram.py: last 10% held out


def batch(split):
    src = train_ids if split == "train" else val_ids
    i = torch.randint(0, len(src) - CTX - 1, (BATCH,))
    x = torch.stack([src[j:j + CTX] for j in i])
    y = torch.stack([src[j + 1:j + CTX + 1] for j in i])  # targets: the same window shifted by one
    return x, y


model = GPT(vocab)
per_layer = sum(p.numel() for p in model.blocks[0].parameters())
total = sum(p.numel() for p in model.parameters())
print(f"tokens: {len(data):,}   vocab: {vocab}   layers {N_LAYER}, d={D}, heads {N_HEAD}, context {CTX}")
print(f"parameters per block: {per_layer:,}   12*d^2 = {12 * D * D:,}")
print(f"parameters total: {total:,}   (embedding + unembedding: {2 * vocab * D:,})")


@torch.no_grad()
def val_loss(k=20):
    model.eval()
    out = sum(F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
              for x, y in (batch("val") for _ in range(k))) / k
    model.train()
    return out


opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.05)
t0 = time.time()
print(f"guessing uniformly: loss = ln({vocab}) = {math.log(vocab):.2f}")
for step in range(STEPS + 1):
    x, y = batch("train")
    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
    if step % 200 == 0:
        print(f"step {step:4d}  train loss {loss.item():.3f}  val loss {val_loss():.3f}"
              f"  ({time.time() - t0:.0f} s)")
    if step == STEPS:
        break
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()

vl = val_loss(50)
bytes_per_token = len(tok.decode(val_ids.tolist()).encode("utf-8")) / len(val_ids)
print(f"val perplexity {math.exp(vl):.1f}   val loss per byte {vl / bytes_per_token:.3f} nats"
      "   (bigram in ch03: 1.942)")

# Attention sanity check on one validation window, first block, first head.
model.eval()
x, _ = batch("val")
with torch.no_grad():
    h = model.embed(x[:1])
    _, w = model.blocks[0].attn(model.blocks[0].norm1(h), model.cos, model.sin, return_weights=True)
w = w[0, 0]                                                       # (T, T) for one head
print("\n== attention check (block 0, head 0) ==")
print(f"rows sum to 1: max |row sum - 1| = {(w.sum(-1) - 1).abs().max().item():.1e}")
print(f"weight on future tokens (above the diagonal): {w.triu(1).sum().item():.1f}")
print("first 5 rows (row i = how token i spreads its attention over tokens 0..i):")
for row in w[:5, :5]:
    print("  " + " ".join(f"{v:.2f}" for v in row.tolist()))

print("\n== sample ==")
start = torch.tensor([[tok.encode("\n")[0]]])
print(tok.decode(model.generate(start, 120)[0].tolist()))
print(f"\ntotal time {time.time() - t0:.0f} s")
