# Run: ~/.venvs/llm/bin/python exercises/ch08_decode.py
# Chapter 8: a toy decoder with and without a KV cache, sampling rules, and serving arithmetic.
import math
import time

import torch

torch.manual_seed(0)
torch.set_num_threads(2)

# ---------------------------------------------------------------------------
# 1. A tiny decoder-only transformer (random weights) with grouped-query attention.
# ---------------------------------------------------------------------------
V, D, L, H, KVH = 1000, 256, 4, 8, 2  # vocab, width, layers, query heads, key/value heads
HD = D // H                           # head dimension


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = torch.nn.Linear(D, H * HD, bias=False)
        self.wk = torch.nn.Linear(D, KVH * HD, bias=False)
        self.wv = torch.nn.Linear(D, KVH * HD, bias=False)
        self.wo = torch.nn.Linear(H * HD, D, bias=False)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(D, 4 * D), torch.nn.GELU(), torch.nn.Linear(4 * D, D))
        self.n1, self.n2 = torch.nn.LayerNorm(D), torch.nn.LayerNorm(D)

    def forward(self, x, cache=None):
        # x: (T_new, D). cache: dict holding past keys/values of shape (KVH, T_past, HD), or None.
        T = x.shape[0]
        h = self.n1(x)
        q = self.wq(h).view(T, H, HD).transpose(0, 1)    # (H, T, HD)
        k = self.wk(h).view(T, KVH, HD).transpose(0, 1)  # (KVH, T, HD)
        v = self.wv(h).view(T, KVH, HD).transpose(0, 1)
        if cache is not None:
            if "k" in cache:  # append the new keys/values to the ones kept from earlier steps
                k = torch.cat([cache["k"], k], dim=1)
                v = torch.cat([cache["v"], v], dim=1)
            cache["k"], cache["v"] = k, v
        Tk = k.shape[1]
        k = k.repeat_interleave(H // KVH, dim=0)          # each K/V head serves H/KVH query heads
        v = v.repeat_interleave(H // KVH, dim=0)
        scores = q @ k.transpose(1, 2) / math.sqrt(HD)    # (H, T, Tk)
        # causal mask: the new token at position (Tk - T + i) may only see positions <= itself
        mask = torch.ones(T, Tk, dtype=torch.bool).tril(diagonal=Tk - T)
        scores = scores.masked_fill(~mask, float("-inf"))
        att = torch.softmax(scores, dim=-1) @ v           # (H, T, HD)
        x = x + self.wo(att.transpose(0, 1).reshape(T, D))
        return x + self.mlp(self.n2(x))


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(V, D)
        self.pos = torch.nn.Embedding(2048, D)
        self.layers = torch.nn.ModuleList(Layer() for _ in range(L))
        self.out = torch.nn.Linear(D, V, bias=False)

    def forward(self, ids, start=0, caches=None):
        x = self.emb(ids) + self.pos(torch.arange(start, start + len(ids)))
        for i, layer in enumerate(self.layers):
            x = layer(x, None if caches is None else caches[i])
        return self.out(x)  # logits, (T_new, V)


@torch.no_grad()
def generate_no_cache(model, prompt, n):
    ids = prompt.clone()
    for _ in range(n):  # every step re-runs the whole sequence so far
        logits = model(ids)
        ids = torch.cat([ids, logits[-1].argmax().view(1)])
    return ids


@torch.no_grad()
def generate_cache(model, prompt, n):
    caches = [dict() for _ in range(L)]
    logits = model(prompt, 0, caches)             # prefill: the whole prompt in one pass
    ids = torch.cat([prompt, logits[-1].argmax().view(1)])
    for _ in range(n - 1):                        # decode: one new token per pass
        logits = model(ids[-1:], len(ids) - 1, caches)
        ids = torch.cat([ids, logits[-1].argmax().view(1)])
    return ids


model = TinyLM().eval()
prompt = torch.randint(0, V, (64,))
N_NEW = 256
generate_cache(model, prompt, 4)  # warm-up

t0 = time.perf_counter(); a = generate_no_cache(model, prompt, N_NEW); t1 = time.perf_counter()
b = generate_cache(model, prompt, N_NEW); t2 = time.perf_counter()
print(f"1. prompt {len(prompt)} tokens, generate {N_NEW} greedily")
print(f"   identical outputs: {torch.equal(a, b)}")
print(f"   no cache : {t1 - t0:6.2f} s  ({N_NEW / (t1 - t0):5.0f} tokens/s)")
print(f"   KV cache : {t2 - t1:6.2f} s  ({N_NEW / (t2 - t1):5.0f} tokens/s)")
kv_bytes = 2 * L * KVH * HD * 4 * len(b)  # fp32 here
print(f"   KV cache size at the end: 2*{L}*{KVH}*{HD}*4 bytes*{len(b)} tokens = {kv_bytes / 1024:.0f} KiB")

# ---------------------------------------------------------------------------
# 2. Sampling: how temperature, top-k, top-p and min-p reshape one distribution.
# ---------------------------------------------------------------------------
logits = torch.tensor([4.0, 3.5, 3.0, 2.0, 1.0, 0.5, 0.0, -1.0])


def show(name, p):
    kept = int((p > 0).sum())
    print(f"   {name:<16} kept={kept}  " + " ".join(f"{x:.2f}" for x in p.tolist()))


def top_k(p, k):
    out = torch.zeros_like(p); idx = p.topk(k).indices; out[idx] = p[idx]
    return out / out.sum()


def top_p(p, thr):
    sp, idx = p.sort(descending=True)
    keep = (sp.cumsum(0) - sp) < thr  # keep tokens until the kept mass reaches thr
    out = torch.zeros_like(p); out[idx[keep]] = sp[keep]
    return out / out.sum()


def min_p(p, ratio):
    out = torch.where(p >= ratio * p.max(), p, torch.zeros_like(p))
    return out / out.sum()


print("\n2. one set of logits, probabilities over 8 tokens")
for T in (0.5, 1.0, 2.0):
    show(f"temperature {T}", torch.softmax(logits / T, dim=0))
p = torch.softmax(logits, dim=0)
show("top-k k=3", top_k(p, 3))
show("top-p p=0.9", top_p(p, 0.9))
show("min-p 0.1", min_p(p, 0.1))
g = torch.Generator().manual_seed(0)
draws = torch.multinomial(top_p(p, 0.9), 10_000, replacement=True, generator=g)
print("   10,000 draws with top-p 0.9, counts per token:", torch.bincount(draws, minlength=8).tolist())

# ---------------------------------------------------------------------------
# 3. Serving arithmetic for real models on one H100 (80 GB, 3.35 TB/s, ~989 TFLOP/s bf16 dense).
# ---------------------------------------------------------------------------
HBM, BW, PEAK = 80e9, 3.35e12, 989e12
GiB = 2**30


def kv_per_token(layers, kv_heads, head_dim, bytes_=2):
    return 2 * layers * kv_heads * head_dim * bytes_


print("\n3. KV cache per token (bf16)")
models = {
    "Llama 3.1 8B  (32 layers, 8 KV heads x 128)": kv_per_token(32, 8, 128),
    "Llama 3.1 70B (80 layers, 8 KV heads x 128)": kv_per_token(80, 8, 128),
    "same 70B if every head kept K/V (64 heads)": kv_per_token(80, 64, 128),
    "DeepSeek-V3 MLA (61 layers, 512+64 latent)": 61 * (512 + 64) * 2,
}
for name, b_tok in models.items():
    print(f"   {name}: {b_tok / 1024:7.1f} KiB/token, {b_tok * 131072 / GiB:6.1f} GiB per 128K-token sequence")

w8 = 8.03e9 * 2  # Llama 3.1 8B weights in bf16
print(f"\n   Llama 3.1 8B: weights {w8 / 1e9:.1f} GB; decode upper bound at batch 1 = "
      f"{BW / w8:.0f} tokens/s (bandwidth / bytes read per step)")
free = HBM * 0.9 - w8  # leave ~10% for activations and runtime
for ctx in (2048, 8192, 32768):
    print(f"   room for {free / (models['Llama 3.1 8B  (32 layers, 8 KV heads x 128)'] * ctx):5.0f} "
          f"sequences of {ctx:>6} tokens in the KV cache")

print(f"\n4. arithmetic intensity of a decode step (8B model, weights only), ridge = {PEAK / BW:.0f} FLOPs/byte")
for B in (1, 8, 64, 256, 512):
    flops, bytes_ = 2 * 8.03e9 * B, w8
    t = max(flops / PEAK, bytes_ / BW)
    bound = "memory" if bytes_ / BW > flops / PEAK else "compute"
    print(f"   batch {B:>3}: {flops / bytes_:5.0f} FLOPs/byte, step >= {t * 1e3:5.2f} ms ({bound}-bound), "
          f"<= {B / t:7.0f} tokens/s total")
