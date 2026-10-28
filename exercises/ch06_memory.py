# Run: ~/.venvs/llm/bin/python exercises/ch06_memory.py
# Part 1: training memory per GPU under DP, ZeRO-1/2/3, tensor and pipeline parallelism.
# Part 2: a ring all-reduce simulated with NumPy shards, checked against a plain sum.
import numpy as np

GB = 1e9

# ---------------------------------------------------------------- Part 1: memory calculator
# Mixed-precision Adam keeps, per parameter:
#   bf16 weights (2 B) + bf16 gradients (2 B) + fp32 master weights, Adam m, Adam v (4 + 4 + 4 B) = 16 B
W, G, OPT = 2, 2, 12


def model_state_bytes(params, dp, tp=1, pp=1, zero=0):
    """Bytes of weights + grads + optimizer state on one GPU.

    TP and PP cut the model into tp*pp pieces. ZeRO then shards pieces of what is left over the dp copies:
    stage 1 shards optimizer state, stage 2 also gradients, stage 3 also weights.
    """
    p = params / (tp * pp)
    w = W * p / (dp if zero >= 3 else 1)
    g = G * p / (dp if zero >= 2 else 1)
    o = OPT * p / (dp if zero >= 1 else 1)
    return w + g + o


def activation_bytes(layers, hidden, seq, micro_batch, tp=1, recompute=False):
    """Activations kept for the backward pass on one GPU (the busiest pipeline stage).

    Per layer, Korthikanti et al. (2022) count about 34*s*b*h bytes in bf16 when the attention
    score matrix is not stored (FlashAttention or selective recompute). With full recompute, each
    layer keeps only its input (2*s*b*h bytes) plus one layer's full set while it is recomputed.
    Under 1F1B the first pipeline stage holds pp micro-batches of layers/pp layers each, which is
    `layers` layers' worth in total, so pp does not appear here. TP (with sequence parallelism)
    splits each layer's activations tp ways.
    """
    sbh = seq * micro_batch * hidden
    if recompute:
        return (2 * sbh * layers + 34 * sbh) / tp
    return 34 * sbh * layers / tp


def report(name, params, layers, hidden, seq, mbs, configs):
    print(f"\n{name}: {params / 1e9:.0f}B params, {layers} layers, hidden {hidden}, "
          f"seq {seq}, micro-batch {mbs}")
    print(f"{'layout':<28}{'GPUs':>6}{'model state':>13}{'activ.':>9}{'activ.+rc':>11}{'fits 80GB?':>12}")
    for label, dp, tp, pp, zero in configs:
        ms = model_state_bytes(params, dp, tp, pp, zero)
        act = activation_bytes(layers, hidden, seq, mbs, tp)
        act_rc = activation_bytes(layers, hidden, seq, mbs, tp, recompute=True)
        best = ms + min(act, act_rc)
        fits = "yes" if ms + act < 80 * GB else ("with rc" if best < 80 * GB else "no")
        print(f"{label:<28}{dp * tp * pp:>6}{ms / GB:>11.1f}GB{act / GB:>7.1f}GB{act_rc / GB:>9.1f}GB{fits:>12}")


# (label, dp, tp, pp, zero stage)
report("7B (Llama-2-7B-like)", 7e9, 32, 4096, 4096, 1, [
    ("plain DP x8", 8, 1, 1, 0),
    ("ZeRO-1 x8", 8, 1, 1, 1),
    ("ZeRO-2 x8", 8, 1, 1, 2),
    ("ZeRO-3 / FSDP x8", 8, 1, 1, 3),
])
report("70B (Llama-2-70B-like)", 70e9, 80, 8192, 4096, 1, [
    ("plain DP x1024", 1024, 1, 1, 0),
    ("ZeRO-3 / FSDP x1024", 1024, 1, 1, 3),
    ("TP8 x DP128, ZeRO-1", 128, 8, 1, 1),
    ("TP8 x PP8 x DP16, ZeRO-1", 16, 8, 8, 1),
])
report("405B (Llama-3-405B-like)", 405e9, 126, 16384, 8192, 1, [
    ("TP8 x DP2048, ZeRO-3", 2048, 8, 1, 3),
    ("TP8 x PP16 x DP128, ZeRO-1", 128, 8, 16, 1),
    ("TP8 x PP16 x DP128, ZeRO-3", 128, 8, 16, 3),
])

# ---------------------------------------------------------------- Part 2: ring all-reduce
rng = np.random.default_rng(0)


def ring_all_reduce(grads):
    """grads: list of N equal-length arrays, one per worker. Returns the per-worker results
    and the number of elements each worker sent."""
    n = len(grads)
    chunks = [np.array_split(g.copy(), n) for g in grads]  # worker i holds n chunks
    sent = [0] * n
    # Phase 1, reduce-scatter: in step s, worker i sends chunk (i - s) to worker i+1, which adds it.
    for s in range(n - 1):
        msgs = [(i, (i - s) % n, chunks[i][(i - s) % n].copy()) for i in range(n)]
        for i, c, data in msgs:  # all sends in a step happen "at once"
            chunks[(i + 1) % n][c] += data
            sent[i] += data.size
    # Now worker i owns the full sum of chunk (i + 1) % n.
    # Phase 2, all-gather: pass the finished chunks around the ring, overwriting.
    for s in range(n - 1):
        msgs = [(i, (i + 1 - s) % n, chunks[i][(i + 1 - s) % n].copy()) for i in range(n)]
        for i, c, data in msgs:
            chunks[(i + 1) % n][c] = data
            sent[i] += data.size
    return [np.concatenate(c) for c in chunks], sent


N, M = 8, 1_000_000
grads = [rng.standard_normal(M).astype(np.float64) for _ in range(N)]
out, sent = ring_all_reduce(grads)
expected = np.sum(grads, axis=0)
print(f"\nRing all-reduce: {N} workers, {M:,} gradient values each")
print("every worker has the full sum:", all(np.allclose(o, expected) for o in out))
print(f"elements sent per worker: {sent[0]:,}  (= 2(N-1)/N * M = {2 * (N - 1) / N * M:,.0f})")
print(f"a central parameter server would receive N*M = {N * M:,} and send the same back")

print("\nPer-worker traffic and step count as the ring grows (gradient size M = 1):")
print(f"{'N':>6}{'sent per worker':>17}{'sequential steps':>18}")
for n in [2, 8, 64, 1024, 16384]:
    print(f"{n:>6}{2 * (n - 1) / n:>17.3f}{2 * (n - 1):>18}")
