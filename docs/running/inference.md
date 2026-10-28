---
title: "8. Inference and Serving"
---

<script setup>
import { cards } from './inference-review'
</script>

# 8. Inference and Serving

Inference is running a trained model to produce text, and serving is doing it for many users at once on
expensive GPUs. Interviewers for infrastructure roles ask about it more than anything else in this primer,
because it is a memory-bandwidth and scheduling problem dressed up as an ML problem.

::: info Before you start
- A model reads and writes **tokens** (word pieces) and is a stack of identical **layers**
  ([Chapter 4](/foundations/transformer)). One pass through the stack turns the tokens so far into scores for
  the next token.
- The model's behaviour lives in its **weights**: billions of numbers fixed after training
  ([Chapter 5](/building/pretraining)). Serving only reads them.
- Running a model costs about 2 floating-point operations per weight per token
  ([Chapter 0](/start/math-toolkit)). [Chapter 6](/building/training-at-scale) covers splitting work across GPUs.
- You know caches, memory bandwidth and virtual memory. This chapter builds on all three.
:::

## Generating text one token at a time

**In short:** a model produces one token per pass and feeds it back in, so a 500-token answer takes 500
sequential passes through every layer.

Ask a model "The capital of France is". It runs one <Term id="forward-pass">forward pass</Term>: the tokens go
in, and out comes a score for every token in its vocabulary. Those scores are <Term id="logit">logits</Term>. <Term id="softmax">Softmax</Term> turns them into a <Term id="probability-distribution">probability distribution</Term>, and we pick one token, say " Paris". Then we append " Paris" to the input and run the
model again to get the token after it.

This loop is <Term id="autoregressive">autoregressive</Term> generation: each output becomes part of the
next input. It cannot be parallelised across output tokens, because token 2 depends on the choice of token 1.
That one fact shapes the whole serving stack.

Inside each layer, two things happen to every token's vector (a list of a few thousand numbers):

- **Attention** lets a token gather information from earlier tokens. Each token computes three vectors from
  its own vector: a **query** ("what am I looking for?"), a **key** ("what do I contain?") and a **value**
  ("what do I pass on if chosen?"). The token's query is compared by dot product with the keys of all earlier
  tokens. Softmax turns those scores into proportions that add up to 1, and the token takes a mix of the earlier
  tokens' values in those proportions.
- **The <Term id="mlp">MLP</Term>** (a small two-matrix network) then transforms each token's vector on its
  own, without looking at other tokens.

Attention is the only place where tokens talk to each other. Remember that the keys and values of past
tokens are what later tokens read; it matters in the KV cache section below.

## Prefill and decode: two different workloads

**In short:** processing the prompt (prefill) does a lot of math per byte read and is limited by compute;
generating each new token (decode) reads all the weights to do a tiny amount of math and is limited by
memory bandwidth.

A request has two phases. In **<Term id="prefill">prefill</Term>**, the model reads the whole prompt in one
forward pass. All prompt tokens are known in advance, so the model processes them together: a 2,000-token
prompt is a matrix of 2,000 rows pushed through every weight matrix. In
**<Term id="decode">decode</Term>**, the model generates one token per forward pass, as described above.

Both phases read every weight from GPU memory once per forward pass. The difference is how much work they do
with each weight once it is read. Prefill multiplies each weight by 2,000 token vectors. Decode for one user
multiplies it by one.

The useful number is **<Term id="arithmetic-intensity">arithmetic intensity</Term>**: floating-point
operations done per byte moved from memory. A GPU has a fixed peak arithmetic rate and a fixed memory
bandwidth. Their ratio is the intensity at which both run out at the same time. Below it, the chip waits on
memory; above it, on arithmetic. Plotting achievable speed against intensity gives the
**<Term id="roofline">roofline</Term>** below.

<InfRooflineDiagram />

The math for one decode step at batch size 1, with N weights stored in 16-bit format (2 bytes each):

```
FLOPs  = 2 · N          (one multiply and one add per weight)
bytes  = 2 · N          (read each weight once)
intensity = 1 FLOP per byte
```

Here N is the parameter count. An H100 GPU needs about 295 FLOPs per byte to keep its arithmetic busy. Decode
at batch 1 delivers 1. So the arithmetic units are idle more than 99% of the time, and the speed of decode is
set by how fast you can stream the weights out of memory.

If B users decode together, the weights are read once and used B times. FLOPs become 2 · N · B, bytes stay
about 2 · N, and intensity becomes about B. **Batching is the main lever for decode efficiency.** Prefill
already has an intensity in the thousands, so it is compute-bound on its own.

::: info Back-of-envelope
**Upper bound on decode speed.** Each decode step must read all the weights. So:

tokens/s per sequence ≤ memory bandwidth / bytes of weights

[Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B) in 16-bit: 8 × 10⁹ weights × 2 bytes ≈ 16 GB.
One H100 reads its memory at about 3.35 TB/s. 3.35 × 10¹² / 16 × 10⁹ ≈ **209 tokens/s** at best, for one user.
Real systems reach maybe 70–80% of that.

Llama 3.1 70B in 16-bit is about 141 GB. It does not fit on one 80 GB GPU. Split over two GPUs (each reads its
half in parallel): 6.7 TB/s / 141 GB ≈ **47 tokens/s**, before any communication cost.

**Prefill time.** A 2,000-token prompt on the 8B model: 2 × 8 × 10⁹ × 2,000 ≈ 3.2 × 10¹³ FLOPs. At the 989
TFLOP/s peak that is 33 ms; at a realistic 50% of peak, about 65 ms.
:::

::: details Going deeper: the hardware numbers, and why decode never fully escapes memory
An [H100 SXM](https://www.nvidia.com/en-us/data-center/h100/) has 80 GB of <Term id="hbm">high-bandwidth memory (HBM)</Term> at about 3.35 TB/s, and about 989 TFLOP/s of dense <Term id="bf16">bf16</Term> matrix math. The ridge point is 989 / 3.35 ≈ 295 FLOPs per byte. Across GPU
generations, compute has mostly grown faster than bandwidth, so the ridge has moved right: about 140 on a
V100, 150 on an A100, 295 on an H100.

The weights are shared by every sequence in the batch, but each sequence's past keys and values (the KV
cache, next section) are private to it. During decode, attention reads each sequence's whole KV cache to do
a few FLOPs per element. That part stays memory-bound no matter how large the batch is. At large batch and
long context, KV-cache reads can exceed weight reads: 64 sequences of 4,096 tokens on Llama 3.1 8B hold 32
GiB of cache, twice the weights. This is why shrinking the KV cache (grouped-query attention, MLA, KV
quantization) speeds up decode, not only saves memory.
:::

## The KV cache

**In short:** keys and values of past tokens never change, so the server stores them instead of
recomputing them; this cache is often the largest thing in GPU memory after the weights, and it decides how
many users fit on a GPU.

Without a cache, step 500 of decode would re-run all 499 earlier tokens through every layer to get their keys
and values. Total work grows with the square of the output length. But a token's key and value at a given
layer depend only on that token and the tokens before it. Later tokens cannot change them. So the server
computes them once, stores them, and each decode step only computes the new token's query, key and value.

That store is the <Term id="kv-cache">KV cache</Term>. The exercise below generates the same 256 tokens with
and without it: identical output, about 7× faster on a 2-core CPU, and the gap grows with length. A
systems engineer would call it memoisation. Its cost is memory.

The size formula:

```
KV bytes = 2 × layers × kv_heads × head_dim × bytes_per_number × tokens
```

- **2**: one key and one value.
- **layers**: every layer has its own attention, so its own keys and values.
- **kv_heads × head_dim**: attention runs as several parallel "heads", each a smaller attention with its own
  queries, keys and values of size head_dim. kv_heads is the number of separate key/value sets stored.
- **bytes_per_number**: 2 for bf16, 1 for 8-bit.
- **tokens**: every token in every active sequence, prompt plus output.

::: info Back-of-envelope
**Llama 3.1 8B** ([config](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct/blob/main/config.json):
32 layers, 8 KV heads, head_dim 128), bf16:

2 × 32 × 8 × 128 × 2 bytes = 131,072 bytes = **128 KiB per token**.

- One 128K-token conversation (the model's maximum context): 128 KiB × 131,072 ≈ **16 GiB**, the same as the
  weights.
- On an 80 GB H100 with 16 GB of weights and ~10% headroom, about 56 GB is left: room for about **52
  sequences of 8K tokens**, or 13 of 32K.

**Llama 3.1 70B** (80 layers, 8 KV heads, head_dim 128): 2 × 80 × 8 × 128 × 2 = **320 KiB per token**, so
40 GiB for one 128K-token sequence.
:::

Memory for the KV cache sets the maximum batch size, and batch size sets decode efficiency. So the main
architecture changes for inference all shrink the KV cache.

**Fewer key/value heads.** Standard **multi-head attention** stores keys and values for every query head.
Llama 3.1 70B has 64 query heads; storing all 64 would cost 2,560 KiB per token, 8× more.

- <Term id="mqa">Multi-query attention (MQA)</Term> ([Shazeer, 2019](https://arxiv.org/abs/1911.02150)) keeps
  one key/value head shared by all query heads. Smallest cache, some quality loss.
- <Term id="gqa">Grouped-query attention (GQA)</Term> ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))
  shares each key/value head among a group of query heads. Llama 3.1 uses 8 groups. It keeps nearly all the
  quality of the full version and is the default in most open models.

**Compress the keys and values.** <Term id="mla">Multi-head latent attention (MLA)</Term>, from
[DeepSeek-V2](https://arxiv.org/abs/2405.04434), stores one small compressed vector per token per layer and
reconstructs keys and values from it on the fly. DeepSeek-V2 reports a 93% smaller KV cache than its earlier
67B model, with better quality.

::: details Going deeper: MLA in numbers
[DeepSeek-V3's config](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) has 61 layers.
Per token and layer it caches a 512-number compressed vector plus a 64-number piece that carries position
information. In bf16: 61 × (512 + 64) × 2 bytes ≈ 70 KB per token, about half of Llama 3.1 8B's, for a model
with 671 billion weights. The trick costs extra matrix multiplies to decompress. Those are cheap because
decode has spare arithmetic (the roofline section).

Other KV-cache reductions: storing the cache in 8-bit (halves it); **sliding-window attention**, where some
layers only look at the last few thousand tokens; and evicting or compressing old tokens (lossy, and
quality effects are debated).
:::

::: warning Thread
The KV cache is the only state a deployed model builds up while it works. Everything the model "learned" in a
conversation lives there: the user's name, the bug it found, the correction you gave it. When the request
ends, the server frees those blocks and the next request starts from the same frozen weights. Inference is
where all the experience happens, and none of it is kept. This is the gap between per-token activations
and months-frozen weights that the [thread](/extras/thread) is about.

The strongest counterargument is that the cache is large and getting larger. 128K tokens of context is
a novel's worth of working memory. In-context learning (picking up a pattern from examples in the prompt)
inside it is real. Prefix caching (below) even keeps a cache alive across requests. But a stored cache is a recording, not a consolidation: it grows linearly, costs
memory every time it is reused, and never changes the weights that produce the next intuition.
:::

## Sampling: choosing the next token

**In short:** the model outputs a probability for every token; temperature reshapes those probabilities,
and top-k, top-p and min-p cut off the unlikely tail before one token is drawn at random.

The model gives logits. Softmax turns them into probabilities: exp of each logit, divided by the sum of
all of them. Now the server must pick one token.

- **<Term id="greedy-decoding">Greedy decoding</Term>** always picks the most likely token. It is
  deterministic and good for short factual answers, but long outputs tend to loop and repeat.
- **Sampling** draws a token at random, in proportion to its probability. It gives varied text, but every
  so often draws a very unlikely token that derails the rest of the output.

The other methods sit between these two.

**<Term id="temperature">Temperature</Term>** divides every logit by a number T before softmax:

```
p_i = exp(z_i / T) / Σ_j exp(z_j / T)
```

Here z_i is the logit of token i and p_i its probability. With T below 1, gaps between logits grow, so the
distribution gets sharper. As T approaches 0, this becomes greedy. With T above 1, gaps shrink and the
distribution flattens toward uniform. The exercise shows one set of logits: the top token's probability is
0.66 at T = 0.5, 0.45 at T = 1 and 0.30 at T = 2.

The next three methods remove tokens and renormalise what is left so it sums to 1:

- **<Term id="top-k">Top-k</Term>** keeps the k most likely tokens. It is simple, but a fixed k is wrong
  for both confident steps (too many kept) and open ones (too few).
- **<Term id="top-p">Top-p</Term>** (nucleus sampling, [Holtzman et al., 2019](https://arxiv.org/abs/1904.09751))
  keeps the smallest set of top tokens whose probabilities add up to at least p, such as 0.9. The set
  shrinks when the model is confident and grows when it is not.
- **<Term id="min-p">Min-p</Term>** ([Nguyen et al., ICLR 2025](https://arxiv.org/abs/2407.01082)) keeps
  tokens with at least a fraction (such as 0.1) of the top token's probability. It adapts like top-p but
  uses a threshold relative to the best token. The authors argue it stays coherent at higher temperatures.

Production APIs typically expose temperature and top-p; serving engines such as vLLM implement all of these.
Sampling is a tiny cost next to the forward pass, but it has one systems consequence. Set T = 0 for
reproducibility and you still may not get identical outputs across runs, because batch composition changes
the order of floating-point additions inside the GPU kernels.

::: tip Interview checkpoint
<details><summary>Why is decode memory-bandwidth-bound while prefill is compute-bound?</summary>

Both read all the weights once per forward pass. Prefill uses each weight for every prompt token (thousands),
so it does thousands of FLOPs per byte read. Decode for one sequence uses each weight once: about 1 FLOP per
byte, far below the ~300 the GPU needs to keep its arithmetic busy.

</details>

<details><summary>Compute the KV cache for Llama 3.1 70B at 32K context, batch 16, bf16.</summary>

Per token: 2 × 80 layers × 8 KV heads × 128 × 2 bytes = 320 KiB. Tokens: 16 × 32,768 = 524,288.
Total: 320 KiB × 524,288 = 160 GiB, more than the 141 GB of weights. You would need to shard across GPUs,
quantize the cache, or lower the batch.

</details>

<details><summary>What is GQA, and why did it become the default?</summary>

Grouped-query attention lets several query heads share one key/value head, so the KV cache shrinks by the
group size (8× for Llama 3.1 70B). Quality stays close to full multi-head attention. A smaller cache means
bigger batches and fewer bytes read per decode step, so both throughput and latency improve.

</details>

<details><summary>How does temperature interact with top-p?</summary>

Temperature reshapes the distribution first (logits divided by T, then softmax), and top-p then cuts the
tail of the reshaped distribution. A high temperature flattens it, so top-p keeps more tokens. Senior add-on:
at high temperature the flattened tail can push many junk tokens inside the top-p set; min-p was proposed
partly to handle that case.

</details>
:::

## Batching: keeping the GPU busy

**In short:** continuous batching lets requests join and leave the batch at every decode step, and
PagedAttention stores the KV cache in small blocks so memory is not wasted; together they multiplied serving
throughput.

Batching raises decode intensity, so servers want big batches. The naive way is **static batching**: collect
B requests, run them together until all are done, then take the next B. Outputs have very different lengths.
A request that finishes after 20 tokens holds its slot while another runs to 2,000, and new arrivals wait in
the queue the whole time.

**<Term id="continuous-batching">Continuous batching</Term>** schedules per step instead of per request.
After every decode step, finished sequences leave and waiting ones join. This idea, called iteration-level
scheduling, comes from [Orca (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu). The
batch stays full, and a short request no longer waits for a long one.

New arrivals need a prefill, and a long prefill inside a decode step stalls every decoding user for that
step. **Chunked prefill** ([Sarathi-Serve, OSDI 2024](https://arxiv.org/abs/2403.02310)) splits a long prompt
into chunks and mixes one chunk into each step alongside the decodes. Users see steady token output.

### PagedAttention

Continuous batching creates a memory problem. You do not know in advance how long a sequence will get. Early
servers reserved one contiguous region per request, sized for the maximum length. Most of it was never
used, and the holes between regions could not be reused. The vLLM authors
[measured](https://vllm.ai/blog/2023-06-20-vllm) that existing systems wasted 60–80% of KV-cache memory this way.

**<Term id="paged-attention">PagedAttention</Term>** ([Kwon et al., SOSP 2023](https://arxiv.org/abs/2309.06180)),
the core of vLLM, borrows virtual memory from operating systems:

- The KV cache is split into fixed-size **blocks** (vLLM's default is 16 tokens), like pages.
- Each request has a **block table** mapping its logical blocks to physical blocks anywhere in GPU memory,
  like a page table.
- Blocks are allocated on demand as the sequence grows, so waste is at most one partly filled block per
  request. vLLM reports under 4% waste.
- Requests can share physical blocks (for example, the same prompt), with reference counts and
  copy-on-write, like forked processes.

<InfPagedKvDiagram />

The analogy fits well, and the paper uses it explicitly. Where it breaks: there is no hardware page-table
walker. The attention kernel itself reads the block table and gathers scattered blocks, so the kernel had to be
rewritten. Also, "swapping" to CPU memory is an explicit scheduler decision, often replaced by throwing the
cache away and recomputing it later. With less waste, more sequences fit, and vLLM reported 2–4× higher
throughput than earlier systems at the same latency.

### Prefix caching

Many requests start with the same text: a long system prompt, a shared document, the earlier turns of a chat.
Their KV cache for that prefix is identical, because a token's keys and values depend only on the tokens
before it. **<Term id="prefix-caching">Prefix caching</Term>** keeps those blocks after a request ends and
reuses them when a new request starts with the same tokens. It skips most of that request's prefill.

vLLM hashes each full block together with everything before it. SGLang's
[RadixAttention](https://arxiv.org/abs/2312.07104) keeps cached prefixes in a radix tree with LRU eviction.
Hosted APIs expose this as "prompt caching" with a discount for cached input tokens. The match must be exact
from the first token, so put stable content first and per-request content last. This also makes request
routing matter: sending a conversation's next turn to the server that already holds its cache saves the
prefill.

## FlashAttention: attention without the big matrix

**In short:** FlashAttention computes exactly the same attention but in tiles that stay in on-chip memory,
so it never writes the huge score matrix to GPU main memory; it is a cache-blocking optimisation.

Attention for a sequence of n tokens compares every query with every key: an n × n matrix of scores per head.
At n = 32,000 that is a billion numbers per head per layer. The straightforward implementation writes the
score matrix to GPU main memory (HBM), reads it back for softmax, writes the result, and reads it again to multiply by the
values. The arithmetic is cheap; the memory traffic dominates.

**<Term id="flash-attention">FlashAttention</Term>** ([Dao et al., 2022](https://arxiv.org/abs/2205.14135))
splits queries, keys and values into tiles that fit in the GPU's small on-chip memory (SRAM, about 200 KB per
compute unit). It computes one tile of scores, applies it, and moves on, never storing the full matrix.
The paper calls this "IO-aware". A systems engineer would call it loop tiling for cache locality, the same
trick as a blocked matrix multiply.

The hard part is softmax, which needs the sum over a whole row before any output is final. FlashAttention
uses an **online softmax**: it keeps a running maximum and running sum per row, and rescales the partial
output when a larger maximum appears. The result is exact, not an approximation. Memory for attention drops
from growing with n² to growing with n, one of the changes that made long context practical.

::: details Going deeper: versions and decode
[FlashAttention-2](https://arxiv.org/abs/2307.08691) improved how work is split across GPU threads.
[FlashAttention-3](https://arxiv.org/abs/2407.08608) targets H100 features (asynchronous copies, fp8) and
reports up to 740 TFLOP/s in fp16, about 75% of peak, versus 35% for FlashAttention-2 on the same chip.

FlashAttention helps most in prefill and training, where n queries meet n keys. In decode there is one query
per sequence, so the work is reading the KV cache. Decode kernels (such as Flash-Decoding) split a long
cache across many compute units and combine the partial results, so that a single long sequence can use the
whole GPU's bandwidth.
:::

## Quantization: fewer bytes per weight

**In short:** storing weights in 8 or 4 bits instead of 16 cuts the bytes read per decode step, which
speeds up memory-bound decode almost proportionally; quantizing activations too also speeds up compute.

Decode speed is bandwidth divided by bytes of weights. Halve the bytes and you nearly double the speed, and
you also free memory for the KV cache. <Term id="quantization">Quantization</Term> stores numbers in fewer
bits. A group of weights shares one scale factor. Each weight is stored as a small integer, and the real
value is integer × scale.

Two kinds, with different payoffs:

- **Weight-only** (for example "W4A16": 4-bit weights, 16-bit activations). The kernel reads 4-bit weights,
  converts them to bf16 on chip, and does normal bf16 math. It cuts bytes, so it speeds up decode. It does
  not speed up compute-bound prefill.
- **Weights and activations** ("W8A8" in int8 or <Term id="fp8">fp8</Term>). Both inputs to the matrix
  multiply are 8-bit, so the GPU uses its 8-bit math units, about twice as fast as bf16 on an H100. This helps
  prefill and large batches too.

Activations are harder to quantize than weights. Some <Term id="activations">activation</Term> channels have
values 100× larger than the rest, and one shared scale either clips them or crushes everything else to zero.
[LLM.int8()](https://arxiv.org/abs/2208.07339) found these outliers in models from about 6.7 billion
parameters up. [SmoothQuant](https://arxiv.org/abs/2211.10438) moves the difficulty from activations into
weights with a per-channel rescaling.

Rounding each weight to the nearest 4-bit value hurts quality. The two common smarter methods use a small
calibration dataset:

- **[GPTQ](https://arxiv.org/abs/2210.17323)** quantizes weights one column at a time and adjusts the
  not-yet-quantized weights to cancel the error introduced so far. It uses second-order information (how
  sensitive the layer's output is to each weight).
- **[AWQ](https://arxiv.org/abs/2306.00978)** (activation-aware weight quantization) observes that about 1%
  of weight channels matter most: the ones multiplied by large activations. It scales those channels up before
  quantizing, so they lose less precision.

::: info Back-of-envelope
**Llama 3.1 70B at batch 1 on H100s.**

| Format | Weight bytes | GPUs needed | Decode bound |
|---|---|---|---|
| bf16 | 141 GB | 2 | 6.7 TB/s ÷ 141 GB ≈ 47 tokens/s |
| fp8 / int8 | 70 GB | 1 (tight) | 3.35 TB/s ÷ 70 GB ≈ 48 tokens/s |
| int4 | ~37 GB (with scales) | 1 | 3.35 TB/s ÷ 37 GB ≈ 90 tokens/s |

8-bit halves the GPU count for the same speed; 4-bit nearly doubles the speed on one GPU and leaves ~35 GB for
KV cache.
:::

::: details Going deeper: 4-bit floats, and what quantization costs
8-bit weights are close to lossless for most large models. At 4 bits, losses depend on model, method and task.
Reasoning and long outputs tend to suffer more than short benchmarks, so check task metrics, not only average loss.
NVIDIA's Blackwell GPUs add hardware 4-bit floats.
[NVFP4](https://research.nvidia.com/labs/eai/blogs/pushing-intelligence-to-4-bit/) stores 4-bit values with one
fp8 scale per 16 values, about 4.5 bits per weight in total. The KV cache can be quantized separately, usually to
fp8.
:::

## Speculative decoding: guess cheaply, check in parallel

**In short:** a small draft model guesses several tokens and the big model checks them all in one forward
pass; an acceptance rule makes the output distribution exactly the big model's, so it is faster at no cost
in quality.

Decode at small batch leaves the arithmetic idle. Checking 5 tokens in one forward pass costs about the same
as generating 1, because the bytes read are the same. **<Term id="speculative-decoding">Speculative decoding</Term>** ([Leviathan et al., 2022](https://arxiv.org/abs/2211.17192);
[Chen et al., 2023](https://arxiv.org/abs/2302.01318)) uses that spare compute.

1. A small, fast **draft model** proposes k tokens, one at a time. It also records q(x), the probability it
   gave each one.
2. The big **target model** runs once on the prompt plus all k guesses. Like prefill, this gives its own
   probability p(x) at every position in one pass.
3. Go through the guesses in order. Accept guess x with probability min(1, p(x) / q(x)). At the first
   rejection, draw a replacement from the leftover distribution max(0, p − q), renormalised, and stop.
4. If all k are accepted, the target's pass already gives the distribution for one more token, so draw it free.

**Why it is lossless.** The rule is a form of rejection sampling. Where the draft over-proposes a token
(q > p), it is accepted only p/q of the time. Where it under-proposes (p > q), the leftover distribution
makes up exactly the missing probability. The papers prove the accepted token follows p exactly. With greedy
decoding it is simpler: accept guesses while they equal the target's top choice. Either way, the output is
what the big model alone would produce; only the speed changes.

The speedup depends on the acceptance rate α. With k guesses, the expected tokens per target pass is
(1 − α^(k+1)) / (1 − α). At α = 0.8 and k = 4, that is about 3.4 tokens per pass, minus the draft's own cost.
Papers report 2–3× in practice.

::: details Going deeper: drafts without a separate model, and when it stops helping
Newer methods attach small extra heads to the target model instead of a separate draft:
[Medusa](https://arxiv.org/abs/2401.10774) and [EAGLE](https://arxiv.org/abs/2401.15077) predict future tokens
from the target's own internal vectors. DeepSeek-V3 was trained with a multi-token prediction module and
[reports](https://arxiv.org/abs/2412.19437) that its second-token guess is accepted 85–90% of the time,
giving about 1.8× more tokens per second. **N-gram or prompt lookup** drafting copies spans from the prompt,
and works well for code editing.

Speculation spends spare compute. At large batch, decode is already closer to compute-bound and the
verification work competes with other users, so gains shrink. Servers often turn it off or shorten k under
heavy load.
:::

## Serving mixture-of-experts models

**In short:** a mixture-of-experts model needs memory for all its weights but reads only a few experts per
token, which is cheap at batch 1 and turns into a load-balancing and communication problem at large batch.

A <Term id="moe">mixture-of-experts (MoE)</Term> model replaces each MLP with many alternative MLPs, the
experts. A small router picks a few for each token. [DeepSeek-V3](https://arxiv.org/abs/2412.19437) has 671
billion weights in total but uses 37 billion per token: 8 of 256 routed experts per MoE layer, plus one shared
expert. ([Chapter 6](/building/training-at-scale) covers how MoE is trained.)

For serving, this splits the two costs apart:

- **Memory** is set by total weights: 671 billion numbers must sit in GPU memory. In its native fp8 that is about
  671 GB, more than one 8-GPU H100 server holds.
- **Compute per token** is set by active weights, like a 37B dense model.
- **Bytes read per step** depend on the batch. At batch 1, only the chosen experts are read, so decode runs
  at roughly 37B-model speed. At large batch, different tokens pick different experts, so almost every
  expert gets read each step. Each expert then sees only a slice of the batch, so the intensity per expert
  is lower than for a dense model with the same batch.

Large deployments use <Term id="expert-parallelism">expert parallelism</Term>: experts are spread across many
GPUs, and each token is shipped to the GPUs holding its chosen experts with an <Term id="all-to-all">all-to-all</Term> exchange, then shipped back. The engineering problems are the network
cost of that exchange, and hot experts that get far more tokens than others (DeepSeek
[replicates](https://arxiv.org/abs/2412.19437) heavily loaded experts). DeepSeek-V3's decode deployment
spreads one model over 40 servers (320 GPUs) so that each GPU holds only a few experts and runs a large batch
for them.

## Latency metrics and the throughput tradeoff

**In short:** users feel time to first token and time per output token; operators pay for total tokens per
second per GPU; larger batches raise throughput but worsen both latencies.

The standard metrics:

- **<Term id="ttft">Time to first token (TTFT)</Term>**: from request arrival to the first output token.
  Queueing plus prefill. It is what makes a chat feel responsive.
- **<Term id="tpot">Time per output token (TPOT)</Term>**, also called inter-token latency: the gap between
  output tokens during decode. People read very roughly 5–10 tokens per second, so above ~20 tokens/s per user
  most people stop noticing. Agents and code generation want much more.
- **End-to-end latency** ≈ TTFT + TPOT × (output tokens − 1).
- **Throughput**: total output tokens per second across all users, per GPU. This sets the cost per token.

The central tradeoff: a bigger batch reads the weights once for more users, so throughput rises. But each step
does more work and takes longer, so every user's TPOT gets worse. A decode step is memory-bound while the batch
is small. Here, adding users is nearly free: the exercise's table shows step time flat from batch 1 to 256 (in
the ideal, weights-only model). Past the ridge, every added user slows everyone.

Operators therefore set a service-level objective (SLO), for example "TTFT under 1 s and TPOT under 50 ms for
99% of requests". They maximise **goodput**: the request rate served while meeting it. Raw throughput numbers
without a latency target mean little.

## Splitting prefill and decode across machines

**In short:** prefill and decode want different hardware settings and interfere when they share a GPU, so
large deployments run them on separate GPU pools and ship the KV cache between them.

On one GPU, a burst of long prompts makes every decoding user stall (bad TPOT). Meanwhile decode work queued
behind prefills delays first tokens (bad TTFT). Chunked prefill softens this, but the two phases still compete
for the same GPU. They also want different parallelism: prefill benefits from splitting one prompt across GPUs
for speed, decode from huge batches.

**<Term id="pd-disaggregation">Prefill/decode disaggregation</Term>** runs a pool of prefill GPUs and a pool of
decode GPUs. A request is prefilled on one, its KV cache is copied over the network to another, and decode
continues there. Each pool is sized and tuned for its own SLO. [DistServe (OSDI 2024)](https://arxiv.org/abs/2401.09670)
reports serving up to 7.4× more requests within latency targets; [Splitwise](https://arxiv.org/abs/2311.18677)
and Moonshot AI's [Mooncake](https://arxiv.org/abs/2407.00079) (which serves Kimi) reached similar conclusions.
Open-source stacks such as [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) and
[llm-d](https://github.com/llm-d/llm-d) now package it for vLLM and SGLang.

The cost is moving the KV cache. That is why disaggregation needs fast networking, and why it pays off most
for large models and long prompts, where prefill is expensive.

::: info Back-of-envelope
**KV transfer for one request.** A 2,000-token prompt on Llama 3.1 70B: 2,000 × 320 KiB ≈ 625 MiB. Over a
400 Gb/s (50 GB/s) network link: about **13 ms**. Prefill of that prompt on 70B is about 2 × 70 × 10⁹ × 2,000
≈ 2.8 × 10¹⁴ FLOPs, roughly 0.3–0.6 s of one H100's compute at 50–100% of peak. The transfer is small next to the work it
moves, and it can overlap with prefill layer by layer.
:::

## Exercise: decoding with and without a KV cache

The script builds a tiny 4-layer decoder with random weights and grouped-query attention (8 query heads, 2
KV heads). It generates the same 256 tokens greedily twice: once re-running the whole sequence each step, once
with a KV cache (one prefill, then one token per step). Then it applies each sampling rule to one set of
logits, and prints the KV cache and decode-speed arithmetic for real models on an H100.

<<< @/../exercises/ch08_decode.py

Output (the timings in part 1 depend on your machine; everything else is exact):

```
1. prompt 64 tokens, generate 256 greedily
   identical outputs: True
   no cache :   6.34 s  (   40 tokens/s)
   KV cache :   0.95 s  (  268 tokens/s)
   KV cache size at the end: 2*4*2*32*4 bytes*320 tokens = 640 KiB

2. one set of logits, probabilities over 8 tokens
   temperature 0.5  kept=8  0.66 0.24 0.09 0.01 0.00 0.00 0.00 0.00
   temperature 1.0  kept=8  0.45 0.27 0.17 0.06 0.02 0.01 0.01 0.00
   temperature 2.0  kept=8  0.30 0.23 0.18 0.11 0.07 0.05 0.04 0.02
   top-k k=3        kept=3  0.51 0.31 0.19 0.00 0.00 0.00 0.00 0.00
   top-p p=0.9      kept=4  0.47 0.29 0.17 0.06 0.00 0.00 0.00 0.00
   min-p 0.1        kept=4  0.47 0.29 0.17 0.06 0.00 0.00 0.00 0.00
   10,000 draws with top-p 0.9, counts per token: [4683, 2938, 1711, 668, 0, 0, 0, 0]

3. KV cache per token (bf16)
   Llama 3.1 8B  (32 layers, 8 KV heads x 128):   128.0 KiB/token,   16.0 GiB per 128K-token sequence
   Llama 3.1 70B (80 layers, 8 KV heads x 128):   320.0 KiB/token,   40.0 GiB per 128K-token sequence
   same 70B if every head kept K/V (64 heads):  2560.0 KiB/token,  320.0 GiB per 128K-token sequence
   DeepSeek-V3 MLA (61 layers, 512+64 latent):    68.6 KiB/token,    8.6 GiB per 128K-token sequence

   Llama 3.1 8B: weights 16.1 GB; decode upper bound at batch 1 = 209 tokens/s (bandwidth / bytes read per step)
   room for   208 sequences of   2048 tokens in the KV cache
   room for    52 sequences of   8192 tokens in the KV cache
   room for    13 sequences of  32768 tokens in the KV cache

4. arithmetic intensity of a decode step (8B model, weights only), ridge = 295 FLOPs/byte
   batch   1:     1 FLOPs/byte, step >=  4.79 ms (memory-bound), <=     209 tokens/s total
   batch   8:     8 FLOPs/byte, step >=  4.79 ms (memory-bound), <=    1669 tokens/s total
   batch  64:    64 FLOPs/byte, step >=  4.79 ms (memory-bound), <=   13350 tokens/s total
   batch 256:   256 FLOPs/byte, step >=  4.79 ms (memory-bound), <=   53400 tokens/s total
   batch 512:   512 FLOPs/byte, step >=  8.31 ms (compute-bound), <=   61582 tokens/s total
```

Things to notice:

- The cached and uncached runs produce identical tokens. The cache is an exact optimisation, not an
  approximation. Try `N_NEW = 512`: the uncached time roughly quadruples, the cached time roughly doubles.
- Top-p 0.9 kept 4 tokens here because the top 3 sum to only 0.89. Draw counts match the renormalised
  probabilities.
- Part 4 ignores KV-cache reads. With them, large batches stay memory-bound longer (see "Going deeper" in the
  prefill/decode section).

## In practice

- **Engines.** [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang)
  and NVIDIA's TensorRT-LLM implement everything in this chapter: continuous batching, paged KV cache, prefix
  caching, quantized kernels, speculative decoding. The knobs you tune are the maximum batch size, the share of
  GPU memory for the KV cache, the chunked-prefill size, and the <Term id="tensor-parallelism">tensor-parallel</Term> degree (how many GPUs share each weight matrix).
- **Capacity planning.** Start from the KV-cache formula. Free memory ÷ (KV bytes per token × typical
  context length) gives concurrent sequences. That, not FLOPs, usually caps throughput.
- **Routing.** A load balancer that sends a conversation back to the replica holding its prefix cache saves
  prefill on every turn. Round-robin routing throws that away.
- **Long context is expensive twice.** Prefill attention grows with the square of the prompt length, and the
  KV cache grows linearly, crowding out other users. This is why providers charge more per token above some
  context length.
- **Pricing.** Output tokens cost several times more than input tokens in most APIs. Input is processed in
  compute-efficient prefill; output needs one memory-bound decode step each.

::: tip Interview checkpoint
<details><summary>Design the serving setup for a 70B chat model. Where do you start?</summary>

Start from memory: 141 GB of bf16 weights, so fp8 on 1 GPU or bf16 split across 2–4 GPUs with tensor
parallelism. Then the KV cache: 320 KiB per token tells you how many concurrent conversations fit. Use
continuous batching, a paged KV cache and prefix caching for shared system prompts. Set TTFT and TPOT
targets and tune the batch size for goodput. Senior add-on: at scale, consider prefill/decode
disaggregation and cache-aware routing, and measure acceptance rates before turning on speculative decoding.

</details>

<details><summary>Why is speculative decoding lossless?</summary>

The target model checks every guessed token. A guess is accepted with probability min(1, p/q), and on
rejection a replacement is drawn from the normalised leftover max(0, p − q). This is rejection sampling, and
it makes each emitted token follow the target's distribution p exactly. The draft only changes speed.

</details>

<details><summary>What problem does PagedAttention solve, and what is the OS analogy?</summary>

Reserving one contiguous KV-cache region per request at maximum length wastes most of the memory through
over-reservation and fragmentation. PagedAttention allocates fixed-size blocks on demand and maps them with a
per-request block table, like pages and page tables. Sharing blocks across requests works like copy-on-write.
Senior add-on: the attention kernel itself walks the block table; no hardware does it.

</details>

<details><summary>Why does weight-only int4 speed up decode but not prefill?</summary>

Decode is limited by bytes read, and 4-bit weights are a quarter of the bytes. Prefill is limited by
arithmetic, and weight-only kernels still compute in bf16 after converting the weights on chip, so the math
is unchanged. To speed up prefill you quantize activations too (W8A8, fp8).

</details>

<details><summary>Why can't you make a single user's decode faster by adding GPUs without limit?</summary>

Tensor parallelism splits the weights, so each GPU reads fewer bytes per step, which helps. But every layer
then needs an exchange of results between GPUs, a fixed cost per layer that does not shrink. Past a few GPUs
(usually within one server's NVLink domain), that communication dominates.

</details>
:::

## Common misconceptions

- **"The KV cache is an approximation."** It is exact: the stored keys and values are the ones a full
  recomputation would produce.
- **"Generation is slow because GPUs are not fast enough at math."** Single-user decode uses under 1% of the
  arithmetic. It is waiting on memory.
- **"Temperature 0 means fully deterministic."** Greedy choice is deterministic, but GPU floating-point sums
  can vary with batch composition, so outputs can still differ.
- **"Speculative decoding trades quality for speed."** With the standard acceptance rule, it does not change
  the output distribution at all.
- **"An MoE model with 37B active weights needs the memory of a 37B model."** It needs memory for all its
  weights; only the compute per token is small.

## Key takeaways

- Generation is one forward pass per token. Prefill is compute-bound; decode is memory-bandwidth-bound, with
  an upper bound of bandwidth ÷ weight bytes tokens/s per sequence.
- The KV cache is 2 × layers × kv_heads × head_dim × bytes × tokens. It caps batch size, so GQA, MLA and KV
  quantization exist to shrink it.
- Continuous batching, PagedAttention and prefix caching keep GPUs full and memory used; FlashAttention
  removes attention's memory traffic without changing its result.
- Quantization and speculative decoding attack the bytes-per-token bottleneck; speculative decoding is
  exactly lossless.
- Serving is a tradeoff between throughput (cost) and TTFT/TPOT (user experience); disaggregation separates
  prefill and decode so each can meet its target.

## Further reading

- [Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention*](https://arxiv.org/abs/2309.06180):
  the vLLM paper; reads like an OS paper and is the best single source for this chapter.
- [Dao et al., *FlashAttention*](https://arxiv.org/abs/2205.14135): IO-aware tiling and the online softmax,
  explained for systems people.
- [Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192):
  short, with the proof of losslessness.
- [Pope et al., *Efficiently Scaling Transformer Inference*](https://arxiv.org/abs/2211.05102): Google's
  analysis of partitioning, latency and cost; the source of much back-of-envelope reasoning.
- [Zhong et al., *DistServe*](https://arxiv.org/abs/2401.09670): why prefill and decode should be split, with
  goodput as the metric.
- [Google DeepMind, *How to Scale Your Model*](https://jax-ml.github.io/scaling-book/): rooflines and
  inference math worked through in detail (uses TPUs, the reasoning transfers).

## Review

<Flashcards id="inference" :cards="cards" />

<MarkDone id="inference" />
