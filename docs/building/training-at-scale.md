---
title: "6. Training at Scale"
---

<script setup>
import { cards } from './training-at-scale-review'
</script>

# 6. Training at Scale

A frontier model does not fit on one GPU, and one GPU would need decades to train it. This chapter is about
how labs split one training run across thousands of GPUs, and keep it running when hardware fails every few
hours. It is the part of an AI lab interview where an infrastructure engineer can shine.

::: info Before you start
- Training repeats one step: run a batch of text forward, measure the loss, run backward to get a
  <Term id="gradient">gradient</Term> for every parameter, then update the parameters
  ([chapter 2](/foundations/learning)).
- A transformer is a stack of identical layers, each an attention block followed by an MLP block. Both are
  mostly large matrix multiplies ([chapter 4](/foundations/transformer)).
- Training a model with N parameters on D tokens costs about 6·N·D floating-point operations
  ([chapter 5](/building/pretraining)).

You do not need to know the details. Each idea is restated where it is used.
:::

## The memory budget: why one GPU is not enough

**In short:** training needs about 16 bytes per parameter before counting activations, so a 70B model needs
over a terabyte. An H100 has 80 GB.

When you run a model to answer questions, you only need its weights. Training needs much more. For every
<Term id="parameter">parameter</Term> it keeps four things:

- **The weight itself**, in a 16-bit format for fast math: 2 bytes.
- **Its gradient**, also 16-bit: 2 bytes.
- **A 32-bit master copy of the weight**: 4 bytes. The next section on precision explains why.
- **The optimizer state.** The <Term id="adam">Adam</Term> optimizer keeps two running averages per
  parameter: of the gradient (the momentum) and of the squared gradient. In 32-bit floats that is 8 bytes.

That adds up to **16 bytes per parameter**. Using 16-bit math while keeping 32-bit copies is called
<Term id="mixed-precision">mixed precision</Term>. The 12 bytes of master weight and Adam averages together
are usually called the <Term id="optimizer-state">optimizer state</Term>.

Then come the <Term id="activations">activations</Term>: the intermediate results of the forward pass. The
backward pass needs them to compute gradients, so they are kept until it runs. Their size grows with the
number of tokens in flight, not with the parameter count. For a long sequence they can be larger than
everything else.

::: info Back-of-envelope: a 70B model
- Model state: 70 × 10⁹ params × 16 bytes = **1.12 TB**. That is 14 H100s just to hold it, with no room for
  anything else.
- Activations, one 4,096-token sequence, 80 layers of width 8,192, about 34 bytes per token per unit of
  width per layer: 34 × 4,096 × 8,192 × 80 ≈ **91 GB**. One sequence already overflows a GPU.
- Compute: 6 × 70 × 10⁹ params × 2 × 10¹² tokens (Llama 2's budget) ≈ 8.4 × 10²³ FLOPs. An H100 does about 4 × 10¹⁴ useful FLOP/s
  in practice. One GPU: 2 × 10⁹ seconds, about **66 years**. 1,024 GPUs: about **24 days**.

So you must split both the memory and the work. Everything below is a way of doing one or both.
:::

::: details Going deeper: where the activation estimate comes from
[Korthikanti et al. (2022)](https://arxiv.org/abs/2205.05198) count the bytes one transformer layer stores
for backward: about s·b·h·(34 + 5·a·s/h) in 16-bit, where s is sequence length, b the number of sequences,
h the hidden width and a the number of attention heads. The 5·a·s/h term is the attention score matrix,
which grows with the square of the sequence length. Modern kernels such as FlashAttention never store it
(they recompute it), which leaves about 34·s·b·h. The exercise uses that.
:::

## Data parallelism and all-reduce

**In short:** give every GPU a full copy of the model and a different slice of the batch, then average the
gradients with an all-reduce every step.

The simplest split is the one you would design for any batch job. Each GPU holds the whole model. Each
processes a different slice of the batch and computes its own gradients. Before anyone updates, all GPUs
average their gradients, so every copy applies the same update and stays identical. This is
<Term id="data-parallelism">data parallelism</Term>.

The averaging step is an <Term id="all-reduce">all-reduce</Term>: every GPU contributes an array, and every
GPU ends up with the element-wise sum. It is one of a family of group operations called
<Term id="collective">collectives</Term>. On NVIDIA hardware the NCCL library implements them.

The shape will feel familiar from MapReduce: a map (forward and backward on each shard) then a reduce (sum
the gradients). The analogy breaks in two places. Every worker needs the full reduced result, not one
reducer. And it repeats every step, a few seconds apart, with a hard barrier: no GPU can start step t+1
until the all-reduce of step t is done. It is bulk-synchronous parallel, so the slowest GPU sets the pace.
A single straggler slows all 16,000.

### The ring all-reduce

A central parameter server would receive N full gradient arrays and send N back, so its link becomes the
bottleneck. The ring all-reduce removes the hot spot. Arrange the N GPUs in a ring and cut each gradient
array into N chunks. It runs in two phases:

1. **Reduce-scatter.** In each of N−1 steps, every GPU sends one chunk to its right neighbour, which adds it
   to its own copy of that chunk. After N−1 steps, each GPU holds the complete sum for one chunk. This phase
   on its own is a collective called <Term id="reduce-scatter">reduce-scatter</Term>.
2. **All-gather.** In N−1 more steps, the finished chunks travel around the ring until every GPU has all of
   them. This phase on its own is called <Term id="all-gather">all-gather</Term>.

Each GPU sends N−1 chunks in each phase. With a gradient of M bytes, that is:

> bytes sent per GPU = 2 · (N−1)/N · M

In words: each GPU sends about twice the gradient size, **no matter how many GPUs there are**. Total work
grows with N, but it is spread evenly, and every link is busy in every step. That is why this pattern is
called bandwidth-optimal.

The catch is latency. The ring takes 2(N−1) sequential steps, and each has a fixed start-up cost. At
thousands of GPUs that adds up, so NCCL also uses tree algorithms and splits the job hierarchically: first
inside each server, then across servers.

::: info Back-of-envelope: all-reducing a 7B model's gradients
- Gradients in 16-bit: 7 × 10⁹ × 2 bytes = 14 GB. Each GPU sends about 2 × 14 = 28 GB.
- Inside one 8-GPU server, links of about 450 GB/s each way: roughly **60 ms**.
- Across servers at about 50 GB/s per GPU: roughly **0.5 s**.
- The compute for one step with 32K tokens per GPU: 6 × 7 × 10⁹ × 32,768 ≈ 1.4 × 10¹⁵ FLOPs, about
  **3.5 s** at 4 × 10¹⁴ FLOP/s.

So the communication fits inside the compute. Frameworks overlap them: they start all-reducing the last
layers' gradients while the backward pass is still working on the first layers.
:::

::: details Going deeper: the cost model NCCL lives by
The standard model is time = α + n/β per message: a fixed latency α plus size over bandwidth β. A ring
all-reduce costs about 2(N−1)·α + 2·(N−1)/N·M/β. The bandwidth term is optimal; the latency term grows
with N. Tree algorithms cut the latency term to about log N steps. NCCL picks an algorithm per call based on
size and topology. PyTorch's DistributedDataParallel groups gradients into buckets of about 25 MB so it
can launch all-reduces during the backward pass. The original description for deep learning is
[Baidu's 2017 post](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/); the algorithm
itself is older ([Patarasuk and Yuan, 2009](https://doi.org/10.1016/j.jpdc.2008.09.002)).
:::

## ZeRO and FSDP: stop storing N copies

**In short:** data parallelism stores the same 16 bytes per parameter on every GPU. ZeRO shards that state
across the GPUs instead, in three stages.

Data parallelism splits the work but not the memory. With 64 GPUs, there are 64 identical copies of the
weights, gradients and optimizer state. [ZeRO](https://arxiv.org/abs/1910.02054) (Zero Redundancy
Optimizer) removes the copies by giving each GPU one shard of each. <Term id="zero">ZeRO</Term> comes in
three stages, each sharding one more thing:

| Stage | Sharded across N GPUs | Bytes per parameter per GPU | Extra communication vs plain data parallel |
|---|---|---|---|
| Plain data parallel | nothing | 16 | none |
| ZeRO-1 | optimizer state (12 B) | 4 + 12/N | none |
| ZeRO-2 | + gradients | 2 + 14/N | none |
| ZeRO-3 | + weights | 16/N | about 1.5× |

The first two stages are nearly free. Look at the ring all-reduce again: it is a reduce-scatter plus an
all-gather. ZeRO-2 stops halfway. After the reduce-scatter, each GPU holds the summed gradient for its own
shard only, so it updates only its own shard of the weights. The all-gather then sends out updated weights
instead of summed gradients. Same bytes, much less memory.

ZeRO-3 also shards the weights themselves. Before running a layer, the GPUs all-gather that layer's full
weights, use them, and drop them again. It is like demand paging with prefetch: fetch the layer just before
it is needed, then evict it. The backward pass has to fetch them again, which is where the extra 50% of
traffic comes from. PyTorch's version of ZeRO-3 is <Term id="fsdp">FSDP</Term> (Fully Sharded Data
Parallel).

In the ZeRO paper's example, a 7.5B model on 64 GPUs needs 120 GB per GPU with plain data parallelism,
31.4 GB with stage 1, 16.6 GB with stage 2, and 1.9 GB with stage 3.

### Where ZeRO runs out

ZeRO alone cannot carry a frontier model, for three reasons:

- **It does not shrink activations.** Each GPU still runs whole sequences through whole layers.
- **One layer must fit at a time.** A 405B model's single layer is about 3 billion parameters, 6 GB in
  16-bit, plus prefetch buffers for the next one.
- **The batch cannot grow forever.** Every data-parallel GPU needs at least one sequence. With 16,000 GPUs
  and 8,000-token sequences that is 128 million tokens per step. Past a certain batch size, called the
  critical batch size, a bigger batch stops making training faster per token.

So you also have to split the model itself. There are two ways: across the width of each layer, and across
the depth of the stack.

## Tensor parallelism: split every matrix multiply

**In short:** cut each layer's weight matrices across GPUs so every GPU does part of every multiply. It
needs very fast links, so it stays inside one server.

The MLP block of a transformer does two matrix multiplies with a nonlinearity between them. Call the input
X, the first weight matrix A and the second B. The block computes Z = GELU(X·A)·B. GELU is applied to each
number on its own.

[Megatron-LM](https://arxiv.org/abs/1909.08053) splits it across two GPUs like this:

1. **Split A by columns** into A₁ and A₂. GPU 1 computes Y₁ = GELU(X·A₁), GPU 2 computes Y₂ = GELU(X·A₂).
   Each column of the output depends only on its own column of A, and GELU works number by number, so no
   communication is needed yet.
2. **Split B by rows** into B₁ and B₂. GPU 1 computes Y₁·B₁ and GPU 2 computes Y₂·B₂. Matrix multiply
   splits this way: Z = Y₁·B₁ + Y₂·B₂.
3. **All-reduce** to add the two partial results. Now both GPUs hold Z.

That is one all-reduce per MLP block in the forward pass and one in the backward pass. Attention splits the
same way. It already runs as several independent "heads", so each GPU takes a subset of them. This is
<Term id="tensor-parallelism">tensor parallelism</Term>. It splits weights, gradients, optimizer state and
most activations, all at once.

The cost is that the all-reduce sits on the critical path of every layer. The GPUs cannot overlap it with
much, and there are two per layer per direction. So tensor parallelism needs the fastest links available:
<Term id="nvlink">NVLink</Term>, the direct GPU-to-GPU links inside one server. A typical server has 8 GPUs,
so tensor parallelism is usually 8-way or less.

::: details Going deeper: sequence parallelism (the Megatron kind)
Tensor parallelism splits the matrix multiplies, but the small per-token steps between them (layer norm and
dropout) still run on full copies of the activations on every GPU. [Korthikanti et al. (2022)](https://arxiv.org/abs/2205.05198) split those parts
along the sequence instead: each GPU handles a slice of the tokens. The all-reduce becomes a reduce-scatter
(into sequence slices) plus an all-gather (back to full), which costs the same bytes. Activation memory per
GPU then drops by the tensor-parallel degree. They call this
<Term id="sequence-parallelism">sequence parallelism</Term>. Do not confuse it with context parallelism
below, which splits the attention itself.
:::

## Pipeline parallelism: split the stack of layers

**In short:** put consecutive layers on different GPUs and pass activations along. Cut the batch into
micro-batches so the stages work at the same time, and accept some idle time at the ends.

An 80-layer model can be cut into 8 stages of 10 layers, each on a different GPU or server. Stage 1 runs its
layers and sends the activations to stage 2, and so on. The backward pass flows the other way. Only the
activations at the cut cross the network, as point-to-point sends, which is little data. That makes this
split a good fit for the slower links between servers. It is called
<Term id="pipeline-parallelism">pipeline parallelism</Term>.

With one batch, only one stage works at a time while the other seven wait. So the batch is cut into
<Term id="micro-batch">micro-batches</Term>. While stage 2 works on micro-batch 1, stage 1 starts
micro-batch 2. Gradients from all micro-batches are summed before the single update, so the result is the
same as one big batch. It is the same idea as a CPU instruction pipeline: throughput improves, but filling
and draining the pipe costs time.

<TasPipelineBubbleDiagram />

The idle time is the <Term id="pipeline-bubble">pipeline bubble</Term>. With p stages and m micro-batches,
the naive schedule from [GPipe](https://arxiv.org/abs/1811.06965) wastes:

> bubble fraction = (p−1) / (m+p−1)

In words: the p−1 slots to fill the pipe, out of the m+p−1 slots the step takes. With 8 stages and 8
micro-batches, 7/15 ≈ 47% of the time is idle. With 64 micro-batches it falls to 7/71 ≈ 10%. So pipelines
want many micro-batches per step, which competes with data parallelism for the same limited batch.

GPipe has a memory problem too. It runs every micro-batch forward before any goes backward, so each stage
holds the activations of all m micro-batches at once. The **1F1B** (one forward, one backward) schedule from
[PipeDream](https://arxiv.org/abs/1806.03377) starts each micro-batch's backward pass as soon as possible.
After a warm-up, each stage alternates one forward and one backward. The bubble is the same size, but a
stage never holds more than p micro-batches. So memory no longer grows with m.

::: details Going deeper: interleaving and the uneven first and last stages
[Narayanan et al. (2021)](https://arxiv.org/abs/2104.04473) give each GPU several smaller, non-adjacent
chunks of layers (an interleaved schedule). With v chunks per GPU the bubble shrinks by about v, at the cost
of more point-to-point messages. Llama 3 used an interleaved schedule with a bubble of (p−1)/(v·m). Its
first stage also holds the token embedding table and the last stage computes the output layer and loss, so
[the Llama 3 paper](https://arxiv.org/abs/2407.21783) removed one transformer layer from each to balance
them. DeepSeek-V3's DualPipe schedule goes further and overlaps the communication of one micro-batch with
the computation of another ([DeepSeek-V3 report](https://arxiv.org/abs/2412.19437)).
:::

## Long sequences: context parallelism

**In short:** activations grow with sequence length, so very long sequences are split across GPUs by
token position. Attention then has to fetch keys and values from the other GPUs.

Recall that attention lets each token look at all earlier tokens. Each token produces a key vector and a
value vector. Each later token compares its own query vector against all earlier keys, then takes a weighted
mix of their values. At 128,000 tokens, the activations for one
sequence no longer fit even after tensor parallelism.

<Term id="context-parallelism">Context parallelism</Term> splits one sequence across GPUs by position: GPU
1 gets tokens 1–8,000, GPU 2 the next 8,000, and so on. Most of the layer works on each token alone, so it
needs no communication. Attention is the exception: tokens on GPU 2 need the keys and values from GPU 1.
[Ring attention](https://arxiv.org/abs/2310.01889) passes blocks of keys and values around a ring while each
GPU computes attention on the block it holds, overlapping the two. Llama 3 instead all-gathers the keys and
values. That handles unusual attention patterns more easily. In Llama 3 several query heads share one key and
value head, so keys and values are small and the transfer is cheap.

Llama 3 405B used 16-way context parallelism only for its long-context stage at 131,072 tokens.

::: details Going deeper: load balancing a causal mask
A token only attends to earlier tokens. So the GPU holding the last chunk of a sequence does far more
attention work than the GPU holding the first. The usual fix is to cut the sequence into 2·c chunks for c
GPUs and give each GPU one early and one late chunk, so the work evens out. Llama 3 does this.
:::

## Mixture of experts and expert parallelism

**In short:** a mixture-of-experts layer has many MLPs but sends each token to only a few. Spreading the
experts over GPUs means shuffling tokens between GPUs twice per layer.

A <Term id="moe">mixture-of-experts (MoE)</Term> layer replaces the single MLP block with many, called
experts. A small router network picks, for each token, the top few experts (often 1 to 8) and only those
run. So the parameter count grows with the number of experts, but the compute per token does not.
[DeepSeek-V3](https://arxiv.org/abs/2412.19437) has 671B parameters, of which 37B are active for each token.

<Term id="expert-parallelism">Expert parallelism</Term> places different experts on different GPUs. Each
GPU starts with its own tokens, the router decides where each must go, and the tokens are sent there. After
the experts run, the results are sent back. Each exchange is an <Term id="all-to-all">all-to-all</Term>:
every GPU sends a different piece to every other GPU.

Here the MapReduce analogy fits exactly. The router is the partition function, and the all-to-all is the
shuffle. The same problem follows: skew. If the router sends most tokens to a few popular experts, those
GPUs become hot partitions and everyone waits. MoE training adds a balancing term to the loss to discourage
this, caps how many tokens an expert accepts (a "capacity factor"), and sometimes drops the excess tokens.
DeepSeek-V3 balances with a per-expert bias the router adjusts instead of a loss term.

## Putting it together: 3D and 4D parallelism

**In short:** real runs combine tensor, context, pipeline and data parallelism. The chattiest split goes on
the fastest links: tensor parallel inside a server, data parallel across the whole cluster.

The splits are independent axes, so they multiply. A cluster of 1,024 GPUs might run 8-way tensor ×
8-way pipeline × 16-way data parallel. Each GPU belongs to one group along each axis. Combining tensor,
pipeline and data parallelism is called **3D parallelism**; adding context parallelism makes it 4D.

The network is not flat. Inside a server, GPUs talk over NVLink at about 450 GB/s per GPU in each direction.
Between servers, each GPU usually has its own <Term id="infiniband">InfiniBand</Term> link of about
50 GB/s each way. That is a gap of about 9×, so placement matters more than any other choice. The rule is to put the most frequent, most
blocking communication on the fastest links:

| Split | What crosses the wire | How often | Where it goes |
|---|---|---|---|
| Tensor | partial layer outputs (all-reduce) | 2–4 times per layer, blocking | inside a server (NVLink) |
| Context | keys and values (all-gather or ring) | every attention layer | inside a server or nearby |
| Expert | tokens (all-to-all) | twice per MoE layer | within or across servers |
| Pipeline | activations at stage cuts (point-to-point) | per micro-batch, small | across servers |
| Data / ZeRO | gradients and weight shards | once per step, overlapped | across the cluster |

<TasTopologyDiagram />

[Llama 3](https://arxiv.org/abs/2407.21783) states this order outright: it groups GPUs in the order
[tensor, context, pipeline, data], innermost first, with FSDP as the data-parallel axis. On 16,384 H100s in
the main stage it ran 8-way tensor × 16-way pipeline × 128-way data parallel, and reached 41% of the GPUs'
peak 16-bit arithmetic. That share is called <Term id="mfu">model FLOPs utilization (MFU)</Term>; 40%
is good at this scale.

::: tip Interview checkpoint
<details><summary>You have 1,024 H100s (128 servers of 8) and a 70B dense model. How do you shard it?</summary>

Start from memory: 70B × 16 bytes ≈ 1.1 TB of model state, and about 90 GB of activations per 4K-token
sequence. Tensor parallel 8 inside each server, over NVLink. That cuts state and activations by 8. Then
either FSDP across the remaining 128 data-parallel groups, or pipeline parallel 4 across servers with ZeRO-1
over 32 data-parallel groups. In the second, tensor × pipeline splits the model 32 ways (≈ 2.2B params per
GPU), and ZeRO-1 shards their optimizer state 32 ways: about 10 GB of state per GPU.
Check the batch: a 4M-token batch is 1,024 sequences, so 32 micro-batches per pipeline, and a bubble of
3/35 ≈ 9%. Add activation checkpointing if activations do not fit.

Senior add-on: measure, do not guess. Profile MFU for a few candidate layouts, watch for exposed
communication, and check that the data-parallel groups map onto the network's switch layout.
</details>

<details><summary>Why is tensor parallelism almost never wider than one server?</summary>

It does blocking all-reduces inside every layer, and they cannot be hidden behind compute. Across servers
the link is roughly 9× slower than NVLink, so the GPUs would spend most of their time waiting.

Senior add-on: systems with larger NVLink domains (NVIDIA's GB200 NVL72 connects 72 GPUs) raise this
ceiling, which changes the best layout.
</details>

<details><summary>Why doesn't ring all-reduce get slower per GPU as you add GPUs?</summary>

Each GPU sends 2·(N−1)/N times the gradient size, which approaches 2× and stops growing. The load is spread
evenly over all links. What grows is the number of sequential steps, 2(N−1), so latency rises; at scale
libraries switch to trees and hierarchical schemes.
</details>

<details><summary>ZeRO-3 versus tensor parallelism: when would you choose each?</summary>

ZeRO-3 splits storage but every GPU still does whole-layer math on whole sequences. Its traffic is
per-layer weight gathers that can be prefetched and overlapped, so it tolerates slower links. Tensor
parallelism splits the math and the activations too, but needs NVLink. Small and medium models often use
FSDP alone; large ones combine both.
</details>
:::

## Mixed precision: bf16, fp8 and loss scaling

**In short:** do the matrix math in 16-bit or 8-bit floats for speed, but keep 32-bit copies for the
update. bf16 keeps the range of 32-bit floats, which removes most of the trouble.

A floating-point number has exponent bits (range: how big or small it can be) and mantissa bits (precision:
how many significant digits). Fewer bits make the math faster and the data smaller. On an H100, 16-bit
matrix multiplies run many times faster than 32-bit ones, and 8-bit ones about twice as fast again. The
question is what you lose.

| Format | Exponent bits | Mantissa bits | Largest value | Significant decimal digits |
|---|---|---|---|---|
| fp32 | 8 | 23 | ~3 × 10³⁸ | ~7 |
| fp16 | 5 | 10 | 65,504 | ~3 |
| bf16 | 8 | 7 | ~3 × 10³⁸ | ~2–3 |
| fp8 (E4M3) | 4 | 3 | 448 | ~1 |

**fp16 and loss scaling.** Early mixed-precision training used fp16. Its problem is range: gradients smaller
than about 6 × 10⁻⁸ round to zero, and many real gradients are that small. The fix
([Micikevicius et al., 2017](https://arxiv.org/abs/1710.03740)) is <Term id="loss-scaling">loss
scaling</Term>. Multiply the loss by a large number, say 2¹⁶, before the backward pass. Every gradient gets
scaled by the same factor, which lifts it into range. Divide it back out, in 32-bit, before the update. If
any gradient overflows to infinity, skip the step and lower the scale.

**bf16.** <Term id="bf16">bf16</Term> ("brain float 16") takes the opposite trade. It keeps fp32's 8
exponent bits, so it has the same range, and gives up precision. Small gradients no longer vanish, so loss
scaling is not needed. Today it is the default for LLM training.

**Why keep 32-bit master weights.** With about 3 significant digits, bf16 cannot represent 1.000 + 0.0001:
the sum rounds back to 1.000. A typical update is a tiny fraction of the weight it changes. Applied to bf16
weights, most updates would vanish. So the optimizer updates a 32-bit master copy, then rounds it to bf16 for
the next forward pass. Those are the 4 extra bytes in the 16-byte budget.

**fp8.** <Term id="fp8">fp8</Term> doubles matrix-multiply throughput again on H100-class GPUs, but its range
is tiny. Each tensor, or each small block of a tensor, gets its own scale factor so its values fit. Sums are
accumulated in higher precision. [DeepSeek-V3](https://arxiv.org/abs/2412.19437) is the best-known
large-scale fp8 run: it ran every weight-matrix multiply in fp8 (the forward one and both backward ones), with a
scale per small tile, and kept sensitive parts (embeddings, normalization, the MoE router, optimizer state) in higher
precision. Llama 3 405B trained in bf16.

::: details Going deeper: the two fp8 formats
There are two fp8 layouts ([Micikevicius et al., 2022](https://arxiv.org/abs/2209.05433)): E4M3 (4 exponent,
3 mantissa bits) for more precision, and E5M2 for more range. A common recipe uses E4M3 for weights and
activations and E5M2 for gradients. DeepSeek-V3 used E4M3 everywhere and relied on fine-grained scaling
(1×128 tiles for activations, 128×128 blocks for weights) to handle range.
:::

## Activation checkpointing: trade compute for memory

**In short:** throw away most activations during the forward pass and recompute them during the backward
pass. It costs about a third more compute and saves most of the activation memory.

The backward pass of a layer needs that layer's activations from the forward pass. Storing them all for 80
layers is what made one 4K-token sequence of a 70B model take 91 GB. The alternative is to keep only each
layer's input, which is a small fraction of its activations. When the backward pass reaches a layer, it runs
that layer forward again from the saved input, then does its backward. This is
<Term id="activation-checkpointing">activation checkpointing</Term>, also called recomputation. (It has
nothing to do with saving the model to disk, which comes later in this chapter.)

It is the classic cache-or-recompute trade, like dropping a materialized view and recomputing it on read.
The forward pass is about a third of a training step's compute (2 of the 6 in 6·N·D). Recomputing it all
adds up to about 33%. **Selective recomputation** keeps the activations that are expensive to recompute but
cheap to store, and recomputes the rest, such as the attention scores. That captures most of the saving for
a few percent of compute ([Korthikanti et al., 2022](https://arxiv.org/abs/2205.05198)).

## Try it: a memory calculator and a ring all-reduce

The exercise has two parts. Part 1 computes model state and activation memory per GPU for several layouts,
using the 16-bytes-per-parameter budget and the 34·s·b·h activation estimate. "activ.+rc" is with full
activation checkpointing. Part 2 runs a ring all-reduce over 8 NumPy "GPUs", checks it against a plain
sum, and counts what each worker sent.

<<< @/../exercises/ch06_memory.py

Output:

```
7B (Llama-2-7B-like): 7B params, 32 layers, hidden 4096, seq 4096, micro-batch 1
layout                        GPUs  model state   activ.  activ.+rc  fits 80GB?
plain DP x8                      8      112.0GB   18.3GB      1.6GB          no
ZeRO-1 x8                        8       38.5GB   18.3GB      1.6GB         yes
ZeRO-2 x8                        8       26.2GB   18.3GB      1.6GB         yes
ZeRO-3 / FSDP x8                 8       14.0GB   18.3GB      1.6GB         yes

70B (Llama-2-70B-like): 70B params, 80 layers, hidden 8192, seq 4096, micro-batch 1
layout                        GPUs  model state   activ.  activ.+rc  fits 80GB?
plain DP x1024                1024     1120.0GB   91.3GB      6.5GB          no
ZeRO-3 / FSDP x1024           1024        1.1GB   91.3GB      6.5GB     with rc
TP8 x DP128, ZeRO-1           1024       35.8GB   11.4GB      0.8GB         yes
TP8 x PP8 x DP16, ZeRO-1      1024        5.2GB   11.4GB      0.8GB         yes

405B (Llama-3-405B-like): 405B params, 126 layers, hidden 16384, seq 8192, micro-batch 1
layout                        GPUs  model state   activ.  activ.+rc  fits 80GB?
TP8 x DP2048, ZeRO-3         16384        0.4GB   71.9GB      4.8GB         yes
TP8 x PP16 x DP128, ZeRO-1   16384       13.0GB   71.9GB      4.8GB     with rc
TP8 x PP16 x DP128, ZeRO-3   16384        0.4GB   71.9GB      4.8GB         yes

Ring all-reduce: 8 workers, 1,000,000 gradient values each
every worker has the full sum: True
elements sent per worker: 1,750,000  (= 2(N-1)/N * M = 1,750,000)
a central parameter server would receive N*M = 8,000,000 and send the same back

Per-worker traffic and step count as the ring grows (gradient size M = 1):
     N  sent per worker  sequential steps
     2            1.000                 2
     8            1.750                14
    64            1.969               126
  1024            1.998              2046
 16384            2.000             32766
```

What to notice:

- **ZeRO-3 on 1,024 GPUs shrinks a 70B model's state to 1.1 GB, but activations still overflow.** Sharding
  storage does nothing for activations; checkpointing, tensor or context parallelism do.
- **Pipeline parallelism does not reduce activations on the busiest stage.** Under 1F1B the first stage
  holds p micro-batches of L/p layers each, which is L layers' worth. Tensor parallelism divides them.
- **"Fits" is optimistic.** The calculator ignores temporary buffers, ZeRO-3 gather buffers, the CUDA
  context and memory fragmentation, which together take several GB. Real teams aim well below 80 GB.
- **The ring's per-worker traffic levels off at 2×, but its step count keeps growing.** That second column
  is why large clusters do not use one flat ring.

## When things break: failures, checkpoints and silent corruption

**In short:** at 16,000 GPUs, something fails every few hours and one failure stops the whole synchronous
job. Runs survive by checkpointing often, restarting fast, and hunting for hardware that lies.

A data pipeline with 16,000 workers retries failed tasks and moves on. Synchronous training cannot. Every
step needs every GPU, so one dead GPU, one flaky cable or one hung collective stops all of them.

[Meta's Llama 3 paper](https://arxiv.org/abs/2407.21783) reports a 54-day snapshot of pre-training on up to
16,384 H100s:

- **466 job interruptions**: 47 planned (such as upgrades) and **419 unexpected**.
- About **78%** of the unexpected ones were confirmed or suspected hardware issues. GPU issues alone were
  58.7%; the largest single causes were faulty GPUs (148, 30.1%) and GPU memory (72, 17.2%).
- **6** were silent data corruption, covered below.
- Despite this, the job spent **more than 90%** of the time training. Significant manual intervention was
  needed only three times; automation handled the rest.

That is one unexpected interruption about every 3 hours. The paper also notes softer effects: throughput
varied 1–2% with the time of day, from temperature, and the whole cluster's power draw swung by tens of
megawatts when all GPUs paused together, for example at a checkpoint.

### Checkpoint and restart

A training <Term id="checkpoint">checkpoint</Term> here means the full state needed to resume: weights,
optimizer state, the data loader's position and the random-number state. At 16 bytes per parameter, a 405B
model's checkpoint is around 6.5 TB. Each GPU writes its own shard in parallel, so it behaves like a
distributed snapshot, not one big file.

Checkpoint too rarely and each failure throws away hours of work on every GPU. Checkpoint too often and
the saving itself eats the run. HPC has a classic answer, Young's approximation: the best interval is about
the square root of 2 × (time to write a checkpoint) × (mean time between failures). Asynchronous
checkpointing shrinks the first term: copy the state to CPU memory in seconds, resume training, and write to
storage in the background.

::: info Back-of-envelope: how often to checkpoint
- Mean time between failures: 54 days / 419 ≈ **3 hours** ≈ 11,000 s.
- Training blocked for about 60 s per checkpoint.
- Young: √(2 × 60 × 11,000) ≈ 1,150 s, so checkpoint about **every 20 minutes**.
- Cost: each failure loses half an interval of work on all 16,384 GPUs, about 10 minutes per 3 hours (≈ 6%).
  Checkpointing itself takes 60 s per 20 minutes (≈ 5%). Restart time comes on top, so fast restart matters
  as much as frequent checkpoints.
:::

### Hangs and stragglers

A crash is easy to see. A hang is worse: one GPU stops answering, and 16,383 others wait inside a
collective forever. Training stacks use watchdog timeouts on collectives. Llama 3's team describes using
PyTorch's NCCL flight recorder, which keeps a ring buffer of recent collective calls per rank, to find which
rank stalled. Stragglers are the same problem in slow motion: one GPU running hot and throttling slows every
step for everyone. Teams find them by comparing per-rank timings and then drain the slow node.

### Silent data corruption

The worst failure raises no error. A defective core occasionally computes a wrong product, and the wrong
number flows into the gradient. This is <Term id="silent-data-corruption">silent data corruption</Term>.
At fleet scale it is not rare. [Meta](https://arxiv.org/abs/2102.11245) and
[Google](https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s01-hochschild.pdf) both found CPUs in
their fleets that silently miscompute; Google estimated a few such cores per several thousand machines. GPUs
are not immune: Llama 3 logged 6 cases in 54 days.

In training, a corrupted step can show up as a sudden spike in loss, NaNs, or nothing at all. Defences
include:

- **Watching the loss and the gradient's overall size** (its norm) for spikes and NaNs. When a spike appears, roll back to an earlier
  checkpoint. PaLM's team restarted about 100 steps before a spike and skipped 200–500 batches
  ([Chowdhery et al., 2022](https://arxiv.org/abs/2204.02311)).
- **Comparing replicas.** Data-parallel copies should hold bit-identical weights, so a checksum that
  disagrees points at the faulty machine.
- **Burn-in and periodic hardware tests**, and draining any node that ever fails one.

::: tip Interview checkpoint
<details><summary>A 16,000-GPU job fails every 3 hours. How do you keep it productive?</summary>

Make failure cheap rather than rare. Checkpoint asynchronously and often (every 10–30 minutes, following
Young's rule), shard checkpoints so all ranks write in parallel, and keep hot spare nodes so a restart does
not wait for a repair. Detect hangs with collective timeouts and per-rank tracing, and drain stragglers.

Senior add-on: restart time is often the real bottleneck: scheduling, loading the checkpoint, warming up
NCCL communicators and compiling. Llama 3 kept over 90% effective training time this way.
</details>

<details><summary>Training loss suddenly spikes at step 300,000. What do you do?</summary>

Roll back to a checkpoint before the spike. Check whether it reproduces with the same data. If it does, the
cause is the data or the optimizer: skip those batches or lower the learning rate. If it does not, suspect
hardware, and look for a node with errors or a replica whose weights disagree.
</details>

<details><summary>Why can't a training job just retry the failed task, like Spark does?</summary>

Spark tasks are independent, so one can be rerun alone. A training step is one tightly coupled computation
across all GPUs, joined by collectives, and every GPU's state depends on every other's. The only consistent
recovery point is a global checkpoint.
</details>
:::

::: warning Thread
One reason today's models are frozen is visible in this chapter. A pretraining run ties up thousands of GPUs
for months, and every step is a global, synchronous update. Nothing in that machinery is built to let one
deployed copy learn from its own conversation. A model you can train is a cluster-sized job; a model you
serve is a read-only file copied to many machines.

The counterargument is strong. Cost is a barrier to retraining from scratch, not to updating. Fine-tuning a
small set of extra weights takes a few GPUs and hours, and labs already ship updated models every few months.
DeepSeek reports its final V3 run at about 2.8M GPU-hours, roughly $5.6M at rental prices, though that
excludes research and failed runs. So the hard part of learning during deployment is probably not compute.
It is deciding what to learn from, without forgetting or being poisoned. See
[Continual Learning](/beyond/continual-learning).
:::

## In practice

- **Most teams never write a collective by hand.** They pick a framework (PyTorch FSDP, Megatron-LM,
  DeepSpeed, JAX with sharding annotations) and choose the parallel degrees. The skill is choosing those
  degrees and reading profiles.
- **Small and medium models (up to roughly 10–30B) often use FSDP alone** on one or a few servers. Tensor
  and pipeline parallelism earn their complexity at larger sizes or longer contexts.
- **Network design follows the parallelism.** Training clusters use "rail-optimized" fabrics, where GPU k
  of every server connects to the same switch, because data-parallel groups talk GPU k to GPU k.
- **Fine-tuning is the same problem, smaller.** Full fine-tuning of a 70B model still needs about 1.1 TB of
  model state. That is one reason adapter methods that train only a small number of extra weights are
  popular.
- **Inference uses some of the same splits** (tensor and expert parallelism), but has no gradients or
  optimizer state; its memory problem is the KV cache instead ([chapter 8](/running/inference)).

## Common misconceptions

- **"A 70B model needs 140 GB to train."** That is the 16-bit weights alone. Training needs about 16 bytes
  per parameter plus activations: over 1 TB.
- **"Adding GPUs makes ring all-reduce more expensive per GPU."** Per-GPU traffic stays near 2× the gradient
  size. Latency grows, bandwidth does not.
- **"Pipeline parallelism divides activation memory by the number of stages."** Under 1F1B the first stage
  still holds a full model's depth of activations, spread over p micro-batches.
- **"ZeRO-3 solves memory."** It shards model state, not activations, and it cannot raise the batch-size
  ceiling on data parallelism.
- **"bf16 training is all bf16."** The update happens in 32-bit master weights; the optimizer state is
  32-bit too.
- **"Hardware failures are rare."** At 16,000 GPUs, Llama 3 saw one unexpected interruption every 3 hours.

## Key takeaways

- **Training needs about 16 bytes per parameter** (16-bit weights and gradients, 32-bit master weights and
  two Adam averages) **plus activations**, which grow with tokens in flight.
- **Data parallelism splits the work; ZeRO/FSDP splits its memory.** The ring all-reduce costs each GPU about
  2× the gradient size, regardless of GPU count.
- **Tensor parallelism splits each layer and needs NVLink; pipeline parallelism splits the stack and pays a
  bubble; context and expert parallelism handle long sequences and MoE.** Place the chattiest split on the
  fastest links.
- **bf16 with 32-bit master weights is standard; fp8 is arriving.** Activation checkpointing buys memory for
  about a third more compute.
- **At scale, failure is routine.** Effective training time depends on fast checkpoint and restart, hang
  detection, and catching hardware that silently computes wrong answers.

## Further reading

- [ZeRO (Rajbhandari et al., 2019)](https://arxiv.org/abs/1910.02054): the memory accounting behind the
  16-bytes-per-parameter rule, and the three stages.
- [Megatron-LM (Shoeybi et al., 2019)](https://arxiv.org/abs/1909.08053): the column/row split for tensor
  parallelism, in a few pages.
- [Efficient Large-Scale Training on GPU Clusters (Narayanan et al., 2021)](https://arxiv.org/abs/2104.04473):
  how to combine tensor, pipeline and data parallelism, and interleaved pipelines.
- [The Llama 3 Herd of Models, section 3.3](https://arxiv.org/abs/2407.21783): the most detailed public
  account of a 16K-GPU run, including the failure table.
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437): fp8 training, DualPipe and expert
  parallelism on a smaller cluster.
- [The Ultra-Scale Playbook (Hugging Face)](https://huggingface.co/spaces/nanotron/ultrascale-playbook): an
  interactive, hands-on tour of all of this, with measurements.
- [How to Scale Your Model (Google DeepMind)](https://jax-ml.github.io/scaling-book/): the same ideas
  from the TPU side, with careful roofline arithmetic.

## Review

<Flashcards id="training-at-scale" :cards="cards" />

<MarkDone id="training-at-scale" />
