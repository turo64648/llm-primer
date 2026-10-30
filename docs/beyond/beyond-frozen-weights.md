---
title: "13. Beyond Frozen Weights"
---

<script setup>
import { cards } from './beyond-frozen-weights-review'
</script>

# 13. Beyond Frozen Weights

This chapter is a tour of the research that tries to give language models memory between "this token" and
"the last training run": fast weights, test-time training, learned memories, direct weight edits, and
generated adapters. Interviewers at AI labs ask about it because linear-attention hybrids already ship, and
because "how would you make a model learn after deployment?" is an open question they think about daily.

::: info Before you start
- A transformer turns each token into a **query**, a **key** and a **value** (three vectors). Attention
  compares the query with every earlier key and mixes their values. See [Chapter 4](/foundations/transformer).
- At inference, the keys and values of all earlier tokens are stored in the <Term id="kv-cache">KV cache</Term>, which grows with every token. See [Chapter 8](/running/inference).
- Training follows the <Term id="gradient">gradient</Term> of a loss, computed by <Term id="backpropagation">backpropagation</Term>. See [Chapter 2](/foundations/learning). <Term id="lora">LoRA</Term> adapts a model with a small add-on matrix. See [Chapter 7](/building/post-training).
- [Chapter 11](/beyond/agents-memory) covered memory outside the model (retrieval, notes). [Chapter 12](/beyond/continual-learning)
  covered why updating the weights themselves causes forgetting.
:::

## The missing middle

**In short:** a deployed LLM has two memories: activations that last one token and weights that last
months. Everything in this chapter is an attempt to add something in between, inside the model.

Recap of the last two chapters in two sentences. Agents can write notes and retrieve documents
([Chapter 11](/beyond/agents-memory)), but the model only sees them as more input text. Changing the weights
after deployment works in principle, but each update tends to erase older skills
([Chapter 12](/beyond/continual-learning)).

So a production model has <Term id="activations">activations</Term>, the numbers computed inside one forward pass,
and frozen weights. The context window sits between them, but it is input, not memory the model writes to
itself. A brain, by contrast, has synapses that change over milliseconds, minutes, days and years.

This chapter asks: can a network carry weights that change while it runs? The answer is yes, and the idea
is almost 40 years old. The hard part is making those weights hold the right things, at scale.

## Fast weights: a second set of weights that changes while you think

**In short:** fast weights are a matrix the network writes to as it runs, using outer products. Reading
it with a key returns the value stored under that key. It is a fixed-size key→value memory.

### The intuition

Picture a reader in the middle of a novel. They remember that "Anna" is the doctor and "Tom" is her brother.
That memory is not in their long-term knowledge, and it does not live only in the current sentence. It lasts
for the book, then fades.

The idea of <Term id="fast-weights">fast weights</Term> is to give a network that middle layer. The ordinary weights,
learned in training, are now called **slow weights**. On top of them sits a matrix that changes with every
token and decays over time. [Hinton and Plaut (1987)](https://www.cs.toronto.edu/~hinton/absps/fastweights87.pdf)
proposed this split; [Ba et al. (2016)](https://arxiv.org/abs/1610.06258) revived it for recurrent networks.

### The mechanism: an associative memory in a matrix

The thing we want is an <Term id="associative-memory">associative memory</Term>: store pairs like
"Anna → doctor", then hand it "Anna" and get back "doctor". Both sides are vectors.

The write uses the <Term id="outer-product">outer product</Term>. For a key vector `k` and a value vector
`v`, the outer product `v kᵀ` is a matrix whose entry in row i, column j is `v[i] · k[j]`. Add it to the
memory matrix `M`.

```
write:  M ← M + v kᵀ
read:   M · q   =  Σ over stored pairs of  v · (k · q)
```

In words: reading multiplies the matrix by a query `q`. Each stored value comes back, weighted by how much
its key matches the query (the dot product `k · q`). If the query equals one stored key, and all keys are
at right angles to each other, the dot product is 1 for that key and 0 for the rest. You get exactly its value.

This write is <Term id="hebbian-learning">Hebbian learning</Term>, after the psychologist Donald Hebb: "cells
that fire together wire together". The connection from input j to output i grows by the product of their
activities, `k[j] · v[i]`. Nothing else is needed: no error, no backward pass.

### Fast weight programmers

[Schmidhuber (1992)](https://people.idsia.ch/~juergen/fastweights/ncfastweightsrev.html) added the step that
matters for modern models. A slow network looks at each input and *decides* what key and value to write into
a fast network. He called it a fast weight programmer. The slow network is trained normally; the fast
network is programmed on the fly.

That is the whole design pattern of this chapter. Slow weights learn *how to write*. Fast weights hold
*what was written*.

### The limit: interference

The catch is that keys are not perfectly at right angles. Reading key `k₁` also returns a little of every
other value, weighted by how similar its key is to `k₁`. This leftover is called **crosstalk**.

In `d` dimensions, at most `d` vectors can be exactly at right angles. So a `d × d` memory holds about `d`
clean pairs. Past that, crosstalk grows with every write, and recall blurs. Real keys are worse than random,
because they are produced by a network and tend to resemble each other.

## Linear attention is a fast-weight memory

**In short:** drop the softmax from attention and the sum over past tokens collapses into one matrix,
updated by an outer product per token. The KV cache becomes a fixed-size fast-weight memory.

### From attention to a running sum

Standard attention for the newest token, with query `q`, reads every earlier key `kᵢ` and value `vᵢ`:

```
output = Σᵢ  weightᵢ · vᵢ        where  weightᵢ = softmax over i of (q · kᵢ)
```

In words: score each past token by how well its key matches the query, turn scores into weights that sum to
1, and average the values. To do this you must keep every key and value. That is the KV cache, and it grows
by one entry per token.

[Katharopoulos et al. (2020)](https://arxiv.org/abs/2006.16236), "Transformers are RNNs", replaced the softmax
weight with a plain dot product, after passing queries and keys through a fixed function (a "feature map"). Then the sum can be regrouped:

```
output = Σᵢ vᵢ (kᵢ · q)  =  (Σᵢ vᵢ kᵢᵀ) · q  =  S · q
S_t   = S_{t−1} + v_t k_tᵀ
```

In words: the state `S` is the sum of outer products of every value with its key. Each new token adds one
outer product. The output is `S` times the query. This is <Term id="linear-attention">linear attention</Term>,
named because its total cost grows linearly with sequence length, not quadratically.

Look at the second line again. It is the fast-weight write from the last section. [Schlag, Irie and Schmidhuber (2021)](https://arxiv.org/abs/2102.11174)
made this explicit: linear transformers *are* fast weight programmers. The slow weights produce `k`, `v`, `q`;
the state `S` is the fast weight matrix.

<BfwKvVsFastWeightDiagram />

::: details Going deeper: the feature map and the normaliser
Katharopoulos et al. apply a positive feature map φ (they use elu(x) + 1) to queries and keys, and divide by
a running sum `z_t = Σ φ(kᵢ)` so the weights still behave like an average:
`output = S φ(q) / (z · φ(q))`. Later work (RetNet, GLA, Mamba-2) mostly drops the normaliser and adds a
decay instead. The systems view: attention is a lookup in an append-only log; linear attention is a
fixed-size summary updated in place. The analogy breaks because the summary cannot be queried exactly.
:::

### The delta rule: write the correction, not the value

Plain outer-product writes have a second problem besides crosstalk. Writing the same key twice, with an
old value A and a new value B, stores `A + B`. You cannot overwrite; you can only pile on.

The <Term id="delta-rule">delta rule</Term> fixes this. Before writing, read what the memory currently
returns for the key, and write only the difference:

```
S_t = S_{t−1} + β_t · (v_t − S_{t−1} k_t) · k_tᵀ
```

In words: `S_{t−1} k_t` is what the memory currently thinks the value is. `v_t − S_{t−1} k_t` is the error.
`β_t` is a write strength between 0 and 1, chosen by the network for each token. With `β = 1` and a unit-length
key, the memory now returns exactly `v_t` for `k_t`, and the old value is gone.

Schlag et al. proposed this for linear transformers. [DeltaNet (Yang et al., 2024)](https://arxiv.org/abs/2406.06484)
showed how to train it fast on GPUs. [Gated DeltaNet (Yang, Kautz and Hatamizadeh, 2024)](https://arxiv.org/abs/2412.06464)
adds a decay gate `α_t` that shrinks the whole state before each write, so the model can also forget in bulk.

There is a deeper point here. The delta-rule update is exactly one step of gradient descent on the loss
`‖S k_t − v_t‖²`, "how badly does the memory recall this pair". The exercise checks this numerically. So
the delta rule is already a tiny case of *learning at test time*. Hold that thought for the next section.

### State-space models and the recall gap

A separate family arrived at fixed-size states from signal processing. A <Term id="state-space-model">state-space model</Term>
carries a state from token to token: `state_t = A_t · state_{t−1} + B_t · x_t`, and outputs `C_t · state_t`.
[Mamba (Gu and Dao, 2023)](https://arxiv.org/abs/2312.00752) made `A`, `B`, `C` depend on the current input,
so it can choose what to keep. [Mamba-2 (2024)](https://arxiv.org/abs/2405.21060) showed this is close to
linear attention with a decay. All these models share one property: the memory has a fixed size.

A fixed size has a price, and the price is exact recall. [Zoology (Arora et al., 2023)](https://arxiv.org/abs/2312.04927)
found most of the quality gap between such models and attention came from "associative recall": finding the
value that went with a key seen earlier. [Jelassi et al. (2024)](https://arxiv.org/abs/2402.01032) proved, and
showed in experiments, that a fixed-state model cannot copy a string longer than its state can hold, while a small
transformer can. Attention keeps every token, so it never has to compress.

The practical answer is the <Term id="hybrid-model">hybrid model</Term>: mostly fixed-state layers, plus a few
full attention layers for exact lookup. [Qwen3-Next (Sept 2025)](https://vllm.ai/blog/2025-09-11-qwen3-next)
uses Gated DeltaNet in three of every four layers. [Kimi Linear (Moonshot, Oct 2025)](https://arxiv.org/abs/2510.26692)
uses the same ratio with a refined delta-rule layer.

::: info Back-of-envelope: KV cache vs fixed state at 1 million tokens
Take a 70B-class model shaped like Llama 3 70B: 80 layers, 8 key/value heads of size 128, 16-bit numbers.

- **KV cache per token:** 2 (key and value) × 80 layers × 8 heads × 128 × 2 bytes ≈ 330 KB.
- **At 1M tokens:** 330 KB × 10⁶ ≈ **330 GB** for one conversation. That is four 80 GB GPUs of memory, for the cache alone.
- **Fixed state instead:** say every layer is linear attention with 64 heads, each a 128 × 128 matrix kept in
  32-bit floats. 80 × 64 × 16,384 × 4 bytes ≈ **335 MB**, at 1 token or 1 billion tokens.
- **Break-even:** 335 MB ÷ 330 KB ≈ 1,000 tokens. Past that, the fixed state is smaller, by about 1,000× at 1M tokens.
- **A 3:1 hybrid** keeps attention in 20 of the 80 layers: about 82 GB of cache at 1M tokens, plus about 250 MB of state.

The catch is in the capacity: 64 heads × 128 dimensions holds on the order of 8,000 clean key→value pairs per
layer, not a million.
:::

::: warning Thread
The thread says LLMs have per-token activations and months-frozen weights, "nothing in between". Linear
attention and DeltaNet are evidence against the structural version of that claim. A matrix that the network
writes to with a learned rule, token after token, is a fast weight. Production hybrids ship it today.

The evidence for the thread is what those fast weights are used for. The state is wiped at the end of each
sequence. The write rule is frozen. Nothing learned in conversation one is available in conversation two.
So this is a working memory, closer to "remembering the plot of this book" than to learning.
:::

::: tip Interview checkpoint
<details><summary>Why can linear attention run in constant memory per token while softmax attention cannot?</summary>

Without the softmax, the output is `(Σ vᵢ kᵢᵀ) q`, and that sum can be kept as one running matrix. The
softmax normalises over all past scores together, so it does not factor this way; you need every key.
</details>
<details><summary>What is the capacity of a d × d linear associative memory, and why?</summary>

About `d` clean pairs. Clean recall needs keys at right angles, and at most `d` vectors can be at right
angles in `d` dimensions. Beyond that, crosstalk grows with each write.
</details>
<details><summary>What does the delta rule add over linear attention?</summary>

It writes the error `v − S k` instead of `v`. A repeated key overwrites its old value, and similar keys
interfere less. It equals one gradient step on the recall loss `‖S k − v‖²`.
</details>
<details><summary>Why do Qwen3-Next and Kimi Linear keep some full attention layers?</summary>

Fixed-state layers lose information when compressing long inputs, so they are weak at exact recall and
copying. A few full attention layers restore exact lookup while most layers stay cheap.
</details>
:::

## Test-time training: the hidden state is a model

**In short:** instead of a matrix updated by a fixed rule, make the memory a small neural network and train
it with real gradient steps on each new token. The outer model learns how to set up that training.

### The idea, from images to sequences

[Sun et al. (2020)](https://arxiv.org/abs/1909.13231) started with images. A classifier faces a test image
from a shifted distribution, say blurred. Before classifying it, the model takes a gradient step on a
made-up task on that one image: rotate it, then predict the rotation. The answer comes from the image itself, so no
label is needed; this is called a self-supervised task. The shared
features adapt to the blur. This is <Term id="test-time-training">test-time training</Term> (TTT).

[Sun et al. (2024), "Learning to (Learn at Test Time)"](https://arxiv.org/abs/2407.04620) moved the idea inside
a sequence model. In a TTT layer the hidden state is the weights `W` of a small model, either a linear map or
a two-layer <Term id="mlp">MLP</Term>. Each token is one training example for it.

### The mechanism

For each token `x_t`, the layer makes a key, a value and a query from it with slow weights (`θ_K`, `θ_V`,
`θ_Q`). Then:

```
loss_t = ‖ f_W(θ_K x_t) − θ_V x_t ‖²        "how badly does the inner model map key to value?"
W_t    = W_{t−1} − η ∇loss_t               one gradient step
output = f_{W_t}(θ_Q x_t)
```

In words: the inner model `f_W` tries to predict the value from the key. It takes one gradient step to do
better on this token. Then it is applied to the query. If `f_W` is linear, this is the delta rule again.
If it is an MLP, the memory can store more complex mappings in the same size.

There are two loops. The inner loop updates `W` at test time. The outer loop is normal training: it learns
`θ_K`, `θ_V`, `θ_Q`, the learning rate and the starting `W`, so that the inner loop's learning helps predict
the next token. The model *learns how to learn* from its context.

[Wang, Shi and Fox (2025)](https://arxiv.org/abs/2501.12352) show that this one frame covers the whole
family. Linear attention, DeltaNet, Mamba-style decays and TTT all fit it.
They are all *online regression*: learning a key→value mapping one pair at a time. They differ in the loss,
the inner model and the optimiser.

**Status.** The TTT paper trained models from 125M to 1.3B parameters. TTT-MLP kept improving with longer
context where Mamba plateaued, but its inner loop was slow in practice. It is a research architecture, not a
production one.

## Titans: a neural long-term memory driven by surprise

**In short:** Titans (Google, 2025) adds an MLP memory next to attention. It writes in proportion to how
surprised it is, keeps momentum so that what follows a surprise is also stored, and forgets with a learned gate.

### The intuition

You do not remember your commute. You do remember the day a deer ran across the road, and also the next few
minutes. [Titans (Behrouz, Zhong and Mirrokni)](https://arxiv.org/abs/2501.00663) builds that into a memory.

Attention handles the short term: exact, but limited to a window. A <Term id="titans">Titans</Term> neural
memory handles the long term: a deep MLP that compresses the past into its weights. Some fixed, learned
"persistent memory" tokens hold task knowledge that never changes.

### The mechanism

The memory `M` is an MLP. For each token it makes a key and value and measures the loss `‖M(k_t) − v_t‖²`.
The gradient of that loss is the <Term id="surprise-signal">surprise</Term>: large when the memory predicted badly.

```
S_t = η_t · S_{t−1}  −  θ_t · ∇loss(M_{t−1}; x_t)     surprise with momentum
M_t = (1 − α_t) · M_{t−1}  +  S_t                      forget a little, then write
```

In words: `S_t` is a running surprise. It carries part of the last step's surprise (`η_t`, like the <Term id="momentum">momentum</Term> used in training) and adds the new gradient (`θ_t`, like a learning rate).
`α_t` is a forget gate: it shrinks the memory a little before each write, like <Term id="weight-decay">weight decay</Term>.
All three numbers are computed from the current token, so the model chooses how much to learn and forget.

The paper offers three ways to wire memory and attention: memory output as extra context for attention,
memory and attention mixed by a gate, or memory as its own layer. Training is parallelised by computing
gradients over chunks of tokens at once.

**Status.** The paper reports better language modelling than transformers and linear-recurrent baselines at
its scale (models under 1B parameters), and strong results on long-context recall benchmarks beyond 2M
tokens. These are the authors' numbers. Whether it beats strong hybrids at frontier scale is not established
in public.

### Follow-ups (2025–2026) {#follow-ups-2025-2026}

The same group has published a fast series. [*It's All Connected* (Miras)](https://arxiv.org/abs/2504.13173)
recasts these models as choices of loss, retention (forgetting) and optimiser, matching Wang et al.'s frame.
[ATLAS](https://arxiv.org/abs/2505.23735) optimises the memory over a window of recent tokens, not only the
current one, and reports large gains on 10M-token recall. [*Nested Learning*](https://arxiv.org/abs/2512.24695)
views a whole model, including its optimiser, as nested learning problems running at different speeds. Its
"Hope" model adds a self-modifying memory and a "continuum memory system" with many update rates.

The newest, [*Language Models Need Sleep* (2026)](https://arxiv.org/abs/2606.03979), proposes an offline phase
that moves in-context knowledge into durable weights by distillation (training one model to imitate another) and replays self-generated data. That is
a direct attempt at consolidation ([Chapter 12](/beyond/continual-learning)). It is a recent preprint; treat its
claims as unreplicated.

::: details Going deeper: why momentum and forgetting matter
Without momentum, the memory stores only the single surprising token and ignores the tokens that follow,
which are often the useful part ("the deer ran across the road *and hit the car in front*"). Momentum keeps
the write going for a few tokens after the surprise. Without forgetting, a fixed-size memory fills with
crosstalk, as with Hebbian writes. The gate lets the model clear space, at the cost of losing old content.
:::

::: warning Thread
Titans-style memory is a real "in between" timescale. It changes weights during use, it is driven by a
gradient, and it can hold millions of tokens' worth of compressed past. That is evidence against the claim
that the LLM design *structurally* lacks a middle timescale.

But look at the signal. Surprise means "I did not predict this", not "this is true" or "I was wrong to
believe that". Trick a Titans model once, and it stores the trick as a surprising fact. Nothing in it
learns *not to be tricked*. The rules for writing are frozen slow weights, and the memory is normally reset
between sequences. It is learning to remember, not learning to judge.
:::

## Model editing: change one fact by surgery

**In short:** ROME finds where a fact lives in a middle-layer MLP and changes it with one outer-product
update. MEMIT does thousands at once. Edits work on the exact prompt but often fail to spread to related facts.

### The intuition: MLPs as key→value tables

[Geva et al. (2021)](https://arxiv.org/abs/2012.14913) found that a transformer's MLP layers behave like
key→value memories. The first weight matrix detects input patterns (keys). The second writes out vectors
(values) that push the prediction towards certain tokens. This is the same associative memory as above, but
written once in training and then frozen.

If that is true, you should be able to change one fact by changing one entry in that table. That is the
idea behind <Term id="model-editing">model editing</Term>.

### ROME: locate, then write a rank-one update

[ROME (Meng et al., 2022)](https://arxiv.org/abs/2202.05262) has two steps. First, a method called <Term id="causal-tracing">causal tracing</Term> finds where the fact is recalled. Add noise to the subject
tokens ("The Eiffel Tower") so the model forgets the answer. Then restore one hidden state at a time from the
clean run and see which restores "Paris". The answer: MLPs in middle layers, at the last subject token.

Second, edit. Take the MLP input at that position as the key `k*`. Find a value `v*` that makes the model
say "Rome". Then update the MLP's output matrix `W`:

```
W_new = W + Λ (C⁻¹ k*)ᵀ        where  Λ is chosen so that  W_new k* = v*
```

In words: add one outer product, so the matrix changes in only one direction (a "rank-one" update). `C` is
a statistic of the keys the MLP normally sees. Dividing by it aims the change at `k*` and away from other
common keys, so other facts are disturbed as little as possible. <Term id="rome">ROME</Term> is the fast-weight
write, done once, by an outside editor, on the slow weights.

[MEMIT (Meng et al., 2022)](https://arxiv.org/abs/2210.07229) solves for many facts together and spreads the
change over several layers. <Term id="memit">MEMIT</Term> edited thousands of facts in GPT-J (6B) in one batch.

### The limits

The edits pass the test they were built for: the edited prompt now gives the new answer. Critiques found
problems around it.

- **Ripple effects.** If Messi now plays for a new club, his new teammates should change too.
  [Cohen et al. (2023)](https://arxiv.org/abs/2307.12976) found edited models often miss these
  <Term id="ripple-effect">ripple effects</Term>. [MQuAKE (Zhong et al., 2023)](https://arxiv.org/abs/2305.14795)
  found the same for multi-step questions.
- **Location is murky.** [Hase et al. (2023)](https://arxiv.org/abs/2301.04213) found that where causal
  tracing points barely predicts which layer is best to edit.
- **Edits accumulate damage.** [Gupta et al. (2024)](https://arxiv.org/abs/2401.07453) applied edits one after
  another. Models gradually lost other abilities, then collapsed suddenly: catastrophic forgetting again.

**Status.** Model editing is a useful tool for interpretability research. It is not how labs fix facts in
production; they use retrieval or fine-tuning.

## Hypernetworks and self-editing: models that write weights

**In short:** a hypernetwork outputs weights for another network. Text-to-LoRA and Doc-to-LoRA turn a task
description or a document into a LoRA adapter in one forward pass. SEAL trains a model to write its own
fine-tuning data.

### Hypernetworks

A <Term id="hypernetwork">hypernetwork</Term> is a network whose output is the weights of another network
([Ha, Dai and Le, 2016](https://arxiv.org/abs/1609.09106)). It is the fast weight programmer idea again, at a
coarser timescale: instead of a write per token, one set of weights per task or document.

[Text-to-LoRA (Sakana AI, 2025)](https://arxiv.org/abs/2506.06105) takes a written task description and outputs
a LoRA adapter for an LLM. It is trained to reproduce hundreds of existing task-specific adapters, or trained
directly on many tasks. The authors report it matches task-tuned adapters on many benchmarks, and produces
adapters for unseen tasks, in one cheap step instead of a fine-tuning run.

[Doc-to-LoRA (2026)](https://arxiv.org/abs/2602.15902) does the same for a document. It turns a long context
into an adapter, so later questions do not need the document in the KV cache. The authors report near-perfect
needle-in-a-haystack recall (finding one planted fact in a long text) at over 4× the model's native context length.

### SEAL: the model writes its own training data

[SEAL (Zweiger et al., MIT, 2025)](https://arxiv.org/abs/2506.10943) asks the model itself to decide how to
learn. Given a new passage, the model writes a "self-edit": restated facts, implications, even training
settings. The model is fine-tuned on that self-edit (with LoRA), then tested on questions about the passage.

The score after the update is a reward. Reinforcement learning trains the model to write self-edits that
lead to better scores. On a knowledge-incorporation task, accuracy on questions about a passage rose from
32.7% with no adaptation to 47.0%, beating fine-tuning on the raw passage.

**Status.** These are small-scale research results. SEAL's authors report that repeated self-edits still
cause forgetting of earlier ones, and each reward needs a full fine-tune and evaluation, which is slow.

::: warning Thread
SEAL is the closest thing here to the thread's "learn from being tricked once": the model decides what to
learn and changes its weights. But its reward comes from a downstream test with known answers. In the hard
case, where nobody can check the answer, there is no reward, and the method has nothing to learn from.

Generated adapters do fit the thread's per-user question well. A LoRA per user or per document is small,
cheap to store, and leaves the shared model untouched, so one user's updates cannot damage another's.
Serving systems already load thousands of LoRAs on one base model (for example [S-LoRA](https://arxiv.org/abs/2311.03285)).
The open question is what writes them, and from what signal.
:::

## Local learning rules: can learning happen without backprop?

**In short:** brains appear to learn with local rules at many timescales. Backpropagation is global and
needs stored activations, which makes it awkward to run continuously. Local alternatives exist but lag far
behind at scale.

### Why backprop is hard to do online

Backprop answers <Term id="credit-assignment">credit assignment</Term>: which of billions of weights caused
this error, and by how much? It does so exactly, but at a price:

- It needs the stored activations of every layer from the forward pass, until the error is known.
- It needs a separate backward pass that uses the same weights as the forward pass, mirrored.
- It needs the outcome first. For a conversation, the outcome may arrive minutes later, or never.
- One gradient step on one example tends to overwrite older knowledge ([Chapter 12](/beyond/continual-learning)).

For a model serving millions of users, doing this per token is costly. It also means each update depends on
the whole network at once.

### Local rules

A local rule updates each connection using only what happens at its two ends. The Hebbian outer product is
the classic example; the delta rule adds one locally available error. Every fast-weight layer in this chapter
uses a local rule *inside* a network trained by backprop. That combination is what works today.

Fully local training of a deep network is harder. [Feedback alignment (Lillicrap et al., 2016)](https://www.nature.com/articles/ncomms13276)
showed that random backward weights can still train small networks, removing the need for mirrored weights.
Hinton's [forward-forward algorithm (2022)](https://arxiv.org/abs/2212.13345) drops the backward pass
entirely. Each layer learns to give high "goodness" to real data and low goodness to fake data, using two
forward passes. The <Term id="forward-forward">forward-forward algorithm</Term> works on small image tasks, but
no local rule has come close to backprop on large language models.

### Brains vs artificial networks (speculative)

Synapses change on many timescales: milliseconds (short-term facilitation), hours to years (long-term
potentiation), and days to years as memories move between brain regions ([Chapter 12](/beyond/continual-learning)).
Many neuroscientists argue the brain approximates gradient-like credit assignment by local means
([Lillicrap et al., 2020](https://www.nature.com/articles/s41583-020-0277-3)); this is contested.

Mapping these to fast weights, adapters and slow weights is tempting. It is speculative. The analogy breaks
in at least one way: brains seem to learn *what matters* from reward, emotion and social feedback, not only
from prediction surprise. No method here has that signal.

## Putting it together: the timescale map

**In short:** the research now covers every timescale between a token and a training run. What it does not
yet cover is a trustworthy signal for what to learn, and memory that persists safely across users and sessions.

<BfwTimescaleDiagram />

| Approach | Timescale | Persists across sessions? | Learning signal | Forgetting | Per-user? |
| --- | --- | --- | --- | --- | --- |
| Long context, RAG | one conversation | only via notes | none (input text) | none | yes |
| Linear attention, DeltaNet, Mamba | one sequence | no | fixed rule / recall error | decay gate | yes, trivially |
| TTT, Titans, Atlas | one long sequence | no, by default | self-supervised surprise | forget gate | yes, trivially |
| ROME, MEMIT | until removed | yes | an editor's target fact | grows with many edits | possible |
| Text-to-LoRA, Doc-to-LoRA | until removed | yes | hypernetwork, one pass | none within adapter | yes |
| SEAL | until removed | yes | RL on checkable tests | still forgets | possible |
| Retraining | months | yes | big curated dataset | handled by data mix | no |

Against the thread's hard parts:

- **Signal at inference time.** Solved only for self-supervised prediction (Titans) or checkable tasks (SEAL).
  Where answers cannot be checked, nothing here knows whether a new memory is true.
- **Forgetting.** Fixed-state memories forget by design, through decay gates. Persistent edits and adapters
  still interfere. Consolidation work ("sleep") is early (speculative).
- **Per-user vs shared.** Adapters and per-sequence memories give per-user state cheaply. Merging what many
  users teach into the shared model is unsolved.
- **Safety.** A memory that writes from surprise can be poisoned by whoever controls the input. A model that
  edits itself can drift away from what was tested. None of the papers above evaluate this in depth.

## Try it: a fast-weight memory

The exercise builds a 64 × 64 matrix memory. It stores random key→value pairs with Hebbian outer-product
writes and with the delta rule, and measures recall: the share of reads whose output is closest to the right
value. It then checks that a delta-rule write equals one gradient step, and runs a tiny Titans-style MLP
memory that learns from its own surprise.

<<< @/../exercises/ch13_fast_weights.py

Output:

```
Recall accuracy (%) from a 64x64 memory, random keys
 pairs N | Hebbian | delta, 1 pass | delta, 5 passes
      16 |     100 |           100 |             100
      32 |     100 |           100 |             100
      64 |     100 |            92 |             100
     128 |      95 |            63 |              71
     256 |      70 |            25 |              26

Recall accuracy (%) from a 64x64 memory, similar keys (half shared direction)
 pairs N | Hebbian | delta, 1 pass | delta, 5 passes
      16 |      75 |            69 |             100
      32 |      16 |            38 |             100
      64 |       9 |            20 |              88
     128 |       2 |            20 |              55
     256 |       2 |            10 |              20

Overwrite test: write key k with value A, then 20 other pairs, then k again with value B
  Hebbian: similarity of read to old A = +0.78, to new B = +0.49
    delta: similarity of read to old A = -0.08, to new B = +1.00

Gradient check: one SGD step (lr 0.5) on ||M k - v||^2 equals one delta-rule write (beta 1)
  max difference: 2.2e-16

Part 3: an MLP memory learning at test time from its own surprise (Titans-style, tiny)
  round 1 over the 32 old pairs: mean surprise 1.15
  round 2 over the 32 old pairs: mean surprise 0.96
  round 3 over the 32 old pairs: mean surprise 0.83
  8 never-seen pairs:              mean surprise 1.34
  final recall, old pairs: mean similarity to true value 0.49
  final recall, new pairs: mean similarity to true value 0.58
```

How to read it:

- **Random keys, Hebbian.** Recall stays perfect while pairs are well below the dimension (64), and falls
  once they exceed it: 70% at 256 pairs. This metric only asks for the *closest* value, so it is forgiving;
  the read itself is already blurred.
- **Similar keys.** Hebbian recall collapses: 16% at 32 pairs. Crosstalk from a shared direction swamps
  every read. The delta rule, given 5 passes, recalls 100% up to 32 pairs and 88% at 64. It subtracts what is
  already stored, so the shared part stops piling up.
- **One pass of the delta rule** is worse than Hebbian on random keys past 64 pairs. It favours recent
  writes and overwrites old ones. A real model learns `β` per token to decide when that is worth it.
- **Overwrite.** Hebbian mixes old and new values (0.78 vs 0.49). The delta rule returns the new value
  exactly (1.00) and keeps nothing of the old.
- **Surprise.** The MLP memory's surprise drops each time it sees the same pairs again, and jumps for
  new pairs. Its final recall favours the newest pairs: test-time learning brings its own small forgetting.

## In practice

- **Hybrids ship.** Qwen3-Next and Kimi Linear use delta-rule layers for three quarters of their depth. This
  is fast-weight memory in production, chosen for long-context cost, not for learning.
- **Context still beats memory for exact facts.** Teams that need precise recall over long documents keep
  full attention or use retrieval ([Chapter 11](/beyond/agents-memory)).
- **Editing is a research tool.** ROME and MEMIT are used to study where knowledge lives, rarely to patch
  deployed models.
- **Per-user adapters are cheap to serve.** Multi-LoRA serving makes "one adapter per customer" practical. The
  open part is producing good adapters without a fine-tuning job; Text-to-LoRA and Doc-to-LoRA are early answers.

::: tip Interview checkpoint
<details><summary>How is test-time training different from fine-tuning?</summary>

Fine-tuning updates the model's main weights offline on a dataset, and the result persists. Test-time
training updates a small inner model on the input being processed, with a self-supervised loss, and usually
discards it after the sequence. An outer training loop teaches the model how to do that inner learning well.
</details>
<details><summary>Explain Titans' update rule in plain words.</summary>

The memory is an MLP. For each token it checks how badly it maps key to value; the gradient of that error
is the surprise. A running surprise with momentum is added to the memory, after shrinking the memory a little
with a forget gate. All the rates depend on the token.
</details>
<details><summary>How does ROME edit a fact, and why might it fail?</summary>

It locates the fact in a middle-layer MLP by causal tracing, then adds a rank-one update so the subject's
key maps to a new value. It can miss consequences of the fact (ripple effects), and many edits in a row
gradually damage the model.
</details>
<details><summary>If you had to let a deployed model learn from users, which of these would you use?</summary>

A reasonable answer: per-user adapters or per-session memory, so one user cannot corrupt others; retrieval
for exact facts; periodic retraining of the shared model on vetted data. Name the gaps: no reliable signal
for uncheckable answers, poisoning risk, and no safe merge of user updates into the shared model.
</details>
:::

## Common misconceptions

- "Linear attention is an approximation of attention." It is a different memory: a fixed-size summary with
  a capacity limit, not a cheaper version of the same lookup.
- "Titans learns permanently from what you tell it." Its memory is written during a sequence and reset by
  default. Its learning rules are frozen.
- "Model editing cleanly replaces a fact." It changes the answer to the edited prompt; related facts often
  stay old, and many edits harm the model.
- "Brains learn with Hebbian rules, so we should too." Brains also have reward, neuromodulation and
  consolidation, and how they assign credit is unresolved (speculative).
- "Fixed-state models have unlimited context." They accept unlimited input, but they keep only what fits in
  the state.

## Key takeaways

- Fast weights are a matrix memory written by outer products while the network runs. Linear attention is one.
- A `d × d` memory holds about `d` clean pairs. The delta rule writes only the error, so it can overwrite and
  copes better with similar keys. It is one gradient step on recall loss.
- TTT and Titans make the memory a small network trained by gradients at test time, driven by surprise, with
  momentum and forgetting.
- ROME and MEMIT edit facts in MLPs directly but miss ripple effects and degrade with many edits; hypernetworks
  and SEAL generate weight updates.
- Every timescale now has a mechanism. Still missing: a signal for what is *true*, safe persistence, and merging
  per-user learning into a shared model.

## Further reading

- [Schlag, Irie and Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers* (2021)](https://arxiv.org/abs/2102.11174):
  the paper that connects attention, fast weights and the delta rule.
- [Songlin Yang's DeltaNet blog series](https://sustcsonglin.github.io/blog/2024/deltanet-1/): the clearest
  explanation of delta-rule layers and how to train them fast.
- [Wang, Shi and Fox, *Test-time regression* (2025)](https://arxiv.org/abs/2501.12352): one framework for linear
  attention, SSMs, TTT and Titans.
- [Sun et al., *Learning to (Learn at Test Time)* (2024)](https://arxiv.org/abs/2407.04620): TTT layers, with a
  good discussion of the inner and outer loops.
- [Behrouz et al., *Titans* (2025)](https://arxiv.org/abs/2501.00663): surprise-driven neural memory; read sections 3 and 4.
- [Meng et al., *ROME* (2022)](https://arxiv.org/abs/2202.05262) and [Cohen et al., *ripple effects* (2023)](https://arxiv.org/abs/2307.12976):
  the method and its sharpest critique.
- [Zweiger et al., *SEAL* (2025)](https://arxiv.org/abs/2506.10943): a model that learns how to update itself, with honest limits.

## Review

<Flashcards id="beyond-frozen-weights" :cards="cards" />

<MarkDone id="beyond-frozen-weights" />
