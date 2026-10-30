---
title: "4. The Transformer"
---

<script setup>
import { cards } from './transformer-review'
</script>

# 4. The Transformer

The transformer is the network inside every current LLM: a stack of identical blocks that let each token read
from the tokens before it. Interviewers expect you to derive it on a whiteboard: attention, the √d, the mask,
the parameter count, and why each piece is there. This chapter ends with a small GPT you train on a laptop.

::: info Before you start
- A layer computes h = f(Wx + b); training nudges every parameter downhill on a loss using gradients
  ([Chapter 1](/foundations/neural-nets), [Chapter 2](/foundations/learning)).
- Text becomes a list of token ids; each id selects one row of an embedding table, a vector of width d. At
  the end, an unembedding turns a vector into one score (logit) per vocabulary entry, and softmax turns scores
  into probabilities ([Chapter 3](/foundations/tokens-embeddings)).
- The bigram model of Chapter 3 predicted the next token from the current token alone. This chapter fixes
  that.

The chapter explains what it needs as it goes.
:::

## The job: let tokens use their context

**In short:** to predict the next token, a model must combine information from many earlier tokens. The
transformer is a way to do that which trains fast on GPUs.

Chapter 3 ended with a bigram model. It turned each token into a vector and predicted the next token from that
vector alone. After "the" it guessed the same thing every time, whatever came before. Real text needs more:
in "The keys to the cabinet ___", the verb should be "are", because of "keys" four tokens back.

So we need a layer that lets each token's vector take in information from other tokens. Everything else in
the model can work on one token at a time, with ordinary layers from Chapter 1.
The <Term id="transformer">transformer</Term> (Vaswani et al., 2017) supplies that mixing layer, called attention, and wraps it
in a few pieces that make deep stacks trainable.

This book covers the decoder-only form used by GPT, Llama, Claude and other LLMs. It reads tokens left to
right and predicts the next one at every position.

## Why not recurrent networks?

**In short:** the earlier answer, recurrent networks, read one token at a time and carry one fixed-size
state. That is slow to train and forgets.

Before 2017 the standard tool for text was the <Term id="rnn">recurrent neural network (RNN)</Term>. It reads token 1 and
produces a state vector. It reads token 2 together with that state and produces a new state, and so on. The
state is a running summary of everything so far, and the prediction comes from it.

This has two problems:

- **It is sequential.** Step 500 needs the state from step 499. Training on a 1,000-token text takes 1,000
  steps in a row, which leaves most of a GPU idle. GPUs are fast only when they do many independent
  multiplications at once.
- **It is a bottleneck.** The whole past must fit into one vector of fixed size. A detail from 800 tokens
  back must survive 800 updates. In practice it fades, and gradients across that distance fade too
  ([vanishing gradients](/foundations/learning)).

Attention removes both problems. Every token can read every earlier token directly, so distance does not
matter. And the computation for all positions happens in parallel, as large matrix multiplies.

The trade is cost. An RNN does a fixed amount of work per token. Attention compares each token with all
earlier ones, so the work grows with the length of the text. Research on recurrent-style models never stopped
for this reason; [Chapter 13](/beyond/beyond-frozen-weights#linear-attention-is-a-fast-weight-memory) covers
the modern versions.

## Attention: the intuition

**In short:** each token asks a question, every earlier token advertises what it holds, and the token takes a
blend of the information whose advertisement best matches its question.

Take "The keys to the cabinet are". When the model processes "are", it wants to know: what is the subject, and
is it plural? Somewhere earlier, "keys" can say: I am a plural noun that could be a subject. The token "are"
should find "keys", and copy some of its information into its own vector.

The layer that does this is <Term id="attention">attention</Term>. It uses three vectors per token:

- a **query**: what this token is looking for ("a subject noun");
- a **key**: what this token offers, the label others match against ("plural noun here");
- a **value**: the information it hands over if someone attends to it.

Each token compares its query with the key of every earlier token. Good matches get large weights, poor
matches small ones. The token's output is the weighted average of the values. These are
the <Term id="query-key-value">queries, keys and values (Q, K, V)</Term>.

To a systems engineer this looks like a key-value store. A query comes in, it is compared with stored keys,
and you get back values. The analogy fits well, with three differences:

- **The lookup is soft.** A hash map returns the one value whose key matches exactly. Attention returns a
  blend of all values, weighted by how well each key matches. The blend is what makes it trainable: small
  changes to a query change the weights a little, so gradients exist.
- **Nothing is stored ahead of time.** The keys and values are computed from the current text on the fly.
  The "store" exists for one input and is gone afterwards. What is learned and kept are the three matrices
  that turn a token's vector into its query, key and value.
- **Every token is both client and entry.** Each token issues a query and also serves as an entry for later
  tokens. Because they all read from the same sequence, this is called <Term id="self-attention">self-attention</Term>.

## Attention: the mechanism

**In short:** project each token vector into a query, key and value; score each query against earlier keys
with a dot product; scale by √d, mask the future, softmax into weights, and average the values.

### Step by step, in words

Each token enters the layer as a vector x of width d (128 in this chapter's exercise, about 4,000 to 16,000 in
large models). The layer has three learned d × d matrices, W_Q, W_K and W_V. For each token it computes:

```text
q = W_Q · x      what I am looking for
k = W_K · x      what I can be found by
v = W_V · x      what I hand over
```

For token i, the score against an earlier token j is the dot product of i's query with j's key. A dot product
is large when two vectors point the same way ([Chapter 0](/start/math-toolkit)), so it measures how well
they match:

```text
score(i, j) = (q_i · k_j) / √d
```

The division by √d is explained below. Next, softmax turns token i's scores into weights that are all positive
and sum to 1. Higher scores get much more weight, because softmax exponentiates:

```text
weight(i, j) = exp(score(i, j)) / Σ_j' exp(score(i, j'))      sum over j' = 1..i
```

Finally, the output for token i is the weighted sum of the values:

```text
out_i = Σ_j weight(i, j) · v_j
```

A last matrix, W_O, maps the result back into the model's working vector. That is the whole layer: four
matrix multiplies, one softmax.

### Why divide by √d?

Suppose the numbers in q and k start out random, each with spread (standard deviation) about 1, as they are
at the start of training. A dot product adds up d such products. A sum of d random terms has a spread of
about √d. With d = 128, raw scores would typically be around ±11.

Softmax on numbers that large is almost one-hot: one weight near 1, the rest near 0. In that region softmax is
flat, so its gradients are close to zero and the layer barely learns. Dividing by √d brings the spread of the
scores back to about 1, where softmax gives a smooth blend and useful gradients.

::: details Going deeper: the variance argument
If q and k have independent entries with mean 0 and variance 1, then q · k = Σ q_m k_m is a sum of d terms,
each with mean 0 and variance 1. Variances of independent terms add, so Var(q · k) = d and the standard
deviation is √d. Dividing by √d gives variance 1. The argument is about initialisation; trained models can
and do learn larger scores when they need sharp attention. Some current models also normalise q and k
(QK-norm) to stop scores growing too large during training.
:::

### The causal mask

During generation the model cannot see tokens it has not produced yet. Training must match: when the model
predicts token 6, it must not peek at token 6. Otherwise it would learn to copy the answer, and fail at
generation time.

The <Term id="causal-mask">causal mask</Term> enforces this. Before the softmax, every score for a later token (j > i) is set
to minus infinity. Since exp(−∞) = 0, those weights are exactly 0, and the rest still sum to 1.

The mask also makes training efficient. A single pass over a 128-token window gives 128 predictions at once,
each using only its own past. That is 128 training examples for the price of one forward pass.

### The attention matrix and its cost

Write all the weights for a sequence as a grid: row i holds token i's weights over tokens 1..i. This
n × n grid is the **attention matrix**. With the mask, everything above the diagonal is 0.

<TfAttentionDiagram />

The grid shows the cost. For n tokens there are about n² scores, each a dot product of length d. Doubling the
context length doubles the work for the rest of the model, but quadruples the work for attention. This
quadratic cost is why long context is expensive, and why so much engineering goes into attention kernels.

::: details Going deeper: the whole layer as matrix multiplies
Stack the n token vectors as rows of an n × d matrix X. Then Q = X·W_Q, K = X·W_K and V = X·W_V are each
n × d. The score matrix is S = Q·Kᵀ / √d, of shape n × n. Add the mask M (0 on and below the diagonal, −∞
above), apply softmax to each row, and multiply by V:

```text
Attention(Q, K, V) = softmax(Q·Kᵀ / √d + M) · V        shape n × d
```

This is the formula from *Attention Is All You Need*. Everything is a large matrix multiply, which is what
GPUs do best. Kernels such as <Term id="flash-attention">FlashAttention</Term> compute it in tiles without ever storing the full
n × n matrix ([Chapter 8](/running/inference#flashattention-attention-without-the-big-matrix)).
:::

## Multi-head attention

**In short:** split the width into several smaller heads, each running its own attention, so one layer can
follow several relations at once.

One attention computation produces one set of weights per token: one blend. But a token often needs several
things at once. "are" wants the subject; it may also want the previous token, or the tense of the sentence. A
single softmax tends to commit to one of these.

So the layer runs several attentions in parallel, each called an <Term id="attention-head">attention head</Term>. Each head has
its own query, key and value projections, of a smaller width. Their outputs are joined side by side and mixed
by W_O. This is <Term id="multi-head-attention">multi-head attention</Term>.

The shapes, with h heads and model width d:

```text
head width          d_h = d / h               e.g. 128 / 4 = 32 in the exercise, 4096 / 32 = 128 in an 8B model
q, k, v per head    n × d_h
scores per head     n × n
joined output       n × (h · d_h) = n × d     then multiplied by W_O (d × d)
```

The cost is about the same as one big head, because the head widths add up to d. Trained heads do specialise.
Some attend to the previous token, some to matching brackets, some copy earlier patterns (<Term id="induction-head">induction heads</Term>,
[Chapter 10](/steering/interp-alignment#induction-heads)). Many heads can be removed with little loss,
so not every head does something distinct.

## The MLP: per-token processing

**In short:** after attention mixes information between tokens, a two-layer network processes each token on
its own. It holds most of the parameters.

Attention only moves and blends information. The weighted average of values is a linear operation on them.
The model also needs to compute with what it gathered: combine facts, apply rules. That is the job of
the <Term id="mlp">MLP</Term> (multilayer perceptron) from Chapter 1, applied to each token separately:

```text
MLP(x) = W_2 · GELU(W_1 · x)        W_1: 4d × d,  W_2: d × 4d
```

W_1 widens the vector four times, <Term id="gelu">GELU</Term> (a smooth version of ReLU) bends it, and W_2 brings it back to
width d. The same weights are used for every token; nothing is shared across tokens here.

There is a useful way to read this. Each row of W_1 acts like a key: it fires when the input matches a
pattern. The matching column of W_2 is a value, added to the output when its key fires. Geva et al. (2021)
[found](https://arxiv.org/abs/2012.14913) many such keys that respond to recognisable text patterns. Much of
a model's stored knowledge seems to live in these MLPs. This is why one line of research edits facts by
changing MLP weights ([Chapter 13](/beyond/beyond-frozen-weights#model-editing-change-one-fact-by-surgery)).

So a transformer has two key-value memories with very different lifetimes. Attention's keys and values come
from the current text and vanish after it. The MLP's keys and values are weights, fixed after training.

## The residual stream

**In short:** each block adds its output to a running vector instead of replacing it. That gives gradients a
clean path through deep stacks, and gives layers a shared channel.

Each sublayer does not overwrite the token's vector. It reads the vector, computes something, and adds the
result back:

```text
x = x + Attention(x)
x = x + MLP(x)
```

The running vector x is called the <Term id="residual-stream">residual stream</Term>. It starts as the token's embedding and
ends as the input to the unembedding. Every attention and MLP sublayer reads from it and writes into it.

Two things follow:

- **Training works at depth.** During backpropagation, the "+ x" passes gradients straight down to earlier
  layers. Without it, gradients pass through every sublayer and shrink or blow up on the way. Residual
  connections came from image networks (ResNets, 2015) for this reason.
- **Layers communicate through a shared space.** A head in layer 2 can write a fact, and an MLP in layer 20
  can read it, without the layers between needing to pass it on. Interpretability research reads models
  this way ([Chapter 10](/steering/interp-alignment#the-residual-stream-a-shared-bus)).

A systems analogy: a shared bus that each module reads and writes. It breaks in one place. Writes add rather
than replace, so the stream holds a sum of everything written, and later layers must pull out the parts they
need.

## Normalisation

**In short:** before each sublayer, rescale the token vector to a standard size. This keeps deep stacks
numerically stable.

Each block adds to the residual stream, so the vectors tend to grow as they pass through many layers. Their
scale also shifts during training as weights change. A sublayer that sees inputs of size 1 at one step and 50
at another has a hard time learning.

The fix is <Term id="layer-norm">layer normalisation (LayerNorm)</Term>. For each token, it shifts the vector to mean 0,
scales it to spread 1, then applies a learned per-dimension scale and shift. Most current models use the
simpler <Term id="rmsnorm">RMSNorm</Term>, which only divides by the root-mean-square:

```text
RMSNorm(x) = g ⊙ x / √(mean(x²) + ε)        g: learned scale per dimension, ε: tiny constant
```

The ⊙ means element-by-element multiplication. Dropping the mean saves a little compute and works as well
in practice.

Where the normalisation goes matters. The 2017 paper normalised after each addition (post-norm). Models since
GPT-2 normalise the input of each branch and leave the stream itself alone (<Term id="pre-norm">pre-norm</Term>):

```text
x = x + Attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

Pre-norm keeps the residual path an untouched sum, so gradients flow through it cleanly. Post-norm models
needed careful learning-rate warmup to train at all ([Xiong et al., 2020](https://arxiv.org/abs/2002.04745)).
A final norm sits before the unembedding.

## Position: where is each token?

**In short:** attention ignores order on its own, so position has to be added. Current models do it by
rotating queries and keys (RoPE), so that scores depend on how far apart two tokens are.

Look again at the attention output: a weighted sum over earlier tokens. Sums do not care about order. Shuffle
the earlier tokens and each token gets the same scores and the same output. "Dog bites man" and "man bites
dog" would look alike. Some <Term id="positional-encoding">positional encoding</Term> must break the symmetry.

### Absolute positions: learned and sinusoidal

The simplest fix: a second embedding table, one row per position (0, 1, 2 …), added to the token embedding at
the start. GPT-2 did this. The rows are learned like any other weights. The catch is that the table has a
fixed number of rows, so the model knows nothing about positions beyond the longest it was trained on.

The 2017 paper used fixed sine and cosine waves instead of a learned table. Position p gets a vector whose
entries are sin(p·ω) and cos(p·ω) for a range of frequencies ω, from fast to very slow, like the hands of a
clock. Nothing is learned, and any position has a vector. In practice, models still did poorly past their
training length.

Both are **absolute**: they tell the model "this is token 517". But meaning mostly depends on **relative**
position: "the word two tokens back", wherever that is.

### RoPE: rotate queries and keys

Current models use <Term id="rope">rotary position embedding (RoPE)</Term> ([Su et al., 2021](https://arxiv.org/abs/2104.09864)). It puts
relative position directly into the attention scores. Nothing is added to the embeddings. Instead, each query and key is
rotated, by an amount that depends on its position, just before the dot product.

Here is the idea in two dimensions. Treat a pair of numbers (a, b) as a point on a plane. Rotate the query of
token m by angle m·θ, and the key of token n by angle n·θ. The dot product of two vectors depends on their
lengths and the angle between them. After rotation, the angle between them has changed by (m − n)·θ. So the
score depends on the content and on the distance m − n, not on where the pair sits in the text.

The full vector has d_h numbers per head, so RoPE splits it into d_h / 2 pairs. Each pair turns at its own
speed:

```text
angle for pair i at position p  =  p · θ_i,     θ_i = base^(−2i / d_h),   base usually 10,000 or more
rotate (a, b) by angle α        →  (a·cos α − b·sin α,  a·sin α + b·cos α)
```

The first pairs turn fast, so they can tell neighbours apart. The last pairs turn very slowly, so they can
track distances of thousands of tokens. The exercise implements this in about ten lines (`apply_rope`).

::: details Going deeper: why the score depends only on m − n
Write a pair as a complex number. Rotation by α is multiplication by e^(iα). The query pair becomes q·e^(imθ),
the key pair k·e^(inθ). The 2D dot product is the real part of one times the conjugate of the other:
Re(q·e^(imθ) · conj(k)·e^(−inθ)) = Re(q·conj(k)·e^(i(m−n)θ)). Positions appear only as m − n. Because rotation
does not change length, RoPE also leaves the size of the vectors alone. Values are not rotated; only the
scores need position.
:::

### Stretching to longer contexts

A model trained on 4,000-token windows has never seen the rotation angles of position 100,000. Its slow pairs
reach angles it never trained on, and quality collapses. Several fixes rescale the angles so long positions
look like ones the model has seen. Position interpolation squeezes all positions into the trained range;
[YaRN](https://arxiv.org/abs/2309.00071) scales the fast and slow pairs differently. Models also raise the base
(Qwen3 uses 1,000,000) and finish pretraining with a short phase on long documents. Qwen3, for example, is
[supports 32,768 tokens natively and 131,072 with YaRN](https://huggingface.co/Qwen/Qwen3-8B).

A long window is not the same as using it well. Models often recall facts from the middle of long inputs
worse than from the ends ([Chapter 11](/beyond/agents-memory)).

## The full block and the stack

**In short:** a block is "norm, attention, add; norm, MLP, add". A model is an embedding, a stack of N
identical blocks, a final norm and an unembedding.

Putting the pieces together, one <Term id="decoder-block">decoder block</Term> is:

```text
x = x + Attention(RMSNorm(x))      tokens exchange information
x = x + MLP(RMSNorm(x))            each token processes what it has
```

<TfBlockDiagram />

The whole model, for a window of n token ids:

1. Look up each id in the embedding table: n vectors of width d. This starts the residual stream.
2. Run the stream through N blocks, each with its own weights. N is 4 in the exercise, about 30 to 100 in
   large models.
3. Apply a final norm and the unembedding: one logit per vocabulary entry, at every position.
4. Softmax at position i gives the model's prediction for token i + 1.

During training, all n positions are predicted at once and the losses averaged. During generation, the model
runs, samples one token from the last position, appends it, and runs again. Chapter 3 called
this <Term id="autoregressive">autoregressive</Term> generation.

Rerunning the whole window for every new token would be wasteful. Because of the causal mask, earlier tokens
never see later ones, so their keys and values never change. They can be stored and reused, and each new token
computes only its own query, key and value. This store is the <Term id="kv-cache">KV cache</Term>, the central data structure of
inference ([Chapter 8](/running/inference#the-kv-cache)).

## Counting parameters and FLOPs

**In short:** one block has about 12·d² parameters. A forward pass costs about 2 FLOPs per parameter per
token, plus an attention term that grows with context length.

### Parameters per block

Count the matrices, ignoring the tiny norm scales:

```text
attention   W_Q, W_K, W_V, W_O         4 matrices of d × d            =  4·d²
MLP         W_1 (d → 4d), W_2 (4d → d)  2 matrices of d × 4d           =  8·d²
total per block                                                         ≈ 12·d²
```

The heads do not change the count: h heads of width d/h use the same d × d matrices, cut into slices. The
whole model is N·12·d² plus the embedding and unembedding, V × d each for a vocabulary of V tokens. Two
thirds of the block parameters sit in the MLP.

The exercise prints this. With d = 128, one block has 196,864 parameters, and 12·d² = 196,608. The difference
of 256 is the two RMSNorm scales of 128 each.

### FLOPs per token

Each parameter in a matrix multiply is used once per token, as one multiply and one add: 2 <Term id="flop">FLOPs</Term>. So
the weights cost about 2N FLOPs per token for a model with N parameters.

Attention adds work with no parameters attached. A token that sees n earlier tokens computes n scores of
length d (2·n·d FLOPs) and a weighted sum of n values of width d (another 2·n·d). Per layer that is about
4·n·d, so for the model:

```text
forward FLOPs per token ≈ 2·N + 4·L·n·d        L layers, n tokens of context, width d
```

Averaged over a training window, a token sees about half the window, so the attention term becomes
2·L·n_ctx·d, the form in [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361). Training costs about three
times the forward pass, which gives the 6·N·D rule of
[Chapter 5](/building/pretraining#where-the-6-comes-from).

::: info Back-of-envelope
**Parameters.** Llama 2 7B has d = 4,096, L = 32 and a 32,000-token vocabulary.
12·d² = 12 × 16.8 M ≈ 201 M per block. Times 32 blocks ≈ 6.4 B. Embedding and unembedding add
2 × 32,000 × 4,096 ≈ 0.26 B. Total ≈ 6.7 B, against the published 6.74 B. (Its MLP is shaped differently,
see below, but has about the same size.)

**FLOPs per generated token.** Weights: 2 × 6.7 B ≈ 13 GFLOP. Attention at 4,096 tokens of context:
4 × 32 × 4,096 × 4,096 ≈ 2.1 GFLOP, about 15% extra. At 128,000 tokens of context the same term is
4 × 32 × 128,000 × 4,096 ≈ 67 GFLOP, five times the weights. Long context changes what dominates.

**The attention matrix.** At n = 128,000, one head's score matrix has n² ≈ 1.6 × 10¹⁰ entries: about 33 GB
in 16-bit numbers, for one head of one layer. This is why real kernels never store it whole.
:::

## What current models change (2026)

**In short:** the 2017 block is still the core. Current open models swap in RMSNorm, SwiGLU MLPs, RoPE,
grouped-query attention, no biases, and often mixture-of-experts.

As an example, [Qwen3-8B's configuration](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) reads:
36 layers, d = 4,096, 32 query heads of width 128, 8 key-value heads, an MLP of width 12,288 with SiLU, RMSNorm,
RoPE with base 1,000,000, and no bias terms (details in the [Qwen3 report](https://arxiv.org/abs/2505.09388)).
Each line maps to one change:

- **RMSNorm, pre-norm.** As above.
- **No biases.** The "+ b" in each linear layer is dropped. It adds little, and removing it slightly
  improves training stability.
- **SwiGLU MLP.** Two input projections instead of one. One goes through SiLU (a smooth curve like GELU), and
  the two are multiplied element by element: a gate. Then W_2 projects back. With three matrices, the hidden
  width is cut (to about 2.7 to 3 × d) to keep the size near 8·d². See <Term id="swiglu">SwiGLU</Term>
  ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)).
- **Grouped-query attention.** Several query heads share one key head and one value head (here 32 query heads
  and 8 key-value heads). The KV cache shrinks four times with little quality loss. See
  <Term id="gqa">GQA</Term> ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)).
- **Mixture-of-experts.** The MLP is replaced by many smaller MLPs ("experts"), and a small router sends each
  token to a few of them. [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json)
  has 128 experts per layer and uses 8 per token: 235 B parameters, about 22 B active per token. See
  <Term id="moe">MoE</Term> ([Chapter 6](/building/training-at-scale)).

Some models change attention itself: DeepSeek compresses keys and values (multi-head latent attention), and
several labs mix in sliding-window or linear-attention layers to cut long-context cost. None of these
replace the block structure.

## Try it: a tiny GPT

**In short:** about 200 lines of PyTorch: the whole architecture of this chapter, trained on Shakespeare in
about 3 minutes on 2 CPU cores.

The exercise reuses the BPE tokenizer and data from Chapter 3 (`ch03_bpe.py`). The model has 4 blocks,
d = 128, 4 heads of width 32, a 128-token context, RoPE, pre-norm RMSNorm and a GELU MLP. Attention is written
out by hand in `CausalSelfAttention`: projections, head split, rotation, scaled scores, mask, softmax,
weighted sum. It trains for 1,000 steps of 12 windows each, then checks the attention weights and samples
text.

<<< @/../exercises/ch04_gpt.py

Output (about 190 seconds on 2 CPU cores):

```text
tokens: 502,657   vocab: 769   layers 4, d=128, heads 4, context 128
parameters per block: 196,864   12*d^2 = 196,608
parameters total: 984,448   (embedding + unembedding: 196,864)
guessing uniformly: loss = ln(769) = 6.65
step    0  train loss 6.649  val loss 6.657  (1 s)
step  200  train loss 3.772  val loss 3.950  (31 s)
step  400  train loss 3.394  val loss 3.679  (74 s)
step  600  train loss 3.158  val loss 3.539  (129 s)
step  800  train loss 3.117  val loss 3.470  (159 s)
step 1000  train loss 3.064  val loss 3.384  (190 s)
val perplexity 30.0   val loss per byte 1.594 nats   (bigram in ch03: 1.942)

== attention check (block 0, head 0) ==
rows sum to 1: max |row sum - 1| = 1.2e-07
weight on future tokens (above the diagonal): 0.0
first 5 rows (row i = how token i spreads its attention over tokens 0..i):
  1.00 0.00 0.00 0.00 0.00
  0.49 0.51 0.00 0.00 0.00
  0.15 0.13 0.72 0.00 0.00
  0.02 0.06 0.87 0.05 0.00
  0.15 0.11 0.37 0.20 0.18

== sample ==

with a back canget the gold of thee;
With fieven, in such a whiles twain cousin
Affer tyrinkles shempartins,
Whit rethool: whom the vow of father did
Masters the truumpease of his gET: his blood no
To man of the present will I hoon,
We make both throth fall of King Marcius.

P

total time 193 s
```

What to read from it:

- **The parameter count matches 12·d².** Embedding and unembedding are a fifth of this small model. In large
  models they are a few percent.
- **The loss falls well below the bigram's.** Validation loss reaches 3.38 nats per token. Per byte of text
  (which allows comparison across tokenizers) it is 1.59, against 1.94 for the Chapter 3 bigram on the same
  data. Validation perplexity drops from 63 to 30. Context is doing work.
- **The mask works.** Every row of the attention matrix sums to 1, and no weight lands on a future token.
  Row 0 must put all its weight on itself, since there is nothing else to see.
- **The sample has structure the bigram lacked.** Line breaks follow verse, speaker names appear in capitals,
  and phrases like "the gold of thee" are grammatical. Words are still often invented; a 1 M-parameter model
  trained for 3 minutes is far from fluent.
- **Train loss is below validation loss.** The gap (3.06 vs 3.38) is early overfitting: the model starts to
  memorise its training text. Longer runs would need more data or regularisation.

Things to try: replace RoPE with a learned position table; remove the mask and watch training loss drop while
generation turns to nonsense; remove the √ in the scaling and compare the first few hundred steps.

::: warning Thread
Attention is the transformer's only memory that changes while it runs. Within one context, the keys and
values are a per-token memory of everything read so far, and the model can use it to learn: follow a new
format from examples, adopt a definition given on page 1. This is <Term id="in-context-learning">in-context learning</Term>, and it is the
strongest counterargument to "frozen models cannot learn": they can, within the window.

The intuitions still bite at the edges. That memory lasts exactly as long as the context, and costs n² to
use. The weights, including the MLP key-value memory, stay fixed until the next training run. Nothing in the
block moves a lesson from the first memory to the second. That is the "per-token activations and frozen
weights, nothing in between" picture made concrete.

Long contexts (hundreds of thousands of tokens) and caching stretch the per-token memory a long way, and may
be enough for many tasks. Whether they can replace slow, compressed learning is open (contested).
[Chapter 13](/beyond/beyond-frozen-weights) looks at designs that fill the gap.
:::

::: tip Interview checkpoint
<details><summary>Why scale attention scores by √d?</summary>

The dot product of two d-dimensional vectors with unit-variance entries has variance d, so scores grow like
√d. Large scores saturate softmax to near one-hot, where gradients vanish. Dividing by √d keeps scores at
spread about 1 at initialisation, so attention starts smooth and trainable.
</details>

<details><summary>Why multi-head attention rather than one head of full width?</summary>

Each head produces one softmax-weighted blend per token, so a single head tends to follow one relation at a
time. Several heads with separate projections can attend to different things at once (previous token, the
subject, a matching bracket) for the same total cost, since the head widths add up to d. W_O then combines
them.
</details>

<details><summary>Walk me through generating text with a KV cache.</summary>

Run the prompt through the model once (prefill), storing each layer's keys and values for every token. Then
for each new token, compute only its query, key and value, append the key and value to the cache, and attend
over the cache. The causal mask guarantees earlier keys and values never change, so this matches a full
recompute. Cost per step is linear in context instead of quadratic, at the price of memory that grows with
context.
</details>

<details><summary>Derive the parameter count of a transformer layer.</summary>

Attention: query, key, value and output projections, four d × d matrices, 4d². MLP: d to 4d and back, 8d².
So about 12d² per layer, plus small norm parameters. The model adds V × d for embedding and again for
unembedding unless tied. Check: d = 4,096 and 32 layers give about 6.4 B, plus 0.26 B of embeddings, close to
Llama 2 7B.
</details>

<details><summary>Why are residual connections essential?</summary>

Each sublayer adds to x instead of replacing it, so backpropagation has an identity path from the loss to
every layer and gradients do not shrink across depth. It also lets each sublayer learn a small correction,
and gives layers a shared stream to read and write.
*Senior add-on:* with pre-norm the stream is never normalised, which is what makes very deep stacks train
without careful warmup; the cost is that the stream's scale grows with depth.
</details>
:::

## In practice

- **Inference cost is a transformer property.** Prefill is big matrix multiplies and compute-bound; decode
  reads all weights and the whole KV cache for each token and is memory-bound. The KV cache, not the
  weights, often limits batch size at long context ([Chapter 8](/running/inference)).
- **Long-context pricing.** Providers charge more, or cap context, because attention work and KV-cache memory
  grow with length. Prompt caching saves money because cached keys and values skip prefill.
- **Architecture choices are inference choices.** GQA, latent attention and sliding windows exist mainly to
  shrink the KV cache. Mixture-of-experts trades memory for fewer FLOPs per token.
- **Reading a config file.** hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads and
  intermediate_size give you parameter count, KV cache size and FLOPs per token in a minute. Interviewers
  like asking for exactly that.

## Common misconceptions

- **"Attention is where the knowledge lives."** Attention moves information between tokens. Most parameters,
  and much stored knowledge, sit in the MLPs.
- **"The model attends over all tokens, including later ones."** In a decoder-only LLM the causal mask
  forbids it, in training and in generation.
- **"Attention weights explain the model's reasoning."** They show where one head read from, not what it
  did with it or what other paths contributed. They are a weak explanation on their own.
- **"Attention's quadratic cost dominates everything."** At short and medium context the 2N weight term
  dominates. The n² term matters at long context.
- **"Transformers understand word order natively."** Attention alone treats the past as a set; order comes
  only from the positional encoding.

## Key takeaways

- Attention lets each token take a weighted average of earlier tokens' values, with weights from
  query–key dot products, scaled by √d, masked to the past, softmaxed.
- A block is "norm, attention, add; norm, MLP, add" on a residual stream. Attention mixes tokens; the MLP
  processes each token and holds two thirds of the parameters.
- Position comes from outside attention; current models rotate queries and keys (RoPE) so scores depend on
  relative distance.
- About 12·d² parameters per layer; about 2N forward FLOPs per token plus 4·L·n·d for attention.
- Current models keep the 2017 skeleton with RMSNorm, SwiGLU, no biases, GQA and often mixture-of-experts.

## Further reading

- [Vaswani et al., *Attention Is All You Need* (2017)](https://arxiv.org/abs/1706.03762): the original paper.
  Short; read sections 3.1 to 3.5 with this chapter beside it.
- [Jay Alammar, *The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/): the
  standard visual walk-through of the same model (encoder-decoder form).
- [Andrej Karpathy, *Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY)
  and [nanoGPT](https://github.com/karpathy/nanoGPT): builds the exercise's model live; nanoGPT scales it to
  GPT-2.
- [Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021)](https://arxiv.org/abs/2104.09864):
  the RoPE paper; section 3 has the rotation math.
- [Elhage et al., *A Mathematical Framework for Transformer Circuits* (Anthropic, 2021)](https://transformer-circuits.pub/2021/framework/index.html):
  reads the transformer as a residual stream with heads reading and writing; the basis of Chapter 10.
- [Geva et al., *Transformer Feed-Forward Layers Are Key-Value Memories* (2021)](https://arxiv.org/abs/2012.14913):
  the evidence behind the MLP-as-memory view.

## Review

<Flashcards id="transformer" :cards="cards" />

<MarkDone id="transformer" />
