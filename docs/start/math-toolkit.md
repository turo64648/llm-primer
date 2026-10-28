---
title: "0. The Math You Need"
---

<script setup>
import { cards } from './math-toolkit-review'
</script>

# 0. The Math You Need

This chapter is the minimum math for the rest of the primer: vectors, matrix multiply, derivatives, softmax,
log-probabilities and array shapes. Interviewers expect you to estimate compute from matrix sizes and to
explain what a loss number means, and both rest on this toolkit.

::: info Before you start
- A **language model** reads text as **tokens**: short pieces of words. For each position, it gives a score
  to every possible next token.
- Its behaviour is set by a large file of numbers called **weights** (also called parameters). Training
  adjusts the weights; using the model only reads them.
- You need no school math beyond multiplying and adding. Every formula here is explained in words.

[Chapter 1](/foundations/neural-nets) builds a network from these pieces. This chapter only gives you the
tools.
:::

## Vectors and dot products

**In short:** a vector is a list of numbers, and the dot product of two vectors is one number that says how
much they point the same way.

A model stores the meaning of each token as a list of a few thousand numbers. Two tokens with related
meanings get lists that are "close" in a sense we can measure. That list is a
<Term id="vector">vector</Term>, and its length is its **dimension**. A 3-dimensional vector looks like
`[1, 2, 3]`.

You can picture a vector as an arrow from the origin. To compare two arrows, multiply them position by
position and add up the results. That sum is the <Term id="dot-product">dot product</Term>:

```
a · b = a[0]·b[0] + a[1]·b[1] + ... + a[d-1]·b[d-1]
```

Here `a` and `b` are vectors with the same dimension `d`. The result is a single number:

- **Large and positive** when the arrows point the same way.
- **Near zero** when they are unrelated (at right angles).
- **Large and negative** when they point in opposite directions.

The dot product also grows with the arrows' lengths. To compare only direction, divide by both lengths. The
result is the <Term id="cosine-similarity">cosine similarity</Term>, which is always between −1 and 1.

**Where you will use this.** Token meanings are vectors called embeddings
([Chapter 3](/foundations/tokens-embeddings)). Attention, the core of the transformer, decides which earlier
tokens matter by taking dot products between vectors ([Chapter 4](/foundations/transformer)). Retrieval
systems find related documents by cosine similarity ([Chapter 11](/beyond/agents-memory)).

::: details Going deeper: random vectors in high dimensions are nearly at right angles
Draw two random vectors with 512 entries each (each entry from a standard normal distribution). Their
cosine similarity is almost always close to 0; the typical size is about 1/√512 ≈ 0.044. The exercise below
gets 0.047. So in a space with thousands of dimensions, "unrelated" is the default, and a cosine of 0.3 is
already a strong signal.

The raw dot product of two such vectors has a spread of about √d. This is why attention divides its dot
products by √d before using them: it keeps the scores around 1 no matter how wide the vectors are.
:::

## Matrix multiply is many dot products

**In short:** multiplying an m × k matrix by a k × n matrix computes m · n dot products of length k, which
costs about 2 · m · n · k floating-point operations.

A <Term id="matrix">matrix</Term> is a grid of numbers: m rows and k columns, written m × k. You can read it
as a stack of m vectors (the rows), each of dimension k.

To multiply matrix A (m × k) by matrix B (k × n), take every row of A and every column of B. Each pair gives
one dot product, which becomes one cell of the result C (m × n). The inner sizes must match (both k),
because a dot product needs two vectors of the same length.

<MathMatmulDiagram />

Counting the work is simple. There are m · n cells. Each cell needs k multiplies and about k adds. So the
total is about **2 · m · n · k** floating-point operations, or <Term id="flop">FLOPs</Term>. A
multiply-then-add pair counts as 2 FLOPs.

**Where you will use this.** A layer of a neural network is a matrix multiply: input vectors times a weight
matrix ([Chapter 1](/foundations/neural-nets)). Almost all the compute in training and serving is matrix
multiplies. The 2 · m · n · k count is the start of every compute estimate in
[Chapter 5](/building/pretraining) and [Chapter 8](/running/inference).

::: info Back-of-envelope
**One layer, one token.** A token's vector has 4,096 numbers. A weight matrix maps it to another 4,096
numbers. That is m = 1, k = 4,096, n = 4,096:

2 · 1 · 4,096 · 4,096 ≈ 33.5 million FLOPs.

**A whole model, one token.** Each weight is used in roughly one multiply and one add per token. So a model with N
weights costs about **2 · N FLOPs per token** to run. For 7 billion weights: 2 · 7 × 10⁹ ≈ 14 billion FLOPs
per token. Training costs about three times that: running the model forward is 1 share, and working out the
gradients (Chapter 2) costs about 2 more. That is where "6 · N · D" in [Chapter 5](/building/pretraining) comes from.
:::

::: details Going deeper: how fast is hardware at this?
The exercise below measures about 50 billion FLOPs per second (50 GFLOP/s) on a 2-core cloud VM with NumPy.
A data-center GPU such as the [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/) is rated at
roughly 1,000 TFLOP/s (10¹⁵ per second) for dense matrix multiply in 16-bit numbers. That is about 20,000
times faster, and it is why large models need GPUs. Real workloads reach only part of the rated peak;
[Chapter 5](/building/pretraining) covers how much.
:::

## Derivatives: sensitivity

**In short:** a derivative says how much an output changes when you nudge one input a tiny bit. A gradient
is the list of those numbers, one per input.

Take f(x) = x². At x = 3, f is 9. Nudge x to 3.01, and f becomes 9.0601. The output moved 0.0601 for an input
nudge of 0.01, about 6 times as much. That ratio, 6, is the <Term id="derivative">derivative</Term> of f at
x = 3. It is written f′(3) or df/dx.

So a derivative is a **sensitivity**: "if I move this input a little, the output moves about this many times
as much." A positive derivative means the output goes up with the input; negative means it goes down. For
x², the derivative at any x is 2x, so the sensitivity grows as x grows.

Most functions in ML have many inputs. A **partial derivative** asks the same question about one input,
holding all the others fixed. Collect the partial derivatives for every input, and you get the
<Term id="gradient">gradient</Term>: a vector with one sensitivity per input.

Training uses this directly. The model's error on some data is one number, called the **loss**. The model
has billions of weights. The gradient of the loss says, for every weight, which way to move it to lower
the loss, and how strongly the loss reacts. Training repeatedly takes a small step against the gradient.
This is **gradient descent**, covered in [Chapter 2](/foundations/learning).

::: details Going deeper: why "small" matters
The derivative describes the function only very close to the current point. A step that is too large can
overshoot, because the slope changes along the way. This is the reason training uses a small step size,
called the learning rate.

You can estimate any derivative numerically: compute (f(x + h) − f(x − h)) / 2h for a tiny h. This is too
slow for billions of weights, but it is a standard way to test a hand-written gradient.
:::

## The chain rule

**In short:** when functions are chained, their sensitivities multiply.

Suppose x feeds into u = 3x + 1, and u feeds into y = u². How sensitive is y to x?

- u reacts to x with sensitivity 3 (every unit of x adds 3 to u).
- y reacts to u with sensitivity 2u (the derivative of u²).
- So y reacts to x with sensitivity 2u · 3.

At x = 2, u = 7, so dy/dx = 2 · 7 · 3 = 42. The exercise checks this number by nudging x. This is the
<Term id="chain-rule">chain rule</Term>: along a chain of steps, multiply the sensitivities of each step.

```
dy/dx = dy/du · du/dx
```

Read it as "how much y moves per unit of u, times how much u moves per unit of x." A gearbox is a good
picture: gear ratios multiply. The picture breaks in one way. If x reaches y through several paths, you
**add** the contributions of the paths.

**Where you will use this.** A neural network is a long chain of simple steps, ending in the loss. To get
the gradient for a weight deep inside, you multiply sensitivities backward from the loss, step by step. That
procedure is **backpropagation** ([Chapter 2](/foundations/learning)). It is the chain rule, applied
mechanically, reusing the products it has already computed.

## exp, log and softmax

**In short:** softmax turns any list of scores into probabilities. It uses exp to make every score positive,
then divides by the total.

Two functions show up everywhere:

- **exp(x)**, or eˣ with e ≈ 2.718, turns any number into a positive one. It grows very fast: exp(0) = 1,
  exp(10) ≈ 22,000, exp(100) ≈ 10⁴³.
- **log(x)**, the natural logarithm, undoes exp: log(exp(x)) = x. It only takes positive inputs. It turns
  products into sums: log(a · b) = log(a) + log(b). For a number between 0 and 1, log is negative.

The set of all tokens a model knows is its **vocabulary**. A model's last layer gives one raw score per
token in the vocabulary. These scores are called
<Term id="logit">logits</Term>. They can be any real number, and they do not add up to anything.

To turn them into probabilities, <Term id="softmax">softmax</Term> applies exp to each score, then divides
by the sum:

```
softmax(z)[i] = exp(z[i]) / (exp(z[0]) + exp(z[1]) + ... + exp(z[V-1]))
```

Here z is the vector of logits and V is the vocabulary size. Every output is positive, and the
outputs add up to 1. A higher score always gets a higher probability. Because of exp, a lead of a few units
in score becomes a large lead in probability.

**Only differences matter.** Adding the same number c to every logit leaves the result unchanged: the extra
factor exp(c) appears on top and bottom, and cancels. Softmax of `[1000, 1001, 1002]` equals softmax of
`[0, 1, 2]`.

**Why subtract the max.** exp(1000) is too large for any floating-point format, so it becomes infinity. The
naive softmax then computes infinity divided by infinity, which is "not a number" (NaN). The fix uses the
rule above: subtract the largest logit from all logits first. Now the largest exponent is exp(0) = 1, so
nothing overflows. The sum is at least 1, so nothing divides by zero. The answer is mathematically the same.

**Where you will use this.** Every next-token prediction ends with a softmax over the vocabulary
([Chapter 4](/foundations/transformer)). Attention uses a softmax to turn dot-product scores into weights
that add up to 1. When a model generates text, a setting called **temperature** divides the logits by a number before the softmax
([Chapter 8](/running/inference)).

::: details Going deeper: the overflow limits
In 64-bit floats, exp(x) overflows above x ≈ 709. In 32-bit floats it overflows above x ≈ 88.7. The 16-bit format bfloat16 has the same
limit as 32-bit; the older float16 overflows above x ≈ 11. Logits of 100 are unusual but possible, and a single
overflow poisons the whole result with NaN. Every real implementation subtracts the max, or uses an
equivalent trick.

The related function **log-sum-exp**, log(Σ exp(z[i])), is computed the same way: max(z) + log(Σ exp(z[i] −
max(z))). It appears inside the loss below.
:::

## Probabilities and log-probabilities

**In short:** a model outputs a probability distribution over the next token. Working with logs of
probabilities turns long products into sums and avoids numbers too small to store.

A <Term id="probability-distribution">probability distribution</Term> over a set of outcomes gives each one
a number between 0 and 1, and the numbers add up to 1. The softmax output is exactly this: one probability
per token in the vocabulary.

A language model writes text one token at a time. The probability of a whole sentence is the product of the
probability of each token, given the tokens before it:

```
P("the cat sat") = P("the") · P("cat" | "the") · P("sat" | "the cat")
```

The bar "|" reads "given". Each factor is one softmax output.

Multiply a thousand probabilities of about 0.1 and you get 10⁻¹⁰⁰⁰. That is far below the smallest number a
64-bit float can hold (around 10⁻³⁰⁸), so it rounds to 0. Taking logs fixes this. The log of a product is a
sum of logs, so the sentence's **log-probability** is a sum of per-token
<Term id="log-probability">log-probabilities</Term>: here, 1,000 · log(0.1) ≈ −2,303. That fits easily.

Log-probabilities are always 0 or negative. 0 means "certain"; more negative means "less likely".

**Where you will use this.** Training and evaluation work with sums of log-probabilities over tokens
([Chapter 5](/building/pretraining), [Chapter 9](/running/evaluation)). Several post-training methods (the tuning done after the main training) compare
the log-probability of the same answer under two models ([Chapter 7](/building/post-training)).

## Cross-entropy: surprise

**In short:** the standard training loss is −log of the probability the model gave to the correct next
token. It measures how surprised the model was.

Suppose the correct next token is "sat". If the model gave it probability 0.9, it was barely surprised. If it
gave 0.01, it was very surprised. A good measure of **surprise** is −log(p):

| Probability given to the correct token | Surprise, −log(p) |
|---|---|
| 1.0 | 0 |
| 0.9 | 0.105 |
| 0.5 | 0.693 |
| 0.01 | 4.6 |

The surprise is 0 when the model is certain and right. It grows without limit as the probability of the
right answer falls toward 0. Averaged over every token in the training text, this is the
<Term id="cross-entropy">cross-entropy</Term> loss. Training a language model means making this one number smaller.

A useful baseline is a model that knows nothing and spreads probability evenly over V tokens. Its surprise
on every token is −log(1/V) = log(V). For V = 50,000, that is about 10.8. A trained model's loss is far
below this, so a loss number tells you where you are between "knows nothing" and "certain".

The unit is the **nat** when you use the natural log, as ML code does. With log base 2, the unit is the
bit: the number of yes/no questions the surprise is worth.

**Where you will use this.** Cross-entropy is the loss in [Chapter 2](/foundations/learning) and the
pretraining objective in [Chapter 5](/building/pretraining). Scaling laws in Chapter 5 predict how this
number falls with model size and data.

::: details Going deeper: the formal definition, perplexity, and a tidy gradient
For a true distribution p and a model distribution q, cross-entropy is −Σ p(x) · log q(x). In language
modelling, the "true distribution" for one position puts all its weight on the actual next token. The sum
then has one term left: −log q(correct token).

**Perplexity** is exp(loss). It reads as "the model is as unsure as if it were choosing evenly among this many
tokens". A loss of log(50,000) is a perplexity of 50,000.

Softmax followed by cross-entropy has a simple gradient with respect to the logits: the predicted
probabilities, minus 1 at the correct token. So the loss pushes the correct token's logit up, and every
other logit down in proportion to its probability. [Chapter 2](/foundations/learning) uses this.
:::

## Shapes and broadcasting in NumPy

**In short:** every array has a shape, and most bugs in model code are shape bugs. Broadcasting lets NumPy
combine arrays of different shapes by stretching size-1 dimensions.

Model code works on many vectors at once. A batch of 4 text sequences, each 3 tokens long, each token a
vector of 5 numbers, is one array with **shape** `(4, 3, 5)`. The usual names for these sizes are
**batch**, **sequence length** and **model width** (often written B, T, D). PyTorch calls such arrays
<Term id="tensor">tensors</Term>; the idea is the same.

The `@` operator multiplies matrices using the **last two** dimensions. Any dimensions before them are
treated as a batch. So `(4, 3, 5) @ (5, 2)` gives `(4, 3, 2)`: every token vector is multiplied by the same
5 × 2 matrix.

<Term id="broadcasting">Broadcasting</Term> is how NumPy combines arrays whose shapes differ. It lines up
shapes from the right. Two sizes are compatible if they are equal or one of them is 1 (or missing). A size-1
dimension is stretched to match, without copying memory.

| Left shape | Right shape | Result | Why |
|---|---|---|---|
| `(4, 3, 5)` | `(5,)` | `(4, 3, 5)` | One bias vector added to every token |
| `(4, 3, 5)` | `(4, 3, 1)` | `(4, 3, 5)` | One number per token, such as its max |
| `(4, 3, 5)` | `(3,)` | error | 5 and 3 do not match on the right |
| `(3, 1)` | `(1, 3)` | `(3, 3)` | Both stretch: no error, possibly a bug |

The last row is the dangerous one. Mixing a column with a row silently builds a full table, and the code
runs. Two habits help. Pass `keepdims=True` to reductions such as `max` and `sum`, so the result keeps a
size-1 dimension and broadcasts back correctly. And always say which axis you reduce over: softmax over the
vocabulary is `axis=-1`, the last dimension.

**Where you will use this.** Every exercise from [Chapter 1](/foundations/neural-nets) on uses these
shapes. Attention in [Chapter 4](/foundations/transformer) is a sequence of matrix multiplies and
broadcasts over `(B, T, D)` arrays.

## Exercise: the toolkit in NumPy

The script checks each idea in this chapter with real numbers. It shows a dot product as similarity, checks
one cell of a matrix multiply and measures FLOP/s, checks the chain rule by nudging the input, breaks and
fixes softmax, prints surprise values, and runs the broadcasting cases from the table.

<<< @/../exercises/ch00_toolkit.py

Output (the GFLOP/s line depends on your machine):

```
1. dot(a, a)  = 14.0
   dot(a, -a) = -14.0
   cosine of two random 512-d vectors = 0.047
2. C[3, 7] == row 3 of A . column 7 of B: True
   2*m*n*k = 33,554,432 FLOPs; ~50 GFLOP/s on this CPU
3. dy/dx numeric = 42.0000, chain rule = 42.0000
4. naive  : [nan nan nan]
   stable : [0.09   0.2447 0.6652]
   same as softmax([0, 1, 2]): [0.09   0.2447 0.6652]
5. p(correct) = 0.9  -> loss = 0.105 nats
5. p(correct) = 0.5  -> loss = 0.693 nats
5. p(correct) = 0.01 -> loss = 4.605 nats
   uniform guess over 50,000 tokens -> loss = 10.82
6. (X + bias).shape = (4, 3, 5)
   (X @ W).shape     = (4, 3, 2)
   max keepdims      = (4, 3, 1) -> (X - max).shape = (4, 3, 5)
   X + ones(3) fails: operands could not be broadcast together with shapes (4,3,5) (3,)
```

Try changing the softmax input to `[1, 2, 3]` and `[10, 20, 30]`. The first gives the same result as
`[0, 1, 2]`; the second is much more peaked, because the gaps are ten times larger.

## In practice

- **Compute estimates.** "How long would this take on N GPUs?" starts with 2 · m · n · k per matrix
  multiply, or 2 · N FLOPs per token for a whole model. Divide by the hardware's realistic FLOP/s.
- **Reading a loss curve.** A loss of 10.8 at step 0 with a 50,000-token vocabulary is expected: the model is
  guessing evenly. A loss that starts far above log(V) usually points to an initialisation problem: the starting logits are too large.
- **NaN in training.** A NaN loss often comes from an overflow in exp, or a log of 0. Frameworks provide
  fused, stable functions (such as PyTorch's `cross_entropy`, which takes logits) for this reason. Pass them
  logits, not probabilities.
- **Silent shape bugs.** A `(B, 1)` array added to a `(B,)` array becomes `(B, B)`. The code runs and the
  loss looks plausible. Asserting shapes at function boundaries catches this early.

::: tip Interview checkpoint
<details><summary>How many FLOPs does multiplying a 1,024 × 4,096 matrix by a 4,096 × 4,096 matrix take?</summary>

2 · 1,024 · 4,096 · 4,096 ≈ 34 billion (3.4 × 10¹⁰). There are 1,024 · 4,096 output cells, each a dot
product of length 4,096, and each multiply-add counts as 2 FLOPs.
</details>

<details><summary>Why do softmax implementations subtract the maximum logit?</summary>

exp of a large logit overflows to infinity, and infinity divided by infinity is NaN. Softmax does not change
when you add the same constant to every input, so subtracting the max gives the same answer with every
exponent at most 0.
</details>

<details><summary>A model's training loss starts at about 10.8. What does that tell you?</summary>

10.8 ≈ log(50,000), so the model is spreading probability about evenly over a vocabulary of roughly 50,000
tokens. That is the expected starting point for an untrained model.
</details>

<details><summary>What is the chain rule, and why does it matter for training?</summary>

When functions are chained, the sensitivity of the end to the start is the product of each step's
sensitivity. A network is a chain of steps ending in the loss, so backpropagation uses the chain rule to get
every weight's gradient in one backward pass.
</details>
:::

## Common misconceptions

- **"A bigger dot product means more similar."** Only if the lengths are comparable. Long vectors give large
  dot products. Cosine similarity removes the effect of length.
- **"Softmax picks the maximum."** It gives every option some probability. It only approaches "pick the max"
  when the gaps between logits are large.
- **"Logits are probabilities."** They are raw scores: any real number, not adding up to 1. Softmax turns them
  into probabilities.
- **"A loss of 0 is the goal."** On real text, the next token is often truly uncertain, so even a perfect
  model has a loss above 0.
- **"No error means the shapes are right."** Broadcasting can turn a mistake into a valid, wrong result.

## Key takeaways

- A dot product measures how much two vectors point the same way. Matrix multiply is many dot products and
  costs about 2 · m · n · k FLOPs.
- A derivative is a sensitivity. The gradient lists the sensitivity of the loss to every weight.
- The chain rule multiplies sensitivities along a chain. Backpropagation is the chain rule, applied
  mechanically.
- Softmax turns logits into probabilities. Subtract the max first to avoid overflow.
- Cross-entropy is −log of the probability given to the right answer: the model's surprise. log(V) is the
  "knows nothing" baseline.

## Further reading

- [3Blue1Brown, Essence of linear algebra](https://www.3blue1brown.com/topics/linear-algebra): short videos
  that make vectors, dot products and matrices visual.
- [3Blue1Brown, Essence of calculus](https://www.3blue1brown.com/topics/calculus): derivatives and the chain
  rule as pictures, not formulas.
- [Andrej Karpathy, The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0):
  derivatives and the chain rule built up in code; good preparation for Chapter 2.
- [Christopher Olah, Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/):
  why −log(p) is the right measure of surprise, and what cross-entropy means.
- [NumPy, Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html): the exact rules, with
  diagrams.

## Review

<Flashcards id="math-toolkit" :cards="cards" />

<MarkDone id="math-toolkit" />
