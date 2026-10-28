---
title: "2. How Networks Learn"
---

<script setup>
import { cards } from './learning-review'
</script>

# 2. How Networks Learn

Training finds the weights of a neural network: it measures how wrong the network is, works out which way
to nudge every weight, and nudges. Interviewers expect you to explain each part of that loop, from
backpropagation to Adam, and what it costs in memory.

::: info Before you start
- A neural network is layers of the form h = f(Wx + b): a matrix multiply, a bias, and a simple bend
  called an activation function. Its weights and biases are its parameters ([Chapter 1](/foundations/neural-nets)).
- A derivative is a sensitivity: how much an output moves per tiny nudge of one input. The chain rule says
  sensitivities along a chain of steps multiply ([Chapter 0](/start/math-toolkit)).
- You will see Python, NumPy and a little PyTorch.

The chapter explains what it needs as it goes.
:::

## Loss: one number for "how wrong"

**In short:** training needs a single number that is small when the network is right and large when it is
wrong. That number is the loss, and training means making it smaller.

In [Chapter 1](/foundations/neural-nets) a person chose the weights that solve XOR. For a network with a
billion weights, nobody can. Instead you show the network examples with known answers, and let a procedure
adjust the weights. The procedure needs a score to improve.

Say the network predicts tomorrow's temperature. It says 21 °C; the answer is 18 °C. The error is 3. Square
it to get 9, so that errors in either direction count, and big errors count much more than small ones.
Average this over many examples:

```text
MSE = (1/N) · Σ (prediction_i − target_i)²
```

N is the number of examples, and Σ means "add up over all of them". This is the <Term id="mse">mean squared error (MSE)</Term>. It is one example of a <Term id="loss">loss</Term>: one
number that says how wrong the network is on some data. The loss depends on the weights, because the
predictions do. Training means finding weights that make it small.

### Classification: cross-entropy

Now the network must pick one answer from a fixed set: "cat or dog", "XOR is 0 or 1", or "which token comes
next". Here the last layer outputs one raw score per option. These scores are called <Term id="logit">logits</Term>. A function called <Term id="softmax">softmax</Term> turns them into
probabilities: it applies exp to each score, so they are all positive, then divides by their sum, so they add
up to 1.

The natural loss asks: what probability did the network give to the correct answer? Call it p. The loss is
−log(p). It is 0 when p = 1, about 0.7 when p = 0.5, and grows without limit as p falls toward 0. Chapter 0
called this number "surprise". Averaged over examples, it is the <Term id="cross-entropy">cross-entropy</Term> loss:

```text
cross-entropy = (1/N) · Σ −log(p_i)        p_i = probability given to the correct answer for example i
```

A language model is a classifier with one option per token in its vocabulary. At every position in the
training text, the correct answer is the token that actually comes next. Its loss is the average surprise at
the real next token, over billions of positions. Every LLM's main training starts from this one formula
([Chapter 5](/building/pretraining)).

Why not use MSE on the probabilities? Take a network that is confidently wrong: it gives the right answer
p = 0.001. With cross-entropy, the push on the right answer's score stays near its maximum. With MSE, the push
shrinks along with p, so the most wrong examples barely move the weights. The exercise
uses cross-entropy for this reason.

::: details Going deeper: the gradients, and maximum likelihood
For a two-way choice with a sigmoid output p = 1 / (1 + e^(−z)) and target 1:
- Cross-entropy −log(p) has derivative p − 1 with respect to the logit z. At p = 0.001 that is about −1.
- MSE (1 − p)² has derivative −2(1 − p) · p(1 − p). At p = 0.001 that is about −0.002.

With softmax over many options, the derivative of cross-entropy with respect to each logit is the
predicted probability minus 1 for the correct option, and minus 0 for the others: `p − onehot`. It pushes
the correct logit up and every other logit down in proportion to its probability.

Minimising cross-entropy is the same as maximising the probability the model gives to the whole training
set (maximum likelihood), because the log of a product of probabilities is a sum of log-probabilities.
:::

## The loss landscape

**In short:** picture the loss as the height of a surface over all possible weight settings. Training is a
walk downhill.

Take a network with only two weights. For every pair of values (w1, w2) you can compute the loss on the
training data. Plot the loss as a height above the (w1, w2) plane and you get hills and valleys. Good weights
are low points.

A real network has billions of weights, so its surface lives in billions of dimensions. Nobody can picture
that, but the idea carries over: every weight setting has a loss, and you want a low one. This surface is the <Term id="loss-landscape">loss landscape</Term>.

You cannot see the landscape. Computing the loss at one point means running the network over the data, and
there are far too many points to try. What you *can* compute cheaply is the slope at the point where you
stand. Gradient descent uses only that.

::: details Going deeper: what the landscape is really like
For a network with nonlinear layers, the landscape is not a simple bowl. It has many minima, flat regions
and saddle points (low in some directions, high in others). In very high dimensions, most points where the
slope is zero are saddles, not traps, and many minima found in practice have similar loss. How the shape of
the minimum relates to how well the network generalises (for example "flat minima generalise better") is
debated.
:::

## Gradient descent

**In short:** the gradient points in the direction where the loss rises fastest. Step the other way, a
little, and repeat.

Stand on a hillside in fog. You cannot see the valley, but you can feel which way the ground slopes under
your feet. Take a small step downhill, feel again, step again. You will reach a low point, though not
necessarily the lowest.

For one weight w, the derivative dL/dw says how fast the loss L rises as w grows. If it is positive, lower w;
if negative, raise w. For many weights, collect one derivative per weight. That vector is the <Term id="gradient">gradient</Term>, written ∇L. It has exactly one number per parameter. Taken together,
it points in the direction in weight space where the loss rises fastest. Its length says how steep that is.

So the update is:

```text
w ← w − lr · ∇L(w)
```

w is the vector of all weights. ∇L(w) is the gradient at the current weights. lr is the <Term id="learning-rate">learning rate</Term>, a small positive number such as 0.001 that sets the step
size. The minus sign means "go downhill". Repeating this step is <Term id="gradient-descent">gradient descent</Term>.

The learning rate matters because the gradient only describes the slope where you stand. Take too large a
step and you overshoot the valley onto the opposite slope, possibly higher than before. Take too small a step
and training takes forever.

<LrnStepSizeDiagram />

::: details Going deeper: why "steepest"
For a tiny step d, the loss changes by about ∇L · d (a dot product). Among all steps of the same length, a
dot product is most negative when d points exactly opposite to ∇L. That is why −∇L is the direction of
steepest descent. The approximation only holds for small steps, which is the formal reason the learning
rate must be small.
:::

## Backpropagation: the chain rule, run backwards

**In short:** record every small step of the forward pass, then walk the record backwards from the loss,
multiplying local derivatives. One backward walk gives the gradient for every parameter.

Gradient descent needs the gradient, and a network can have billions of weights. You could nudge each weight
and re-run the network, but that is one full forward pass per weight. Backpropagation gets all of them with one
backward pass, which costs about as much as two forward passes.

### A worked example

Take one weight w, one input x, a bias b and a target t, with the loss L = (w·x + b − t)². Use w = 3,
x = 2, b = 1, t = 5. Break the forward pass into small steps, keeping every result:

```text
u = w · x   = 6
z = u + b   = 7
d = z − t   = 2
L = d²      = 4
```

Now go backwards. Each step knows its own local derivative, which is easy:

- L = d², so dL/dd = 2d = **4**.
- d = z − t, so d moves exactly with z: dL/dz = 4 · 1 = **4**.
- z = u + b, so z moves exactly with u and with b: dL/du = **4**, dL/db = **4**.
- u = w · x, so u moves x times as much as w: dL/dw = 4 · x = 4 · 2 = **8**.

Each line multiplies the gradient arriving from the right by one local derivative. That is the <Term id="chain-rule">chain rule</Term>, applied one step at a time.

<LrnBackpropDiagram />

Now take one gradient descent step with lr = 0.1. w becomes 3 − 0.1 · 8 = 2.2, and b becomes
1 − 0.1 · 4 = 0.6. The new prediction is 2.2 · 2 + 0.6 = 5.0, exactly the target, so the loss is 0. Here the
step lands exactly; usually it only gets closer.

### The mechanism

The record of steps is a <Term id="computation-graph">computation graph</Term>: each node is one operation,
and its edges point to the values it used. **Backpropagation** walks this graph backwards:

1. **Forward pass.** Compute every node in order, and save the values the backward pass will need.
2. **Seed.** Set the gradient of the loss with respect to itself to 1.
3. **Backward pass.** Visit nodes in reverse order. Each node multiplies its own gradient by its local
   derivative for each input, and *adds* the result to that input's gradient.

Step 3 adds because one value can feed several later steps. A weight used at many positions, for example,
gets the sum of the contributions from all of them. That is the chain rule's "add over paths".

<Term id="backpropagation">Backpropagation</Term> is not a learning theory. It is bookkeeping: an exact,
efficient way to compute a gradient. Libraries that do it for you, like PyTorch, are called <Term id="autograd">autograd</Term> (automatic differentiation) engines.

### Why backwards?

The chain rule lets you multiply the factors in either order. Going forwards from one input, you learn how
*that one input* affects *every output*. Going backwards from one output, you learn how *every input* affects
*that one output*.

Training has one output, the loss, and billions of inputs, the weights. So one backward pass gives the whole
gradient. The forward order would need one pass per weight. This choice is called **reverse-mode**
automatic differentiation. Forward mode wins in the opposite case: few inputs, many outputs.

### A systems view

The computation graph is a dependency graph, like a build graph. The forward pass runs it in dependency
order. The backward pass runs it in reverse dependency order and needs each node's saved inputs.

That has a cost. Every intermediate value the backward pass needs must stay in memory from the moment the
forward pass makes it until the backward pass uses it. For a large model these saved <Term id="activations">activations</Term> can take more memory than the weights. They grow with batch size,
sequence length and number of layers. Inference keeps none of them: it can throw each layer's output away
once the next layer has used it.

The standard trade is <Term id="activation-checkpointing">activation checkpointing</Term>: save only some
activations, and recompute the rest during the backward pass. It costs extra compute to save memory
([Training at Scale](/building/training-at-scale)).

::: details Going deeper: the cost of a backward pass
A forward pass through a weight matrix is one matrix multiply. The backward pass needs two: one for the
gradient with respect to the weights, and one for the gradient with respect to the layer's input, which is
passed on to the layer before. So the backward pass costs about twice the forward pass.

For an LLM with N parameters, the forward pass costs about 2N floating-point operations per token
([Chapter 1](/foundations/neural-nets)). Training costs about 2N + 4N = **6N per token**. This "6N" rule is
the basis of training compute estimates in [Chapter 5](/building/pretraining).

The algorithm has a long history. Its popular form for neural networks is
[Rumelhart, Hinton and Williams (1986)](https://www.nature.com/articles/323533a0); reverse-mode
differentiation itself is older (Linnainmaa, 1970).
:::

## Try it: an autograd engine from scratch

**In short:** about 60 lines of Python give every number a memory of how it was made. That is enough to
train a small network on XOR and to check every gradient by brute force.

Each `Value` holds a number, the `Value`s it was made from, and the local derivative for each of them, saved
during the forward pass. `backward()` sorts the graph so that every node comes after the nodes it depends on
(a topological sort). Then it walks that order in reverse, as in the worked example. The network uses
tanh, a smooth S-shaped activation function, and a sigmoid at the output to turn its score into a
probability.

The **gradient check** in Part 2 is how you test a gradient you computed. Nudge one weight up by a tiny
amount ε (here 10⁻⁶), then down, and see how the loss changes: (L(w + ε) − L(w − ε)) / 2ε. It needs two forward passes
per weight, so it is far too slow for training, but it is a reliable test. This style of engine follows
Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

<<< @/../exercises/ch02_autograd.py

Output:

```text
== Part 1: the chain rule, by hand and by the engine ==
y = 49.0,  dy/dx by hand = 2*(3*2+1)*3 = 42,  by the engine = 42.0

== Part 2: gradient check on 17 parameters ==
loss = 0.9249; the graph for one loss has 220 nodes, all kept for backward
first 3 grads (engine):   [-0.00785, -0.127251, -0.004686]
largest |engine - finite difference| over all params: 1.0e-10

== Part 3: train with gradient descent (learning rate 0.5) ==
step   0  loss 0.9249
step 100  loss 0.1281
step 200  loss 0.0392
step 300  loss 0.0223
step 400  loss 0.0155
step 500  loss 0.0118
x=[0, 0]  target=0  predicted P(1)=0.003
x=[0, 1]  target=1  predicted P(1)=0.987
x=[1, 0]  target=1  predicted P(1)=0.986
x=[1, 1]  target=0  predicted P(1)=0.017
```

The engine and the finite differences agree to about 10⁻¹⁰, so the backward pass is right. The network
starts with random weights and a loss near 0.93. Gradient descent alone finds weights that solve XOR,
where Chapter 1 had to set them by hand.

Note the 220 nodes: a network with 17 parameters builds a graph of 220 values for four examples, and every
one stays in memory until `backward()` runs. That is the saved-activations cost, in miniature.

::: details Going deeper: variations to try
- Comment out the line that resets `p.grad = 0.0`. Gradients pile up across steps and training goes wrong.
  PyTorch has the same behaviour, which is why every loop calls `zero_grad()`.
- Set `lr = 20`. The loss jumps around instead of falling: the right panel of the step-size diagram.
- Replace the loss with MSE on the probability, `(p - t) ** 2`, and compare how fast it trains.
:::

## Minibatches: cheaper, noisier steps

**In short:** computing the gradient over all the data for every step is too slow. Use a small random
sample per step instead. Each step is noisier, but you can take far more of them.

The true loss is an average over the whole training set. For an LLM that is trillions of tokens. One exact
gradient step would mean a forward and backward pass over all of them.

So each step uses a random <Term id="minibatch">minibatch</Term>: for example 32 examples, or a few million
tokens for an LLM. The average gradient over the minibatch is a noisy estimate of the true gradient. It
points roughly downhill and is right on average. Gradient descent with these estimates is <Term id="sgd">stochastic gradient descent (SGD)</Term>. "Stochastic" means "random".

One full pass over the training data is an <Term id="epoch">epoch</Term>. Small networks often train for
many epochs. LLM pretraining usually sees most of its data about once.

Batch size is a trade-off:

- **Bigger batches** give less noisy gradients and keep hardware busy, because many examples share one
  matrix multiply. Past some size, doubling the batch no longer halves the number of steps you need.
- **Smaller batches** give more steps for the same compute, but each step is noisier.

The noise is not only a cost. It shakes the weights out of narrow dips and saddle points (spots that are flat but not a minimum). Some researchers
argue it also steers training toward solutions that generalise better; how much this matters is contested.

::: details Going deeper: batch size and learning rate
Averaging over B examples shrinks the noise in the gradient by about √B. A common rule of thumb scales the
learning rate up with the batch size, linearly up to a point ([Goyal et al., 2017](https://arxiv.org/abs/1706.02677)).
Beyond a "critical batch size" the returns fall off ([McCandlish et al., 2018](https://arxiv.org/abs/1812.06162)).
Some LLM runs grow the batch size during training.
:::

## Better steps: momentum and Adam

**In short:** plain SGD reacts only to the current noisy gradient, with the same step size for every
weight. Momentum averages over recent steps. Adam also gives each weight its own step size. Both need extra
memory per parameter.

The rule that turns a gradient into a weight update is called the **optimizer**. Plain SGD is the simplest
one.

### Momentum

Picture a long, narrow valley that slopes gently along its length and steeply across. Plain gradient
descent mostly sees the steep sides. It zigzags from wall to wall and crawls along the floor.

<Term id="momentum">Momentum</Term> fixes this by keeping a running average of recent gradients and
stepping along that:

```text
v ← μ · v + g
w ← w − lr · v
```

g is the current gradient and v is the running average, called the velocity. μ (often 0.9) sets how much of
the past it keeps; with 0.9, it averages roughly the last 10 steps. Across the valley, the gradient flips
sign each step, so those parts cancel. Along the valley, it points the same way every time, so they add up and
the step speeds up. Like a ball rolling downhill, it builds speed in a consistent direction.

### Adam

Different weights see gradients of very different sizes. A weight whose gradient is always tiny barely moves
under one shared learning rate. A weight with huge gradients may overshoot. <Term id="adam">Adam</Term>
(adaptive moment estimation) gives each weight its own step size by also tracking how large that weight's
gradients usually are:

```text
m ← β1 · m + (1 − β1) · g          running average of the gradient (the momentum)
v ← β2 · v + (1 − β2) · g²         running average of the squared gradient (the typical size)
w ← w − lr · m / (√v + ε)
```

All of this is per weight. m is a momentum term, as above. v tracks the typical squared gradient, so √v is
roughly the typical gradient size. Dividing m by √v cancels out the scale: each weight moves about lr per
step in its consistent direction, whatever the size of its gradients. β1 and β2 set how long each average
remembers (typically 0.9 and 0.95 to 0.999). ε is a tiny number that prevents division by zero.

Adam works well with little tuning, which is why it, in the form of AdamW below, is the default optimizer
for LLMs. Its cost is memory. The two averages are the optimizer's state, kept between steps for every
parameter: **two extra numbers per parameter**, usually stored as 32-bit floats, so 8 bytes. For a 70B model
that is 560 GB before counting anything else. Splitting this <Term id="optimizer-state">optimizer state</Term>
across GPUs is a central topic of [Training at Scale](/building/training-at-scale).

### AdamW and weight decay

<Term id="weight-decay">Weight decay</Term> shrinks every weight a little toward zero at each step. It
discourages the network from relying on a few huge weights. The section on overfitting below says why that
helps.

The old way to do it was to add a penalty λ · Σ w² to the loss. With Adam that goes wrong. The penalty's
gradient passes through Adam's division by √v, so weights with large gradients get barely any decay. <Term id="adamw">AdamW</Term> decouples the two: it applies the decay directly to the weights, outside
Adam's scaling ([Loshchilov and Hutter, 2017](https://arxiv.org/abs/1711.05101)):

```text
w ← w − lr · ( m / (√v + ε) + λ · w )
```

λ is the decay strength, often 0.1 for LLMs.

::: details Going deeper: bias correction, real settings, newer optimizers
m and v start at zero, so early on they are too small. Adam divides them by (1 − β1ᵗ) and (1 − β2ᵗ) at step
t to correct this ([Kingma and Ba, 2014](https://arxiv.org/abs/1412.6980)).

Llama 2 used AdamW with β1 = 0.9, β2 = 0.95, ε = 10⁻⁵, weight decay 0.1, and gradient clipping at 1.0
([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)). β2 below the textbook 0.999 makes v react faster
to sudden changes in gradient size, which helps stability in large runs.

Newer optimizers aim to beat AdamW. One example is Muon, which treats each weight matrix as a whole
rather than each number separately; Moonshot AI reported using it for large LLMs
([Liu et al., 2025](https://arxiv.org/abs/2502.16982)). AdamW remains the common default.
:::

::: info Back-of-envelope
**Memory per parameter: training vs inference**

Inference in bf16 (a 2-byte number format):
- Weights: **2 bytes** per parameter.

Training with AdamW in mixed precision (16-bit math, 32-bit copies for the update):
- Weights in bf16: 2 bytes
- Gradient: 2 bytes (one number per parameter)
- 32-bit master copy of the weights: 4 bytes
- Adam's m and v, 32-bit: 4 + 4 = 8 bytes
- Total: **16 bytes** per parameter, 8 times inference. Plus the saved activations, which depend on batch
  size and sequence length.

For a 7B model: 7 × 10⁹ × 2 ≈ **14 GB** to serve, 7 × 10⁹ × 16 ≈ **112 GB** to train before activations.
That does not fit on one 80 GB GPU, which is why training is spread across many
([Training at Scale](/building/training-at-scale)).

Compute: inference costs about 2N operations per token; training costs about 6N (forward plus a backward
pass twice as expensive).
:::

::: tip Interview checkpoint
<details><summary>Why cross-entropy rather than MSE for classification?</summary>

Cross-entropy is −log of the probability given to the right answer, so it punishes confident mistakes very
hard and its gradient with respect to the logits is simply p − onehot. With MSE on probabilities, a
confidently wrong prediction has a tiny gradient, so the worst examples learn slowest.
Senior add-on: cross-entropy is maximum likelihood, and paired with softmax it gives well-scaled gradients
whatever the probabilities.
</details>

<details><summary>What does backpropagation compute, and why is it done in reverse?</summary>

It computes the exact gradient of the loss with respect to every parameter, by applying the chain rule
over the computation graph from the loss back to the weights. Reverse order fits one output with many
inputs: one backward pass gives all the gradients, at about twice the cost of a forward pass.
Senior add-on: the price is memory, because every activation the backward pass needs is kept from the
forward pass; activation checkpointing trades recompute for that memory.
</details>

<details><summary>What does Adam store, and why?</summary>

Two running averages per parameter: of the gradient (momentum, to smooth noise and build speed in
consistent directions) and of the squared gradient (to scale each parameter's step by its typical gradient
size). That is 8 bytes per parameter in 32-bit floats, on top of weights and gradients: about 16 bytes per parameter
in total for mixed-precision training.
</details>

<details><summary>What is the difference between Adam with L2 regularisation and AdamW?</summary>

L2 adds λ·Σw² to the loss, so its gradient goes through Adam's per-parameter scaling, and weights with
large gradients get little decay. AdamW applies the decay directly to the weights, separately from the
adaptive step, so every weight decays at the same rate.
</details>

<details><summary>Why use minibatches rather than the full dataset per step?</summary>

The full-data gradient for an LLM would need a pass over trillions of tokens per step. A minibatch gives a
noisy estimate that is right on average, far more cheaply, so you take many more steps for the same compute. The noise
also helps escape saddle points.
</details>
:::

## Learning rate and schedules

**In short:** the learning rate is the most important setting in training. LLMs ramp it up over the first
steps, then lower it slowly to the end.

The learning rate is usually the first thing to tune. Too high and the loss spikes or turns into NaN ("not a
number"). Too low and training wastes compute. The best value also changes during training, so runs follow a <Term id="lr-schedule">learning rate schedule</Term>:

- **Warmup.** Start near zero and grow linearly to the peak over the first few hundred or thousand steps. At
  the start the weights are random and Adam's averages are still empty, so gradients are large and
  unreliable. Full-size steps then can wreck the run.
- **Decay.** After warmup, lower the rate slowly. Large steps make fast progress early. Later, small steps
  let the weights settle into the bottom of a valley instead of bouncing around it.

The common decay shape for LLMs is **cosine**: the rate follows half a cosine wave from the peak down to a
floor, often 10% of the peak. Llama 2, for example, used 2,000 warmup steps, then cosine decay to 10%.

```text
lr(t) = floor + (peak − floor) · ½ · (1 + cos(π · t / T))      t = step after warmup, T = steps left
```

At t = 0 the cosine is 1, so the rate is the peak. At t = T the cosine is −1, so the rate is the floor.

::: details Going deeper: gradient clipping and other schedules
**Gradient clipping** caps the length of the whole gradient vector: if it is longer than a threshold (often
1.0), scale it down to that length. One bad batch then cannot throw the weights far. Almost every LLM run uses
it.

Cosine decay fixes the total length of the run in advance. **Warmup-stable-decay** schedules hold the rate
flat and decay only at the end, so a run can be extended or branched from a checkpoint before the decay
([Hu et al., 2024, MiniCPM](https://arxiv.org/abs/2404.06395)). Cosine schedules with restarts come from
[Loshchilov and Hutter (2016)](https://arxiv.org/abs/1608.03983).
:::

## Initialisation, and vanishing and exploding gradients

**In short:** the starting weights must be random and of the right size. Otherwise the gradient
shrinks to nothing or blows up as it passes back through many layers.

**Why random?** Start every weight at zero, and every neuron in a layer computes the same thing and gets the
same gradient. They stay identical forever, so the layer is effectively one neuron. Random starting values
break this symmetry.

**Why the right size?** The backward pass multiplies by one factor per layer on the way back. In a network
with 50 layers, the gradient for the first layer is a product of 50 such factors. If they are typically 0.5,
the product is about 10⁻¹⁵: the early layers stop learning. If they are typically 2, the product is about
10¹⁵: the steps become huge and training diverges. These are <Term id="vanishing-gradients">vanishing and exploding gradients</Term>.

The fixes, from oldest to newest:

- **Scaled random initialisation.** Draw each weight with a spread of about 1/√(n_in), where n_in is the
  number of inputs to the layer. Each layer then keeps its outputs at about the same size as its inputs.
  Chapter 1's exercise did exactly this.
- **ReLU-like activation functions.** Older networks used sigmoid (a smooth S-shaped bend from 0 to 1), whose derivative is at most 0.25, so it
  shrinks the gradient at every layer. ReLU's derivative is 1 for positive inputs.
- **Residual connections and normalisation layers.** Transformers, the design behind LLMs, add each block's output to its input and
  rescale values to a standard size. The gradient then has a direct path back through every block
  ([Chapter 4](/foundations/transformer)).
- **Gradient clipping**, capping the gradient's length, handles the occasional explosion that remains.

::: details Going deeper: the names
Draw weights with variance 1/n_in for tanh-like functions ([Glorot and Bengio, 2010](https://proceedings.mlr.press/v9/glorot10a.html))
or 2/n_in for ReLU ([He et al., 2015](https://arxiv.org/abs/1502.01852)). The factor 2 compensates for
ReLU zeroing about half its inputs. LLMs often use a small fixed spread, such as 0.02, and scale down
the weights of layers that feed residual connections, by a factor that depends on depth.
:::

## Overfitting and regularisation

**In short:** the goal is good predictions on new data, not on the training data. A network can memorise its
training set, noise included. A held-out set tells you when that happens.

A student who memorises past exam answers scores 100% on those papers and fails a new one. A big network can
do the same: fit every training example perfectly, including its noise, and predict new examples badly. This
is <Term id="overfitting">overfitting</Term>.

To detect it, split your data before training. Train on one part. Keep a second part, the <Term id="validation-set">validation set</Term>, aside and only measure on it. (A third part, the test set,
is used once at the end, so you do not tune your choices to it.) While both losses fall, the network is
learning patterns that carry over. When the training loss keeps falling but the validation loss rises, it is
memorising.

Methods that fight overfitting are called <Term id="regularization">regularisation</Term>:

- **More data.** The most reliable fix: noise in different examples does not line up, so memorising it
  stops paying.
- **Weight decay.** Keeps weights small, which favours smoother functions.
- **Early stopping.** Keep the checkpoint with the best validation loss.
- **Dropout.** During training, zero a random fraction of activations at each step, so no neuron can rely
  on any single other one.

LLM pretraining is a special case. The data is so large that most of it is seen once, so the training loss
on new batches already measures performance on unseen text. Overfitting matters more in fine-tuning (further training of a finished model for one task) on
small datasets ([Chapter 7](/building/post-training)).

## Try it: the same in PyTorch

**In short:** PyTorch builds the graph and runs the backward pass for you. The script compares SGD,
momentum and Adam, then shows overfitting and two remedies.

Part 1 is the XOR network again: you write only the forward pass and the loss. `loss.backward()` is the
engine you built above. Part 2 trains on two interleaved spirals with minibatches of 32. Part 3 trains a
large network on only 60 noisy points and measures it on 1,000 held-out ones.

<<< @/../exercises/ch02_torch.py

Output (about 30 seconds on 2 CPU cores):

```text
== Part 1: XOR in PyTorch ==
step   0  loss 0.6996
step 100  loss 0.4664
step 200  loss 0.0815
step 300  loss 0.0245
step 400  loss 0.0132
step 500  loss 0.0088
P(1): [0.005 0.991 0.992 0.012]

== Part 2: SGD vs momentum vs Adam on two spirals (batch 32, 1500 steps) ==
optimizer             loss@100    @500   @1500  val acc
SGD        lr=0.05       0.642   0.602   0.195    98.3%
SGD+moment lr=0.05       0.612   0.013   0.002    99.9%
Adam       lr=0.01       0.285   0.003   0.000   100.0%
parameters: 4482; Adam state tensors hold 8964 numbers (2 per parameter)

== Part 3: overfitting; weight decay; more data ==
60 points, no weight decay  
  step  100  train loss 0.044 acc  96.7%   val loss 2.112 acc  73.6%
  step 1000  train loss 0.000 acc 100.0%   val loss 5.612 acc  74.4%
  step 3000  train loss 0.000 acc 100.0%   val loss 6.962 acc  74.5%
60 points, weight decay 1.0 
  step  100  train loss 0.058 acc  96.7%   val loss 1.565 acc  72.8%
  step 1000  train loss 0.009 acc 100.0%   val loss 1.980 acc  76.8%
  step 3000  train loss 0.004 acc 100.0%   val loss 1.925 acc  76.3%
1000 points, no weight decay
  step  100  train loss 0.106 acc  95.8%   val loss 0.174 acc  93.2%
  step 1000  train loss 0.064 acc  97.0%   val loss 0.334 acc  92.8%
  step 3000  train loss 0.046 acc  97.7%   val loss 0.413 acc  93.3%
```

What to read from it:

- **Part 2.** Plain SGD is still on a plateau at step 500 (loss 0.60, close to the 0.69 of pure guessing).
  Momentum escapes it; Adam is fastest. Adam's state is exactly 2 numbers per parameter. One run on a toy
  problem is not a benchmark: well-tuned SGD can match Adam on some tasks.
- **Part 3, 60 points.** Training accuracy reaches 100% while validation accuracy stays near 75%, and
  validation loss climbs: the network memorised the noise and grew ever more confident on wrong answers.
- **Weight decay** keeps validation loss far lower (1.9 instead of 7.0), because the weights cannot grow to
  make wild, confident predictions. Here it barely changes accuracy.
- **More data** helps most: validation accuracy goes from about 75% to 93% with the same network.

Note that validation loss can rise while validation accuracy stays flat. Loss also measures confidence,
so it catches overconfidence that accuracy misses.

## The generalisation puzzle

**In short:** classical theory says a model with far more parameters than data points should overfit.
Big neural networks often do not. Why is still an active research question.

Classical statistics gives a U-shaped picture. Make a model more flexible and test error first falls, as it
captures real patterns. Then it rises, as it starts fitting noise. The advice: pick a model just complex
enough.

Neural networks break this picture. [Zhang et al. (2017)](https://arxiv.org/abs/1611.03530) showed that
standard image networks can perfectly fit training labels that were replaced by random ones. So these
networks are flexible enough to memorise anything. Yet with real labels, the same networks generalise well.
Their capacity alone cannot explain it.

Push model size further and something stranger appears. Test error falls, rises near the size where the
model can just barely fit the training data exactly, then falls again as the model keeps growing. This is <Term id="double-descent">double descent</Term>
([Belkin et al., 2019](https://arxiv.org/abs/1812.11118); [Nakkiran et al., 2019](https://arxiv.org/abs/1912.02292)).
Past that threshold, many weight settings fit the data, and training seems to find smooth ones among them.

Why training prefers those solutions is an area of active research. Candidate explanations include the
implicit bias of gradient descent toward simple solutions, the noise of SGD, and properties of flat minima.
None is settled. In practice, LLMs follow a more predictable rule: bigger models trained on more data reach
lower loss on unseen text ([Chapter 5](/building/pretraining)).

::: warning Thread
Everything in this chapter happens only during training. Learning means weight updates driven by a loss,
and a loss needs a target: the correct next token, the right label. At deployment there is no target, so
there is no loss, no gradient and no update. The forward pass runs; the backward pass never does.

The way SGD learns also differs from your "tricked once, learns forever" intuition. Each step moves the
weights a tiny amount, averaged over a minibatch, and a fact must usually appear many times before the model
learns it. LLMs answer questions about facts that appear in few pretraining documents much worse than about
common ones ([Kandpal et al., 2023](https://arxiv.org/abs/2211.08411)). Making one surprise count by taking a
big step on it tends to damage what the model already knew (catastrophic forgetting,
[Chapter 12](/beyond/continual-learning)).

The counterpoints: cross-entropy *is* a surprise signal, so the machinery for "learn from surprise" exists;
what is missing at deployment is the target. And a model can adapt to a single example given in its prompt,
with no weight change at all ([Chapter 4](/foundations/transformer)). That adaptation disappears when the
conversation ends.
:::

::: tip Interview checkpoint
<details><summary>Your training loss suddenly spikes to NaN. What do you check?</summary>

Most often the learning rate is too high for this stage (missing or too short warmup), or one bad batch gave a
huge gradient. Check gradient norms before the spike, enable gradient clipping, lower the peak learning rate,
and look for numerical overflow, such as exp of a large logit in a hand-written softmax.
</details>

<details><summary>Why warm up the learning rate?</summary>

At the start the weights are random and Adam's running averages are empty, so gradients are large and
Adam's per-weight step sizes are unreliable. Full-size steps then can push the weights into a bad region.
Starting small and ramping up gives the statistics time to settle.
</details>

<details><summary>What are vanishing gradients, and what fixed them?</summary>

The gradient for an early layer is a product of one factor per later layer, so it can shrink toward zero
(or blow up) exponentially with depth. Fixes: scaled random initialisation, ReLU-like activation functions,
residual connections and normalisation layers, and gradient clipping for explosions.
</details>

<details><summary>How do you tell overfitting from underfitting?</summary>

Overfitting: training loss low and still falling, validation loss rising. Underfitting: both losses high and
close together. The first needs regularisation or more data; the second needs a bigger model, longer
training or a better learning rate.
</details>
:::

## In practice

- **Training loops are five lines.** Forward, loss, `zero_grad()`, `backward()`, `step()`. Most bugs live
  around them: forgetting to zero gradients, wrong loss inputs (probabilities instead of logits), or leaving
  dropout on during evaluation.
- **Loss curves are the main dashboard.** Teams watch training loss, validation loss, gradient norm and
  learning rate for every run. Spikes in the gradient norm often come before a loss spike.
- **Memory planning starts at 16 bytes per parameter.** Weights, gradients and AdamW state set the floor for
  training memory; activations come on top and grow with batch and sequence length.
- **The learning rate gets the most tuning.** Large labs fit the best learning rate on small models and
  carry it over to big ones, because a single big run is too expensive to repeat.

## Common misconceptions

- **"Backpropagation is how the network learns."** Backpropagation only computes the gradient. The
  optimizer (SGD, Adam) decides how to change the weights.
- **"Gradient descent finds the best weights."** It finds a low point near where it started. For big
  networks, that turns out to be good enough, for reasons still being studied.
- **"The noise in SGD is just an approximation error."** It also shapes which solutions training finds.
- **"Lower training loss means a better model."** Only held-out loss tells you that.
- **"Adam is free."** Its m and v add two 32-bit floats per parameter: 8 bytes, four times the size of the
  bf16 weights themselves.

## Key takeaways

- The loss is one number for "how wrong". Classification and next-token prediction use cross-entropy:
  −log of the probability given to the right answer.
- Gradient descent steps every weight against the gradient, scaled by the learning rate.
- Backpropagation is the chain rule run backwards over the computation graph: all gradients for about twice
  the forward cost, paid for in saved activations.
- Minibatch SGD trades exact gradients for many cheap, noisy steps. AdamW adds momentum and a per-weight
  step size, at 8 extra bytes per parameter: about 16 bytes per parameter for training versus 2 for inference.
- Generalisation is measured on held-out data. Why huge networks generalise so well is still open.

## Further reading

- [Andrej Karpathy, *The spelled-out intro to neural networks and backpropagation*](https://www.youtube.com/watch?v=VMj-3S1tku0):
  builds micrograd live; the exercise follows the same path.
- [Christopher Olah, *Calculus on Computational Graphs: Backpropagation*](https://colah.github.io/posts/2015-08-Backprop/):
  the clearest explanation of why reverse mode wins.
- [3Blue1Brown, *Backpropagation, intuitively*](https://www.3blue1brown.com/lessons/backpropagation):
  visual intuition for the backward pass.
- [Sebastian Ruder, *An overview of gradient descent optimization algorithms*](https://www.ruder.io/optimizing-gradient-descent/):
  momentum, Adam and relatives side by side.
- [Kingma and Ba, *Adam* (2014)](https://arxiv.org/abs/1412.6980) and [Loshchilov and Hutter, *AdamW* (2017)](https://arxiv.org/abs/1711.05101):
  short papers; read the algorithm boxes.
- [Nakkiran et al., *Deep Double Descent* (2019)](https://arxiv.org/abs/1912.02292): the clearest
  experiments on the generalisation puzzle.

## Review

<Flashcards id="learning" :cards="cards" />

<MarkDone id="learning" />
