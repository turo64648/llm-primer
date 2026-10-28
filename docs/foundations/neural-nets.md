---
title: "1. Neural Networks from Scratch"
---

<script setup>
import { cards } from './neural-nets-review'
</script>

# 1. Neural Networks from Scratch

A neural network is a fixed program that does matrix multiplies, with millions or billions of tunable numbers
plugged in. Every large language model (LLM) is one. Interviewers expect you to explain what a layer
computes, why it needs a nonlinearity, and what a "7B model" physically is.

::: info Before you start
- A <Term id="vector">vector</Term> is a list of numbers. A <Term id="dot-product">dot product</Term> multiplies
  two vectors element by element and adds up the results.
- A **matrix multiply** is many dot products at once: each row of the matrix is dotted with the input.
- You will see NumPy code with arrays and shapes like `(4, 32)`.

[Chapter 0](/start/math-toolkit) covers all of this. This chapter explains what it needs as it goes.
:::

## A neuron: a weighted vote with a threshold

**In short:** a neuron multiplies each input by a weight, adds them up with a bias, and passes the result
through a simple bend. That is all.

Suppose you want a rule that outputs 1 only when two switches are both on. Give each switch a vote of 1.
Add the votes, then subtract 1.5. If the result is positive, output 1; otherwise output 0. Both on gives
0.5, which is positive. One on gives −0.5, which is not.

That rule is a <Term id="neuron">neuron</Term>. The vote sizes are its <Term id="weight">weights</Term>.
The −1.5 is its <Term id="bias">bias</Term>, which sets the threshold. Written out:

```text
z = w1·x1 + w2·x2 + ... + wn·xn + b   =   w · x + b
```

Here x1 … xn are the inputs, w1 … wn the weights, and b the bias. The right-hand form says
the same thing as a dot product of the weight vector **w** with the input vector **x**. The
number z is the neuron's raw score.

The neuron then applies an <Term id="activation-function">activation function</Term> to z. The score z
is a linear function of the inputs: a straight line, or a flat plane in more dimensions. The activation
function is a fixed, simple function that bends it. The AND rule above used a hard step (1 if positive, else 0).
Modern networks use smoother bends, covered below. The output is one number.

::: details Going deeper: the name, and the biology
This unit with a hard step is Rosenblatt's **perceptron** (1958). The word "neuron" is a loose
borrowing from biology. Real neurons spike over time, have complex dendrites and chemistry, and learn by
different rules. Treat the artificial neuron as a weighted sum plus a bend, nothing more.
:::

## Layers are matrix multiplies

**In short:** a layer is many neurons reading the same inputs. Stack their weight vectors as rows of a
matrix, and the whole layer is one matrix multiply plus a bias, then the activation function.

One neuron gives one number. To detect many patterns in the same input, use many neurons side by side, each
with its own weights. A group of neurons that all read the same inputs is a <Term id="layer">layer</Term>.

Put each neuron's weight vector in one row of a matrix W. Then "every neuron computes its dot product"
is exactly a matrix–vector multiply:

```text
h = f(W x + b)
```

x is the input vector, with n_in numbers. W has one row per neuron, so its shape is n_out × n_in.
b holds one bias per neuron. f is the activation function, applied to each number separately. h is the
layer's output: one number per neuron.

<NnNeuronLayerDiagram />

The layer's output becomes the next layer's input. A stack of such layers is a
<Term id="mlp">multi-layer perceptron (MLP)</Term>. The layers between input and output are called
**hidden layers**, because you never see their outputs directly.

Running an input through every layer in order is a <Term id="forward-pass">forward pass</Term>. The
intermediate numbers it produces, such as each hidden layer's h, are the network's
<Term id="activations">activations</Term>. Note the two similar words: the *activation function* is the
bend; the *activations* are the values flowing through.

The weights and biases together are the network's <Term id="parameter">parameters</Term>. A layer from
n_in to n_out numbers has n_in × n_out weights plus n_out biases.

::: details Going deeper: batches, and why rows hold examples in code
In practice you run many inputs at once. Put them as rows of a matrix X of shape
`(batch, n_in)`. Then the layer is `X @ W.T + b`, with shape `(batch, n_out)`.
The bias vector is added to every row (NumPy calls this broadcasting).

This turns many matrix–vector products into one matrix–matrix product. Hardware runs that far more
efficiently, because each weight loaded from memory is reused for every row in the batch. PyTorch's
`nn.Linear` stores W as `(out, in)` and computes exactly `x @ W.T + b`.
:::

## Why the nonlinearity matters

**In short:** without an activation function, any stack of layers equals one single layer. The bend is
what lets depth add power.

Take two layers with no activation function. The first computes Ax, the second computes B(Ax).
Matrix multiplication is associative, so B(Ax) = (BA)x. And BA is one matrix. So the two layers are
one layer in disguise. Biases do not change this: the stack is still one matrix multiply plus one bias.
A hundred linear layers collapse the same way.

A single linear layer is very limited. Look at XOR: output 1 when exactly one of two inputs is 1. Plot the
four inputs on a square. The two "1" cases sit on opposite corners. No straight line puts both of them on
one side and the two "0" cases on the other. The best any linear model can do is output 0.5 for everything.

Now add a bend between the layers. The bend breaks the collapse, because f(Ax) is no longer a
matrix times x. Each hidden neuron can now carve the input space along its own line. The next
layer combines those pieces. With two hidden neurons, XOR becomes easy; the exercise below sets the weights
by hand.

The standard bend is the <Term id="relu">rectified linear unit (ReLU)</Term>:

```text
ReLU(z) = max(0, z)
```

It passes positive values unchanged and turns negative values into 0. It is a straight line with one kink
at zero. That single kink per neuron is enough. Many neurons, each with its kink in a different place,
can build any bent shape you need, like a polyline built from straight segments.

The <Term id="gelu">Gaussian error linear unit (GELU)</Term> is a smooth version of ReLU. It is close to
0 for very negative inputs, close to z for positive ones, and curves gently in between. GPT-2 and BERT
use it. In practice it trains slightly better than ReLU in some models. Training is the topic of
[Chapter 2](/foundations/learning).

::: details Going deeper: older and newer activation functions
- **Sigmoid** `1 / (1 + e^(−z))` and **tanh** were the standard choices before about 2010. They flatten out
  for large inputs, which makes deep stacks hard to train. Chapter 2 explains why.
- **GELU** is `z · Φ(z)`, where Φ is the standard normal cumulative distribution function
  ([Hendrycks and Gimpel, 2016](https://arxiv.org/abs/1606.08415)).
- **SwiGLU**, used in Llama and many recent LLMs, multiplies two linear projections, one passed through a
  smooth ReLU-like function called SiLU ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)). It needs three
  weight matrices per MLP block instead of two.
:::

## Try it: an MLP forward pass in NumPy

**In short:** one neuron computes AND; no linear model computes XOR; two ReLU neurons do; linear layers
collapse; then a random three-layer network shows the shapes and the parameter count.

The script has five short parts. Parts 1–4 use weights set by hand. Part 5 uses random weights, because a
network nobody has trained yet looks exactly like this: right shapes, meaningless outputs.

<<< @/../exercises/ch01_mlp.py

Output:

```text
== Part 1: one neuron computes AND ==
x=[0. 0.]  z=-1.5  out=0
x=[0. 1.]  z=-0.5  out=0
x=[1. 0.]  z=-0.5  out=0
x=[1. 1.]  z=+0.5  out=1

== Part 2: the best purely linear model for XOR ==
best weights and bias: [0.  0.  0.5]
its outputs:           [0.5 0.5 0.5 0.5]  wanted: [0. 1. 1. 0.]

== Part 3: a 2-2-1 network with ReLU solves XOR ==
x=[0. 0.]  hidden=[0. 0.]  out=0
x=[0. 1.]  hidden=[1. 0.]  out=1
x=[1. 0.]  hidden=[1. 0.]  out=1
x=[1. 1.]  hidden=[2. 1.]  out=0

== Part 4: without ReLU, two layers collapse into one ==
two layers: [-4.357  1.785]
one layer:  [-4.357  1.785]
same? True

== Part 5: a random MLP on a batch ==
input         (4, 8)
after layer 1 (4, 32)   W1 (32, 8), b1 (32,)
after layer 2 (4, 32)   W2 (32, 32), b2 (32,)
after layer 3 (4, 3)   W3 (3, 32), b3 (3,)
output for example 0: [-0.102  0.761  0.328]

parameter count: 1443
saved as float32: 5900 bytes (1443 params x 4 bytes = 5772, plus a small header)
```

How the XOR network works: hidden neuron 1 counts how many inputs are on. Hidden neuron 2 is the same count
minus 1, cut at zero by ReLU, so it fires only when both are on. The output is h1 − 2·h2: the count,
with a penalty that cancels the "both on" case. Part 2 is the best straight-line fit (least squares), and
it can do no better than 0.5 everywhere.

In Part 5, the parameter count is (8 × 32 + 32) + (32 × 32 + 32) + (32 × 3 + 3) = 1443. Each
float32 number takes 4 bytes, so the saved file is about 5.8 KB. That file is the whole state of
this network. Nothing else would be needed to restore it.

::: details Going deeper: variations to try
- Remove the `relu` in Part 3. The outputs become 2, 1, 1, 0. Without the kink, neuron 2 is −1 for
  the input (0, 0) instead of 0, so its penalty turns into a bonus there.
- In Part 5, change `sizes` to `[4096, 11008, 4096]`, the shape of one MLP block in Llama 2 7B. Print
  the parameter count (about 90 million, ignoring the third SwiGLU matrix) and time one forward pass.
- Where did the weights in Part 3 come from? A person chose them. For anything bigger, training finds
  them; [Chapter 2](/foundations/learning) builds that.
:::

## What can a network represent?

**In short:** with enough neurons, one hidden layer can approximate any reasonable function. That says the
right weights exist, not that you can find them or that they generalise.

The XOR example hints at a general fact. Each ReLU neuron adds one kink. Enough kinks, placed well, can
trace any continuous curve as closely as you like. A similar argument works with more inputs.

This is the <Term id="universal-approximation">universal approximation theorem</Term>. Stated carefully:
take any continuous function on a bounded, closed region of inputs, and any error tolerance. Then a network
with **one** hidden layer and a non-polynomial activation function (ReLU, GELU, sigmoid all qualify) can
match that function within the tolerance everywhere in the region, if it has enough hidden neurons.

What the theorem does **not** say matters more in interviews:

- **How many neurons.** For some functions, a single hidden layer needs an enormous number of them.
- **How to find the weights.** It proves good weights exist. Training may or may not find them.
- **Generalisation.** It is about matching a known function. It says nothing about predicting well on
  inputs you never trained on, which is the actual goal.
- **Why depth helps.** Deep, narrower networks can represent some functions with far fewer neurons than one
  wide layer. Each layer builds on features found by the layer before.

So the theorem is a reassurance that the model family is expressive enough. It does not explain why neural
networks work well in practice. That question is still partly open.

::: details Going deeper: the sources
[Cybenko (1989)](https://link.springer.com/article/10.1007/BF02551274) proved it for sigmoid activation
functions, and [Hornik (1991)](https://doi.org/10.1016/0893-6080(91)90009-T) for general bounded ones.
[Leshno et al. (1993)](https://doi.org/10.1016/S0893-6080(05)80131-5) showed the exact condition: any
continuous activation function that is not a polynomial. [Telgarsky
(2016)](https://arxiv.org/abs/1602.04485) gave functions that deep networks represent compactly but shallow
networks need exponentially many neurons for.
:::

## What "the model" physically is

**In short:** a trained model is two things. A file of named arrays of numbers, and a short, fixed program
that says how to multiply them. The file is almost all of it.

A model you download, such as an LLM whose weights are published, is a <Term id="checkpoint">checkpoint</Term>. It is a
file (or a few) holding the parameters. Each parameter array has a name and a shape. A multi-dimensional
array like this is called a <Term id="tensor">tensor</Term>. One tensor in Llama 2 7B, for example, is
named `model.layers.0.mlp.up_proj.weight` with shape `[11008, 4096]`.

The second part is the architecture: code that says which tensors to multiply, in what order, with which
activation functions. It is a few hundred lines and is the same for every model of that design. Load
different numbers into the same code, and you get a different model.

A systems analogy: the program is like an interpreter, and the checkpoint is the data it runs on. The
analogy breaks in one place. Nobody wrote the numbers, and they do not read as instructions. The behaviour
you care about lives in billions of numbers found by training, and reading meaning out of them is a
research field of its own.

Three properties follow from this structure:

- **Fixed compute per input.** The program has no loops that depend on the data. A trivial input and a
  hard one cost the same multiplies.
- **Size is parameters times bytes.** Model files are almost pure numbers, so their size is easy to
  estimate.
- **Using the model changes nothing.** Using a trained model is called **inference**. Each forward pass
  reads the weights and writes only activations. The file is identical after a billion requests.

::: details Going deeper: file formats and number formats
Most open models ship as **safetensors** files: a small JSON header listing each tensor's name, shape,
number format and byte offset, followed by the raw bytes. Loaders can memory-map the file and read
tensors in place. Older PyTorch `.bin` files used Python pickle, which can run code when loaded, so
safetensors replaced them.

Common number formats per parameter:
- **float32**: 4 bytes. Used for training state, rarely for shipping LLMs.
- **<Term id="bf16">bf16</Term>** (bfloat16): 2 bytes, same range as float32 but fewer digits. The usual
  format for LLM weights.
- **int8 / 4-bit**: 1 byte or half a byte, after quantisation (rounding each weight to fewer bits, at some cost in
  accuracy). Covered in
  [Inference](/running/inference).
:::

## From a tiny MLP to an LLM

**In short:** an LLM is built from the same parts: matrix multiplies and activation functions. It adds a
few layer types and is far wider and deeper.

Here is the shape of a large language model at the level of this chapter. Text is split into pieces called
tokens, and each token becomes a vector of a few thousand numbers ([Chapter 3](/foundations/tokens-embeddings)).
That vector then passes through a stack of identical blocks, 32 in Llama 2 7B. Each block has two parts:

- **Attention**, which lets each token's vector take in information from earlier tokens. It is built from
  matrix multiplies too ([Chapter 4](/foundations/transformer)).
- **An MLP block**, exactly the kind of network in this chapter: widen from 4096 to 11008 numbers, apply
  the activation function, narrow back to 4096. (Llama's version uses three matrices; see SwiGLU above.)

A final layer turns the last vector into one score for every possible next token. That is the whole model.

Most of the parameters sit in plain layers like the ones in the exercise. In Llama 2 7B, the MLP blocks
alone hold about two-thirds of all parameters. The rest are attention weights and the token lookup tables.
Scaling up mostly means wider matrices, more layers, and much more training.

::: details Going deeper: counting Llama 2 7B
With width 4096, MLP width 11008, 32 blocks and 32,000 tokens in the vocabulary
([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)):
- Each attention part: 4 matrices of 4096 × 4096 ≈ 67 M.
- Each MLP block (SwiGLU, 3 matrices): 3 × 4096 × 11008 ≈ 135 M.
- 32 blocks × 202 M ≈ 6.5 B.
- Token lookup table and output layer: 2 × 32,000 × 4096 ≈ 0.26 B.
- Total ≈ 6.7 B, the "7B". MLP share: 32 × 135 M ≈ 4.3 B, about 64%.

Llama uses no biases in these layers, and many recent LLMs also leave them out. The count above ignores a
few small normalisation vectors.
:::

::: info Back-of-envelope
**How big is a 7B model, and what does one token cost?**

- Parameters: 7 billion.
- In bf16, 2 bytes each: 7 × 10⁹ × 2 = 14 × 10⁹ bytes ≈ **14 GB**.
- In float32: ≈ 28 GB. In 4-bit: ≈ 3.5 GB.
- A 70B model in bf16 is ≈ 140 GB, more than one 80 GB GPU holds.

Compute per token: the model runs one forward pass per token. Each weight is used once, in one multiply
and one add. So a forward pass costs about 2 × parameters operations: 2 × 7 × 10⁹ ≈ **14 billion
floating-point operations (<Term id="flop">FLOPs</Term>)** per token.

Memory traffic per token: generating one token for one user reads all 14 GB of weights. At about 3 TB/s of
GPU memory bandwidth, that is about 5 ms, so at most about 200 tokens per second. The bottleneck is
reading the weights from memory, not the arithmetic. [Inference](/running/inference) builds on this.
:::

::: warning Thread
Training sets the weights; after that they are frozen. Everything the model knows from training lives in
the checkpoint. During use, the only thing that changes is the activations, which exist for one request and
then vanish. Between the two there is nothing: no layer of slowly changing weights that absorbs what
happened today. That is the gap your intuition about "per-token activations and months-frozen weights"
points at.

The strongest counterpoint: activations are not as weak as "per-token" sounds. In an LLM, earlier text in
the conversation shapes the activations for every later token, so a model can pick up a new pattern from
examples in its prompt. [Chapter 4](/foundations/transformer) shows the mechanism. Whether that kind of
short-lived adaptation, plus retrieval and frequent retraining, can stand in for changing weights is the
open question.
:::

::: tip Interview checkpoint
<details><summary>Why do neural networks need nonlinear activation functions?</summary>

A composition of linear layers is a single linear layer: B(Ax) = (BA)x. Without a
nonlinearity, depth adds nothing, and the model cannot even represent XOR. The activation function breaks
the collapse, so each layer can build new features from the previous ones.
</details>

<details><summary>How many parameters does a linear layer from 4096 to 11008 have, and what does one forward pass through it cost?</summary>

4096 × 11008 ≈ 45 M weights, plus 11008 biases if it has them. Each weight is one multiply
and one add per input, so about 90 M floating-point operations per token. In bf16 the weights take about
90 MB.
</details>

<details><summary>What is actually inside a model checkpoint? How large is a 70B model?</summary>

Named tensors of weights (plus biases and a few small scaling vectors, if the design has them), with their shapes and number format. The
architecture code is separate and small. A 70B model in bf16 is 70 × 10⁹ × 2 bytes ≈ 140 GB,
so serving it needs at least two 80 GB GPUs before any working memory.
</details>

<details><summary>What does the universal approximation theorem say, and why is it not the reason deep learning works?</summary>

One hidden layer with enough neurons and a non-polynomial activation function can approximate any
continuous function on a bounded region. It does not say how many neurons, how to find the weights, or
whether the result generalises to new inputs. Those are the questions that matter in practice.
</details>

<details><summary>Does a deployed LLM learn from the conversations it has?</summary>

Not in its weights: a forward pass only reads them. Within one conversation it can adapt, because earlier
text shapes later activations. Lasting changes need a new training run and a new checkpoint.
</details>
:::

## In practice

- **Sizing hardware.** "Will this model fit?" starts with parameters × bytes per parameter. A 7B model in
  bf16 (14 GB) fits on a 24 GB consumer GPU; a 70B model does not fit on one 80 GB GPU without quantisation.
- **Downloads are numbers.** Pulling a model from a hub means pulling tens of gigabytes of tensors.
  Teams cache them like large container images and memory-map them at load time.
- **Same code, many models.** Fine-tuned variants (copies of a model trained further for one task) share
  the architecture code. Only the checkpoint differs, so one serving engine can run all of them.
- **Throughput is matrix-multiply throughput.** GPUs and TPUs are built around fast dense matrix
  multiplies, because that is nearly all a forward pass does.

## Common misconceptions

- **"Artificial neurons work like brain neurons."** They are a weighted sum plus a bend. The name is
  historical.
- **"Activation functions are a minor detail."** Without them, any depth collapses to one linear layer.
- **"Universal approximation means a network can learn anything."** It only says suitable weights exist.
  Finding them, and generalising from them, are separate problems.
- **"The model contains a program someone wrote for each skill."** The code is small and generic. The
  skills live in the trained numbers.
- **"Chatting with a model trains it."** Inference only reads the weights. Any lasting change comes from a
  later training run.

## Key takeaways

- A neuron is a weighted sum plus a bias, passed through an activation function. A layer is
  f(Wx + b): one matrix multiply.
- Without nonlinearity, stacked layers collapse into one. ReLU's single kink per neuron is enough to fix it.
- Universal approximation says good weights exist, not that training finds them or that they generalise.
- A model is a checkpoint of named tensors plus a small fixed program. Size ≈ parameters × bytes: 7B in
  bf16 ≈ 14 GB.
- An LLM uses the same matrix multiplies at huge scale; about two-thirds of its parameters are MLP layers.

## Further reading

- [3Blue1Brown, *Neural networks* series](https://www.3blue1brown.com/topics/neural-networks): the best
  visual intuition for neurons, layers and (next chapter) learning.
- [Michael Nielsen, *Neural Networks and Deep Learning*, chapter 4](http://neuralnetworksanddeeplearning.com/chap4.html):
  an interactive, visual proof of universal approximation.
- [Andrej Karpathy, *Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html): builds
  everything from scratch in code; the path chapters 2–4 follow.
- [Hugging Face, safetensors docs](https://huggingface.co/docs/safetensors): what a checkpoint file
  looks like on disk.
- [Touvron et al., *Llama 2* (2023)](https://arxiv.org/abs/2307.09288): a real LLM's sizes, in a
  readable table.

## Review

<Flashcards id="neural-nets" :cards="cards" />

<MarkDone id="neural-nets" />
