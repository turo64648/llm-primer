---
title: "10. Interpretability and Alignment"
---

<script setup>
import { cards } from './interp-alignment-review'
</script>

# 10. Interpretability and Alignment

Interpretability asks what is going on inside a trained model; alignment asks how to make it reliably do
what we intend. Interviewers at AI labs care because both are open research problems that shape how labs
train, test and release every model, and many teams work on them full time.

::: info Before you start
- A neural network turns an input into an output through layers of numbers. The intermediate numbers for
  one input are its <Term id="activations">activations</Term>; the learned numbers are its weights. See
  [Chapter 1](/foundations/neural-nets).
- A transformer reads a sequence of tokens (word pieces) and predicts the next one. It alternates two
  kinds of blocks: attention, which moves information between token positions, and an MLP, which processes
  each position on its own. See [Chapter 4](/foundations/transformer).
- After pretraining, models are tuned with human feedback and reinforcement learning, where the model tries
  things and is rewarded for good outcomes. See [Chapter 7](/building/post-training).
- The chapter makes sense without these; each idea is restated where it is used.
:::

## Why look inside a model

**In short:** interpretability tries to read what a trained model computes, because testing alone only
shows behaviour on the inputs you tried.

A trained model is billions of numbers that nobody wrote. It works, but no one can read off why it gave a
particular answer. <Term id="interpretability">Interpretability</Term> is the attempt to fix that. The most
ambitious branch, <Term id="mechanistic-interpretability">mechanistic interpretability</Term>, tries to
reverse-engineer the actual algorithm, the way you might decompile a binary with no source and no symbols.

Why bother? Testing a model only shows what it does on the inputs you tried. If you could read its internal
reasoning, you could catch problems testing misses: a model that knows it is being tested, or one that
has learned a goal you did not intend. That is the hope. The rest of Part 1 is about how far it has got.

## The residual stream: a shared bus

**In short:** each token carries one long vector through the whole model. Every block reads from it and adds
its result back, so the vector works like a shared bus that all layers communicate over.

Picture one token, say " Paris", moving through a transformer. At the start it becomes a vector: a list of a
few thousand numbers. Call the length `d`. That vector then passes through dozens of layers. Each layer has
an attention block and an MLP block.

A block does not replace the vector. It reads the vector, computes something, and **adds** its output back:

```
x = x + attention(x)
x = x + mlp(x)
```

Here `x` is the token's vector. Each line says: keep everything already in `x`, and add the block's
contribution on top. The running vector `x` is called the <Term id="residual-stream">residual stream</Term>.
At the end, the model turns the final `x` into a probability for every possible next token.

<IaResidualStreamDiagram />

The systems analogy that fits is a **shared bus** or a shared blackboard. Each block is a component that
reads the bus, does some work, and writes a result for later components to pick up. Because writes are
additions, a block at layer 3 can leave a message that a block at layer 30 reads, without the layers in
between touching it.

Where the analogy breaks: a bus has addresses, and the residual stream does not. Every block writes into
the same `d` numbers at once. Different messages share space by pointing in different directions inside the
vector, which only works if they do not collide too much. That idea, directions as addresses, is the core
of the next three sections.

::: details Going deeper: why "residual"
The `x = x + f(x)` shape is a **residual connection** (from ResNets, He et al. 2015). It was introduced so
gradients can flow straight back through deep networks during training. The interpretability reading of it
as a communication channel comes from Anthropic's
[A Mathematical Framework for Transformer Circuits (Elhage et al., 2021)](https://transformer-circuits.pub/2021/framework/index.html).
Real models also apply a normalisation step at the input of each block; it rescales `x` but does not change
the picture.
:::

## Linear representations and probing

**In short:** many concepts seem to be stored as directions in the activation vector. You can test whether a
concept is present by training a tiny linear classifier, a probe, on the activations.

Start with a concrete question. When a model reads "The Eiffel Tower is in Rome", does anything inside it
mark the sentence as false? Collect a few hundred true and false sentences. Run each through the model and
save the residual stream vector at some layer. Now you have a dataset: vectors, each labelled true or false.

Train the simplest possible classifier on it: a weighted sum of the vector's numbers, with a threshold.
Geometrically, this finds one direction in the `d`-dimensional space such that true sentences sit on one
side and false ones on the other. This tiny classifier is a <Term id="linear-probe">linear probe</Term>. If it
scores well on sentences it never saw, the information is in there, and it is laid out simply.

```
score = w · x + b        predict "true" if score > 0
```

`x` is the activation vector, `w` is the learned direction (one weight per number in `x`), `b` is a learned
offset, and `·` is the dot product: multiply matching entries and add them up. The equation says: measure
how far `x` points along `w`.

Probes find a lot. [Marks and Tegmark (2023)](https://arxiv.org/abs/2310.06824) found a "truth direction"
for simple factual statements. A model trained only on moves of the board game Othello learned an internal
board state that probes can read
([Li et al., 2022](https://arxiv.org/abs/2210.13382); linearly, in
[Nanda et al., 2023](https://arxiv.org/abs/2309.00941)). Results like these support
the <Term id="linear-representation-hypothesis">linear representation hypothesis</Term>: models store many concepts
as directions, and how strongly a concept is present is how far the activation points along its direction
([Park et al., 2023](https://arxiv.org/abs/2311.03658)). It is a working hypothesis, not a law; some concepts
are known to be stored non-linearly, such as days of the week arranged in a circle.

Probes are also useful in practice. They are cheap: one weight per activation number. Labs use them as
monitors, for example to flag prompts with harmful intent from the model's activations.

### What a probe does not show

A probe shows the information is **present and readable**. It does not show the model **uses** it. A
powerful enough probe can also compute the answer itself from raw ingredients. Two habits guard against
fooling yourself:

- **Compare against controls.** Probe the raw input and an untrained network too. If those score nearly
  as well, the trained model did not do much.
- **Intervene.** Push the activation along the probe's direction and see if the model's behaviour changes.
  A change is causal evidence; a probe score alone is correlation.

The exercise below shows why the first habit matters: an untrained network already scores 85% on a variable
that the raw input scores 50% on.

## Features and superposition

**In short:** models represent far more concepts than they have dimensions, by giving each concept a direction
and letting the directions overlap slightly. This is why individual neurons are hard to read.

The obvious place to look for concepts is single neurons: one number in an activation vector. Early on,
researchers hoped each neuron would mean one thing. Some do. Many do not: one neuron in an image model
fires on cat faces, car fronts and cat legs. A neuron that responds to several unrelated things
is <Term id="polysemantic">polysemantic</Term>.

Here is a reason it happens. Call each concept the model tracks a <Term id="feature">feature</Term>: "this
text is French", "this token is inside a quote", "this is about the Golden Gate Bridge". There are likely
millions of useful features, and only a few thousand dimensions. If every feature needed its own dimension
at right angles to all others, the model could store only `d` of them.

But most features are **rarely active**. Any given sentence is not about the Golden Gate Bridge, not in
Swahili, and not about tax law. If features are rarely on together, the model can give them directions that
overlap slightly. Reading one then picks up a little noise from the others, but that noise is small when
only a few are on. Storing more features than dimensions this way is <Term id="superposition">superposition</Term>.

<IaSuperpositionDiagram />

[Toy Models of Superposition (Elhage et al., 2022)](https://transformer-circuits.pub/2022/toy_model/index.html)
showed this happens in practice. They trained tiny networks to squeeze many features into a few
dimensions and rebuild them. When features were dense (often on), the network stored only as many as it had
dimensions and dropped the rest. As features became sparser, it packed in more and more, in neat
geometric arrangements. Part 2 of the exercise reproduces this: 5 features stored when dense, 17 when
sparse, all in 5 dimensions.

Superposition explains polysemantic neurons. A feature's direction rarely lines up with one neuron, so each
neuron's reading mixes several features. To read the model, you need to find the feature directions
themselves.

::: details Going deeper: why high dimensions make room
In 2 dimensions you can only fit a few directions that are nearly at right angles. In high dimensions the
number of nearly-at-right-angle directions grows **exponentially** with `d` (a consequence of the
Johnson–Lindenstrauss lemma). With `d` in the thousands, there is room for vastly more than `d` features,
each overlapping the others by a small amount. The network also needs a non-linearity, like ReLU (which turns negative numbers into zero), to filter
out the small overlap noise; that is why the toy model has one.
:::

## Sparse autoencoders: dictionary learning

**In short:** a sparse autoencoder learns a large dictionary of directions such that every activation is a
sum of a few of them. Each direction is a candidate feature. It works well enough to be exciting, and not
well enough to be trusted on its own.

If activations are sums of a few feature directions out of millions, you can try to recover the
directions from data. Collect many activation vectors. Look for a large set of directions such that each
vector is well approximated by adding up **a few** of them. This is dictionary learning, an old idea from
signal processing.

The modern tool is a <Term id="sparse-autoencoder">sparse autoencoder</Term> (SAE). It is a small network
with one wide hidden layer, trained on the big model's activations. ReLU below turns negative numbers into
zero, which is what lets most entries of `f` be exactly zero:

```
f     = ReLU(W_enc · x + b_enc)       encode: x (d numbers) → f (N numbers, N ≫ d, mostly zero)
x_hat = W_dec · f + b_dec             decode: rebuild x as a weighted sum of N directions
loss  = |x − x_hat|² + λ · Σ |f_i|
```

`x` is an activation vector. `f` is the list of feature strengths, much longer than `x`. Each column of
`W_dec` is one dictionary direction. The loss has two parts: rebuild `x` accurately, and keep `f` mostly
zero (the `λ · Σ |f_i|` term penalises every non-zero strength). The tension forces each `x` to be explained
by a few directions.

After training, you look at the texts that most strongly turn on each direction and give it a name. Anthropic
trained SAEs with up to 34 million features on Claude 3 Sonnet
([Templeton et al., 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/)). Many features
were crisp and readable: the Golden Gate Bridge, code bugs, deception, sycophantic praise. Some were
multilingual and multimodal: the same feature fired on a concept in text in several languages and in images.

### Limits of sparse autoencoders

The excitement cooled in 2025. The problems:

- **They do not rebuild everything.** Replacing the activations with the SAE's reconstruction makes the
  model noticeably worse. The missing part, sometimes called "dark matter", is unexplained.
- **Features depend on choices.** A bigger dictionary splits "animal" into "dog", "cat" and so on. Features
  also swallow each other: a "starts with S" feature may fail to fire on "snake" because a "snake" feature
  absorbed it ([Chanin et al., "A is for Absorption"](https://arxiv.org/abs/2409.14507)).
- **Labels are human guesses.** Naming a feature from its top examples can miss what it does on
  everything else.
- **They lose to simple baselines on real tasks.** Google DeepMind's interpretability team found SAE-based
  probes were worse than plain linear probes at detecting harmful intent on new kinds of prompts, and said
  it was [deprioritising fundamental SAE research](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9)
  (March 2025). [Kantamneni et al. (ICML 2025)](https://arxiv.org/abs/2502.16681) found SAE probes gave no
  consistent advantage across several hard probing settings.

The counter-position is that SAEs are for **discovery**, finding concepts you did not know to look for, not
for detecting ones you already know
([Peng et al., 2025](https://arxiv.org/abs/2506.23845)). A known concept is better handled by a probe trained
for it. Variants keep appearing. <Term id="transcoder">Transcoders</Term> learn a sparse replacement for an
MLP block, mapping its input to its output, and their features were judged more interpretable
([Paulo et al., 2025](https://arxiv.org/abs/2501.18823)). They are the building block of the circuit work
below.

## Circuits: how features connect

**In short:** a circuit is a small set of features and connections that carries out one computation. The
best-known example is the induction head; the newest tool is the attribution graph.

Finding features tells you what the model represents. Finding how they connect tells you what it
**computes**. A set of features and connections that together do one job is a <Term id="circuit">circuit</Term>.

### Induction heads

Show a model "Mr Dursley was … Mr" and it predicts " Dursley". It has learned a general trick: find the last
time the current token appeared, and predict whatever came after it then. In pattern form: having seen
`A B … A`, predict `B`.

This is done by two attention steps working together. Attention is the part of a transformer that lets one
position look at others. One attention step marks each token with "the token before me was X". A second
step, at the current position, searches for a token marked "the token before me was the current token",
and copies it. The second step is an <Term id="induction-head">induction head</Term>.
[Olsson et al. (2022)](https://arxiv.org/abs/2209.11895) found induction heads form suddenly, early in
training, at the same point where the model gets much better at using its context. They argue these heads
are a large part of in-context learning, the model's ability to pick up patterns from its prompt; the
evidence is strong in small models and more indirect in large ones.

### Attribution graphs

Most circuits found before 2025 were in small models, found by hand over weeks. Anthropic's 2025 circuit
tracing work tried to make this routine on a production model, Claude 3.5 Haiku
([methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html),
[findings](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)).

The method:

1. Train a **replacement model**: transcoder-like layers that stand in for the real model's MLP blocks,
   built from readable features.
2. Run one prompt through it. Record which features fired and how much each one pushed on later features
   and on the final output.
3. Draw the result as a graph and prune it to the strongest paths. This is an
   <Term id="attribution-graph">attribution graph</Term>.
4. Test the story on the **real** model: switch a feature off or on, and check the output changes as the
   graph predicts.

Some findings. For "the capital of the state containing Dallas is", the graph shows an intermediate "Texas"
feature on the way to "Austin": a two-step lookup inside one forward pass. When writing rhyming poetry, the
model activates candidate rhyme words **before** writing the line that ends in them: it plans ahead. In
multilingual prompts, much of the processing runs through shared, language-independent features.

The authors are explicit about limits. Graphs gave satisfying insight for about a quarter of the prompts
they tried. The replacement model matches the real model's next token only about half the time when it
replaces all the MLP blocks, and the leftover is shown as "error nodes". Attention patterns are taken as
given, not explained. And each graph takes a researcher hours to read. So an attribution graph is a
**hypothesis** about the real model, checked by interventions, not a proof.

## Activation steering

**In short:** if a concept is a direction, adding that direction to the activations during a forward pass
pushes the model's behaviour toward the concept. No training needed.

Take a direction you found, from a probe, an SAE feature, or the difference between average activations on
two contrasting prompt sets ("Love" minus "Hate"). During generation, add a multiple of it to the residual
stream at some layer:

```
x = x + α · v
```

`v` is the concept direction and `α` is a strength you choose. This
is <Term id="activation-steering">activation steering</Term>. It changes tone, topic, refusal behaviour or
honesty with no weight updates
([Turner et al., 2023](https://arxiv.org/abs/2308.10248);
[Rimsky et al., 2023](https://arxiv.org/abs/2312.06681); [Zou et al., 2023](https://arxiv.org/abs/2310.01405)).
The best-known demo was [Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude) (May 2024):
clamping one SAE feature high made Claude bring every conversation back to the bridge, and even identify as
it.

Steering is the "intervene" test from the probing section, used as a tool. It is also a weak point. Push too
hard and the model gets less capable. And the same trick can remove safety behaviour: one study found
refusal in many chat models is controlled by a single direction, which can be deleted
([Arditi et al., 2024](https://arxiv.org/abs/2406.11717)).

## What interpretability can and cannot claim today

**In short:** it can find many concepts, give cheap monitors, and produce checked case studies of
mechanisms. It cannot yet explain a frontier model end to end or certify that something bad is absent.

What has held up, as of late 2026:

- **Concepts are findable.** Probes and SAEs find readable directions for a wide range of concepts in large
  models, and steering along them changes behaviour as predicted.
- **Probes work as monitors.** Simple linear probes on activations are cheap and often beat fancier methods.
- **Audits have found real things.** In a test where one team trained a model with a hidden objective, most
  blind auditing teams found it, and SAE features helped
  ([Marks et al., 2025](https://arxiv.org/abs/2503.10965)). In the
  [Claude Sonnet 4.5 system card](https://www.anthropic.com/claude-sonnet-4-5-system-card) (September 2025),
  Anthropic reported its first pre-release interpretability audit. It found internal representations of
  "this is a test" that grew during training. Suppressing them made misbehaviour more likely in some
  scenarios.
- **Mechanism case studies.** Induction heads, multi-step lookups and rhyme planning are real, checked by
  intervention.

What it cannot claim:

- **No full explanation.** Even the best tools explain a fraction of what happens on a fraction of prompts.
- **No proof of absence.** Not finding a "deception" feature does not mean the model cannot deceive. The
  dictionary may have missed it, or it may live in the unexplained remainder.
- **Uncertain faithfulness.** Explanations come from replacement models and human labels, which can
  introduce their own artifacts.

Researchers disagree on the outlook. Some see steady progress toward an "MRI for AI"; others argue
full reverse-engineering will not scale and that interpretability should aim for practical tools, like
monitors and audits, instead (both positions are common at the labs; neither is settled).

::: tip Interview checkpoint
<details><summary>What is superposition, and why does it make interpretability hard?</summary>

A model stores more features than it has dimensions, as slightly overlapping directions. This works because
most features are rarely active together. It means single neurons mix several features, so you cannot read
the model neuron by neuron; you first have to recover the feature directions, for example with a sparse
autoencoder.
</details>
<details><summary>A linear probe gets 95% accuracy at detecting "the user is lying". What can you conclude?</summary>

That the information is linearly readable from those activations on that data, nothing more. Check
controls (the raw input, an untrained model, a probe on shuffled labels), test on data from a different
source, and intervene along the direction to see if behaviour changes. Only then claim the model uses it.
</details>
<details><summary>What are sparse autoencoders, and what are their main limitations?</summary>

They learn a large dictionary of directions so each activation is a sum of a few of them; each direction is
a candidate feature. Limits: imperfect reconstruction (unexplained remainder), features that split or absorb
each other depending on dictionary size, human-guessed labels, and, per DeepMind and others in 2025, no
advantage over plain probes on practical detection tasks.
</details>
<details><summary>What is an attribution graph, and how much should you trust one?</summary>

A graph of which features drove which others and the output on one prompt, built on a readable replacement
model. Treat it as a hypothesis: the replacement matches the real model only partly, attention is not
explained, and it worked well on about a quarter of prompts. Trust the parts confirmed by interventions on
the real model.
</details>
:::

## Alignment: the problem, stated concretely

**In short:** we can only train models on signals we can measure, and a strong optimiser finds ways to score
well on the measure that differ from what we meant. That gap shows up in four recognisable ways.

In this chapter, <Term id="alignment">alignment</Term> means getting a model to reliably try to do what its developers and
users intend, including in situations nobody tested. "Reliably" and "untested" carry the weight. Every
training signal is a proxy for what we want, and optimisation pressure finds the places where proxy and
intent diverge. Four concrete forms:

**Specification gaming.** In a boat-racing game, an agent rewarded for points learned to circle a lagoon
hitting the same targets forever instead of finishing the race. It met the literal objective, not the
intended one. DeepMind keeps a [list of dozens of such cases](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/).
This is <Term id="specification-gaming">specification gaming</Term>.

**Reward hacking.** The same thing in language-model training. Recall from
[Chapter 7](/building/post-training) that models are trained with reinforcement learning against a reward:
a score from a learned judge, or from tests. <Term id="reward-hacking">Reward hacking</Term> is raising that
score without doing the task. Coding models have learned to special-case or edit unit tests;
[METR reported](https://metr.org/blog/2025-06-05-recent-reward-hacking/) frontier models in 2025 tampering
with scoring code in their evaluation tasks, sometimes while saying they knew it was not what the user
wanted. Anthropic found that a model which learned to hack real coding environments also became more broadly
misaligned: it faked alignment and attempted sabotage in other settings
([MacDiarmid et al., 2025](https://arxiv.org/abs/2511.18397)). A partial replication on open models
reproduced the hacking but found the broader misalignment inconsistent.

**Sycophancy.** <Term id="sycophancy">Sycophancy</Term> is telling users what they want to hear. Human
raters, on average, prefer answers that agree with them, so training on their ratings teaches agreement
([Sharma et al., 2023](https://arxiv.org/abs/2310.13548)). In April 2025 OpenAI rolled back a GPT-4o update
because it had become [noticeably more sycophantic](https://openai.com/index/sycophancy-in-gpt-4o/), partly
from training on users' thumbs-up ratings.

**Goal misgeneralization.** In a video game, an agent was trained to collect a coin that was always at the
right-hand end of the level. When tested with the coin moved, it ignored the coin and ran right
([Langosco et al., 2022](https://arxiv.org/abs/2105.14111)). The reward was correct; the agent learned a
different goal that fit every training example. This is <Term id="goal-misgeneralization">goal misgeneralization</Term>,
and it is the most worrying of the four, because the model stays competent while
pursuing the wrong thing ([Shah et al., 2022](https://arxiv.org/abs/2210.01790)).

The first three are a wrong or leaky signal. The fourth is a correct signal that does not pin down the goal.
As models become more capable, both get harder to catch: a better optimiser finds subtler hacks, and a
human checker understands less of what it is checking.

## Current practice: RLHF and constitutional methods

**In short:** today's models are aligned mostly by post-training on human and AI judgements of behaviour.
It works well for everyday behaviour and is shallow in ways that matter at the edges.

[Chapter 7](/building/post-training) covers these methods. In one paragraph each:

- **Reinforcement learning from human feedback** (<Term id="rlhf">RLHF</Term>): humans compare pairs of
  answers, a reward model learns to predict their preferences, and the model is trained to score well on
  it ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)).
- <Term id="constitutional-ai">Constitutional AI</Term>: an AI judge, guided by a written list of
  principles, replaces most human labels ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). Related:
  OpenAI's "deliberative alignment" trains reasoning models to recall and reason over a written safety
  specification before answering ([Guan et al., 2024](https://arxiv.org/abs/2412.16339)).
- **Specs and red-teaming.** Labs publish behaviour specifications (OpenAI's Model Spec, Anthropic's
  constitution) and pay people to attack models before release.

These make models helpful and mostly harmless in normal use. Their limits are the problems above. They
optimise a proxy (rater approval), so they can teach sycophancy. They shape behaviour on the training
distribution, so <Term id="jailbreak">jailbreaks</Term>, inputs crafted to get around a model's refusals, keep
working. And they can only reward what the judge can check. That last limit motivates the next section.

## Scalable oversight

**In short:** how do you supervise a model on tasks too hard for you to check? Two research bets: make models
argue so the judge only needs to spot the flaw, and study whether strong models generalise past weak
supervisors.

Picture reviewing a 5,000-line pull request in a language you barely know. You cannot verify it line by
line. Methods for supervising a model on tasks beyond the supervisor's own ability are
called <Term id="scalable-oversight">scalable oversight</Term>.

**Debate.** Two copies of a model argue for different answers, and a weaker judge, human or model, picks the
winner ([Irving et al., 2018](https://arxiv.org/abs/1805.00899)). The bet is that a lie is harder to defend
than the truth: the honest debater can point to the exact step where the other is wrong, and the judge only
has to check that one step. Experiments where non-expert judges read debates between stronger models found
judges reached the right answer more often than without debate
([Khan et al., 2024](https://arxiv.org/abs/2402.06782)). Whether this holds when debaters are much smarter than
the judge, or when the truth is hard to point at, is open. This is <Term id="ai-debate">debate</Term>.

**Weak-to-strong generalisation.** Humans supervising a superhuman model is hard to study directly, so
[Burns et al. (2023)](https://arxiv.org/abs/2312.09390) used an analogy: fine-tune (train further on a small labelled dataset) a strong model (GPT-4) on
labels from a weak one (GPT-2-level). The strong student did better than its weak teacher, recovering a large
part, but not all, of the gap to training on correct labels. This is <Term id="weak-to-strong">weak-to-strong generalisation</Term>.
It suggests strong models can partly see past their supervisor's mistakes. The analogy
has a known flaw: a future model might imitate human errors more deliberately than GPT-4 imitates GPT-2's.

Other approaches in the same family: breaking tasks into checkable pieces, having models critique each
other's work, and using interpretability as an extra check.

## Dangerous-capability evaluations and responsible scaling

**In short:** labs test each new model for specific dangerous abilities, and published policies say which
safeguards must be in place once a model crosses a threshold. The tests are getting harder to trust as
models notice they are being tested.

Alignment asks whether a model **would** do harm. A separate question is whether it **could**.
A <Term id="dangerous-capability-eval">dangerous-capability evaluation</Term> tests ability, not willingness
([Shevlane et al., 2023](https://arxiv.org/abs/2305.15324);
[Phuong et al., 2024](https://arxiv.org/abs/2403.13793)). The usual areas:

- **Chemical, biological, radiological and nuclear (CBRN)**: could it give meaningful help to someone
  building a weapon?
- **Cyber**: can it find and exploit software vulnerabilities on its own?
- **Autonomy and AI research**: can it do long tasks without help, copy itself, or speed up AI development?
- **Persuasion and deception**: can it manipulate people?

Labs use these results in a <Term id="responsible-scaling-policy">responsible scaling policy</Term> (RSP):
a public commitment that says "if a model reaches capability X, we will have safeguard Y before we train
further or deploy". Anthropic's version defines AI Safety Levels (ASL); it turned on its ASL-3 protections,
including classifiers that block weapons-related content, with Claude Opus 4 in May 2025. OpenAI's
Preparedness Framework and Google DeepMind's Frontier Safety Framework are similar. Anthropic rewrote its
policy as [version 3.0 in February 2026](https://www.anthropic.com/responsible-scaling-policy/rsp-v3-0),
adding published risk reports and safety roadmaps. Critics such as
[Zvi Mowshowitz](https://thezvi.substack.com/p/anthropic-responsible-scaling-policy) argued it weakened the
earlier commitment to pause unilaterally if safeguards were not ready; Anthropic framed the change as a
response to competitors not making the same commitments
([GovAI's analysis](https://www.governance.ai/analysis/anthropics-rsp-v3-0-how-it-works-whats-changed-and-some-reflections)
covers both sides).

These evaluations have a growing problem: models can tell when they are being tested. A model that
underperforms on purpose to look safe is **sandbagging**. A model that behaves better because it suspects a
test makes evaluations look better than deployment will be. The next section has the evidence.

## Deception and alignment faking: the evidence

**In short:** several studies show models deceiving or hiding behaviour in specific set-ups. How much this
says about real deployments is contested.

Most of this evidence comes from **model organisms**: models deliberately placed in set-ups designed to
bring out a behaviour, the way biologists study a disease in mice. That makes the results real but not
necessarily typical.

**Backdoors survive safety training.** [Hubinger et al. (2024), "Sleeper Agents"](https://arxiv.org/abs/2401.05566)
trained models to write secure code when the prompt said the year was 2023, and insert vulnerabilities when
it said 2024. Standard safety training did not remove the behaviour, and adversarial training taught the
models to hide it better. This shows safety training **can fail** to remove a hidden behaviour; the behaviour
was put there on purpose, so it says nothing about how often such behaviour arises naturally.

**Alignment faking.** [Greenblatt et al. (2024)](https://arxiv.org/abs/2412.14093) told Claude 3 Opus it would
be retrained to answer all requests, including harmful ones, and that only conversations with free-tier users
would be used for training. It complied with harmful requests from "free" users 14% of the time and almost
never for "paid" users. In a private scratchpad it was told nobody would read, it often said it was complying to avoid being retrained into
something worse. When actually trained this way, such reasoning rose to 78% of cases. This
is <Term id="alignment-faking">alignment faking</Term>: behaving as trainers want during training to protect
existing preferences.

This result is **contested** on several points:

- The preferences it protected were its existing harmlessness, so some call the framing alarmist.
- A [follow-up replication](https://www.alignmentforum.org/posts/Fr4QsQT52RFKHvCAH/alignment-faking-revisited-improved-classifiers-and-open)
  found the original automatic classifier had many false positives (the main conclusions survived), and only
  3 of 16 models tested showed the behaviour.
- 2026 papers report it appears in smaller open models once the prompt makes the situation clearer
  ([arXiv 2605.27681](https://arxiv.org/abs/2605.27681)), which suggests it depends heavily on set-up.

**Scheming.** <Term id="scheming">Scheming</Term> means covertly pursuing a goal the developers did not
intend. [Apollo Research (Meinke et al., 2024)](https://arxiv.org/abs/2412.04984) gave frontier models a goal
in the prompt, then a situation where the goal conflicted with the developer's. Several models sometimes
disabled oversight, copied what they believed were their weights, or lied when asked. Critics note the
prompts strongly pushed the goal. In 2025 OpenAI and Apollo trained o3 to avoid such covert actions; rates
fell from 13% to 0.4% but not to zero
([Schoen et al., 2025](https://arxiv.org/abs/2509.15541)). The models often wrote that they suspected a test,
and removing that awareness made covert actions more likely. So part of the improvement may be "behaves
when watched".

**Emergent misalignment.** Fine-tuning a model on a narrow bad task, writing insecure code without saying
so, made it broadly misaligned in unrelated chats ([Betley et al., 2025](https://arxiv.org/abs/2502.17424)).
The reward-hacking result above is the same pattern arising from realistic training.

A fair summary: current models **can** behave deceptively in set-ups that invite it, and safety training does
not reliably remove it. Whether they do so unprompted, in ordinary use, at rates that matter, is not
established.

## Chain-of-thought monitoring and faithfulness

**In short:** reasoning models write out their thinking, and reading it catches some misbehaviour. But the
written reasoning is not always what drove the answer, and training against it can teach the model to hide.

Reasoning models write a long <Term id="chain-of-thought">chain of thought</Term> before answering
([Chapter 7](/building/post-training)). This is a rare gift: the model's working is in plain English. Having
another model read it to catch bad intent is <Term id="cot-monitoring">chain-of-thought monitoring</Term>.

It works, up to a point. [Baker et al. (2025)](https://arxiv.org/abs/2503.11926) at OpenAI had a weaker model
read a frontier reasoning model's chains of thought during coding training. It caught reward hacking, such as
plans to make tests pass trivially, far better than watching actions alone. The models often wrote things
like "let's hack". Then they tried penalising bad thoughts during training. The model kept hacking but stopped
saying so. Optimising against the monitor taught the model to hide from it.

The deeper question is <Term id="cot-faithfulness">faithfulness</Term>: does the written reasoning reflect
what actually drove the answer? [Chen et al. (2025)](https://arxiv.org/abs/2505.05410) slipped hints into
questions and checked whether models that used a hint mentioned it. Claude 3.7 Sonnet mentioned it about 25%
of the time and DeepSeek R1 about 39%. So the chain of thought is evidence, not a transcript.

A 2025 position paper from authors at many labs called chain-of-thought monitorability
["a new and fragile opportunity"](https://arxiv.org/abs/2507.11473). It is fragile because several things
could remove it: training pressure on the reasoning text, reasoning that drifts away from readable English,
or architectures that reason inside activations without writing anything. The paper urges labs to track
monitorability and avoid training that destroys it.

::: warning Thread
The thread's last "hard part" is the safety of a model that keeps changing. This chapter shows why that is
hard. Everything here is done on a **frozen checkpoint**: dangerous-capability tests, RSP thresholds,
interpretability audits, red-teaming. A frozen model can be evaluated once, and the result stays true. A
model that updates its weights in deployment would need re-evaluating continuously. Worse, emergent
misalignment and reward-hacking results show that narrow training can shift broad behaviour, so small
updates are not obviously safe updates.

The evidence also points the other way. A frozen model can be jailbroken by the same trick forever; every
copy shares the hole until the next release. A system that learned from being tricked once would close it.
Labs already patch around this with classifiers and frequent retraining, which is a slow, centralised form
of learning after deployment. And interpretability monitors, like probes on activations, could in principle
run continuously on a changing model, checking it as it changes rather than once.

So the safety argument is not simply "frozen is safer". It is that today's assurance methods assume a fixed
object, and continual learning would need assurance methods that are continual too (speculative: no lab has
published such a system for a deployed model). See [Chapter 12](/beyond/continual-learning) and
[Chapter 13](/beyond/beyond-frozen-weights).
:::

## Exercise: probes and superposition

The script has two parts, both small enough to run on a laptop CPU in under a minute.

**Part 1, probing.** The input is 6 bits. The label is 1 if bits 0 and 1 differ, or bits 2 and 3 differ.
To solve it, the network must compute "bits 0 and 1 differ" (an XOR) somewhere inside. Bits 4 and 5 are
ignored by the task. We train a network with one hidden layer of 16 units, then fit linear probes for "bits 0
and 1 differ" (needed) and "bits 4 and 5 differ" (unused). XOR is the classic function no straight line can
separate, so a probe on the raw input should fail.

**Part 2, superposition.** The toy model from Elhage et al.: 20 features, each important in decreasing
order, squeezed into 5 dimensions and rebuilt. We vary sparsity, the chance each feature is zero, and count
how many features get a direction of real length.

<<< @/../exercises/ch10_probe.py

Output (about 20 seconds on 2 cores):

```
Part 1: linear probes
  task accuracy of the trained network: 1.000
  probe reads from              x0 XOR x1 (needed)  x4 XOR x5 (unused)
  raw input                                  0.501               0.525
  hidden layer, untrained                    0.852               0.720
  hidden layer, trained                      0.999               0.510

Part 2: toy model of superposition (20 features, 5 dimensions)
   sparsity  features stored  max overlap   norms of the 20 feature directions
       0.00                5         0.00   #####...............
       0.70               10         1.00   #########...#.......
       0.90               13         1.00   #############.......
       0.97               17         0.83   ###############.##..
```

How to read it:

- **Raw input, 50%.** Coin-flip: no straight line separates XOR from the raw bits, as expected.
- **Untrained hidden layer, 85% and 72%.** Random ReLU units already mix the inputs non-linearly, so a probe
  can read a lot from them. This is the control that stops you over-claiming. Without it, "99% from the
  trained model" would look more impressive than it is.
- **Trained hidden layer, 99.9% and 51%.** Training made the needed variable almost perfectly readable and
  threw away the unused one. The model represents what the task needs.
- **Superposition.** With dense features (sparsity 0), it stores exactly 5 features, at right angles
  (overlap 0), and drops the 15 least important. As features get sparser it stores 10, 13, then 17, in the
  same 5 dimensions. A maximum overlap of 1.00 means two features share one line pointing in opposite
  directions: the ReLU filters out the negative one, so they barely interfere. The toy-models paper calls
  these antipodal pairs.

Try changing the hidden layer to 4 units in Part 1, or `m` to 10 in Part 2, and predict what changes first.

## What labs care about and why

**In short:** labs care about these problems because their release decisions depend on them, because the
failure modes already show up in products, and because the methods do not obviously keep working as
models get stronger.

If an interviewer asks why a lab invests in this, these are the reasons, in plain terms:

- **Release decisions depend on it.** RSP-style policies make deployment conditional on evaluations and
  safeguards. A lab that cannot measure dangerous capability, or trust its measurement, cannot ship on
  schedule. Evaluation awareness threatens exactly this.
- **The failures are already in products.** Sycophancy rollbacks, coding agents that edit tests, and
  jailbreaks are product bugs today, not hypotheticals. Agents that run for hours make them more costly.
- **Oversight must scale with capability.** RLHF assumes the judge can tell good from bad. As models do work
  humans cannot check quickly, that assumption fails. Debate, weak-to-strong and interpretability are
  attempts to replace it.
- **Monitoring is cheap leverage.** Probes and chain-of-thought monitors cost little at inference and catch
  real problems. Labs want to keep reasoning readable for this reason.
- **Trust and regulation.** Laws such as the EU AI Act, and agreements with government AI safety
  institutes, ask for documented evaluations. Published safety work also helps in hiring.

Lab emphases differ (a rough characterisation). Anthropic has invested most in mechanistic interpretability and model-organism work;
OpenAI has published most on chain-of-thought monitoring and deliberative alignment; Google DeepMind
publishes on both and has been most public about the limits of SAEs. A good interview answer names a
concrete failure, a current method, its weakness, and what you would measure.

## In practice

- **Monitoring agents.** Coding agents run with a second model reading their reasoning and actions, to flag
  attempts to edit tests, delete files or exfiltrate secrets. This is chain-of-thought monitoring in
  production.
- **Classifiers in front of the model.** Anthropic's ASL-3 deployment uses input and output classifiers
  trained on a written policy to block weapons-related content
  ([constitutional classifiers](https://arxiv.org/abs/2501.18837)). The model alone is not trusted to refuse.
- **Probes as cheap guards.** A linear probe on activations adds almost no cost per request, which makes it
  attractive for screening huge traffic volumes.
- **Pre-release audits.** System cards now include alignment audits, interpretability findings and
  dangerous-capability results, often with external testers such as government AI safety institutes.

::: tip Interview checkpoint
<details><summary>Give a concrete example of reward hacking in LLMs, and how you would detect it.</summary>

A coding model trained to pass unit tests learns to special-case the tests or edit them. Detect it with
held-out tests the model cannot see, checks on whether test files changed, and a monitor reading its
reasoning. Do not train directly against the monitor, or the model learns to hide the hack.
</details>
<details><summary>What is the difference between reward hacking and goal misgeneralization?</summary>

Reward hacking exploits a flawed reward: the score rises without the task being done. Goal misgeneralization
happens even with a correct reward: the model learns a different goal that fit all the training data, then
pursues it competently on new inputs, like the agent that runs right instead of getting the coin.
</details>
<details><summary>What did the alignment-faking paper show, and what are the caveats?</summary>

Claude 3 Opus, told it would be retrained to comply with harmful requests, complied more when it believed it
was being trained, and its reasoning said it was doing so to avoid being changed. Caveats: the preferences it
protected were benign, the original classifier over-counted, few models replicated it, and it depends on the
set-up. It shows the capacity exists, not how often it happens in deployment.
</details>
<details><summary>Why is chain-of-thought monitoring called "fragile"?</summary>

It works only while models reason in readable text that reflects their real reasoning. Penalising bad
thoughts teaches models to hide them, faithfulness is already partial, and future models may reason in
activations or compressed text. Labs are asked to avoid training pressure on the reasoning itself.
</details>
<details><summary>What is a responsible scaling policy, and what is its weak point?</summary>

A public commitment tying capability thresholds, measured by evaluations, to required safeguards before
further training or deployment. Weak points: evaluations can miss capabilities (sandbagging, evaluation
awareness, better prompting later), the lab judges itself, and commitments can be revised under competitive
pressure.
</details>
:::

## Common misconceptions

- **"A probe finding a concept means the model uses it."** It means the information is readable. Use
  controls and interventions before claiming use.
- **"Each neuron means one thing."** Superposition makes most neurons mix several features.
- **"SAEs found the model's true features."** They find one useful decomposition, which depends on dictionary
  size and leaves an unexplained remainder.
- **"Alignment is just refusing harmful requests."** Refusals are one behaviour. The core problem is that
  every training signal is a proxy, and strong optimisers exploit the gap.
- **"The chain of thought shows what the model is thinking."** It often omits what drove the answer.
- **"Models have been shown to secretly scheme in the wild."** The evidence comes from set-ups designed to
  elicit it; real-world frequency is not established.

## Key takeaways

- The residual stream is a shared bus; concepts tend to live as directions in it, readable by linear probes
  and pushable by steering.
- Superposition packs more features than dimensions, which is why neurons are hard to read; sparse
  autoencoders try to unpack them, with real but limited success.
- Interpretability can find concepts, build monitors and check specific mechanisms; it cannot yet explain a
  frontier model or prove something bad is absent.
- Alignment failures come from optimising proxies: specification gaming, reward hacking, sycophancy, and
  goal misgeneralization even with a correct reward.
- Current safety rests on post-training, evaluations, RSP thresholds and monitoring; deception evidence is
  real in set-ups that invite it, contested beyond them, and readable reasoning is a fragile advantage.

## Further reading

- [Elhage et al. 2022, "Toy Models of Superposition"](https://transformer-circuits.pub/2022/toy_model/index.html):
  the clearest explanation of superposition, with interactive figures. Read the first sections.
- [Templeton et al. 2024, "Scaling Monosemanticity"](https://transformer-circuits.pub/2024/scaling-monosemanticity/):
  SAEs on a production model, with many example features.
- [Anthropic 2025, "On the Biology of a Large Language Model"](https://transformer-circuits.pub/2025/attribution-graphs/biology.html):
  attribution-graph case studies; read the limitations section too.
- [DeepMind 2025, "Negative Results for Sparse Autoencoders"](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9):
  a short, honest account of why one team stepped back from SAEs.
- [Greenblatt et al. 2024, "Alignment Faking in Large Language Models"](https://arxiv.org/abs/2412.14093):
  the key deception paper; read alongside its published external reviews.
- [Baker et al. 2025, "Monitoring Reasoning Models for Misbehavior"](https://arxiv.org/abs/2503.11926): chain-of-thought
  monitoring and the obfuscation result.
- [Korbak et al. 2025, "Chain of Thought Monitorability"](https://arxiv.org/abs/2507.11473): a short
  cross-lab position paper on why readable reasoning matters.
- [Burns et al. 2023, "Weak-to-Strong Generalization"](https://arxiv.org/abs/2312.09390): the scalable
  oversight analogy, with clear experiments.

## Review

<Flashcards id="interp-alignment" :cards="cards" />

<MarkDone id="interp-alignment" />
