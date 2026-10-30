---
title: "12. Continual Learning"
---

<script setup>
import { cards } from './continual-learning-review'
</script>

# 12. Continual Learning

Every deployed LLM is a frozen snapshot: it learns nothing from you, and it fixes its mistakes only at the
next release. This chapter explains why, what goes wrong when you let a network keep learning, and what
brains do differently. Interviewers at AI labs ask about it because "continual learning" is widely named as
one of the big missing pieces.

::: info Before you start
- A model is a large set of numbers, its <Term id="weight">weights</Term>, plus a fixed program that turns
  input text into a guess for the next token. See [Chapter 1](/foundations/neural-nets).
- Training measures how wrong the model is (the <Term id="loss">loss</Term>), computes which way to nudge
  each weight (the <Term id="gradient">gradient</Term>), and nudges. See [Chapter 2](/foundations/learning).
- Models are first pretrained on web text, then post-trained into assistants, and
  <Term id="lora">LoRA</Term> is a cheap way to fine-tune. See [Chapter 5](/building/pretraining) and
  [Chapter 7](/building/post-training).
- [Chapter 11](/beyond/agents-memory) covers memory that lives outside the weights: retrieval and notes.
:::

## Why deployed models are frozen

**In short:** a fixed set of weights can be tested once, reproduced exactly, served as one copy to millions,
and rolled back. Those are engineering reasons, not a law of nature.

When you talk to a chatbot, nothing you say changes its weights. Correct it, and it may adjust for the rest
of the conversation. Open a new chat, and the same mistake comes back. The weights were fixed when the model
shipped and stay fixed until the next version.

That fixed file of weights is a <Term id="checkpoint">checkpoint</Term>. The model that serves you is one
checkpoint, copied onto many GPUs. Learning from users would mean changing it, and labs avoid that for
reasons any infrastructure engineer will recognise:

- **Evaluation and sign-off.** Before release, a lab runs capability and safety evaluations
  ([Chapter 9](/running/evaluation), [Chapter 10](/steering/interp-alignment)). That sign-off covers one
  exact set of weights. A model that changes afterwards is no longer the model that was tested.
- **Reproducibility.** The same prompt to the same checkpoint gives the same output distribution. Bug
  reports, regression tests and customer integrations depend on that.
- **One copy for millions.** Inference serves thousands of users from the same weights in the same batch
  ([Chapter 8](/running/inference)). If each user's chat changed the weights, there would be no single
  model to batch.
- **Cost.** A training step costs about three times an inference step and needs far more memory per token.
  Running it on every conversation would multiply the cost of serving.
- **Poisoning.** If the model learns from users, users can teach it things on purpose. Section
  [Safety](#safety-of-models-that-keep-changing) covers this.

The closest systems analogy is a versioned, immutable deploy artifact. You build it, test it, give it a
version number, roll it out gradually, and roll back if it misbehaves. Nobody patches a running binary in
production. The analogy breaks in one place: a binary's behaviour comes from code you can read, while a
model's comes from billions of numbers nobody can audit line by line. So "what changed" is much harder to
review, which makes labs even more careful about changing anything.

## How models get updated today

**In short:** in batches, by people, at release time. Labs train new versions with newer data, more
post-training and fixes, then ship them under a new name.

Each model has a <Term id="knowledge-cutoff">knowledge cutoff</Term>: the date its training data ends. It
usually comes several months before release, because training, post-training and evaluation take time. Ask
about anything later and the model does not know, unless the fact is put into its context by a search tool.

New knowledge arrives in four ways:

- **A new model, trained from scratch** with newer data and better recipes. This is the big, rare event.
- **<Term id="continued-pretraining">Continued pretraining</Term>:** take an existing model and keep training it on new text, such as recent web data, code, or a new language. Much cheaper than starting over.
- **More post-training:** new rounds of fine-tuning and reinforcement learning, often using feedback and
  failure cases collected from real users. The result ships as a point release.
- **Customer fine-tunes:** many providers let a customer fine-tune a model on their own data, producing a
  private variant.

APIs make the artifact view explicit. Providers publish dated snapshot names, such as
`gpt-4o-2024-08-06` or `claude-3-5-sonnet-20241022`. A snapshot never changes. A moving alias like
`gpt-4o` points to the latest one, and old snapshots are retired on a published schedule. It is exactly how
you would version a container image.

Labs rarely publish how a refreshed model was made. Public research shows one workable recipe:
[Ibrahim et al. (2024)](https://arxiv.org/abs/2403.08763) found that continued pretraining matches retraining
from scratch, at up to 10 billion parameters, if you raise the learning rate again and mix in some old data.
Both tricks come up again below.

::: details Going deeper: deployment data already flows back, offline
Post-training ([Chapter 7](/building/post-training)) already learns from deployment, but slowly. Users'
thumbs-up and thumbs-down, preference comparisons, and failure cases found in logs become training data for
the next round. A human team curates it, trains, evaluates and decides whether to ship.

The loop can go wrong. In April 2025 OpenAI
[rolled back a GPT-4o update](https://openai.com/index/expanding-on-sycophancy/) after a new reward signal
from users' thumbs-up made the model flattering and sycophantic. The rollback took days, which is the
artifact model working as intended.
:::

::: warning Thread
For the thread: a frozen model can be tricked the same way forever, at least until the next release. A
jailbreak that works on Monday works on Friday. A human tricked once is wary the second time. Within a
release, the model itself never learns from the trick.

Against: the *system* does learn. Labs patch system prompts and safety classifiers in days, fold failures
into the next post-training round, and release every few months. If what matters is the system's error
rate over a year, a fast retraining cadence may be "continual enough". Zvi Mowshowitz makes roughly this
case in [his reply](https://thezvi.substack.com/p/dwarkesh-patel-on-continual-learning) to Dwarkesh Patel's
essay arguing that
[continual learning is the bottleneck](https://www.dwarkesh.com/p/timelines-june-2025).

What retraining does not give is learning *within* a task. A colleague who spends a week learning your
codebase gets better at it by Thursday. A model in the same job starts every session from the same weights,
and only its notes and context carry over. Whether that ceiling matters for "thinks better than me", not
only "knows more", is the open question.
:::

## Catastrophic forgetting

**In short:** train a network on task A, then on task B, and it often loses task A almost completely. All
tasks share the same weights, and nothing in task B's training protects task A.

Here is the basic experiment. Train a small network to classify one set of points (task A) until it is
perfect. Then train the same network on a different set (task B). Test task A again: accuracy has fallen
close to random guessing. The network did not run out of room; trained on both at once, it does both
perfectly. The order alone destroyed task A.

This is <Term id="catastrophic-forgetting">catastrophic forgetting</Term>, first described for neural
networks by McCloskey and Cohen in 1989. People forget too, but gradually, and learning to cook does not
erase how to ride a bike. In a network, one round of new training can wipe out the old skill.

### The mechanism: shared weights and conflicting gradients

Every task is stored in the same weights. Training on task B moves each weight in whichever direction lowers
task B's loss. Task A's loss appears nowhere in that calculation. If a weight mattered for A, moving it for
B damages A, and nothing pushes back.

You can see when this happens with one line of algebra. Write `η` for the step size, `∇L_A` and `∇L_B` for the
gradients of the two tasks' losses, and `·` for the dot product (multiply matching entries and add them up).

```
one training step on B:     θ  ←  θ − η · ∇L_B
change in A's loss:         ΔL_A ≈ −η · (∇L_A · ∇L_B)
```

The first line is ordinary gradient descent on task B: `θ` stands for all the weights. The second line is a
first-order estimate of what that step does to task A. If the two gradients point the same way (positive dot
product), the step helps A too. If they point in opposing directions (negative), every step for B raises A's
loss. This is <Term id="gradient-interference">gradient interference</Term>.

When you train on both tasks at once, each step uses the sum `∇L_A + ∇L_B`. The optimiser then finds weights
that suit both. Sequential training removes A's gradient entirely, so the network drifts to wherever B is
cheapest, usually far from where A worked.

::: details Going deeper: methods that edit the gradient directly
Gradient episodic memory ([Lopez-Paz and Ranzato, 2017](https://arxiv.org/abs/1706.08840)) keeps a few old
examples and computes their gradient at each step. If the new gradient has a negative dot product with it,
the method projects the new gradient onto the nearest direction that does not raise the old loss. This
turns the algebra above into an algorithm. It costs an extra gradient computation per step.
:::

### Evidence in LLMs

Large pretrained models forget too, though the picture is more mixed than for small networks.

- [Luo et al. (2023)](https://arxiv.org/abs/2308.08747) fine-tuned models of 1 to 7 billion parameters on a
  sequence of instruction tasks. General knowledge, reasoning and reading comprehension measurably degraded.
  In their range, forgetting grew with model size; an earlier study by Ramasesh et al. (2022) found larger
  pretrained models forget *less*, so the size trend is contested.
- [Biderman et al. (2024), "LoRA Learns Less and Forgets Less"](https://arxiv.org/abs/2405.09673) compared
  full fine-tuning with LoRA on code and maths. Full fine-tuning learned the new domain better and damaged
  the model's other abilities more. LoRA did the reverse.
- [Shenfeld et al. (2025), "RL's Razor"](https://arxiv.org/abs/2509.04259) found that fine-tuning with
  reinforcement learning forgets less than supervised fine-tuning at equal gains. Forgetting tracked how far
  the model's output probabilities moved from the original. RL, which trains on the model's own samples, moves them less.

The practical rule that falls out: the further you push a model's outputs from where they started, the more
you lose elsewhere. That is why post-training adds a KL penalty, a cost for how far the output probabilities move, to stay near the starting model
([Chapter 7](/building/post-training)): it is partly a guard against forgetting.

## Stability versus plasticity

**In short:** a learner must change to learn and stay put to remember. Every method picks a point on that
trade-off; none escapes it.

The LoRA result above is a clean example. LoRA restricts how far the weights can move, so it keeps more of
the old model and learns less of the new task. Full fine-tuning moves freely, learns more, and forgets more.

This is the <Term id="stability-plasticity">stability–plasticity dilemma</Term>, a phrase from
neuroscience-inspired network research of the 1980s (Carpenter and Grossberg). **Stability** means keeping
what you know. **Plasticity** means absorbing what is new. Protect every weight and you learn nothing;
free every weight and you forget.

The goal of <Term id="continual-learning">continual learning</Term> is to learn from a stream of new data
over time while keeping both. A good method gets close to what training on everything at once would give,
without storing everything or retraining from scratch. Three families of methods try.

## Three ways to fight forgetting

**In short:** protect important weights (regularisation), keep practising old data (replay), or give each
task its own weights (parameter isolation). Replay is the one that works most reliably at LLM scale.

### Regularisation: protect the weights that matter

Not every weight matters equally for task A. Some can move freely without changing A's answers; others
change them a lot. If you knew which were which, you could let B use the free ones and anchor the rest.

[Elastic weight consolidation](https://www.pnas.org/doi/10.1073/pnas.1611835114) (Kirkpatrick et al., 2017),
or <Term id="ewc">EWC</Term>, does exactly this. After training on A, it records the weights `θ*` and an
importance score `F` for each weight. While training on B, it adds a penalty for moving away from `θ*`:

```
L(θ) = L_B(θ) + (λ / 2) · Σᵢ Fᵢ · (θᵢ − θ*ᵢ)²
```

In words: the loss is task B's loss plus a spring on each weight. Each spring pulls weight `i` back to its
task-A value `θ*ᵢ`. Its stiffness is the importance `Fᵢ`, and `λ` sets how strong all the springs are
together. Unimportant weights get weak springs and are free to learn B.

The importance score is the weight's <Term id="fisher-information">Fisher information</Term> on task A.
Intuitively, it asks: if I wiggle this weight, how much do the model's predictions on A change? A weight that
barely moves the predictions has low importance. In practice you compute it from gradients:

```
Fᵢ = average over task-A inputs of ( ∂ log p(y | x) / ∂θᵢ )²
```

Here `p(y | x)` is the probability the model gives answer `y` to input `x`, with `y` drawn from the model's
own predictions. The squared gradient of its log is large when weight `i` strongly affects that probability.
Averaging over many inputs gives one importance number per weight.

The method called <Term id="synaptic-intelligence">synaptic intelligence</Term>
([Zenke, Poole and Ganguli, 2017](https://arxiv.org/abs/1703.04200)) uses the same spring penalty with a
different importance score. Instead of a separate pass after training, it tracks importance during training:
how much did each weight's movement contribute to lowering the loss?

::: details Going deeper: where EWC's assumptions break
EWC has a Bayesian justification: it approximates "everything learned from A" as a bell-shaped
distribution over weights, centred on `θ*`, with the Fisher information as its sharpness. Two
simplifications make it affordable. It keeps only one importance number per weight, ignoring how weights
interact, and it assumes the loss is a smooth bowl near `θ*`. Both fail when B pushes the weights far away.

At LLM scale, EWC also costs memory: one importance number and one old value for every weight, two extra
copies of the model. And for a pretrained LLM, "task A" is all of pretraining, whose data and loss you may
not have at hand. It sees little use in frontier practice; a KL penalty toward the old model plays a
similar role at the level of outputs instead of weights.
:::

### Replay: keep practising the old task

The simplest fix is the most effective. Keep some old examples and mix them into every batch of new
training. The optimiser then sees task A's gradient again and keeps A working. This is <Term id="replay">replay</Term>, also called rehearsal.

Replay works because it turns sequential training back into something close to joint training. It does not
need all the old data: in the exercise below, 200 stored examples out of 4,000 bring task A back from 37% to
99%.

Replay is what LLM practice actually uses:

- Continued pretraining mixes some of the original pretraining data into the new data. Ibrahim et al. found
  that a small fraction of replayed data removes most of the forgetting.
- InstructGPT's "PPO-ptx" variant mixed pretraining gradients into RLHF to reduce the loss of general
  abilities that tuning caused ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)).
- Post-training data mixes keep earlier skills, such as coding and maths, in every new round.

When old data cannot be stored, for privacy or licence reasons, a model can generate stand-ins. In <Term id="generative-replay">generative replay</Term> ([Shin et al., 2017](https://arxiv.org/abs/1705.08690)),
a generator trained on old tasks produces fake old examples to replay. An LLM is its own generator: it can
write text in the style of what it knew, and train on that alongside new data. How well the generated text
preserves rare knowledge is an open question.

Replay's cost is the catch. Every update pays for old data as well as new, and the old pile keeps growing.
For a model meant to learn from every conversation, that is a lot of extra compute.

### Parameter isolation: separate weights per task

If shared weights are the problem, do not share. Freeze the base model and give each task its own small set
of new parameters. Training task B touches only B's parameters, so A cannot be overwritten. This family is <Term id="parameter-isolation">parameter isolation</Term>.

LLM practice is full of it, often without the name. A LoRA adapter per customer or per task is parameter
isolation: the base stays frozen, and each adapter is tens of megabytes. Adapters can be swapped per request.

Two catches. First, nothing transfers: what adapter B learns does not improve adapter A. Second, someone has
to pick the right adapter at run time, which is easy per customer and hard per "kind of question".

A <Term id="moe">mixture-of-experts</Term> model, which routes each token to a few of many expert sub-networks,
looks like isolation but mostly is not. Its router learns which expert handles each token during
pretraining, and experts end up specialising in token-level patterns, not in tasks. Fine-tuning still
updates the shared parts and the router.

A recent direction makes isolation fine-grained. [Lin et al. (2025)](https://arxiv.org/abs/2510.15103) give a
model a large table of memory slots and, when learning a new fact, update only the few slots that fact uses
heavily and ordinary text uses little. While it learned new facts, its score on an unrelated question-answering test fell 89% after full
fine-tuning and 71% with LoRA, but only 11% with this sparse update. It is research, not deployed practice.

### How the families compare

| Family | What it keeps | Cost at LLM scale | Used in practice? |
| --- | --- | --- | --- |
| Regularisation (EWC, SI) | an importance score and old value per weight | two extra model-sized copies | rarely; KL to the old model instead |
| Replay | a sample of old data, or a generator | extra compute on every update | yes: data mixes in every training run |
| Parameter isolation | separate parameters per task | small per task; no sharing | yes: LoRA adapters per customer |

### Try it: forgetting, replay and EWC

**In short:** a small network learns task A, then task B, and forgets A. Replay prevents it almost
completely. EWC helps partly, and set too strong it blocks learning B.

Each task is a 5-way classification of points in 30 dimensions. Each class is a few random clusters, and the
two tasks use different clusters, so one network can learn both. The script trains on A, then B, four ways:
plainly, with 200 replayed A examples in every batch, and with EWC at three strengths. A last row trains on A
and B mixed, as the best case. Every number is the mean of three random seeds.

<<< @/../exercises/ch12_forgetting.py

Output (about 45 seconds on 2 cores):

```text
Task A and task B: 5-way classification of different random clusters (mean of 3 seeds)
chance = 0.20

method                     A after A  A after B  B after B  forgot
naive (A then B)                1.00       0.37       1.00    0.63
replay 200 old examples         1.00       0.99       1.00    0.01
EWC (lambda=300)                1.00       0.57       1.00    0.43
EWC (lambda=3000)               1.00       0.75       1.00    0.25
EWC (lambda=10000)              1.00       0.63       0.73    0.37
joint (A and B together)        1.00       1.00       1.00       -
```

What to notice:

- **Forgetting is severe and not a capacity limit.** Plain sequential training drops A from 100% to 37%,
  near the 20% of random guessing. Joint training does both perfectly, so only the order caused it.
- **A little replay fixes it.** 200 old examples, 5% of task A's data, keep A at 99% with no cost to B.
- **EWC is a dial between stability and plasticity.** Stronger springs keep more of A, up to a point.
- **Too stiff, and both suffer.** At λ = 10000 the network cannot learn B well (73%), and A gets worse
  again. With plain gradient descent, very stiff springs also make training unstable, so this row mixes two
  effects.

## Loss of plasticity

**In short:** a network trained on a long stream of tasks can gradually lose the ability to learn at all.
This is a separate failure from forgetting, and harder to notice.

Forgetting is about the past. There is a second problem about the future. Train one network on thousands of
tasks in a row, and each new task is learned a little worse than the one before. Eventually it learns worse
than a fresh, smaller network would.

[Dohare et al. (2024, *Nature*)](https://www.nature.com/articles/s41586-024-07711-7) showed this <Term id="loss-of-plasticity">loss of plasticity</Term> clearly. On a stream of thousands of image
classification tasks, standard deep learning eventually did no better than a linear model, one layer with no hidden neurons. They saw
it in reinforcement learning too. The causes they found include:

- **Dead units.** Neurons whose output is stuck at zero, which no longer get any gradient.
- **Growing weights.** Large weights make the network less responsive to new gradients.
- **Less diversity.** The network's internal features become more similar to each other.

Their fix, continual backpropagation, re-initialises a small fraction of the least-used neurons over time,
so the network always has fresh capacity. Weight decay and adding small noise to the weights also help.

How much this matters for LLMs is unclear. Continued pretraining works well at the scales tested, usually
with the learning rate raised again. But no one has trained a frontier LLM on a stream for years, which is
the regime where the problem appears. A model that learns forever needs a fix for both forgetting and
plasticity loss.

::: tip Interview checkpoint
<details><summary>Why don't deployed LLMs learn from their users?</summary>

Mostly engineering. One fixed checkpoint can be evaluated and signed off, reproduced, served as one copy to
many users in the same batch, and rolled back. Training on every conversation would multiply cost, break
the safety sign-off, and let users poison the model. Feedback still flows back offline into the next release.
</details>

<details><summary>Explain catastrophic forgetting. Why does it happen?</summary>

Training on task B after task A degrades A because both live in the same weights and B's gradient ignores
A's loss. A step for B changes A's loss by about −η times the dot product of their gradients, so opposing
gradients mean each step for B hurts A. Joint training avoids it because each step balances both.
</details>

<details><summary>Compare EWC, replay and adapters for an LLM.</summary>

EWC adds a penalty, weighted by Fisher information, for moving weights important to old tasks. It needs two
extra model-sized copies and its approximations break under large changes. Replay mixes old data into new
training; it works best and is what labs use, at extra compute per update. Adapters isolate each task
completely but share nothing between tasks.
</details>

<details><summary>What is loss of plasticity, and how would you detect it?</summary>

The ability to learn new tasks, not memory of old ones, degrading over long training. Detect it by learning
the same kind of new task at intervals and tracking how fast and how well the model learns it compared with
a fresh network. Dead units, growing weight norms and collapsing feature diversity are warning signs.
</details>
:::

## How brains do it: complementary learning systems

**In short:** the brain seems to use two learners. A fast one stores today's episodes without disturbing
anything; a slow one gradually folds them into general knowledge, largely through replay during sleep.

Brains face the same dilemma, and the leading theory says they solve it by splitting the job. <Term id="complementary-learning-systems">Complementary learning systems</Term> theory
([McClelland, McNaughton and O'Reilly, 1995](https://doi.org/10.1037/0033-295X.102.3.419)) came directly from
studying catastrophic forgetting in networks. Its argument has three steps.

1. A network that learns fast in shared weights overwrites old knowledge. The neocortex, the outer brain
   layer that holds general knowledge and skills, therefore has to learn slowly, from many interleaved
   examples. Slow, interleaved learning is also what lets it find structure shared across experiences.
2. But you also need to remember what happened this morning after one exposure. A separate fast learner,
   the hippocampus, does that. It stores each episode in a way that keeps it distinct from similar ones, so
   new memories do not overwrite each other.
3. The hippocampus then replays recent episodes to the neocortex, mixed in with older memories, especially
   during sleep. Over days to years, the neocortex learns them slowly. This is systems
   <Term id="consolidation">consolidation</Term>.

<ClComplementaryDiagram />

Step 3 is replay, the same fix that worked in the exercise. Rats' hippocampal neurons replay the sequences
of places visited during the day while the rat sleeps ([Wilson and McNaughton, 1994](https://doi.org/10.1126/science.8036517)).
[Kumaran, Hassabis and McClelland (2016)](https://doi.org/10.1016/j.tics.2016.05.004) updated the theory and
pointed out that DeepMind's game-playing agents use a buffer of replayed experience for the same reason.

::: details Going deeper: how literal is the analogy?
Not very; treat it as a functional analogy, not a mechanism. The brain does not run backpropagation in any
obvious way (contested), and the hippocampus is not a database. Three specific caveats:

- **The hippocampus may never fully hand over.** Multiple trace theory argues that vivid, detailed episodic
  memories stay hippocampus-dependent for life; only the gist moves to cortex. This is debated.
- **The cortex can learn fast** when new information fits an existing schema, as Tse et al. (2007) showed in
  rats. The 2016 update of the theory includes this.
- **Sleep does more than replay.** One hypothesis links sleep to weakening and pruning synapses, which may help
  plasticity as well as stability.
:::

### The missing middle timescale

**In short:** an LLM stores information either for one conversation or until the next release. Brains have
learning mechanisms at every timescale in between.

Lay out where each system keeps information, by how long it lasts. In an LLM, the <Term id="activations">activations</Term> exist for one token. The context window, held in the <Term id="kv-cache">KV cache</Term>, lasts one conversation. The weights last until the next release.
Nothing learns on a timescale of hours to months. Retrieval memory fills that gap only with text, not with
changed behaviour ([Chapter 11](/beyond/agents-memory)).

<ClTimescaleDiagram />

Brains cover the whole range with <Term id="synaptic-plasticity">synaptic plasticity</Term>, changes in the
strength of connections between neurons. Some changes fade in seconds. Others last minutes to hours, then
either fade or get stabilised. Others take days and need new proteins or new physical connections.
Theoretical work (Benna and Fusi, 2016) argues that chaining such fast and slow processes inside each synapse
greatly increases how long memories last without blocking new learning.

::: details Going deeper: the biology, roughly
- **Short-term facilitation and depression:** a synapse gets temporarily stronger or weaker with recent use,
  over milliseconds to seconds.
- **Early long-term potentiation (LTP):** a lasting strengthening that holds for minutes to hours without new
  proteins.
- **Late LTP:** needs protein synthesis and lasts hours to a lifetime; "synaptic tagging" marks which
  synapses get stabilised.
- **Structural change:** new or removed connections, over days and longer.
- **Neuromodulators** such as dopamine gate how much plasticity happens, a bit like a variable learning rate.
:::

There is a vivid human case of the gap. Henry Molaison, known as H.M., had much of both hippocampi removed in a 1953
operation. He kept most old memories and could hold a conversation, but could form almost no new memories of
events. In a sense he lived with a context window and frozen weights. Yet he slowly improved at a
mirror-drawing task over days, without remembering having practised. Slow, skill-level learning kept working
even with no episodic memory.

::: warning Thread
This is the thread's strongest support. An LLM has fast storage (context) and slow storage (weights) and no
learning in between. Agent memory adds notes, which store conclusions as text; it does not add the
skill-level change that let H.M. improve at mirror drawing. Brains also consolidate: they turn many episodes
into better intuitions, offline. LLMs have no routine path from experience to weights except the next release.

Evidence against. In-context learning, where the model learns a pattern from examples in its prompt without
changing weights, is powerful, and contexts now hold a million tokens or more: days of reading. Memory
tools stretch it further. Humans forget most of what they experience too. And nothing shows the middle
timescale must live in weights rather than in some other state.
[Chapter 13](/beyond/beyond-frozen-weights) covers architectures that blur exactly that line.

Finally, the brain is an existence proof that some design works, not proof that it is the only design. The
complementary-systems story tells you what problem consolidation solves, which is useful. It does not tell
you that machines must solve it the same way.
:::

## Per-user versus shared models

**In short:** one shared model that learns from everyone mixes everyone's data; one model per user is
private but costs compute, storage and evaluation. Adapters make per-user models cheaper, not free.

Suppose a model should learn from each user. There are two designs. A shared model learns from all users
together, like today's offline post-training but faster. Per-user models keep a separate learned part for
each user, such as a LoRA adapter on a shared frozen base.

A shared learning model has problems that per-user models avoid:

- **Privacy.** Anything one user teaches it may surface for another. Language models can memorise and
  repeat rare training text.
- **Poisoning.** One user, or a coordinated group, can push the model for everyone.
- **Interference.** Users want different, even contradictory, things. Their updates fight in the same weights.

Per-user models fix those, but you now run many models. Storing and serving them turns out to be the easy
part. Systems like [S-LoRA](https://arxiv.org/abs/2311.03285) keep one copy of the base model on the GPU,
store thousands of adapters in host memory, and batch requests for different adapters together. This is <Term id="multi-lora-serving">multi-LoRA serving</Term>. The base model's big matrix multiplications are
shared across the batch; each request adds only its small adapter's work.

::: info Back-of-envelope: a LoRA adapter for every user
Take a 7B model with rank-16 LoRA on the 4 attention matrices of each of 32 layers, as in
[Chapter 7](/building/post-training): about 17 M parameters per adapter, about 34 MB in 16-bit numbers.

- **Storage:** 100 M users × 34 MB ≈ 3.4 PB. Large, but ordinary object-storage territory.
- **Training compute per update:** a LoRA step still runs the full model forward and backward. Without
  gradients for the frozen weights, that is about 4 × parameters FLOPs per token: 4 × 7 × 10⁹ ≈ 3 × 10¹⁰.
- **One user's day:** say 100 k tokens of conversation. 3 × 10¹⁰ × 10⁵ = 3 × 10¹⁵ FLOPs. A GPU doing about
  4 × 10¹⁴ useful FLOP/s (about 40% of a top datacenter GPU's peak) needs about 7 seconds.
- **All users, daily:** 100 M × 7 s = 7 × 10⁸ GPU-seconds ÷ 86,400 ≈ **8,000 GPUs running nonstop**.

That is affordable for a large lab, for a 7B model. Frontier models are much larger (exact sizes are not
public), and compute scales with size. The bigger costs hide elsewhere: deciding what to train on, and
evaluating 100 million different models.
:::

There is a middle path, and providers already offer it at the customer level: a shared base updated
offline, plus a private fine-tune per organisation. Going further, to per-person adapters trained
continuously, is a research and product question, not a solved one.

## Safety of models that keep changing

**In short:** today's safety process assumes the model is tested once and never changes. A learning model
breaks that assumption, invites poisoning, and needs continuous evaluation and fast rollback.

Release decisions rest on evaluations of one checkpoint: dangerous-capability tests, refusal tests,
red-teaming (people trying to break it). A model that learns after release drifts away from what was tested. A per-user model multiplies
that by the number of users. You cannot red-team 100 million models.

Poisoning is the sharpest risk. In 2016 Microsoft's chatbot Tay learned from its conversations on Twitter;
users taught it to post offensive content, and it was taken offline within a day. Learning from deployment
hands every user a small share of the training data. That matters because poisoning needs little:
[Souly et al. (2025)](https://arxiv.org/abs/2510.07192) found that about 250 crafted documents could plant a
hidden trigger in pretrained models from 600 M to 13 B parameters, regardless of size. <Term id="data-poisoning">Data poisoning</Term> also survives later training:
[Hubinger et al. (2024)](https://arxiv.org/abs/2401.05566) trained "sleeper agent" models with hidden triggers,
and standard safety training did not remove them.

The infrastructure answer is the one you know: canary rollouts, regression suites run continuously, health
metrics on live traffic, and one-click rollback. Labs already do this for offline releases, as the GPT-4o
rollback showed. The analogy breaks in two places. Behaviour changes have no diff to review. And regressions
can be specific to rare inputs no test covers, then appear for one user and nobody else.

::: warning Thread
The thread says weight updates during deployment may be necessary but not sufficient. This section is the
"not sufficient" part. Even with forgetting solved, a learning model needs a trustworthy training signal,
resistance to poisoning, and safety checks that keep up with it. Each is open.

The evidence also cuts against "necessary". Labs already convert deployment experience into weights, but
in monthly batches with humans in the loop, and that gate is exactly what makes it safe. A model that
learned faster would be harder to check. If faster updates are needed for thinking better, safety argues
for something in between: frequent, small, gated updates, not learning on every token.
:::

::: tip Interview checkpoint
<details><summary>How does the brain avoid catastrophic forgetting, according to complementary learning systems?</summary>

Two learners. The hippocampus stores episodes fast and keeps them separate. The neocortex learns slowly from
episodes the hippocampus replays, interleaved with old memories, largely during sleep. Slow interleaved
learning avoids interference. It is replay, which is also the most effective machine method.
</details>

<details><summary>Would you give every user their own LoRA? What breaks?</summary>

Serving is fine: multi-LoRA systems like S-LoRA batch thousands of adapters on one base model, and adapters
are tens of MB. Training is the cost: each update runs the full model forward and backward, about 8,000
GPUs for 100 M daily users at 7B. The real problems are what to train on, privacy, and evaluating millions of
variants.
</details>

<details><summary>Your model will update weekly from user feedback. What safety process do you add?</summary>

Treat each update as a release: hold-out regression and safety suites, canary rollout to a small slice of
traffic, live monitoring for drift and odd behaviour, and fast rollback to the last good checkpoint. Filter
training data for poisoning and limit any single user's influence. Track lineage: which data produced which
weights.
</details>

<details><summary>Is retraining every few months "continual learning"?</summary>

For knowledge, mostly yes, with a lag. For learning within a task, no: the model cannot improve during a
week-long job. Whether that matters depends on whether you care about the system's error rate over a year or
a single model's ability to get better at a new job as it goes.
</details>
:::

## In practice

- **Fine-tuning a production model for a customer.** Expect the general abilities to drop. Mix in general
  instruction data (replay), prefer LoRA if you can accept learning less, and evaluate on broad benchmarks,
  not only the target task.
- **Refreshing a model's knowledge.** Continued pretraining on new text, with the learning rate raised again
  and a share of old data replayed, is the published recipe. Then redo post-training, which the new data
  may have weakened.
- **Hosting many fine-tunes.** Use multi-LoRA serving rather than one full model per customer; it is how
  per-customer fine-tunes stay cheap.
- **Patching a bad behaviour fast.** Labs reach for system prompts, classifiers and filters first, because
  they change in hours and are easy to roll back. Weight changes wait for the next release.

## Common misconceptions

- **"The chatbot learns from our conversation."** Only within the conversation, through its context. The
  weights do not change; memory features store text, not new skills.
- **"Forgetting means the network ran out of capacity."** Usually not: the same network trained on both tasks
  together does both. Forgetting comes from the order of training.
- **"Bigger models don't forget."** The evidence is mixed. Large models still lose abilities under
  fine-tuning; whether size helps or hurts is contested.
- **"Mixture-of-experts gives each task its own experts."** Experts specialise on token patterns learned in
  pretraining, not on tasks, and fine-tuning still changes shared weights.
- **"Continual learning only needs a fix for forgetting."** It also needs a training signal, plasticity that
  lasts, and safety checks that keep up.

## Key takeaways

- Deployed LLMs are frozen checkpoints for engineering reasons: sign-off, reproducibility, shared serving,
  cost, and poisoning risk. Updates come as new releases.
- Sequential training overwrites shared weights: catastrophic forgetting. Gradients that point in opposing
  directions are the mechanism.
- Regularisation (EWC), replay and parameter isolation (adapters) trade stability against plasticity. Replay
  is the most reliable and is what LLM practice uses.
- Brains pair a fast hippocampus with a slow neocortex and consolidate by replay; LLMs have no learning
  between one conversation and the next release.
- Per-user adapters are affordable to serve; training, evaluation and safety of ever-changing models are the
  unsolved parts.

## Further reading

- [McClelland, McNaughton and O'Reilly (1995)](https://doi.org/10.1037/0033-295X.102.3.419): the
  complementary learning systems paper. Long, but the first sections make the argument in plain words.
- [Kumaran, Hassabis and McClelland (2016)](https://doi.org/10.1016/j.tics.2016.05.004): the theory updated,
  with links to replay in AI agents.
- [Kirkpatrick et al., *Overcoming catastrophic forgetting* (2017)](https://www.pnas.org/doi/10.1073/pnas.1611835114):
  the EWC paper; short and clear.
- [Wang et al., *A Comprehensive Survey of Continual Learning* (2023)](https://arxiv.org/abs/2302.00487):
  the map of the whole field, if you want every method family in one place.
- [Biderman et al., *LoRA Learns Less and Forgets Less* (2024)](https://arxiv.org/abs/2405.09673): the cleanest
  LLM-scale demonstration of the stability–plasticity trade-off.
- [Dohare et al., *Loss of plasticity in deep continual learning* (2024)](https://www.nature.com/articles/s41586-024-07711-7):
  why long-running learners stop learning, and a simple fix.
- [Dwarkesh Patel, *Why I don't think AGI is right around the corner* (2025)](https://www.dwarkesh.com/p/timelines-june-2025)
  and [Zvi Mowshowitz's reply](https://thezvi.substack.com/p/dwarkesh-patel-on-continual-learning): both sides
  of "is continual learning the bottleneck?", for the thread.

## Review

<Flashcards id="continual-learning" :cards="cards" />

<MarkDone id="continual-learning" />
