---
title: "The Frozen-Weights Thread"
---

<script setup>
import { cards } from './thread-review'
</script>

# The Frozen-Weights Thread

This page pulls together one claim that runs through the whole primer: frozen, checkpointed LLMs cannot reach
"thinks better than me", and learning during deployment may be necessary but is not sufficient. It states the
claim carefully, weighs the evidence from every chapter, and turns it into questions you can test and discuss.

::: info Before you start
This is a synthesis page, not a lesson. It assumes you have read the Thread boxes in the chapters, especially
[Chapter 11](/beyond/agents-memory), [Chapter 12](/beyond/continual-learning) and
[Chapter 13](/beyond/beyond-frozen-weights). Each point links back to where the evidence is explained. Much of
what follows is open research, and speculative claims are marked.
:::

## The claim, stated carefully

**In short:** the claim is about how a system improves with experience after release, not about what it knows or
how good it is on day one.

A loose version, "frozen models can't be AGI", is easy to dismiss and hard to test. Here is a sharper version
in three parts.

1. **Frozen is not enough.** A system whose weights are fixed at release cannot reach better judgement than a
   skilled human across open-ended, unfamiliar domains. This holds even with any amount of context, retrieval
   and tools.
2. **Weight change is necessary.** Reaching that level needs some lasting change to parameters during
   deployment. It must run on a timescale between one conversation and one release: hours to weeks.
3. **Weight change is not sufficient.** Updating weights also needs a trustworthy training signal, protection
   against forgetting, a per-user or shared design, and safety checks that keep up.

Three words need pinning down. A **frozen** model is a <Term id="checkpoint">checkpoint</Term> whose weights
do not change between releases; the system around it (notes, retrieval, prompts) may still change. **Thinks
better** means better judgement and skill on problems the system has not seen, not a larger store of facts.
**Experience** means the tasks, mistakes and feedback the deployed system meets.

The first part has a strong objection built in. Nothing in principle stops a frozen model from reasoning
better than you on day one. A chess engine is frozen during a game and still beats every human. So the
defensible claim is narrower: a frozen model cannot keep *improving from its own experience* the way a person
does over weeks in a new job. The claim is about the slope of a learning curve, not its starting height.

### What would count as evidence

Because the claim is about slopes, the evidence must be learning curves over time. Useful measurements look
like this:

- **Improvement within a deployment.** Give a system many episodes of a new task, spread over days. Does its
  success rate keep rising, or does it plateau once the context is full?
- **Transfer of a lesson.** After one correction, does the system avoid the same mistake in a paraphrased
  form, in a later session, with no hint in the prompt?
- **Equal-budget comparisons.** Compare frozen-plus-memory against weight-updating systems at the same
  compute and the same data. Otherwise one design wins only because it got more resources.

Today almost no published result measures this directly (speculative: the field has many static benchmarks
and few long-run learning-curve benchmarks, see [Chapter 9](/running/evaluation)). That gap is itself an
opportunity for a newcomer.

## The four intuitions, weighed

**In short:** each intuition has real support in the chapters, and each has a serious counterargument. The
honest summary is that the gap is real today, but nobody has shown it must be filled with weights.

### 1. "A frozen model can be tricked the same way forever"

**Best evidence for.** Within one release, the model never learns from a trick. A jailbreak that works on
Monday works on Friday, and every copy shares the hole
([Chapter 12](/beyond/continual-learning#how-models-get-updated-today),
[Chapter 10](/steering/interp-alignment#chain-of-thought-monitoring-and-faithfulness)). Glitch tokens break the
same model on the same input until someone retrains it
([Chapter 3](/foundations/tokens-embeddings#try-it-a-bigram-language-model)).

**Strongest counterargument.** Inside a conversation the model does learn: tell it it was tricked and it
avoids the trick for the rest of the session. This is <Term id="in-context-learning">in-context learning</Term>,
and it can run a real learning algorithm in the activations
([Chapter 11](/beyond/agents-memory#limits-of-in-context-learning)). Also, the *system* learns. Labs patch
system prompts and classifiers in days and retrain every few months.

**A subtler point against.** Weight updates do not deliver "learn once" for free either. Gradient descent
moves weights a tiny amount per step, and a fact usually has to appear many times before the model learns it
([Kandpal et al., 2023](https://arxiv.org/abs/2211.08411), in [Chapter 2](/foundations/learning#the-generalisation-puzzle)).
A big step on one surprise damages old knowledge. So today the only one-shot learning mechanism LLMs have
lives in context, not in weights.

### 2. "Memory stores conclusions, and retrieval needs you to know what to look for"

**Best evidence for.** Every agent memory system stores conclusions as text
([Chapter 11](/beyond/agents-memory#what-they-have-in-common)). Retrieval fails when the query does not hint
at the cause ([Chapter 11](/beyond/agents-memory#where-rag-fails)). A person tricked once does not look up
"have I been tricked like this before?"; the experience changed how they see the situation. The patient H.M.
improved at mirror drawing with no memory of practising: skill learning without any notes
([Chapter 12](/beyond/continual-learning#the-missing-middle-timescale)).

**Strongest counterargument.** Humans rely on notes, checklists and documentation too, and it works.
Million-token contexts can hold a whole history, which removes the need to know what to look for
([Chapter 11](/beyond/agents-memory#long-context-vs-retrieval)). Agents that write and test their own skills
store procedures, not only facts. And Doc-to-LoRA turns a document into weights in one pass, which blurs the
line between "notes" and "intuition" ([Chapter 13](/beyond/beyond-frozen-weights#hypernetworks)).

**Where it stands.** Long contexts are used unevenly: models miss facts in the middle and when question and
answer share no words ([Chapter 11](/beyond/agents-memory#long-context-vs-retrieval)). So "just paste
everything" weakens the counterargument in practice, though not in principle.

### 3. "Nothing in between per-token activations and months-frozen weights"

**Best evidence for.** Training sets the weights, then only activations change, and they vanish after one
request ([Chapter 1](/foundations/neural-nets#from-a-tiny-mlp-to-an-llm)). The <Term id="kv-cache">KV cache</Term>
holds everything learned in a conversation and is freed when it ends
([Chapter 8](/running/inference#the-kv-cache)). Pretraining runs once and stops
([Chapter 5](/building/pretraining#other-responses)). Brains, by contrast, change connections on every
timescale and turn many episodes into intuitions offline, through <Term id="consolidation">consolidation</Term>
([Chapter 12](/beyond/continual-learning#the-missing-middle-timescale)).

**Strongest counterargument.** The structural version of this claim is already false. Linear attention and
DeltaNet keep a <Term id="fast-weights">fast weight</Term> matrix that the network writes to every token, and
production hybrids ship it ([Chapter 13](/beyond/beyond-frozen-weights#state-space-models-and-the-recall-gap)).
The <Term id="titans">Titans</Term> architecture changes the weights of a memory network during use, driven by a gradient
([Chapter 13](/beyond/beyond-frozen-weights#titans-a-neural-long-term-memory-driven-by-surprise)). Nothing
shows the middle timescale must live in weights rather than some other state. And the brain is proof that one
design works, not that it is the only one.

**Where it stands.** These fast weights are wiped at the end of each sequence, and the rules that write them
are frozen. Titans learns to *remember* surprising input, not to *judge* better: trick it once and it stores
the trick as a surprising fact. Persistent, cross-session consolidation is only beginning, for example the
2026 preprint [*Language Models Need Sleep*](https://arxiv.org/abs/2606.03979), which is unreplicated
([Chapter 13](/beyond/beyond-frozen-weights#follow-ups-2025-2026)).

### 4. "Retraining is too slow to count"

This intuition is implicit in the thread: monthly releases are not learning. It has the strongest
counterarguments of the four.

**Best evidence for.** Retraining gives no learning *within* a task. A colleague learning your codebase is
better by Thursday; a model starts every session from the same weights
([Chapter 12](/beyond/continual-learning#how-models-get-updated-today)).
Reinforcement learning on verifiable rewards (<Term id="rlvr">RLVR</Term>) is genuine learning from failure, but it happens offline, on curated problems, and the
deployed model is frozen again
([Chapter 7](/building/post-training#does-rl-teach-new-skills-or-surface-old-ones-contested)).

**Strongest counterargument.** Deployed models already learn from users, in batches with a human release gate.
In April 2025 OpenAI rolled back a GPT-4o update that had used thumbs-up and thumbs-down signal, because it
made the model sycophantic ([Chapter 7](/building/post-training#does-rl-teach-new-skills-or-surface-old-ones-contested)).
If what matters is the system's error rate over a year, a fast retraining cadence may be "continual enough".
Zvi Mowshowitz argues roughly this in
[his reply](https://thezvi.substack.com/p/dwarkesh-patel-on-continual-learning) to Dwarkesh Patel's
[essay](https://www.dwarkesh.com/p/timelines-june-2025). Cost is not the barrier: fine-tuning a small set of
extra weights takes hours on a few GPUs ([Chapter 6](/building/training-at-scale#silent-data-corruption)).

## The timescale picture

**In short:** lay every mechanism out by how long it keeps information. LLM research now has something at most
timescales; what is missing is persistence across sessions and a trustworthy signal for what to keep.

<ClTimescaleDiagram />

| Timescale | Brain (roughly) | LLM today | Research proposals |
| --- | --- | --- | --- |
| One token to seconds | short-term synaptic changes | activations | none needed |
| One conversation | working memory, early plasticity | context, KV cache | linear attention, DeltaNet, TTT, Titans |
| Hours to weeks | consolidation, often during sleep | notes and retrieval (text only) | per-user LoRA, Doc-to-LoRA, SEAL, "sleep" distillation |
| Months to years | slow structural change | the next release | continued pretraining, post-training rounds |

The row that matters is "hours to weeks". Today it holds only text. Every proposal for it is either
research-scale or resets between sessions
([Chapter 13](/beyond/beyond-frozen-weights#putting-it-together-the-timescale-map)). The brain analogy is
useful for naming the gap. Mapping specific mechanisms one-to-one onto brain processes is speculative.

## The hard parts

**In short:** even if weights could change during deployment, four problems remain. Partial answers exist for
forgetting and per-user state; the training signal and safety are the most open.

| Hard part | What addresses it (Chapters 12–13) | What is open |
| --- | --- | --- |
| Training signal at inference time | Self-supervised surprise (Titans); checkable rewards (RLVR, SEAL); user feedback and judges | Domains where nobody can check the answer; signals staying reliable under constant optimisation |
| Catastrophic forgetting | Regularisation (EWC), replay, LoRA and parameter isolation, RL's smaller output shift, consolidation | Thousands of sequential updates at LLM scale; SEAL still forgets earlier self-edits |
| Per-user vs shared | Per-user LoRA served by multi-LoRA systems; hypernetworks that write adapters | Merging what many users teach into the shared model without privacy leaks or poisoning |
| Safety of a changing model | Canary rollouts, continuous regression suites, rollback; probes running on activations | Assurance for a model that differs per user and per day; no lab has published such a system |

**The signal.** This is the hardest part, and [Chapter 9](/running/evaluation#why-models-get-good-at-verifiable-tasks-first)
explains why. If you cannot build a reliable evaluation for a skill, you probably cannot build a reliable
online reward for it either. User feedback does carry signal, but the GPT-4o rollback shows what continuous
optimisation against it can do. That is <Term id="goodharts-law">Goodhart's law</Term>: a measure that becomes
a target stops being a good measure.

**Forgetting.** <Term id="catastrophic-forgetting">Catastrophic forgetting</Term> is the best-studied of the
four ([Chapter 12](/beyond/continual-learning#catastrophic-forgetting)). The practical rule from the evidence:
the further you push a model's outputs from where they started, the more it loses elsewhere. "LoRA learns
less and forgets less" is the same trade-off seen from the other side. Long-running learners can also lose
the ability to learn at all ([Chapter 12](/beyond/continual-learning#loss-of-plasticity)).

**Per-user vs shared.** Storing and serving millions of adapters is the easy part
([Chapter 12](/beyond/continual-learning#per-user-versus-shared-models)). Deciding what each adapter learns,
and evaluating millions of different models, is the hard part.

**Safety.** Today's assurance tests one frozen checkpoint
([Chapter 10](/steering/interp-alignment#chain-of-thought-monitoring-and-faithfulness)). A learning model
hands every user a share of its training data, and <Term id="data-poisoning">data poisoning</Term> needs
little: about 250 crafted documents planted a trigger across model sizes
([Chapter 12](/beyond/continual-learning#safety-of-models-that-keep-changing)). This argues for frequent,
small, gated updates rather than learning on every token. It also cuts against "necessary": the human gate
on today's slow learning is what makes it safe.

## What would change your mind

**In short:** decide in advance which results would move you. Each is concrete enough to check when it
appears.

**Results that would weaken the claim:**

- A frozen model with a memory scaffold shows a rising learning curve on a new, long task (a week inside an
  unfamiliar codebase), matching a weight-updating baseline at equal compute.
- Lessons stored as notes generalise to paraphrased tricks in later sessions as well as a weight update does.
- Frozen models keep extending the length of task they can complete, as
  [METR measures it](https://arxiv.org/abs/2503.14499), with no deployment-time learning added.
- Many-shot in-context learning keeps improving as you add examples, with no plateau below what fine-tuning
  on the same examples reaches.

**Results that would strengthen it:**

- On a long-run learning benchmark, frozen-plus-memory systems plateau while a test-time-training or
  consolidation method keeps improving at equal compute.
- Adding more context stops helping on a skill, but a small weight update on the same experience does help.
- A "sleep"-style consolidation method replicates, and models that use it beat equal-size frozen models on
  tasks that need accumulated skill, not only recall.
- An online learner succeeds in a domain without a checker, using only user feedback, without the drift seen
  in the GPT-4o incident. That would show the "not sufficient" problems are solvable.

## Research directions for a newcomer

**In short:** the field lacks good measurements more than it lacks ideas. Small, careful experiments that
measure learning curves are within reach on a laptop or one consumer GPU. All of these are suggestions
(speculative), not established results.

**1. A "tricked once" benchmark.** Build 50 tricks (misleading premises, fake facts, prompt injections), each
with five paraphrases. Teach a small open model (around 0.1–0.5 billion parameters) the correction in three
ways: in context, as a retrieved note, and as a tiny LoRA update. Measure the paraphrase success rate in a
fresh session, and the damage on a held-out set. This tests intuitions 1 and 2 directly.

**2. A toy consolidation loop.** Extend the fast-weight exercise from
[Chapter 13](/beyond/beyond-frozen-weights) with "sessions". The fast memory is wiped between sessions. After
each session, run a "sleep" phase that distils what the fast memory held into the slow weights, mixed with
replay. Track recall of old sessions against new-session performance. This is the stability–plasticity
trade-off made visible on a CPU.

**3. Forgetting over a long stream.** The [Chapter 12](/beyond/continual-learning) exercise uses two tasks.
Run 100 small tasks in sequence instead, and plot accuracy on task 1 over time for plain fine-tuning, replay,
EWC and LoRA. Then measure how much each still learns per task, to check for loss of plasticity.

**4. Goodhart in miniature.** Train a small reward model on limited preference data, then keep optimising a
small policy against it while tracking a held-out "true" reward. Find where the two diverge, and test whether
gating updates (accept only if a fixed eval set does not drop) delays it. This is the unverifiable-signal
problem at toy scale.

**5. Poisoning an online learner.** Take any small model that learns from a stream. Mix in a growing fraction
of adversarial updates and find how many it takes to plant a behaviour, with and without a gate. Small
numbers here are a useful result.

Write each one up with the learning curve, not a single score. A clean negative result is still worth
publishing as a blog post.

::: tip Interview checkpoint
<details><summary>How would you discuss "frozen LLMs can't reach AGI" without overclaiming?</summary>

State it as a hypothesis about learning curves, not a verdict. Say what is true today: deployed models do not
change their weights from experience, and their only one-shot learning lives in context. Name the strongest
counters yourself (in-context learning, long context, fast retraining, Titans-style memory, chess engines that
are frozen and superhuman). Then propose a measurement that would settle part of it. Avoid "impossible",
"AGI" without a definition, and brain analogies offered as proof.
</details>

<details><summary>Isn't frequent retraining already continual learning?</summary>

Partly. Labs fold deployment data into post-training and release every few months, so the system learns,
slowly, behind a human gate. What retraining does not give is improvement within a task or for one user. And
the gate is what keeps it safe; removing it brings back poisoning and drift, as the GPT-4o rollback showed.
</details>

<details><summary>Which hard part would you work on first, and why?</summary>

A good answer picks one and gives a reason. The training signal is the most fundamental: forgetting and
per-user serving have partial fixes, but without a trustworthy signal there is nothing worth learning. A
practical first step is better learning-curve evaluations, because the same problem that makes online rewards
hard makes evaluation hard.
</details>

<details><summary>Do test-time training or Titans refute the "nothing in between" claim?</summary>

They refute the structural version: these models change weights during use, driven by a gradient. They do not
yet give persistence across sessions, and their signal is surprise, not correctness. So they fill the
timescale with memory, not with better judgement.
</details>
:::

## Common misconceptions

- **"Frozen means the model cannot learn anything after release."** In-context learning is real learning
  within a conversation, and the system learns through retraining.
- **"Adding test-time training solves it."** Weight change is at most necessary. The signal, forgetting,
  per-user design and safety all remain.
- **"Brains prove you need weight updates."** Brains show one design that works, not that it is the only one.
- **"Weight updates give one-shot learning."** Gradient descent usually needs many repetitions, and big single
  steps cause forgetting.
- **"Compute is the barrier."** Small adapter updates are cheap; deciding what to learn from is the hard part.

## Key takeaways

- The defensible claim is about improving from experience after release, not about capability on day one.
- Each intuition has real support and a serious counter; the "hours to weeks" timescale holds only text today.
- Fast weights, TTT and Titans fill the middle timescale with memory that resets, driven by surprise, not truth.
- The training signal in unverifiable domains, and safety of a changing model, are the most open problems.
- Progress depends on measuring learning curves, which is a good place for a newcomer to contribute.

## Further reading

- [Dwarkesh Patel, *Why I don't think AGI is right around the corner* (2025)](https://www.dwarkesh.com/p/timelines-june-2025)
  and [Zvi Mowshowitz's reply](https://thezvi.substack.com/p/dwarkesh-patel-on-continual-learning): the claim
  and its strongest counter, argued well.
- [McClelland, McNaughton and O'Reilly (1995)](https://doi.org/10.1037/0033-295X.102.3.419): complementary
  learning systems, the origin of the "consolidation" picture.
- [Biderman et al., *LoRA Learns Less and Forgets Less* (2024)](https://arxiv.org/abs/2405.09673): the
  stability–plasticity trade-off at LLM scale.
- [Behrouz et al., *Titans* (2025)](https://arxiv.org/abs/2501.00663) and
  [Wang, Shi and Fox, *Test-time regression* (2025)](https://arxiv.org/abs/2501.12352): the best current
  attempts at a middle timescale, and one frame that covers them.
- [Zweiger et al., *SEAL* (2025)](https://arxiv.org/abs/2506.10943): a model that learns how to update itself,
  and where it needs a checker.
- [Kandpal et al. (2023)](https://arxiv.org/abs/2211.08411): why rare facts are learned poorly, which tempers
  "learn once" hopes for weight updates.
- [Souly et al. (2025)](https://arxiv.org/abs/2510.07192): how little poisoning takes, for the safety side.
- [Many-shot jailbreaking (Anthropic, 2024)](https://www.anthropic.com/research/many-shot-jailbreaking): in-context
  learning cutting the wrong way.

## Review

<Flashcards id="thread" :cards="cards" />

<MarkDone id="thread" />
