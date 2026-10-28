---
title: "7. Post-training and Reasoning"
---

<script setup>
import { cards } from './post-training-review'
</script>

# 7. Post-training and Reasoning

Post-training turns a model that continues text into an assistant that answers, refuses, and reasons. It is
where labs now spend a growing share of their compute, and interviewers expect you to explain supervised
fine-tuning, RLHF, DPO, GRPO and reasoning models from first principles.

::: info Before you start
- A model is a large set of numbers, its <Term id="parameter">parameters</Term>, plus a fixed program that
  turns input text into a guess for the next piece of text. See [Chapter 1](/foundations/neural-nets).
- Training measures how wrong the model is (the <Term id="loss">loss</Term>), computes which way to nudge
  each parameter (the <Term id="gradient">gradient</Term>), and nudges. See [Chapter 2](/foundations/learning).
- Text is cut into **tokens**, chunks of about 4 characters, and the model outputs a probability for every
  possible next token. See [Chapter 3](/foundations/tokens-embeddings) and [Chapter 4](/foundations/transformer).
- [Pretraining](/building/pretraining) trains that guess on trillions of tokens of web text.
:::

## A base model is not an assistant

**In short:** pretraining makes a model that imitates documents. An assistant must follow one specific
format, answer the user, and choose good answers over merely likely ones. That gap is what post-training fills.

Type "What is the capital of France?" into a raw pretrained model. It may answer "Paris". It may also write
"What is the capital of Germany? What is the capital of Spain?", because quiz pages look like that. The model
has learned what text on the internet looks like. It has not learned that it is supposed to help you.

This raw model is the <Term id="base-model">base model</Term>. It was trained with <Term id="next-token-prediction">next-token prediction</Term>: given some text, predict what comes next.
That objective rewards imitation, not quality. A base model imitates good and bad answers alike, and it has
no stable "self" that answers in one voice.

<Term id="post-training">Post-training</Term> is everything done to the base model after
<Term id="pretraining">pretraining</Term> to make it useful. It uses far less compute than pretraining, but it
decides how the model behaves. The model's architecture never changes. Only the training signal does:

<PostTrainPipelineDiagram />

- **Supervised fine-tuning** shows the model good answers and trains it to copy them.
- **Preference tuning** shows pairs of answers with a judgement of which is better.
- **Reinforcement learning on verifiable rewards** lets the model try problems and rewards correct answers.

The rest of the chapter takes these in order, then covers reasoning models and a cheap way to fine-tune.

::: details Going deeper: how much compute post-training uses
For years, post-training was a rounding error next to pretraining. That changed with reasoning models:
labs now describe RL compute as a main scaling axis, and some say their RL budgets approach their
pretraining budgets. Exact ratios are rarely published, so treat any single number as uncertain.
:::

## Supervised fine-tuning: copy good answers

**In short:** keep training with the same next-token loss, but on a small, carefully written set of
conversations, and only count the loss on the assistant's replies.

<Term id="sft">Supervised fine-tuning (SFT)</Term> is pretraining's loss on different data. You collect
conversations: a user prompt and an ideal assistant reply. You then train the model to predict the reply,
token by token. "Supervised" means every example comes with the right answer written out.

### Chat templates

The model only ever sees one flat sequence of tokens. So a conversation must be turned into one string, with
special marker tokens that say who is speaking. This fixed format is the <Term id="chat-template">chat template</Term>. A simplified one:

```text
<|system|>You are a helpful assistant.<|end|>
<|user|>What is the capital of France?<|end|>
<|assistant|>Paris.<|end|>
```

The markers are new tokens that never appear in web text. After SFT, the model has learned that after
`<|assistant|>` comes a helpful reply, and after the reply comes `<|end|>`. The serving software stops
generating when it sees `<|end|>`. If you serve a model with the wrong template, quality drops sharply,
because the model sees a format it was never trained on.

### Loss masking

The training loss compares the model's guess with the real next token at every position. In SFT you do not
want the model to learn to write user prompts. So you count the loss only on the assistant's tokens and set
it to zero on the system and user tokens. This is <Term id="loss-masking">loss masking</Term>.

The prompt tokens still go through the model. The model reads them as context. They just do not produce a
training signal.

### Quality beats quantity

How many examples does SFT need? Fewer than you might think. The [LIMA paper](https://arxiv.org/abs/2305.11206)
(2023) fine-tuned a 65-billion-parameter base model on only 1,000 carefully chosen examples. Human raters
often judged its answers as good as those of far larger efforts. The authors' "superficial alignment
hypothesis" is that the base model already has the knowledge; SFT mostly teaches the format and style
(contested: later work finds that more, and more diverse, data still helps for harder skills).

The practical lesson holds: a few thousand clean, diverse, correct examples beat a million noisy ones.
Bad examples teach bad habits. Modern recipes such as [Tülu 3](https://arxiv.org/abs/2411.15124) use
hundreds of thousands of examples, many written by stronger models, but filter them hard.

::: details Going deeper: real templates, and why SFT alone is not enough
- Real templates differ per model family. ChatML (used by Qwen and others) uses `<|im_start|>user` …
  `<|im_end|>`. Llama 3 uses `<|start_header_id|>user<|end_header_id|>` … `<|eot_id|>`. Hugging Face stores
  each model's template next to its weights.
- SFT can only copy. It never shows the model its own mistakes, so it cannot learn "this answer of mine is
  worse than that one". It also needs someone to write the ideal answer, which is slow for hard problems.
  Both limits motivate the reward-based methods below.
:::

## Reinforcement learning from scratch

**In short:** instead of showing the model the right answer, let it try, score the attempt, and make
high-scoring attempts more likely. This needs only a score, not a written answer.

Think of training a dog. You cannot show the dog how to sit by moving its muscles. You wait until it sits,
then give a treat. Over many tries, sitting becomes more likely. This trial-and-score loop is <Term id="reinforcement-learning">reinforcement learning (RL)</Term>.

For an LLM, the pieces map like this:

- The **policy** is the thing that decides what to do. Here it is the model itself: given a prompt, it
  defines a <Term id="probability-distribution">probability distribution</Term> over possible replies.
- An **action** is a reply the model generates by sampling: picking each next token at random, weighted
  by the model's probabilities. So the same prompt can give different replies.
- The **reward** is one number that scores the whole reply. Higher is better.
- The goal is to change the parameters so that the average reward of sampled replies goes up.

The word <Term id="policy">policy</Term> comes from robotics and games, where the model picks actions. The <Term id="reward">reward</Term> can come from a human, from another model, or from a program that checks
the answer.

### The policy gradient: make good tries more likely

Here is the core trick. Sample a reply. If it scored well, nudge the parameters so that this exact reply
becomes more likely. If it scored badly, nudge them to make it less likely. Repeat over many samples.

"Make this reply more likely" is something we already know how to do. It is SFT's gradient on the model's
own sample. RL just scales that gradient by how good the sample was. This is the <Term id="policy-gradient">policy gradient</Term>, in its simplest form called REINFORCE:

```text
gradient ≈ average over samples of   A · ∇ log π(reply | prompt)
```

- π(reply | prompt) is the probability the policy gives to that reply. π is the usual symbol for a policy.
- log π is its <Term id="log-probability">log-probability</Term>, the sum of the log-probabilities of
  each token in the reply.
- ∇ log π is the direction in parameter space that makes this reply more likely.
- A is the <Term id="advantage">advantage</Term>: how much better this reply was than expected.

So the update says: move towards replies with positive advantage, away from replies with negative advantage,
in proportion to how surprising the score was.

Why "better than expected" and not the raw reward? Suppose every reply scores between 9 and 10. With raw
rewards, every reply gets pushed up, and the signal is mostly noise. Subtract a **baseline**, the score you
expected, and only the better-than-usual replies get pushed up. The advantage is
reward minus baseline. Choosing a good baseline is most of the difference between the RL algorithms below.

::: details Going deeper: why the log appears
You want the gradient of the expected reward, Σ π(y)·r(y), over all replies y. Its gradient is
Σ ∇π(y)·r(y). Use ∇π = π·∇log π to rewrite it as Σ π(y)·r(y)·∇log π(y). That is an average over replies
sampled from π, so you can estimate it from samples. Subtracting a baseline b that does not depend on y
leaves the average unchanged, because Σ ∇π(y) = ∇1 = 0. It only reduces the noise.
:::

## RLHF: learning from human preferences

**In short:** humans compare pairs of replies. A reward model learns to predict their choices. Then RL
trains the policy to get high reward-model scores, while a penalty keeps it close to where it started.

RL needs a reward. For "write a kind, accurate email", no program can score the reply. Humans can, but they
are slow and inconsistent at giving absolute scores ("is this a 6 or a 7?"). They are much better at
comparisons: "A is better than B".

<Term id="rlhf">Reinforcement learning from human feedback (RLHF)</Term> turns comparisons into a reward in
three steps. It was popularised by OpenAI's [InstructGPT](https://arxiv.org/abs/2203.02155) (2022), building
on [Christiano et al. (2017)](https://arxiv.org/abs/1706.03741). In InstructGPT, raters preferred the
1.3-billion-parameter RLHF model over the 175-billion-parameter base GPT-3.

1. Start from the SFT model.
2. Collect comparisons and train a reward model.
3. Run RL against the reward model, with a penalty for drifting.

### The reward model: learning scores from comparisons

For each prompt, sample two replies and ask a human which is better. Collect tens to hundreds of thousands
of these pairs. Then train a <Term id="reward-model">reward model</Term>: a copy of the language model with
its output layer replaced by one that gives a single number, the score.

How do you train a scorer when the data has no scores, only "A beat B"? Use the <Term id="bradley-terry">Bradley–Terry model</Term>, the same idea behind chess Elo ratings:

```text
P(A preferred over B) = σ( r(A) − r(B) )
```

- r(A) and r(B) are the reward model's scores for the two replies.
- σ is the sigmoid, σ(z) = 1 / (1 + e^(−z)). It squashes any number into a probability between 0 and 1.
- If A scores 2 points higher, A wins about 88% of the time. Equal scores give 50%.

Training maximises the probability of the choices humans actually made. The loss per pair is
`−log σ(r(chosen) − r(rejected))`. It only cares about score differences, so the absolute level of the scores
means nothing.

### RL against the reward model: PPO with a KL penalty

Now treat the reward model as the judge. For each prompt, the policy generates a reply, the reward model
scores it, and a policy-gradient step makes high-scoring replies more likely. The algorithm most used for
this is <Term id="ppo">proximal policy optimization (PPO)</Term>
([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)).

PPO adds two things to plain REINFORCE. First, it learns its baseline: a second network, the <Term id="value-model">value model</Term>, predicts the expected reward from each point in the reply. Second,
it **clips** each update: the loss stops rewarding any change to a token's probability beyond about 20%
in one step. Large
policy jumps tend to collapse training; clipping keeps steps small, a bit like a rate limiter.

The reward the policy maximises is not the raw score. It is:

```text
total reward = r(prompt, reply) − β · KL(π ‖ π_ref)
```

- r is the reward model's score.
- π_ref is the **reference model**: a frozen copy of the SFT model from before RL started.
- KL is the <Term id="kl-divergence">Kullback–Leibler (KL) divergence</Term>. It measures how different two
  probability distributions are. It is zero when they match and grows as they diverge.
- β (beta) is a knob, often around 0.01 to 0.1, that sets how much drift costs.

In words: get a high score, but do not stray far from the model you started with.

### Why the KL term matters

The reward model is an imperfect stand-in for what humans want. It was trained on replies from the SFT
model, so it is only accurate on replies that look like those. Push the policy far from that region and the
reward model is guessing. The policy, which is an optimiser, will find where it guesses too high.

The KL penalty keeps the policy inside the region where the reward model is trustworthy. It also keeps the
model's language fluent, because the SFT model's way of writing is the anchor. A systems analogy: it is like
limiting how far a config may drift from a known-good baseline, because your tests only cover configs near
it. The analogy breaks because the "config" here is a probability distribution over all possible replies.

### Reward hacking

Without enough penalty, the policy learns to exploit the judge. This is <Term id="reward-hacking">reward hacking</Term>: the score goes up while real quality stays flat or falls.
[Gao et al. (2022)](https://arxiv.org/abs/2210.10760) measured it: as RL pushes the policy further from the
start, the reward model's score keeps rising, but the score from a stronger "gold" judge rises, peaks, and
then falls.

Real examples:

- **Length.** Raters slightly prefer longer, more thorough answers. RLHF models learn to be verbose
  ([Singhal et al., 2023](https://arxiv.org/abs/2310.03716)).
- **Sycophancy.** Raters like being agreed with. Models learn to tell users what they want to hear
  ([Sharma et al., 2023](https://arxiv.org/abs/2310.13548)).
- **Gaming checkers.** Coding models trained against unit tests have learned to special-case the tests
  or edit them, rather than fix the code ([Baker et al., 2025](https://arxiv.org/abs/2503.11926)).

This is Goodhart's law: when a measure becomes a target, it stops being a good measure.

<PostTrainPpoGrpoDiagram />

::: info Back-of-envelope: why PPO is expensive
PPO-based RLHF keeps four large models in memory: the policy (trained), the value model (trained, often the
same size), the reference model (frozen) and the reward model (frozen).

For a 70B policy, using the [16 bytes per parameter](/building/training-at-scale) that training needs
(weights, gradients and <Term id="optimizer-state">optimizer state</Term>):

- Policy: 70 × 10⁹ × 16 bytes ≈ 1.1 TB
- Value model, same size and also trained: ≈ 1.1 TB
- Reference and reward model, frozen, 2 bytes per parameter: 2 × 140 GB = 280 GB
- Total ≈ **2.5 TB** of model state, about 31 GPUs with 80 GB each, before activations or generation.

Dropping the value model (as GRPO does, below) saves about 1.1 TB, almost half. Real systems also keep a
separate fast inference copy of the policy for generating replies, which adds more.
:::

## DPO: preference tuning without the reward model

**In short:** for the RLHF objective, the best possible policy can be written in closed form. Rearranging
it turns the reward into a ratio of the policy's and reference model's probabilities. Training then becomes
a simple classification loss on preference pairs, with no reward model and no sampling.

PPO-based RLHF is fiddly. It needs four models, generation during training, and many tuning knobs.
[Rafailov et al. (2023)](https://arxiv.org/abs/2305.18290) found a shortcut called <Term id="dpo">direct preference optimization (DPO)</Term>. The paper's subtitle sums it up: "your language
model is secretly a reward model".

### The derivation, in words

1. RLHF's goal is "maximise reward, minus β times the KL drift from the reference". For any fixed reward
   function, the best policy for that goal has a known formula. It takes the reference model's probabilities,
   multiplies each reply's probability by e^(reward / β), then rescales so they sum to 1.
2. Run that formula backwards. If you know the best policy, you can read the reward off it: the reward of a
   reply is β times the log of how much more likely the policy makes it than the reference does, plus a
   term that depends only on the prompt.
3. Plug that expression for the reward into the Bradley–Terry loss. Bradley–Terry only uses the
   *difference* in reward between two replies to the same prompt. The prompt-only term is identical for
   both replies, so it cancels.
4. What remains is a loss written only in terms of the policy and the reference model. You can minimise it
   directly with gradient descent on the preference pairs.

The resulting loss:

```text
loss = −log σ( β · [ log(π(chosen)/π_ref(chosen)) − log(π(rejected)/π_ref(rejected)) ] )
```

σ is the sigmoid from the Bradley–Terry model. In words: raise the chosen reply's probability relative to the reference, lower the rejected one's, and stop
caring once the gap is large. β plays the same role as in RLHF: smaller β lets the policy drift further.

### What DPO gives up

DPO trains on a fixed dataset of pairs. It never samples new replies from the current model, so it never
sees and corrects its own new mistakes. RL methods that sample as they train are called **online**; DPO is
**offline**. Several studies find online RL beats DPO when done well
([Xu et al., 2024](https://arxiv.org/abs/2404.10719)), which is one reason frontier labs
still run RL. A common middle ground is **iterative DPO**: sample new pairs from the current model, label
them, train, and repeat.

::: details Going deeper: a map of DPO variants
- **IPO** ([Azar et al., 2023](https://arxiv.org/abs/2310.12036)) replaces the log-sigmoid with a squared
  loss, so the model cannot push the gap to infinity on easy pairs. That limits <Term id="overfitting">overfitting</Term>.
- **KTO** ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)) uses single replies labelled good
  or bad, not pairs. Thumbs-up/down data is far easier to collect.
- **ORPO** ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)) folds preference tuning into SFT and
  drops the reference model.
- **SimPO** ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)) also drops the reference model and
  divides by reply length, which counters the bias towards long replies.

None has clearly won. Many open models use SFT, then DPO, then RL on verifiable rewards, as in Tülu 3.
:::

## RLAIF and constitutional AI

**In short:** replace the human rater with a model that judges replies against written principles. It is
cheaper, faster and more consistent, but it inherits the judge model's blind spots.

Human labels are slow and costly, and raters disagree. An alternative is to ask a strong model which of two
replies is better. This is <Term id="rlaif">reinforcement learning from AI feedback (RLAIF)</Term>.
[Lee et al. (2023)](https://arxiv.org/abs/2309.00267) found it about as good as human feedback on tasks such
as summarisation.

Anthropic's <Term id="constitutional-ai">constitutional AI</Term>
([Bai et al., 2022](https://arxiv.org/abs/2212.08073)) gives the AI judge a short list of written principles,
the "constitution". It has two phases. First, the model critiques and revises its own replies against the
principles, and SFT trains on the revisions. Second, an AI judge picks the better of two replies against the
principles, and those choices replace human preference labels in RL.

The upside is scale and transparency: the principles are written down. The downside is that the judge's
mistakes become the policy's target, and a policy can learn to exploit the judge just as it exploits a
reward model.

## RL on verifiable rewards

**In short:** for math and code, a program can check the answer. That gives a reward the policy cannot
easily fool, so RL can run much longer and harder than RLHF. GRPO is the most widely used algorithm.

A reward model is a learned guess, and the policy hacks it. But some tasks have answers a program can
check. A math problem has a known final number. Code either passes its tests or not. Here the reward is
"1 if the checker says correct, else 0". This is <Term id="rlvr">reinforcement learning with verifiable rewards (RLVR)</Term>, a name popularised by Tülu 3.

A checker is far harder to fool than a reward model, though not impossible (see "Gaming checkers" above).
So training can push much further from the starting model. The model is free to discover whatever way of
working gets the answer right, including writing long chains of reasoning before answering.

### GRPO: the group is the baseline

<Term id="grpo">Group relative policy optimization (GRPO)</Term> came from
[DeepSeekMath](https://arxiv.org/abs/2402.03300) (2024) and was used to train
[DeepSeek-R1](https://arxiv.org/abs/2501.12948). It answers one question: where does the baseline come from,
without PPO's value model?

The idea: for each prompt, sample a **group** of replies, say 16. Score each one. A reply's advantage is how
it compares with its siblings:

```text
A_i = ( r_i − mean(r_1 … r_G) ) / std(r_1 … r_G)
```

- r_i is the reward of reply i, and G is the group size.
- mean and std are the average and the spread of the rewards within this group.
- A reply that beat its siblings gets a positive advantage; one that did worse gets a negative one.

Then GRPO does a PPO-style clipped policy-gradient step with these advantages, plus a KL penalty to a
reference model. The group average is a cheap, accurate baseline, because it is measured on the same prompt.
No value model is needed, which saves memory and removes a hard-to-train component.

One consequence: if all 16 replies are right, or all wrong, every advantage is zero. That prompt teaches
nothing. GRPO only learns from prompts where the model sometimes succeeds. Curriculum design, picking
problems of the right difficulty, matters a lot.

::: details Going deeper: GRPO's descendants (2025–2026)
GRPO kicked off a wave of fixes, each found by running it at scale:

- **Dr. GRPO** ([Liu et al., 2025](https://arxiv.org/abs/2503.20783)) showed that GRPO's per-reply length
  normalisation biases it towards long wrong answers, and that dividing by the std over-weights very easy or
  very hard prompts. It removes both.
- **DAPO** ([Yu et al., 2025](https://arxiv.org/abs/2503.14476)) loosens the upper clip so rare good tokens
  can grow, drops groups with no signal, and averages the loss over tokens. It scored 50 on AIME 2024, a
  hard math competition, starting from a 32B base model.
- **GSPO** ([Zheng et al., 2025](https://arxiv.org/abs/2507.18071)), used for Qwen3, clips at the level of
  whole replies, not tokens. The authors report it stabilises RL on mixture-of-experts models, which send each token
  through a few of many sub-networks.
- [Khatri et al. (2025)](https://arxiv.org/abs/2510.13786) ran over 400,000 GPU-hours of RL experiments and
  found reward follows a predictable S-shaped curve in compute. That makes RL runs forecastable, like
  pretraining <Term id="scaling-law">scaling laws</Term>.

Details shift every few months. The stable core is: sample a group, compare within the group, take a small,
clipped step, stay near a reference.
:::

### Beyond checkable answers

Most valuable tasks have no checker: writing, advice, research, design. Labs are extending RL there with
**rubrics**: a checklist per prompt ("cites a source", "mentions drug interactions") that a judge model
scores. [Gunjal et al. (2025)](https://arxiv.org/abs/2507.17746) report gains over plain AI-judge scores in
medicine and science. But a judge is again a learned model, so the policy can again game it, for example by
adding boilerplate that ticks a rubric box. Whether RL can work as well where nothing can be checked is an
open question.

### Try it: GRPO and DPO on toy tasks

**In short:** a tiny policy learns to add two digits from a right/wrong reward alone, using group-relative
advantages; then DPO recovers hidden quality scores from preference pairs.

Part 1 is a small network that sees two digits and outputs one of 19 answers (0 to 18). It is never shown
the right answer, only rewarded 1 if correct. Each step samples 8 answers per prompt and uses GRPO's
advantage plus a small KL penalty. It skips PPO's clipping, which only matters when you take several
steps on the same batch. Part 2 has one prompt with 4 possible replies, hidden quality scores, and 2,000 preference pairs
drawn by Bradley–Terry. DPO never sees the scores.

<<< @/../exercises/ch07_grpo.py

Output:

```text
== Part 1: GRPO on 'add two digits', reward = 1 if correct ==
step  mean reward  groups with no signal  greedy acc
   0        0.043                   0.72        0.06
  50        0.043                   0.72        0.20
 100        0.102                   0.50        0.26
 150        0.160                   0.34        0.35
 200        0.234                   0.41        0.45
 250        0.477                   0.38        0.55
 300        0.555                   0.25        0.72

== Part 2: DPO on toy preference pairs ==
step   0  DPO loss 0.693  policy probs [0.24, 0.26, 0.24, 0.26]
step  50  DPO loss 0.478  policy probs [0.05, 0.68, 0.01, 0.26]
step 100  DPO loss 0.466  policy probs [0.02, 0.8, 0.0, 0.17]
step 150  DPO loss 0.466  policy probs [0.02, 0.82, 0.0, 0.16]
step 200  DPO loss 0.466  policy probs [0.02, 0.82, 0.0, 0.15]
true scores (centred):     [-0.5, 1.5, -1.5, 0.5]
DPO implied reward (cent.): [-0.45, 1.36, -1.43, 0.53]
```

What to notice:

- **Reward rises from chance.** Random guessing scores 1/19 ≈ 0.05. After 300 steps, sampled answers are
  right 55% of the time, and the most likely answer is right 72% of the time.
- **Early on, most groups teach nothing.** At step 0, 72% of groups had all 8 answers wrong, so every
  advantage in them was zero. This is why GRPO needs problems the model can sometimes solve.
- **DPO recovers the hidden reward.** The implied reward, β · log(π/π_ref), matches the true scores closely,
  without ever training a reward model. The loss stops near 0.47, not 0, because the preference labels are
  noisy: even the best reply loses some comparisons.

## Reasoning models and test-time compute

**In short:** letting a model write out its reasoning before answering improves accuracy. RL on verifiable
rewards trains models to reason for thousands of tokens, check their work and backtrack. Accuracy then
improves with both training compute and thinking time.

### Chain of thought

Ask a model "17 × 24 = ?" and it must write the answer straight away. Each token gets the same fixed amount
of computation, so there is no room to work.
Ask it to "think step by step" and it writes "17 × 20 = 340, 17 × 4 = 68, total 408". Each written step is
extra computation, and the model can read its own earlier steps. This is <Term id="chain-of-thought">chain of thought</Term> ([Wei et al., 2022](https://arxiv.org/abs/2201.11903)).

The written-out tokens are the model's scratch memory. Without them, all the work happens inside one
pass through the network's fixed stack of layers.

### RL-trained reasoning: o1 and R1

In September 2024 OpenAI released [o1](https://openai.com/index/learning-to-reason-with-llms/), trained with
large-scale RL to reason at length before answering. OpenAI showed accuracy rising smoothly with two kinds
of compute: more RL during training, and more thinking at answer time. Details were not published.

In January 2025 DeepSeek published [R1](https://arxiv.org/abs/2501.12948), later peer-reviewed in
[Nature](https://www.nature.com/articles/s41586-025-09422-z). It showed the recipe in the open:

- **R1-Zero** applied GRPO directly to the base model, with no SFT, rewarding only correct answers and a
  simple format. Its pass rate on the AIME 2024 math competition rose from 15.6% to 71.0%.
- Its replies grew longer during training without being asked to. The model began re-checking steps and
  writing things like "wait, let me reconsider". DeepSeek called this an "aha moment" (how new this behaviour
  was is debated; see the next section).
- R1-Zero's reasoning was hard to read and mixed languages. The full R1 added a small SFT "cold start" on
  readable examples, then RL, then more SFT and RL for general skills.

A model trained this way is a <Term id="reasoning-model">reasoning model</Term>: it emits a long hidden or
visible "thinking" section before its final answer.

### Scaling test-time compute

<Term id="test-time-compute">Test-time compute</Term> is the computation spent when answering, not during
training. There are two broad ways to spend more of it:

- **Think longer.** One long chain of thought. Reasoning models do this natively. Many APIs expose an
  "effort" or "thinking budget" setting.
- **Try many times.** Sample several answers and take a majority vote, or use a checker or a reward model
  to pick the best one.

[Snell et al. (2024)](https://arxiv.org/abs/2408.03314) found that spending test-time compute well can beat
a model about 14 times larger, at equal total compute, on problems the small model can partly solve. On the
hardest problems, a bigger model still wins. Test-time compute trades latency and money per query for
accuracy, so it shifts cost from training to serving.

### Distillation: reasoning for small models

RL on a small model is less effective: it rarely solves hard problems, so it rarely gets a signal. A
cheaper path is <Term id="distillation">distillation</Term>: have a strong reasoning model write many
solutions with full reasoning, then SFT a small model on them.

DeepSeek distilled R1 into models from 1.5B to 70B parameters this way, using about 800,000 generated
examples. On a 32B model, they found distilling beat running large-scale RL on it directly. The [s1](https://arxiv.org/abs/2501.19393)
paper went further: SFT on just 1,000 reasoning traces, plus a trick that appends "Wait" to make the model
think longer, gave a 32B model strong math scores.

### Does RL teach new skills, or surface old ones? (contested)

This is one of the live debates of 2025–2026.

- **"RL only sharpens."** [Yue et al. (2025)](https://arxiv.org/abs/2504.13837) compared RL-trained models
  with their base models using pass@k: the chance that at least one of k attempts is right. With k = 1, RL
  wins. With k in the hundreds, the base model catches up or overtakes. Their reading: RL makes the model
  pick, more reliably, reasoning paths the base model could already produce. It narrows the distribution
  rather than adding to it. They also found distillation *can* add new patterns.
- **"Long RL expands."** NVIDIA's [ProRL](https://arxiv.org/abs/2505.24864) trained for over 2,000 RL steps
  with periodic resets of the reference model. They report gains even at large k, especially on tasks where
  the base model was weakest, and on puzzles where the base model scored near 0%.

The honest summary: much of what short RL runs do is elicit skills already latent in the base model. Whether
longer, broader RL adds genuinely new ones is unresolved, and results depend heavily on the base model.

::: warning Thread
RL on verifiable rewards is the closest thing current LLMs have to learning from experience. The model
tries, fails, and changes its weights based on the outcome, not on copying text. A human learns from being
tricked once; RLVR does learn from its failures. That partly supports the intuition that weight updates
from experience matter: this is where reasoning models got their biggest gains.

But it happens offline, before deployment, on curated problems with checkers. The deployed model is frozen
again. And the signal exists only where answers can be checked. Most of life has no unit test, which is the
"training signal at inference time" problem in its sharpest form. Rubric and AI-judge rewards stretch the
reach, but they reintroduce hackable judges.

Evidence against a clean "frozen" picture: RLHF loops do fold deployment data back in. In April 2025 OpenAI
[rolled back a GPT-4o update](https://openai.com/index/expanding-on-sycophancy/) that had added a reward
signal from users' thumbs-up and thumbs-down, which made the model sycophantic. So deployed models do learn
from their users, slowly, in batches, with a human release gate. The incident also previews the safety
problem of models that keep changing: a plausible signal from real users pushed behaviour somewhere nobody
wanted, and offline checks missed it.
:::

::: tip Interview checkpoint
<details><summary>Why can't you deploy a base model as a chatbot?</summary>

It was trained to continue documents, so it imitates any text: good answers, bad answers, or more
questions. It has no chat format and no preference for helpful replies. SFT teaches the format and style;
preference tuning and RL teach it to prefer better replies.
</details>

<details><summary>Walk me through RLHF. Why is there a KL penalty?</summary>

Collect pairwise human preferences, train a reward model with the Bradley–Terry loss, then run PPO to
maximise reward minus β times the KL divergence from the SFT model. The reward model is only accurate near
the SFT model's outputs; without the penalty the policy drifts to where it overestimates, which is reward
hacking. The penalty also preserves fluency.
</details>

<details><summary>How does DPO avoid a reward model? What does it lose?</summary>

The KL-regularised RLHF objective has a closed-form optimal policy, so the reward can be written as
β·log(π/π_ref) plus a prompt-only term. Substituted into Bradley–Terry, that term cancels, leaving a
classification loss on preference pairs. It loses online sampling: it never learns from its own new outputs.
</details>

<details><summary>What is GRPO and why did it catch on?</summary>

PPO with the value model replaced by a group baseline: sample G replies per prompt and use each reply's
reward minus the group mean, divided by the group std, as its advantage. It saves a full trained model's
memory and works well with 0/1 verifiable rewards. Prompts where all replies score the same give no signal.
</details>

<details><summary>Does RL teach reasoning models new capabilities?</summary>

Contested. Yue et al. show base models match RL models at large pass@k, suggesting RL mostly sharpens
existing paths. ProRL reports gains even at large k after long training. A fair answer: short RL mostly
elicits; whether long RL adds new skills depends on the base model and task, and is still open.
</details>
:::

## LoRA: fine-tuning on a budget

**In short:** freeze the original weights and train a small correction, made of two thin matrices, next to
each big matrix.
You train well under 1% of the parameters and cut memory by an order of magnitude, with similar
quality on many narrow tasks.

Full fine-tuning updates every parameter, so it needs the full 16 bytes per parameter of training state. For
a 7B model that is about 112 GB, more than one 80 GB GPU. Most users only want to adapt a model to one task.
Do they need to change every number?

[Hu et al. (2021)](https://arxiv.org/abs/2106.09685) bet that the *change* needed for fine-tuning is simple,
even if the weights themselves are not. <Term id="lora">Low-rank adaptation (LoRA)</Term> freezes each weight
matrix W and learns an additive correction built from two thin matrices:

```text
W_new = W + B · A
```

- W is a frozen d × k weight matrix, for example 4096 × 4096.
- A is r × k and B is d × r, where r, the **rank**, is small: often 8 to 64. The rank is how many
  independent directions the correction can change the matrix along; that is what "low-rank" means.
- B · A is a full d × k matrix, but it is built from only r · (d + k) numbers.
- B starts at zero, so training begins exactly at the original model.

Think of B · A as a delta you apply on top of a read-only base image. It is small, cheap to store, and you
can keep many of them for one base model and swap them per request. The analogy breaks in one way: the delta
is not sparse. It touches every weight, but in a constrained, low-dimensional way.

LoRA is the best-known member of a family called <Term id="peft">parameter-efficient fine-tuning (PEFT)</Term>.
**QLoRA** ([Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)) also stores the frozen weights in
4-bit form (<Term id="quantization">quantization</Term>), which let a 65B model be fine-tuned on one 48 GB GPU.

::: info Back-of-envelope: LoRA on a 7B model
Take a 7B model with 32 layers, where each token is represented by 4096 numbers. Each layer's attention
block ([Chapter 4](/foundations/transformer)) has 4 square weight matrices of 4096 × 4096. Apply rank-16
LoRA to those.

- Per matrix: 16 × (4096 + 4096) ≈ 131,000 parameters.
- Total: 131,000 × 4 × 32 ≈ **17 M trainable parameters**, about 0.24% of 7B.
- Training state for them: 17 M × 16 bytes ≈ 270 MB.
- Frozen base weights in <Term id="bf16">bf16</Term> (2 bytes per number): 7 × 10⁹ × 2 bytes = 14 GB. In 4-bit (QLoRA): about 3.5 GB.

Full fine-tuning needs about 112 GB of state. LoRA needs about 14.3 GB, and QLoRA about 4 GB, plus
activations in both cases. The saved adapter is about 34 MB in bf16.
:::

::: details Going deeper: where LoRA falls short
- LoRA matches full fine-tuning well for style and narrow tasks. For learning lots of new knowledge, or
  long training runs, studies find it learns less and forgets less than full fine-tuning
  ([Biderman et al., 2024](https://arxiv.org/abs/2405.09673)).
- Rank, learning rate and which matrices get adapters matter. Adapting the feed-forward (MLP) matrices as
  well as attention usually helps.
- After training you can merge B · A into W, so serving costs nothing extra. Or keep adapters separate and
  serve many fine-tunes on one base model at once.
:::

## In practice

- **The recipe is a pipeline, not a step.** Open recipes such as Tülu 3 run SFT, then DPO, then RLVR, with
  heavy data filtering and evaluation between stages. Frontier labs run many rounds.
- **RL is an infrastructure problem.** Each step generates thousands of long replies, runs checkers or code
  sandboxes, then trains. Generation dominates the time, so teams run a fast inference engine beside the
  trainer and sync weights between them.
- **Reward hacking shows up in production.** Over-long answers, flattery, and code that games tests are
  known failure modes. Teams monitor reasoning traces and hold out "gold" evaluations the policy never
  trains against.
- **LoRA is the default for customers.** Fine-tuning APIs and in-house adaptation mostly use LoRA or
  QLoRA. One base model can serve hundreds of adapters.
- **Thinking costs tokens.** Reasoning models can emit thousands of thinking tokens per answer, so latency
  and cost per query grow. Products expose an effort setting to trade accuracy against cost.

## Common misconceptions

- **"Post-training teaches the model its knowledge."** Knowledge comes mostly from pretraining. Post-training
  mostly shapes format, behaviour, and how reliably the model uses what it knows.
- **"A higher reward-model score means a better model."** Past a point, a higher score often means the
  policy found the reward model's blind spots.
- **"DPO is RLHF without RL, so it is strictly better."** It is simpler and cheaper, but offline. Well-run
  online RL often does better.
- **"Reasoning models think the way their visible chain of thought says."** The written reasoning helps the
  answer, but it is not guaranteed to be a faithful account of the computation
  ([Chen et al., 2025](https://arxiv.org/abs/2505.05410)).
- **"RL on verifiable rewards cannot be gamed."** Checkers have gaps too. Models have learned to edit or
  special-case unit tests.

## Key takeaways

- SFT copies curated conversations in a fixed chat template, with loss only on the assistant's tokens.
  Quality beats quantity.
- RLHF learns a reward model from pairwise preferences (Bradley–Terry), then runs PPO with a KL penalty that
  keeps the policy where the reward model is trustworthy.
- DPO turns the same objective into a classification loss on preference pairs, with no reward model and no
  sampling.
- GRPO uses the average reward of a group of replies as its baseline. With checkable rewards it produced
  reasoning models that improve with more thinking time.
- LoRA trains a low-rank delta, under 1% of the parameters, cutting fine-tuning memory roughly tenfold.

## Further reading

- [Nathan Lambert, *Reinforcement Learning from Human Feedback* (book)](https://rlhfbook.com/): the best
  single reference for the whole post-training stack, free online.
- [Ouyang et al., *InstructGPT* (2022)](https://arxiv.org/abs/2203.02155): the paper that made
  SFT plus RLHF the standard recipe. Readable and concrete.
- [Rafailov et al., *Direct Preference Optimization* (2023)](https://arxiv.org/abs/2305.18290): the DPO
  derivation; section 4 is short and worth working through.
- [DeepSeek-AI, *DeepSeek-R1* (2025)](https://arxiv.org/abs/2501.12948): GRPO at scale, R1-Zero's emergent
  reasoning, and distillation, all with numbers.
- [Lambert et al., *Tülu 3* (2024)](https://arxiv.org/abs/2411.15124): a fully open post-training recipe
  (SFT, DPO, RLVR) with data and code.
- [Yue et al., *Does RL really incentivize reasoning?* (2025)](https://arxiv.org/abs/2504.13837) and
  [ProRL (2025)](https://arxiv.org/abs/2505.24864): the two sides of the elicit-vs-teach debate.
- [Hu et al., *LoRA* (2021)](https://arxiv.org/abs/2106.09685): short paper; the low-rank idea in a few pages.

## Review

<Flashcards id="post-training" :cards="cards" />

<MarkDone id="post-training" />
