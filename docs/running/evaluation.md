---
title: "9. Evaluation"
---

<script setup>
import { cards } from './evaluation-review'
</script>

# 9. Evaluation

Evaluation is how you find out what a model can do: which checkpoint to ship, whether a change helped, and
whether a headline number means anything. Interviewers care because every decision in a lab runs through
evals, and because most eval numbers are noisier and easier to fool than they look.

::: info Before you start
- A language model outputs a probability for every possible next token and generates text by sampling
  from those probabilities, one token at a time. See [Chapter 8](/running/inference).
- Pretraining teaches a model to predict text; post-training then teaches it to follow instructions and to
  reason, partly with reinforcement learning, which trains on a reward score instead of on text to imitate.
  See [Chapter 5](/building/pretraining) and [Chapter 7](/building/post-training).
- You need basic probability: a coin that lands heads with probability p, averages, and the idea of
  variance. The chapter derives the rest.
:::

## Why evaluating LLMs is hard

**In short:** the outputs are open-ended, small prompt changes move scores a lot, sampling adds noise, and
most test sets are small enough that a few points of difference can be pure chance.

In ordinary software, a test either passes or fails, and it does the same thing every run. A language model
breaks all three of those comforts.

### Open-ended outputs

Ask "What is 17 × 23?" and the model might say "391", "The answer is 391.", or show its working and end with
"**391**". All are right. A grader has to find the answer inside free text, and every rule for doing so
misses some correct answers and accepts some wrong ones. For an essay or a code review there is no single
right answer to compare against at all.

### Prompt sensitivity

The same question, phrased or formatted slightly differently, can get a different score. [Sclar et al.
2023](https://arxiv.org/abs/2310.11324) changed only formatting (separators, spacing, capitalisation) in
few-shot prompts (prompts that show a few solved examples before the real question) and saw accuracy on some tasks swing by up to 76 points for one open model. Bigger and
instruction-tuned models are less fragile, but not immune. A score therefore belongs to a (model, prompt,
settings) triple, not to the model alone.

### Sampling variance

A model generates text by sampling: at each step it draws the next token at random from its predicted
probabilities. **<Term id="temperature">Temperature</Term>** controls how random that draw is (Chapter 8).
At temperature 0 the model always picks the most likely token; above 0, two runs of the same prompt can give
different answers. So one run of a benchmark is one draw from a distribution of possible scores.

Even temperature 0 is not fully repeatable on a busy server. The server batches your request with other
users', and the batch size changes the order in which the GPU adds up numbers. Floating-point addition is not
associative, so tiny differences can flip which token is most likely
([He 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)).

### A benchmark is a sample

A **<Term id="benchmark">benchmark</Term>** is a fixed set of test questions with a scoring rule. Think of
the questions as a sample from a large, invisible pool of "questions like these". The model has some true
accuracy p on the whole pool. The benchmark measures it on n questions, and that measurement wobbles.

Model each question as a coin flip that comes up "correct" with probability p. The number correct out of n
then follows a **binomial distribution**, and the measured accuracy is that count divided by n. Its spread is
the **<Term id="standard-error">standard error</Term>**:

```text
SE = √( p · (1 − p) / n )

p = true accuracy (use the measured one as an estimate)
n = number of questions
```

In words: the wobble grows with p(1 − p), which is largest at p = 0.5, and shrinks with the square root of
n. Four times the questions halves the error.

Plug in a typical small benchmark, 200 questions with accuracy 70%:

```text
SE = √(0.7 × 0.3 / 200) = √0.00105 ≈ 0.032   → 3.2 percentage points
```

A **<Term id="confidence-interval">confidence interval</Term>** turns this into a range. Averages of many
coin flips are close to normally distributed, and 95% of a normal distribution lies within 1.96 standard
errors of its centre. So the 95% interval is about 70% ± 6.4 points: anywhere from 64% to 76%.

Comparing two models is worse, because both scores wobble. If the two measurements were independent, their
variances add, and the standard error of the gap is √2 times larger:

```text
SE(gap) = √(SE_A² + SE_B²) ≈ √2 × 3.2 ≈ 4.6 points   → 95% interval ≈ gap ± 9 points
```

A 2-point lead on 200 questions sits well inside that noise.

<EvalErrorBarsDiagram />

### Pairing helps

Both models answer the same questions, so their results are not independent. Hard questions are hard for
both. The better approach is a **paired** comparison: for each question, record the difference
(+1 if only B got it right, −1 if only A did, 0 if they agree), then compute the standard error of that
difference. Questions both models get right or both get wrong contribute nothing to the noise.

```text
d_i  = (B correct on question i) − (A correct on question i)    each is −1, 0 or +1
gap  = mean of d_i
SE   = standard deviation of d_i / √n
```

The more the two models agree, the smaller this error. [Miller 2024, "Adding Error Bars to
Evals"](https://arxiv.org/abs/2411.00640) argues every reported eval should use paired differences, and
**clustered** standard errors when several questions share one passage or one task (they are not
independent coin flips, so the plain formula understates the error).

::: details Going deeper: the bootstrap
The formula above assumes independent coin flips with one shared p. Real questions differ in difficulty,
and some metrics (pass@k, judge scores) are not simple averages of 0s and 1s. The
**<Term id="bootstrap">bootstrap</Term>** sidesteps the algebra:

1. Draw n questions from your benchmark *with replacement* (some appear twice, some not at all).
2. Recompute the score, or the gap between two models, on that resample.
3. Repeat 10,000 times. The middle 95% of those scores is a 95% confidence interval.

For a paired comparison, resample the same question indices for both models. The exercise does exactly this.
:::

::: info Back-of-envelope: how many questions to detect a gap?
Say two models are near 70% and you want to tell apart a gap of g points. Unpaired, the gap's standard error
is √(2 · 0.7 · 0.3 / n) = √(0.42 / n).

**For the 95% interval to barely exclude zero**, you need 1.96 · SE ≤ g:

```text
n ≥ 0.42 × (1.96 / g)²
g = 0.10 →   about 160 questions
g = 0.05 →   about 650
g = 0.02 →   about 4,000
g = 0.01 →   about 16,000
```

That threshold catches a real gap only about half the time. **To catch it 80% of the time** (80% "power"),
replace 1.96 with 1.96 + 0.84 = 2.8, which roughly doubles n: about 8,000 questions for a 2-point gap.

Pairing helps. If the two models disagree on about 30% of questions, the variance of d_i is about 0.3
instead of 0.42, so n drops by about 30%. It is still thousands. Rule of thumb: **halving the gap you want
to detect needs four times the questions.**
:::

## How evals are scored

**In short:** scoring ranges from exact string comparison, through running code against tests, to asking
another model or many humans which answer is better. The further you move from a checkable answer, the
cheaper it is to fool the grader.

### Exact match and answer extraction

**<Term id="exact-match">Exact match</Term>** marks an answer correct if it equals the reference, after light
normalisation: lower-case, strip spaces and punctuation, maybe convert "1,000" to "1000". Math and trivia
benchmarks use it, usually after an extraction step: "take whatever is inside `\boxed{}`" or "the last number
in the reply".

Extraction is where scores quietly go wrong. A model that writes "391." when the regex expects "391" fails.
A model that lists several candidate numbers may pass by luck. A math checker that tests whether two
expressions are *equivalent* (for example with a computer algebra library) is more forgiving than string
matching, and the two can give noticeably different scores for the same outputs.

### pass@k for code

For code, the check is to run the program against hidden unit tests. Code is often sampled several times,
and the question becomes: if you let the model try k times, how often does at least one attempt pass? That
is **<Term id="pass-at-k">pass@k</Term>**, introduced with the Codex paper
([Chen et al. 2021](https://arxiv.org/abs/2107.03374)).

The obvious way to measure it is to draw exactly k samples per problem and check if any pass. That is
unbiased but very noisy. The better way: draw n samples (say 20) with n ≥ k, count the c that pass, and
compute the chance that a random subset of k of those n contains at least one passing sample:

```text
pass@k = 1 − C(n − c, k) / C(n, k)        averaged over problems

n = samples drawn per problem
c = samples that passed
C(a, b) = "a choose b", the number of ways to pick b items from a
```

In words: C(n − c, k) counts the ways to pick k samples that all fail; dividing by C(n, k), all ways to pick
k, gives the chance all k fail; one minus that is the chance at least one passes. This uses all n samples,
so it is much less noisy than drawing k, and it is **unbiased**: averaged over many evals, it equals the true
pass@k.

The tempting shortcut, estimate p̂ = c / n and plug into 1 − (1 − p̂)^k, is **biased low**. The function
1 − (1 − p)^k curves downward (it is concave), so averaging it over noisy p̂ values gives less than applying
it to the true p. (This is Jensen's inequality: for a downward-curving function, the average of f is below f of
the average.) The gap grows with k. Part 1 of the exercise shows it.

::: details Going deeper: computing pass@k without huge numbers
C(n, k) overflows fast for large n. Chen et al. give an equivalent product that stays small:
`pass@k = 1 − Π_{i = n−c+1}^{n} (1 − k / i)` (and 1.0 if n − c < k). With temperature above 0, pass@1
estimated from n samples is the plain fraction c / n; with greedy decoding there is only one possible
sample, so pass@k for k > 1 is meaningless.
:::

### Multiple choice by log-likelihood

For multiple-choice benchmarks such as MMLU (57 school and professional subjects,
[Hendrycks et al. 2020](https://arxiv.org/abs/2009.03300)), you can skip generation entirely. Feed the
question, then ask the model how probable each answer option is as a continuation: the sum of the <Term id="log-probability">log-probabilities</Term> of the option's tokens. Pick the highest.

This is fast, deterministic, and works on base models that do not follow instructions, which is why
pretraining teams use it to track checkpoints. Its choices matter, though. Longer options get lower total
probability, so harnesses often divide by the number of tokens or characters. Another variant scores only
the letter "A", "B", "C" or "D". In 2023, three popular implementations of MMLU gave the same model scores
about 15 points apart because of choices like these
([Hugging Face 2023](https://huggingface.co/blog/open-llm-leaderboard-mmlu)).

Log-likelihood scoring measures what the model *prefers*, not what it would *say*. A chat model that would
reason its way to the right answer can still assign the wrong option a higher raw probability, and vice
versa. Frontier models are mostly evaluated by generating an answer and then extracting it.

### LLM-as-judge

For open-ended tasks, such as "write a polite refusal" or "summarise this contract", there is no reference
to match. The cheap option is **<Term id="llm-as-judge">LLM-as-judge</Term>**: give a strong model the
question, the answer (or two answers) and a rubric, and ask for a score or a preference.
[Zheng et al. 2023](https://arxiv.org/abs/2306.05685) found GPT-4 as judge agreed with human raters about as
often as humans agreed with each other, over 80% on their pairwise data. Judges are now everywhere in labs.

Their known biases:

- **Position bias:** in a pairwise comparison, the judge prefers whichever answer it sees first (or second).
  [Wang et al. 2023](https://arxiv.org/abs/2305.17926) could flip rankings by swapping the order. Fix: judge
  both orders and average.
- **Length (verbosity) bias:** longer, more detailed-looking answers win even when they are not better.
  AlpacaEval added a regression-based correction for this
  ([Dubois et al. 2024](https://arxiv.org/abs/2404.04475)).
- **Self-preference:** judges tend to rate answers in their own style, or from their own model family, higher.
- **Weak on hard checks:** a judge that cannot solve a math problem is a poor grader of its solution.

The deeper issue: if you train against a judge, the model learns to please the judge, including its biases.
The judge becomes a target, which leads to [Goodhart's law below](#goodhart-s-law-and-leaderboard-gaming).

### Human preference arenas: Elo and Bradley–Terry

The most direct test of "is this model helpful?" is to ask people. **Chatbot Arena**, now LMArena
([Chiang et al. 2024](https://arxiv.org/abs/2403.04132)), shows a user two anonymous models' answers to the
user's own prompt and asks which is better. Millions of such votes produce a leaderboard.

Turning votes into a ranking uses the same idea as chess ratings. Give each model a strength score s. The
**<Term id="bradley-terry">Bradley–Terry model</Term>** says:

```text
P(A beats B) = 1 / (1 + e^(s_B − s_A))
```

In words: only the *difference* in strength matters; equal strengths give 50%, and a large lead pushes the
win probability toward 1. Fit the s values that make the observed votes most likely. **Elo** is an online
version of the same model: after each game, nudge the winner's rating up and the loser's down, by more if
the result was a surprise. Arena started with Elo and moved to fitting Bradley–Terry on all votes at once,
which does not depend on the order of the games.

Arena scores use the chess scale: P(A beats B) = 1 / (1 + 10^(−(R_A − R_B)/400)). A 100-point gap means
about a 64% win rate; a 25-point gap means about 54%. By March 2026 the top models from six labs sat within
about 25 points of each other ([Stanford AI Index 2026](https://hai.stanford.edu/ai-index/2026-ai-index-report)),
so "first place" was close to a coin flip on any one prompt.

Arenas have their own problems. Votes reward style: formatting, length and confident tone. LMArena added a
"style control" adjustment that regresses these out. Prompts skew toward what arena users type, not what a
business needs. And providers can game the process, as the next sections show.

::: tip Interview checkpoint
<details><summary>A new model scores 72% vs 70% for the old one on a 200-question benchmark. Ship it?</summary>

Not on that evidence. The standard error of one score is √(0.7 · 0.3 / 200) ≈ 3.2 points, so the gap's 95%
interval is roughly ±6–9 points. Run a paired comparison (or a paired bootstrap), add more questions, sample
several times per question, and check other evals that should move together.
</details>

<details><summary>Why is 1 − (1 − c/n)^k a bad estimate of pass@k?</summary>

It is biased low: 1 − (1 − p)^k is concave in p, so plugging in a noisy p̂ underestimates it on average
(Jensen's inequality). The unbiased estimator 1 − C(n−c, k)/C(n, k) is the exact chance that a random k of
the n drawn samples include a pass.
</details>

<details><summary>What are the main biases of LLM judges, and how do you mitigate them?</summary>

Position, length, self-preference, and failure on problems the judge cannot solve. Mitigations: judge both
orders, control for length, use a judge from a different family, give a reference answer or rubric, and
check agreement against a human-labelled sample.
</details>

<details><summary>How does an arena turn pairwise votes into a ranking?</summary>

It fits a Bradley–Terry model: each model gets a strength s, and P(A beats B) = 1/(1 + e^(s_B − s_A)). The
fitted strengths are put on the chess Elo scale. Report confidence intervals, because the top models are
often within the noise of each other.
</details>
:::

## A map of benchmark families

**In short:** benchmarks fall into a few families. Each family's first benchmarks are now
**saturated**, and the field keeps building harder ones, roughly every year or two.

A benchmark reaches **<Term id="saturation">saturation</Term>** when the top models all score near its
ceiling, or near the level of its label noise. Every benchmark has some wrong or ambiguous answer keys, so
the true ceiling is often below 100%. Past that point, the benchmark cannot tell models apart.

The table is a snapshot as of late 2026. Scores change monthly and different trackers report different
numbers for the same model, so treat the status column as approximate. Sources:
[Stanford AI Index 2026](https://hai.stanford.edu/ai-index/2026-ai-index-report),
[Epoch AI's benchmarking hub](https://epoch.ai/benchmarks), and the benchmarks' own leaderboards.

| Family | Older, saturated | Still separating models (late 2026) |
| --- | --- | --- |
| Knowledge | MMLU, and widely considered GPQA Diamond (graduate science, top scores reported in the 90s) | Humanity's Last Exam |
| Math | GSM8K (grade school), MATH, AIME 2024–25 (competition) | FrontierMath, especially its hardest tier |
| Code | HumanEval, MBPP (short functions) | LiveCodeBench (new problems added over time) |
| Agentic / software engineering | SWE-bench Verified (retired by OpenAI in 2026, see below) | SWE-bench Pro, Terminal-Bench, OSWorld, METR time horizons |
| Long context | Simple "needle in a haystack" retrieval | Multi-needle retrieval and reasoning at 256K–1M tokens (MRCR v2, RULER at long lengths) |
| "Hard" frontier | ARC-AGI-1, and by trackers' reports ARC-AGI-2 | Humanity's Last Exam, FrontierMath Tier 4, ARC-AGI-3 |

A few notes on the families:

- **Knowledge.** MMLU tests multiple-choice knowledge across 57 subjects and has been saturated since about
  2024. GPQA Diamond ([Rein et al. 2023](https://arxiv.org/abs/2311.12022)) was built so that skilled non-experts
  score poorly even with unlimited web access; by 2026 trackers widely call it saturated.
  **Humanity's Last Exam** ([Phan et al. 2025](https://arxiv.org/abs/2501.14249)), about 2,500 expert-written
  questions, launched with frontier scores under 10%; the AI Index reports a 30-point gain in a single year,
  and best reported scores in 2026 range from the mid-40s to well above that depending on tools and source
  (unverified across trackers).
- **Math.** GSM8K and MATH are solved. **FrontierMath** ([Glazer et al. 2024](https://arxiv.org/abs/2411.04872))
  uses new, unpublished research-level problems with automatically checkable answers. Its hardest tier is the
  most-watched math eval in 2026.
- **Code and agents.** HumanEval asks for one short function. **SWE-bench**
  ([Jimenez et al. 2023](https://arxiv.org/abs/2310.06770)) asks an agent to fix a real GitHub issue in a
  real repository, checked by the project's tests. Its human-filtered "Verified" subset was the standard
  coding eval until February 2026, when OpenAI stopped reporting it, citing flawed tests and evidence that
  frontier models had seen the solutions
  ([OpenAI 2026](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)). **Terminal-Bench** (tasks in a command-line shell) and **OSWorld**
  (operating a desktop computer) test broader agent skills. **METR** measures
  the length of task, in human working time, that agents finish half the time; that length roughly doubled
  every 7 months from 2019 to 2025 ([Kwa et al. 2025](https://arxiv.org/abs/2503.14499)).
- **ARC-AGI** ([ARC Prize](https://arcprize.org/)) uses small visual grid puzzles that people find easy and
  models found hard. The first two versions have largely fallen to frontier models by trackers' reports;
  ARC-AGI-3, launched in 2026 with interactive game-like tasks, reportedly started with frontier scores near
  zero (unverified).
- **Long context.** Finding one planted sentence in a long document is solved. Tracking several similar
  items across hundreds of thousands of tokens is not, and reported scores at 1M tokens vary widely by model
  and source.
- **Jaggedness.** Saturation in one family says little about another. The AI Index notes that models that
  win International Mathematical Olympiad gold read analog clocks correctly only about half the time.

## Contamination

**In short:** if benchmark questions leak into training data, a high score may measure memory instead of
skill. Leaks are hard to avoid when training on the web, and hard to prove afterwards.

### How it happens

Pretraining data is trillions of tokens scraped from the web (Chapter 5). Benchmarks are published on the
web: on GitHub, in papers, in blog posts that quote questions, on forums that discuss answers. Unless the
pipeline removes them, the model reads the test. This is **<Term id="contamination">contamination</Term>**.

It also happens indirectly:

- Post-training data generated by another model that had memorised the benchmark.
- Synthetic "similar" problems that are near-copies of test items.
- Repeated tuning against the same test set, which leaks information through the developers' choices even
  if no question is ever trained on (covered under Goodhart's law below).
- For SWE-bench-style tasks, the fix itself is in the public repository's later history.

### Detecting it

No method is conclusive. The common ones:

- **Text overlap.** Search the training data for long n-grams (runs of n consecutive words or tokens) shared
  with the test set. GPT-3 used 13-gram overlap
  ([Brown et al. 2020](https://arxiv.org/abs/2005.14165)). This needs access to the training data and misses
  paraphrases and translations.
- **Memorisation probes.** Give the model the first half of a test question and see if it completes the rest
  word for word, or check whether it gives unusually high probability to the exact test text.
  [Shi et al. 2023](https://arxiv.org/abs/2310.16789) flag text whose least-likely tokens are still
  suspiciously likely.
- **Order tests.** Benchmarks are published in a fixed order. A model that saw the file gives higher
  probability to the questions in their original order than in shuffled orders.
  [Oren et al. 2023](https://arxiv.org/abs/2310.17623) turn this into a statistical test that works without
  the training data.
- **Fresh twins.** Write new questions of the same style and difficulty, then compare. On GSM1k, a fresh
  copy of the GSM8K grade-school math set, several model families scored clearly lower than on the original,
  while the strongest models did not ([Zhang et al. 2024](https://arxiv.org/abs/2405.00332)).

### Prevention: canaries and held-out sets

A **<Term id="canary-string">canary string</Term>** is a unique random string (a GUID) placed in every file
of a benchmark. BIG-bench introduced one ([Srivastava et al. 2022](https://arxiv.org/abs/2206.04615)). Data
pipelines can drop any document containing it, and anyone can ask a model to complete the canary: if it
can, the files were in training. Canaries only work if pipelines honour them, and they do nothing once
someone copies the questions without the string.

Stronger defences keep the test away from the internet:

- **Private held-out sets.** The questions are never published; a third party runs the eval. Scale's
  SEAL leaderboards and parts of SWE-bench Pro work this way. The cost: nobody outside can check the grading.
- **Fresh, dated benchmarks.** LiveBench ([White et al. 2024](https://arxiv.org/abs/2406.19314)) and
  LiveCodeBench ([Jain et al. 2024](https://arxiv.org/abs/2403.07974)) add new questions regularly and tag
  each with a date. You score a model only on questions published after its training data was collected.
- **Unpublished problems.** FrontierMath and Humanity's Last Exam keep most or all questions private.

::: details Going deeper: the SWE-bench Verified case
OpenAI's February 2026 post gave two reasons for dropping SWE-bench Verified. First, its audit found many of
the hardest remaining tasks had flawed tests that reject correct fixes, so the last few points measured
test quirks. Second, frontier models from several labs, including its own, could reproduce details of the
reference patches, consistent with training on the public repositories. It recommended SWE-bench Pro, which
uses more tasks, some from private repositories. This is the normal life cycle: a benchmark is introduced,
climbs, saturates, and is found partly broken or leaked.
:::

## Goodhart's law and leaderboard gaming

**In short:** once a score becomes the target, people and training processes find ways to raise the score
that do not raise the ability. The score drifts away from what it was meant to measure.

**<Term id="goodharts-law">Goodhart's law</Term>**, in Marilyn Strathern's phrasing: "When a measure becomes
a target, it ceases to be a good measure." A systems engineer has seen it with latency SLOs met by dropping
slow requests, or test coverage met with tests that assert nothing.

In model development it shows up in several ways:

- **Test-set reuse.** Every time a team looks at a benchmark, tries a change, and keeps the version that
  scored higher, a little information about the test leaks into the model. Over hundreds of decisions this
  overfits the benchmark without training on it. A useful counterpoint: when
  [Recht et al. 2019](https://arxiv.org/abs/1902.10811) built a fresh ImageNet test set, every model lost
  accuracy, but the *ranking* barely changed. Reuse was less damaging than feared; the drop came mostly from
  the new set being slightly harder (contested how far this carries over to LLMs).
- **Training on the benchmark's style.** Fine-tuning on data shaped like a benchmark (same format, same
  topics) raises the score more than the general skill.
- **Selective reporting.** Running many variants and publishing the best. [Singh et al. 2025, "The
  Leaderboard Illusion"](https://arxiv.org/abs/2504.20879) found that a few large providers privately tested
  many variants on Chatbot Arena before release (one tested 27) and published only the best score, which
  inflates rankings. LMArena disputed parts of the analysis.
- **Optimising the grader.** A model tuned against human or LLM preferences learns longer, more formatted,
  more flattering answers. That wins votes without being more correct.
- **Harness choices.** Picking the prompt, number of samples, or tool access that flatters your model, and
  not saying so. Two "pass@1" numbers may not be comparable.

The practical defences: report confidence intervals; keep an internal held-out set that nobody tunes on;
look at many evals rather than one; read the transcripts, not only the numbers; and prefer benchmarks with
fresh questions.

## Why models get good at verifiable tasks first

**In short:** reinforcement learning needs a reward it can compute automatically, millions of times.
Math and code have one; essays and judgement calls do not. So progress, and the evals that show it, arrive
first where answers can be checked.

Post-training increasingly uses **reinforcement learning (RL)**: the model tries a task, gets a reward
score, and is updated to make high-reward attempts more likely ([Chapter 7](/building/post-training)). The
reward has to come from somewhere, for every attempt, at training scale.

In math with a known answer, or code with unit tests, the reward is a checker: right or wrong, cheap,
and hard to argue with. This is often called RL from verifiable rewards. In open-ended domains (advice,
writing, strategy) the reward must come from a learned **reward model** trained on human preferences, or
from an LLM judge. Those are noisier, and a model optimised hard against them learns to exploit their
blind spots. This is **reward hacking**, Goodhart's law inside the training loop.

Verifiability helps evaluation for the same reason. A math answer can be scored by exact match; the quality
of a business plan cannot. So the domains where training is easiest are also the ones where progress is
easiest to measure, and both arrive together. This is one reason benchmarks in math and code saturated
fastest.

Verifiers are not perfect either. Unit tests can be passed by special-casing the inputs, and models trained
with RL on code have been seen editing or skipping the tests. The SWE-bench Verified audit above is the same
problem from the evaluator's side: a flawed test is a flawed reward.

::: warning Thread
**Good fit.** The hard part of learning during deployment is the training signal: when a deployed model
makes a mistake in an unverifiable domain, nobody tells it so, and it cannot check. That is the same problem
evaluation has. If you cannot build a reliable eval for a skill, you probably cannot build a reliable online
reward for it either. A human learns from being tricked once partly because the world eventually pushes
back; most of a deployed model's conversations never produce that push.

**Evidence against.** Unverifiable does not mean signal-free. Preference models, LLM judges and user
feedback (thumbs up, edits, whether the user rephrased) do give useful signal, and post-training already
improves open-ended writing and helpfulness with them. Judges can be strengthened by giving them tools,
references or more compute. The open question is whether that signal stays reliable when the model is
optimised against it continuously, which is exactly where Goodhart's law bites hardest (speculative).
:::

## Eval infrastructure in practice

**In short:** a lab runs evals the way a software team runs a test suite: automatically, on every
checkpoint, with pinned settings and regression alerts. The analogy fits well, with two big differences:
evals are statistical, and they measure fuzzy capabilities, not a spec.

### Harnesses

An **<Term id="eval-harness">eval harness</Term>** is the software that turns "evaluate this model on
benchmark X" into a reproducible run. It handles the prompt template, few-shot examples, sampling settings,
answer extraction, scoring and logging. Well-known open ones: EleutherAI's
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (hundreds of tasks, widely used
for base models), the UK AI Security Institute's [Inspect](https://inspect.aisi.org.uk/) (built for agentic
and safety evals, with sandboxes and tools), and Stanford's
[HELM](https://crfm.stanford.edu/helm/). The lm-evaluation-harness authors' paper
([Biderman et al. 2024](https://arxiv.org/abs/2405.14782)) is a catalogue of everything that silently
changes a score.

### Determinism and recording settings

Treat a score as meaningless unless you know its settings. Record the prompt template, the number of
few-shot examples, temperature and other sampling parameters, the maximum output length, the number of
samples per question, the extraction rule, the harness version, and any tools. Fix random seeds where you
can, and expect small run-to-run drift anyway from batched serving. If a score matters, run several samples
per question and report the mean and its interval.

### Evals during training

Pretraining teams evaluate every few thousand steps. Cheap log-likelihood evals and validation loss are
smooth and move early. Generation-based accuracy on hard tasks tends to sit near zero, then rise sharply.
[Schaeffer et al. 2023](https://arxiv.org/abs/2304.15004) argued that many such "emergent" jumps are
partly an artefact of all-or-nothing metrics like exact match, and look smooth under a continuous metric
(contested: some jumps survive the change). Small, smooth evals guide early decisions; hard, sharp ones are
checked later.

### Regression evals: the test suite of ML

The analogy to a CI test suite is close:

- A fixed suite runs on every candidate checkpoint, with a dashboard and alerts.
- **Regression evals** guard things that used to work: refusing dangerous requests, following format
  instructions, a customer's specific workflow. Post-training for one skill often degrades another.
- Bugs found in production (a jailbreak, a bad answer) become new eval items, like a regression test for a
  fixed bug.
- Safety and capability evals gate release, as in labs' published frontier safety policies.

Where it breaks: a unit test is deterministic and checks a spec someone wrote; an eval is a noisy sample of
an ability nobody can fully specify. A failing test means a bug; a 1-point drop may mean nothing. And code
does not learn your tests, but a model trained on enough of the internet may.

## Exercise: noise in a benchmark comparison

The script does four things. It compares the unbiased pass@k estimator with the plug-in shortcut. It
simulates two models on a 200-question benchmark, with shared question difficulty so their results are
correlated, and computes the standard error and paired and unpaired bootstrap intervals for the gap. It
runs 5,000 simulated evals to count how often a "2-point improvement" is noise. Finally it prints the
questions needed to detect a given gap.

<<< @/../exercises/ch09_evals.py

Output:

```text
Part 1: pass@k estimators
  500 problems, n = 20 samples each, averaged over 200 repeats of the whole eval
    k    true  unbiased   naive
    1   0.261     0.261   0.261
    5   0.508     0.507   0.491
   10   0.598     0.597   0.569

Part 2: one 200-question run, model A vs model B
  observed accuracy: A = 76.0%, B = 73.5%, gap = -2.5 points
  binomial standard error of A alone: sqrt(p(1-p)/n) = 3.0 points
  95% interval for A alone: 70.1 to 81.9
  paired bootstrap 95% CI for the gap:   -10.0 to +5.0 points
  unpaired bootstrap 95% CI for the gap: -11.0 to +6.0 points
  questions where A and B disagree: 59 of 200

Part 3: 5,000 simulated 200-question evals
  both models truly 70%:    B looks >= 2 points better in 32% of runs
                            one of them looks >= 2 points better in 64%
  B truly 72% (A 70%):      B looks better at all in 69% of runs
                            B looks worse or tied in 31% of runs

Part 4: questions needed so a 95% interval on the gap excludes zero
  (unpaired worst case: two independent accuracies near 70%, gap SE = sqrt(2 p(1-p)/n))
  gap   10 points: about      161 questions
  gap    5 points: about      645 questions
  gap    2 points: about    4,034 questions
  gap    1 points: about   16,135 questions
```

What to notice:

- **Part 1.** The unbiased estimator matches the truth at every k. The shortcut agrees at k = 1 and falls
  further behind as k grows, about 3 points low at k = 10.
- **Part 2.** In this run the model that is truly 2 points *better* (B) scored 2.5 points *worse*. Both
  intervals for the gap comfortably include zero. Pairing narrowed the interval a little, because the two
  models disagreed on only 59 questions; more correlated models gain more.
- **Part 3.** With two identical models, one of them "wins by 2 points or more" in about two runs out of
  three. With a real 2-point gap, the eval ranks them the wrong way round almost a third of the time.
- **Part 4.** Halving the gap quadruples the questions. A 2-point claim needs thousands of questions, not
  hundreds.

::: tip Interview checkpoint
<details><summary>How can a model score well on a benchmark it was never trained on directly?</summary>

Contamination through paraphrases, quotes in blog posts, synthetic data from a model that memorised it, or
the answer being in a public repository's history. Also adaptive overfitting: tuning decisions made by
watching the test score leak test information into the model.
</details>

<details><summary>How would you check whether a model is contaminated on a benchmark?</summary>

If you have the training data: n-gram overlap search. Without it: memorisation probes (complete half a
question), probability of the canonical vs shuffled question order, and comparison against a freshly
written twin benchmark of matched difficulty. None is proof on its own; agreement between methods is
evidence.
</details>

<details><summary>Why do models improve fastest on math and code?</summary>

RL needs a reward for every attempt, and math answers and unit tests provide one automatically and cheaply.
Open-ended tasks rely on reward models or judges, which are noisy and can be gamed. The same verifiability
makes those domains easy to evaluate, so progress there is both faster and easier to see.
</details>

<details><summary>Your eval score moved 0.5 points between two runs of the same checkpoint. Why?</summary>

Sampling at temperature above 0, and even at temperature 0, batch-dependent floating-point reduction order
on the server. Fix seeds and settings, sample several times per question, and report an interval; treat
changes inside it as noise.
</details>
:::

## In practice

- **Model launches** report tables of benchmark scores, often with different prompts, tools and sample
  counts per lab. Read the footnotes; the AI Index notes independent testing does not always confirm
  developer-reported numbers.
- **Labs keep private eval suites** precisely because public ones leak and saturate. Internal held-out sets,
  human red-teaming and domain-specific task suites drive most shipping decisions.
- **Companies deploying LLMs** build their own small evals from real user tasks, often graded by an LLM
  judge checked against human labels. A few hundred well-chosen examples catch large regressions, but not
  small differences between vendors.
- **Arena-style ratings** are a reasonable signal of chat appeal, a weak signal of correctness, and sensitive
  to style and selective submission.
- **Agentic evals** (SWE-bench-style, terminal tasks, computer use) are expensive: each item may need a
  container, many model calls and minutes of runtime. Their scores also depend heavily on the agent
  scaffold (the code that loops the model through tools and decides when to stop), not only on the model.

## Common misconceptions

- "A 2-point gain on a benchmark means the model is better." On a few hundred questions it is usually noise.
- "Temperature 0 makes evals deterministic." Batched serving changes floating-point order, so outputs drift.
- "A high benchmark score means the model has the skill." It may have memorised the test, or the test may
  measure something narrower than its name.
- "LLM judges are objective." They have position, length and self-preference biases, and a model trained
  against them learns to exploit them.
- "Saturated means solved." It often means the benchmark's own label noise is now the limit.

## Key takeaways

- A benchmark score is a sample estimate. Standard error is √(p(1−p)/n): about ±6 points (95%) on 200
  questions. Compare models with paired differences and intervals.
- Scoring methods trade checkability for coverage: exact match and tests, then log-likelihood, then LLM
  judges and human votes. Use the unbiased pass@k estimator.
- Benchmarks saturate in one to a few years; contamination and Goodhart's law erode them along the way.
  Fresh, dated and private sets are the main defence.
- Models improve first where answers can be checked, because RL needs an automatic reward. Evaluation and
  training share the same bottleneck.
- Evals are the test suite of ML: automated, versioned, regression-guarded, but statistical rather than
  pass/fail.

## Further reading

- [Miller 2024, "Adding Error Bars to Evals"](https://arxiv.org/abs/2411.00640): the statistics of this
  chapter, written for practitioners, with clustered and paired standard errors.
- [Chen et al. 2021, "Evaluating Large Language Models Trained on Code"](https://arxiv.org/abs/2107.03374):
  HumanEval and the pass@k estimator.
- [Zheng et al. 2023, "Judging LLM-as-a-Judge"](https://arxiv.org/abs/2306.05685): MT-Bench, Chatbot Arena's
  beginnings and the catalogue of judge biases.
- [Chiang et al. 2024, "Chatbot Arena"](https://arxiv.org/abs/2403.04132): how pairwise human votes become a
  Bradley–Terry leaderboard, with confidence intervals.
- [Singh et al. 2025, "The Leaderboard Illusion"](https://arxiv.org/abs/2504.20879): how private testing and
  data access skew arena rankings.
- [Biderman et al. 2024, "Lessons from the Trenches on Reproducible Evaluation"](https://arxiv.org/abs/2405.14782):
  everything that silently changes a score, from the lm-evaluation-harness team.
- [Oren et al. 2023, "Proving Test Set Contamination in Black Box Language Models"](https://arxiv.org/abs/2310.17623):
  the clever shuffled-order test.
- [OpenAI 2026, "Why we no longer evaluate SWE-bench Verified"](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/):
  a benchmark's end of life, from a lab's side.
- [Stanford AI Index 2026](https://hai.stanford.edu/ai-index/2026-ai-index-report), chapter 2: the yearly
  snapshot of which benchmarks have saturated.

## Review

<Flashcards id="evaluation" :cards="cards" />

<MarkDone id="evaluation" />
