---
title: "5. Pretraining and Scaling Laws"
---

<script setup>
import { cards } from './pretraining-review'
</script>

# 5. Pretraining and Scaling Laws

Pretraining is the long, expensive first phase where a model reads trillions of words and learns to guess
the next one. Interviewers care because almost everything else follows from it: what the model knows, what
it costs, how big it should be, and why labs fight over data and GPUs.

::: info Before you start
- A neural network is a large set of numbers (its <Term id="parameter">parameters</Term>, or weights) plus a fixed program that
  turns an input into an output using them. See [Chapter 1](/foundations/neural-nets).
- Training means: measure how wrong the output is (the **loss**), work out which way to nudge every
  parameter to reduce it (the <Term id="gradient">gradient</Term>), and nudge. See [Chapter 2](/foundations/learning).
- Text is cut into **tokens**, chunks of about 4 characters of English, before a model sees it. See
  [Chapter 3](/foundations/tokens-embeddings).
- The model in this chapter is a transformer: a network that reads a sequence of tokens and outputs a
  guess for the next one. See [Chapter 4](/foundations/transformer).
:::

## The objective: predict the next token

**In short:** the model sees text and is graded on how much probability it gave to the token that really
came next. That one simple test forces it to learn a great deal about the world.

Take the sentence "The capital of France is Paris". Show the model "The capital of France is" and ask for
the next token. It does not pick one answer. It outputs a probability for every token in its vocabulary (the fixed list of tokens it knows),
perhaps 0.9 for " Paris", 0.01 for " a", and tiny amounts for the other 100,000 or so.

The grade is the negative log of the probability it gave to the right answer. If it gave " Paris" 0.9, the
penalty is −log(0.9) ≈ 0.1. If it gave 0.001, the penalty is −log(0.001) ≈ 6.9. Confident and right costs
almost nothing; confident and wrong costs a lot. This penalty is the <Term id="cross-entropy">cross-entropy loss</Term> from
[Chapter 2](/foundations/learning), measured in **nats** (natural-log units) per token.

This is <Term id="next-token-prediction">next-token prediction</Term>, and
<Term id="pretraining">pretraining</Term> is running it over a huge pile of text. Nobody labels anything.
The text is its own answer key: every position in every document is a question whose answer is the next
token. One document of 4,000 tokens gives 4,000 graded guesses in a single pass, because the transformer
makes all of them at once and each guess may only look at the tokens before it.

### Why such a plain objective teaches so much

To guess the next token well, the model must model whatever produced the text. Some examples:

- "The capital of France is ___" needs a fact.
- "She dropped the glass and it ___" needs some physics and common sense.
- "def fib(n): return ___" needs programming.
- The last line of a worked proof needs the reasoning that led there.
- A forum reply needs a guess at what the writer believes and wants.

The loss keeps falling only as the model picks up more of this structure. That is why one objective, with
no task-specific labels, produces a model that can translate, summarise and write code.

There is a second way to see it. A model that predicts text well can compress text well: good predictions
mean short codes, the same idea a compression tool uses. [Delétang et al. 2023](https://arxiv.org/abs/2309.10668)
showed large language models are strong general-purpose compressors. "Prediction is compression is
understanding" is a popular slogan in the field; how far the last step holds is contested.

### What the objective does not do

The objective rewards imitating the text, not being right. If most of the internet repeats a myth, the
best next-token predictor repeats the myth. The raw pretrained model, the **base model**, also does not
follow instructions. Ask it a question and it may continue with three more questions, because that is what
a list of questions looks like. Turning it into a helpful assistant is post-training,
[Chapter 7](/building/post-training).

Whether next-token prediction can ever reach human-level reasoning is a live argument. Yann LeCun, for one,
has long argued that predicting text one token at a time is a dead end for full intelligence. Most
frontier labs are betting that it is a sound base when combined with post-training and reinforcement
learning, which trains on rewards rather than on text to imitate (contested).

::: details Going deeper: the loss as a number
- The loss for a whole run is the average of −log p(correct token) over every token seen. Lower is better.
- A model that guessed uniformly over a 128,000-token vocabulary would score ln(128,000) ≈ 11.8 nats.
  Trained models score a small fraction of that.
- Loss values are only comparable with the same tokenizer and the same evaluation text. A tokenizer with
  bigger tokens packs more information into each one, so the loss per token goes up even if the model is
  equally good.
- **Perplexity** is exp(loss). A perplexity of 6 means the model is, on average, as unsure as if choosing
  evenly among 6 tokens.
:::

## The data pipeline

**In short:** pretraining data is mostly filtered web crawl, plus code, books, papers and, increasingly,
model-written text. Most of the quality of a model is decided here, in a pipeline that throws away the
large majority of what it starts with.

If you build data pipelines, this part of the field will look familiar: extract, filter, deduplicate,
score, sample. The difference is scale (petabytes of HTML) and the fact that the "consumer" is a model whose
quality you can only measure by training it.

<PretrainDataPipelineDiagram />

### Step 1: the raw crawl

The base of almost every open dataset is <Term id="common-crawl">Common Crawl</Term>, a non-profit that
crawls the web and publishes a new snapshot every month or two. Each snapshot holds billions of pages.
Labs also run their own crawlers.

Raw HTML is mostly not text you want: menus, cookie banners, ads, footers, boilerplate. The first job is
**text extraction**: turn HTML into clean running text. Then a language detector tags each document so the
pipeline can keep, drop or route it by language. The choice of extractor matters more than you would
expect. The [FineWeb](https://arxiv.org/abs/2406.17557) team found that extracting text from the raw HTML
themselves beat using the plain-text files Common Crawl ships.

### Step 2: rule-based filters

Next come cheap rules, each one written after someone looked at a lot of bad data:

- **Length**: drop pages with too few words, or lines that are too short to be prose.
- **Symbols and characters**: drop pages that are mostly punctuation, digits or odd characters.
- **Repetition**: drop pages where the same line or phrase repeats many times (spam, generated listings).
- **Blocklists**: drop known adult, malware and spam domains.
- **Safety and privacy**: remove or mask personal data such as email addresses.

These rules run per document and parallelise trivially, like any stateless map step.

### Step 3: deduplication

The web repeats itself: mirrors, scraped copies, templated pages, the same licence text on a million sites.
Training on duplicates wastes compute and makes the model memorise those passages word for word.
[Lee et al. 2021](https://arxiv.org/abs/2107.06499) showed that removing duplicates reduces verbatim
memorisation and trains a model at least as good with fewer steps.

<Term id="deduplication">Deduplication</Term> happens at several levels:

- **Exact**: hash each document (or each URL) and keep one copy per hash.
- **Near-duplicate**: two pages that differ only in a date or a sidebar should count as copies. Comparing
  every pair is impossible at billions of documents, so pipelines use <Term id="minhash">MinHash</Term>.
- **Line or substring level**: remove lines or long passages that repeat across many documents, such as
  boilerplate that survived extraction.

MinHash works like this. Cut each document into overlapping word n-grams (say, every run of 5 words). Two
documents are similar if they share a large fraction of their n-grams. MinHash builds a short signature per
document so that two signatures agree in a fraction of slots equal, on average, to that overlap. Then
**locality-sensitive hashing** groups signatures into buckets so only documents in the same bucket are
compared. It is the same trick as a hash join, made approximate on purpose.

Deduplication is the one step that is not a stateless map: it needs a global view of the corpus. That
makes it the step most like a large distributed shuffle.

::: details Going deeper: global versus per-snapshot deduplication
The FineWeb team found a surprise. Deduplicating across all 96 snapshots at once made the data **worse**
than deduplicating within each snapshot. When they inspected older snapshots, the documents that global
deduplication removed looked higher in quality than the ones it kept
([FineWeb paper](https://arxiv.org/abs/2406.17557)). A plausible reason is that good pages stay online and
get re-crawled, so they look like duplicates, while one-off junk does not. The lesson generalises: every filter changes the
distribution of what is left, and you only find out by training on it.
:::

### Step 4: model-based quality filtering

Rules catch obvious junk. They cannot tell a thoughtful explainer from a well-formed but empty SEO page.
For that, pipelines train a <Term id="quality-classifier">quality classifier</Term>: a small, cheap model
that scores each document, and the pipeline keeps the top slice.

The recipe is usually:

1. Pick examples of what "good" means. Options include pages cited by Wikipedia, textbook-like text, or
   answers to high-quality questions.
2. Optionally, ask a large language model to grade a sample of documents, which gives cheap labels at scale.
3. Train a small classifier on those labels. It must be fast, because it runs on billions of documents.
4. Score everything and keep documents above a threshold.

Two public examples show the effect. **FineWeb-Edu** asked Llama-3-70B-Instruct to rate about 450,000 web
pages for educational value on a 0–5 scale, trained a small classifier on those ratings, and kept pages
scoring 3 or more. The result was 1.3 trillion tokens, out of FineWeb's 15 trillion, that trained much
better models on knowledge and reasoning tests ([FineWeb paper](https://arxiv.org/abs/2406.17557)).
**DCLM** started from a 240-trillion-token pool, compared many filters, and found that a simple fastText
classifier (a fast, shallow text classifier) was the best one; its filtered set trained a 7B model to 64% on the MMLU knowledge test with 2.6
trillion tokens ([DCLM paper](https://arxiv.org/abs/2406.11794)).

The catch is that "quality" means "looks like my positive examples". Filter too hard toward textbook prose
and you lose dialects, forums, informal writing and rare topics. Many pipelines also run
**decontamination**: removing text that overlaps with benchmark test sets, so later scores are honest
([Chapter 9](/running/evaluation)).

::: details Going deeper: Llama 3's web pipeline
The [Llama 3 paper](https://arxiv.org/abs/2407.21783) (section 3.1) describes a full pipeline:

- A custom HTML parser; personal-data and unsafe-domain filters.
- Deduplication at URL level, document level (global MinHash), and line level (removing lines that appear
  more than 6 times in each bucket of 30 million documents).
- Heuristics: duplicated n-gram ratios, "dirty word" counts, and a check for documents whose token
  distribution looks unlike the rest of the corpus.
- Model-based quality: a fastText classifier trained to recognise text that Wikipedia would cite, and
  RoBERTa-based classifiers trained on quality labels produced by Llama 2.
- Separate classifiers to find code and math-heavy pages, which are processed with their own extractors.
:::

### Step 5: the mixture

The filtered web is one source among several. A <Term id="data-mixture">data mixture</Term> sets how much
of each source the model sees: web text, code, scientific papers, books, math, multilingual text. Mixture
weights are a strong lever. More code tends to help reasoning; too much of any one source hurts the others.

Llama 3's final mix was roughly 50% general knowledge, 25% math and reasoning, 17% code and 8%
multilingual tokens ([Llama 3 paper](https://arxiv.org/abs/2407.21783)). Labs choose these weights by
training many small models on different mixtures and extrapolating, which is itself a scaling-law
exercise (next section).

Mixtures also change during the run. A common pattern is to finish training on a smaller set of the
highest-quality data (**annealing**, covered below). Llama 3 did this, and so did
[Qwen3](https://arxiv.org/abs/2505.09388), whose models trained on about 36 trillion tokens in stages:
general text first, then extra science and code, then long documents.

### Synthetic data

**Synthetic data** is text written by another model, made on purpose for training. It comes in two kinds:

- **Rephrasing**: take a real web page and ask a model to rewrite it as a clear explanation or as
  question-and-answer pairs. The facts come from the web; the model only cleans the form.
  [Maini et al. 2024](https://arxiv.org/abs/2401.16380) found this sped up pretraining about threefold.
- **Generation from scratch**: ask a model to write textbooks, exercises or code. Microsoft's Phi models
  started this way ([Gunasekar et al. 2023](https://arxiv.org/abs/2306.11644)).

Synthetic data is attractive because you control its topic, format and difficulty, and it does not run
out. The risk is narrowing. [Shumailov et al. 2024](https://www.nature.com/articles/s41586-024-07566-y)
showed that models trained repeatedly on their own outputs lose the rare cases first and then degrade,
which they called **model collapse**. Others point out that this happens when synthetic data replaces real
data, and is much milder when it is added alongside it. How far synthetic data can substitute for human
text is contested.

### What the pipeline produces

The output is a set of shuffled, tokenized shards: long arrays of token IDs, packed into fixed-length
sequences (often 4,000–8,000 tokens) with documents joined end to end. Data loaders stream these to the
GPUs. Labs also keep exact records of which shard was read at which step, so a crashed run can resume
without seeing data twice or skipping it.

::: info Back-of-envelope: how big is 15 trillion tokens?
- One token is about 4 bytes of English text, so 15T tokens ≈ 60 TB of plain text.
- One token is about 0.75 English words, so 15T tokens ≈ 11 trillion words.
- A fast human reader at 250 words per minute, reading 8 hours a day, gets through about 44 million
  words a year. 11 trillion words is about 250,000 years of reading.
- The raw crawl it came from was many times larger: every filter above removed a large share.
:::

## Scaling laws

**In short:** loss falls smoothly and predictably as you add parameters, data and compute. That lets labs
test on small models and predict a large run before paying for it.

Here is the observation that changed the field. Train a family of models of different sizes on different
amounts of data. Plot the final loss against model size on log-log axes. The points fall on a nearly
straight line, across many orders of magnitude. A straight line on log-log axes is a **power law**: loss
goes down by a fixed fraction each time you multiply the size by a fixed factor.

These regularities are called <Term id="scaling-law">scaling laws</Term>. They matter for a practical
reason: a frontier run costs tens to hundreds of millions of dollars, and you get one shot. Scaling laws
let you run cheap experiments at 1/1000th of the size, fit the curve, and extrapolate.

### Kaplan 2020: bigger models first

[Kaplan et al. 2020](https://arxiv.org/abs/2001.08361) at OpenAI found power laws in model size, data and
compute separately. They also asked: with a fixed compute budget, how should you split it between a bigger
model and more data? Their answer was to grow the model much faster than the data. That advice shaped GPT-3:
175 billion parameters trained on only about 300 billion tokens, under 2 tokens per parameter.

### Chinchilla 2022: grow both equally

[Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556) at DeepMind redid the experiment more carefully,
with over 400 models, and reached a different answer. For a fixed budget, model size and data should grow
**at the same rate**. Double your compute, and make the model about 1.4× bigger and the data about 1.4×
bigger.

The headline rule of thumb is about **20 tokens per parameter**. They tested it directly. Chinchilla, a 70B
model trained on 1.4 trillion tokens, beat their own Gopher, a 280B model trained on 300 billion tokens,
using the same compute. Gopher had been too big for its data. A model trained at this ratio is called
<Term id="compute-optimal">compute-optimal</Term> (or "Chinchilla-optimal").

Why did Kaplan get a different answer? The Chinchilla authors blamed the learning-rate schedule (how the
step size changes during training, covered below), which Kaplan did not match to each run's length. Later
replications point to more setup details: Kaplan left the embedding table (the lookup table that turns
tokens into vectors) out of the parameter count, worked at small scale, and did not tune warmup and the
optimizer per size ([Pearce and Song 2024](https://arxiv.org/abs/2406.12907);
[Porian et al. 2024](https://arxiv.org/abs/2406.19146)). Both papers recover the Chinchilla result once
these are fixed.

### The Chinchilla formula

Chinchilla fitted a formula for the final loss L of a model with N parameters trained on D tokens:

**L(N, D) = E + A / N<sup>α</sup> + B / D<sup>β</sup>**

In words:

- **E** is the floor: the loss even an infinite model on infinite data would have, because text is partly
  unpredictable. Nobody can guess a random phone number.
- **A / N<sup>α</sup>** is the penalty for a model that is too small. It shrinks as N grows.
- **B / D<sup>β</sup>** is the penalty for too little data. It shrinks as D grows.
- **α** and **β** (both about 0.3) say how fast each penalty shrinks. Small exponents mean slow progress:
  halving a penalty needs roughly 10× more parameters or data.

Compute ties N and D together. To spend a budget C wisely, choose the N and D that make L smallest. The
answer is the 20-tokens-per-parameter rule.

::: details Going deeper: the Chinchilla fit was itself corrected
[Besiroglu et al. 2024](https://arxiv.org/abs/2404.10102) at Epoch AI re-extracted Chinchilla's data from
the paper's plots and refitted the formula. The published constants (E = 1.69, A = 406.4, B = 410.7,
α = 0.34, β = 0.28) imply about 70 tokens per parameter, which contradicts Chinchilla's own other two
methods. Their refit (E = 1.82, A = 482, B = 2085, α = 0.35, β = 0.37) implies about 20, consistent with
the rest of the paper. The exercise below uses the refit. Lesson: the "20" comes from the fixed-budget
experiments, not from the much-quoted formula's original constants.
:::

### Why modern models train far past Chinchilla

Chinchilla answers one question: given a training budget, what gives the lowest loss? Labs care about a
different one: what gives the best model per dollar **over its whole life**, including serving it.

Serving cost depends on model size, not on how long it was trained. Each generated token costs about 2N
operations (next section). A model that will answer billions of requests is cheaper overall if it is
smaller and trained much longer, even if that wastes some training compute.
[Sardana et al. 2024](https://arxiv.org/abs/2401.00448) worked this out formally.

So modern small models are heavily <Term id="overtraining">overtrained</Term>, meaning trained on far more
than 20 tokens per parameter:

| Model | Parameters | Training tokens | Tokens per parameter |
|---|---|---|---|
| GPT-3 (2020) | 175B | ~0.3T | ~2 |
| Chinchilla (2022) | 70B | 1.4T | 20 |
| Llama 3 8B (2024) | 8B | ~15T | ~1,900 |
| Llama 3.1 405B (2024) | 405B | 15.6T | ~40 |
| Qwen3 0.6B (2025) | 0.6B | ~36T | ~60,000 |

Sources: [GPT-3](https://arxiv.org/abs/2005.14165), [Chinchilla](https://arxiv.org/abs/2203.15556),
[Llama 3](https://arxiv.org/abs/2407.21783), [Qwen3](https://arxiv.org/abs/2505.09388) (which says all its
models saw over 30 trillion tokens).

Note the Llama 3.1 405B row. Meta fitted its own scaling law, which pointed to a 402B model on 16.55T
tokens for its budget, and trained close to that. The flagship was near compute-optimal by Meta's fit; the
small models were the ones pushed far past it. Loss keeps falling with more data long past 20 tokens per
parameter; it falls more slowly, but it does fall.

::: info Back-of-envelope: compute-optimal size for a budget
C is the training compute, counted in <Term id="flop">FLOPs</Term> (floating-point operations). Use
C ≈ 6ND (next section) and D ≈ 20N. Then C ≈ 120N², so N ≈ √(C / 120).

- Chinchilla's budget, C ≈ 5.8 × 10<sup>23</sup> FLOPs: N ≈ √(4.8 × 10<sup>21</sup>) ≈ 69B parameters,
  D ≈ 1.4T tokens. That matches the real model.
- A budget of 10<sup>25</sup> FLOPs: N ≈ √(8.3 × 10<sup>22</sup>) ≈ 290B, D ≈ 5.8T tokens.
:::

### What scaling laws do not tell you

Scaling laws predict **loss**, not abilities. Whether a lower loss means the model can now do long
division or write a working web server is a separate question. Some abilities seem to switch on suddenly at
a certain scale ("emergence"). [Schaeffer et al. 2023](https://arxiv.org/abs/2304.15004) argued much of
that suddenness comes from all-or-nothing scoring, and smooth metrics show smooth progress (contested).

Scaling laws are also specific to the data, the architecture and the tokenizer. Better data shifts the
whole curve down. That is why labs re-fit them for every new recipe.

::: tip Interview checkpoint
<details><summary>What did Chinchilla change compared with Kaplan et al.?</summary>

Kaplan said: with more compute, grow the model much faster than the data. Chinchilla, with more careful
experiments, found model size and data should grow equally, about 20 tokens per parameter. Their 70B model
on 1.4T tokens beat a 280B model on 300B tokens with the same compute. Later work traced Kaplan's result to
setup choices: a learning-rate schedule not matched to run length, leaving embedding parameters out of the
count, and small scale.
</details>
<details><summary>Llama 3 8B saw 15T tokens, about 1,900 per parameter. Was that a mistake?</summary>

No. Chinchilla minimises training compute for a given loss. But serving cost scales with parameters, and an
8B model is served billions of times. Spending extra training compute to make a small model better is cheap
compared with serving a bigger model forever. Loss still improves past 20 tokens per parameter, only more
slowly.
</details>
<details><summary>How would you choose model size for a new run with a fixed budget?</summary>

Train a ladder of small models at several compute budgets. For each budget, try several sizes and find the
size with lowest loss (an "iso-FLOP" curve). Fit a power law of best size against compute and extrapolate
to your budget. Then adjust down if inference cost matters, and check the result with a few mid-size runs.
</details>
<details><summary>What does a scaling law not tell you?</summary>

It predicts loss on the training distribution, not specific skills or benchmark scores. It holds for one
data recipe, architecture and tokenizer, so changing any of those means re-fitting. And it says nothing
about whether the run will be stable.
</details>
:::

## Compute: FLOPs ≈ 6ND

**In short:** training costs about 6 floating-point operations per parameter per token. Multiply by tokens,
divide by what your GPUs actually achieve, and you have the GPU-hours.

A **FLOP** is one floating-point operation: one add or one multiply. Training compute is counted in total
FLOPs; hardware speed in FLOP/s, operations per second. The two are easy to mix up in papers.

### Where the 6 comes from

Recall from [Chapter 1](/foundations/neural-nets) that most of a transformer's work is matrix multiplies:
each parameter is a weight that multiplies an incoming number, and the products are added up.

- **Forward pass: 2 per parameter per token.** Each weight takes part in one multiply and one add for each
  token that passes through. So a forward pass costs about 2N FLOPs per token.
- **Backward pass: 4 per parameter per token.** Backpropagation ([Chapter 2](/foundations/learning)) does
  two jobs at each layer. It works out the gradient for the layer's inputs, to pass further back. It also
  works out the gradient for the layer's weights, to update them. Each job is a matrix multiply the same
  size as the forward one, so 2N + 2N = 4N.

Add them: 6N FLOPs per token. Over D training tokens:

**C ≈ 6 · N · D**

where C is total training compute in FLOPs, N is the parameter count, and D is the number of training
tokens. Inference needs only the forward pass, so generating one token costs about **2N**.

::: details Going deeper: what 6ND leaves out
- **Attention** adds work that grows with context length, roughly 6 × (layers) × (context) × (width) extra
  FLOPs per token. For large models at a few thousand tokens of context it is a small share; at very long
  context it matters.
- The optimizer update, normalisation and activation functions are small next to the matrix multiplies.
- **Activation recomputation** (redoing part of the forward pass during the backward pass to save memory,
  [Chapter 6](/building/training-at-scale)) adds real work that 6ND does not count.
- <Term id="moe">Mixture-of-experts</Term> models send each token through only some of their weights. For them, N in 6ND is
  the **active** parameters per token, not the total. DeepSeek-V3 has 671B parameters but uses about 37B
  per token ([DeepSeek-V3 report](https://arxiv.org/abs/2412.19437)).
:::

### MFU: how much of the GPU you actually use

A GPU's spec sheet gives a peak rate. An H100 does about 1,000 trillion FLOP/s (10<sup>15</sup>) of 16-bit
matrix math. Real training gets much less. <Term id="mfu">Model FLOPs utilisation (MFU)</Term> is the
fraction you achieve:

**MFU = (6ND ÷ wall-clock time) ÷ (number of GPUs × peak FLOP/s per GPU)**

MFU counts only the "useful" 6ND work. Time spent on communication between GPUs, memory-bound operations,
GPUs waiting on each other and recomputation all lower it. Good dense-model runs land around 35–50%. Llama 3
405B reported 38–43% on up to 16,384 H100s ([Llama 3 paper](https://arxiv.org/abs/2407.21783)).

MFU does not include time lost to failures and restarts. At this scale those are routine: over a 54-day
window, Llama 3's pretraining saw 466 job interruptions. 419 were unplanned, about one every three hours,
and about 78% of those came from hardware. Keeping that wasted time low is a major engineering effort, covered in [Chapter 6](/building/training-at-scale).

### A worked cost estimate: Llama 3.1 405B

::: info Back-of-envelope: what did Llama 3.1 405B cost?
**FLOPs.** N = 405 × 10<sup>9</sup>, D = 15.6 × 10<sup>12</sup>.
6 × 405 × 10<sup>9</sup> × 15.6 × 10<sup>12</sup> ≈ **3.8 × 10<sup>25</sup> FLOPs**. Meta's paper gives
the same figure.

**GPU-hours.** An H100 at 40% MFU does about 0.4 × 10<sup>15</sup> = 4 × 10<sup>14</sup> FLOP/s.
3.8 × 10<sup>25</sup> ÷ 4 × 10<sup>14</sup> ≈ 9.5 × 10<sup>10</sup> GPU-seconds ≈ **26 million GPU-hours**.
On 16,384 GPUs that is about 1,600 hours, or **68 days**.

**Check.** Meta's model card reports
[30.84 million H100-hours](https://huggingface.co/meta-llama/Llama-3.1-405B) for the 405B model. Our
estimate is within 15%; Meta does not break down the gap.

**Dollars.** H100 rental in late 2026 runs from under $2 to over $10 per GPU-hour, with typical
specialist-cloud prices around $2–4 ([IntuitionLabs price survey](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)).
At $2–4 per hour, 31M GPU-hours is **$60–125 million**, for the final run alone.
:::

The final run is not the whole bill. Small-scale experiments, data processing, failed runs, staff and
post-training add a lot. [DeepSeek-V3](https://arxiv.org/abs/2412.19437) is a useful contrast: it reported
2.79 million H800 GPU-hours, or $5.6M at $2 per hour. It is a mixture-of-experts model, which sends
each token through only part of its weights: 37B of its 671B parameters, on 14.8T tokens. Its report
states that this excludes prior research and ablations. Headlines
that compared $5.6M with other labs' total spending compared different things.

## Exercise: a scaling-law calculator

The script does four things with NumPy:

1. Generates noisy "training runs" from the refitted Chinchilla formula, finds the best model size at each
   budget, and fits a power law, as the Chinchilla paper did.
2. Prints compute-optimal parameters and tokens for several budgets.
3. Compares Llama 3 8B's overtrained recipe with a compute-optimal model of the same training cost.
4. Repeats the 405B cost estimate.

<<< @/../exercises/ch05_scaling.py

Run it with `~/.venvs/llm/bin/python exercises/ch05_scaling.py`. Output:

```text
Part 1: fit N_opt = k * C^a from noisy iso-FLOP sweeps
  fitted exponent a = 0.509  (true value from the formula: 0.513)
  so tokens D_opt = C / (6 N_opt) grow as C^0.491: both grow about as sqrt(C)

Part 2: compute-optimal N and D for a budget
      FLOPs   params N   tokens D    D/N   loss
   1.00e+21   2.78e+09   6.00e+10   21.6  2.306
   1.00e+23   2.94e+10   5.66e+11   19.2  2.032
   5.76e+23   7.22e+10   1.33e+12   18.4  1.974
   1.00e+25   3.12e+11   5.34e+12   17.1  1.912
   3.80e+25   6.19e+11   1.02e+13   16.5  1.892
  (5.76e23 is Chinchilla's own budget; it used 70B params and 1.4T tokens.)

Part 3: overtraining an 8B model on 15T tokens (Llama 3 8B's recipe)
  compute: 7.20e+23 FLOPs, tokens per param = 1875
  8B on 15T:              loss 2.022
  compute-optimal 81B on 1.5T:  loss 1.968
  inference cost per token is about 2N FLOPs, so the optimal model costs 10.1x more to serve
  extra loss from choosing 8B: 0.054 nats per token (small, but not zero)

Part 4: cost of Llama 3.1 405B pretraining from 6ND
  FLOPs = 6 * 4.05e+11 * 1.56e+13 = 3.79e+25
  at 40% MFU: 26.6M GPU-hours = 68 days on 16,384 GPUs
  at $2.00/GPU-hour: $53M
  at $4.00/GPU-hour: $106M
  Meta reports 30.84M GPU-hours for the 405B model (it does not break down the gap).
```

What to notice:

- Part 1 recovers an exponent near 0.5 from noisy data: model size and data each grow as about the square
  root of compute.
- Part 2 shows the optimal ratio drifting from about 22 to 16 tokens per parameter as budgets grow. "20" is
  a rule of thumb, not a constant.
- Part 3 is the inference-economics argument in numbers. The overtrained 8B model pays a small loss penalty
  and is 10× cheaper to serve.
- The loss values come from Chinchilla's data and tokenizer. Do not compare them with other models' losses.

Try changing the budget in Part 3, or swap in the original Chinchilla constants from the "Going deeper" box
above. The ratio jumps to roughly 50–120 tokens per parameter, depending on the budget.

## Learning rate, warmup and loss spikes

**In short:** the learning rate starts near zero, ramps up, then decays. Even so, big runs sometimes see
the loss suddenly jump, and teams roll back and skip data to recover.

### The schedule

Each training step processes one **batch**, a group of sequences, and updates the weights once. The
**learning rate** sets the size of that update ([Chapter 2](/foundations/learning)). Pretraining does not
keep it fixed. A typical <Term id="learning-rate-schedule">learning-rate schedule</Term> has three
phases:

1. **Warmup.** Start with a tiny rate and raise it linearly over the first few thousand steps. At the start
   the weights are random and the gradients are large and erratic. <Term id="adam">Adam</Term>, the usual optimizer, also needs a
   few steps to estimate how large each gradient typically is. Full-size steps at this point can throw the
   model into a bad region it never leaves.
2. **Peak and decay.** Hold or slowly lower the rate for most of the run. A common shape is a **cosine**
   curve that falls smoothly to about a tenth of the peak.
3. **Final decay.** Large steps keep the model bouncing around the bottom of the loss valley. Shrinking
   them at the end lets it settle, and the loss drops noticeably.

The cosine schedule has a practical flaw: you must know the total length in advance. The
**warmup-stable-decay** (WSD) schedule fixes this. It holds the rate constant, then decays it quickly at
the end ([MiniCPM paper](https://arxiv.org/abs/2404.06395)). You can branch a short decay off any
**checkpoint** (a saved copy of the weights and optimizer state) from the stable phase. That makes it cheap
to run scaling experiments at many lengths.

The final low-learning-rate phase is often combined with the best data. This is **annealing**: the model
spends its last, most careful steps on the highest-quality text. Llama 3 decayed the rate to zero over the
final 40 million tokens while up-weighting high-quality sources
([Llama 3 paper](https://arxiv.org/abs/2407.21783)).

::: details Going deeper: Llama 3 405B's settings
From the [Llama 3 paper](https://arxiv.org/abs/2407.21783), section 3.4: peak learning rate 8 × 10<sup>−5</sup>,
linear warmup over 8,000 steps, cosine decay to 8 × 10<sup>−7</sup> over 1.2 million steps. The batch
started at 4M tokens and grew to 8M and then 16M tokens as training went on, because small batches are
more stable early and large batches are more efficient later.
:::

### Loss spikes

Sometimes, deep into a stable run, the loss shoots up within a few steps. This is a
<Term id="loss-spike">loss spike</Term>. Some spikes recover by themselves. Others send the loss climbing
until the run is useless (**divergence**).

The causes are not fully understood (contested). Candidates include:

- A batch of unusual data, such as long runs of repeated or garbled tokens.
- Numerical trouble in 16-bit arithmetic, where some values grow until they overflow.
- Attention scores that grow very large, so attention snaps onto single tokens.
- Adam's per-parameter step sizes going stale for weights that rarely get gradients, then over-stepping.

Teams respond in layers. They watch the loss and gradient sizes live. When a spike diverges, they roll back
to a checkpoint from before it and skip the batches around it. PaLM's team restarted about 100 steps before
a spike and skipped 200–500 batches ([PaLM paper](https://arxiv.org/abs/2204.02311)). Preventive measures
include **gradient clipping** (capping how big a single update can be), lower learning rates, normalising
attention inputs, and a small extra loss that keeps output scores from drifting large. Meta's
[OPT-175B logbook](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles) is a
readable day-by-day record of a team fighting these problems.

## Running out of data

**In short:** compute keeps growing faster than the supply of good human text. Repeating data helps up to a
point; after that labs turn to synthetic data, other sources and reinforcement learning.

Compute-optimal training says data should grow with the square root of compute. Overtraining small models
uses even more. Human writing does not grow nearly as fast.

[Villalobos et al.](https://arxiv.org/abs/2211.04325) at Epoch AI estimated the stock of usable public
human text at about 300 trillion tokens, with a wide range of 100–1,000 trillion. They projected that
training sets would reach that size between 2026 and 2032
([Epoch summary](https://epoch.ai/publications/will-we-run-out-of-data-limits-of-llm-scaling-based-on-human-generated-data)).
Qwen3's 36T tokens is already about a tenth of the central estimate. Ilya Sutskever put it bluntly at NeurIPS
2024: "we have but one internet", and pretraining "as we know it will end"
([The Verge, via Techmeme](https://www.techmeme.com/241213/p33)).

### Repeating data

The first fallback is to train on the same data more than once. One full pass over the dataset is an
<Term id="epoch">epoch</Term>. [Muennighoff et al. 2023](https://arxiv.org/abs/2305.16264) measured what
repeats are worth: up to about 4 epochs is nearly as good as fresh data. Beyond that each repeat is worth
less, and around 16 epochs extra passes add almost nothing. Their data-constrained scaling law extends the
Chinchilla formula with a term for how fast repeated tokens lose value.

### Other responses

- **Better filtering and rephrasing** get more out of the same raw text (see synthetic data above).
- **More sources**: transcribed audio and video, licensed books and news, non-English text, private data.
- **Synthetic data** at large scale, with the collapse risk above.
- **Reinforcement learning** on problems with checkable answers, such as math and code, where the model
  generates its own training signal ([Chapter 7](/building/post-training)). Labs have moved a growing share
  of compute here since 2024; how large a share is not public.

Whether there is a real "data wall" is contested. Optimists point to multi-epoch training, synthetic data
and large untapped sources such as video. Pessimists note that the best human text is already used, and that synthetic
data may not add knowledge the generating model lacked.

::: warning Thread
Pretraining is the one phase where the model learns from the world at large. For a few months it absorbs
trillions of tokens. Then the weights are frozen, and every later fact reaches the model only through its
prompt. The knowledge cutoff is a direct, visible consequence. This fits the thread's intuition about
"months-frozen weights": the slow timescale exists, but it runs once and stops.

The evidence against is real. In-context learning, the model adapting to examples in its prompt, is
itself something pretraining teaches; the frozen model has learned a way to learn within a context. Labs
also retrain often, and **continued pretraining** on new data is routine. And pretraining is not
data-efficient in a brain-like way. Llama 3 saw about 11 trillion words. The
[BabyLM challenge](https://babylm.github.io/) uses 100 million words as a rough estimate of what a
13-year-old has heard, five orders of magnitude less. That gap suggests the issue is not only "frozen
versus updating", but how much each experience teaches.

The running-out-of-data problem cuts both ways. If good human text is nearly used up, a model that learns
from its own deployment would have a new data source. But that is exactly where a training signal is
hardest to trust, which is one of the thread's open problems.
:::

::: tip Interview checkpoint
<details><summary>Derive the 6ND rule.</summary>

The forward pass does one multiply and one add per parameter per token: 2N. The backward pass computes
gradients for each layer's inputs and for its weights, each about as costly as the forward pass: 4N. So
6N per token, times D tokens. Inference is forward only, so about 2N per generated token. It ignores
attention's context-length term and recomputation.
</details>
<details><summary>Estimate the GPU-hours for a 70B model on 15T tokens on H100s.</summary>

6 × 7 × 10<sup>10</sup> × 1.5 × 10<sup>13</sup> ≈ 6.3 × 10<sup>24</sup> FLOPs. At 40% MFU an H100 does
about 4 × 10<sup>14</sup> FLOP/s, so about 1.6 × 10<sup>10</sup> GPU-seconds, or about 4.4 million
GPU-hours. Meta reported 7.0 million for Llama 3.1 70B, so treat a pure 6ND estimate as a lower bound.
</details>
<details><summary>What is MFU and why is it well below 100%?</summary>

Achieved useful FLOP/s (6ND over wall-clock time) divided by the hardware's peak. It is lost to
communication between GPUs, memory-bound operations like normalisation, GPUs waiting on each other, and
recomputation. 35–50% is good for a large dense run. Failures and restarts lower the effective number
further but are tracked separately.
</details>
<details><summary>Why do we warm up the learning rate, and what do you do about a loss spike?</summary>

Early on, weights are random, gradients are erratic and Adam's statistics are not yet reliable, so full
steps can wreck the model. For a spike: if it does not recover, roll back to an earlier checkpoint, skip
the batches around it, and possibly lower the learning rate or tighten gradient clipping.
</details>
<details><summary>Your team has run out of fresh high-quality text. What are the options?</summary>

Repeat data for up to about 4 epochs with little loss; filter and rephrase existing text to extract more
value; add sources (code, transcripts, licensed data, other languages); add synthetic data alongside real
data; and shift compute to reinforcement learning on checkable tasks.
</details>
:::

## In practice

- **Data teams are as important as model teams.** FineWeb and DCLM showed that filtering and
  deduplication alone move benchmark scores a lot at fixed model size and compute. Most of that work is classic
  distributed data engineering with an unusual test: train a small model and measure it.
- **Large runs are planned with scaling laws.** Labs train ladders of small models to choose size, data mix
  and learning rate, then extrapolate. A run that lands off its predicted curve is a red flag.
- **Small models are trained for their serving cost.** Models meant for phones or high-volume APIs are
  trained on thousands of tokens per parameter.
- **Training clusters plan for failure.** At 16,000 GPUs something breaks every few hours, so checkpoints,
  fast restarts and data-loader determinism are core infrastructure ([Chapter 6](/building/training-at-scale)).
- **Loss curves get watched like production dashboards.** On-call engineers watch loss and gradient sizes,
  and roll back on spikes.

## Common misconceptions

- **"Chinchilla says you should train on 20 tokens per parameter."** It says that minimises training
  compute for a given loss. If you will serve the model a lot, train a smaller model for longer.
- **"Pretraining teaches the model to be helpful."** A base model continues text. Helpfulness comes from
  post-training.
- **"More data is always better."** Duplicated or low-quality data can make a model worse. Filtering often
  beats raw volume.
- **"The training run is the cost of a model."** The final run is a fraction of the total once experiments,
  failures, data work and staff are added.
- **"Scaling laws predict what the model will be able to do."** They predict loss. Specific abilities are
  a separate, less predictable question.

## Key takeaways

- Next-token prediction grades the model on the probability it gave the real next token. Doing it well
  forces the model to learn facts, language, code and reasoning.
- The data pipeline (extract, filter, deduplicate, score, mix) decides much of a model's quality.
- Loss follows power laws in parameters and data. Chinchilla's compute-optimal ratio is about 20 tokens per
  parameter, but small models are trained far past it to cut serving cost.
- Training compute is about 6ND FLOPs. Divide by GPUs × peak × MFU (often about 40%) for time, then
  multiply by price.
- Good human text is running short; repetition, synthetic data and reinforcement learning are the main
  responses, and how far they go is contested.

## Further reading

- [Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models"](https://arxiv.org/abs/2203.15556):
  the Chinchilla paper. Read the fixed-budget ("iso-FLOP") figures.
- [Kaplan et al. 2020, "Scaling Laws for Neural Language Models"](https://arxiv.org/abs/2001.08361): the
  original power-law plots, still the clearest picture of smooth scaling.
- [Penedo et al. 2024, "The FineWeb Datasets"](https://arxiv.org/abs/2406.17557): the most detailed public
  account of a web-data pipeline, with ablations for each step. Best read for a data engineer.
- [Li et al. 2024, "DataComp-LM"](https://arxiv.org/abs/2406.11794): a controlled benchmark of filtering
  methods on a 240T-token pool.
- [Llama 3 team 2024, "The Llama 3 Herd of Models"](https://arxiv.org/abs/2407.21783): sections 3 and 3.3–3.4
  cover data, scaling-law fits, infrastructure and the training recipe for a frontier-scale open model.
- [Muennighoff et al. 2023, "Scaling Data-Constrained Language Models"](https://arxiv.org/abs/2305.16264):
  what repeated data is worth.
- [Villalobos et al. 2024, "Will we run out of data?"](https://arxiv.org/abs/2211.04325): the data-supply
  projections.
- [Besiroglu et al. 2024, "Chinchilla Scaling: A replication attempt"](https://arxiv.org/abs/2404.10102):
  a short lesson in checking famous numbers.

## Review

<Flashcards id="pretraining" :cards="cards" />

<MarkDone id="pretraining" />
