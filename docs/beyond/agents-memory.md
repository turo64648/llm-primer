---
title: "11. Agents, Tools, RAG and Memory"
---

<script setup>
import { cards } from './agents-memory-review'
</script>

# 11. Agents, Tools, RAG and Memory

A deployed model's weights are frozen, yet products built on it search the web, run code and "remember" you
between chats, all by putting text into the model's input. Interviewers ask about agents, retrieval and memory because that is where most applied LLM engineering now happens.

::: info Before you start
- A model reads a sequence of **tokens** (chunks of about 4 characters) and outputs a probability for each
  possible next token. It writes text by doing this repeatedly. See [Chapter 3](/foundations/tokens-embeddings).
- Attention lets every token read every earlier token, and the stored per-token state, the KV cache, grows
  with the input. See [Chapter 4](/foundations/transformer) and [Chapter 8](/running/inference).
- After training, the model's numbers (its weights) do not change while it serves users. Post-training
  taught it to follow a chat format and to reason. See [Chapter 7](/building/post-training).
:::

## The context window is everything the model knows right now

**In short:** a deployed model has two kinds of knowledge: what is baked into its weights during training, and
the text in front of it. Agents, tools, retrieval and memory are all ways of choosing that text.

Ask a chat model what happened in the news this morning. Its weights were fixed months ago, so it cannot know. Paste
the morning's headlines into the prompt, and it answers well. Nothing about the model changed; only its input did.

The input has a hard size limit, the <Term id="context-window">context window</Term>. It holds the system
instructions, the conversation, any pasted documents, and the model's own output so far. In 2026 frontier models
advertise windows of about 200,000 to 1 million tokens. Anything outside the window does not exist for the model.

Keep one picture in mind for the whole chapter. The weights are like a compiled program, fixed until the next release.
The context window is like that program's RAM for one request: fast, flexible, and wiped when the request ends. Every
technique below is a way to load the right bytes into that RAM. The analogy breaks in one place: the "program" does not
execute the RAM's contents exactly; it reads them and responds, sometimes ignoring parts.

## Tool use: the model writes text, the harness acts

**In short:** the model cannot run anything. It writes a structured request, such as a function name and JSON
arguments. Ordinary code parses it, runs the real function, and pastes the result back into the context.

Suppose you ask an assistant for the weather in Oslo. The assistant's input includes a short description of a tool:
its name `get_weather`, what it does, and the arguments it takes. The model, instead of answering, writes something like
this and stops:

```json
{"type": "tool_use", "name": "get_weather", "input": {"city": "Oslo"}}
```

A program around the model sees this, calls a real weather API, and appends the result to the conversation:

```json
{"type": "tool_result", "content": "Oslo: 4 °C, light rain"}
```

Then it runs the model again on the longer conversation. Now the model writes "It is 4 °C and raining in Oslo."

This is <Term id="tool-use">tool use</Term>, also called function calling. The program around the model is the <Term id="harness">harness</Term> (or scaffold). It builds the prompt, parses tool calls, runs tools, enforces
permissions, and decides when to stop. The model only ever produces text; the harness turns some of that text into actions.

### Function-calling formats

Each API wraps the same idea differently. Tool definitions are usually JSON Schema: a name, a description, and typed
parameters. The provider renders them into the prompt using special marker tokens from the model's chat template, the
fixed format that turns a conversation into one token sequence. The model's reply then contains a tool-call block
that the API returns as structured data rather than as plain text.

Details vary. Some APIs let the model request several tool calls at once, which the harness can run in parallel.
Some can force the output to be valid JSON by restricting sampling to tokens a grammar allows, called constrained
decoding. The <Term id="mcp">Model Context Protocol (MCP)</Term>, [introduced by Anthropic in late
2024](https://www.anthropic.com/news/model-context-protocol), standardises how a harness discovers tools from outside
servers, so one integration works across many applications.

### How models learn to call tools

A base model has never seen your tool format, so tool use is trained in post-training
([Chapter 7](/building/post-training)). Supervised fine-tuning shows it thousands of conversations with correct tool
calls and results. [Toolformer (2023)](https://arxiv.org/abs/2302.04761) showed a model can label its own training
data by keeping only the tool calls that made its next-token predictions better.

Modern agents are also trained with reinforcement learning (RL): the model attempts real tasks in a sandbox, such as
fixing a bug until the tests pass, and gets rewarded for success. That is how models learn when to call a tool, how to
read its errors, and when to stop. Tool descriptions are prompts, so their wording changes behaviour a lot.

## The agent loop

**In short:** an agent is a model run in a loop. Each step it reads the whole context, writes a thought and an
action, and the harness appends the result. The loop ends when the model writes a final answer or a limit is hit.

A coding assistant asked to "fix the failing test" does not answer in one reply. It runs the tests, reads the error,
opens a file, edits it, runs the tests again, and repeats. Each step is one model call; each tool result becomes part
of the input to the next call.

That is an <Term id="agent">agent</Term>: a model, a set of tools, and a loop. The classic description is
observe, think, act. [ReAct (Yao et al., 2022)](https://arxiv.org/abs/2210.03629), short for reasoning plus acting,
made the pattern standard: the model writes a short thought ("the error says `x` is undefined; check imports"), then
an action, then reads the observation. Writing the thought first helps for the same reason <Term id="chain-of-thought">chain of thought</Term> helps: the model can use its own written words as scratch space.

<AgmAgentLoopDiagram />

In code, the whole loop is short:

```python
context = [system_prompt, tool_descriptions, user_task]
while True:
    reply = model.generate(context)          # one call; weights never change
    context.append(reply)
    if reply.tool_call is None:
        break                                # final answer
    result = harness.run(reply.tool_call)    # shell, search, API...
    context.append(result)                   # the only way the model "sees" it
```

### Context is the agent's only working memory

Look at what persists between steps: only `context`. The model's activations, the intermediate numbers it computes while reading, are recomputed on every
call (the KV cache stores them only to avoid recomputation, not to add anything). So whatever the agent has learned
about the task, such as which file holds the bug or which approach failed, exists only as text in the context.

This has a sharp consequence. If a tool result scrolls out of the window, or the harness drops it, the agent has forgotten
it completely. If the agent wrote down a wrong conclusion three steps ago, it will read that conclusion as fact on
every later step. There is no other memory to fall back on.

## Context engineering

**In short:** because the context is small, expensive and the only memory, choosing what goes in it at each step
is the main design problem for agents. Anthropic and others call this context engineering.

A long agent run fills the window fast. One test run can print 20,000 tokens of logs; one web page can be 50,000.
Models also get worse as the context grows (see "Long context vs retrieval" below). So more context is not free,
even when it fits.

The practice of curating those tokens is <Term id="context-engineering">context engineering</Term>.
[Anthropic's guide (2025)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) frames it as
finding the smallest set of high-signal tokens for each step. Common techniques:

- **Trim tool results:** return the 20 relevant log lines, not 20,000; clear old tool results the agent has used.
- **Load on demand:** give the agent search and file-reading tools instead of pasting everything up front.
- **Compact:** when the window fills, have the model summarise the history and continue from the summary.
- **Take notes:** have the agent write progress to a file it can reread later.
- **Use sub-agents:** a fresh agent with its own clean context does a sub-task and returns a short report.

Each technique trades completeness for focus. Every one of them also means someone, usually the model, decides
what to keep, and that decision can be wrong.

## Long tasks: errors compound

**In short:** if each step succeeds independently with probability p, a task of n steps succeeds with p^n. That
number falls fast, which is why long agent tasks are hard even for models that rarely make mistakes.

Say an agent gets each step right 99% of the time. That sounds reliable. But a 100-step task needs every step right:

```
P(all n steps succeed) = p^n
0.99^100 ≈ 0.37      0.95^20 ≈ 0.36      0.999^500 ≈ 0.61
```

Here p is the per-step success rate and n the number of steps. Half of all runs survive about ln(0.5)/ln(p) steps:
roughly 69 steps at p = 0.99, and about 7 at p = 0.9. The exercise below prints the full table.

The real picture is both worse and better than this formula. Worse, because errors are not independent: a wrong
conclusion written into the context poisons every later step. Better, because good agents recover. They run the tests,
see the failure, and fix it. A step that can be checked and retried has an effective p much closer to 1.

That is why agents do best where results are checkable: code with tests, math with answers. It is also why
[METR measures](https://arxiv.org/abs/2503.14499) agent progress as the length of task, in human working time, an agent
can finish half the time. They found that length roughly doubled every seven months from 2019 to early 2025.

## In-context learning: learning without changing weights

**In short:** models get better at a new task from examples in the prompt, with no weight update. It works
well, it is partly understood, and it vanishes when the context is cleared.

Put this in a prompt:

```
sea otter → loutre de mer
cheese → fromage
peppermint → menthe poivrée
plush giraffe →
```

A large model continues with "girafe en peluche". Nobody said "translate to French"; the examples defined the task. This
is <Term id="in-context-learning">in-context learning</Term> (ICL). Giving a few examples like this is <Term id="few-shot">few-shot</Term> prompting; giving none is zero-shot.

### Evidence that it works

The [GPT-3 paper (Brown et al., 2020)](https://arxiv.org/abs/2005.14165) made ICL famous. Its 175-billion-parameter
model improved on many tasks as examples were added to the prompt, and larger models gained more from them. Later,
[many-shot ICL (Agarwal et al., 2024)](https://arxiv.org/abs/2404.11018) used long contexts to fit hundreds or thousands
of examples, and performance kept rising on many tasks.

The cleanest evidence comes from toy settings. [Garg et al. (2022)](https://arxiv.org/abs/2208.01066) trained small
transformers on prompts made of (x, y) pairs from random linear functions. Given a new prompt, the model predicted y
for a new x about as well as the best classical method, least squares. It learned a learning algorithm that runs in its
forward pass.

But ICL is not always what it looks like. [Min et al. (2022)](https://arxiv.org/abs/2202.12837) found that giving
random labels in the examples barely hurt accuracy on classification tasks. The examples mostly told the model the
format and the set of possible answers; the skill was already in the weights. [Wei et al.
(2023)](https://arxiv.org/abs/2303.03846) then showed that the largest models can follow flipped labels, so they do
learn the new mapping. ICL is a mix of recognising a known task and learning from the examples, in proportions that
depend on the model and task.

### Why it works: two theories

Nobody fully knows. Two theories are popular, and both are partly supported.

**Implicit Bayesian inference.** [Xie et al. (2021)](https://arxiv.org/abs/2111.02080) argue that web text is a
mixture of many hidden "tasks" or topics. To predict the next token well, a model must infer which task the current
document belongs to. Examples in a prompt are evidence that narrows down the task, so ICL is the model locating a
task it already knows. This explains the random-label result well, and explains truly new tasks less well.

**Attention as implicit gradient descent (contested).** [von Oswald et al. (2022)](https://arxiv.org/abs/2212.07677)
showed that one layer of a simplified (linear) attention can compute exactly one step of gradient descent on the
examples in the prompt. So a transformer could be "training" a small model inside its activations. [Shen et al.
(2023)](https://arxiv.org/abs/2310.08540) found that real LLMs' in-context behaviour does not match gradient descent
closely. The construction is real for toy models; whether big models do this is disputed.

A third, mechanistic piece is solid: <Term id="induction-head">induction heads</Term>, attention patterns that find an
earlier copy of the current token and predict what followed it, form during training at the same time ICL ability
jumps ([Chapter 10](/steering/interp-alignment)).

### Limits of in-context learning

- **It does not persist.** Clear the context and the learning is gone. The next user starts from the same weights.
- **It is bounded by the window.** You cannot fit a year of experience into a prompt.
- **It is shallow for hard skills.** Examples rarely teach a model a skill it lacks; they mostly select and steer.
- **It cuts both ways.** [Many-shot jailbreaking (Anthropic, 2024)](https://www.anthropic.com/research/many-shot-jailbreaking)
  fills a long context with fake dialogues of an assistant misbehaving, and the model learns in context to misbehave.

::: warning Thread
In-context learning is the strongest counterexample to "a frozen model cannot learn". Within one context, the model
does change its behaviour from experience. Tell it it was tricked, and it can avoid the trick for the rest of the
conversation. The toy results show it can run a real learning algorithm in its activations, not only look things up.

But the learning lives in activations, the per-token timescale. When the context ends, it is gone, and the next
conversation can be tricked the same way. There is no mechanism to turn a lesson learned in context into a weight change,
which is the "nothing in between" gap. [Chapter 13](/beyond/beyond-frozen-weights) covers proposals to fill it, such as
test-time training and fast weights.
:::

## Retrieval-augmented generation

**In short:** search a document store for text related to the question, paste the best matches into the prompt,
and let the model answer from them. It is a search engine in front of the model, and it fails where search fails.

A company wants its assistant to answer questions about 50,000 internal wiki pages. They do not fit in any context,
and the model never saw them during training. Fine-tuning on them is slow, hard to update, and poor at teaching exact
facts. So at question time you search the wiki, take the five most relevant passages, and put them in the prompt.

This is <Term id="rag">retrieval-augmented generation</Term> (RAG), named by [Lewis et al.
(2020)](https://arxiv.org/abs/2005.11401). It is the most common way to give a model knowledge it was not trained on.

<AgmRagPipelineDiagram />

### Step 1: chunking

Whole documents are too long to paste and too coarse to match, so they are split into pieces of a few hundred tokens.
This is <Term id="chunking">chunking</Term>. Simple systems cut every N tokens with some overlap; better ones cut at
headings and paragraphs. Each chunk is stored with a link to its source so the answer can cite it.

### Step 2: embeddings

To search by meaning rather than by exact words, each chunk is turned into a vector, a list of numbers such as 1,024
floats. An <Term id="embedding-model">embedding model</Term> does this: a transformer trained so that texts with similar
meaning get vectors pointing in similar directions. "How do I reset my password?" and "steps to recover account
access" share no words but get nearby vectors.

Closeness is measured with <Term id="cosine-similarity">cosine similarity</Term>: the cosine of the angle between two
vectors, 1 for the same direction and 0 for unrelated. Embedding models are usually trained contrastively: pull matching
(question, passage) pairs together and push random pairs apart. The query is embedded with the same model at search time.

### Step 3: vector search

Now you need the chunks whose vectors are closest to the query vector. Comparing against all 10 million chunks is a
full scan: fine for thousands, too slow for billions. So vector databases use an index for <Term id="ann">approximate nearest neighbour</Term> (ANN) search, which returns most of the true nearest neighbours
much faster.

The most common index is <Term id="hnsw">HNSW</Term> (hierarchical navigable small world), a layered graph. Each vector links to a few close neighbours.
Search starts at a sparse top layer, walks greedily toward the query, then drops to denser layers and repeats. It is
like a skip list, but for "closest" instead of "next in order".

::: details Going deeper: HNSW and other ANN indexes
HNSW ([Malkov and Yashunin, 2016](https://arxiv.org/abs/1603.09320)) gives roughly logarithmic search time and high
recall (often above 95%). Its costs are memory, since the graph links sit alongside the vectors, and slow inserts.
Parameters trade speed for recall: `M` (links per node) and `ef_search` (candidates kept during the walk).
Alternatives include IVF, which clusters vectors and searches only the nearest clusters, and product quantization,
which compresses vectors to a few bytes each. FAISS, pgvector and most vector databases offer several of these.
:::

### Step 4: hybrid search and reranking

Embeddings miss exact strings: error codes, part numbers, rare names. Classic keyword search catches them. <Term id="bm25">BM25</Term>, the standard keyword-ranking formula, scores a chunk higher when it contains the query's
words, especially rare ones. <Term id="hybrid-search">Hybrid search</Term> runs both and merges the lists.

The fast search returns maybe 50 candidates. A <Term id="reranker">reranker</Term> then reads the query and each
candidate together and scores how well the candidate answers the query. It is slower per item but much more accurate,
because it sees both texts at once instead of comparing two separately made vectors. The top five go into the prompt.

### Where RAG fails

Most RAG failures are retrieval failures: the right text never reaches the model.

- **You must know what to look for.** The query has to resemble the answer. A question that does not hint at the cause
  ("why did the app fail?") retrieves text about the app, not the expired certificate that caused it.
- **Vocabulary mismatch.** Keyword search fails when words differ; embeddings help but miss jargon and exact IDs.
- **Chunk boundaries.** A fact split across two chunks, or a chunk that says "it" without naming the subject, is found
  by neither half of the query.
- **Multi-hop questions.** "Who manages the team that owns the billing service?" needs two lookups, the second
  depending on the first. One search cannot do it.
- **The model ignores or misreads the context.** Even with the right chunk, it may answer from its weights instead.

Agentic search addresses some of these: the model issues several queries, reads results, and refines its search, as a
person would. That helps with multi-hop questions, but it still needs the model to know what the next query should be.

### Try it: a tiny retrieval system and its failures

The exercise builds a retrieval system in about 40 lines of NumPy. It uses TF-IDF (term frequency–inverse document frequency) vectors instead of a learned
embedding model: each word gets a weight that is high when it is frequent in this document and rare in others. It then
ranks documents by cosine similarity, shows three failures, and prints the p^n table from earlier.

<<< @/../exercises/ch11_rag.py

Output:

```
Indexed 10 documents, vocabulary of 72 words.

=== 1. Works: the query shares words with the answer ===

Query: 'how do I rotate the TLS certificates?'
  query words the index knows: ['rotate', 'tls', 'certificates']
  0.71  To rotate TLS certificates, run certctl rotate on the gateway and restart the 
  0.00  The billing service writes invoices to the ledger database every night at 02:0
  0.00  Deploys go through the staging cluster first; promote with deployctl promote.

=== 2. Fails: same fact, no shared words (vocabulary mismatch) ===

Query: 'why was the website down last spring?'
  query words the index knows: []
  0.00  To rotate TLS certificates, run certctl rotate on the gateway and restart the 
  0.00  The billing service writes invoices to the ledger database every night at 02:0
  0.00  Deploys go through the staging cluster first; promote with deployctl promote.

=== 3. Fails: you did not know what to ask for ===

Query: 'why did the mobile app fail for everyone in March?'
  query words the index knows: ['mobile', 'app', 'fail', 'march']
  0.37  The mobile app talks to the gateway over HTTP/2 with retries and backoff.
  0.33  The March 4 incident: an expired certificate on the gateway made every request
  0.00  The billing service writes invoices to the ledger database every night at 02:0

=== 4. Chunk boundaries: a fact split across two chunks ===
  chunk 0: The ledger database runs on host db7. Its primary replica is in
  chunk 1: zone b. Credentials for it live in vault under secret/ledger/admin. Rotate them
  chunk 2: quarterly.

Query: 'where are the ledger database credentials?'
  query words the index knows: ['ledger', 'database', 'credentials']
  0.36  The ledger database runs on host db7. Its primary replica is in
  0.29  zone b. Credentials for it live in vault under secret/ledger/admin. Rotate the
  -> chunk 1 holds the vault path but only says 'it'; chunk 0 says 'ledger database'.

=== 5. Error compounding: P(all n steps succeed) = p^n ===
  p \ n        1      10      20      50     100     500
  0.9      0.900   0.349   0.122   0.005   0.000   0.000
  0.95     0.950   0.599   0.358   0.077   0.006   0.000
  0.99     0.990   0.904   0.818   0.605   0.366   0.007
  0.999    0.999   0.990   0.980   0.951   0.905   0.606
  p=0.9: 50% of runs survive about 7 steps
  p=0.95: 50% of runs survive about 14 steps
  p=0.99: 50% of runs survive about 69 steps
  p=0.999: 50% of runs survive about 693 steps
```

Case 2 scores zero everywhere: "website", "down" and "spring" appear in no document, though the March 4 incident is
the answer. A learned embedding model would likely fix this one, since it knows "website down" is close to "every
request fail". Case 3 is harder. The query is about the mobile app, so the app document wins, and a top-1 system would
miss the real cause. Better embeddings do not fix that: the question does not contain the clue. In case 4 the chunk with
the vault path loses to the chunk with the subject's name.

Things to try: add a synonym map to the tokenizer (`website → gateway`) and watch case 2 recover; change the chunk size
to 30 words in case 4; set `k=1` everywhere.

## Long context vs retrieval

**In short:** with million-token windows you can sometimes skip retrieval and paste everything. It costs far more per
query, models use long contexts unevenly, and the usable length is often much shorter than the advertised one.

If the whole wiki fit in the window, would you need RAG? Sometimes no: for one long contract or one codebase, pasting
it all is simpler and avoids every retrieval failure above. But three problems remain.

**Cost.** Every token in the context is processed on every request. The KV cache grows linearly with the length, and the
attention computation during prefill, the first pass over the prompt, grows with the square of the length. A
million-token prompt costs about 200 times a 5,000-token RAG prompt in API price, more in compute. <Term id="prefix-caching">Prefix caching</Term> ([Chapter 8](/running/inference)) helps when many queries share the same
long prefix, but only then.

**Uneven use.** [Liu et al. (2023)](https://arxiv.org/abs/2307.03172) put the answer to a question at different positions
in a long prompt. Models did best when it was at the start or end and worst in the middle. This is <Term id="lost-in-the-middle">lost in the middle</Term>. The effect has shrunk in newer models but not vanished; [Chroma's
2025 study](https://research.trychroma.com/context-rot) found all 18 models it tested got worse as input grew, even on
simple tasks, and called this "context rot".

**Effective vs advertised length.** A model that accepts a million tokens may only use a fraction well. The <Term id="effective-context">effective context length</Term> is the longest input at which a model still does a task
well. [RULER (2024)](https://arxiv.org/abs/2404.06654) found that only about half the models claiming 32,000 tokens held
up at that length. [NoLiMa (2025)](https://arxiv.org/abs/2502.05167) removed word overlap between question and answer,
so the model had to infer the link; at 32,000 tokens, 11 of 13 models dropped below half their short-context score.
Newer models do better, but "fits in the window" still does not mean "used well".

Note the parallel with RAG: NoLiMa shows long context also struggles most when the question does not share words with
the answer. Matching on surface overlap is a weakness of both.

::: info Back-of-envelope: 1M tokens of context vs RAG
Take a 70B-parameter model with 80 layers, 8 key-value heads of size 128 (grouped-query attention), and 2-byte numbers.

**KV cache per token** = 2 (keys and values) × 80 layers × 8 heads × 128 × 2 bytes ≈ 330 KB.
- 1M tokens: ≈ 330 GB of KV cache, about four 80 GB GPUs just for one user's cache.
- 5,000 tokens (RAG: five chunks plus the question): ≈ 1.6 GB.

**Prefill compute.** The weights cost about 2 × 70B ≈ 1.4 × 10¹¹ FLOPs per token, so 1.4 × 10¹⁷ for 1M tokens.
Attention adds about 2 × n² × d × layers for causal attention, with d = 64 query heads × 128 = 8,192:
2 × (10⁶)² × 8,192 × 80 ≈ 1.3 × 10¹⁸ FLOPs, ten times the weights' share. At 5,000 tokens attention is negligible and
the total is about 7 × 10¹⁴, roughly 2,000 times less.

**Price.** At an illustrative $3 per million input tokens: $3 per long-context query vs $0.015 for RAG, 200 times
more. Cached prefixes are often billed at about a tenth, which narrows the gap for repeated documents.

The embedding search itself is cheap: one small-model call for the query and a few milliseconds of index lookup.
:::

Practice in 2026 combines both: retrieve generously (tens of thousands of tokens rather than five chunks) into a long
window, and let the model do the final selection. Agents often skip vector search for code and use plain `grep` and file
reading as tools, retrieving iteratively.

## How agent memory works today

**In short:** every production memory system writes text somewhere and later puts some of it back into the context.
What differs is who writes it, when, and how it is selected. None of them change the weights.

"Memory" here means anything that carries information from one context to another: across sessions, or across a
compaction inside one long task. Here are the main designs in use, with what they store.

**Memory files.** The simplest design is a text file loaded into every session. Claude Code reads
[CLAUDE.md files](https://docs.anthropic.com/en/docs/claude-code/memory) of project and personal instructions at startup;
other coding agents use similar files such as AGENTS.md. Humans and the agent both edit them. This is a <Term id="memory-file">memory file</Term>: fully visible, easy to correct, and limited by what fits in the context.

**Saved facts and chat history.** [ChatGPT's memory](https://openai.com/index/memory-and-new-controls-for-chatgpt/)
(launched 2024) saves short facts about the user ("prefers Python", "has a daughter") and adds them to future prompts.
In April 2025 OpenAI extended it to reference past chats as well, by drawing relevant snippets from earlier conversations.
Claude's apps gained similar memory and past-chat search in 2025. Anthropic's API also offers a
[memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool): a directory of files the
model reads and writes through tool calls, with storage run by the developer.

**Hierarchical memory (MemGPT / Letta).** [MemGPT (Packer et al., 2023)](https://arxiv.org/abs/2310.08560) borrows the
OS idea of virtual memory. The context is "main memory"; an external database is "disk". The model itself calls
functions to page facts in and out, edit a core block of always-present notes, and search past messages. The project
continues as [Letta](https://www.letta.com/). The analogy is close, but the "page faults" are decided by the model, and
it must know a page exists to ask for it.

**Summarisation and compaction.** When an agent's context fills up, the harness asks the model to summarise the
history so far, then starts a fresh context from that summary. This is <Term id="compaction">compaction</Term>. Claude
Code and Anthropic's API do it automatically near the limit. It lets tasks run for hours, but it is lossy: details the
summary left out are gone.

**Episodic stores.** Some systems log every event (messages, actions, observations) and retrieve old ones by similarity
and recency, which is RAG over the agent's own past. [Generative Agents (Park et al., 2023)](https://arxiv.org/abs/2304.03442)
added periodic "reflection", where the model writes higher-level conclusions from recent memories and stores those too.

**Skills and tools the agent writes.** [Voyager (Wang et al., 2023)](https://arxiv.org/abs/2305.16291), a Minecraft
agent, saved working code for each new skill into a library and reused it later. Anthropic's [Agent
Skills (2025)](https://www.anthropic.com/news/skills) package instructions and scripts in folders that the agent loads
only when relevant. Here the memory is procedure, not fact, and it can be tested.

### What they have in common

All of these are text that ends up in the context. Something, usually the model, decided what was worth writing down,
wrote it in words, and something else decided when to bring it back. So memory stores conclusions, the parts someone
could put into words, and recall depends on a query matching them.

Three limits follow:

- **Lossy writing.** A note says "the user prefers short answers", not the hundred small signals that led there.
- **Recall needs a cue.** A stored lesson only helps if retrieval surfaces it at the right moment, which needs a query
  resembling it.
- **No skill change.** Reading "avoid this trick" in a note is weaker than having learned it; the model must apply it
  correctly every time, from scratch.

Memory also brings new risks: wrong notes persist and compound, and anything an agent reads (a web page, an email) can
plant instructions into its memory. This is prompt injection, now with a long lifetime.

::: warning Thread
This chapter supports two of the thread's intuitions directly. Every memory system above stores conclusions in words,
and the exercise shows retrieval failing when the query lacks the clue. A human who was tricked once does not look up
"have I been tricked like this before?"; the experience changed how they see the situation.

The strongest counterarguments are real. Humans also rely on external notes, checklists and documentation, and that
works. Million-token contexts can hold far more than any note, and pasting a whole history avoids "knowing what to look
for". Agents that write and test their own skills store procedures, not just facts, which is closer to know-how. And
in-context learning shows that reading the right text can change behaviour substantially.

What none of them do is the slow middle step: turning many experiences into changed weights, as human practice and
sleep do. Whether that step is needed, or whether better notes plus longer contexts suffice, is the open question.
[Chapter 12](/beyond/continual-learning) covers why updating weights after deployment is hard, and
[Chapter 13](/beyond/beyond-frozen-weights) covers attempts to make it work.
:::

::: tip Interview checkpoint
<details><summary>Design memory for a coding agent that works on one repo for months.</summary>

Layers, by lifetime. A short project file (conventions, build commands, gotchas) loaded every session and editable by
humans. Within a task, compaction plus a progress-notes file so long tasks survive summarisation. Across tasks, an
episodic log of past tasks and outcomes, retrieved by search, and a library of tested scripts the agent wrote. Use the
code itself as ground truth through grep and file reads rather than embedding it. Add review and expiry for notes, since
wrong memories compound, and treat text from untrusted sources as data, not instructions.
</details>

<details><summary>RAG vs fine-tuning vs long context: when do you use which?</summary>

RAG for large or changing knowledge where you need citations and access control; it fails when retrieval misses.
Long context when the material fits and is used per query (one contract, one codebase); simplest, but costly per query
and weaker in the middle. Fine-tuning for behaviour, format, style or a narrow skill, not for injecting facts that
change; it is slow to update and can hurt other skills. They combine: fine-tune the format, retrieve the facts.
</details>

<details><summary>How does tool calling actually work?</summary>

Tool schemas are rendered into the prompt. The model, trained by supervised fine-tuning and RL to do so, emits a structured call in special
tokens and stops. The harness parses it, runs the tool, appends the result as a new message, and calls the model
again. The model never executes anything; it only writes text.
</details>

<details><summary>Why do agents fail on long tasks even with a strong model?</summary>

Errors compound: with per-step success p, n steps succeed with p^n, so 99% per step gives about 37% over 100 steps.
Errors are also correlated, since a wrong conclusion in the context misleads later steps, and long contexts degrade
the model. Mitigations: checkable steps with retries, smaller contexts via sub-agents and compaction, and RL training
on long tasks.
</details>

<details><summary>Is in-context learning real learning?</summary>

Partly. Toy transformers learn to do least-squares regression in context, and large models can follow flipped labels.
But much ICL is task recognition: random labels barely hurt on many tasks. Theories include implicit Bayesian
inference and attention implementing gradient descent (contested for real LLMs). Either way, it disappears when the
context ends.
</details>
:::

## In practice

- **Most "AI features" are RAG plus a few tools.** Support bots, internal search and document Q&A are usually
  retrieval, a prompt, and a model, with retrieval quality the main lever.
- **Coding agents lean on grep, not vectors.** Code has exact names, so tools like Claude Code mainly search with plain
  text tools and read files, retrieving step by step.
- **Evaluate retrieval separately.** Measure whether the right chunk is in the top k (recall@k) before blaming the model.
- **Tool descriptions are prompts.** Clear names, examples and error messages change agent success rates a lot.
- **Long tasks need checkpoints.** Production agents write progress files, run tests often, and restart from notes.

## Common misconceptions

- **"The model runs the code."** The model writes text; the harness runs it.
- **"Memory means the model learns about me."** Memory is text added to the prompt; the weights are shared and unchanged.
- **"A 1M-token window means 1M tokens of reliable reading."** Effective context is often much shorter.
- **"Embeddings understand everything."** They miss exact identifiers and cannot retrieve what the query does not hint at.
- **"Few-shot examples teach the model the task."** Often they only show the format; the skill was already there.

## Key takeaways

- A deployed model knows only its frozen weights and its context; agents, RAG and memory all choose the context.
- Tool use is text in, text out: the model writes a call, the harness runs it and appends the result.
- Per-step reliability compounds as p^n, so long tasks need checkable steps and recovery.
- RAG fails mainly at retrieval: vocabulary, chunking, and not knowing what to ask; long context trades that for cost
  and uneven attention.
- Every memory system today stores conclusions as text and needs a cue to recall them; none update the weights.

## Further reading

- [Yao et al., *ReAct* (2022)](https://arxiv.org/abs/2210.03629): the thought-action-observation loop behind most agents.
- [Anthropic, *Effective context engineering for AI agents* (2025)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents):
  practical guide to compaction, notes and sub-agents.
- [Lewis et al., *Retrieval-Augmented Generation* (2020)](https://arxiv.org/abs/2005.11401): the original RAG paper.
- [Liu et al., *Lost in the Middle* (2023)](https://arxiv.org/abs/2307.03172) and
  [NoLiMa (2025)](https://arxiv.org/abs/2502.05167): how models really use long contexts.
- [Packer et al., *MemGPT* (2023)](https://arxiv.org/abs/2310.08560): memory as virtual memory; the OS analogy made concrete.
- [Garg et al. (2022)](https://arxiv.org/abs/2208.01066) and [von Oswald et al. (2022)](https://arxiv.org/abs/2212.07677):
  in-context learning as a learned algorithm, in toy settings.
- [Kwa et al., *Measuring AI ability to complete long tasks* (METR, 2025)](https://arxiv.org/abs/2503.14499): the task-horizon trend.

## Review

<Flashcards id="agents-memory" :cards="cards" />

<MarkDone id="agents-memory" />
