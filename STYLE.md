# Writing Style Guide

Every chapter follows this guide. The reader is an experienced engineer who is rusty on OS concepts,
often reading on a phone. The goal is understanding they can explain out loud in an interview, not
memorisation.

## The three rules that matter most

1. **Never use a term before you explain it.** Explain it in plain words where it first appears in the
   chapter, even if an earlier chapter already did. If the chapter does not need the term, leave it out.
2. **Show the example before the name.** Describe the concrete situation first, then give the formal term.
3. **Keep two layers.** The main text must make complete sense on its own, in plain language. Hardware
   names, bit layouts, register names, kernel internals and exact numbers go in "Going deeper" boxes.

## Sentences and words

- One idea per sentence. Aim for 20 words or fewer; never more than 30.
- Paragraphs of at most 4 sentences.
- Active voice: "the CPU checks the table", not "the table is checked".
- Plain words: "before" (not "prior to"), "use" (not "utilise"), "make sure" (not "ensure").
- One word, one meaning. Pick one term for a thing and use only that term in the chapter. For example,
  do not switch between "page table entry", "PTE" and "entry" without saying they are the same.
- Spell out acronyms. Use an acronym only if interviewers use it (TLB, MMU, OOM), and introduce it once:
  "the translation lookaside buffer (TLB)".
- Do not rely on the reader's memory of something far above. Restate it in a few words where needed.
- No filler: no "it's worth noting", "interestingly", "simply", "just", "obviously".

## Chapter structure

Each chapter uses these parts, in this order:

1. **Title and a two-sentence intro**: what this is and why interviewers care.
2. **"Before you start" box** (`::: info Before you start`): 3–4 plain sentences on what the reader needs
   to know, with links to the chapters that explain it. The chapter must still make sense without them.
3. **Main sections.** Each `##` section starts with a one-line summary in bold, beginning with
   **In short:**. Then the explanation.
4. **"Going deeper" boxes** (`::: details Going deeper: <topic>`) inside sections, for precise details.
5. **Diagrams and widgets** where seeing it helps. Label them in plain words; x86 or Linux names may appear
   in brackets.
6. **Code** in C, short, compilable, with a comment on the first line showing how to build and run it.
   Show the expected output.
7. **"Why this matters in real systems"**: short, concrete stories (databases, servers, runtimes, ML).
8. **Interview questions**: each in a `::: details` block. Model answers use plain language first, then a
   "Senior add-on" line with the precise, deeper point.
9. **Common misconceptions**: short bullet list.
10. **Key takeaways**: at most 5 bullets.
11. **Review**: `<Flashcards>` then `<MarkDone>`. There is no multiple-choice quiz.

## Terms and the glossary

- The glossary lives in `docs/.vitepress/glossary.ts`. Every term that appears in more than one chapter
  belongs there, with a one or two sentence plain definition.
- Wrap a glossary term with `<Term id="tlb">TLB</Term>` **the first time** it appears in a chapter (only
  the first time). The reader can tap it to see the definition.
- Add new terms to the glossary when you write a chapter.

## Flashcards

- Ask **why** and **how**, not trivia. Good: "Why do page tables need to be a tree?" Bad: "How many bits
  index each page table level?"
- Answers are 1–3 plain sentences.
- 12–20 cards per chapter.

## Lists and formatting

- At most 5–6 bullets in a list. Split longer lists into groups.
- Use tables only for comparisons across the same attributes.
- Bold sparingly: key terms at the point they are explained, and the "In short" lines.
- Numbers: give an order of magnitude with units ("about 100 microseconds on an SSD"). Put exact,
  hardware-specific numbers in "Going deeper".

## Example

Too dense:

> The CR3 register holds the physical address of the top-level table, the PML4.

Following this guide:

> The CPU needs to know where the current process's page table is. It keeps that location in a register,
> a small storage slot inside the CPU. When the OS switches to another process, it loads that process's
> table location into the register.
>
> `::: details Going deeper: the x86 names` — on x86-64 this register is called CR3, and the top-level
> table is called the PML4.

## LLM Primer additions

These override the rules above where they conflict.

**The reader.** A software engineer with about 2 years in infrastructure (data pipelines, system internals,
some JVM). Strong at programming and systems. Knows no ML beyond a vague idea of gradient descent. Explain ML
from first principles. Do not explain programming or systems concepts (caches, RPC, threads, memory
bandwidth). Use a systems analogy only where it genuinely fits, and say where it breaks.

**Order inside a section.** Intuition first, then the mechanism, then the math. Explain every equation in
words: what each symbol is, and what the equation says.

**Math notation.** No LaTeX: the site has no math renderer. Write formulas in code blocks or inline code, using Unicode where it helps (Σ, √, ·, ×, ≈).

**Code.** Python 3 with NumPy or PyTorch, CPU-only, runs in under about 2 minutes on a 2-core laptop. Put
each exercise in `exercises/chNN_<name>.py`, with a comment on the first line showing how to run it, and a
fixed random seed. Include it in the chapter with `<<< @/../exercises/chNN_<name>.py` (do not paste a
copy). Show the expected output, and run the script to get it. Never show output you did not get.

**Callouts.**
- `::: tip Interview checkpoint` — 3–5 questions an AI lab interviewer might ask. Each question is a
  `<details><summary>` inside the box, with a concise model answer.
- `::: info Back-of-envelope` — a worked estimate (params → bytes, tokens → FLOPs, KV cache size). Show the
  arithmetic and round sensibly.
- `::: warning Thread` — 1–3 short paragraphs connecting the material to the running thread (below),
  including evidence against it. Only where it genuinely connects; most chapters have 0–2.
- Mark contested or speculative claims in the text: "(contested)", "(speculative)", or a sentence saying
  who disagrees.

**Chapter structure.** Same as above, with these changes: the interview questions section is replaced by
the Interview checkpoint box(es); "Why this matters in real systems" becomes **"In practice"**; add
**"Further reading"** before Review, with key papers and posts as links and one line each on why to read it.

**Accuracy and sources.** Web-search to verify anything that changes fast (current techniques, recent
architectures, frontier research, model sizes, prices). Cite with inline links. Do not invent numbers,
quotes, papers or authors. If you cannot verify something, leave it out or mark it as uncertain.

**The running thread.** The reader is exploring the claim that frozen, checkpointed LLMs cannot reach AGI
(loosely: "thinks better than me", not "knows more"), and that updating weights during deployment may be
necessary but not sufficient. Their intuitions:
- A frozen model can be tricked the same way forever; a human learns from being tricked once.
- Agent memory stores conclusions, not intuitions; retrieval needs you to know what to look for.
- LLMs have per-token activations and months-frozen weights, but nothing in between. Brains have the whole
  timescale spectrum, plus consolidation (sleep).
- Hard parts: a training signal at inference time (especially where answers cannot be checked),
  catastrophic forgetting, per-user vs shared models, safety of models that keep changing.
Thread callouts test these intuitions honestly, including the strongest counterarguments (in-context
learning, long context, retrieval, frequent retraining).
