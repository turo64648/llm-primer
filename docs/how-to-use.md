# How to Use This Primer

This primer explains how large language models (LLMs) work, from a single neuron to a serving cluster. It
assumes you can program and know how computers work, and that you know no machine learning. It is meant to be
studied over a few weeks, not skimmed.

## What's in each chapter

- **Core explanation.** Intuition first, then the mechanism, then the math. The math is explained in words.
- **Exercises.** Short Python programs (NumPy or PyTorch) that run on a laptop CPU. Chapters 1–4 build up to a
  tiny working GPT. All code is in the `exercises/` folder of the repository.
- **Back-of-envelope boxes.** How to estimate memory, compute and cost from a model's size. Interviewers at AI
  labs ask for these often.
- **Interview checkpoint.** Questions an interviewer might ask, with model answers. Answer out loud before
  opening them.
- **Thread callouts.** Short asides on one open question: can a model whose weights are frozen after training
  ever think better than a person? They include evidence on both sides. The [Thread page](/extras/thread)
  collects them.
- **Going deeper boxes.** Collapsed by default. The main text makes sense without them.
- **Tap-to-see terms.** Words with a dotted underline show a definition when tapped. All are on the
  [Glossary](/glossary) page.
- **Flashcards.** For review a few days later. They ask *why* and *how*, not trivia.

## Running the exercises

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu   # from Chapter 2 on
python exercises/ch01_mlp.py
```

## A suggested routine

1. Read a chapter in one sitting and run its exercises. Change a number and predict what happens first.
2. The next day, go through the flashcards without re-reading.
3. Answer the interview checkpoint questions **out loud**, then compare.
4. A week later, redo the flashcards. Mark the chapter as done once they feel easy.

Your progress (chapters done, known flashcards) is saved in this browser only.

## How AI lab interviewers probe

Research engineer and infrastructure interviews mix three kinds of question:

> "What does attention compute?" → "Why is decoding memory-bound?" → "Estimate the KV cache for a 70B model
> serving 32 users at 8k context. What would you change to fit twice as many?"

So for every concept, aim to explain **what it is**, **what it costs** (in memory, FLOPs and time), and
**what breaks** when you scale it.
