// Flashcards for the Frozen-Weights Thread synthesis page. Strings may contain inline HTML.

export const cards = [
  {
    q: 'Why is the thread\'s claim about learning curves rather than day-one capability?',
    a: 'Nothing stops a frozen model from reasoning better than a person at release; chess engines are frozen and superhuman. The defensible claim is that a frozen model cannot keep improving from its own experience, as a person does over weeks in a new job.',
  },
  {
    q: 'What does "necessary but not sufficient" mean for weight updates during deployment?',
    a: 'Necessary: some lasting parameter change between one conversation and one release is needed. Not sufficient: you also need a trustworthy training signal, protection from forgetting, a per-user or shared design, and safety checks that keep up.',
  },
  {
    q: 'What kind of experiment would test the thread\'s claim directly?',
    a: 'A learning curve over many episodes of a new task, spread across sessions, comparing a frozen model with memory against a weight-updating one at equal compute and data. A single benchmark score cannot test a claim about improvement.',
  },
  {
    q: 'Why is in-context learning the strongest counter to "tricked the same way forever"?',
    a: 'Within one conversation the model does learn from being told it was tricked, and can run a real learning algorithm in its activations. The learning vanishes when the context ends, which is where the intuition still holds.',
  },
  {
    q: 'Why don\'t weight updates automatically give "learn from being tricked once"?',
    a: 'Gradient descent moves weights a tiny amount per step and usually needs a fact many times. A big step on one example damages old knowledge. Today the only one-shot learning LLMs have lives in context.',
  },
  {
    q: 'How do long contexts weaken the "retrieval needs you to know what to look for" intuition, and why only partly?',
    a: 'Pasting a whole history avoids choosing a query. But models use long contexts unevenly: they miss facts in the middle and when question and answer share no words.',
  },
  {
    q: 'Do linear attention and Titans refute "nothing in between activations and weights"?',
    a: 'They refute the structural version: they keep fast weights that change during use. But the state resets between sequences, the write rules are frozen, and the signal is surprise, so they add memory rather than better judgement.',
  },
  {
    q: 'What is the strongest argument that frequent retraining is "continual enough"?',
    a: 'Labs already fold deployment feedback into post-training and patch prompts and classifiers in days. If what matters is the system\'s error rate over a year, that may suffice. It still gives no improvement within a task or for one user.',
  },
  {
    q: 'Which timescale holds only text in today\'s LLM systems, and what proposals target it?',
    a: 'Hours to weeks: today it is filled only by notes and retrieval. Proposals include per-user LoRA adapters, Doc-to-LoRA, SEAL and "sleep"-style distillation from context into weights, all at research scale.',
  },
  {
    q: 'Why is the training signal the hardest of the four hard parts?',
    a: 'In unverifiable domains nobody tells the model it was wrong. User feedback and judges give some signal, but optimising against them continuously invites Goodhart effects, as the 2025 GPT-4o sycophancy rollback showed.',
  },
  {
    q: 'Why does safety cut against the "necessary" half of the claim?',
    a: 'Today\'s slow learning passes through a human release gate, which is what makes it safe to check. Faster learning hands every user a share of the training data and drifts from what was tested, so safety argues for small, gated updates.',
  },
  {
    q: 'Which result would most weaken the thread\'s claim?',
    a: 'A frozen model with a memory scaffold showing a rising learning curve on a new, long task, matching a weight-updating baseline at equal compute, with lessons transferring to paraphrased mistakes in later sessions.',
  },
  {
    q: 'How should you present this claim to a lab interviewer?',
    a: 'As a hypothesis about learning from experience, with the strongest counters named by you, and a concrete measurement that would settle part of it. Avoid "impossible", undefined "AGI", and brain analogies offered as proof.',
  },
]
