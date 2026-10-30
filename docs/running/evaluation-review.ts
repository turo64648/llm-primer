// Flashcards for the Evaluation chapter. Strings may contain inline HTML.

export const cards = [
  {
    q: 'Why is a benchmark score an estimate rather than a fact about the model?',
    a: 'The questions are a sample from a larger pool of similar questions, and sampling adds noise. The standard error of an accuracy p on n questions is √(p(1−p)/n), about 3 points for 70% on 200 questions.',
  },
  {
    q: 'Why is comparing two models harder than measuring one?',
    a: 'Both scores wobble, so the error of the gap is larger: about √2 times one model’s error if independent. A paired comparison on the same questions helps, because questions both models get right or wrong add no noise.',
  },
  {
    q: 'How does the number of questions needed scale with the gap you want to detect?',
    a: 'As one over the gap squared. Halving the gap needs four times the questions; a 2-point gap near 70% needs thousands of questions, not hundreds.',
  },
  {
    q: 'How does the bootstrap estimate uncertainty?',
    a: 'Resample the questions with replacement many times, recompute the score (or the paired gap) on each resample, and take the middle 95% of results as the interval. It needs no formula for the metric.',
  },
  {
    q: 'Why does temperature 0 not make evals fully repeatable on a server?',
    a: 'Requests are batched with others, and batch size changes the order of floating-point additions. Floating-point addition is not associative, so tiny differences can flip the top token.',
  },
  {
    q: 'How does the unbiased pass@k estimator work?',
    a: 'Draw n ≥ k samples, count c correct, and compute 1 − C(n−c, k)/C(n, k): the chance a random k of them include a pass. It uses all n samples, so it is less noisy than drawing exactly k.',
  },
  {
    q: 'Why is 1 − (1 − c/n)^k biased?',
    a: 'The function curves downward (it is concave), so applying it to a noisy estimate of p gives a lower value on average than applying it to the true p. The bias grows with k.',
  },
  {
    q: 'How is multiple choice scored by log-likelihood, and what is the catch?',
    a: 'Compute the model’s probability of each option as a continuation of the question and pick the highest. It is fast and works on base models, but length normalisation and format choices change scores by many points, and it measures preference rather than what the model would say.',
  },
  {
    q: 'What are the main biases of LLM-as-judge?',
    a: 'Position (prefers first or second answer), length (prefers longer answers), self-preference (prefers its own style or family), and poor grading of problems it cannot solve itself.',
  },
  {
    q: 'How does an arena turn human votes into a leaderboard?',
    a: 'It fits a Bradley–Terry model: each model has a strength, and the chance A beats B depends only on the difference. On the Elo scale, 100 points is about a 64% win rate.',
  },
  {
    q: 'How does contamination happen even if nobody trains on the test on purpose?',
    a: 'Benchmarks are quoted across the web, synthetic data can copy memorised items, and code fixes live in public repository history. Repeated tuning against a test set also leaks information into the model.',
  },
  {
    q: 'What is a canary string, and what are its limits?',
    a: 'A unique random string in every benchmark file, so pipelines can filter those files and anyone can test whether a model learned them. It fails if pipelines ignore it or the questions are copied without it.',
  },
  {
    q: 'What makes a benchmark resistant to contamination?',
    a: 'Keeping questions private and graded by a third party, or adding fresh dated questions and scoring each model only on ones published after its training data was collected.',
  },
  {
    q: 'How does Goodhart’s law show up in model development?',
    a: 'Teams tune on the same tests, train on benchmark-shaped data, report the best of many variants, and optimise for what judges or voters like. The score rises faster than the ability it was meant to track.',
  },
  {
    q: 'Why do models improve first on verifiable tasks like math and code?',
    a: 'Reinforcement learning needs an automatic reward for every attempt. Answer checkers and unit tests give one cheaply; open-ended tasks need reward models or judges, which are noisier and can be gamed.',
  },
  {
    q: 'Where does the "evals are the test suite of ML" analogy break?',
    a: 'Tests are deterministic and check a written spec; evals are noisy samples of a fuzzy ability. A small drop may be noise, and a model can absorb the test set from its training data.',
  },
]
