import type { GlossaryEntry } from '../glossary'

const chapter = '/running/evaluation'

export const terms: Record<string, GlossaryEntry> = {
  benchmark: {
    term: 'Benchmark',
    def: 'A fixed set of test questions or tasks with a scoring rule, used to compare models. Its score is a sample estimate, not the model’s true ability.',
    chapter,
  },
  'standard-error': {
    term: 'Standard error',
    def: 'How much a measured score would wobble if you re-ran the measurement on a fresh sample. For an accuracy p on n questions it is √(p(1−p)/n).',
    chapter,
  },
  'confidence-interval': {
    term: 'Confidence interval',
    def: 'A range around a measured score that would contain the true value in, say, 95% of repeated measurements. Roughly the score ± 2 standard errors.',
    chapter,
  },
  bootstrap: {
    term: 'Bootstrap',
    def: 'Estimating the uncertainty of a score by resampling the test questions with replacement many times and recomputing the score each time. The spread of those scores is the uncertainty.',
    chapter,
  },
  'exact-match': {
    term: 'Exact match',
    def: 'Scoring an answer as correct only if it equals the reference answer after light normalisation (case, spaces, punctuation). Simple and cheap, but brittle to formatting.',
    chapter,
  },
  'pass-at-k': {
    term: 'pass@k',
    def: 'For code or math: the chance that at least one of k sampled attempts is correct. Estimated without bias by drawing n ≥ k samples and computing 1 − C(n−c, k) / C(n, k), where c are correct.',
    chapter,
  },
  'llm-as-judge': {
    term: 'LLM-as-judge',
    def: 'Using a strong language model to grade or compare other models’ answers. Cheap and scalable, but biased toward position, length and its own style.',
    chapter,
  },
  contamination: {
    term: 'Contamination',
    def: 'When benchmark questions or answers end up in a model’s training data, so a high score may reflect memorisation instead of ability.',
    chapter,
  },
  'canary-string': {
    term: 'Canary string',
    def: 'A unique random string embedded in a benchmark’s files, so data pipelines can filter them out and anyone can test whether a model has seen them.',
    chapter,
  },
  'goodharts-law': {
    term: 'Goodhart’s law',
    def: '“When a measure becomes a target, it ceases to be a good measure.” In ML: optimising hard for a benchmark score improves the score faster than the ability it was meant to track.',
    chapter,
  },
  saturation: {
    term: 'Saturation',
    def: 'When top models all score near a benchmark’s ceiling (or near the level of its label noise), so it can no longer tell them apart.',
    chapter,
  },
  'eval-harness': {
    term: 'Eval harness',
    def: 'Software that runs benchmarks against a model in a fixed, reproducible way: prompts, sampling settings, answer extraction, scoring and logging. Examples: lm-evaluation-harness, Inspect.',
    chapter,
  },
}
