import type { GlossaryEntry } from '../glossary'

const chapter = '/building/pretraining'

export const terms: Record<string, GlossaryEntry> = {
  pretraining: {
    term: 'Pretraining',
    def: 'The first and most expensive training phase: next-token prediction over trillions of tokens of mostly web text. It produces a base model that continues text but does not yet follow instructions.',
    chapter,
  },
  'next-token-prediction': {
    term: 'Next-token prediction',
    def: 'The pretraining objective. The model outputs a probability for every possible next token and is penalised by −log of the probability it gave the token that really came next.',
    chapter,
  },
  'common-crawl': {
    term: 'Common Crawl',
    def: 'A non-profit that crawls the web and publishes free snapshots of billions of pages. The raw material for most open pretraining datasets.',
    chapter,
  },
  deduplication: {
    term: 'Deduplication',
    def: 'Removing repeated documents or passages from training data, both exact copies and near-copies. It saves compute and reduces word-for-word memorisation.',
    chapter,
  },
  minhash: {
    term: 'MinHash',
    def: 'A hashing scheme that gives each document a short signature, so that two signatures agree about as often as the documents share n-grams. Used with bucketing to find near-duplicates without comparing every pair.',
    chapter,
  },
  'quality-classifier': {
    term: 'Quality classifier',
    def: 'A small, fast model that scores documents by how much they look like chosen examples of good text. Pipelines keep the top-scoring slice.',
    chapter,
  },
  'data-mixture': {
    term: 'Data mixture',
    def: 'The share of training tokens drawn from each source (web, code, math, books, other languages). Often changed during a run, ending on the highest-quality data.',
    chapter,
  },
  'scaling-law': {
    term: 'Scaling law',
    def: 'An empirical power law relating a model’s loss to its parameters, training tokens or compute. Lets labs predict a large run from cheap small ones.',
    chapter,
  },
  'compute-optimal': {
    term: 'Compute-optimal (Chinchilla-optimal)',
    def: 'The model size and token count that give the lowest loss for a fixed training budget. For Chinchilla-style models, about 20 tokens per parameter.',
    chapter,
  },
  overtraining: {
    term: 'Overtraining',
    def: 'Training a model on far more tokens than compute-optimal, to get a smaller model that is cheaper to serve. Not the same as overfitting.',
    chapter,
  },
  'learning-rate-schedule': {
    term: 'Learning-rate schedule',
    def: 'How the step size changes during training: usually a short warmup from near zero, then a slow decay (cosine, or constant then a quick drop).',
    chapter,
  },
  'loss-spike': {
    term: 'Loss spike',
    def: 'A sudden jump in training loss. Some recover; others diverge, and the team rolls back to an earlier checkpoint and skips the data around it.',
    chapter,
  },
}
