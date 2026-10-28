import type { GlossaryEntry } from '../glossary'

const ch = '/start/math-toolkit'

export const terms: Record<string, GlossaryEntry> = {
  vector: { term: 'Vector', def: 'A list of numbers, such as [1, 2, 3]. Its length is its dimension. Models represent tokens and their meanings as vectors with thousands of numbers.', chapter: ch },
  'dot-product': { term: 'Dot product', def: 'Multiply two equal-length vectors position by position and add up the results. Large and positive when they point the same way, near zero when unrelated.', chapter: ch },
  'cosine-similarity': { term: 'Cosine similarity', def: 'The dot product of two vectors divided by both their lengths. It compares direction only and is always between −1 and 1.', chapter: ch },
  matrix: { term: 'Matrix', def: 'A grid of numbers with m rows and k columns (m × k). It can be read as a stack of m vectors.', chapter: ch },
  flop: { term: 'FLOP', def: 'One floating-point operation, such as a multiply or an add. Multiplying an m × k matrix by a k × n matrix costs about 2 · m · n · k FLOPs.', chapter: ch },
  derivative: { term: 'Derivative', def: 'How much a function’s output changes per unit of change in its input, for a tiny nudge: a sensitivity. A partial derivative does this for one input, holding the others fixed.', chapter: ch },
  gradient: { term: 'Gradient', def: 'The vector of partial derivatives of one output (usually the loss) with respect to every input (usually every weight). Training steps against it to lower the loss.', chapter: ch },
  'chain-rule': { term: 'Chain rule', def: 'When functions are chained, the sensitivity of the end to the start is the product of the sensitivities of each step. Backpropagation applies it mechanically.', chapter: ch },
  logit: { term: 'Logit', def: 'A raw score a model gives to one option, such as one possible next token. Any real number; softmax turns a vector of logits into probabilities.', chapter: ch },
  softmax: { term: 'Softmax', def: 'Turns a vector of scores into probabilities: exp of each score, divided by the sum. Implementations subtract the largest score first to avoid overflow.', chapter: ch },
  'probability-distribution': { term: 'Probability distribution', def: 'A number between 0 and 1 for each possible outcome, adding up to 1. A language model outputs one over its vocabulary at each position.', chapter: ch },
  'log-probability': { term: 'Log-probability', def: 'The natural log of a probability: 0 for certain, more negative for less likely. Logs turn products of many probabilities into sums that do not underflow.', chapter: ch },
  'cross-entropy': { term: 'Cross-entropy', def: 'The standard training loss for language models: −log of the probability the model gave to the correct next token, averaged over tokens. It measures surprise.', chapter: ch },
  broadcasting: { term: 'Broadcasting', def: 'NumPy and PyTorch’s rule for combining arrays of different shapes: line shapes up from the right and stretch size-1 dimensions to match.', chapter: ch },
}
