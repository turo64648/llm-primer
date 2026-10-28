// Flashcards for the Math You Need chapter. Strings may contain inline HTML.

export const cards = [
  {
    q: 'Why is the dot product a measure of similarity?',
    a: 'It multiplies two vectors position by position and sums. Where both have the same sign the terms add up, so it is large when they point the same way, near zero when unrelated, and negative when opposite.',
  },
  {
    q: 'Why use cosine similarity instead of the raw dot product?',
    a: 'The dot product also grows with the vectors’ lengths. Dividing by both lengths leaves only the direction, so long vectors do not look more similar just because they are long.',
  },
  {
    q: 'How is a matrix multiply made of dot products?',
    a: 'Each cell (i, j) of the result is the dot product of row i of the first matrix with column j of the second. An m × k by k × n multiply computes m · n of them.',
  },
  {
    q: 'Why does an m × k by k × n matrix multiply cost about 2 · m · n · k FLOPs?',
    a: 'There are m · n output cells, each a dot product of length k. Each of the k steps is one multiply and one add, which count as 2 FLOPs.',
  },
  {
    q: 'Why does running a model cost about 2 · N FLOPs per token, for N weights?',
    a: 'Almost all the work is matrix multiplies, and each weight is used in one multiply and one add per token.',
  },
  {
    q: 'What does a derivative tell you, in plain words?',
    a: 'Its sensitivity: if you nudge the input a tiny bit, the output moves about this many times as much. The sign says which direction.',
  },
  {
    q: 'What is the gradient of the loss, and how does training use it?',
    a: 'It is one partial derivative per weight: how the loss reacts to nudging that weight. Training takes small steps against it, which lowers the loss.',
  },
  {
    q: 'What does the chain rule say, and why does backpropagation need it?',
    a: 'Along a chain of functions, sensitivities multiply (and add where paths split and rejoin). A network is a chain ending in the loss, so the gradient of each weight is a product of step sensitivities, computed backward.',
  },
  {
    q: 'What does softmax do, and why does it use exp?',
    a: 'It turns any vector of scores (logits) into probabilities that are positive and sum to 1. exp makes every score positive and keeps the order, so higher scores get more probability.',
  },
  {
    q: 'Why do softmax implementations subtract the maximum logit first?',
    a: 'exp of a large logit overflows to infinity, giving NaN. Adding a constant to all logits does not change softmax, so subtracting the max gives the same answer with no overflow.',
  },
  {
    q: 'Why work with log-probabilities instead of probabilities?',
    a: 'A sequence’s probability is a product of many small numbers, which underflows to 0. Logs turn the product into a sum of moderate negative numbers.',
  },
  {
    q: 'Why is cross-entropy described as “surprise”?',
    a: 'For each token it is −log of the probability the model gave to the correct answer: 0 if the model was certain and right, growing without limit as that probability falls toward 0.',
  },
  {
    q: 'What loss does a model that knows nothing get, and why is that useful?',
    a: 'Spreading probability evenly over V tokens gives log(V) per token, about 10.8 for 50,000. It is the baseline: a loss near it at the start is expected, far above it suggests a bug.',
  },
  {
    q: 'How does NumPy broadcasting decide if two shapes fit, and how can it hide bugs?',
    a: 'It lines the shapes up from the right; each pair of sizes must be equal or one must be 1, which is stretched. A (3, 1) plus a (1, 3) silently becomes (3, 3), so a wrong shape can still run.',
  },
]
