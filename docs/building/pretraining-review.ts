// Flashcards for the Pretraining and Scaling Laws chapter. Strings may contain inline HTML.

export const cards = [
  {
    q: 'How is a model graded during pretraining?',
    a: 'At every position it outputs a probability for every possible next token. The loss is −log of the probability it gave the token that really came next, averaged over all positions.',
  },
  {
    q: 'Why does plain next-token prediction teach a model so much?',
    a: 'To predict text well, the model has to capture whatever produced it: grammar, facts, code, reasoning, what the writer wants. The loss only keeps falling as it learns more of that structure.',
  },
  {
    q: 'Why does a base model not reliably answer questions?',
    a: 'It was trained to continue text, not to help. A question might plausibly be followed by more questions. Instruction following comes from post-training.',
  },
  {
    q: 'Why deduplicate pretraining data?',
    a: 'Duplicates waste compute and make the model memorise those passages word for word. Removing them gives an equal or better model in fewer steps.',
  },
  {
    q: 'How do pipelines find near-duplicates among billions of documents?',
    a: 'MinHash gives each document a short signature whose agreement tracks shared n-grams. Locality-sensitive hashing buckets the signatures, so only documents in the same bucket get compared.',
  },
  {
    q: 'How is a model-based quality filter built, and what is its risk?',
    a: 'Pick examples of good text (or have a large model grade a sample), train a small fast classifier, score everything, keep the top slice. The risk is that "quality" means "looks like my examples", which can drop dialects, forums and rare topics.',
  },
  {
    q: 'What is a data mixture, and how do labs choose it?',
    a: 'The share of tokens from each source: web, code, math, books, other languages. Labs train many small models on different mixtures and extrapolate, and often end training on the highest-quality data.',
  },
  {
    q: 'What is the main risk of training on synthetic data?',
    a: 'Narrowing: models trained repeatedly on their own outputs lose rare cases first and can degrade ("model collapse"). The effect is much milder when synthetic data is added to real data rather than replacing it.',
  },
  {
    q: 'What do scaling laws let a lab do?',
    a: 'Predict the loss of a huge, expensive run from a ladder of cheap small ones, and pick the model size and data amount before committing the budget.',
  },
  {
    q: 'What did Chinchilla find, and how did it differ from Kaplan et al.?',
    a: 'For a fixed budget, grow model size and data at the same rate, about 20 tokens per parameter. Kaplan had said to grow the model much faster; later work traced that to setup choices.',
  },
  {
    q: 'Why are small modern models trained on thousands of tokens per parameter?',
    a: 'Serving cost depends on model size, and a popular model is served billions of times. Extra training compute that makes a small model better is cheaper than serving a bigger one. Loss keeps improving past 20 tokens per parameter, only more slowly.',
  },
  {
    q: 'Where does the 6 in C ≈ 6ND come from?',
    a: 'The forward pass costs a multiply and an add per parameter per token (2N). The backward pass computes gradients for inputs and for weights, each about the same cost (4N).',
  },
  {
    q: 'What is MFU, and why is it well below 100%?',
    a: 'The share of the GPUs’ peak FLOP/s that goes into the useful 6ND work. Communication, memory-bound operations, pipeline idle time and recomputation eat the rest; about 40% is good.',
  },
  {
    q: 'How do you estimate the GPU-hours for a training run?',
    a: 'Compute 6ND, then divide by (peak FLOP/s per GPU × MFU) and by 3,600. For Llama 3.1 405B that gives about 26M H100-hours against a reported 31M.',
  },
  {
    q: 'Why warm up the learning rate?',
    a: 'At the start the weights are random, gradients are large and erratic, and Adam’s running statistics are not yet reliable. Full-size steps then can push the model somewhere it never recovers from.',
  },
  {
    q: 'Why decay the learning rate at the end, and why do it on the best data?',
    a: 'Large steps keep the model bouncing around the bottom of the loss valley; small steps let it settle. Pairing that careful final phase with the highest-quality data (annealing) gets the most out of it.',
  },
  {
    q: 'What do teams do when a loss spike does not recover?',
    a: 'Roll back to a checkpoint from before the spike, skip the batches around it, and possibly lower the learning rate or tighten gradient clipping.',
  },
  {
    q: 'How much is repeated data worth?',
    a: 'Up to about 4 epochs is nearly as good as fresh data. Beyond that each pass is worth less, and around 16 epochs extra passes add almost nothing.',
  },
]
