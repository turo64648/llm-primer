// Flashcards for the Text as Numbers chapter. Strings may contain inline HTML.

export const cards = [
  {
    q: 'Why not feed a model one character at a time?',
    a: 'Sequences get about 4 times longer than with tokens. Attention cost grows with the square of the length, the context window holds 4 times less text, and generation needs 4 times more steps.',
  },
  {
    q: 'Why not use whole words as tokens?',
    a: 'The set of words is open: names, typos, code, new words, other languages. A word list cannot cover them, so unknown words are lost (out-of-vocabulary), and a list big enough to come close needs millions of embedding rows.',
  },
  {
    q: 'How does BPE training work?',
    a: 'Start with every byte as a token. Count all adjacent pairs in the training text, merge the most frequent pair into a new token, and repeat until the vocabulary reaches the wanted size. The ordered list of merges is the tokenizer.',
  },
  {
    q: 'How does a BPE tokenizer encode new text?',
    a: 'Convert the text to bytes, then apply the learned merges in the order they were learned. Common words end up as one token; rare words as several pieces; anything else as raw bytes.',
  },
  {
    q: 'What does byte-level BPE guarantee?',
    a: 'Every possible text can be encoded, because the 256 byte values are always in the vocabulary. No input is ever unknown; rare text costs more tokens.',
  },
  {
    q: 'What is pre-tokenization for?',
    a: 'A regular expression splits text into words, numbers, punctuation and whitespace before BPE. Merges never cross these boundaries, so tokens do not glue the end of one word to the start of the next.',
  },
  {
    q: 'Why do LLMs struggle to count the r’s in “strawberry”?',
    a: 'The model never sees letters. It sees a few token ids, and each id’s embedding says nothing directly about spelling. It must have learned the spelling of each token indirectly from training text.',
  },
  {
    q: 'Why does tokenization hurt arithmetic, and what helps?',
    a: 'Older tokenizers split numbers into arbitrary chunks (“1234” as “12”+“34” or “123”+“4”), so digits do not line up by place value. Splitting every digit, or fixed groups of up to 3, gives consistent pieces.',
  },
  {
    q: 'What is the “token tax” on non-English text?',
    a: 'Tokenizers are trained mostly on English, so other languages and scripts get fewer merges and need more tokens per word. Those users pay more, wait longer and fit less text in the context window.',
  },
  {
    q: 'What caused glitch tokens like “ SolidGoldMagikarp”?',
    a: 'The tokenizer was trained on data where the string was common (a Reddit username), but the model saw it almost never. Its embedding row got almost no gradient updates, so it stayed near random and the model behaved erratically on it.',
  },
  {
    q: 'Why is an embedding lookup the same as a matrix multiply?',
    a: 'A one-hot vector (all zeros, one 1 at the token’s id) times the embedding matrix gives exactly that row. Real code skips the multiply and reads the row.',
  },
  {
    q: 'How are embeddings learned?',
    a: 'They are ordinary parameters, random at the start. Backpropagation sends a gradient into the rows of the tokens in each batch, and the optimizer updates them like any other weight.',
  },
  {
    q: 'What is weight tying?',
    a: 'Using the embedding matrix also as the unembedding, the matrix that turns the final vector into logits. It saves vocabulary × width parameters; common in small models, rarer in large ones.',
  },
  {
    q: 'How many parameters does a 128K-token vocabulary cost at width 4096?',
    a: '128,256 × 4,096 ≈ 525 million per matrix, about 1.05 billion for untied embedding and unembedding: roughly 13% of an 8B model.',
  },
  {
    q: 'What is perplexity, and when can you compare it?',
    a: 'exp(average loss per token): “as unsure as choosing evenly among this many tokens”. Only between models with the same tokenizer; otherwise compare loss per byte (or bits per byte).',
  },
  {
    q: 'What does a language model output at each position?',
    a: 'One logit per vocabulary entry. Softmax turns them into a probability distribution over the next token; generation samples one token, appends it, and repeats.',
  },
]
