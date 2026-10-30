import type { GlossaryEntry } from '../glossary'

const chapter = '/foundations/tokens-embeddings'

export const terms: Record<string, GlossaryEntry> = {
  token: {
    term: 'Token',
    def: 'The unit of text a language model reads and writes: a word, part of a word, a punctuation mark or a few bytes. English averages about 4 characters per token.',
    chapter,
  },
  tokenizer: {
    term: 'Tokenizer',
    def: 'The fixed program that cuts text into tokens and maps each to an integer id, and maps ids back to text. It is built before model training and never changes afterwards.',
    chapter,
  },
  vocabulary: {
    term: 'Vocabulary',
    def: 'The fixed list of all tokens a model knows, each with an integer id. Current models use about 32,000 to 260,000 entries.',
    chapter,
  },
  oov: {
    term: 'Out-of-vocabulary (OOV)',
    def: 'A word not in a word-level vocabulary. Old word-level models replaced it with a single "unknown" token, losing it. Byte-level tokenizers have no OOV words.',
    chapter,
  },
  'utf-8': {
    term: 'UTF-8',
    def: 'The standard way to store text as bytes. ASCII characters take 1 byte; other characters take 2 to 4 bytes.',
    chapter,
  },
  bpe: {
    term: 'Byte-pair encoding (BPE)',
    def: 'A tokenizer training method: start from single bytes or characters, and repeatedly merge the most frequent adjacent pair into a new token, until the vocabulary has the wanted size.',
    chapter,
  },
  'byte-level-bpe': {
    term: 'Byte-level BPE',
    def: 'BPE that starts from the 256 byte values, so any text can be encoded and no input is ever unknown. Introduced for GPT-2; used by most current models.',
    chapter,
  },
  'pre-tokenization': {
    term: 'Pre-tokenization',
    def: 'Splitting text into chunks (words, numbers, punctuation, whitespace) with a regular expression before BPE runs. Merges never cross chunk boundaries.',
    chapter,
  },
  'special-token': {
    term: 'Special token',
    def: 'A reserved token, such as end-of-text or a chat role marker, that ordinary text can never produce. Software inserts it on purpose to mark structure.',
    chapter,
  },
  'glitch-token': {
    term: 'Glitch token',
    def: 'A token that exists in the vocabulary but almost never appeared in model training, so its embedding stayed near random and the model behaves erratically on it. Example: " SolidGoldMagikarp" in GPT-2 and GPT-3.',
    chapter,
  },
  'unigram-tokenizer': {
    term: 'Unigram tokenizer',
    def: 'A tokenizer that starts with a large vocabulary and prunes it, keeping the tokens that best explain the text under a simple probability model. Available in the SentencePiece library.',
    chapter,
  },
  'one-hot': {
    term: 'One-hot vector',
    def: 'A vector of zeros with a single 1 at the position of one item. Multiplying it by a matrix picks out one row.',
    chapter,
  },
  embedding: {
    term: 'Embedding',
    def: 'The learned vector that stands for a token. All of them form the embedding table, a matrix with one row per vocabulary entry; looking up a token means reading its row.',
    chapter,
  },
  unembedding: {
    term: 'Unembedding',
    def: 'The last matrix of a language model. It turns the final vector at a position into one logit per vocabulary entry. Also called the output layer or LM head.',
    chapter,
  },
  'weight-tying': {
    term: 'Weight tying',
    def: 'Using the same matrix for the embedding table and the unembedding, which saves vocabulary × width parameters. Common in small models.',
    chapter,
  },
  'language-model': {
    term: 'Language model',
    def: 'A model that gives a probability distribution over the next token, given the tokens before it. Generating text means sampling from it one token at a time.',
    chapter,
  },
  'bigram-model': {
    term: 'Bigram model',
    def: 'A language model that predicts the next token from the current token only. Its whole network can be one table of next-token logits.',
    chapter,
  },
  perplexity: {
    term: 'Perplexity',
    def: 'exp(average cross-entropy loss per token). Reads as "as unsure as choosing evenly among this many tokens". Only comparable between models with the same tokenizer.',
    chapter,
  },
}
