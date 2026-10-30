import type { GlossaryEntry } from '../glossary'

const chapter = '/foundations/transformer'

export const terms: Record<string, GlossaryEntry> = {
  transformer: {
    term: 'Transformer',
    def: 'The neural network design behind current LLMs (2017). A stack of identical blocks, each with an attention step that mixes information between tokens and an MLP that works on each token alone.',
    chapter,
  },
  rnn: {
    term: 'Recurrent neural network (RNN)',
    def: 'A network that reads a sequence one step at a time and carries a fixed-size state vector forward. It must run step by step and squeeze all history into that one state.',
    chapter,
  },
  attention: {
    term: 'Attention',
    def: 'A layer in which each token computes a weighted average of information from other tokens. The weights come from how well its query matches their keys, so each token picks what to read.',
    chapter,
  },
  'self-attention': {
    term: 'Self-attention',
    def: 'Attention where queries, keys and values all come from the same sequence, so the tokens of one text read from each other. The attention in a GPT-style model.',
    chapter,
  },
  'query-key-value': {
    term: 'Query, key, value (Q, K, V)',
    def: 'Three vectors each token computes for attention: the query says what it is looking for, the key says what it offers to be found by, the value is what it hands over when found.',
    chapter,
  },
  'causal-mask': {
    term: 'Causal mask',
    def: 'The rule that a token may attend only to itself and earlier tokens. Implemented by setting scores for later tokens to minus infinity before the softmax, so their weights are 0.',
    chapter,
  },
  'attention-head': {
    term: 'Attention head',
    def: 'One independent attention computation with its own query, key and value projections on a slice of the model width. A layer runs several heads in parallel (multi-head attention).',
    chapter,
  },
  'multi-head-attention': {
    term: 'Multi-head attention',
    def: 'Running several attention heads side by side in one layer, each with smaller vectors, then joining their outputs. Lets one layer look for several kinds of relation at once.',
    chapter,
  },
  'decoder-block': {
    term: 'Decoder block',
    def: 'The repeated unit of a GPT-style transformer: normalise, attention, add back to the residual stream; normalise, MLP, add back. Models stack tens of them.',
    chapter,
  },
  'layer-norm': {
    term: 'Layer normalisation (LayerNorm)',
    def: 'Rescales each token vector to mean 0 and spread 1, then applies a learned scale and shift. Keeps the numbers going into each sublayer in a steady range, which makes deep training stable.',
    chapter,
  },
  rmsnorm: {
    term: 'RMSNorm',
    def: 'A cheaper LayerNorm: divides each vector by its root-mean-square and applies a learned scale, without subtracting the mean. Standard in current open models.',
    chapter,
  },
  'pre-norm': {
    term: 'Pre-norm',
    def: 'Placing the normalisation at the input of each attention or MLP branch, leaving the residual stream itself untouched. More stable to train than the original post-norm layout.',
    chapter,
  },
  'positional-encoding': {
    term: 'Positional encoding',
    def: 'Any way of telling a transformer where each token sits in the sequence. Attention alone ignores order, so position must be added: learned vectors, fixed sine waves, or rotations (RoPE).',
    chapter,
  },
  rope: {
    term: 'Rotary position embedding (RoPE)',
    def: 'Encodes position by rotating pairs of numbers in each query and key by an angle proportional to the position. The query–key dot product then depends on how far apart two tokens are.',
    chapter,
  },
  swiglu: {
    term: 'SwiGLU',
    def: 'An MLP variant used by most current LLMs: two input projections, one passed through the SiLU curve, multiplied element by element (a gate), then projected back.',
    chapter,
  },
}
