// Flashcards for the Transformer chapter. Strings may contain inline HTML.

export const cards = [
  {
    q: 'Why did transformers replace RNNs for language models?',
    a: 'An RNN must process tokens one after another, so training cannot run in parallel across a sequence, and it squeezes all history into one fixed-size state. Attention lets every token read every earlier token directly, and all positions train in parallel.',
  },
  {
    q: 'In what sense is attention a soft key-value lookup?',
    a: 'Each token has a query; every earlier token has a key and a value. Instead of returning the one value whose key matches exactly, attention returns an average of all values, weighted by how well each key matches the query.',
  },
  {
    q: 'Where does the key-value-store analogy for attention break?',
    a: 'The keys, values and queries are all computed from the current text, not stored ahead of time. Nothing is written for later: the "store" is rebuilt for each input and gone after it. And the result is a blend, never one exact hit.',
  },
  {
    q: 'Why divide attention scores by √d?',
    a: 'A dot product of two d-dimensional vectors with unit-variance entries has a spread of about √d. Large scores push softmax to near one-hot, where gradients vanish. Dividing by √d keeps scores near spread 1, so training starts in a useful range.',
  },
  {
    q: 'How is the causal mask implemented, and why is it needed?',
    a: 'Scores for later tokens are set to minus infinity before the softmax, so their weights are exactly 0. Without it, during training each position could read the very token it must predict, and the model would learn to copy instead of predict.',
  },
  {
    q: 'Why is attention O(n²) in sequence length?',
    a: 'Every token computes a score against every earlier token, so the attention matrix has about n² entries per head per layer. Doubling the context quadruples that work. The weights-side cost per token stays fixed.',
  },
  {
    q: 'Why use several attention heads instead of one big one?',
    a: 'One softmax produces one weighted average, so one head can mainly follow one kind of relation at a time. Several smaller heads can each attend to something different (previous token, subject of the verb, a matching bracket) for about the same cost.',
  },
  {
    q: 'What does the MLP in each block do, if attention already mixes tokens?',
    a: 'Attention moves information between tokens; the MLP processes each token on its own. It holds about two thirds of the parameters, and much stored knowledge seems to live there, acting like a key→value memory.',
  },
  {
    q: 'What is the residual stream?',
    a: 'The running vector for each token that every block reads from and adds its output to (x = x + f(x)). It gives gradients a direct path through deep stacks and acts as a shared channel between layers.',
  },
  {
    q: 'Why do transformers normalise, and why pre-norm?',
    a: 'Normalising keeps the scale of each sublayer input steady, so deep stacks train without exploding or vanishing values. Pre-norm normalises only the branch inputs and leaves the residual path clean, which trains more stably than the original post-norm layout.',
  },
  {
    q: 'Why does a transformer need positional information at all?',
    a: 'Attention computes a weighted average over a set; shuffling the earlier tokens gives the same result. Without position signals "dog bites man" and "man bites dog" look alike to it.',
  },
  {
    q: 'How does RoPE encode position?',
    a: 'It rotates each pair of numbers in a query and key by an angle proportional to the token position, at a different speed per pair. The dot product of two rotated vectors depends only on the difference of the angles, so scores depend on relative distance.',
  },
  {
    q: 'Derive the ≈12·d² parameters per transformer layer.',
    a: 'Attention has four d × d matrices (query, key, value, output): 4d². The MLP goes d → 4d → d: 2 × 4d² = 8d². Total 12d², ignoring small norms and biases.',
  },
  {
    q: 'What are the forward FLOPs per token of a transformer?',
    a: 'About 2N, one multiply and one add per parameter, plus attention: about 4·n·d per layer for a token that sees n earlier tokens. At short context the 2N term dominates; at very long context the attention term can exceed it.',
  },
  {
    q: 'What is a KV cache and why does it work?',
    a: 'When generating, earlier tokens\' keys and values never change, because the causal mask stops them from seeing later tokens. So they are stored once and reused, and each new token computes only its own query, key and value.',
  },
  {
    q: 'What changed between the 2017 transformer and a 2025-era open model?',
    a: 'Decoder-only, pre-norm with RMSNorm, RoPE instead of added positions, SwiGLU MLPs, no bias terms, grouped-query attention to shrink the KV cache, and often mixture-of-experts MLPs. The core block is the same.',
  },
  {
    q: 'Attention as memory: what does it hold, and for how long?',
    a: 'Keys and values are a per-token memory of the current context: rich and exact, but gone when the context ends. Weights are the long-term memory, frozen after training. Nothing in a standard transformer moves information from the first to the second.',
  },
]
