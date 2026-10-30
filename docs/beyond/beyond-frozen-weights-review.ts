// Flashcards for the Beyond Frozen Weights chapter. Strings may contain inline HTML.

export const cards = [
  {
    q: 'What are fast weights, and how do they differ from activations and ordinary weights?',
    a: 'Weights that change while the network runs, holding recent information for longer than one token. Activations vanish after each token; ordinary (slow) weights change only in training.',
  },
  {
    q: 'How does an outer product store a key→value pair in a matrix?',
    a: 'Add v kᵀ to the matrix M. Then M·k = v·(k·k) plus crosstalk from other pairs, so reading with the key returns roughly v.',
  },
  {
    q: 'Why is linear attention a fast-weight memory?',
    a: 'Without the softmax, the attention output is (Σ v kᵀ)·q. The sum is a fixed-size matrix written by one outer product per token and read by the query: exactly a fast-weight memory.',
  },
  {
    q: 'Why does a d × d linear memory stop recalling cleanly after about d pairs?',
    a: 'Recall is clean only if each key is nearly orthogonal to all others. In d dimensions at most d keys can be fully orthogonal, so beyond that every read picks up crosstalk from other pairs.',
  },
  {
    q: 'What does the delta rule fix compared with plain outer-product writes?',
    a: 'It writes only the error, v − M k. Writing a key twice replaces the old value instead of averaging it, and similar keys interfere less. It is one gradient step on the recall loss.',
  },
  {
    q: 'Why do fixed-state models (Mamba, DeltaNet) fall behind attention on exact recall and copying?',
    a: 'Their state has a fixed size, so a long input must be compressed and details are lost. Attention keeps every token and can look any of them up exactly, at growing cost.',
  },
  {
    q: 'Why do production long-context models use hybrids rather than pure linear attention?',
    a: 'Fixed-state layers are cheap but recall poorly; a few full attention layers restore exact lookup. Qwen3-Next and Kimi Linear use about three linear layers per attention layer.',
  },
  {
    q: 'What is test-time training?',
    a: 'Taking gradient steps on a self-supervised loss, on the input being processed, while the model is in use. TTT layers do this inside the network: the hidden state is itself a small model being trained.',
  },
  {
    q: 'How does Titans decide what to store in its long-term memory?',
    a: 'It measures surprise: the gradient of the memory’s prediction error on the current key→value pair. Big surprise means a big write; momentum carries recent surprise forward and a forget gate decays old content.',
  },
  {
    q: 'Does Titans give an LLM persistent learning across conversations?',
    a: 'Not by default. The memory is written during one long sequence and reset afterwards; the rules for how to write are frozen weights. It is an in-between timescale, not lifelong learning.',
  },
  {
    q: 'What does ROME do, in one sentence?',
    a: 'It finds the middle-layer MLP that recalls a fact about a subject (causal tracing), treats that MLP as a key→value memory, and adds a rank-one update so the subject’s key now maps to a new value.',
  },
  {
    q: 'Name two known failure modes of model editing.',
    a: 'Edits often fail to ripple to related facts (multi-hop questions still use the old fact), and many sequential edits gradually damage the model’s other abilities.',
  },
  {
    q: 'What is a hypernetwork, and how does Text-to-LoRA use one?',
    a: 'A network that outputs weights for another network. Text-to-LoRA reads a task description and outputs a LoRA adapter in one forward pass, instead of fine-tuning one.',
  },
  {
    q: 'How does SEAL get a training signal for its own weight updates?',
    a: 'The model writes its own fine-tuning data (a self-edit), is updated on it, and is tested afterwards. Reinforcement learning rewards self-edits that improved the test score. It needs checkable downstream tasks.',
  },
  {
    q: 'Why is backpropagation hard to run online, one token at a time?',
    a: 'It needs the stored activations of every layer and a backward pass with the same weights, after the outcome is known. Doing that per step is costly in memory, and a single step on one example tends to overwrite old knowledge.',
  },
  {
    q: 'Which hard part of the thread does none of these methods solve?',
    a: 'A trustworthy training signal at inference time where answers cannot be checked. Test-time memories learn from prediction surprise, SEAL from checkable rewards; neither knows whether the new information is true or safe.',
  },
]
