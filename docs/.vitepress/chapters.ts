// Single source of truth for the book's structure.
// Used by the sidebar (config.mts), the home page chapter list, and stub pages.

export interface Chapter {
  id: string
  num: string
  title: string
  link: string
  ready: boolean
  topics: string[]
}

export interface Part {
  title: string
  chapters: Chapter[]
}

export const parts: Part[] = [
  {
    title: '0. Toolkit',
    chapters: [
      {
        id: 'math-toolkit', num: '0', title: 'The Math You Need', link: '/start/math-toolkit', ready: true,
        topics: ['Vectors, dot products and matrix multiply as "many dot products"', 'Derivatives, partial derivatives, the chain rule', 'exp, log, softmax', 'Probability: distributions, log-probabilities, cross-entropy', 'Shapes and broadcasting in NumPy'],
      },
    ],
  },
  {
    title: 'I. Foundations',
    chapters: [
      {
        id: 'neural-nets', num: '1', title: 'Neural Networks from Scratch', link: '/foundations/neural-nets', ready: true,
        topics: ['A neuron: weighted sum plus nonlinearity', 'Layers as matrix multiplies', 'Activations (ReLU, GELU) and why nonlinearity matters', 'What "the model" physically is: a file of numbers plus a fixed program', 'Exercise: an MLP forward pass in NumPy'],
      },
      {
        id: 'learning', num: '2', title: 'How Networks Learn', link: '/foundations/learning', ready: true,
        topics: ['Loss functions and cross-entropy', 'Gradients and backpropagation (the chain rule, mechanised)', 'SGD, momentum, Adam; learning rate schedules', 'Overfitting, regularisation, generalisation', 'Exercise: a tiny autograd engine, then the same in PyTorch'],
      },
      {
        id: 'tokens-embeddings', num: '3', title: 'Text as Numbers', link: '/foundations/tokens-embeddings', ready: false,
        topics: ['Why not characters or words', 'Byte-pair encoding (BPE), step by step', 'Token quirks: arithmetic, spelling, multilingual cost', 'Embeddings: a lookup table that learns', 'Exercise: train a BPE tokenizer; a bigram language model'],
      },
      {
        id: 'transformer', num: '4', title: 'The Transformer', link: '/foundations/transformer', ready: false,
        topics: ['Attention: intuition, then Q/K/V math', 'Causal masking, multi-head attention', 'MLP blocks, residual stream, layer norm', 'Positional information and RoPE', 'Exercise: a tiny GPT trained on CPU'],
      },
    ],
  },
  {
    title: 'II. Building a Model',
    chapters: [
      {
        id: 'pretraining', num: '5', title: 'Pretraining and Scaling Laws', link: '/building/pretraining', ready: true,
        topics: ['Next-token prediction as the objective', 'Data: sources, filtering, deduplication, mixtures', 'Scaling laws and Chinchilla', 'Compute: FLOPs ≈ 6ND, MFU, what a run costs', 'Why it costs what it costs'],
      },
      {
        id: 'training-at-scale', num: '6', title: 'Training at Scale', link: '/building/training-at-scale', ready: true,
        topics: ['Memory budget of training: weights, grads, optimizer state, activations', 'Data, tensor, pipeline, sequence and expert parallelism', 'ZeRO and FSDP', 'Mixed precision (bf16, fp8), activation checkpointing', 'Collectives, interconnects, failures, checkpoint/restart'],
      },
      {
        id: 'post-training', num: '7', title: 'Post-training and Reasoning', link: '/building/post-training', ready: true,
        topics: ['Supervised fine-tuning (SFT)', 'RLHF: reward models and PPO', 'DPO and friends', 'RL on verifiable rewards (GRPO)', 'Reasoning models and test-time compute'],
      },
    ],
  },
  {
    title: 'III. Running and Judging',
    chapters: [
      {
        id: 'inference', num: '8', title: 'Inference and Serving', link: '/running/inference', ready: true,
        topics: ['Autoregressive decoding: prefill vs decode', 'KV cache and its memory math', 'Sampling: temperature, top-k, top-p', 'Batching, PagedAttention, FlashAttention', 'Quantization, speculative decoding, MoE serving'],
      },
      {
        id: 'evaluation', num: '9', title: 'Evaluation', link: '/running/evaluation', ready: true,
        topics: ['Benchmarks and what they measure', 'Contamination', 'Goodhart’s law and benchmark saturation', 'LLM-as-judge, human preference arenas', 'Why models get good at verifiable tasks first'],
      },
    ],
  },
  {
    title: 'IV. Understanding and Steering',
    chapters: [
      {
        id: 'interp-alignment', num: '10', title: 'Interpretability and Alignment', link: '/steering/interp-alignment', ready: true,
        topics: ['Features, superposition, sparse autoencoders', 'Circuits and probing', 'Specification gaming, reward hacking', 'Scalable oversight, constitutional methods', 'What labs care about and why'],
      },
    ],
  },
  {
    title: 'V. Learning After Deployment',
    chapters: [
      {
        id: 'agents-memory', num: '11', title: 'Agents, Tools, RAG and Memory', link: '/beyond/agents-memory', ready: false,
        topics: ['Tool use and the agent loop', 'Retrieval-augmented generation', 'Long context vs retrieval', 'How agent memory works today', 'Limits: conclusions vs intuitions'],
      },
      {
        id: 'continual-learning', num: '12', title: 'Continual Learning', link: '/beyond/continual-learning', ready: false,
        topics: ['Why deployed models are frozen checkpoints', 'Catastrophic forgetting', 'Stability vs plasticity; EWC, replay', 'Complementary learning systems and consolidation', 'Per-user vs shared models'],
      },
      {
        id: 'beyond-frozen-weights', num: '13', title: 'Beyond Frozen Weights', link: '/beyond/beyond-frozen-weights', ready: false,
        topics: ['Fast weights; attention as fast weights', 'Test-time training and Titans-style learned memory', 'Model editing: ROME, MEMIT', 'Hypernetworks and generated LoRAs', 'Local and Hebbian learning rules'],
      },
    ],
  },
  {
    title: 'Extras',
    chapters: [
      {
        id: 'thread', num: 'T', title: 'The Frozen-Weights Thread', link: '/extras/thread', ready: false,
        topics: ['The claim, stated carefully', 'Evidence for', 'Evidence against', 'Open problems and what would change the picture'],
      },
    ],
  },
]

export const allChapters: Chapter[] = parts.flatMap((p) => p.chapters)
