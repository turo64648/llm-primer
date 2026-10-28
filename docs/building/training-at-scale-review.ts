// Flashcards for the Training at Scale chapter. Strings may contain inline HTML.

export const cards = [
  {
    q: 'Where does “16 bytes per parameter” for training come from?',
    a: '16-bit weights (2) and gradients (2), plus a 32-bit master copy of the weights (4) and Adam’s two 32-bit running averages (8). Activations come on top.',
  },
  {
    q: 'Why does mixed-precision training keep 32-bit master weights?',
    a: 'A typical update is a tiny fraction of the weight. bf16 has only about 3 significant digits, so adding the update to a bf16 weight would usually round it away.',
  },
  {
    q: 'Why does fp16 training need loss scaling, and bf16 usually not?',
    a: 'fp16 has a small range, so tiny gradients round to zero; scaling the loss up lifts them into range. bf16 has the same range as fp32, so small gradients survive.',
  },
  {
    q: 'What does data parallelism split, and what does it not split?',
    a: 'It splits the work (each GPU gets a slice of the batch). It does not split memory: every GPU holds the full weights, gradients and optimizer state.',
  },
  {
    q: 'How does a ring all-reduce work, and what does each GPU send?',
    a: 'A reduce-scatter (N−1 steps where chunks are passed and summed around the ring) then an all-gather (N−1 steps passing finished chunks). Each GPU sends 2·(N−1)/N times the array size, about 2×, regardless of N.',
  },
  {
    q: 'If ring all-reduce bandwidth per GPU is constant, why don’t huge clusters use one flat ring?',
    a: 'It takes 2(N−1) sequential steps, each with a fixed latency. At thousands of GPUs that dominates, so libraries use trees and hierarchical schemes (inside a server first, then across).',
  },
  {
    q: 'What do ZeRO stages 1, 2 and 3 shard?',
    a: 'Stage 1 shards the optimizer state, stage 2 also the gradients, stage 3 also the weights. Stages 1 and 2 cost no extra communication; stage 3 costs about 1.5× because weights are gathered again for backward.',
  },
  {
    q: 'Why is ZeRO-2 nearly free in communication?',
    a: 'An all-reduce is a reduce-scatter plus an all-gather. After the reduce-scatter each GPU has the summed gradient for its shard, updates only that shard, and the all-gather sends updated weights instead of gradients.',
  },
  {
    q: 'How does Megatron split an MLP block across GPUs with only one all-reduce?',
    a: 'Split the first matrix by columns, so each GPU computes part of the hidden layer with no communication (GELU is element-wise). Split the second by rows, so each GPU gets a partial sum. One all-reduce adds them.',
  },
  {
    q: 'Why does tensor parallelism stay inside one server?',
    a: 'It needs blocking all-reduces inside every layer that cannot be hidden behind compute. Only NVLink is fast enough; the network between servers is roughly 9× slower per GPU.',
  },
  {
    q: 'What is the pipeline bubble and how do you shrink it?',
    a: 'Idle time while the pipeline fills and drains: (p−1)/(m+p−1) of the step for p stages and m micro-batches. Use more micro-batches, or interleave several layer chunks per GPU.',
  },
  {
    q: 'What does the 1F1B schedule improve over GPipe?',
    a: 'Not the bubble but memory: by starting each micro-batch’s backward pass early, a stage holds at most p micro-batches of activations instead of all m.',
  },
  {
    q: 'What problem does context parallelism solve, and what does it cost?',
    a: 'Very long sequences whose activations do not fit. It splits the sequence by position across GPUs; attention then needs the keys and values from other GPUs, passed around a ring or all-gathered.',
  },
  {
    q: 'Why is expert parallelism like a MapReduce shuffle?',
    a: 'The router assigns each token to experts on other GPUs, and an all-to-all ships them there and back. Like a shuffle, it suffers from skew: popular experts become hot partitions.',
  },
  {
    q: 'How do you place tensor, pipeline and data parallelism on a cluster?',
    a: 'Most frequent and most blocking communication innermost: tensor parallel inside a server over NVLink, then pipeline across nearby servers, and data parallel (once per step, overlapped) across the whole cluster.',
  },
  {
    q: 'What does activation checkpointing trade?',
    a: 'It keeps only layer inputs and recomputes the rest during backward. About a third more compute (one extra forward pass) for most of the activation memory; selective recomputation gets most of the saving for a few percent.',
  },
  {
    q: 'Why does one GPU failure stop a whole training job, and how do runs cope?',
    a: 'Every step is one synchronous computation across all GPUs, so there is no task to retry alone. Runs checkpoint often (Young: interval ≈ √(2 × write time × MTBF)), write asynchronously, and restart fast.',
  },
  {
    q: 'What is silent data corruption and how is it caught in training?',
    a: 'Hardware that computes wrong answers without raising an error. It shows up as loss spikes or NaNs, or as data-parallel replicas whose weights no longer match; teams roll back and drain the bad node.',
  },
]
