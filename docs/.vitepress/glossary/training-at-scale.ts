import type { GlossaryEntry } from '../glossary'

const chapter = '/building/training-at-scale'

export const terms: Record<string, GlossaryEntry> = {
  adam: {
    term: 'Adam',
    def: 'The standard optimizer for LLMs. Besides the gradient, it keeps two running averages per parameter (of the gradient and of its square) to scale each update.',
    chapter: '/foundations/learning',
  },
  'optimizer-state': {
    term: 'Optimizer state',
    def: 'Extra numbers the optimizer keeps per parameter between steps. For Adam: two running averages, usually in 32-bit floats, so 8 bytes per parameter.',
    chapter,
  },
  'mixed-precision': {
    term: 'Mixed precision',
    def: 'Training with 16-bit (or 8-bit) numbers for the heavy matrix math, while keeping a 32-bit master copy of the weights and optimizer state so small updates are not lost.',
    chapter,
  },
  bf16: {
    term: 'bf16 (bfloat16)',
    def: 'A 16-bit float with the same range as a 32-bit float but only about 3 significant decimal digits. The default number format for LLM training math.',
    chapter,
  },
  fp8: {
    term: 'fp8',
    def: 'An 8-bit float format. Recent GPUs multiply fp8 matrices about twice as fast as bf16, but the tiny range needs careful per-block scaling.',
    chapter,
  },
  'loss-scaling': {
    term: 'Loss scaling',
    def: 'Multiplying the loss by a large constant before backpropagation so tiny gradients do not round to zero in fp16, then dividing it back out before the update.',
    chapter,
  },
  'data-parallelism': {
    term: 'Data parallelism',
    def: 'Every GPU holds a full model copy and processes a different slice of the batch. Gradients are averaged across GPUs every step.',
    chapter,
  },
  collective: {
    term: 'Collective',
    def: 'A communication operation that a whole group of GPUs takes part in at once, such as all-reduce, all-gather, reduce-scatter or all-to-all. NCCL is the usual library.',
    chapter,
  },
  'all-reduce': {
    term: 'All-reduce',
    def: 'A collective where every GPU starts with its own array and ends with the element-wise sum (or average) of all of them. Usually built as a reduce-scatter followed by an all-gather.',
    chapter,
  },
  'all-gather': {
    term: 'All-gather',
    def: 'A collective where each GPU holds one shard and every GPU ends up with all shards concatenated.',
    chapter,
  },
  'reduce-scatter': {
    term: 'Reduce-scatter',
    def: 'A collective where every GPU contributes a full array and each ends up with the sum of only its own shard.',
    chapter,
  },
  zero: {
    term: 'ZeRO',
    def: 'Zero Redundancy Optimizer: shard the optimizer state (stage 1), also the gradients (stage 2), and also the weights (stage 3) across data-parallel GPUs instead of copying them on each.',
    chapter,
  },
  fsdp: {
    term: 'FSDP',
    def: 'Fully Sharded Data Parallel: PyTorch’s implementation of ZeRO stage 3. Each GPU stores a shard of everything and gathers a layer’s full weights just before using it.',
    chapter,
  },
  'tensor-parallelism': {
    term: 'Tensor parallelism',
    def: 'Splitting each weight matrix of a layer across GPUs, so every GPU computes part of every matrix multiply. Needs fast links, because GPUs exchange results inside every layer.',
    chapter,
  },
  'pipeline-parallelism': {
    term: 'Pipeline parallelism',
    def: 'Splitting the model’s layers into consecutive stages on different GPUs, passing activations forward and gradients backward between them.',
    chapter,
  },
  'micro-batch': {
    term: 'Micro-batch',
    def: 'A slice of a batch processed as one unit. Pipelines feed many micro-batches so stages can work at the same time; gradients from all of them are summed before the update.',
    chapter,
  },
  'pipeline-bubble': {
    term: 'Pipeline bubble',
    def: 'Time a pipeline stage sits idle while the pipeline fills at the start of a step and drains at the end.',
    chapter,
  },
  'context-parallelism': {
    term: 'Context parallelism',
    def: 'Splitting one long sequence across GPUs, so each holds the activations for a chunk of tokens. Attention then needs the keys and values from other chunks.',
    chapter,
  },
  'sequence-parallelism': {
    term: 'Sequence parallelism (Megatron)',
    def: 'A companion to tensor parallelism: the parts of a layer that tensor parallelism does not split are split along the sequence instead, to save activation memory.',
    chapter,
  },
  moe: {
    term: 'Mixture of experts (MoE)',
    def: 'A layer with many alternative MLPs (experts) where a small router sends each token to only a few. Total parameters grow without growing the compute per token.',
    chapter,
  },
  'expert-parallelism': {
    term: 'Expert parallelism',
    def: 'Placing different experts of a mixture-of-experts layer on different GPUs and shipping each token to the GPUs holding its chosen experts with an all-to-all.',
    chapter,
  },
  'all-to-all': {
    term: 'All-to-all',
    def: 'A collective where every GPU sends a different piece of data to every other GPU, like the shuffle phase of MapReduce.',
    chapter,
  },
  nvlink: {
    term: 'NVLink',
    def: 'NVIDIA’s direct GPU-to-GPU links inside a server. About 900 GB/s per H100 (450 each way), roughly 9 times the network link each GPU has to other servers.',
    chapter,
  },
  infiniband: {
    term: 'InfiniBand',
    def: 'The low-latency network usually used between GPU servers in a training cluster, typically one 400 Gb/s (50 GB/s) link per GPU. RoCE (RDMA over Ethernet) is the main alternative.',
    chapter,
  },
  'activation-checkpointing': {
    term: 'Activation checkpointing',
    def: 'Keeping only some activations during the forward pass and recomputing the rest during the backward pass. Trades about a third more compute for much less memory. Also called recomputation.',
    chapter,
  },
  mfu: {
    term: 'MFU',
    def: 'Model FLOPs utilization: the share of the GPUs’ peak arithmetic that goes into useful model math. 35–45% is typical for large training runs.',
    chapter: '/building/pretraining',
  },
  'silent-data-corruption': {
    term: 'Silent data corruption',
    def: 'A hardware fault that produces a wrong result without any error being raised, such as a defective core computing a bad product.',
    chapter,
  },
}
