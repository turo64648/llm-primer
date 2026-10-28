import type { GlossaryEntry } from '../glossary'

const chapter = '/running/inference'

export const terms: Record<string, GlossaryEntry> = {
  autoregressive: { term: 'Autoregressive generation', def: 'Producing text one token at a time, appending each new token to the input before computing the next. Output tokens cannot be generated in parallel.', chapter },
  prefill: { term: 'Prefill', def: 'The first phase of a request: one forward pass over the whole prompt. It does many FLOPs per byte of weights read, so it is limited by compute.', chapter },
  decode: { term: 'Decode', def: 'The generation phase: one forward pass per new token. Each step reads all the weights for little math, so it is limited by memory bandwidth.', chapter },
  'arithmetic-intensity': { term: 'Arithmetic intensity', def: 'FLOPs done per byte moved from memory. Below the hardware’s ratio of peak FLOP/s to bandwidth (about 295 on an H100), a workload is memory-bound.', chapter },
  roofline: { term: 'Roofline model', def: 'A plot of achievable speed against arithmetic intensity: a slanted line where memory bandwidth limits, then a flat roof where peak compute limits.', chapter },
  hbm: { term: 'HBM (high-bandwidth memory)', def: 'The GPU’s main memory, stacked next to the chip. Tens of GB at a few TB/s; holds the weights and the KV cache.', chapter },
  'kv-cache': { term: 'KV cache', def: 'The stored keys and values of every earlier token, at every layer, so decode does not recompute them. Size: 2 × layers × kv_heads × head_dim × bytes × tokens.', chapter },
  mqa: { term: 'Multi-query attention (MQA)', def: 'Attention where all query heads share one key/value head. Shrinks the KV cache by the number of heads, at some cost in quality.', chapter },
  gqa: { term: 'Grouped-query attention (GQA)', def: 'Attention where groups of query heads share a key/value head (for example 64 query heads, 8 KV heads). Shrinks the KV cache with little quality loss.', chapter },
  mla: { term: 'Multi-head latent attention (MLA)', def: 'DeepSeek’s attention variant that caches one small compressed vector per token and layer and rebuilds keys and values from it, shrinking the KV cache further.', chapter },
  'greedy-decoding': { term: 'Greedy decoding', def: 'Always choosing the single most likely next token. Deterministic in principle, but long outputs tend to repeat.', chapter },
  temperature: { term: 'Temperature', def: 'A number the logits are divided by before softmax. Below 1 sharpens the distribution toward the top token; above 1 flattens it.', chapter },
  'top-k': { term: 'Top-k sampling', def: 'Sampling only from the k most likely tokens, after renormalising their probabilities.', chapter },
  'top-p': { term: 'Top-p (nucleus) sampling', def: 'Sampling from the smallest set of top tokens whose probabilities add up to at least p. The set adapts to how confident the model is.', chapter },
  'min-p': { term: 'Min-p sampling', def: 'Keeping only tokens whose probability is at least a fixed fraction of the top token’s, then sampling among them.', chapter },
  'continuous-batching': { term: 'Continuous batching', def: 'Scheduling at every decode step: finished requests leave and new ones join the running batch, so the GPU stays full and short requests do not wait for long ones.', chapter },
  'paged-attention': { term: 'PagedAttention', def: 'vLLM’s technique of storing the KV cache in fixed-size blocks mapped by a per-request block table, like virtual-memory pages. Nearly eliminates wasted cache memory.', chapter },
  'prefix-caching': { term: 'Prefix caching', def: 'Keeping the KV cache of a shared prompt prefix (system prompt, document, earlier turns) and reusing it for later requests that start with the same tokens.', chapter },
  'flash-attention': { term: 'FlashAttention', def: 'An exact attention algorithm that works in tiles held in on-chip memory and never writes the full score matrix to GPU main memory.', chapter },
  quantization: { term: 'Quantization', def: 'Storing weights (and sometimes activations or the KV cache) in fewer bits, such as 8 or 4, with shared scale factors. Cuts memory and bytes read per token.', chapter },
  'speculative-decoding': { term: 'Speculative decoding', def: 'A cheap draft guesses several tokens and the big model checks them in one pass. An acceptance rule keeps the output distribution exactly the big model’s.', chapter },
  ttft: { term: 'TTFT (time to first token)', def: 'Time from a request’s arrival to its first output token: queueing plus prefill.', chapter },
  tpot: { term: 'TPOT (time per output token)', def: 'The average gap between output tokens during decode, also called inter-token latency.', chapter },
  'pd-disaggregation': { term: 'Prefill/decode disaggregation', def: 'Running prefill and decode on separate GPU pools and copying each request’s KV cache between them, so the two phases stop interfering.', chapter },
}
