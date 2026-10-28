// Flashcards for the Inference and Serving chapter. Strings may contain inline HTML.

export const cards = [
  {
    q: 'Why can’t a model generate the tokens of one answer in parallel?',
    a: 'Each token is chosen from the model’s output given all tokens before it, including the ones it just generated. Token 2 depends on the choice of token 1, so each needs its own forward pass.',
  },
  {
    q: 'Why is prefill compute-bound and decode memory-bound?',
    a: 'Both read every weight once per pass. Prefill uses each weight for thousands of prompt tokens; decode for one sequence uses it once, about 1 FLOP per byte, far below the ~300 a GPU needs to keep its arithmetic busy.',
  },
  {
    q: 'How do you estimate the maximum decode speed of one sequence?',
    a: 'Memory bandwidth divided by bytes of weights, since every step reads them all. An 8B model in bf16 (16 GB) on a 3.35 TB/s H100 tops out near 200 tokens/s.',
  },
  {
    q: 'Why does batching help decode so much?',
    a: 'The weights are read once per step and reused by every sequence in the batch. Arithmetic intensity grows with batch size, so throughput rises almost for free until the batch reaches the compute limit.',
  },
  {
    q: 'Why can keys and values be cached, and why is the cache exact?',
    a: 'A token’s key and value at a layer depend only on that token and earlier ones. Later tokens never change them, so storing them gives exactly what recomputation would.',
  },
  {
    q: 'What is the KV cache size formula, and what is each factor?',
    a: '2 (key and value) × layers × KV heads × head dimension × bytes per number × tokens. For Llama 3.1 8B in bf16 that is 128 KiB per token.',
  },
  {
    q: 'Why does the KV cache, not compute, usually cap serving throughput?',
    a: 'Every active sequence needs its own cache in GPU memory. Free memory ÷ cache per sequence sets the maximum batch, and batch size sets decode efficiency.',
  },
  {
    q: 'How do GQA and MLA speed up decode, not only save memory?',
    a: 'They shrink the KV cache that every decode step must read for attention. Fewer bytes per step means faster steps and room for bigger batches.',
  },
  {
    q: 'What does temperature do to the distribution, and why?',
    a: 'Logits are divided by T before softmax. T below 1 widens the gaps so the top token dominates; T above 1 narrows them so the distribution flattens.',
  },
  {
    q: 'Why is top-p usually preferred to top-k?',
    a: 'A fixed k keeps too many tokens when the model is confident and too few when many continuations are fine. Top-p keeps however many tokens cover a fixed share of probability, so it adapts.',
  },
  {
    q: 'What does continuous batching fix compared with static batching?',
    a: 'Static batches run until their longest request finishes, idling slots and making new requests wait. Continuous batching swaps requests in and out after every step.',
  },
  {
    q: 'How is PagedAttention like virtual memory, and where does the analogy break?',
    a: 'The KV cache lives in fixed-size blocks mapped by a per-request block table, allocated on demand and shareable like copy-on-write pages. But no hardware walks the table: the attention kernel reads it itself.',
  },
  {
    q: 'Why does prefix caching require an exact match from the first token?',
    a: 'A token’s keys and values depend on every token before it. Change anything early and every later cached value is wrong, so put stable content first.',
  },
  {
    q: 'What does FlashAttention change, and what does it not change?',
    a: 'It changes memory traffic: attention is computed in tiles in on-chip memory, with an online softmax, so the n × n score matrix never goes to GPU main memory. The result is the same.',
  },
  {
    q: 'Why does weight-only 4-bit quantization speed up decode but not prefill?',
    a: 'Decode is limited by bytes read, and 4-bit weights are a quarter of bf16. Prefill is limited by arithmetic, which weight-only kernels still do in bf16.',
  },
  {
    q: 'Why is speculative decoding lossless?',
    a: 'The big model checks every guess and accepts it with probability min(1, p/q), drawing a replacement from the leftover distribution on rejection. This rejection-sampling rule makes every token follow the big model’s distribution exactly.',
  },
  {
    q: 'What makes serving mixture-of-experts models different from dense ones?',
    a: 'Memory must hold all experts, but each token computes with only a few. At large batch nearly every expert is read each step, so deployments spread experts over many GPUs and route tokens with all-to-all.',
  },
  {
    q: 'Why do larger batches improve throughput but hurt latency?',
    a: 'A larger batch shares each weight read among more users, so total tokens per second rise. But each step does more work, so each user waits longer between tokens, and new requests queue longer.',
  },
  {
    q: 'Why split prefill and decode onto separate GPUs?',
    a: 'On a shared GPU, long prefills stall decoding users and queued decodes delay first tokens. Separate pools can each be tuned for their own latency target, at the cost of copying the KV cache across the network.',
  },
]
