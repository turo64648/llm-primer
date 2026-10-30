// Flashcards for the Agents, Tools, RAG and Memory chapter. Strings may contain inline HTML.

export const cards = [
  {
    q: 'What are the only two places a deployed model’s knowledge can come from?',
    a: 'Its frozen weights, set during training, and the text in its context window for this request. Tools, RAG and memory all work by putting text into the context.',
  },
  {
    q: 'How does tool calling work mechanically?',
    a: 'Tool schemas are put in the prompt. The model writes a structured call (name plus JSON arguments) and stops; the harness runs the real function and appends the result, then calls the model again.',
  },
  {
    q: 'How do models learn to call tools?',
    a: 'In post-training: supervised fine-tuning on conversations with correct calls, and reinforcement learning on real tasks in sandboxes, rewarded for success. Toolformer showed a model can label its own tool-use data.',
  },
  {
    q: 'Why is the context the agent’s only working memory?',
    a: 'Activations are recomputed from the context on every call and weights do not change. Anything the agent learned during the task exists only as text in the context; if it is dropped, it is forgotten.',
  },
  {
    q: 'What is ReAct, and why write the thought before the action?',
    a: 'A loop of thought, action, observation. Writing the reasoning first lets the model use its own words as scratch space, the same reason chain of thought helps.',
  },
  {
    q: 'Why do long agent tasks fail even when each step is reliable?',
    a: 'Success compounds: n independent steps at success rate p all succeed with p^n, so 0.99^100 ≈ 0.37. Errors are also correlated, because a wrong conclusion in the context misleads later steps.',
  },
  {
    q: 'What makes some long tasks tractable despite p^n?',
    a: 'Checkable steps. If the agent can run tests or verify an answer and retry, each step’s effective success rate gets close to 1.',
  },
  {
    q: 'What is context engineering, and why is it needed?',
    a: 'Choosing what goes into the limited context at each step: trimmed tool results, on-demand loading, summaries, notes, sub-agents. Context is small, costly, and models degrade as it grows.',
  },
  {
    q: 'What is in-context learning, and what is the best evidence it is real learning?',
    a: 'Better performance from examples in the prompt, with no weight change. Toy transformers learn least-squares regression in context, and large models can follow deliberately flipped labels.',
  },
  {
    q: 'What does the random-labels result say about few-shot prompting?',
    a: 'On many classification tasks, randomising the example labels barely hurts. The examples mainly convey the format and answer set; the skill was already in the weights.',
  },
  {
    q: 'What are the two main theories of in-context learning?',
    a: 'Implicit Bayesian inference: examples help the model infer which known task it is doing. Attention as gradient descent: attention layers can run optimisation steps on the examples (shown for toy linear attention; contested for real LLMs).',
  },
  {
    q: 'Walk through a RAG pipeline.',
    a: 'Offline: chunk documents, embed each chunk, store vectors in an ANN index. Online: embed the query, fetch nearest chunks plus keyword (BM25) hits, rerank, and paste the best few into the prompt.',
  },
  {
    q: 'Why add BM25 keyword search to embedding search?',
    a: 'Embeddings capture paraphrases but miss exact strings such as error codes, IDs and rare names. Keyword search catches those; hybrid search merges both.',
  },
  {
    q: 'Why is a reranker more accurate than the first-stage vector search?',
    a: 'It reads the query and the candidate together and scores the pair, instead of comparing two vectors made separately. It is too slow to run on everything, so it runs only on the top candidates.',
  },
  {
    q: 'What retrieval failures can better embeddings not fix?',
    a: 'Queries that do not hint at the needed fact (you must know what to look for), multi-hop questions needing a second lookup, and facts split across chunk boundaries.',
  },
  {
    q: 'Why not just put everything in a 1M-token context instead of using RAG?',
    a: 'Cost: about 200 times more tokens per query, a KV cache in the hundreds of GB, and attention compute growing with the square of the length. Also, models use the middle of long contexts poorly and effective length is often far shorter than advertised.',
  },
  {
    q: 'What do CLAUDE.md files, ChatGPT memory, MemGPT and compaction have in common?',
    a: 'All write text somewhere and later put some of it back into the context. None change the weights, so memory holds put-into-words conclusions that must be recalled by a matching cue.',
  },
  {
    q: 'What is the strongest argument that external memory might be enough?',
    a: 'Humans also rely heavily on notes and documentation; huge contexts reduce the need to know what to look for; and agents that write and test their own skills store procedures, not just facts.',
  },
]
