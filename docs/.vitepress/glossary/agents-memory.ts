import type { GlossaryEntry } from '../glossary'

const chapter = '/beyond/agents-memory'

export const terms: Record<string, GlossaryEntry> = {
  'context-window': { term: 'Context window', def: 'The maximum number of tokens a model can read at once: prompt, conversation, tool results and its own output. Anything outside it does not exist for the model.', chapter },
  'tool-use': { term: 'Tool use (function calling)', def: 'The model writes a structured request, such as a function name and JSON arguments; outside code runs it and pastes the result back into the context.', chapter },
  harness: { term: 'Harness (agent scaffold)', def: 'The ordinary program around a model that builds its prompt, parses its tool calls, runs the tools, appends the results and decides when to stop.', chapter },
  agent: { term: 'Agent', def: 'A model run in a loop with tools: it reads the context, picks an action, sees the result, and repeats until the task is done.', chapter },
  react: { term: 'ReAct', def: 'Reasoning plus acting: a prompting pattern where the model alternates a written thought, a tool call, and the observed result.', chapter },
  mcp: { term: 'Model Context Protocol (MCP)', def: 'An open protocol, introduced by Anthropic in 2024, for exposing tools and data sources to any model harness in a standard way.', chapter },
  'context-engineering': { term: 'Context engineering', def: 'Deciding what goes into a model’s limited context at each step: instructions, tool results, retrieved text, summaries and notes.', chapter },
  'in-context-learning': { term: 'In-context learning (ICL)', def: 'A model doing a new task better because of examples or instructions in its prompt, with no change to its weights.', chapter },
  'few-shot': { term: 'Few-shot prompting', def: 'Putting a handful of worked examples of a task in the prompt before the real input. Zero-shot means no examples.', chapter },
  rag: { term: 'Retrieval-augmented generation (RAG)', def: 'Searching a document store for text relevant to the query and pasting it into the prompt before the model answers.', chapter },
  chunking: { term: 'Chunking', def: 'Splitting documents into pieces of a few hundred tokens before indexing them, so retrieval can return just the relevant part.', chapter },
  'embedding-model': { term: 'Embedding model', def: 'A network that maps a piece of text to one vector so that texts with similar meaning get vectors with high cosine similarity.', chapter },
  ann: { term: 'Approximate nearest neighbour (ANN) search', def: 'Finding vectors close to a query vector without comparing against all of them, trading a little recall for a large speedup.', chapter },
  hnsw: { term: 'HNSW', def: 'Hierarchical navigable small world: a layered graph index for approximate nearest neighbour search. Search walks greedily from coarse layers to fine ones.', chapter },
  bm25: { term: 'BM25', def: 'A classic keyword-ranking formula: scores documents by how often they contain the query words, weighting rare words more and long documents less.', chapter },
  'hybrid-search': { term: 'Hybrid search', def: 'Running keyword search (such as BM25) and vector search together and merging the results, to catch both exact terms and paraphrases.', chapter },
  reranker: { term: 'Reranker', def: 'A slower, more accurate model that reads the query and each candidate chunk together and rescores them, used on the top results of a fast search.', chapter },
  'lost-in-the-middle': { term: 'Lost in the middle', def: 'The finding that models use information at the start and end of a long context better than information in the middle.', chapter },
  'effective-context': { term: 'Effective context length', def: 'The longest input at which a model still does a task well, usually measured by benchmarks; often much shorter than the advertised context window.', chapter },
  compaction: { term: 'Compaction', def: 'Replacing an agent’s long history with a model-written summary when the context fills up, so the task can continue in a fresh context.', chapter },
  'memory-file': { term: 'Memory file', def: 'A plain text file of instructions or notes, such as CLAUDE.md, that a harness loads into the context at the start of every session.', chapter },
}
