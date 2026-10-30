<script setup lang="ts">
// RAG pipeline: offline indexing, online retrieval, rerank, generate.
</script>

<template>
  <figure class="figure">
    <svg viewBox="0 0 640 330" role="img" aria-label="Offline, documents are split into chunks, each chunk is turned into a vector by an embedding model, and the vectors go into a vector index. Online, the query is embedded, the index returns the nearest chunks, a keyword search returns more, a reranker picks the best few, and they are pasted into the prompt for the model to answer.">
      <defs>
        <marker id="agr-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" class="arrowhead" />
        </marker>
      </defs>

      <text x="20" y="22" class="h">Offline: build the index</text>
      <rect x="20" y="34" width="130" height="50" rx="8" class="box" />
      <text x="85" y="64" text-anchor="middle" class="t">Documents</text>
      <rect x="175" y="34" width="130" height="50" rx="8" class="box-a" />
      <text x="240" y="64" text-anchor="middle" class="t">Split into chunks</text>
      <rect x="330" y="34" width="130" height="50" rx="8" class="box-a" />
      <text x="395" y="56" text-anchor="middle" class="t">Embedding model</text>
      <text x="395" y="74" text-anchor="middle" class="m">chunk → vector</text>
      <rect x="485" y="34" width="135" height="50" rx="8" class="box-b" />
      <text x="552" y="56" text-anchor="middle" class="t">Vector index</text>
      <text x="552" y="74" text-anchor="middle" class="m">e.g. HNSW graph</text>
      <line x1="150" y1="59" x2="173" y2="59" class="ln" marker-end="url(#agr-ah)" />
      <line x1="305" y1="59" x2="328" y2="59" class="ln" marker-end="url(#agr-ah)" />
      <line x1="460" y1="59" x2="483" y2="59" class="ln" marker-end="url(#agr-ah)" />

      <text x="20" y="128" class="h">Online: answer a query</text>
      <rect x="20" y="140" width="130" height="50" rx="8" class="box" />
      <text x="85" y="170" text-anchor="middle" class="t">User query</text>
      <rect x="175" y="140" width="130" height="50" rx="8" class="box-a" />
      <text x="240" y="162" text-anchor="middle" class="t">Embed query</text>
      <text x="240" y="180" text-anchor="middle" class="m">same model</text>
      <rect x="330" y="140" width="290" height="50" rx="8" class="box-b" />
      <text x="475" y="162" text-anchor="middle" class="t">Nearest-neighbour search: top ~50</text>
      <text x="475" y="180" text-anchor="middle" class="m">plus keyword search (BM25), merged</text>
      <line x1="150" y1="165" x2="173" y2="165" class="ln" marker-end="url(#agr-ah)" />
      <line x1="305" y1="165" x2="328" y2="165" class="ln" marker-end="url(#agr-ah)" />
      <line x1="552" y1="84" x2="552" y2="138" class="ln" marker-end="url(#agr-ah)" />

      <rect x="330" y="230" width="290" height="50" rx="8" class="box-c" />
      <text x="475" y="252" text-anchor="middle" class="t">Reranker: score each (query, chunk)</text>
      <text x="475" y="270" text-anchor="middle" class="m">keep the best ~5</text>
      <rect x="20" y="230" width="285" height="50" rx="8" class="box-d" />
      <text x="162" y="252" text-anchor="middle" class="t">LLM: prompt = chunks + query</text>
      <text x="162" y="270" text-anchor="middle" class="m">answer, ideally citing chunks</text>
      <line x1="475" y1="190" x2="475" y2="228" class="ln" marker-end="url(#agr-ah)" />
      <line x1="330" y1="255" x2="307" y2="255" class="ln" marker-end="url(#agr-ah)" />
      <text x="20" y="314" class="m">Each arrow can lose the answer: bad split, bad vector, missed neighbour, wrong cut.</text>
    </svg>
    <figcaption>A typical retrieval-augmented generation pipeline. The model's weights never change; retrieval only decides which text lands in the prompt.</figcaption>
  </figure>
</template>
