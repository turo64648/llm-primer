<script setup lang="ts">
// Text -> tokens -> ids -> embedding rows -> model -> logits -> softmax probabilities.
const toks = [
  { s: 'The', id: '17', x: 200 },
  { s: '␣cat', id: '902', x: 330 },
  { s: '␣sat', id: '655', x: 460 },
]
const probs = [
  { s: '␣on', p: 0.42 },
  { s: '␣down', p: 0.21 },
  { s: '.', p: 0.12 },
  { s: '␣in', p: 0.09 },
  { s: 'other', p: 0.16 },
]
const cells = [0.8, -0.3, 0.5, -0.9, 0.2]
</script>

<template>
  <figure class="figure">
    <svg viewBox="0 0 640 470" role="img" aria-label="A pipeline from top to bottom. The text The cat sat is cut into three tokens. Each token becomes an id number. Each id selects one row of the embedding table, a short list of numbers. The model turns the vectors into one score per vocabulary entry, and softmax turns the scores into probabilities for the next token, highest for on.">
      <defs>
        <marker id="tok-pipe-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" class="arrowhead" />
        </marker>
      </defs>
      <text x="20" y="38" class="tb">Text</text>
      <rect x="200" y="16" width="320" height="34" rx="6" class="box" />
      <text x="360" y="38" text-anchor="middle" class="t mono">"The cat sat"</text>

      <text x="20" y="100" class="tb">Tokens</text>
      <text x="20" y="160" class="tb">Token ids</text>
      <text x="20" y="232" class="tb">Embeddings</text>
      <text x="20" y="250" class="m">one table row each</text>
      <g v-for="t in toks" :key="t.s">
        <path :d="`M${t.x + 50} 52 L${t.x + 50} 76`" class="ln" marker-end="url(#tok-pipe-ah)" />
        <rect :x="t.x" y="80" width="100" height="32" rx="6" class="box-a" />
        <text :x="t.x + 50" y="101" text-anchor="middle" class="t mono">{{ t.s }}</text>
        <path :d="`M${t.x + 50} 114 L${t.x + 50} 136`" class="ln" marker-end="url(#tok-pipe-ah)" />
        <rect :x="t.x" y="140" width="100" height="32" rx="6" class="box-b" />
        <text :x="t.x + 50" y="161" text-anchor="middle" class="t mono">{{ t.id }}</text>
        <path :d="`M${t.x + 50} 174 L${t.x + 50} 196`" class="ln" marker-end="url(#tok-pipe-ah)" />
        <rect v-for="(c, i) in cells" :key="i" :x="t.x + 10 + i * 16" y="200" width="16" height="44"
              :class="(i + t.s.length) % 2 ? 'box-c' : 'box'" />
        <path :d="`M${t.x + 50} 246 L${t.x + 50} 270`" class="ln" marker-end="url(#tok-pipe-ah)" />
      </g>

      <rect x="200" y="274" width="360" height="40" rx="6" class="box-d" />
      <text x="380" y="299" text-anchor="middle" class="tb">the model (Chapter 4)</text>
      <path d="M380 316 L380 340" class="ln" marker-end="url(#tok-pipe-ah)" />

      <text x="20" y="380" class="tb">Next-token</text>
      <text x="20" y="398" class="tb">probabilities</text>
      <text x="20" y="416" class="m">softmax of the logits</text>
      <g v-for="(q, i) in probs" :key="q.s">
        <text :x="248" :y="362 + i * 22" text-anchor="end" class="t mono">{{ q.s }}</text>
        <rect :x="258" :y="348 + i * 22" :width="q.p * 600" height="17" class="box-a" />
        <text :x="266 + q.p * 600" :y="362 + i * 22" class="m">{{ q.p.toFixed(2) }}</text>
      </g>
      <text x="320" y="464" text-anchor="middle" class="m">ids are made up; ␣ marks a space that belongs to the token</text>
    </svg>
    <figcaption>
      From text to a prediction. The tokenizer cuts the text into tokens and maps each to an id. Each id picks
      one row of the embedding table. The model turns these vectors into one score (logit) per vocabulary
      entry, and softmax turns the scores into probabilities for the next token.
    </figcaption>
  </figure>
</template>
