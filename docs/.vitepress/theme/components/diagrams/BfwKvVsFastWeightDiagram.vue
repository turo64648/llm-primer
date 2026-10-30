<script setup lang="ts">
// KV cache (grows per token) vs fast-weight matrix memory (fixed size, written by outer products).
</script>

<template>
  <figure class="figure">
    <svg viewBox="0 0 640 330" role="img" aria-label="Left: softmax attention keeps every key and value in a cache that grows by one row per token, and a query compares against all rows. Right: linear attention or a fast-weight memory keeps one fixed-size matrix; each token writes by adding value times key transposed, and a query reads by multiplying the matrix by the query.">
      <defs>
        <marker id="bfw-kv-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" class="arrowhead" />
        </marker>
      </defs>

      <text x="160" y="22" text-anchor="middle" class="h">Attention: KV cache</text>
      <text x="480" y="22" text-anchor="middle" class="h">Fast weights: one matrix</text>

      <rect x="40" y="40" width="240" height="26" rx="4" class="box-a" />
      <text x="160" y="58" text-anchor="middle" class="t">token 1: key, value</text>
      <rect x="40" y="70" width="240" height="26" rx="4" class="box-a" />
      <text x="160" y="88" text-anchor="middle" class="t">token 2: key, value</text>
      <rect x="40" y="100" width="240" height="26" rx="4" class="box-a" />
      <text x="160" y="118" text-anchor="middle" class="t">token 3: key, value</text>
      <text x="160" y="146" text-anchor="middle" class="m">⋮</text>
      <rect x="40" y="156" width="240" height="26" rx="4" class="box-a" />
      <text x="160" y="174" text-anchor="middle" class="t">token t: key, value</text>
      <text x="160" y="202" text-anchor="middle" class="m">one new row per token: grows forever</text>

      <rect x="40" y="226" width="240" height="40" rx="8" class="box" />
      <text x="160" y="251" text-anchor="middle" class="t">query compares with every row</text>
      <text x="160" y="292" text-anchor="middle" class="m">exact recall; memory and time grow with t</text>

      <rect x="390" y="60" width="180" height="120" rx="6" class="box-b" />
      <text x="480" y="115" text-anchor="middle" class="tb">matrix M</text>
      <text x="480" y="137" text-anchor="middle" class="m">d × d numbers, always</text>

      <text x="330" y="88" text-anchor="middle" class="t">write</text>
      <text x="330" y="106" text-anchor="middle" class="m">M += v kᵀ</text>
      <line x1="300" y1="118" x2="386" y2="118" class="ln" marker-end="url(#bfw-kv-ah)" />

      <line x1="480" y1="182" x2="480" y2="224" class="ln" marker-end="url(#bfw-kv-ah)" />
      <rect x="390" y="226" width="180" height="40" rx="8" class="box" />
      <text x="480" y="251" text-anchor="middle" class="t">read: output = M q</text>
      <text x="480" y="292" text-anchor="middle" class="m">constant cost per token;</text>
      <text x="480" y="310" text-anchor="middle" class="m">pairs blur together as it fills</text>
    </svg>
    <figcaption>Softmax attention keeps everything and searches it. A fast-weight memory squeezes everything into one fixed-size matrix: cheap per token, but it can only hold about as many clean pairs as its width.</figcaption>
  </figure>
</template>
