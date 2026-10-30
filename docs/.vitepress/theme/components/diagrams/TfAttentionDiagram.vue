<script setup lang="ts">
// Causal attention matrix for a 5-token sentence, plus how one row is computed.
const toks = ['The', '␣cat', '␣sat', '␣on', '␣the']
// Illustrative weights (rows sum to 1); row i only covers tokens 0..i.
const W = [
  [1.0],
  [0.3, 0.7],
  [0.1, 0.6, 0.3],
  [0.05, 0.25, 0.6, 0.1],
  [0.05, 0.15, 0.2, 0.5, 0.1],
]
const x0 = 110
const y0 = 80
const c = 46
</script>

<template>
  <figure class="figure">
    <svg viewBox="0 0 640 400" role="img" aria-label="A five by five grid. Rows are the token doing the looking, columns are the token being looked at. Only cells on or below the diagonal have weights; cells above the diagonal are masked because they are future tokens. Each row sums to one. On the right, the steps for one row: compare the query with every earlier key, scale and softmax the scores into weights, and take the weighted sum of the values.">
      <text x="225" y="24" text-anchor="middle" class="tb">attention weights</text>
      <text x="225" y="44" text-anchor="middle" class="m">columns: the token looked at (key)</text>
      <text v-for="(t, j) in toks" :key="'c' + j" :x="x0 + j * c + c / 2" :y="70" text-anchor="middle" class="t mono">{{ t }}</text>
      <text v-for="(t, i) in toks" :key="'r' + i" :x="x0 - 8" :y="y0 + i * c + c / 2 + 5" text-anchor="end" class="t mono">{{ t }}</text>
      <text x="16" y="200" class="m" transform="rotate(-90 16 200)" text-anchor="middle">rows: the token looking (query)</text>
      <g v-for="(row, i) in toks" :key="'g' + i">
        <g v-for="(col, j) in toks" :key="'g' + i + '-' + j">
          <rect v-if="j <= i" :x="x0 + j * c" :y="y0 + i * c" :width="c" :height="c" class="box-a"
                :style="{ fillOpacity: 0.15 + 0.85 * W[i][j] }" />
          <rect v-else :x="x0 + j * c" :y="y0 + i * c" :width="c" :height="c" class="box" style="opacity: 0.35" />
          <text :x="x0 + j * c + c / 2" :y="y0 + i * c + c / 2 + 5" text-anchor="middle" class="m">{{ j <= i ? W[i][j].toFixed(2) : '—' }}</text>
        </g>
      </g>
      <text x="225" y="330" text-anchor="middle" class="m">— = masked (a future token): weight 0</text>
      <text x="225" y="348" text-anchor="middle" class="m">each row sums to 1</text>

      <text x="500" y="24" text-anchor="middle" class="tb">one row (query "␣sat")</text>
      <rect x="390" y="44" width="220" height="56" rx="6" class="box-b" />
      <text x="500" y="66" text-anchor="middle" class="t">score = q · k for each</text>
      <text x="500" y="86" text-anchor="middle" class="t">earlier token, ÷ √d</text>
      <path d="M500 102 L500 122" class="ln" />
      <rect x="390" y="124" width="220" height="56" rx="6" class="box-c" />
      <text x="500" y="146" text-anchor="middle" class="t">softmax: scores →</text>
      <text x="500" y="166" text-anchor="middle" class="t">weights 0.1, 0.6, 0.3</text>
      <path d="M500 182 L500 202" class="ln" />
      <rect x="390" y="204" width="220" height="56" rx="6" class="box-d" />
      <text x="500" y="226" text-anchor="middle" class="t">output = 0.1·v(The)</text>
      <text x="500" y="246" text-anchor="middle" class="t">+ 0.6·v(cat) + 0.3·v(sat)</text>
      <text x="500" y="290" text-anchor="middle" class="m">weights are made up</text>
      <text x="500" y="308" text-anchor="middle" class="m">for illustration</text>
    </svg>
    <figcaption>
      The attention matrix for one head. Row i shows how token i splits its attention over itself and the
      tokens before it. The causal mask blanks everything above the diagonal, so no token sees the future. On
      the right, how one row is computed from queries (q), keys (k) and values (v).
    </figcaption>
  </figure>
</template>
