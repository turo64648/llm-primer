<script setup lang="ts">
// One neuron (left) and a layer of neurons as one matrix multiply (right).
const ins = [70, 130, 190]
const outs = [55, 105, 155, 205]
</script>

<template>
  <figure class="figure">
    <svg viewBox="0 0 640 300" role="img" aria-label="Left: a neuron multiplies each input by a weight, adds them and a bias, then applies ReLU. Right: a layer of four neurons reading three inputs is one 4 by 3 weight matrix.">
      <defs>
        <marker id="nn-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" class="arrowhead" />
        </marker>
      </defs>

      <text x="150" y="24" text-anchor="middle" class="h">One neuron</text>
      <g v-for="(y, i) in ins" :key="'i' + i">
        <circle cx="30" :cy="y" r="16" class="box" />
        <text x="30" :y="y + 5" text-anchor="middle" class="t">x{{ i + 1 }}</text>
        <path :d="`M46 ${y} L132 130`" class="ln" />
        <text x="80" :y="y + (130 - y) * 0.45 - 6" text-anchor="middle" class="m">w{{ i + 1 }}</text>
      </g>
      <circle cx="150" cy="130" r="22" class="box-a" />
      <text x="150" y="136" text-anchor="middle" class="tb">Σ</text>
      <text x="150" y="176" text-anchor="middle" class="m">sum + bias b</text>
      <path d="M172 130 L198 130" class="ln" marker-end="url(#nn-ah)" />
      <rect x="200" y="112" width="58" height="36" rx="6" class="box-b" />
      <text x="229" y="136" text-anchor="middle" class="t">ReLU</text>
      <path d="M258 130 L288 130" class="ln" marker-end="url(#nn-ah)" />
      <text x="298" y="136" class="tb">h</text>
      <text x="150" y="232" text-anchor="middle" class="t mono">h = ReLU(w·x + b)</text>
      <text x="150" y="256" text-anchor="middle" class="m">one number out</text>

      <line x1="330" y1="40" x2="330" y2="270" class="ln-dash" />

      <text x="490" y="24" text-anchor="middle" class="h">A layer of 4 neurons</text>
      <g v-for="(yi, i) in ins" :key="'li' + i">
        <circle cx="380" :cy="yi" r="16" class="box" />
        <text x="380" :y="yi + 5" text-anchor="middle" class="t">x{{ i + 1 }}</text>
        <path v-for="(yo, j) in outs" :key="'e' + i + j" :d="`M396 ${yi} L494 ${yo}`" class="ln" />
      </g>
      <g v-for="(yo, j) in outs" :key="'o' + j">
        <circle cx="510" :cy="yo" r="16" class="box-a" />
        <text x="510" :y="yo + 5" text-anchor="middle" class="t">h{{ j + 1 }}</text>
      </g>
      <text x="490" y="250" text-anchor="middle" class="t mono">h = ReLU(W x + b)</text>
      <text x="490" y="274" text-anchor="middle" class="m">W is 4×3: one row of weights per neuron</text>
    </svg>
    <figcaption>
      A neuron multiplies each input by its own weight, adds the results and a bias, then applies an
      activation function such as ReLU. A layer is many neurons reading the same inputs. Stacking their weight
      rows gives a matrix, so the whole layer is one matrix multiply.
    </figcaption>
  </figure>
</template>
