<script setup lang="ts">
// The residual stream drawn as a shared bus. Each block reads the current vector and adds its output back.
const blocks = [
  { x: 120, label: 'Attention 1', up: true },
  { x: 240, label: 'MLP 1', up: false },
  { x: 360, label: 'Attention 2', up: true },
  { x: 480, label: 'MLP 2', up: false },
]
const busY = 150
</script>

<template>
  <figure class="figure">
    <svg viewBox="0 0 640 300" role="img" aria-label="A horizontal line labelled residual stream runs from the token embedding on the left to the next-token prediction on the right. Attention and MLP blocks sit above and below it. Each block reads from the line and adds its output back onto it at a plus sign.">
      <defs>
        <marker id="iars-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" class="arrowhead" />
        </marker>
      </defs>
      <rect x="10" y="128" width="78" height="44" rx="6" class="box-a" />
      <text x="49" y="147" text-anchor="middle" class="t">token</text>
      <text x="49" y="164" text-anchor="middle" class="t">vector</text>
      <rect x="552" y="128" width="78" height="44" rx="6" class="box-a" />
      <text x="591" y="147" text-anchor="middle" class="t">next-token</text>
      <text x="591" y="164" text-anchor="middle" class="t">guess</text>
      <line x1="88" :y1="busY" x2="550" :y2="busY" class="ln-b" style="stroke-width: 5" marker-end="url(#iars-ah)" />
      <text x="320" y="290" text-anchor="middle" class="m">residual stream: one vector per token, d numbers wide</text>
      <g v-for="b in blocks" :key="b.label">
        <rect :x="b.x - 50" :y="b.up ? 30 : 222" width="100" height="40" rx="6" class="box-c" />
        <text :x="b.x" :y="b.up ? 55 : 247" text-anchor="middle" class="tb">{{ b.label }}</text>
        <!-- read: from bus up/down into block -->
        <line :x1="b.x - 30" :y1="b.up ? busY - 4 : busY + 4" :x2="b.x - 30" :y2="b.up ? 72 : 220" class="ln" marker-end="url(#iars-ah)" />
        <text :x="b.x - 36" :y="b.up ? 112 : 196" text-anchor="end" class="m">read</text>
        <!-- write: from block back to a plus on the bus -->
        <line :x1="b.x + 30" :y1="b.up ? 72 : 220" :x2="b.x + 30" :y2="b.up ? busY - 12 : busY + 12" class="ln" marker-end="url(#iars-ah)" />
        <text :x="b.x + 36" :y="b.up ? 112 : 196" class="m">add</text>
        <circle :cx="b.x + 30" :cy="busY" r="9" class="box" />
        <text :x="b.x + 30" :y="busY + 5" text-anchor="middle" class="tb">+</text>
      </g>
    </svg>
    <figcaption>
      The residual stream as a shared bus. Each attention or MLP block reads the current vector, computes
      something, and adds its result back. Nothing is overwritten, so a later block can read what any earlier
      block wrote. Real models have dozens of layers, not two.
    </figcaption>
  </figure>
</template>
