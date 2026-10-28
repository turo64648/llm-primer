<script setup lang="ts">
// Left: 2 features in 2 dimensions, at right angles. Right: 5 features in 2 dimensions, as a pentagon.
const r = 85
const left = { cx: 160, cy: 145 }
const right = { cx: 480, cy: 145 }
const five = Array.from({ length: 5 }, (_, i) => {
  const a = -Math.PI / 2 + (2 * Math.PI * i) / 5
  return { x: right.cx + r * Math.cos(a), y: right.cy + r * Math.sin(a), lx: right.cx + (r + 18) * Math.cos(a), ly: right.cy + (r + 18) * Math.sin(a) + 5, n: i + 1 }
})
</script>

<template>
  <figure class="figure">
    <svg viewBox="0 0 640 280" role="img" aria-label="Left panel: two arrows at right angles, one per feature, with no overlap. Right panel: five arrows spread evenly around a circle in the same two dimensions, like a pentagon; each pair overlaps a little.">
      <defs>
        <marker id="iasp-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" class="arrowhead" />
        </marker>
      </defs>
      <text :x="left.cx" y="26" text-anchor="middle" class="tb">2 features, 2 dimensions</text>
      <line :x1="left.cx" :y1="left.cy" :x2="left.cx + r" :y2="left.cy" class="ln-a" marker-end="url(#iasp-ah)" />
      <line :x1="left.cx" :y1="left.cy" :x2="left.cx" :y2="left.cy - r" class="ln-a" marker-end="url(#iasp-ah)" />
      <text :x="left.cx + r + 8" :y="left.cy + 5" class="t">1</text>
      <text :x="left.cx - 4" :y="left.cy - r - 8" class="t">2</text>
      <text :x="left.cx" y="262" text-anchor="middle" class="m">no overlap: reading one never picks up the other</text>
      <text :x="right.cx" y="26" text-anchor="middle" class="tb">5 features, 2 dimensions</text>
      <g v-for="p in five" :key="p.n">
        <line :x1="right.cx" :y1="right.cy" :x2="p.x" :y2="p.y" class="ln-b" marker-end="url(#iasp-ah)" />
        <text :x="p.lx" :y="p.ly" text-anchor="middle" class="t">{{ p.n }}</text>
      </g>
      <text :x="right.cx" y="262" text-anchor="middle" class="m">some overlap: fine if features are rarely active together</text>
    </svg>
    <figcaption>
      Superposition. With as many dimensions as features, each feature gets its own direction at right angles
      to the rest. With more features than dimensions, a network can still store them as directions that
      overlap slightly. Reading one feature then picks up a little of the others, which only hurts when several
      are active at once.
    </figcaption>
  </figure>
</template>
