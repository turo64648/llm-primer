<script setup lang="ts">
// GPipe-style schedule: 4 stages, 4 micro-batches, forward and backward each take one slot.
const P = 4
const M = 4
const x0 = 92
const w = 38
const rowY = (s: number) => 50 + s * 46
const cells: { s: number; t: number; label: string; kind: 'f' | 'b' }[] = []
for (let s = 0; s < P; s++) {
  for (let m = 0; m < M; m++) {
    cells.push({ s, t: s + m, label: `F${m + 1}`, kind: 'f' })
    cells.push({ s, t: 2 * P - 1 + (P - 1 - s) + m, label: `B${M - m}`, kind: 'b' })
  }
}
const total = 2 * (M + P - 1)
</script>

<template>
  <figure class="figure">
    <svg viewBox="0 0 640 290" role="img" aria-label="Pipeline schedule with 4 stages and 4 micro-batches: each stage is idle at the start and the end, which is the pipeline bubble">
      <text v-for="s in P" :key="'l' + s" :x="10" :y="rowY(s - 1) + 24" class="tb">GPU {{ s }}</text>
      <rect v-for="s in P" :key="'r' + s" :x="x0" :y="rowY(s - 1)" :width="total * w" height="36" rx="4" class="ghost" />
      <g v-for="c in cells" :key="c.s + '-' + c.t">
        <rect :x="x0 + c.t * w + 1" :y="rowY(c.s)" :width="w - 2" height="36" rx="4" :class="c.kind === 'f' ? 'box-a' : 'box-b'" />
        <text :x="x0 + c.t * w + w / 2" :y="rowY(c.s) + 23" text-anchor="middle" class="t">{{ c.label }}</text>
      </g>
      <line :x1="x0" y1="240" :x2="x0 + total * w" y2="240" class="ln" />
      <text :x="x0" y="258" class="m">time →</text>
      <text x="320" y="282" text-anchor="middle" class="m">F = forward, B = backward of one micro-batch. Dashed gaps are idle time: the bubble.</text>
    </svg>
    <figcaption>
      A naive (GPipe-style) pipeline: 4 stages, 4 micro-batches, with backward drawn the same width as forward
      for simplicity. Each GPU is busy 8 of 14 slots; the idle 6 are the bubble, (4−1)/(4+4−1) = 3/7 of the time.
      More micro-batches shrink the bubble.
    </figcaption>
  </figure>
</template>
