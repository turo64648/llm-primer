<script setup lang="ts">
// Two models, true accuracy 70% and 72%, measured on 200 vs 5,000 questions. Bars are 95% intervals (±1.96 SE).
const x = (pct: number) => 60 + (pct - 60) * 26 // 60%..82% maps to 60..632
const se = (p: number, n: number) => 196 * Math.sqrt((p * (1 - p)) / n) // in percentage points
const rows = [
  { label: 'Model A, 200 questions', p: 0.7, n: 200, y: 80, cls: 'ln-a', dot: 'box-a' },
  { label: 'Model B, 200 questions', p: 0.72, n: 200, y: 125, cls: 'ln-b', dot: 'box-b' },
  { label: 'Model A, 5,000 questions', p: 0.7, n: 5000, y: 205, cls: 'ln-a', dot: 'box-a' },
  { label: 'Model B, 5,000 questions', p: 0.72, n: 5000, y: 250, cls: 'ln-b', dot: 'box-b' },
]
const ticks = [60, 65, 70, 75, 80]
</script>

<template>
  <figure class="figure">
    <svg viewBox="0 0 640 300" role="img" aria-label="Error bars for two models two points apart. On 200 questions the 95 percent intervals overlap almost completely; on 5,000 questions they barely overlap.">
      <text x="20" y="30" class="h">Same two models, same 2-point gap, two benchmark sizes</text>
      <g v-for="r in rows" :key="r.label">
        <text :x="x(100 * r.p)" :y="r.y - 12" text-anchor="middle" class="m">{{ r.label }}</text>
        <line :x1="x(100 * r.p - se(r.p, r.n))" :y1="r.y" :x2="x(100 * r.p + se(r.p, r.n))" :y2="r.y" :class="r.cls" stroke-width="3" />
        <line :x1="x(100 * r.p - se(r.p, r.n))" :y1="r.y - 6" :x2="x(100 * r.p - se(r.p, r.n))" :y2="r.y + 6" :class="r.cls" />
        <line :x1="x(100 * r.p + se(r.p, r.n))" :y1="r.y - 6" :x2="x(100 * r.p + se(r.p, r.n))" :y2="r.y + 6" :class="r.cls" />
        <circle :cx="x(100 * r.p)" :cy="r.y" r="5" :class="r.dot" />
      </g>
      <line x1="60" y1="275" x2="632" y2="275" class="ln" />
      <g v-for="t in ticks" :key="t">
        <line :x1="x(t)" y1="271" :x2="x(t)" y2="279" class="ln" />
        <text :x="x(t)" y="295" text-anchor="middle" class="m">{{ t }}%</text>
      </g>
      <line x1="20" y1="160" x2="620" y2="160" class="ln-dash" />
    </svg>
    <figcaption>
      Each bar is a 95% interval: the range the measured score lands in 95% of the time. On 200 questions the
      intervals are about ±6 points wide and overlap almost entirely, so a 2-point gap tells you little. On 5,000
      questions they shrink to about ±1.3 points and the gap starts to show.
    </figcaption>
  </figure>
</template>
