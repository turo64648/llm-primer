<script setup lang="ts">
// Gradient descent on L(w) = w², from w = −1.6, with three learning rates.
// Each step: w ← w − lr · 2w.
const panels = [
  { title: 'Too small (0.05)', lr: 0.05, steps: 6, cx: 110 },
  { title: 'About right (0.4)', lr: 0.4, steps: 3, cx: 320 },
  { title: 'Too large (1.05)', lr: 1.05, steps: 3, cx: 530 },
]
const sx = 45, sy = 30, base = 210
const px = (cx: number, w: number) => cx + w * sx
const py = (w: number) => base - w * w * sy
function curve(cx: number) {
  const pts: string[] = []
  for (let w = -2.1; w <= 2.101; w += 0.05) pts.push(`${px(cx, w).toFixed(1)},${py(w).toFixed(1)}`)
  return 'M' + pts.join(' L')
}
function path(lr: number, n: number) {
  const ws = [-1.6]
  for (let i = 0; i < n; i++) ws.push(ws[ws.length - 1] * (1 - 2 * lr))
  return ws
}
</script>

<template>
  <figure class="figure">
    <svg viewBox="0 0 640 270" role="img" aria-label="Three copies of a bowl-shaped loss curve. With a small learning rate the steps creep down slowly. With a good one they reach the bottom in a few steps. With a too large one they jump from side to side and climb higher each time.">
      <g v-for="p in panels" :key="p.title">
        <text :x="p.cx" y="24" text-anchor="middle" class="tb">{{ p.title }}</text>
        <path :d="curve(p.cx)" class="ln" />
        <line :x1="p.cx - 100" :y1="base + 1" :x2="p.cx + 100" :y2="base + 1" class="ln-dash" />
        <polyline :points="path(p.lr, p.steps).map((w) => `${px(p.cx, w)},${py(w)}`).join(' ')" class="ln-b" />
        <circle v-for="(w, i) in path(p.lr, p.steps)" :key="i" :cx="px(p.cx, w)" :cy="py(w)" r="4.5" :class="i === 0 ? 'box-a' : 'box-b'" />
        <text :x="p.cx" :y="base + 25" text-anchor="middle" class="m">weight w</text>
      </g>
      <text x="320" y="258" text-anchor="middle" class="m">height = loss; first dot = starting weight; each later dot = one step</text>
    </svg>
    <figcaption>
      Gradient descent on a one-weight loss shaped like a bowl, from the same start with three learning rates.
      Too small crawls. About right reaches the bottom in a few steps. Too large overshoots further each time
      and the loss grows: training diverges.
    </figcaption>
  </figure>
</template>
