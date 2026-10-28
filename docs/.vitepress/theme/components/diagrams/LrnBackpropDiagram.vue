<script setup lang="ts">
// Forward and backward pass over the graph of L = (w·x + b − t)², with w=3, x=2, b=1, t=5.
const nodes = [
  { x: 40, y: 70, label: 'w', val: '3', grad: '8', cls: 'box-a' },
  { x: 40, y: 150, label: 'x', val: '2', grad: '', cls: 'box' },
  { x: 160, y: 110, label: '×', val: 'u = 6', grad: '4', cls: 'box' },
  { x: 160, y: 200, label: 'b', val: '1', grad: '4', cls: 'box-a' },
  { x: 290, y: 155, label: '+', val: 'z = 7', grad: '4', cls: 'box' },
  { x: 290, y: 245, label: 't', val: '5', grad: '', cls: 'box' },
  { x: 420, y: 200, label: '−', val: 'd = 2', grad: '4', cls: 'box' },
  { x: 550, y: 200, label: '( )²', val: 'L = 4', grad: '1', cls: 'box-b' },
]
const edges = [[0, 2], [1, 2], [2, 4], [3, 4], [4, 6], [5, 6], [6, 7]]
const r = 22
function seg(a: number, b: number) {
  const A = nodes[a], B = nodes[b]
  const dx = B.x - A.x, dy = B.y - A.y, n = Math.hypot(dx, dy)
  return `M${A.x + (dx / n) * r} ${A.y + (dy / n) * r} L${B.x - (dx / n) * (r + 3)} ${B.y - (dy / n) * (r + 3)}`
}
</script>

<template>
  <figure class="figure">
    <svg viewBox="0 0 640 320" role="img" aria-label="A computation graph for the loss (w times x plus b minus t) squared. Forward values flow left to right; gradients flow right to left. The gradient for w is 8 and for b is 4.">
      <defs>
        <marker id="lrn-bp-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" class="arrowhead" />
        </marker>
      </defs>
      <path v-for="(e, i) in edges" :key="'e' + i" :d="seg(e[0], e[1])" class="ln" marker-end="url(#lrn-bp-ah)" />
      <g v-for="(n, i) in nodes" :key="'n' + i">
        <circle :cx="n.x" :cy="n.y" :r="r" :class="n.cls" />
        <text :x="n.x" :y="n.y + 5" text-anchor="middle" class="tb">{{ n.label }}</text>
        <text :x="n.x" :y="n.y - 30" text-anchor="middle" class="t mono">{{ n.val }}</text>
        <text v-if="n.grad" :x="n.x" :y="n.y + 42" text-anchor="middle" class="m mono">grad {{ n.grad }}</text>
      </g>
      <path d="M560 40 L380 40" class="ln-b" marker-end="url(#lrn-bp-ah)" />
      <text x="470" y="30" text-anchor="middle" class="m">backward: gradients, right to left</text>
      <path d="M40 300 L230 300" class="ln" marker-end="url(#lrn-bp-ah)" />
      <text x="135" y="290" text-anchor="middle" class="m">forward: values, left to right</text>
    </svg>
    <figcaption>
      The loss L = (w·x + b − t)² as a graph of small steps. The forward pass computes each value (top
      labels) and keeps it. The backward pass starts with 1 at the loss and multiplies by each step's local
      derivative on the way back (bottom labels). It ends with the gradient for every parameter: 8 for w and 4
      for b.
    </figcaption>
  </figure>
</template>
