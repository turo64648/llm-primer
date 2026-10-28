<script setup lang="ts">
const nodes = [
  { x: 30, y: 50, title: 'Node 1: layers 1–40', cls: 'box-a' },
  { x: 360, y: 50, title: 'Node 2: layers 41–80', cls: 'box-b' },
  { x: 30, y: 200, title: 'Node 3: layers 1–40', cls: 'box-a' },
  { x: 360, y: 200, title: 'Node 4: layers 41–80', cls: 'box-b' },
]
</script>

<template>
  <figure class="figure">
    <svg viewBox="0 0 640 340" role="img" aria-label="Four nodes of eight GPUs: tensor parallelism inside each node over NVLink, pipeline parallelism between nodes of one model copy, data parallelism between model copies, both over InfiniBand">
      <defs>
        <marker id="tas-topo-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" class="arrowhead" />
        </marker>
      </defs>
      <text x="320" y="36" text-anchor="middle" class="h">Model copy 1</text>
      <text x="320" y="190" text-anchor="middle" class="h">Model copy 2</text>
      <g v-for="(n, i) in nodes" :key="i">
        <rect :x="n.x" :y="n.y" width="250" height="100" rx="8" class="box" />
        <text :x="n.x + 125" :y="n.y + 22" text-anchor="middle" class="tb">{{ n.title }}</text>
        <rect v-for="g in 8" :key="g" :x="n.x + 12 + (g - 1) * 29" :y="n.y + 36" width="24" height="24" rx="3" :class="n.cls" />
        <text :x="n.x + 125" :y="n.y + 84" text-anchor="middle" class="m">8 GPUs, tensor parallel (NVLink)</text>
      </g>
      <path d="M282 100 L358 100" class="ln-c" marker-start="url(#tas-topo-ah)" marker-end="url(#tas-topo-ah)" />
      <text x="320" y="90" text-anchor="middle" class="m">pipeline</text>
      <path d="M282 250 L358 250" class="ln-c" marker-start="url(#tas-topo-ah)" marker-end="url(#tas-topo-ah)" />
      <text x="320" y="240" text-anchor="middle" class="m">pipeline</text>
      <path d="M155 152 L155 198" class="ln-a" marker-start="url(#tas-topo-ah)" marker-end="url(#tas-topo-ah)" />
      <text x="165" y="180" class="m">data parallel</text>
      <path d="M485 152 L485 198" class="ln-b" marker-start="url(#tas-topo-ah)" marker-end="url(#tas-topo-ah)" />
      <text x="495" y="180" class="m">data parallel</text>
      <text x="320" y="328" text-anchor="middle" class="m">Inside a node: NVLink, ~900 GB/s per GPU. Between nodes: InfiniBand, ~50 GB/s per GPU.</text>
    </svg>
    <figcaption>
      A 70B model on 32 GPUs: tensor parallel 8 × pipeline parallel 2 × data parallel 2. The chattiest split
      (tensor parallel, every layer) stays on the fast links inside a node. Pipeline and data parallel traffic
      crosses the slower network between nodes.
    </figcaption>
  </figure>
</template>
