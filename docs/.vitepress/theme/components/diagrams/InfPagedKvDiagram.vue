<script setup lang="ts">
// Two requests' KV caches stored in fixed-size blocks; a per-request block table maps logical to physical blocks.
const seqs = [
  { name: 'Request A', y: 60, blocks: [{ l: 'A0', p: 6 }, { l: 'A1', p: 1 }, { l: 'A2', p: 3 }], cls: 'box-a' },
  { name: 'Request B', y: 190, blocks: [{ l: 'B0', p: 6 }, { l: 'B1', p: 4 }], cls: 'box-b' },
]
// physical pool: 8 blocks in a 2 x 4 grid
const phys = (i: number) => ({ x: 420 + (i % 4) * 52, y: 80 + Math.floor(i / 4) * 110 })
const owner: Record<number, string> = { 6: 'box-c', 1: 'box-a', 3: 'box-a', 4: 'box-b' }
</script>

<template>
  <figure class="figure">
    <svg viewBox="0 0 640 300" role="img" aria-label="Paged KV cache: each request's blocks map through a block table to scattered physical blocks; both requests share block 6, which holds the same prompt prefix.">
      <defs>
        <marker id="ipk-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" class="arrowhead" />
        </marker>
      </defs>
      <text x="20" y="30" class="h">Logical blocks (16 tokens each)</text>
      <text x="420" y="30" class="h">GPU memory</text>
      <g v-for="s in seqs" :key="s.name">
        <text x="20" :y="s.y - 10" class="tb">{{ s.name }}</text>
        <g v-for="(b, i) in s.blocks" :key="b.l">
          <rect :x="20 + i * 64" :y="s.y" width="56" height="40" rx="4" :class="b.l === 'A0' || b.l === 'B0' ? 'box-c' : s.cls" />
          <text :x="48 + i * 64" :y="s.y + 25" text-anchor="middle" class="t">{{ b.l }}</text>
          <line :x1="48 + i * 64" :y1="s.y + 40" :x2="phys(b.p).x + 24" :y2="phys(b.p).y + (phys(b.p).y > s.y + 40 ? 0 : 44)" class="ln-dash" marker-end="url(#ipk-ah)" />
        </g>
      </g>
      <g v-for="i in 8" :key="'p' + i">
        <rect :x="phys(i - 1).x" :y="phys(i - 1).y" width="48" height="44" rx="4" :class="owner[i - 1] || 'ghost'" />
        <text :x="phys(i - 1).x + 24" :y="phys(i - 1).y + 27" text-anchor="middle" class="m">{{ i - 1 }}</text>
      </g>
      <text x="420" y="260" class="m">dashed = free block</text>
      <text x="20" y="285" class="m">A0 and B0 hold the same prompt prefix, so both point at block 6 (shared, reference-counted).</text>
    </svg>
    <figcaption>
      PagedAttention stores each request's KV cache in small fixed-size blocks anywhere in GPU memory. A block
      table per request maps "my block 2" to "physical block 3", like a page table. Requests that share a prompt
      prefix can point at the same physical blocks.
    </figcaption>
  </figure>
</template>
