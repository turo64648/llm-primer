<script setup lang="ts">
// Roofline for one H100 (bf16 dense): ~989 TFLOP/s peak, ~3.35 TB/s memory bandwidth. Log-log axes.
const PEAK = 989
const BW = 3.35
const px = (x: number) => 80 + 130 * Math.log10(x)
const py = (y: number) => 250 - 70 * Math.log10(y)
const ridge = PEAK / BW
const roof = (x: number) => Math.min(BW * x, PEAK)
const xs = [1, 10, 100, 1000, 10000]
const ys = [1, 10, 100, 1000]
const pts = [
  { x: 1, label: 'decode, batch 1', dx: 10, dy: 22, cls: 'box-b' },
  { x: 64, label: 'decode, batch 64', dx: -150, dy: -6, cls: 'box-b' },
  { x: 2000, label: 'prefill, 2K-token prompt', dx: -150, dy: 28, cls: 'box-a' },
]
</script>

<template>
  <figure class="figure">
    <svg viewBox="0 0 640 310" role="img" aria-label="Roofline: performance rises with arithmetic intensity until about 295 FLOPs per byte, then is flat at peak compute. Decode sits low on the slanted memory-bound part, prefill sits on the flat compute-bound part.">
      <line :x1="px(1)" :y1="py(1)" :x2="px(10000)" :y2="py(1)" class="ln" />
      <line :x1="px(1)" :y1="py(1)" :x2="px(1)" :y2="py(1000) - 10" class="ln" />
      <g v-for="x in xs" :key="'x' + x">
        <line :x1="px(x)" :y1="py(1)" :x2="px(x)" :y2="py(1) + 5" class="ln" />
        <text :x="px(x)" :y="py(1) + 20" text-anchor="middle" class="m">{{ x.toLocaleString('en-US') }}</text>
      </g>
      <g v-for="y in ys" :key="'y' + y">
        <text :x="px(1) - 8" :y="py(y) + 4" text-anchor="end" class="m">{{ y }}</text>
      </g>
      <polyline :points="`${px(1)},${py(roof(1))} ${px(ridge)},${py(PEAK)} ${px(10000)},${py(PEAK)}`" class="ln-a" />
      <line :x1="px(ridge)" :y1="py(PEAK)" :x2="px(ridge)" :y2="py(1)" class="ln-dash" />
      <text :x="px(ridge) + 6" :y="py(3)" class="m">ridge ≈ 295</text>
      <text :x="px(3)" :y="py(60)" class="m">memory-bound:</text>
      <text :x="px(3)" :y="py(60) + 16" class="m">limited by bandwidth</text>
      <text x="634" :y="py(PEAK) - 8" text-anchor="end" class="m">compute-bound: limited by peak FLOP/s</text>
      <g v-for="p in pts" :key="p.label">
        <circle :cx="px(p.x)" :cy="py(roof(p.x))" r="6" :class="p.cls" />
        <text :x="px(p.x) + p.dx" :y="py(roof(p.x)) + p.dy" class="t">{{ p.label }}</text>
      </g>
      <text x="340" y="296" text-anchor="middle" class="m">arithmetic intensity: FLOPs per byte read from memory (log scale)</text>
      <text x="14" y="22" class="m">TFLOP/s (log)</text>
    </svg>
    <figcaption>
      A roofline for one H100 GPU in 16-bit math. Below about 295 FLOPs per byte, the chip waits on memory;
      above it, on arithmetic. Generating one token for one user does about 1 FLOP per byte of weights, so it
      uses well under 1% of the arithmetic. Batching moves decode up the slope; a long prompt is already on the flat part.
    </figcaption>
  </figure>
</template>
