<script setup lang="ts">
// Matrix multiply as many dot products: one row of A times one column of B gives one cell of C.
</script>

<template>
  <figure class="figure">
    <svg viewBox="0 0 640 260" role="img" aria-label="A is m by k, B is k by n, C is m by n. Row i of A dotted with column j of B gives cell i, j of C.">
      <!-- A: m x k -->
      <rect x="20" y="50" width="180" height="120" rx="4" class="box" />
      <rect x="20" y="90" width="180" height="24" class="box-a" />
      <text x="110" y="36" text-anchor="middle" class="tb">A (m × k)</text>
      <text x="110" y="107" text-anchor="middle" class="m">row i</text>
      <text x="110" y="192" text-anchor="middle" class="m">k numbers across</text>

      <text x="222" y="116" text-anchor="middle" class="h">×</text>

      <!-- B: k x n -->
      <rect x="244" y="30" width="120" height="180" rx="4" class="box" />
      <rect x="300" y="30" width="24" height="180" class="box-b" />
      <text x="304" y="22" text-anchor="middle" class="tb">B (k × n)</text>
      <text x="312" y="232" text-anchor="middle" class="m">column j</text>
      <text x="312" y="250" text-anchor="middle" class="m">(k numbers down)</text>

      <text x="386" y="116" text-anchor="middle" class="h">=</text>

      <!-- C: m x n -->
      <rect x="408" y="50" width="120" height="120" rx="4" class="box" />
      <rect x="464" y="90" width="24" height="24" class="box-c" />
      <text x="468" y="36" text-anchor="middle" class="tb">C (m × n)</text>
      <text x="476" y="192" text-anchor="middle" class="m">cell (i, j)</text>

      <text x="584" y="96" text-anchor="middle" class="m">m × n cells,</text>
      <text x="584" y="114" text-anchor="middle" class="m">each a dot</text>
      <text x="584" y="132" text-anchor="middle" class="m">product of</text>
      <text x="584" y="150" text-anchor="middle" class="m">length k</text>
    </svg>
    <figcaption>
      Each cell of the result is one dot product: a row of A against a column of B. There are m × n cells, and
      each costs k multiplies and k adds, so the whole multiply costs about 2 · m · n · k operations.
    </figcaption>
  </figure>
</template>
