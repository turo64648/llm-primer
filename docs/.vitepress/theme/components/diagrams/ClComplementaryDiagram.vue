<script setup lang="ts">
// Complementary learning systems: fast hippocampus, slow neocortex, replay; and the LLM analogue.
</script>

<template>
  <figure class="figure">
    <svg viewBox="0 0 640 380" role="img" aria-label="Experience goes to a fast learner, the hippocampus, which stores each episode after one exposure. During sleep it replays episodes, interleaved with old memories, to a slow learner, the neocortex, which builds structured knowledge. Below, the LLM analogue: context and retrieval memory as the fast store, weights as the slow store, and the consolidation step marked as missing.">
      <defs>
        <marker id="cls-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" class="arrowhead" />
        </marker>
      </defs>

      <text x="20" y="24" class="h">Brain (complementary learning systems)</text>

      <rect x="20" y="44" width="120" height="70" rx="8" class="box" />
      <text x="80" y="74" text-anchor="middle" class="tb">Experience</text>
      <text x="80" y="94" text-anchor="middle" class="m">today's events</text>

      <rect x="200" y="40" width="200" height="78" rx="8" class="box-a" />
      <text x="300" y="64" text-anchor="middle" class="tb">Hippocampus</text>
      <text x="300" y="84" text-anchor="middle" class="m">fast: one exposure</text>
      <text x="300" y="102" text-anchor="middle" class="m">keeps episodes separate</text>

      <rect x="200" y="160" width="200" height="78" rx="8" class="box-c" />
      <text x="300" y="184" text-anchor="middle" class="tb">Neocortex</text>
      <text x="300" y="204" text-anchor="middle" class="m">slow: small changes</text>
      <text x="300" y="222" text-anchor="middle" class="m">shared, structured knowledge</text>

      <line x1="140" y1="79" x2="198" y2="79" class="ln" marker-end="url(#cls-ah)" />
      <line x1="300" y1="118" x2="300" y2="158" class="ln-a" marker-end="url(#cls-ah)" />
      <text x="312" y="134" class="t">replay during sleep,</text>
      <text x="312" y="152" class="m">mixed with old memories</text>
      <line x1="140" y1="100" x2="198" y2="190" class="ln-dash" marker-end="url(#cls-ah)" />
      <text x="70" y="160" class="m">a little,</text>
      <text x="70" y="176" class="m">directly</text>

      <line x1="20" y1="262" x2="620" y2="262" class="ln-dash" />
      <text x="20" y="290" class="h">LLM today</text>

      <rect x="200" y="300" width="200" height="34" rx="8" class="box-a" />
      <text x="300" y="322" text-anchor="middle" class="t">context + retrieval memory</text>
      <rect x="430" y="300" width="190" height="34" rx="8" class="box-c" />
      <text x="525" y="322" text-anchor="middle" class="t">weights</text>
      <line x1="400" y1="317" x2="428" y2="317" class="ln-dash" />
      <text x="415" y="356" text-anchor="middle" class="m">no consolidation step (only the next release)</text>
    </svg>
    <figcaption>Complementary learning systems in one picture. The fast store learns at once without disturbing old knowledge; the slow store learns structure from replayed, interleaved examples. LLMs have rough analogues of both stores but no routine path between them.</figcaption>
  </figure>
</template>
