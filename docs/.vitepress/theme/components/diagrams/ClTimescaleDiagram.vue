<script setup lang="ts">
// Timescale spectrum: where an LLM stores information vs where a brain does.
</script>

<template>
  <figure class="figure">
    <svg viewBox="0 0 640 400" role="img" aria-label="Two rows on a time axis from milliseconds to years. The LLM row has activations lasting one token, the context window lasting one conversation, an empty gap from hours to months, and weights that change with each release. The brain row has overlapping mechanisms covering every timescale: neural activity, short-term synaptic changes, early long-term potentiation, hippocampal memory, and slow cortical change through consolidation.">
      <text x="20" y="24" class="h">How long a change lasts</text>

      <line x1="120" y1="44" x2="620" y2="44" class="ln" />
      <text x="120" y="38" text-anchor="middle" class="m">ms</text>
      <text x="203" y="38" text-anchor="middle" class="m">second</text>
      <text x="286" y="38" text-anchor="middle" class="m">minute</text>
      <text x="370" y="38" text-anchor="middle" class="m">hour</text>
      <text x="453" y="38" text-anchor="middle" class="m">day</text>
      <text x="536" y="38" text-anchor="middle" class="m">month</text>
      <text x="620" y="38" text-anchor="middle" class="m">year</text>

      <text x="20" y="90" class="tb">LLM</text>
      <rect x="120" y="66" width="80" height="44" rx="6" class="box-a" />
      <text x="160" y="86" text-anchor="middle" class="t">activations</text>
      <text x="160" y="102" text-anchor="middle" class="m">one token</text>
      <rect x="204" y="66" width="170" height="44" rx="6" class="box-b" />
      <text x="289" y="86" text-anchor="middle" class="t">context / KV cache</text>
      <text x="289" y="102" text-anchor="middle" class="m">one conversation</text>
      <rect x="378" y="66" width="150" height="44" rx="6" class="ghost" />
      <text x="453" y="86" text-anchor="middle" class="t">nothing learned</text>
      <text x="453" y="102" text-anchor="middle" class="m">(memory tools are text)</text>
      <rect x="532" y="66" width="88" height="44" rx="6" class="box-c" />
      <text x="576" y="86" text-anchor="middle" class="t">weights</text>
      <text x="576" y="102" text-anchor="middle" class="m">per release</text>

      <line x1="20" y1="134" x2="620" y2="134" class="ln-dash" />

      <text x="20" y="190" class="tb">Brain</text>
      <rect x="120" y="150" width="110" height="40" rx="6" class="box-a" />
      <text x="175" y="175" text-anchor="middle" class="t">neural firing</text>
      <rect x="170" y="198" width="160" height="40" rx="6" class="box-b" />
      <text x="250" y="223" text-anchor="middle" class="t">short-term synaptic</text>
      <rect x="270" y="246" width="150" height="40" rx="6" class="box-b" />
      <text x="345" y="271" text-anchor="middle" class="t">early LTP</text>
      <rect x="360" y="294" width="190" height="40" rx="6" class="box-d" />
      <text x="455" y="319" text-anchor="middle" class="t">hippocampal memory</text>
      <rect x="450" y="342" width="170" height="40" rx="6" class="box-c" />
      <text x="535" y="367" text-anchor="middle" class="t">cortex, via sleep</text>
    </svg>
    <figcaption>Rough timescales of storage. An LLM has fast storage (activations, context) and very slow storage (weights), with nothing that learns in between. A brain has overlapping mechanisms at every scale; LTP is long-term potentiation, a lasting strengthening of a synapse. Boundaries are approximate.</figcaption>
  </figure>
</template>
