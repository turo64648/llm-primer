<script setup lang="ts">
// Post-training pipeline: base model to assistant / reasoning model.
</script>

<template>
  <figure class="figure">
    <svg viewBox="0 0 640 420" role="img" aria-label="A base model goes through supervised fine-tuning on demonstrations, then preference tuning with RLHF or DPO, then reinforcement learning on verifiable rewards, producing an assistant that can reason. Each stage names the signal it learns from.">
      <defs>
        <marker id="ptp-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" class="arrowhead" />
        </marker>
      </defs>

      <text x="20" y="22" class="h">Stage</text>
      <text x="400" y="22" class="h">What it learns from</text>

      <rect x="20" y="36" width="340" height="56" rx="8" class="box" />
      <text x="190" y="60" text-anchor="middle" class="tb">Base model (pretrained)</text>
      <text x="190" y="80" text-anchor="middle" class="m">continues any text; not an assistant</text>
      <text x="400" y="60" class="t">Trillions of tokens</text>
      <text x="400" y="80" class="m">predict the next token</text>

      <rect x="20" y="118" width="340" height="56" rx="8" class="box-a" />
      <text x="190" y="142" text-anchor="middle" class="tb">1. Supervised fine-tuning</text>
      <text x="190" y="162" text-anchor="middle" class="m">learns the chat format and to answer</text>
      <text x="400" y="142" class="t">~10k to 1M demonstrations</text>
      <text x="400" y="162" class="m">"here is a good answer, copy it"</text>

      <rect x="20" y="200" width="340" height="56" rx="8" class="box-b" />
      <text x="190" y="224" text-anchor="middle" class="tb">2. Preference tuning (RLHF / DPO)</text>
      <text x="190" y="244" text-anchor="middle" class="m">learns tone, helpfulness, refusals</text>
      <text x="400" y="224" class="t">Pairs: A is better than B</text>
      <text x="400" y="244" class="m">from humans or an AI judge</text>

      <rect x="20" y="282" width="340" height="56" rx="8" class="box-c" />
      <text x="190" y="306" text-anchor="middle" class="tb">3. RL on verifiable rewards</text>
      <text x="190" y="326" text-anchor="middle" class="m">learns long, self-checking reasoning</text>
      <text x="400" y="306" class="t">A checker: right or wrong</text>
      <text x="400" y="326" class="m">unit tests, math answers</text>

      <rect x="20" y="364" width="340" height="48" rx="8" class="box-d" />
      <text x="190" y="393" text-anchor="middle" class="tb">Assistant / reasoning model</text>
      <text x="630" y="393" class="m" text-anchor="end">stages 2 and 3 often repeat and interleave</text>

      <line x1="190" y1="92" x2="190" y2="116" class="ln" marker-end="url(#ptp-ah)" />
      <line x1="190" y1="174" x2="190" y2="198" class="ln" marker-end="url(#ptp-ah)" />
      <line x1="190" y1="256" x2="190" y2="280" class="ln" marker-end="url(#ptp-ah)" />
      <line x1="190" y1="338" x2="190" y2="362" class="ln" marker-end="url(#ptp-ah)" />
    </svg>
    <figcaption>A typical post-training pipeline. The model architecture never changes; only the training signal does. Real recipes run several rounds and reorder stages.</figcaption>
  </figure>
</template>
