<script setup lang="ts">
// PPO-based RLHF vs GRPO: which model copies each needs.
</script>

<template>
  <figure class="figure">
    <svg viewBox="0 0 640 330" role="img" aria-label="PPO needs four models: the policy being trained, a frozen reference copy, a reward model, and a value model being trained. GRPO needs only the policy, the frozen reference, and a reward source such as a checker; it replaces the value model by comparing a group of answers to the same prompt.">
      <text x="160" y="24" text-anchor="middle" class="h">PPO (classic RLHF)</text>
      <text x="480" y="24" text-anchor="middle" class="h">GRPO</text>
      <line x1="320" y1="10" x2="320" y2="320" class="ln-dash" />

      <rect x="20" y="40" width="280" height="50" rx="8" class="box-a" />
      <text x="160" y="62" text-anchor="middle" class="tb">Policy (trained)</text>
      <text x="160" y="80" text-anchor="middle" class="m">the model being improved</text>

      <rect x="20" y="102" width="280" height="50" rx="8" class="box" />
      <text x="160" y="124" text-anchor="middle" class="tb">Reference (frozen)</text>
      <text x="160" y="142" text-anchor="middle" class="m">for the KL penalty</text>

      <rect x="20" y="164" width="280" height="50" rx="8" class="box-b" />
      <text x="160" y="186" text-anchor="middle" class="tb">Reward model (frozen)</text>
      <text x="160" y="204" text-anchor="middle" class="m">scores each answer</text>

      <rect x="20" y="226" width="280" height="50" rx="8" class="box-c" />
      <text x="160" y="248" text-anchor="middle" class="tb">Value model (trained)</text>
      <text x="160" y="266" text-anchor="middle" class="m">predicts the expected score</text>
      <text x="160" y="304" text-anchor="middle" class="m">4 large models in memory</text>

      <rect x="340" y="40" width="280" height="50" rx="8" class="box-a" />
      <text x="480" y="62" text-anchor="middle" class="tb">Policy (trained)</text>
      <text x="480" y="80" text-anchor="middle" class="m">samples a group of answers</text>

      <rect x="340" y="102" width="280" height="50" rx="8" class="box" />
      <text x="480" y="124" text-anchor="middle" class="tb">Reference (frozen)</text>
      <text x="480" y="142" text-anchor="middle" class="m">for the KL penalty</text>

      <rect x="340" y="164" width="280" height="50" rx="8" class="box-b" />
      <text x="480" y="186" text-anchor="middle" class="tb">Checker or reward model</text>
      <text x="480" y="204" text-anchor="middle" class="m">often a program, not a network</text>

      <rect x="340" y="226" width="280" height="50" rx="8" class="ghost" />
      <text x="480" y="248" text-anchor="middle" class="tb">No value model</text>
      <text x="480" y="266" text-anchor="middle" class="m">baseline = group's average score</text>
      <text x="480" y="304" text-anchor="middle" class="m">2 large models (+ a checker)</text>
    </svg>
    <figcaption>What sits in GPU memory during RL. GRPO drops the value model by comparing each answer with other answers to the same prompt.</figcaption>
  </figure>
</template>
