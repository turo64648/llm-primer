<script setup lang="ts">
// Timescale spectrum: where each memory mechanism sits, from per-token activations to months-frozen weights.
</script>

<template>
  <figure class="figure">
    <svg viewBox="0 0 640 400" role="img" aria-label="Five rows from fastest to slowest. Activations last one token. Context and KV cache last one conversation. Fast weights and test-time memory such as linear attention, DeltaNet and Titans last one long sequence. Edits and adapters such as ROME, MEMIT, generated LoRAs and SEAL last until removed, per user or task. Slow weights from pretraining and post-training last months and are shared by all users.">
      <text x="20" y="22" class="h">Timescale</text>
      <text x="200" y="22" class="h">Mechanisms</text>
      <text x="450" y="22" class="h">Written by</text>

      <rect x="20" y="36" width="600" height="56" rx="8" class="box" />
      <text x="30" y="60" class="tb">Activations</text>
      <text x="30" y="80" class="m">one token</text>
      <text x="200" y="60" class="t">hidden states in one forward pass</text>
      <text x="450" y="60" class="m">the forward pass</text>

      <rect x="20" y="100" width="600" height="56" rx="8" class="box-a" />
      <text x="30" y="124" class="tb">Context</text>
      <text x="30" y="144" class="m">one conversation</text>
      <text x="200" y="124" class="t">KV cache, prompt, retrieved text</text>
      <text x="200" y="144" class="m">in-context learning, RAG (ch. 11)</text>
      <text x="450" y="124" class="m">appending tokens</text>

      <rect x="20" y="164" width="600" height="72" rx="8" class="box-b" />
      <text x="30" y="188" class="tb">Fast weights</text>
      <text x="30" y="208" class="m">one long sequence</text>
      <text x="200" y="188" class="t">linear attention, DeltaNet, Mamba</text>
      <text x="200" y="208" class="t">TTT layers, Titans, Atlas</text>
      <text x="450" y="188" class="m">a fixed update rule</text>
      <text x="450" y="208" class="m">or a gradient on surprise</text>

      <rect x="20" y="244" width="600" height="72" rx="8" class="box-c" />
      <text x="30" y="268" class="tb">Edits, adapters</text>
      <text x="30" y="288" class="m">until removed;</text>
      <text x="30" y="304" class="m">per user or task</text>
      <text x="200" y="268" class="t">ROME, MEMIT, LoRA</text>
      <text x="200" y="288" class="t">Text-to-LoRA, Doc-to-LoRA, SEAL</text>
      <text x="450" y="268" class="m">an editor, a hypernetwork,</text>
      <text x="450" y="288" class="m">or the model's own edits</text>

      <rect x="20" y="324" width="600" height="56" rx="8" class="box-d" />
      <text x="30" y="348" class="tb">Slow weights</text>
      <text x="30" y="368" class="m">months; shared</text>
      <text x="200" y="348" class="t">pretraining, post-training</text>
      <text x="450" y="348" class="m">backprop on a big dataset</text>
    </svg>
    <figcaption>The timescale spectrum. Production LLMs mostly use the first, second and last rows. The middle rows are where the research in this chapter lives, and most of it still resets or stays small.</figcaption>
  </figure>
</template>
