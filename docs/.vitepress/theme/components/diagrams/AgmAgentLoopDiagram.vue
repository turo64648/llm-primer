<script setup lang="ts">
// Agent loop: the model only writes text; the harness runs tools and appends results to the context.
</script>

<template>
  <figure class="figure">
    <svg viewBox="0 0 640 400" role="img" aria-label="The context window holds the system prompt, tool descriptions, the task, and a growing list of tool calls and results. The model reads the whole context and writes either a tool call or a final answer. The harness parses the tool call, runs the real tool, and appends the result to the context, then the model runs again.">
      <defs>
        <marker id="agl-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" class="arrowhead" />
        </marker>
      </defs>

      <text x="20" y="22" class="h">Context window (the only working memory)</text>
      <rect x="20" y="34" width="300" height="350" rx="8" class="box" />
      <rect x="34" y="48" width="272" height="36" rx="6" class="box-a" />
      <text x="170" y="71" text-anchor="middle" class="t">System prompt + memory files</text>
      <rect x="34" y="92" width="272" height="36" rx="6" class="box-a" />
      <text x="170" y="115" text-anchor="middle" class="t">Tool descriptions (name, schema)</text>
      <rect x="34" y="136" width="272" height="36" rx="6" class="box-b" />
      <text x="170" y="159" text-anchor="middle" class="t">User task</text>
      <rect x="34" y="180" width="272" height="36" rx="6" class="box-c" />
      <text x="170" y="203" text-anchor="middle" class="t">Model: thought + tool call</text>
      <rect x="34" y="224" width="272" height="36" rx="6" class="box-d" />
      <text x="170" y="247" text-anchor="middle" class="t">Harness: tool result</text>
      <rect x="34" y="268" width="272" height="36" rx="6" class="box-c" />
      <text x="170" y="291" text-anchor="middle" class="t">Model: thought + tool call</text>
      <rect x="34" y="312" width="272" height="36" rx="6" class="box-d" />
      <text x="170" y="335" text-anchor="middle" class="t">Harness: tool result</text>
      <text x="170" y="372" text-anchor="middle" class="m">... grows every step until full</text>

      <rect x="400" y="60" width="220" height="70" rx="8" class="box-c" />
      <text x="510" y="88" text-anchor="middle" class="tb">Model (frozen weights)</text>
      <text x="510" y="110" text-anchor="middle" class="m">reads context, writes text</text>

      <rect x="400" y="230" width="220" height="90" rx="8" class="box-d" />
      <text x="510" y="258" text-anchor="middle" class="tb">Harness (ordinary code)</text>
      <text x="510" y="280" text-anchor="middle" class="m">parses the tool call, runs it:</text>
      <text x="510" y="300" text-anchor="middle" class="m">shell, search, API, file edit</text>

      <line x1="322" y1="95" x2="398" y2="95" class="ln" marker-end="url(#agl-ah)" />
      <text x="360" y="86" text-anchor="middle" class="m">read</text>
      <line x1="510" y1="132" x2="510" y2="228" class="ln" marker-end="url(#agl-ah)" />
      <text x="518" y="185" class="m">tool call (text)</text>
      <line x1="398" y1="275" x2="322" y2="275" class="ln" marker-end="url(#agl-ah)" />
      <text x="360" y="266" text-anchor="middle" class="m">append</text>
    </svg>
    <figcaption>The agent loop. The model never runs anything: it writes a tool call as text, the harness executes it and appends the result, and the model reads the longer context on the next step. Everything the agent "knows" during the task is in that box on the left.</figcaption>
  </figure>
</template>
