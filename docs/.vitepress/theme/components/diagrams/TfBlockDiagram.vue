<script setup lang="ts">
// A pre-norm decoder block on the residual stream, stacked N times.
</script>

<template>
  <figure class="figure">
    <svg viewBox="0 0 640 560" role="img" aria-label="Bottom to top: token ids go into the embedding table, which starts the residual stream, a vertical line. Inside a block, a branch reads the stream through a normalisation step into attention and adds the result back. A second branch reads through another normalisation into the MLP and adds the result back. The block repeats N times. At the top, a final normalisation and the unembedding produce logits for the next token.">
      <defs>
        <marker id="tf-block-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" class="arrowhead" />
        </marker>
      </defs>
      <!-- bottom: tokens and embedding -->
      <rect x="170" y="510" width="220" height="34" rx="6" class="box" />
      <text x="280" y="532" text-anchor="middle" class="t">token ids</text>
      <path d="M280 508 L280 482" class="ln" marker-end="url(#tf-block-ah)" />
      <rect x="170" y="446" width="220" height="34" rx="6" class="box-b" />
      <text x="280" y="468" text-anchor="middle" class="t">embedding (+ learned positions, if used)</text>

      <!-- residual stream -->
      <path d="M280 444 L280 96" class="ln" style="stroke-width: 4" marker-end="url(#tf-block-ah)" />
      <text x="60" y="270" class="tb">residual</text>
      <text x="60" y="288" class="tb">stream</text>
      <text x="60" y="306" class="m">one vector of width d</text>
      <text x="60" y="322" class="m">per token</text>

      <!-- block outline -->
      <rect x="150" y="120" width="420" height="304" rx="10" class="box" style="fill-opacity: 0.08" />
      <text x="556" y="146" text-anchor="end" class="tb">block × N</text>

      <!-- attention branch -->
      <path d="M280 398 L340 398" class="ln" marker-end="url(#tf-block-ah)" />
      <rect x="342" y="382" width="90" height="32" rx="6" class="box-c" />
      <text x="387" y="403" text-anchor="middle" class="t">norm</text>
      <path d="M434 398 L452 398" class="ln" marker-end="url(#tf-block-ah)" />
      <rect x="454" y="374" width="100" height="48" rx="6" class="box-a" />
      <text x="504" y="395" text-anchor="middle" class="t">attention</text>
      <text x="504" y="412" text-anchor="middle" class="m">mixes tokens</text>
      <path d="M504 372 L504 330 L294 330" class="ln" marker-end="url(#tf-block-ah)" />
      <circle cx="280" cy="330" r="12" class="box-d" />
      <text x="280" y="335" text-anchor="middle" class="tb">+</text>

      <!-- mlp branch -->
      <path d="M280 276 L340 276" class="ln" marker-end="url(#tf-block-ah)" />
      <rect x="342" y="260" width="90" height="32" rx="6" class="box-c" />
      <text x="387" y="281" text-anchor="middle" class="t">norm</text>
      <path d="M434 276 L452 276" class="ln" marker-end="url(#tf-block-ah)" />
      <rect x="454" y="252" width="100" height="48" rx="6" class="box-a" />
      <text x="504" y="273" text-anchor="middle" class="t">MLP</text>
      <text x="504" y="290" text-anchor="middle" class="m">per token</text>
      <path d="M504 250 L504 200 L294 200" class="ln" marker-end="url(#tf-block-ah)" />
      <circle cx="280" cy="200" r="12" class="box-d" />
      <text x="280" y="205" text-anchor="middle" class="tb">+</text>

      <!-- top -->
      <rect x="170" y="60" width="220" height="34" rx="6" class="box-c" />
      <text x="280" y="82" text-anchor="middle" class="t">final norm</text>
      <path d="M280 58 L280 46" class="ln" />
      <rect x="170" y="12" width="220" height="34" rx="6" class="box-b" />
      <text x="280" y="34" text-anchor="middle" class="t">unembedding → logits</text>
    </svg>
    <figcaption>
      A decoder-only transformer. The embedding starts the residual stream, one vector per token. Each block
      reads from the stream twice, through a normalisation step: attention moves information between tokens,
      the MLP works on each token alone, and each adds its result back. The block repeats N times (4 in the
      exercise, about 30 to 100 in large models).
    </figcaption>
  </figure>
</template>
