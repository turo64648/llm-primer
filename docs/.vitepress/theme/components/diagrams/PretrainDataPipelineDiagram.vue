<script setup lang="ts">
// Pretraining data pipeline: crawl to training mixture.
</script>

<template>
  <figure class="figure">
    <svg viewBox="0 0 640 400" role="img" aria-label="Raw web crawl goes through text extraction, language and heuristic filters, deduplication, a quality classifier, and is then mixed with code, books, math and synthetic data into the training mixture">
      <defs>
        <marker id="pdp-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" class="arrowhead" />
        </marker>
      </defs>

      <rect x="20" y="10" width="600" height="44" rx="8" class="box" />
      <text x="320" y="38" text-anchor="middle" class="tb">Raw web crawl: billions of HTML pages</text>

      <rect x="60" y="72" width="520" height="40" rx="8" class="box-a" />
      <text x="320" y="97" text-anchor="middle" class="t">1. Extract text from HTML, detect language</text>

      <rect x="100" y="130" width="440" height="40" rx="8" class="box-a" />
      <text x="320" y="155" text-anchor="middle" class="t">2. Rule filters: length, symbols, repetition, URL lists</text>

      <rect x="140" y="188" width="360" height="40" rx="8" class="box-a" />
      <text x="320" y="213" text-anchor="middle" class="t">3. Deduplicate: exact, then near-duplicate</text>

      <rect x="180" y="246" width="280" height="40" rx="8" class="box-a" />
      <text x="320" y="271" text-anchor="middle" class="t">4. Quality classifier keeps the top slice</text>

      <rect x="20" y="246" width="140" height="70" rx="8" class="box-c" />
      <text x="90" y="272" text-anchor="middle" class="tb">Other sources</text>
      <text x="90" y="292" text-anchor="middle" class="m">code, books, papers,</text>
      <text x="90" y="308" text-anchor="middle" class="m">math, synthetic</text>

      <rect x="180" y="330" width="280" height="56" rx="8" class="box-b" />
      <text x="320" y="354" text-anchor="middle" class="tb">5. Mixture: weights per source</text>
      <text x="320" y="374" text-anchor="middle" class="m">trillions of tokens, shuffled</text>

      <line x1="320" y1="54" x2="320" y2="70" class="ln" marker-end="url(#pdp-ah)" />
      <line x1="320" y1="112" x2="320" y2="128" class="ln" marker-end="url(#pdp-ah)" />
      <line x1="320" y1="170" x2="320" y2="186" class="ln" marker-end="url(#pdp-ah)" />
      <line x1="320" y1="228" x2="320" y2="244" class="ln" marker-end="url(#pdp-ah)" />
      <line x1="320" y1="286" x2="320" y2="328" class="ln" marker-end="url(#pdp-ah)" />
      <line x1="90" y1="316" x2="200" y2="340" class="ln" marker-end="url(#pdp-ah)" />

      <text x="620" y="204" text-anchor="end" class="m">each step</text>
      <text x="620" y="220" text-anchor="end" class="m">removes a</text>
      <text x="620" y="236" text-anchor="end" class="m">large share</text>
    </svg>
    <figcaption>A typical web-data pipeline. Each stage narrows the funnel; the mixture step decides how often the model sees each source.</figcaption>
  </figure>
</template>
