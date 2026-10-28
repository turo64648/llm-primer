<script setup lang="ts">
import { withBase } from 'vitepress'
import { glossary } from '../../glossary'

const entries = Object.entries(glossary).sort((a, b) => a[1].term.localeCompare(b[1].term))
</script>

<template>
  <dl class="glossary">
    <template v-for="[id, e] in entries" :key="id">
      <dt :id="id">{{ e.term }}</dt>
      <dd>
        {{ e.def }}
        <a v-if="e.chapter" :href="withBase(e.chapter)">Read more →</a>
      </dd>
    </template>
  </dl>
</template>

<style scoped>
dt {
  font-weight: 700;
  margin-top: 20px;
}
dd {
  margin: 4px 0 0;
}
dd a {
  font-size: 14px;
  font-weight: 600;
  white-space: nowrap;
}
</style>
