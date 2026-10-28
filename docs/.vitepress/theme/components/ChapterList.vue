<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { withBase } from 'vitepress'
import { parts } from '../../chapters'
import { load } from '../storage'

const done = ref<Record<string, boolean>>({})
onMounted(() => {
  done.value = load('done', {})
})
</script>

<template>
  <div class="chapter-list">
    <section v-for="part in parts" :key="part.title">
      <h3>{{ part.title }}</h3>
      <a
        v-for="c in part.chapters"
        :key="c.id"
        :href="withBase(c.link)"
        class="chapter"
        :class="{ soon: !c.ready }"
      >
        <span class="num">{{ c.num }}</span>
        <span class="body">
          <span class="title">{{ c.title }}</span>
          <span class="topics">{{ c.topics.join(' · ') }}</span>
        </span>
        <span class="status">
          <template v-if="done[c.id]">✓ Done</template>
          <template v-else-if="!c.ready">Soon</template>
        </span>
      </a>
    </section>
  </div>
</template>

<style scoped>
section {
  margin-bottom: 28px;
}
h3 {
  font-size: 13px !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--vp-c-text-2);
  margin: 0 0 8px !important;
  border: 0 !important;
}
.chapter {
  display: flex;
  gap: 14px;
  align-items: flex-start;
  padding: 12px 14px;
  border-radius: 10px;
  text-decoration: none !important;
  color: inherit !important;
  border: 1px solid transparent;
  transition: border-color 0.15s, background 0.15s;
}
.chapter:hover {
  border-color: var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
}
.num {
  flex: 0 0 28px;
  font-weight: 700;
  color: var(--vp-c-brand-1);
  font-variant-numeric: tabular-nums;
}
.body {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.title {
  font-weight: 600;
}
.topics {
  font-size: 14px;
  line-height: 1.5;
  color: var(--vp-c-text-2);
}
.status {
  flex: 0 0 auto;
  font-size: 13px;
  font-weight: 600;
  color: var(--ok);
}
.soon .status {
  color: var(--vp-c-text-3);
  font-weight: 500;
}
.soon .title {
  color: var(--vp-c-text-2);
}
</style>
