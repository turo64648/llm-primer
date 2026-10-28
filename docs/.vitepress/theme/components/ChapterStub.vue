<script setup lang="ts">
import { computed } from 'vue'
import { allChapters } from '../../chapters'

const props = defineProps<{ id: string }>()
const chapter = computed(() => allChapters.find((c) => c.id === props.id))
</script>

<template>
  <div v-if="chapter" class="stub">
    <p class="note">This chapter hasn't been written yet. It will cover:</p>
    <ul>
      <li v-for="t in chapter.topics" :key="t">{{ t }}</li>
    </ul>
  </div>
</template>

<style scoped>
.stub {
  padding: 4px 20px;
  border-radius: 12px;
  border: 1px dashed var(--vp-c-divider);
}
.note {
  color: var(--vp-c-text-2);
}
</style>
