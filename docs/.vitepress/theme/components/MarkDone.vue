<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { load, save } from '../storage'

const props = defineProps<{ id: string }>()
const done = ref(false)

onMounted(() => {
  done.value = load<Record<string, boolean>>('done', {})[props.id] === true
})

function toggle() {
  const all = load<Record<string, boolean>>('done', {})
  done.value = !done.value
  all[props.id] = done.value
  save('done', all)
}
</script>

<template>
  <div class="mark-done" :class="{ done }">
    <span>{{ done ? 'You marked this chapter as done.' : 'Finished the chapter and the flashcards?' }}</span>
    <button @click="toggle">{{ done ? 'Mark as not done' : 'Mark chapter as done' }}</button>
  </div>
</template>

<style scoped>
.mark-done {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
  justify-content: space-between;
  margin: 40px 0 8px;
  padding: 16px 20px;
  border-radius: 12px;
  border: 1px dashed var(--vp-c-divider);
  font-size: 15px;
}
.mark-done.done {
  border-style: solid;
  border-color: var(--ok);
}
button {
  font: inherit;
  font-weight: 600;
  padding: 6px 14px;
  border-radius: 8px;
  border: 1px solid var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
  background: transparent;
  cursor: pointer;
}
</style>
