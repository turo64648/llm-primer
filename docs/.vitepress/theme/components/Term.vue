<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref } from 'vue'
import { withBase } from 'vitepress'
import { glossary } from '../../glossary'

const props = defineProps<{ id: string }>()
const entry = computed(() => glossary[props.id])

const open = ref(false)
const anchor = ref<HTMLElement | null>(null)
const pos = ref({ top: 0, left: 0 })

const WIDTH = 300

function place() {
  const r = anchor.value?.getBoundingClientRect()
  if (!r) return
  const vw = document.documentElement.clientWidth
  const width = Math.min(WIDTH, vw - 24)
  const left = Math.max(12, Math.min(r.left, vw - width - 12))
  pos.value = { top: r.bottom + 8, left }
}

function onOutside(e: Event) {
  if (!anchor.value?.contains(e.target as Node)) close()
}

function close() {
  open.value = false
  document.removeEventListener('click', onOutside, true)
  window.removeEventListener('scroll', place, true)
  window.removeEventListener('resize', place)
}

async function toggle() {
  if (open.value) return close()
  place()
  open.value = true
  await nextTick()
  document.addEventListener('click', onOutside, true)
  window.addEventListener('scroll', place, true)
  window.addEventListener('resize', place)
}

onBeforeUnmount(close)
</script>

<template>
  <span
    ref="anchor"
    class="term"
    role="button"
    tabindex="0"
    :aria-expanded="open"
    @click.stop="toggle"
    @keydown.enter.prevent="toggle"
    @keydown.esc="close"
    ><slot /><span
        v-if="open && entry"
        class="term-pop"
        role="tooltip"
        :style="{ top: pos.top + 'px', left: pos.left + 'px', maxWidth: `min(${WIDTH}px, calc(100vw - 24px))` }"
        @click.stop
        ><strong>{{ entry.term }}</strong><span class="def">{{ entry.def }}</span
        ><a v-if="entry.chapter" :href="withBase(entry.chapter)" class="more">Read more →</a></span
    ></span
  >
</template>

<style>
.term {
  cursor: help;
  text-decoration: underline dotted;
  text-decoration-color: var(--vp-c-brand-1);
  text-underline-offset: 3px;
  text-decoration-thickness: 1.5px;
}
.term:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  border-radius: 2px;
}
.term-pop {
  position: fixed;
  z-index: 100;
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 12px 14px;
  border-radius: 10px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-elv);
  box-shadow: var(--vp-shadow-3);
  font-size: 14px;
  line-height: 1.55;
  color: var(--vp-c-text-1);
}
.term-pop .more {
  font-size: 13px;
  font-weight: 600;
  color: var(--vp-c-brand-1);
  text-decoration: none;
}
</style>
