<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { load, save } from '../storage'

interface Card {
  q: string
  a: string
}

const props = defineProps<{ id: string; cards: Card[] }>()

const order = ref<number[]>(props.cards.map((_, i) => i))
const pos = ref(0)
const flipped = ref(false)
const known = ref<Record<number, boolean>>({})
const hideKnown = ref(false)

const key = computed(() => `cards:${props.id}`)
onMounted(() => {
  known.value = load(key.value, {})
})

const visible = computed(() => (hideKnown.value ? order.value.filter((i) => !known.value[i]) : order.value))
const current = computed(() => {
  const i = visible.value[Math.min(pos.value, visible.value.length - 1)]
  return i === undefined ? null : { i, ...props.cards[i] }
})
const knownCount = computed(() => Object.values(known.value).filter(Boolean).length)

function go(delta: number) {
  const n = visible.value.length
  if (!n) return
  pos.value = (pos.value + delta + n) % n
  flipped.value = false
}

function shuffle() {
  const a = [...order.value]
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  order.value = a
  pos.value = 0
  flipped.value = false
}

function toggleKnown() {
  if (!current.value) return
  const i = current.value.i
  known.value = { ...known.value, [i]: !known.value[i] }
  save(key.value, known.value)
  if (hideKnown.value && known.value[i]) {
    flipped.value = false
    if (pos.value >= visible.value.length) pos.value = 0
  }
}

function reset() {
  known.value = {}
  save(key.value, {})
}
</script>

<template>
  <div class="widget flashcards">
    <h4>Flashcards</h4>
    <div class="meta">
      <span v-if="current">Card {{ Math.min(pos + 1, visible.length) }} of {{ visible.length }}</span>
      <span>{{ knownCount }} / {{ cards.length }} known</span>
    </div>

    <button v-if="current" class="card" :class="{ flipped }" @click="flipped = !flipped">
      <span class="label">{{ flipped ? 'Answer' : 'Question' }}</span>
      <span class="text" v-html="flipped ? current.a : current.q" />
      <span class="hint">{{ flipped ? 'Tap to see the question' : 'Tap to reveal' }}</span>
    </button>
    <div v-else class="card empty">
      All cards are marked as known.
      <button @click="reset">Reset</button>
    </div>

    <div class="row controls">
      <button @click="go(-1)">← Prev</button>
      <button class="primary" @click="go(1)">Next →</button>
      <button v-if="current" @click="toggleKnown">{{ known[current.i] ? 'Unmark known' : 'I know this' }}</button>
      <button @click="shuffle">Shuffle</button>
      <label class="toggle"><input v-model="hideKnown" type="checkbox" @change="pos = 0" /> Hide known</label>
    </div>
  </div>
</template>

<style scoped>
.meta {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  color: var(--vp-c-text-2);
  margin-bottom: 8px;
}
.card {
  display: flex !important;
  flex-direction: column;
  justify-content: center;
  gap: 12px;
  width: 100%;
  min-height: 180px;
  padding: 24px !important;
  border-radius: 12px !important;
  text-align: left;
  font-size: 17px !important;
  line-height: 1.6;
  background: var(--vp-c-bg) !important;
}
.card.flipped {
  border-color: var(--vp-c-brand-1) !important;
}
.card.empty {
  align-items: center;
  border: 1px solid var(--vp-c-divider);
}
.label {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--vp-c-brand-1);
  font-weight: 700;
}
.text {
  font-weight: 500;
}
.hint {
  font-size: 12px;
  color: var(--vp-c-text-3);
}
.controls {
  margin-top: 12px;
}
.toggle {
  font-size: 14px;
  display: flex;
  gap: 6px;
  align-items: center;
  margin-left: auto;
}
</style>
