// Per-browser study progress. Everything is best-effort: storage can be
// unavailable (private mode, blocked site data), so reads fall back to defaults.

const PREFIX = 'llm-primer:'

export function load<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(PREFIX + key)
    return raw === null ? fallback : (JSON.parse(raw) as T)
  } catch {
    return fallback
  }
}

export function save(key: string, value: unknown): void {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(value))
  } catch {
    /* ignore */
  }
}
