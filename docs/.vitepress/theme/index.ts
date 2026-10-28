import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import './custom.css'

// Every .vue file under components/ (including diagrams/) is registered globally by its file name,
// so new chapters never need to edit this file.
const components = import.meta.glob('./components/**/*.vue', { eager: true }) as Record<string, { default: any }>

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    for (const [path, mod] of Object.entries(components)) {
      app.component(path.split('/').pop()!.replace(/\.vue$/, ''), mod.default)
    }
  },
} satisfies Theme
