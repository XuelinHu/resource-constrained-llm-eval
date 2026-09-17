import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 4024,
    proxy: {
      '/api': 'http://127.0.0.1:8029',
    },
  },
  preview: {
    port: 4024,
    strictPort: true,
    proxy: {
      '/api': 'http://127.0.0.1:8029',
    },
  },
})
