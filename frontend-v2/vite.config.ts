import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: {
      '/api/lab': {
        target: 'http://localhost:8005',
        changeOrigin: true,
        ws: true,
      },
      '/api': {
        target: 'http://localhost:8003',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8003',
        ws: true,
      },
    },
  },
})
