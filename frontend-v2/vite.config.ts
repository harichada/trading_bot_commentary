import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiPort = process.env.VITE_API_PORT || '8010'

export default defineConfig({
  plugins: [react()],
  base: '/',
  build: {
    outDir: '../frontend-v2-dist',
    emptyOutDir: true,
  },
  server: {
    port: 5175,
    proxy: {
      '/api/lab': {
        target: 'http://localhost:8005',
        changeOrigin: true,
        ws: true,
      },
      '/api': {
        target: `http://localhost:${apiPort}`,
        changeOrigin: true,
      },
      '/ws': {
        target: `ws://localhost:${apiPort}`,
        ws: true,
      },
    },
  },
})
