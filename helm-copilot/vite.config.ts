import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3001,
    proxy: {
      '/api': {
        target: process.env.VITE_API_BASE || 'http://127.0.0.1:9000',
        changeOrigin: true,
      },
      '/ws': {
        target: (process.env.VITE_API_BASE || 'http://127.0.0.1:9000').replace('http', 'ws'),
        ws: true,
      },
    },
  },
})
