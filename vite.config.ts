import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

// VITE_BASE lets the static build live under a sub-path (for example GitHub Pages).
// Vite does not put .env files into process.env for this file, so read them with loadEnv;
// a real shell variable still wins.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), 'VITE_')
  return {
    base: process.env.VITE_BASE ?? env.VITE_BASE ?? '/',
    plugins: [react()],
  }
})
