/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          primary: '#06060f',
          secondary: '#0c0c1e',
          card: '#0e0e24',
          glass: '#141432',
          input: '#19193c',
        },
        accent: {
          1: '#6366f1',
          2: '#8b5cf6',
          3: '#a855f7',
          4: '#06b6d4',
          5: '#3b82f6',
        },
        text: {
          primary: '#f0f0ff',
          secondary: '#9d9dba',
          muted: '#5a5a7a',
          accent: '#8b5cf6',
        }
      }
    },
  },
  corePlugins: {
    preflight: false,
  },
  plugins: [],
}

