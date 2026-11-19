/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Space Grotesk"', 'Inter', 'system-ui', 'sans-serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      colors: {
        obsidian: '#050608',
        panel: '#0B0D11',
        surface: '#11151C',
        limebrand: '#C9FF3B',
        cyanbrand: '#38BDF8',
      },
    },
  },
  plugins: [],
};
