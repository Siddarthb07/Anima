/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["DM Sans", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      colors: {
        ink: { 950: "#0c0f14", 900: "#141922", 800: "#1e2533" },
        accent: { cyan: "#22d3ee", rose: "#fb7185", amber: "#fbbf24" },
      },
    },
  },
  plugins: [],
};
