import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"]
      },
      colors: {
        zero: {
          bg: "#1a1a1a",
          panel: "#262421",
          panel2: "#312e2b",
          border: "#3d3a36",
          light: "#EEEED2",
          dark: "#769656",
          accent: "#81b64c",
          yellow: "#f7f769",
          red: "#bb3e3e"
        }
      }
    }
  },
  plugins: []
};

export default config;
