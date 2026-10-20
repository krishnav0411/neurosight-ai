/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#f8fafc",
        primary: { DEFAULT: "#1e40af", dark: "#173bab" },
        success: "#16a34a",
        warning: "#d97706",
        danger: "#dc2626",
        ink: "#0f172a",
        ink2: "#475569",
        ink3: "#94a3b8",
        hairline: "#e2e8f0",
      },
      fontFamily: { sans: ["Inter", "system-ui", "sans-serif"] },
      borderRadius: { card: "12px", control: "8px" },
      boxShadow: {
        card: "0 1px 3px 0 rgba(15, 23, 42, 0.05)",
        modal: "0 4px 12px -2px rgba(15, 23, 42, 0.08)",
      },
    },
  },
  plugins: [],
};
