import { heroui } from "@heroui/theme";

export default heroui({
  themes: {
    light: {
      colors: {
        background: "#E8BCB8", // Light peach
        foreground: "#1B1931", // Dark navy
        primary: {
          50: "#f5f3ff",
          100: "#ede9fe",
          200: "#ddd6fe",
          300: "#c4b5fd",
          400: "#a78bfa",
          500: "#862249", // Purple-burgundy
          600: "#7c3aed",
          700: "#6d28d9",
          800: "#5b21b6",
          900: "#44174E", // Deep purple
          DEFAULT: "#862249",
          foreground: "#ffffff",
        },
        secondary: {
          50: "#fdf4f3",
          100: "#fce8e5",
          200: "#f9d5cf",
          300: "#f4b8ad",
          400: "#ED9E59", // Peach/orange
          500: "#A34054", // Mauve/rose
          600: "#862249",
          700: "#6d1f3a",
          800: "#5a1b30",
          900: "#44174E",
          DEFAULT: "#A34054",
          foreground: "#ffffff",
        },
        focus: "#ED9E59",
      },
    },
    dark: {
      colors: {
        background: "#1B1931", // Dark navy
        foreground: "#E8BCB8", // Light peach
        primary: {
          50: "#44174E",
          100: "#5a1b30",
          200: "#6d1f3a",
          300: "#862249", // Purple-burgundy
          400: "#A34054", // Mauve/rose
          500: "#ED9E59", // Peach/orange
          600: "#f4b8ad",
          700: "#f9d5cf",
          800: "#fce8e5",
          900: "#fdf4f3",
          DEFAULT: "#862249",
          foreground: "#ffffff",
        },
        secondary: {
          50: "#1B1931",
          100: "#44174E",
          200: "#862249",
          300: "#A34054",
          400: "#ED9E59",
          500: "#E8BCB8",
          600: "#f0cbc7",
          700: "#f5dbd8",
          800: "#f9e9e7",
          900: "#fdf6f5",
          DEFAULT: "#44174E",
          foreground: "#E8BCB8",
        },
        focus: "#ED9E59",
      },
    },
  },
});
