/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        youtube: {
          red: '#FF0000',
          dark: '#282828',
          light: '#F1F1F1'
        }
      },
      fontFamily: {
        'youtube': ['Roboto', 'Arial', 'sans-serif']
      }
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
  ],
}
