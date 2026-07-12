import { defineConfig } from "cypress"

export default defineConfig({
  e2e: {
    baseUrl: "http://localhost:3000",
    specPattern: "__tests__/e2e/**/*.cy.ts",
    supportFile: false,
    video: false,
    screenshotOnRunFailure: true,
    defaultCommandTimeout: 8000,
  },
})