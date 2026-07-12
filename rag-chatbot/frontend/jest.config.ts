import type { Config } from "jest"

const config: Config = {
  testEnvironment: "jsdom",
  setupFilesAfterEnv: ["<rootDir>/jest.setup.ts"],
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/$1",
  },
  transform: {
    "^.+\\.tsx?$": ["ts-jest", {
      tsconfig: { jsx: "react-jsx" },
    }],
  },
  testMatch: [
    "**/__tests__/unit/**/*.test.ts",
    "**/__tests__/integration/**/*.test.ts",
  ],
  collectCoverageFrom: [
    "services/**/*.ts",
    "hooks/**/*.ts",
    "lib/validations.ts",
    "!**/*.d.ts",
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
  },
}

export default config