// cypress/e2e/auth.cy.ts
// E2E Tests — Full user flows

const BASE = "http://localhost:3000"
const API  = "http://127.0.0.1:8000"

// ─── Login ────────────────────────────────────────────────────────────────────
describe("Login Page", () => {
  beforeEach(() => {
    cy.visit(`${BASE}/login`)
  })

  it("shows login form elements", () => {
    cy.contains("Welcome back").should("be.visible")
    cy.get("input[placeholder='Enter your username']").should("exist")
    cy.get("input[placeholder='Enter your password']").should("exist")
    cy.contains("Sign in").should("exist")
    cy.contains("Forgot password?").should("exist")
    cy.contains("Register").should("exist")
    cy.contains("Sign in with Google").should("exist")
  })

  it("shows error on empty submit", () => {
    cy.get("button[type='submit']").click()
    // browser native validation prevents submit — inputs stay empty
    cy.url().should("include", "/login")
  })

  it("shows error on wrong credentials", () => {
    cy.intercept("POST", `${API}/auth/login`, {
      statusCode: 401,
      body: { detail: "Invalid username or password" },
    }).as("loginFail")

    cy.get("input[placeholder='Enter your username']").type("wronguser")
    cy.get("input[placeholder='Enter your password']").type("wrongpass")
    cy.get("button[type='submit']").click()

    cy.wait("@loginFail")
    cy.contains("Invalid username or password").should("be.visible")
  })

  it("redirects to /chat on successful login", () => {
    cy.intercept("POST", `${API}/auth/login`, {
      statusCode: 200,
      body: { access_token: "fake-token", token_type: "bearer" },
    }).as("login")

    cy.intercept("GET", `${API}/auth/me`, {
      statusCode: 200,
      body: { id: 1, username: "fatma", email: "fatma@test.com", is_active: true },
    }).as("getMe")

    cy.get("input[placeholder='Enter your username']").type("fatma")
    cy.get("input[placeholder='Enter your password']").type("pass123")
    cy.get("button[type='submit']").click()

    cy.wait("@login")
    cy.wait("@getMe")
    cy.url().should("include", "/chat")
  })

  it("toggles password visibility", () => {
    cy.get("input[placeholder='Enter your password']").should("have.attr", "type", "password")
    cy.get("button[type='button']").first().click()
    cy.get("input[placeholder='Enter your password']").should("have.attr", "type", "text")
  })

  it("navigates to register page", () => {
    cy.contains("Register").click()
    cy.url().should("include", "/register")
  })

  it("navigates to forgot password page", () => {
    cy.contains("Forgot password?").click()
    cy.url().should("include", "/forgot-password")
  })
})

// ─── Register ─────────────────────────────────────────────────────────────────
describe("Register Page", () => {
  beforeEach(() => {
    cy.visit(`${BASE}/register`)
  })

  it("shows register form elements", () => {
    cy.contains("Create account").should("be.visible")
    cy.get("input[placeholder='Username']").should("exist")
    cy.get("input[placeholder='Email']").should("exist")
    cy.get("input[placeholder='Password']").should("exist")
    cy.get("input[placeholder='Confirm Password']").should("exist")
  })

  it("shows error when passwords don't match", () => {
    cy.get("input[placeholder='Username']").type("fatma123")
    cy.get("input[placeholder='Email']").type("fatma@test.com")
    cy.get("input[placeholder='Password']").type("pass123")
    cy.get("input[placeholder='Confirm Password']").type("different")
    cy.get("button[type='submit']").click()
    cy.contains("Passwords do not match").should("be.visible")
  })

  it("shows error when password is too short", () => {
    cy.get("input[placeholder='Username']").type("fatma123")
    cy.get("input[placeholder='Email']").type("fatma@test.com")
    cy.get("input[placeholder='Password']").type("123")
    cy.get("input[placeholder='Confirm Password']").type("123")
    cy.get("button[type='submit']").click()
    cy.contains("Password must be at least 6 characters").should("be.visible")
  })

  it("shows success and redirects on valid register", () => {
    cy.intercept("POST", `${API}/auth/register`, {
      statusCode: 201,
      body: { id: 1, username: "newuser", email: "new@test.com", is_active: true },
    }).as("register")

    cy.get("input[placeholder='Username']").type("newuser")
    cy.get("input[placeholder='Email']").type("new@test.com")
    cy.get("input[placeholder='Password']").type("pass123")
    cy.get("input[placeholder='Confirm Password']").type("pass123")
    cy.get("button[type='submit']").click()

    cy.wait("@register")
    cy.contains("Account created!").should("be.visible")
    cy.url({ timeout: 3000 }).should("include", "/login")
  })

  it("shows error when username is taken", () => {
    cy.intercept("POST", `${API}/auth/register`, {
      statusCode: 400,
      body: { detail: "Username already taken" },
    }).as("registerFail")

    cy.get("input[placeholder='Username']").type("takenuser")
    cy.get("input[placeholder='Email']").type("new@test.com")
    cy.get("input[placeholder='Password']").type("pass123")
    cy.get("input[placeholder='Confirm Password']").type("pass123")
    cy.get("button[type='submit']").click()

    cy.wait("@registerFail")
    cy.contains("Username already taken").should("be.visible")
  })
})

// ─── Forgot Password ──────────────────────────────────────────────────────────
describe("Forgot Password Page", () => {
  beforeEach(() => {
    cy.visit(`${BASE}/forgot-password`)
  })

  it("shows email step by default", () => {
    cy.contains("Reset password").should("be.visible")
    cy.get("input[placeholder='Email address']").should("exist")
    cy.contains("Send OTP").should("exist")
  })

  it("moves to OTP step after sending email", () => {
    cy.intercept("POST", `${API}/auth/forgot-password`, {
      statusCode: 200,
      body: { message: "OTP sent to your email", otp_sent: true },
    }).as("forgotPassword")

    cy.get("input[placeholder='Email address']").type("fatma@test.com")
    cy.contains("Send OTP").click()
    cy.wait("@forgotPassword")

    cy.get("input[placeholder='OTP code']").should("exist")
    cy.contains("Verify OTP").should("exist")
  })

  it("moves to reset step after valid OTP", () => {
    cy.intercept("POST", `${API}/auth/forgot-password`, {
      statusCode: 200,
      body: { message: "OTP sent", otp_sent: true },
    })
    cy.intercept("POST", `${API}/auth/verify-otp`, {
      statusCode: 200,
      body: { reset_token: "reset-token-xyz" },
    }).as("verifyOtp")

    cy.get("input[placeholder='Email address']").type("fatma@test.com")
    cy.contains("Send OTP").click()

    cy.get("input[placeholder='OTP code']").type("123456")
    cy.contains("Verify OTP").click()
    cy.wait("@verifyOtp")

    cy.get("input[placeholder='New password']").should("exist")
    cy.contains("Reset password").should("exist")
  })

  it("shows error on invalid OTP", () => {
    cy.intercept("POST", `${API}/auth/forgot-password`, {
      statusCode: 200,
      body: { message: "OTP sent", otp_sent: true },
    })
    cy.intercept("POST", `${API}/auth/verify-otp`, {
      statusCode: 400,
      body: { detail: "Invalid OTP" },
    })

    cy.get("input[placeholder='Email address']").type("fatma@test.com")
    cy.contains("Send OTP").click()
    cy.get("input[placeholder='OTP code']").type("000000")
    cy.contains("Verify OTP").click()
    cy.contains("Invalid OTP").should("be.visible")
  })

  it("shows done message after successful reset", () => {
    cy.intercept("POST", `${API}/auth/forgot-password`, { statusCode: 200, body: {} })
    cy.intercept("POST", `${API}/auth/verify-otp`, {
      statusCode: 200,
      body: { reset_token: "token-xyz" },
    })
    cy.intercept("POST", `${API}/auth/reset-password`, {
      statusCode: 200,
      body: { message: "Password reset successfully" },
    }).as("resetPassword")

    cy.get("input[placeholder='Email address']").type("fatma@test.com")
    cy.contains("Send OTP").click()
    cy.get("input[placeholder='OTP code']").type("123456")
    cy.contains("Verify OTP").click()
    cy.get("input[placeholder='New password']").type("newpass123")
    cy.get("input[placeholder='Confirm new password']").type("newpass123")
    cy.contains("Reset password").click()
    cy.wait("@resetPassword")
    cy.contains("Password reset successfully").should("be.visible")
  })
})

// ─── Chat ─────────────────────────────────────────────────────────────────────
describe("Chat Page", () => {
  beforeEach(() => {
    // mock auth
    cy.window().then((win) => {
      win.localStorage.setItem("token", "fake-token")
      win.localStorage.setItem("user", JSON.stringify({
        id: 1, username: "fatma", email: "fatma@test.com", is_active: true,
      }))
    })

    cy.intercept("GET", `${API}/auth/me`, {
      statusCode: 200,
      body: { id: 1, username: "fatma", email: "fatma@test.com", is_active: true },
    })
    cy.intercept("GET", `${API}/chat/history`, { statusCode: 200, body: [] })

    cy.visit(`${BASE}/chat`)
  })

  it("shows chat UI elements", () => {
    cy.contains("BrownBox AI").should("be.visible")
    cy.get("textarea[placeholder='Ask a question...']").should("exist")
  })

  it("shows user in sidebar", () => {
    cy.contains("fatma").should("be.visible")
  })

  it("sends a message and shows response", () => {
    cy.intercept("POST", `${API}/chat/ask`, {
      statusCode: 200,
      body: {
        answer: "Shipping takes 3-5 business days.",
        conversation_id: 1,
        message_id: 1,
        sources: [],
        response_time: 1.2,
      },
    }).as("askQuestion")

    cy.intercept("GET", `${API}/chat/history`, {
      statusCode: 200,
      body: [{ id: 1, title: "Shipping?", created_at: "2026-04-25T02:39:27.278Z", updated_at: "2026-04-25T02:39:27.278Z" }],
    })

    cy.get("textarea[placeholder='Ask a question...']").type("What is shipping time?")
    cy.get("button[type='submit']").click()

    cy.wait("@askQuestion")
    cy.contains("What is shipping time?").should("be.visible")
    cy.contains("Shipping takes 3-5 business days.").should("be.visible")
  })

  it("clears messages on new chat", () => {
    cy.intercept("POST", `${API}/chat/ask`, {
      statusCode: 200,
      body: { answer: "Hello!", conversation_id: 1, message_id: 1, sources: [], response_time: 1 },
    })

    cy.get("textarea[placeholder='Ask a question...']").type("Hello")
    cy.get("button[type='submit']").click()
    cy.contains("Hello!").should("be.visible")

    cy.contains("New Chat").click()
    cy.contains("Hello!").should("not.exist")
  })
})