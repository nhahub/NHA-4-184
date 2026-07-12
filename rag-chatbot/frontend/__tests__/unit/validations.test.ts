import {
  loginSchema,
  registerSchema,
  forgotPasswordSchema,
  otpSchema,
  resetPasswordSchema,
  chatMessageSchema,
  userResponseSchema,
  tokenResponseSchema,
  chatResponseSchema,
  conversationListSchema,
  conversationDetailSchema,
} from "../../lib/validations"

// ─── Login ────────────────────────────────────────────────────────────────────
describe("loginSchema", () => {
  it("passes with valid data", () => {
    expect(loginSchema.safeParse({ username: "fatma123", password: "pass123" }).success).toBe(true)
  })

  it("fails when username is empty", () => {
    const r = loginSchema.safeParse({ username: "", password: "pass123" })
    expect(r.success).toBe(false)
    expect(r.error?.issues[0].message).toBe("Username is required")
  })

  it("fails when username is less than 3 chars", () => {
    expect(loginSchema.safeParse({ username: "ab", password: "pass123" }).success).toBe(false)
  })

  it("fails when password is empty", () => {
    const r = loginSchema.safeParse({ username: "fatma123", password: "" })
    expect(r.success).toBe(false)
    expect(r.error?.issues[0].message).toBe("Password is required")
  })

  it("fails when password is less than 6 chars", () => {
    expect(loginSchema.safeParse({ username: "fatma123", password: "123" }).success).toBe(false)
  })
})

// ─── Register ─────────────────────────────────────────────────────────────────
describe("registerSchema", () => {
  const valid = {
    username: "fatma_123",
    email: "fatma@example.com",
    password: "pass123",
    confirmPassword: "pass123",
  }

  it("passes with valid data", () => {
    expect(registerSchema.safeParse(valid).success).toBe(true)
  })

  it("fails when passwords don't match", () => {
    const r = registerSchema.safeParse({ ...valid, confirmPassword: "different" })
    expect(r.success).toBe(false)
    expect(r.error?.issues[0].message).toBe("Passwords do not match")
  })

  it("fails with invalid email", () => {
    const r = registerSchema.safeParse({ ...valid, email: "not-an-email" })
    expect(r.success).toBe(false)
    expect(r.error?.issues[0].message).toBe("Invalid email address")
  })

  it("fails with special chars in username", () => {
    expect(registerSchema.safeParse({ ...valid, username: "fatma@#!" }).success).toBe(false)
  })

  it("fails when username is too short", () => {
    expect(registerSchema.safeParse({ ...valid, username: "ab" }).success).toBe(false)
  })

  it("fails when confirmPassword is empty", () => {
    const r = registerSchema.safeParse({ ...valid, confirmPassword: "" })
    expect(r.success).toBe(false)
  })
})

// ─── Forgot Password ──────────────────────────────────────────────────────────
describe("forgotPasswordSchema", () => {
  it("passes with valid email", () => {
    expect(forgotPasswordSchema.safeParse({ email: "test@test.com" }).success).toBe(true)
  })

  it("fails with invalid email", () => {
    expect(forgotPasswordSchema.safeParse({ email: "not-email" }).success).toBe(false)
  })

  it("fails with empty email", () => {
    const r = forgotPasswordSchema.safeParse({ email: "" })
    expect(r.success).toBe(false)
    expect(r.error?.issues[0].message).toBe("Email is required")
  })
})

// ─── OTP ──────────────────────────────────────────────────────────────────────
describe("otpSchema", () => {
  it("passes with valid 6-digit OTP", () => {
    expect(otpSchema.safeParse({ otp: "123456" }).success).toBe(true)
  })

  it("fails with less than 6 digits", () => {
    const r = otpSchema.safeParse({ otp: "1234" })
    expect(r.success).toBe(false)
    expect(r.error?.issues[0].message).toBe("OTP must be 6 digits")
  })

  it("fails with more than 6 digits", () => {
    expect(otpSchema.safeParse({ otp: "1234567" }).success).toBe(false)
  })

  it("fails with letters", () => {
    const r = otpSchema.safeParse({ otp: "12345a" })
    expect(r.success).toBe(false)
    expect(r.error?.issues[0].message).toBe("OTP must contain numbers only")
  })

  it("fails when empty", () => {
    const r = otpSchema.safeParse({ otp: "" })
    expect(r.success).toBe(false)
    expect(r.error?.issues[0].message).toBe("OTP is required")
  })
})

// ─── Reset Password ───────────────────────────────────────────────────────────
describe("resetPasswordSchema", () => {
  it("passes with matching passwords", () => {
    expect(resetPasswordSchema.safeParse({
      new_password: "newpass123",
      confirm_password: "newpass123",
    }).success).toBe(true)
  })

  it("fails when passwords don't match", () => {
    const r = resetPasswordSchema.safeParse({
      new_password: "newpass123",
      confirm_password: "different",
    })
    expect(r.success).toBe(false)
    expect(r.error?.issues[0].message).toBe("Passwords do not match")
  })

  it("fails when password is too short", () => {
    expect(resetPasswordSchema.safeParse({
      new_password: "123",
      confirm_password: "123",
    }).success).toBe(false)
  })
})

// ─── Chat Message ─────────────────────────────────────────────────────────────
describe("chatMessageSchema", () => {
  it("passes with valid message", () => {
    expect(chatMessageSchema.safeParse({ content: "Hello" }).success).toBe(true)
  })

  it("fails with empty message", () => {
    const r = chatMessageSchema.safeParse({ content: "" })
    expect(r.success).toBe(false)
    expect(r.error?.issues[0].message).toBe("Message cannot be empty")
  })

  it("fails with message over 2000 chars", () => {
    const r = chatMessageSchema.safeParse({ content: "a".repeat(2001) })
    expect(r.success).toBe(false)
    expect(r.error?.issues[0].message).toBe("Message is too long (max 2000 characters)")
  })

  it("passes with exactly 2000 chars", () => {
    expect(chatMessageSchema.safeParse({ content: "a".repeat(2000) }).success).toBe(true)
  })
})

// ─── API Response Schemas ─────────────────────────────────────────────────────
describe("userResponseSchema", () => {
  it("passes with valid user", () => {
    expect(userResponseSchema.safeParse({
      id: 1, username: "fatma", email: "fatma@test.com", is_active: true,
    }).success).toBe(true)
  })

  it("fails with invalid email", () => {
    expect(userResponseSchema.safeParse({
      id: 1, username: "fatma", email: "bad-email", is_active: true,
    }).success).toBe(false)
  })

  it("fails when id is missing", () => {
    expect(userResponseSchema.safeParse({
      username: "fatma", email: "fatma@test.com", is_active: true,
    }).success).toBe(false)
  })
})

describe("tokenResponseSchema", () => {
  it("passes with valid token", () => {
    expect(tokenResponseSchema.safeParse({
      access_token: "eyJhbGci...",
      token_type: "bearer",
    }).success).toBe(true)
  })

  it("fails with empty access_token", () => {
    expect(tokenResponseSchema.safeParse({
      access_token: "",
      token_type: "bearer",
    }).success).toBe(false)
  })
})

describe("chatResponseSchema", () => {
  it("passes with full response", () => {
    expect(chatResponseSchema.safeParse({
      answer: "Hello!",
      conversation_id: 1,
      message_id: 5,
      sources: [],
      response_time: 1.2,
    }).success).toBe(true)
  })

  it("passes with only conversation_id", () => {
    expect(chatResponseSchema.safeParse({ conversation_id: 1 }).success).toBe(true)
  })

  it("fails when conversation_id is missing", () => {
    expect(chatResponseSchema.safeParse({ answer: "Hello!" }).success).toBe(false)
  })
})

describe("conversationListSchema", () => {
  it("passes with valid list", () => {
    expect(conversationListSchema.safeParse([{
      id: 1,
      title: "Shipping question",
      created_at: "2026-04-25T02:39:27.278Z",
      updated_at: "2026-04-25T02:39:27.278Z",
    }]).success).toBe(true)
  })

  it("passes with empty array", () => {
    expect(conversationListSchema.safeParse([]).success).toBe(true)
  })

  it("fails when id is missing", () => {
    expect(conversationListSchema.safeParse([{
      title: "test",
      created_at: "2026-04-25T02:39:27.278Z",
      updated_at: "2026-04-25T02:39:27.278Z",
    }]).success).toBe(false)
  })
})

describe("conversationDetailSchema", () => {
  const validDetail = {
    id: 1,
    title: "Test",
    created_at: "2026-04-25T02:39:27.278Z",
    updated_at: "2026-04-25T02:39:27.278Z",
    messages: [{
      id: 1,
      user_query: "Hello?",
      llm_response: "Hi!",
      response_time: 1.5,
      created_at: "2026-04-25T02:39:27.278Z",
      feedback: null,
    }],
  }

  it("passes with valid detail", () => {
    expect(conversationDetailSchema.safeParse(validDetail).success).toBe(true)
  })

  it("passes with feedback object", () => {
    const withFeedback = {
      ...validDetail,
      messages: [{
        ...validDetail.messages[0],
        feedback: { rating: 5, comment: "Great!", created_at: "2026-04-25T02:39:27.278Z" },
      }],
    }
    expect(conversationDetailSchema.safeParse(withFeedback).success).toBe(true)
  })

  it("passes with empty messages", () => {
    expect(conversationDetailSchema.safeParse({ ...validDetail, messages: [] }).success).toBe(true)
  })
})