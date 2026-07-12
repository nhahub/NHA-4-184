import { authService } from "@/services/auth.service"
import { apiClient } from "@/lib/api-client"

// mock axios
jest.mock("@/lib/api-client", () => ({
  apiClient: {
    post: jest.fn(),
    get: jest.fn(),
  },
}))

const mockPost = apiClient.post as jest.Mock
const mockGet  = apiClient.get  as jest.Mock

// mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {}
  return {
    getItem: (k: string) => store[k] ?? null,
    setItem: (k: string, v: string) => { store[k] = v },
    removeItem: (k: string) => { delete store[k] },
    clear: () => { store = {} },
  }
})()
Object.defineProperty(window, "localStorage", { value: localStorageMock })

beforeEach(() => {
  jest.clearAllMocks()
  localStorage.clear()
})

// ─── login ────────────────────────────────────────────────────────────────────
describe("authService.login", () => {
  it("saves token to localStorage on success", async () => {
    mockPost.mockResolvedValue({
      data: { access_token: "test-token-123", token_type: "bearer" },
    })

    await authService.login({ username: "fatma", password: "pass123" })
    expect(localStorage.getItem("token")).toBe("test-token-123")
  })

  it("returns token response", async () => {
    mockPost.mockResolvedValue({
      data: { access_token: "abc", token_type: "bearer" },
    })

    const result = await authService.login({ username: "fatma", password: "pass123" })
    expect(result.access_token).toBe("abc")
  })

  it("calls correct endpoint", async () => {
    mockPost.mockResolvedValue({ data: { access_token: "x", token_type: "bearer" } })
    await authService.login({ username: "fatma", password: "pass123" })
    expect(mockPost).toHaveBeenCalledWith("/auth/login", { username: "fatma", password: "pass123" })
  })

  it("throws on invalid credentials", async () => {
    mockPost.mockRejectedValue({
      response: { data: { detail: "Invalid username or password" }, status: 401 },
    })
    await expect(authService.login({ username: "wrong", password: "wrong" })).rejects.toBeTruthy()
  })
})

// ─── register ─────────────────────────────────────────────────────────────────
describe("authService.register", () => {
  it("returns user on success", async () => {
    mockPost.mockResolvedValue({
      data: { id: 1, username: "fatma", email: "fatma@test.com", is_active: true },
    })

    const result = await authService.register({
      username: "fatma",
      email: "fatma@test.com",
      password: "pass123",
      confirm_password: "pass123",
    })
    expect(result.username).toBe("fatma")
  })

  it("calls correct endpoint", async () => {
    mockPost.mockResolvedValue({
      data: { id: 1, username: "fatma", email: "fatma@test.com", is_active: true },
    })
    const payload = {
      username: "fatma", email: "fatma@test.com",
      password: "pass123", confirm_password: "pass123",
    }
    await authService.register(payload)
    expect(mockPost).toHaveBeenCalledWith("/auth/register", payload)
  })

  it("throws when username already taken", async () => {
    mockPost.mockRejectedValue({
      response: { data: { detail: "Username already taken" }, status: 400 },
    })
    await expect(authService.register({
      username: "taken", email: "x@x.com",
      password: "pass123", confirm_password: "pass123",
    })).rejects.toBeTruthy()
  })
})

// ─── getMe ────────────────────────────────────────────────────────────────────
describe("authService.getMe", () => {
  it("returns user and saves to localStorage", async () => {
    mockGet.mockResolvedValue({
      data: { id: 1, username: "fatma", email: "fatma@test.com", is_active: true },
    })

    const user = await authService.getMe()
    expect(user.username).toBe("fatma")
    expect(localStorage.getItem("user")).toBe(JSON.stringify(user))
  })

  it("calls correct endpoint", async () => {
    mockGet.mockResolvedValue({
      data: { id: 1, username: "fatma", email: "fatma@test.com", is_active: true },
    })
    await authService.getMe()
    expect(mockGet).toHaveBeenCalledWith("/auth/me")
  })
})

// ─── logout ───────────────────────────────────────────────────────────────────
describe("authService.logout", () => {
  it("clears token and user from localStorage", () => {
    localStorage.setItem("token", "abc")
    localStorage.setItem("user", JSON.stringify({ id: 1 }))

    authService.logout()

    expect(localStorage.getItem("token")).toBeNull()
    expect(localStorage.getItem("user")).toBeNull()
  })
})

// ─── forgotPassword ───────────────────────────────────────────────────────────
describe("authService.forgotPassword", () => {
  it("returns OTP response", async () => {
    mockPost.mockResolvedValue({
      data: { message: "OTP sent to your email", otp_sent: true },
    })
    const result = await authService.forgotPassword({ email: "fatma@test.com" })
    expect(result.message).toBe("OTP sent to your email")
  })

  it("throws when email not found", async () => {
    mockPost.mockRejectedValue({
      response: { data: { detail: "Email not found" }, status: 404 },
    })
    await expect(authService.forgotPassword({ email: "notfound@test.com" })).rejects.toBeTruthy()
  })
})

// ─── verifyOtp ────────────────────────────────────────────────────────────────
describe("authService.verifyOtp", () => {
  it("returns reset token on valid OTP", async () => {
    mockPost.mockResolvedValue({
      data: { reset_token: "reset-token-xyz" },
    })
    const result = await authService.verifyOtp({ email: "fatma@test.com", otp: "123456" })
    expect(result.reset_token).toBe("reset-token-xyz")
  })

  it("throws on invalid OTP", async () => {
    mockPost.mockRejectedValue({
      response: { data: { detail: "Invalid OTP" }, status: 400 },
    })
    await expect(authService.verifyOtp({ email: "fatma@test.com", otp: "000000" })).rejects.toBeTruthy()
  })
})