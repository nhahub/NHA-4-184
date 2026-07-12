import { apiClient } from "@/lib/api-client"
import type {
  RegisterRequest,
  LoginRequest,
  TokenResponse,
  UserResponse,
  ForgotPasswordRequest,
  OTPResponse,
  VerifyOTPRequest,
  ResetTokenResponse,
  ResetPasswordRequest,
} from "@/types"

export const authService = {
  register: async (payload: RegisterRequest): Promise<UserResponse> => {
    return (await apiClient.post<UserResponse>("/auth/register", payload)).data
  },

  login: async (payload: LoginRequest): Promise<TokenResponse> => {
    const response = await apiClient.post<TokenResponse>("/auth/login", payload)
    const data = response.data
    localStorage.setItem("token", data.access_token)
    return data
  },

  getMe: async (): Promise<UserResponse> => {
    const response = await apiClient.get<UserResponse>("/auth/me")
    const data = response.data
    localStorage.setItem("user", JSON.stringify(data))
    return data
  },

  forgotPassword: async (payload: ForgotPasswordRequest): Promise<OTPResponse> => {
    return (await apiClient.post<OTPResponse>("/auth/forgot-password", payload)).data
  },

  verifyOtp: async (payload: VerifyOTPRequest): Promise<ResetTokenResponse> => {
    return (await apiClient.post<ResetTokenResponse>("/auth/verify-otp", payload)).data
  },

  resetPassword: async (payload: ResetPasswordRequest): Promise<void> => {
    await apiClient.post("/auth/reset-password", payload)
  },

  googleLogin: (): void => {
      const base = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000"
      window.location.href = `${base}/auth/google/login`
  },

  logout: (): void => {
    localStorage.removeItem("token")
    localStorage.removeItem("user")
  },
}
