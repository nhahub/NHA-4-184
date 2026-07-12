import axios, { AxiosError, AxiosResponse, InternalAxiosRequestConfig } from "axios"

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000"

export const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: { "Content-Type": "application/json" },
  withCredentials: true, // ← بيبعت الـ HTTP-only cookie مع كل request
})

// ─── Request interceptor: attach token (localStorage fallback) ───────────────
apiClient.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  const token = localStorage.getItem("token")
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// ─── Response interceptor: handle 401 ────────────────────────────────────────
apiClient.interceptors.response.use(
  (response: AxiosResponse) => response, // ← بيرجع data مباشرة زي fetch client
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      localStorage.removeItem("token")
      localStorage.removeItem("user")
      if (typeof window !== "undefined") {
        window.location.href = "/login"
      }
    }
    return Promise.reject(error)
  }
)
