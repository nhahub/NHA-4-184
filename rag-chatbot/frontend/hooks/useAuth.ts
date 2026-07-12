"use client"

import { useState, useEffect, useCallback } from "react"
import { useRouter } from "next/navigation"
import { authService } from "@/services/auth.service"
import type { UserResponse, LoginRequest, RegisterRequest } from "@/types"

interface AuthState {
  user: UserResponse | null
  isLoading: boolean
  error: string | null
}

export function useAuth() {
  const router = useRouter()
  const [state, setState] = useState<AuthState>({
    user: null,
    isLoading: false,
    error: null,
  })

  // Rehydrate user — always fetch from /auth/me if token exists
  useEffect(() => {
    const token = localStorage.getItem("token")
    if (!token) return

    authService.getMe()
      .then((user) => setState((prev) => ({ ...prev, user })))
      .catch(() => {
        // token expired or invalid — clear and redirect
        authService.logout()
        router.push("/login")
      })
  }, [router])

  const setError = (error: string | null) =>
    setState((prev) => ({ ...prev, error }))

  const login = useCallback(
    async (payload: LoginRequest) => {
      setState((prev) => ({ ...prev, isLoading: true, error: null }))
      try {
        await authService.login(payload)
        const user = await authService.getMe()
        setState({ user, isLoading: false, error: null })
        router.push("/chat")
      } catch (err: any) {
        const msg = err.response?.data?.detail ?? err.message ?? "Login failed"
        setState((prev) => ({ ...prev, isLoading: false, error: msg }))
      }
    },
    [router]
  )

  const register = useCallback(async (payload: RegisterRequest) => {
    setState((prev) => ({ ...prev, isLoading: true, error: null }))
    try {
      await authService.register(payload)
      setState((prev) => ({ ...prev, isLoading: false }))
      return true
    } catch (err: any) {
      const msg = err.response?.data?.detail ?? err.message ?? "Registration failed"
      setState((prev) => ({ ...prev, isLoading: false, error: msg }))
      return false
    }
  }, [])

  const logout = useCallback(() => {
    authService.logout()
    setState({ user: null, isLoading: false, error: null })
    router.push("/login")
  }, [router])

  return {
    user: state.user,
    isLoading: state.isLoading,
    error: state.error,
    setError,
    login,
    register,
    logout,
  }
}