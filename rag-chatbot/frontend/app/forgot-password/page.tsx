"use client"

import { useState } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { Package } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ThemeToggle } from "@/components/theme-toggle"
import { authService } from "@/services/auth.service"

type Step = "email" | "otp" | "reset" | "done"

export default function ForgotPasswordPage() {
  const router = useRouter()
  const [step, setStep] = useState<Step>("email")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")

  const [email, setEmail] = useState("")
  const [otp, setOtp] = useState("")
  const [resetToken, setResetToken] = useState("")
  const [passwords, setPasswords] = useState({ new: "", confirm: "" })

  const handleSendOtp = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError("")
    try {
      await authService.forgotPassword({ email })
      setStep("otp")
    } catch (err: any) {
      setError(err.response?.data?.detail ?? "Failed to send OTP")
    } finally {
      setIsLoading(false)
    }
  }

  const handleVerifyOtp = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError("")
    try {
      const { reset_token } = await authService.verifyOtp({ email, otp })
      setResetToken(reset_token)
      setStep("reset")
    } catch (err: any) {
      setError(err.response?.data?.detail ?? "Invalid OTP")
    } finally {
      setIsLoading(false)
    }
  }

  const handleResetPassword = async (e: React.FormEvent) => {
    e.preventDefault()
    if (passwords.new !== passwords.confirm) {
      setError("Passwords do not match")
      return
    }
    setIsLoading(true)
    setError("")
    try {
      await authService.resetPassword({
        reset_token: resetToken,
        new_password: passwords.new,
        confirm_password: passwords.confirm,
      })
      setStep("done")
      setTimeout(() => router.push("/login"), 2000)
    } catch (err: any) {
      setError(err.response?.data?.detail ?? "Reset failed")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <header className="flex items-center justify-between p-4 border-b border-border">
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-primary">
            <Package className="w-5 h-5 text-primary-foreground" />
          </div>
          <span className="font-semibold text-lg">BrownBox</span>
        </div>
        <ThemeToggle />
      </header>

      <main className="flex-1 flex items-center justify-center p-4">
        <div className="w-full max-w-md">
          <div className="bg-card rounded-xl border border-border p-8 shadow-lg">
            <h1 className="text-2xl font-bold text-foreground mb-2">Reset password</h1>

            {error && (
              <div className="mb-4 p-3 rounded-lg bg-red-100 text-red-600 text-sm">
                {error}
              </div>
            )}

            {/* Step 1 — Email */}
            {step === "email" && (
              <form onSubmit={handleSendOtp} className="space-y-4">
                <p className="text-sm text-muted-foreground mb-4">
                  Enter your email to receive a one-time password.
                </p>
                <Input
                  type="email"
                  placeholder="Email address"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
                <Button type="submit" className="w-full h-11" disabled={isLoading}>
                  {isLoading ? "Sending…" : "Send OTP"}
                </Button>
              </form>
            )}

            {/* Step 2 — OTP */}
            {step === "otp" && (
              <form onSubmit={handleVerifyOtp} className="space-y-4">
                <p className="text-sm text-muted-foreground mb-4">
                  We sent a code to <strong>{email}</strong>. Enter it below.
                </p>
                <Input
                  placeholder="OTP code"
                  value={otp}
                  onChange={(e) => setOtp(e.target.value)}
                  required
                />
                <Button type="submit" className="w-full h-11" disabled={isLoading}>
                  {isLoading ? "Verifying…" : "Verify OTP"}
                </Button>
              </form>
            )}

            {/* Step 3 — New password */}
            {step === "reset" && (
              <form onSubmit={handleResetPassword} className="space-y-4">
                <Input
                  type="password"
                  placeholder="New password"
                  value={passwords.new}
                  onChange={(e) => setPasswords((p) => ({ ...p, new: e.target.value }))}
                  required
                />
                <Input
                  type="password"
                  placeholder="Confirm new password"
                  value={passwords.confirm}
                  onChange={(e) => setPasswords((p) => ({ ...p, confirm: e.target.value }))}
                  required
                />
                <Button type="submit" className="w-full h-11" disabled={isLoading}>
                  {isLoading ? "Saving…" : "Reset password"}
                </Button>
              </form>
            )}

            {/* Done */}
            {step === "done" && (
              <p className="text-sm text-green-600 text-center">
                Password reset successfully! Redirecting to login…
              </p>
            )}

            <p className="text-center text-sm mt-6">
              <Link href="/login" className="text-primary font-medium">
                ← Back to login
              </Link>
            </p>
          </div>
        </div>
      </main>
    </div>
  )
}