"use client"

import { useState } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { Package, Eye, EyeOff } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ThemeToggle } from "@/components/theme-toggle"
import { useAuth } from "@/hooks/useAuth"

export default function RegisterPage() {
  const router = useRouter()
  const { register, isLoading, error, setError } = useAuth()
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)
  const [success, setSuccess] = useState("")
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    confirmPassword: "",
  })

  const handleChange = (field: keyof typeof formData) =>
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setError(null)
      setFormData((prev) => ({ ...prev, [field]: e.target.value }))
    }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setSuccess("")

    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match")
      return
    }
    if (formData.password.length < 6) {
      setError("Password must be at least 6 characters")
      return
    }

    const ok = await register({
      username: formData.username,
      email: formData.email,
      password: formData.password,
      confirm_password: formData.confirmPassword,
    })

    if (ok) {
      setSuccess("Account created! Redirecting to login…")
      setTimeout(() => router.push("/login"), 1500)
    }
  }

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <header className="flex items-center justify-between p-4 border-b border-border">
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-primary">
            <Package className="w-5 h-5 text-primary-foreground" />
          </div>
          <span className="font-semibold text-lg text-foreground">BrownBox</span>
        </div>
        <ThemeToggle />
      </header>

      <main className="flex-1 flex items-center justify-center p-4">
        <div className="w-full max-w-md">
          <div className="bg-card rounded-xl border border-border p-8 shadow-lg">
            <div className="flex flex-col items-center mb-8">
              <div className="flex items-center justify-center w-14 h-14 rounded-xl bg-primary mb-4">
                <Package className="w-8 h-8 text-primary-foreground" />
              </div>
              <h1 className="text-2xl font-bold text-foreground">Create account</h1>
              <p className="text-muted-foreground mt-1">Get started with BrownBox Support</p>
            </div>

            {error && (
              <div className="mb-4 p-3 rounded-lg bg-red-100 text-red-600 text-sm text-center">
                {error}
              </div>
            )}
            {success && (
              <div className="mb-4 p-3 rounded-lg bg-green-100 text-green-600 text-sm text-center">
                {success}
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-4">
              <Input
                placeholder="Username"
                value={formData.username}
                onChange={handleChange("username")}
                required
              />
              <Input
                type="email"
                placeholder="Email"
                value={formData.email}
                onChange={handleChange("email")}
                required
              />

              <div className="relative">
                <Input
                  type={showPassword ? "text" : "password"}
                  placeholder="Password"
                  value={formData.password}
                  onChange={handleChange("password")}
                  className="pr-10"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((v) => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground"
                >
                  {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>

              <div className="relative">
                <Input
                  type={showConfirm ? "text" : "password"}
                  placeholder="Confirm Password"
                  value={formData.confirmPassword}
                  onChange={handleChange("confirmPassword")}
                  className="pr-10"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowConfirm((v) => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground"
                >
                  {showConfirm ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>

              <Button type="submit" className="w-full h-11" disabled={isLoading}>
                {isLoading ? "Creating account..." : "Create account"}
              </Button>
            </form>

            <p className="text-center text-sm mt-6">
              Already have an account?{" "}
              <Link href="/login" className="text-primary font-medium">
                Sign in
              </Link>
            </p>
          </div>
        </div>
      </main>
    </div>
  )
}