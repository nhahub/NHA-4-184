import { z } from "zod"

// ─── Form Schemas ─────────────────────────────────────────────────────────────

export const loginSchema = z.object({
  username: z
    .string()
    .min(1, "Username is required")
    .min(3, "Username must be at least 3 characters"),
  password: z
    .string()
    .min(1, "Password is required")
    .min(6, "Password must be at least 6 characters"),
})

export const registerSchema = z
  .object({
    username: z
      .string()
      .min(1, "Username is required")
      .min(3, "Username must be at least 3 characters")
      .regex(/^[a-zA-Z0-9_]+$/, "Username can only contain letters, numbers, and underscores"),
    email: z
      .string()
      .min(1, "Email is required")
      .email("Invalid email address"),
    password: z
      .string()
      .min(1, "Password is required")
      .min(6, "Password must be at least 6 characters"),
    confirmPassword: z
      .string()
      .min(1, "Please confirm your password"),
  })
  .refine((d) => d.password === d.confirmPassword, {
    message: "Passwords do not match",
    path: ["confirmPassword"],
  })

export const forgotPasswordSchema = z.object({
  email: z
    .string()
    .min(1, "Email is required")
    .email("Invalid email address"),
})

export const otpSchema = z.object({
  otp: z
    .string()
    .min(1, "OTP is required")
    .length(6, "OTP must be 6 digits")
    .regex(/^\d+$/, "OTP must contain numbers only"),
})

export const resetPasswordSchema = z
  .object({
    new_password: z
      .string()
      .min(1, "Password is required")
      .min(6, "Password must be at least 6 characters"),
    confirm_password: z
      .string()
      .min(1, "Please confirm your password"),
  })
  .refine((d) => d.new_password === d.confirm_password, {
    message: "Passwords do not match",
    path: ["confirm_password"],
  })

export const chatMessageSchema = z.object({
  content: z
    .string()
    .min(1, "Message cannot be empty")
    .max(2000, "Message is too long (max 2000 characters)"),
})

// ─── API Response Schemas ─────────────────────────────────────────────────────

export const userResponseSchema = z.object({
  id: z.number(),
  username: z.string(),
  email: z.string().email(),
  is_active: z.boolean(),
})

export const tokenResponseSchema = z.object({
  access_token: z.string().min(1),
  token_type: z.string(),
})

export const chatResponseSchema = z.object({
  answer: z.string().optional(),
  response: z.string().optional(),
  conversation_id: z.number(),
  message_id: z.number().optional(),
  sources: z.array(z.object({
    title: z.string().optional(),
    category: z.string().optional(),
    url: z.string().optional(),
    score: z.number().optional(),
  })).optional(),
  response_time: z.number().optional(),
})

export const conversationListSchema = z.array(z.object({
  id: z.number(),
  title: z.string(),
  created_at: z.string(),
  updated_at: z.string(),
}))

export const conversationDetailSchema = z.object({
  id: z.number(),
  title: z.string(),
  created_at: z.string(),
  updated_at: z.string(),
  messages: z.array(z.object({
    id: z.number(),
    user_query: z.string(),
    llm_response: z.string(),
    response_time: z.number(),
    created_at: z.string(),
    feedback: z.object({
      rating: z.number(),
      comment: z.string(),
      created_at: z.string(),
    }).nullable(),
  })),
})

export const feedbackResponseSchema = z.object({
  id: z.number(),
  message_id: z.number(),
  rating: z.number().min(1).max(5),
  comment: z.string().optional(),
  created_at: z.string(),
})

// ─── Types ────────────────────────────────────────────────────────────────────

export type LoginFormData       = z.infer<typeof loginSchema>
export type RegisterFormData    = z.infer<typeof registerSchema>
export type ForgotPasswordData  = z.infer<typeof forgotPasswordSchema>
export type OTPFormData         = z.infer<typeof otpSchema>
export type ResetPasswordData   = z.infer<typeof resetPasswordSchema>
export type ChatMessageData     = z.infer<typeof chatMessageSchema>