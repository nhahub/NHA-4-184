// ─── Auth ────────────────────────────────────────────────────────────────────

export interface RegisterRequest {
  username: string
  email: string
  password: string
  confirm_password: string
}

export interface LoginRequest {
  username: string
  password: string
}

export interface UserResponse {
  id: number
  username: string
  email: string
  is_active: boolean
  is_admin: boolean
}

export interface TokenResponse {
  access_token: string
  token_type: string
}

export interface ForgotPasswordRequest {
  email: string
}

export interface OTPResponse {
  message: string
  otp_sent: boolean
}

export interface VerifyOTPRequest {
  email: string
  otp: string
}

export interface ResetTokenResponse {
  reset_token: string
}

export interface ResetPasswordRequest {
  reset_token: string
  new_password: string
  confirm_password: string
}

// ─── Chat ────────────────────────────────────────────────────────────────────

export interface ChatRequest {
  question: string
  conversation_id?: number
  n_results?: number
}

export interface SourceChunk {
  title?: string
  category?: string
  url?: string
  content?: string
  score?: number
}

export interface ChatResponse {
  answer: string
  response?: string
  conversation_id: number
  message_id: number
  sources: SourceChunk[]
  response_time: number
}

export interface ConversationListItem {
  id: number
  title: string
  created_at: string
  updated_at: string
}

export interface FeedbackInfo {
  rating: number
  comment: string
  created_at: string
}

export interface MessageItem {
  id: number
  user_query: string
  llm_response: string
  response_time: number
  created_at: string
  feedback: FeedbackInfo | null
}

export interface ConversationDetail extends ConversationListItem {
  messages: MessageItem[]
}

// ─── Feedback ────────────────────────────────────────────────────────────────

export interface FeedbackRequest {
  message_id: number
  rating: number
  comment?: string
}

export interface FeedbackResponse {
  id: number
  message_id: number
  rating: number
  comment?: string
  created_at: string
}

// ─── Voice ───────────────────────────────────────────────────────────────────

export interface VoiceQueryParams {
  audio: File
  conversation_id?: number
  n_results?: number
}

// ─── UI Normalized ───────────────────────────────────────────────────────────

/** Normalized message shape used across UI components */
export interface UIMessage {
  id: number | string
  content: string
  role: "user" | "assistant"
  timestamp: string
  sources?: { title: string; url: string }[]
  isTicketReply?: boolean
}

/** Normalized conversation shape used in sidebar */
export interface UIConversation {
  id: string
  title: string
  timestamp: string
}