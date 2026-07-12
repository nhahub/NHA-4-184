"use client"

import { useState } from "react"
import { ThumbsUp, ThumbsDown, ChevronDown, ChevronUp, Bot, User, Headset } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/lib/utils"

interface Source {
  title: string
  url: string
}

interface ChatMessageProps {
  id: number | string
  content: string
  role: "user" | "assistant"
  sources?: Source[]
  timestamp: string
  isTicketReply?: boolean
  onFeedback?: (messageId: number | string, feedback: "up" | "down", comment?: string) => void
}

export function ChatMessage({
  id,
  content,
  role,
  sources,
  timestamp,
  isTicketReply,
  onFeedback,
}: ChatMessageProps) {
  const [showSources, setShowSources] = useState(false)
  const [feedback, setFeedback] = useState<"up" | "down" | null>(null)
  const [showFeedbackInput, setShowFeedbackInput] = useState(false)
  const [feedbackComment, setFeedbackComment] = useState("")

  const handleFeedback = (type: "up" | "down") => {
    setFeedback(type)
    setShowFeedbackInput(true)
  }

  const submitFeedback = () => {
    if (onFeedback && feedback) {
      onFeedback(id, feedback, feedbackComment)
    }
    setShowFeedbackInput(false)
  }

  const isUser = role === "user"

  return (
    <div className={cn("flex gap-3 p-4", isUser && "flex-row-reverse")}>
      {/* Avatar */}
      <Avatar
        className={cn(
          "h-8 w-8 shrink-0",
          isUser ? "bg-primary" : isTicketReply ? "bg-amber-500" : "bg-muted"
        )}
      >
        <AvatarFallback
          className={cn(
            isUser
              ? "bg-primary text-primary-foreground"
              : isTicketReply
                ? "bg-amber-500 text-white"
                : "bg-muted text-muted-foreground"
          )}
        >
          {isUser ? (
            <User className="h-4 w-4" />
          ) : isTicketReply ? (
            <Headset className="h-4 w-4" />
          ) : (
            <Bot className="h-4 w-4" />
          )}
        </AvatarFallback>
      </Avatar>

      {/* Message Content */}
      <div className={cn("flex-1 max-w-[80%]", isUser && "flex flex-col items-end")}>
        {/* Support Team badge */}
        {!isUser && isTicketReply && (
          <div className="flex items-center gap-1.5 mb-1.5">
            <span className="flex items-center gap-1 text-xs font-medium text-amber-600 dark:text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded-full border border-amber-500/20">
              <Headset className="h-3 w-3" />
              Support Team
            </span>
          </div>
        )}

        <div
          className={cn(
            "rounded-2xl px-4 py-3",
            isUser
              ? "bg-primary text-primary-foreground rounded-tr-sm"
              : isTicketReply
                ? "bg-amber-500/10 border border-amber-500/30 rounded-tl-sm"
                : "bg-card border border-border rounded-tl-sm"
          )}
        >
          <p className="text-sm leading-relaxed whitespace-pre-wrap">{content}</p>
        </div>

        {/* Sources Section (AI only) */}
        {!isUser && !isTicketReply && sources && sources.length > 0 && (
          <div className="mt-2">
            <button
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              {showSources ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
              {sources.length} source{sources.length > 1 ? "s" : ""}
            </button>
            {showSources && (
              <div className="mt-2 p-3 bg-muted/50 rounded-lg border border-border">
                <p className="text-xs font-medium text-muted-foreground mb-2">Sources:</p>
                <ul className="space-y-1">
                  {sources.map((source, index) => (
                    <li key={index}>
                      <a
                        href={source.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-primary hover:underline"
                      >
                        {source.title}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Feedback Section (AI only, not for ticket replies) */}
        {!isUser && !isTicketReply && (
          <div className="mt-2">
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleFeedback("up")}
                className={cn(
                  "h-7 px-2 text-muted-foreground hover:text-foreground",
                  feedback === "up" && "text-green-500 hover:text-green-500"
                )}
              >
                <ThumbsUp className="h-3.5 w-3.5" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleFeedback("down")}
                className={cn(
                  "h-7 px-2 text-muted-foreground hover:text-foreground",
                  feedback === "down" && "text-red-500 hover:text-red-500"
                )}
              >
                <ThumbsDown className="h-3.5 w-3.5" />
              </Button>
              <span className="text-xs text-muted-foreground">{timestamp}</span>
            </div>

            {/* Feedback Input */}
            {showFeedbackInput && (
              <div className="mt-2 p-3 bg-muted/50 rounded-lg border border-border">
                <Textarea
                  placeholder="Tell us more about your experience (optional)"
                  value={feedbackComment}
                  onChange={(e) => setFeedbackComment(e.target.value)}
                  className="min-h-[60px] text-sm resize-none bg-background"
                />
                <div className="flex justify-end gap-2 mt-2">
                  <Button variant="ghost" size="sm" onClick={() => setShowFeedbackInput(false)}>
                    Cancel
                  </Button>
                  <Button size="sm" onClick={submitFeedback}>
                    Submit
                  </Button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Timestamp for ticket replies (no feedback buttons) */}
        {!isUser && isTicketReply && (
          <span className="text-xs text-muted-foreground mt-1">{timestamp}</span>
        )}

        {/* Timestamp for user messages */}
        {isUser && <span className="text-xs text-muted-foreground mt-1">{timestamp}</span>}
      </div>
    </div>
  )
}