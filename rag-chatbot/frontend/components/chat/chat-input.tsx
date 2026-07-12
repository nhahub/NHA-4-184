"use client"

import { useState, useRef, useEffect } from "react"
import { Send, Mic, MicOff } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface ChatInputProps {
  onSend: (message: string) => void
  isLoading: boolean
  isRecording?: boolean
  onStartRecording?: () => void
  onStopRecording?: () => void
}

export function ChatInput({
  onSend,
  isLoading,
  isRecording = false,
  onStartRecording,
  onStopRecording,
}: ChatInputProps) {
  const [message, setMessage] = useState("")
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`
    }
  }, [message])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (message.trim() && !isLoading) {
      onSend(message.trim())
      setMessage("")
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const handleVoiceClick = () => {
    if (isRecording) {
      onStopRecording?.()
    } else {
      onStartRecording?.()
    }
  }

  return (
    <div className="border-t border-border bg-background p-4">
      <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
        <div className={cn(
          "flex items-end gap-2 bg-input rounded-xl border p-2 transition-colors",
          isRecording ? "border-red-500" : "border-border"
        )}>
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isRecording ? "Recording... press mic to send" : "Ask a question..."}
            rows={1}
            className={cn(
              "flex-1 resize-none bg-transparent px-3 py-2 text-sm",
              "placeholder:text-muted-foreground focus:outline-none",
              "max-h-[150px] min-h-[40px]"
            )}
            disabled={isLoading || isRecording}
          />
          <div className="flex items-center gap-1">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className={cn(
                "h-9 w-9 transition-colors",
                isRecording
                  ? "text-red-500 animate-pulse hover:text-red-600"
                  : "text-muted-foreground hover:text-foreground"
              )}
              onClick={handleVoiceClick}
              disabled={isLoading}
              title={isRecording ? "Stop recording" : "Voice message"}
            >
              {isRecording ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
            </Button>
            <Button
              type="submit"
              size="icon"
              className="h-9 w-9"
              disabled={!message.trim() || isLoading || isRecording}
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
        <p className="text-xs text-center text-muted-foreground mt-2">
          BrownBox AI can make mistakes. Please verify important information.
        </p>
      </form>
    </div>
  )
}