"use client"

import { Bot } from "lucide-react"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"

export function TypingIndicator() {
  return (
    <div className="flex gap-3 p-4">
      <Avatar className="h-8 w-8 shrink-0 bg-muted">
        <AvatarFallback className="bg-muted text-muted-foreground">
          <Bot className="h-4 w-4" />
        </AvatarFallback>
      </Avatar>
      <div className="flex items-center gap-1 bg-card border border-border rounded-2xl rounded-tl-sm px-4 py-3">
        <div className="flex gap-1">
          <span className="w-2 h-2 bg-muted-foreground/50 rounded-full animate-bounce [animation-delay:-0.3s]" />
          <span className="w-2 h-2 bg-muted-foreground/50 rounded-full animate-bounce [animation-delay:-0.15s]" />
          <span className="w-2 h-2 bg-muted-foreground/50 rounded-full animate-bounce" />
        </div>
      </div>
    </div>
  )
}
