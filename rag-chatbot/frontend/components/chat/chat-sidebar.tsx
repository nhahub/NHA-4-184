"use client"

import { useState } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Package, Plus, MessageSquare, LogOut, ChevronLeft, ChevronRight, TicketCheck } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { ThemeToggle } from "@/components/theme-toggle"
import { cn } from "@/lib/utils"

interface ChatHistory {
  id: string
  title: string
  timestamp: string
  isActive: boolean
}

interface User {
  id?: number
  username: string
  email?: string
  is_admin?: boolean
}

interface ChatSidebarProps {
  chatHistory: ChatHistory[]
  onNewChat: () => void
  onSelectChat: (id: string) => void
  currentChatId: string | null
  user?: User | null
  onLogout?: () => void
}

function getInitials(username: string): string {
  return username
    .split(/[\s_-]/)
    .map((w) => w[0]?.toUpperCase() ?? "")
    .slice(0, 2)
    .join("")
}

export function ChatSidebar({
  chatHistory,
  onNewChat,
  onSelectChat,
  currentChatId,
  user,
  onLogout,
}: ChatSidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const pathname = usePathname()

  return (
    <aside
      className={cn(
        "flex flex-col h-full bg-sidebar border-r border-sidebar-border transition-all duration-300",
        isCollapsed ? "w-16" : "w-72"
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-sidebar-border">
        {!isCollapsed && (
          <div className="flex items-center gap-2">
            <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-sidebar-primary">
              <Package className="w-4 h-4 text-sidebar-primary-foreground" />
            </div>
            <span className="font-semibold text-sidebar-foreground">BrownBox</span>
          </div>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="h-8 w-8 text-sidebar-foreground hover:bg-sidebar-accent"
        >
          {isCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </Button>
      </div>

      {/* User Profile — real data from API */}
      {user && (
        <div className={cn("p-4 border-b border-sidebar-border", isCollapsed && "flex justify-center")}>
          <div className={cn("flex items-center gap-3", isCollapsed && "flex-col")}>
            <Avatar className="h-10 w-10">
              <AvatarFallback className="bg-sidebar-accent text-sidebar-accent-foreground font-medium">
                {getInitials(user.username)}
              </AvatarFallback>
            </Avatar>
            {!isCollapsed && (
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-sidebar-foreground truncate">{user.username}</p>
                {user.email && (
                  <p className="text-xs text-muted-foreground truncate">{user.email}</p>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* New Chat */}
      <div className="p-3">
        <Button
          onClick={onNewChat}
          className={cn(
            "w-full justify-start gap-2 bg-sidebar-primary text-sidebar-primary-foreground hover:bg-sidebar-primary/90",
            isCollapsed && "justify-center px-0"
          )}
        >
          <Plus className="h-4 w-4" />
          {!isCollapsed && "New Chat"}
        </Button>
      </div>

      {/* Chat History */}
      <div className="flex-1 overflow-y-auto p-2">
        {!isCollapsed && (
          <p className="px-2 py-1 text-xs font-medium text-muted-foreground uppercase tracking-wider">
            Recent Chats
          </p>
        )}
        <div className="space-y-1 mt-2">
          {chatHistory.map((chat) => (
            <button
              key={chat.id}
              onClick={() => onSelectChat(chat.id)}
              className={cn(
                "w-full flex items-center gap-2 px-3 py-2 rounded-lg text-left transition-colors",
                currentChatId === chat.id
                  ? "bg-sidebar-accent text-sidebar-accent-foreground"
                  : "text-sidebar-foreground hover:bg-sidebar-accent/50",
                isCollapsed && "justify-center px-2"
              )}
            >
              <MessageSquare className="h-4 w-4 shrink-0" />
              {!isCollapsed && (
                <div className="flex-1 min-w-0">
                  <p className="text-sm truncate">{chat.title}</p>
                  <p className="text-xs text-muted-foreground">{chat.timestamp}</p>
                </div>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Admin: Support Tickets link */}
      {user?.is_admin && (
        <div className="px-3 pb-2">
          <Link
            href="/tickets"
            className={cn(
              "w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors",
              pathname === "/tickets"
                ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium"
                : "text-sidebar-foreground hover:bg-sidebar-accent/50",
              isCollapsed && "justify-center px-2"
            )}
          >
            <TicketCheck className="h-4 w-4 shrink-0" />
            {!isCollapsed && "Support Tickets"}
          </Link>
        </div>
      )}

      {/* Footer */}
      <div
        className={cn(
          "p-3 border-t border-sidebar-border",
          isCollapsed ? "flex flex-col items-center gap-2" : "flex items-center justify-between"
        )}
      >
        <ThemeToggle />
        <Button
          variant="ghost"
          size={isCollapsed ? "icon" : "sm"}
          onClick={onLogout}
          className="text-sidebar-foreground hover:bg-sidebar-accent gap-2"
        >
          <LogOut className="h-4 w-4" />
          {!isCollapsed && "Logout"}
        </Button>
      </div>
    </aside>
  )
}