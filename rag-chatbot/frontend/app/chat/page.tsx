"use client"

import { useEffect } from "react"
import { Package } from "lucide-react"
import { ChatSidebar } from "@/components/chat/chat-sidebar"
import { ChatMessage } from "@/components/chat/chat-message"
import { ChatInput } from "@/components/chat/chat-input"
import { TypingIndicator } from "@/components/chat/typing-indicator"
import { EmptyState } from "@/components/chat/empty-state"
import { useAuth } from "@/hooks/useAuth"
import { useChat } from "@/hooks/useChat"
import { useFeedback } from "@/hooks/useFeedback"

export default function ChatPage() {
  const { user, logout } = useAuth()
  const {
    messages,
    conversations,
    currentConversationId,
    isLoading,
    isRecording,
    error,
    messagesEndRef,
    loadHistory,
    selectConversation,
    newChat,
    sendMessage,
    startRecording,
    stopRecording,
  } = useChat()
  const { submit: submitFeedback } = useFeedback()

  useEffect(() => {
    loadHistory()
  }, [loadHistory])

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar — receives real user from /auth/me */}
      <div className="hidden md:block">
        <ChatSidebar
          chatHistory={conversations.map((c) => ({
            id: c.id,
            title: c.title,
            timestamp: c.timestamp,
            isActive: c.id === String(currentConversationId),
            messages: [],
          }))}
          onNewChat={newChat}
          onSelectChat={selectConversation}
          currentChatId={currentConversationId ? String(currentConversationId) : null}
          user={user}
          onLogout={logout}
        />
      </div>

      {/* Main */}
      <div className="flex-1 flex flex-col min-w-0">
        <header className="flex items-center justify-between p-3 border-b shrink-0">
          <div className="flex items-center gap-2">
            <Package className="w-5 h-5" />
            <span className="font-semibold">BrownBox AI</span>
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <EmptyState onSuggestionClick={sendMessage} />
          ) : (
            <div className="max-w-3xl mx-auto px-4 pb-4">
              {messages.map((m) => (
                <ChatMessage
                  key={m.id}
                  {...m}
                  onFeedback={(msgId, type, comment) => {
                    const numericRating = type === "up" ? 1 : -1;
                    submitFeedback(Number(m.id), numericRating, comment);
                  }}
                />
              ))}
              {isLoading && <TypingIndicator />}
              {isRecording && (
                <p className="text-sm text-red-500 text-center py-2 animate-pulse">
                  🎙️ Recording... tap the mic again to send
                </p>
              )}
              {error && (
                <p className="text-sm text-red-500 text-center py-2">{error}</p>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input with voice button inside — ChatGPT style */}
        <ChatInput
          onSend={sendMessage}
          isLoading={isLoading}
          isRecording={isRecording}
          onStartRecording={startRecording}
          onStopRecording={stopRecording}
        />
      </div>
    </div>
  )
}