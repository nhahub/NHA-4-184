"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import {
  chatService,
  normalizeAssistantMessage,
  normalizeConversationMessages,
} from "@/services/chat.service"
import { ticketsService } from "@/services/tickets.service"
import type { UIMessage, UIConversation } from "@/types"

function formatTimestamp(iso: string): string {
  return new Date(iso).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  })
}

function now(): string {
  return new Date().toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", hour12: true })
}

// Inserts a ticket reply directly after the chat message it answers,
// instead of appending it to the end of the conversation.
function insertReplyAfterSource(prev: UIMessage[], ticket: any): UIMessage[] {
  const replyMessage = {
    id: `ticket-${ticket.id}-${Date.now()}`,
    content: `📩 Support team reply:\n\n${ticket.answer}`,
    role: "assistant" as const,
    timestamp: now(),
    isTicketReply: true,
  }

  const sourceIndex = prev.findIndex((m) => String(m.id) === String(ticket.chat_message_id))

  if (sourceIndex === -1) {
    // Couldn't find the original question message (rare) — fall back to appending.
    return [...prev, replyMessage]
  }

  const copy = [...prev]
  copy.splice(sourceIndex + 1, 0, replyMessage)
  return copy
}

export function useChat() {
  const [messages, setMessages] = useState<UIMessage[]>([])
  const [conversations, setConversations] = useState<UIConversation[]>([])
  const [currentConversationId, setCurrentConversationId] = useState<number | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const seenTicketIds = useRef<Set<number>>(new Set())

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, isLoading])

  const loadHistory = useCallback(async () => {
    try {
      const res = await chatService.getHistory()
      const raw: any[] = Array.isArray(res)
        ? res
        : (res as any)?.conversations ?? (res as any)?.data ?? []
      setConversations(
        raw.map((c: any) => ({
          id: String(c.id),
          title: c.title,
          timestamp: formatTimestamp(c.created_at),
        }))
      )
    } catch (err) {
      console.error("[useChat] loadHistory:", err)
    }
  }, [])

  // ── Human handoff: check for admin replies to support tickets (background polling) ──
  const checkTicketUpdates = useCallback(
    async (conversationIdOverride?: number | null) => {
      try {
        const convId = conversationIdOverride !== undefined ? conversationIdOverride : currentConversationId
        const tickets = await ticketsService.getMine()
        const newlyAnswered = tickets.filter(
          (t) =>
            (t.status === "resolved" || t.status === "answered") &&
            t.answer &&
            !seenTicketIds.current.has(t.id)
        )

        if (newlyAnswered.length === 0) return

        const matching = newlyAnswered.filter((t) => convId && t.conversation_id === convId)
        if (matching.length === 0) return

        matching.forEach((t) => seenTicketIds.current.add(t.id))

        setMessages((prev) => {
          let result = prev
          for (const t of matching) {
            result = insertReplyAfterSource(result, t)
          }
          return result
        })
      } catch (err) {
        console.error("[useChat] checkTicketUpdates:", err)
      }
    },
    [currentConversationId]
  )

  // ── Human handoff: re-inject ticket replies every time a conversation is opened ──
  const injectTicketRepliesForConversation = useCallback(async (convId: number) => {
    try {
      const tickets = await ticketsService.getMine()
      const matching = tickets.filter(
        (t) =>
          (t.status === "resolved" || t.status === "answered") &&
          t.answer &&
          t.conversation_id === convId
      )

      if (matching.length === 0) return

      matching.forEach((t) => seenTicketIds.current.add(t.id))

      setMessages((prev) => {
        let result = prev
        for (const t of matching) {
          result = insertReplyAfterSource(result, t)
        }
        return result
      })
    } catch (err) {
      console.error("[useChat] injectTicketRepliesForConversation:", err)
    }
  }, [])

  const selectConversation = useCallback(
    async (id: string) => {
      const numericId = Number(id)
      setCurrentConversationId(numericId)
      setError(null)
      try {
        const detail = await chatService.getConversation(numericId)
        setMessages(normalizeConversationMessages(detail))
       
        await injectTicketRepliesForConversation(numericId)
      } catch {
        setError("Failed to load conversation")
      }
    },
    [injectTicketRepliesForConversation]
  )

  const newChat = useCallback(() => {
    setMessages([])
    setCurrentConversationId(null)
    setError(null)
  }, [])

  const sendMessage = useCallback(
    async (content: string) => {
      setMessages((prev) => [
        ...prev,
        { id: Date.now(), content, role: "user", timestamp: now() },
      ])
      setIsLoading(true)
      setError(null)
      try {
        const response = await chatService.ask({
          question: content,
          conversation_id: currentConversationId ?? 0,
          n_results: 3,
        })
        setMessages((prev) => [...prev, normalizeAssistantMessage(response)])
        if (!currentConversationId && response.conversation_id) {
          setCurrentConversationId(response.conversation_id)
        }
        await loadHistory()
      } catch (err: any) {
        setError(err.message ?? "Something went wrong.")
      } finally {
        setIsLoading(false)
      }
    },
    [currentConversationId, loadHistory]
  )

  const sendVoiceFile = useCallback(
    async (audio: File) => {
      setMessages((prev) => [
        ...prev,
        { id: Date.now(), content: "🎙️ Voice message", role: "user", timestamp: now() },
      ])
      setIsLoading(true)
      setError(null)
      try {
        const response = await chatService.voiceQuery({
          audio,
          conversation_id: currentConversationId ?? undefined,
          n_results: 3,
        })

        const data = response as any

        setMessages((prev) => [
          ...prev.slice(0, -1),
          { id: Date.now(), content: `🎙️ ${data.transcription}`, role: "user", timestamp: now() },
          {
            id: data.message_id,
            content: data.answer ?? "No response from server",
            role: "assistant",
            timestamp: now(),
          },
        ])

        if (data.audio_base64) {
          const audioData = `data:audio/mp3;base64,${data.audio_base64}`
          const audioEl = new Audio(audioData)
          audioEl.play()
        }

        if (!currentConversationId && data.conversation_id) {
          setCurrentConversationId(data.conversation_id)
        }

        await loadHistory()
      } catch {
        setError("Voice query failed. Please try again.")
      } finally {
        setIsLoading(false)
      }
    },
    [currentConversationId, loadHistory]
  )

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(chunksRef.current, { type: "audio/webm" })
        const audioFile = new File([audioBlob], "voice.webm", { type: "audio/webm" })
        stream.getTracks().forEach((t) => t.stop())
        await sendVoiceFile(audioFile)
      }

      mediaRecorder.start()
      setIsRecording(true)
    } catch {
      setError("Microphone access denied.")
    }
  }, [sendVoiceFile])

  const stopRecording = useCallback(() => {
    mediaRecorderRef.current?.stop()
    setIsRecording(false)
  }, [])

  // Background polling — safe because it only touches messages when there's a match
  useEffect(() => {
    const interval = setInterval(() => checkTicketUpdates(), 15000)
    return () => clearInterval(interval)
  }, [checkTicketUpdates])

  return {
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
    checkTicketUpdates,
  }
}