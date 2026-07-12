"use client"

import { useState, useCallback } from "react"
import { feedbackService } from "@/services/feedback.service"
import type { FeedbackResponse } from "@/types"

export function useFeedback() {
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = useCallback(
    async (messageId: number, rating: number, comment?: string): Promise<FeedbackResponse | null> => {
      setIsSubmitting(true)
      setError(null)
      try {
        const result = await feedbackService.submit({ message_id: messageId, rating, comment })
        return result
      } catch (err: any) {
        const msg = err.response?.data?.detail ?? "Failed to submit feedback"
        setError(msg)
        return null
      } finally {
        setIsSubmitting(false)
      }
    },
    []
  )

  const getFeedback = useCallback(async (messageId: number): Promise<FeedbackResponse | null> => {
    try {
      return await feedbackService.get(messageId)
    } catch {
      return null
    }
  }, [])

  return { submit, getFeedback, isSubmitting, error }
}