import { apiClient } from "@/lib/api-client"
import type { FeedbackRequest, FeedbackResponse } from "@/types"

export const feedbackService = {
  submit: async (payload: FeedbackRequest): Promise<FeedbackResponse> => {
    return (await apiClient.post<FeedbackResponse>("/feedback/", payload)).data
  },

  get: async (message_id: number): Promise<FeedbackResponse> => {
    return (await apiClient.get<FeedbackResponse>(`/feedback/${message_id}`)).data
  },
}