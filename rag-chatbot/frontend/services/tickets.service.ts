import { apiClient } from "@/lib/api-client"

export interface TicketResponse {
  id: number
  conversation_id: number | null
  chat_message_id: number | null
  question: string
  status: "open" | "answered" | "resolved" | "closed"
  answer: string | null       
  created_at: string
}

export const ticketsService = {
  // Admin
  getAll: async (): Promise<TicketResponse[]> => {
    const res = await apiClient.get("/tickets/")
    return res.data
  },

  getOne: async (id: number): Promise<TicketResponse> => {
    const res = await apiClient.get(`/tickets/${id}`)
    return res.data
  },

  respond: async (id: number, answer: string): Promise<TicketResponse> => {
    const res = await apiClient.post(`/tickets/${id}/respond`, { answer })
    return res.data
  },

  updateStatus: async (id: number, status: string): Promise<TicketResponse> => {
    const res = await apiClient.patch(`/tickets/${id}/status`, { status })
    return res.data
  },

  // User — used by useChat for the human-handoff polling
  getMine: async (): Promise<TicketResponse[]> => {
    const res = await apiClient.get("/tickets/user/mine")
    return res.data
  },
}