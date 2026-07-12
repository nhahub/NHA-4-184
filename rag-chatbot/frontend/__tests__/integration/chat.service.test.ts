import {
  chatService,
  normalizeAssistantMessage,
  normalizeConversationList,
  normalizeConversationMessages,
} from "@/services/chat.service"
import { apiClient } from "@/lib/api-client"

jest.mock("@/lib/api-client", () => ({
  apiClient: {
    post: jest.fn(),
    get: jest.fn(),
  },
}))

const mockPost = apiClient.post as jest.Mock
const mockGet  = apiClient.get  as jest.Mock

beforeEach(() => jest.clearAllMocks())

// ─── ask ──────────────────────────────────────────────────────────────────────
describe("chatService.ask", () => {
  const mockResponse = {
    answer: "Shipping takes 3-5 days.",
    conversation_id: 1,
    message_id: 10,
    sources: [{ title: "Shipping Policy", url: "/docs/shipping" }],
    response_time: 1.2,
  }

  it("returns chat response", async () => {
    mockPost.mockResolvedValue({ data: mockResponse })
    const result = await chatService.ask({ question: "shipping time?", conversation_id: 0, n_results: 3 })
    ;(expect(result.answer) as any).toBe("Shipping takes 3-5 days.")
  })

  it("calls correct endpoint", async () => {
    mockPost.mockResolvedValue({ data: mockResponse })
    await chatService.ask({ question: "test", conversation_id: 0, n_results: 3 })
    expect(mockPost).toHaveBeenCalledWith("/chat/ask", { question: "test", conversation_id: 0, n_results: 3 })
  })

  it("throws on server error", async () => {
    mockPost.mockRejectedValue({ response: { status: 500 } })
    await expect(chatService.ask({ question: "test", conversation_id: 0, n_results: 3 })).rejects.toBeTruthy()
  })
})

// ─── getHistory ───────────────────────────────────────────────────────────────
describe("chatService.getHistory", () => {
  it("returns conversation list", async () => {
    mockGet.mockResolvedValue({
      data: [
        { id: 1, title: "Shipping?", created_at: "2026-04-25T02:39:27.278Z", updated_at: "2026-04-25T02:39:27.278Z" },
        { id: 2, title: "Return policy", created_at: "2026-04-25T02:39:27.278Z", updated_at: "2026-04-25T02:39:27.278Z" },
      ],
    })
    const result = await chatService.getHistory()
    expect(result).toHaveLength(2)
    expect(result[0].title).toBe("Shipping?")
  })

  it("calls correct endpoint", async () => {
    mockGet.mockResolvedValue({ data: [] })
    await chatService.getHistory()
    expect(mockGet).toHaveBeenCalledWith("/chat/history")
  })
})

// ─── getConversation ──────────────────────────────────────────────────────────
describe("chatService.getConversation", () => {
  it("returns conversation detail", async () => {
    mockGet.mockResolvedValue({
      data: {
        id: 1,
        title: "Shipping?",
        created_at: "2026-04-25T02:39:27.278Z",
        updated_at: "2026-04-25T02:39:27.278Z",
        messages: [{
          id: 1,
          user_query: "How long shipping?",
          llm_response: "3-5 days.",
          response_time: 1.2,
          created_at: "2026-04-25T02:39:27.278Z",
          feedback: null,
        }],
      },
    })

    const result = await chatService.getConversation(1)
    expect(result.messages).toHaveLength(1)
    expect(result.messages[0].user_query).toBe("How long shipping?")
  })

  it("calls correct endpoint with id", async () => {
    mockGet.mockResolvedValue({ data: { id: 5, title: "", created_at: "", updated_at: "", messages: [] } })
    await chatService.getConversation(5)
    expect(mockGet).toHaveBeenCalledWith("/chat/history/5")
  })
})

// ─── normalizeAssistantMessage ────────────────────────────────────────────────
describe("normalizeAssistantMessage", () => {
  it("returns assistant message with answer", () => {
    const msg = normalizeAssistantMessage({
      answer: "Hello!",
      conversation_id: 1,
      message_id: 5,
      sources: [],
      response_time: 1,
    })
    expect(msg.role).toBe("assistant")
    expect(msg.content).toBe("Hello!")
  })

  it("falls back to response field if answer missing", () => {
    const msg = normalizeAssistantMessage({
      response: "Fallback response",
      conversation_id: 1,
      message_id: 5,
      sources: [],
      response_time: 1,
    } as any)
    expect(msg.content).toBe("Fallback response")
  })

  it("shows fallback when both missing", () => {
    const msg = normalizeAssistantMessage({
      conversation_id: 1,
      message_id: 5,
      sources: [],
      response_time: 1,
    } as any)
    expect(msg.content).toBe("No response from server")
  })

  it("normalizes sources correctly", () => {
    const msg = normalizeAssistantMessage({
      answer: "Hi",
      conversation_id: 1,
      message_id: 1,
      sources: [{ category: "Shipping", url: "/docs/shipping" }],
      response_time: 1,
    })
    expect(msg.sources?.[0].title).toBe("Shipping")
    expect(msg.sources?.[0].url).toBe("/docs/shipping")
  })

  it("uses message_id as id", () => {
    const msg = normalizeAssistantMessage({
      answer: "Hi",
      conversation_id: 1,
      message_id: 42,
      sources: [],
      response_time: 1,
    })
    expect(msg.id).toBe(42)
  })
})

// ─── normalizeConversationMessages ────────────────────────────────────────────
describe("normalizeConversationMessages", () => {
  const detail = {
    id: 1,
    title: "Test",
    created_at: "2026-04-25T02:39:27.278Z",
    updated_at: "2026-04-25T02:39:27.278Z",
    messages: [
      {
        id: 10,
        user_query: "What is shipping time?",
        llm_response: "3-5 days.",
        response_time: 1,
        created_at: "2026-04-25T02:39:27.278Z",
        feedback: null,
      },
    ],
  }

  it("expands each message to 2 bubbles (user + assistant)", () => {
    const msgs = normalizeConversationMessages(detail)
    expect(msgs).toHaveLength(2)
  })

  it("first bubble is user with user_query", () => {
    const msgs = normalizeConversationMessages(detail)
    expect(msgs[0].role).toBe("user")
    expect(msgs[0].content).toBe("What is shipping time?")
  })

  it("second bubble is assistant with llm_response", () => {
    const msgs = normalizeConversationMessages(detail)
    expect(msgs[1].role).toBe("assistant")
    expect(msgs[1].content).toBe("3-5 days.")
  })

  it("returns empty array for empty messages", () => {
    const msgs = normalizeConversationMessages({ ...detail, messages: [] })
    expect(msgs).toHaveLength(0)
  })
})