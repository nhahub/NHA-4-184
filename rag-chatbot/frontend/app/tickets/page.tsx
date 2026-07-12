"use client"

import { useEffect, useState } from "react"
import { usePathname, useRouter } from "next/navigation"
import { ticketsService } from "@/services/tickets.service"
import { authService } from "@/services/auth.service"

export default function TicketsPage() {
  const pathname = usePathname()
  const router = useRouter()
  const [tickets, setTickets] = useState<any[]>([])
  const [selected, setSelected] = useState<any>(null)
  const [answer, setAnswer] = useState("")
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Admin guard — redirect non-admins immediately
  useEffect(() => {
    authService.getMe().then((user) => {
      if (!user.is_admin) {
        router.replace("/chat")
      }
    }).catch(() => {
      router.replace("/login")
    })
  }, [router])

  const loadTickets = () => {
    setLoading(true)
    ticketsService
      .getAll()
      .then((data) => {
        setTickets(Array.isArray(data) ? data : [])
        setError(null)
      })
      .catch((err) => {
        console.error("Failed to load tickets:", err)
        setError("Failed to load tickets")
        setTickets([])
      })
      .finally(() => setLoading(false))
  }

  // يعيد التحميل كل مرة الصفحة دي "تتفتح" فعليًا من التنقل، مش مرة واحدة بس أول mount
  useEffect(() => {
    loadTickets()
  }, [pathname])

  // fallback إضافي: لو المستخدم رجع للتاب بعد ما كان في تاب/نافذة تانية
  useEffect(() => {
    const handleFocus = () => loadTickets()
    window.addEventListener("focus", handleFocus)
    return () => window.removeEventListener("focus", handleFocus)
  }, [])

  const handleRespond = async (ticketId: number) => {
    if (!answer.trim()) return
    try {
      await ticketsService.respond(ticketId, answer)
      setAnswer("")
      setSelected(null)
      loadTickets()
    } catch (err) {
      console.error("Failed to send response:", err)
    }
  }

  if (loading) return <div className="p-6">Loading tickets...</div>
  if (error) return <div className="p-6 text-red-500">{error}</div>

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">Support Tickets</h1>

      {/* Stats */}
      <div className="flex gap-4 mb-6">
        <div className="bg-red-500/20 px-4 py-2 rounded">
          🔴 Open: {tickets.filter((t) => t.status === "open").length}
        </div>
        <div className="bg-green-500/20 px-4 py-2 rounded">
          ✅ Resolved: {tickets.filter((t) => t.status === "resolved").length}
        </div>
      </div>

      {/* Tickets List */}
      <div className="space-y-3">
        {tickets.length === 0 && (
          <p className="text-muted-foreground">No tickets yet.</p>
        )}
        {tickets.map((ticket) => (
          <div key={ticket.id} className="border rounded-lg p-4">
            <div className="flex justify-between">
              <div>
                <span className="text-sm text-muted-foreground">#{ticket.id}</span>
                <p className="font-medium mt-1">{ticket.question}</p>
                <span
                  className={`text-xs px-2 py-1 rounded mt-2 inline-block
                  ${ticket.status === "open" ? "bg-red-500/20 text-red-400" : "bg-green-500/20 text-green-400"}`}
                >
                  {ticket.status}
                </span>
              </div>
              {ticket.status === "open" && (
                <button
                  onClick={() => setSelected(ticket)}
                  className="bg-primary text-white px-4 py-2 rounded h-fit"
                >
                  Respond
                </button>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Response Modal */}
      {selected && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center">
          <div className="bg-card p-6 rounded-xl w-full max-w-lg">
            <h2 className="font-bold text-lg mb-2">Ticket #{selected.id}</h2>
            <p className="text-muted-foreground mb-4">{selected.question}</p>
            <textarea
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              placeholder="اكتب الإجابة هنا..."
              className="w-full h-32 bg-input border rounded p-3 mb-4"
            />
            <div className="flex gap-3">
              <button
                onClick={() => handleRespond(selected.id)}
                className="bg-primary text-white px-6 py-2 rounded flex-1"
              >
                Send Response
              </button>
              <button onClick={() => setSelected(null)} className="border px-6 py-2 rounded">
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}