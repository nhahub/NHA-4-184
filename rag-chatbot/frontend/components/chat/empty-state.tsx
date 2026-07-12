"use client"

import { Package, RotateCcw, MapPin, KeyRound, CreditCard, Truck, HelpCircle } from "lucide-react"
import { Button } from "@/components/ui/button"

interface EmptyStateProps {
  onSuggestionClick: (suggestion: string) => void
}

const suggestions = [
  {
    icon: RotateCcw,
    text: "How to return a product?",
  },
  {
    icon: MapPin,
    text: "Where is my order?",
  },
  {
    icon: KeyRound,
    text: "How to change my password?",
  },
  {
    icon: CreditCard,
    text: "Payment methods available?",
  },
  {
    icon: Truck,
    text: "Shipping and delivery times?",
  },
  {
    icon: HelpCircle,
    text: "Contact customer support",
  },
]

export function EmptyState({ onSuggestionClick }: EmptyStateProps) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-8">
      <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-primary/10 mb-6">
        <Package className="w-8 h-8 text-primary" />
      </div>
      <h2 className="text-2xl font-bold text-foreground mb-2 text-balance text-center">
        Welcome to BrownBox Support
      </h2>
      <p className="text-muted-foreground text-center max-w-md mb-8">
        I'm your AI assistant. Ask me anything about orders, returns, shipping, or account management.
      </p>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-xl">
        {suggestions.map((suggestion, index) => {
          const Icon = suggestion.icon
          return (
            <Button
              key={index}
              variant="outline"
              className="h-auto py-3 px-4 justify-start gap-3 text-left hover:bg-accent"
              onClick={() => onSuggestionClick(suggestion.text)}
            >
              <Icon className="h-4 w-4 text-primary shrink-0" />
              <span className="text-sm">{suggestion.text}</span>
            </Button>
          )
        })}
      </div>
    </div>
  )
}
