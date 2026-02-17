"use client"

import { Button } from "@/components/ui/button"
import {
  FREE_PLAN_DAILY_REQUEST_UNITS,
  REQUEST_MULTIPLIER_TIERS,
  getRequestMultiplierFromPricing,
} from "@/lib/request-units"
import { ArrowLeft, Check, Sparkles, Zap, Building2 } from "lucide-react"
import { motion } from "framer-motion"
import Link from "next/link"
import { useEffect, useState } from "react"
import { cn } from "@/lib/utils"

type UsageResponse = {
  usage?: Array<{ created_at?: string | null }>
}

function formatEuro(value: number) {
  return new Intl.NumberFormat("fr-FR", {
    style: "currency",
    currency: "EUR",
    maximumFractionDigits: 2,
  }).format(value)
}

export default function PricingPage() {
  const [billingPeriod, setBillingPeriod] = useState<"monthly" | "yearly">("monthly")
  const [detectedMonthlyUsage, setDetectedMonthlyUsage] = useState(300)

  useEffect(() => {
    let cancelled = false

    const loadUsage = async () => {
      try {
        const response = await fetch("/api/model-usage?limit=1000&offset=0", {
          method: "GET",
          cache: "no-store",
        })

        if (!response.ok) {
          return
        }

        const payload = (await response.json()) as UsageResponse
        const usageRows = Array.isArray(payload.usage) ? payload.usage : []
        const now = new Date()
        const currentMonth = now.getMonth()
        const currentYear = now.getFullYear()

        const monthCount = usageRows.filter((row) => {
          if (!row.created_at) {
            return false
          }

          const date = new Date(row.created_at)
          return (
            date.getMonth() === currentMonth && date.getFullYear() === currentYear
          )
        }).length

        if (!cancelled && monthCount > 0) {
          setDetectedMonthlyUsage(monthCount)
        }
      } catch {
        // Keep fallback usage value
      }
    }

    loadUsage()

    return () => {
      cancelled = true
    }
  }, [])

  const hero = {
    initial: { opacity: 0, y: 14 },
    animate: { opacity: 1, y: 0 },
  }

  const plans = [
    {
      name: "Starter",
      icon: Zap,
      description: "Perfect for individuals and small projects",
      monthlyPrice: 9,
      yearlyPrice: 90,
      requestUnits: 500,
      features: [
        "500 request units/month",
        "Access to all AI models",
        "Pay-per-use after limit",
        "Basic support",
        "Standard templates",
      ],
      cta: "Get Started",
      popular: false,
    },
    {
      name: "Professional",
      icon: Sparkles,
      description: "Ideal for growing teams and businesses",
      monthlyPrice: 29,
      yearlyPrice: 290,
      requestUnits: 2000,
      features: [
        "2,000 request units/month",
        "Access to all AI models",
        "Pay-per-use after limit",
        "Priority support",
        "Advanced analytics",
        "Team collaboration",
        "API access",
      ],
      cta: "Get Started",
      popular: true,
    },
    {
      name: "Enterprise",
      icon: Building2,
      description: "For large organizations with advanced needs",
      monthlyPrice: 99,
      yearlyPrice: 990,
      requestUnits: 10000,
      features: [
        "10,000 request units/month",
        "Access to all AI models",
        "Discounted pay-per-use",
        "24/7 dedicated support",
        "Custom development",
        "Advanced security",
        "SSO integration",
        "White-label options",
      ],
      cta: "Get Started",
      popular: false,
    },
  ]

  return (
    <main className="bg-background text-foreground relative min-h-dvh overflow-hidden">
      {/* Background gradients */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="pointer-events-none absolute inset-0">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_-10%,oklch(0.72_0.09_280_/_0.18),transparent_55%)]" />
      </motion.div>

      <div className="relative mx-auto flex min-h-dvh w-full max-w-7xl flex-col px-4 py-8 sm:px-8 sm:py-12">
        {/* Header */}
        <motion.div {...hero} transition={{ duration: 0.35 }} className="mb-8 flex w-full items-center">
          <Button asChild variant="ghost" className="rounded-xl">
            <Link href="/chat" className="inline-flex items-center gap-2">
              <ArrowLeft className="size-4" />
              Back to chat
            </Link>
          </Button>
        </motion.div>

        {/* Hero Section */}
        <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col items-center justify-center gap-12 pb-12">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, delay: 0.05 }}
            className="space-y-4 text-center"
          >
            <h1 className="bg-gradient-to-b from-foreground to-foreground/70 bg-clip-text text-5xl font-bold tracking-tight text-transparent sm:text-6xl">
              Choose Your Plan
            </h1>
            <p className="text-muted-foreground mx-auto max-w-2xl text-base sm:text-lg">
              Select the perfect plan for your needs. Upgrade or downgrade at any time.
            </p>

            {/* Billing Toggle */}
            <div className="mt-6 flex items-center justify-center gap-3">
              <span className={cn("text-sm font-medium transition-colors", billingPeriod === "monthly" ? "text-foreground" : "text-muted-foreground")}>
                Monthly
              </span>
              <button
                onClick={() => setBillingPeriod(billingPeriod === "monthly" ? "yearly" : "monthly")}
                className={cn(
                  "relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                  billingPeriod === "yearly" ? "bg-primary" : "bg-input"
                )}
              >
                <span
                  className={cn(
                    "inline-block h-4 w-4 transform rounded-full bg-background transition-transform",
                    billingPeriod === "yearly" ? "translate-x-6" : "translate-x-1"
                  )}
                />
              </button>
              <span className={cn("text-sm font-medium transition-colors", billingPeriod === "yearly" ? "text-foreground" : "text-muted-foreground")}>
                Yearly
              </span>
              <span className="bg-primary/10 text-primary ml-2 rounded-full px-2.5 py-1 text-xs font-semibold">
                Save 17%
              </span>
            </div>
          </motion.div>

          {/* Pricing Cards */}
          <motion.section
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.1 }}
            className="grid w-full gap-6 lg:grid-cols-3"
          >
            {plans.map((plan, index) => {
              const Icon = plan.icon
              const price = billingPeriod === "monthly" ? plan.monthlyPrice : plan.yearlyPrice

              return (
                <motion.div
                  key={plan.name}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.35, delay: 0.1 + index * 0.05 }}
                  className={cn(
                    "relative rounded-2xl border bg-card p-8 shadow-lg transition-all hover:shadow-xl",
                    plan.popular
                      ? "border-primary/50 ring-2 ring-primary/20 scale-105"
                      : "border-border/70"
                  )}
                >
                  {plan.popular && (
                    <div className="bg-primary text-primary-foreground absolute -top-4 left-1/2 -translate-x-1/2 rounded-full px-4 py-1 text-sm font-semibold">
                      Most Popular
                    </div>
                  )}

                  <div className="flex flex-col gap-6">
                    {/* Icon & Name */}
                    <div>
                      <div className="bg-primary/10 text-primary mb-4 inline-flex rounded-xl p-3">
                        <Icon className="size-6" />
                      </div>
                      <h3 className="text-2xl font-bold">{plan.name}</h3>
                      <p className="text-muted-foreground mt-2 text-sm">{plan.description}</p>
                    </div>

                    {/* Price */}
                    <div className="flex items-baseline gap-1">
                      <span className="text-5xl font-bold">{formatEuro(price)}</span>
                      <span className="text-muted-foreground text-sm">/{billingPeriod === "monthly" ? "month" : "year"}</span>
                    </div>

                    {plan.requestUnits && (
                      <p className="text-muted-foreground -mt-4 text-sm">
                        {plan.requestUnits.toLocaleString("fr-FR")} request units / month
                      </p>
                    )}

                    {/* Features */}
                    <ul className="space-y-3">
                      {plan.features.map((feature) => (
                        <li key={feature} className="flex items-start gap-3">
                          <Check className="text-primary mt-0.5 size-5 shrink-0" />
                          <span className="text-sm">{feature}</span>
                        </li>
                      ))}
                    </ul>

                    {/* CTA */}
                    <Button
                      className={cn(
                        "w-full rounded-xl",
                        plan.popular
                          ? "bg-primary hover:bg-primary/90"
                          : "bg-secondary hover:bg-secondary/80"
                      )}
                      size="lg"
                    >
                      {plan.cta}
                    </Button>
                  </div>
                </motion.div>
              )
            })}
          </motion.section>

          {/* Model Multipliers Section */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.2 }}
            className="bg-card/50 border-border/60 w-full rounded-2xl border p-8 backdrop-blur-sm"
          >
            <div className="text-center">
              <h2 className="text-2xl font-bold">Model Multipliers</h2>
              <p className="text-muted-foreground mx-auto mt-3 max-w-3xl text-sm">
                The multiplier is calculated based on the model's input/output cost. More expensive models consume more request units.
              </p>
            </div>

            <div className="mt-8 grid gap-4 text-sm sm:grid-cols-2 lg:grid-cols-3">
              {REQUEST_MULTIPLIER_TIERS.map((tier) => (
                <div key={tier.maxCostPerMillion} className="bg-muted/50 border-border/60 rounded-xl border p-4">
                  <div className="font-semibold">Up to ${tier.maxCostPerMillion}/1M tokens</div>
                  <div className="text-primary mt-2 text-lg font-bold">x{tier.multiplier} multiplier</div>
                </div>
              ))}
              <div className="bg-muted/50 border-border/60 rounded-xl border p-4">
                <div className="font-semibold">Above ${REQUEST_MULTIPLIER_TIERS[REQUEST_MULTIPLIER_TIERS.length - 1].maxCostPerMillion}/1M</div>
                <div className="text-primary mt-2 text-lg font-bold">x6 multiplier</div>
              </div>
            </div>

            <div className="mt-6 grid gap-4 text-sm sm:grid-cols-2">
              {[
                `Model at $1.5/$6.0 per 1M tokens → x${getRequestMultiplierFromPricing({ inputCostPerMillion: 1.5, outputCostPerMillion: 6.0 })} multiplier`,
                `Model at $15/$60 per 1M tokens → x${getRequestMultiplierFromPricing({ inputCostPerMillion: 15, outputCostPerMillion: 60 })} multiplier`,
              ].map((item) => (
                <div key={item} className="bg-primary/5 border-primary/20 flex items-start gap-3 rounded-xl border p-4">
                  <Check className="text-primary mt-0.5 size-5 shrink-0" />
                  <span>{item}</span>
                </div>
              ))}
            </div>
          </motion.section>

          {/* Pay-per-use Section */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.25 }}
            className="bg-gradient-to-br from-primary/10 to-primary/5 border-primary/30 w-full rounded-2xl border p-8"
          >
            <div className="text-center">
              <h2 className="text-2xl font-bold">Pay-Per-Use After Limit</h2>
              <p className="text-muted-foreground mx-auto mt-3 max-w-3xl text-base">
                Never run out of requests! Once your monthly units are exhausted, you can continue using Ziro with our pay-per-use system.
              </p>
            </div>

            <div className="mt-8 grid gap-6 md:grid-cols-3">
              <div className="bg-card/80 rounded-xl border border-border/60 p-6 text-center">
                <div className="text-primary mb-2 text-3xl font-bold">€0.02</div>
                <div className="text-sm font-medium">per request unit</div>
                <div className="text-muted-foreground mt-2 text-xs">Starter & Professional</div>
              </div>
              <div className="bg-card/80 rounded-xl border border-border/60 p-6 text-center">
                <div className="text-primary mb-2 text-3xl font-bold">€0.015</div>
                <div className="text-sm font-medium">per request unit</div>
                <div className="text-muted-foreground mt-2 text-xs">Enterprise (25% discount)</div>
              </div>
              <div className="bg-card/80 rounded-xl border border-border/60 p-6 text-center">
                <div className="text-primary mb-2 text-3xl font-bold">€0</div>
                <div className="text-sm font-medium">hidden fees</div>
                <div className="text-muted-foreground mt-2 text-xs">100% transparent pricing</div>
              </div>
            </div>

            <div className="mt-6 text-center">
              <p className="text-muted-foreground text-sm">
                💡 <strong>Example:</strong> With Professional plan (2,000 units/month), if you use 2,500 units, you only pay €29 + (500 × €0.02) = <strong>€39 total</strong>
              </p>
            </div>
          </motion.section>
        </div>
      </div>
    </main>
  )
}
